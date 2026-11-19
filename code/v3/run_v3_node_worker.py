#!/usr/bin/env python3
"""
Phase V3 Per-Node Governed End-to-End Orchestrator (`plane1`, `plane2`, `confirm-and-holdout`, `longitudinal`).

Designed for parallel 3-node execution across `24x A100-80GB`:
  - `node-0` (`8x A100`): `qwen38_27b` (`27B` `FSDP FULL_SHARD` + `vLLM tp=4`)
  - `node-1` (`8x A100`): `qwen35_4b`  (`4B`  `DDP`             + `vLLM tp=1` x 8)
  - `node-2` (`8x A100`): `qwen35_9b`  (`9B`  `DDP`             + `vLLM tp=1` x 8)

Enforces `IMMUTABLE_MANIFEST_V3.json` verification before every stage, append-only hash-chained
ledgers, order-invariant `select-search-winner` across the 3-Proposal RLVR tournament (`R1`, `R2`, `R3`),
Intersection-Union Confirmation (`IUT`) against `B_0` + prior promoted champions, and atomic
write-once Single-Touch Protected Holdout (`gate-holdout`).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

V1_ROOT = Path("/scratch/governed_slm_rsi_v1")
V2_ROOT = Path("/scratch/governed_slm_rsi_v2")
V3_ROOT = Path("/scratch/governed_slm_rsi_v3")
TRUST_PIN_FILE = Path("/scratch/governed_slm_rsi_v3_trust_root.sha256")

EVAL_V3_DIR = V3_ROOT / "eval_v3"
CODE_V3_DIR = V3_ROOT / "code/v3"
for p in (EVAL_V3_DIR, CODE_V3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isolated_scorer_v3 import score_math_tasks_safely  # noqa: E402
from lean4_verifier import verify_lean_candidate  # noqa: E402
from model_registry_v3 import MODEL_REGISTRY_V3  # noqa: E402

GUARD_PY = EVAL_V3_DIR / "governed_loop_guard_v37.py"
SCORER_PY = EVAL_V3_DIR / "isolated_scorer_v3.py"
VLLM_PY = CODE_V3_DIR / "vllm_eval_v3.py"
SFT_PY = CODE_V3_DIR / "train_sft_v3.py"
RLVR_PY = CODE_V3_DIR / "train_rlvr_v3.py"
AXION_PY = CODE_V3_DIR / "axion_strengthen_and_frontier_v3.py"


PRIOR_CHAMPION_PATHS: dict[tuple[str, str], tuple[str, str]] = {
    ("math", "qwen35_4b"): ("v2_m_astar_qwen35_4b", str(V2_ROOT / "checkpoints/track_m/m3_qwen35_4b_dapo_chunked12k")),
    ("math", "qwen35_9b"): ("v2_m_astar_qwen35_9b", str(V2_ROOT / "checkpoints/track_m/m3_qwen35_9b_dapo_chunked12k")),
    ("lean", "qwen35_4b"): ("v1_l_astar_qwen35_4b", str(V1_ROOT / "checkpoints/track_l/l2_q4b_restem_repair")),
}


STAGE_HP: dict[tuple[str, str], dict[str, Any]] = {
    ("math", "qwen35_4b"): {
        "s1_lr": 1.2e-4, "s1_epochs": 2, "s1_max_steps": 220,
        "s2_lr": 6.0e-5, "s2_epochs": 2, "s2_max_steps": 100,
        "rl_pool_n": 320, "rl_lr": 2.0e-5, "rl_max_steps": 36,
    },
    ("math", "qwen35_9b"): {
        "s1_lr": 9.0e-5, "s1_epochs": 2, "s1_max_steps": 200,
        "s2_lr": 4.5e-5, "s2_epochs": 2, "s2_max_steps": 90,
        "rl_pool_n": 320, "rl_lr": 1.5e-5, "rl_max_steps": 36,
    },
    ("math", "qwen38_27b"): {
        "s1_lr": 6.0e-5, "s1_epochs": 1, "s1_max_steps": 100,
        "s2_lr": 3.0e-5, "s2_epochs": 1, "s2_max_steps": 56,
        "rl_pool_n": 192, "rl_lr": 1.0e-5, "rl_max_steps": 24,
    },
    ("lean", "qwen35_4b"): {
        "s1_lr": 1.4e-4, "s1_epochs": 3, "s1_max_steps": 280,
        "s2_lr": 7.0e-5, "s2_epochs": 2, "s2_max_steps": 120,
        "rl_pool_n": 320, "rl_lr": 2.0e-5, "rl_max_steps": 36,
    },
    ("lean", "qwen35_9b"): {
        "s1_lr": 1.0e-4, "s1_epochs": 3, "s1_max_steps": 240,
        "s2_lr": 5.0e-5, "s2_epochs": 2, "s2_max_steps": 110,
        "rl_pool_n": 320, "rl_lr": 1.5e-5, "rl_max_steps": 36,
    },
    ("lean", "qwen38_27b"): {
        "s1_lr": 7.0e-5, "s1_epochs": 2, "s1_max_steps": 140,
        "s2_lr": 3.5e-5, "s2_epochs": 2, "s2_max_steps": 72,
        "rl_pool_n": 240, "rl_lr": 1.0e-5, "rl_max_steps": 24,
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_checked(cmd: list[str], *, allow_rc: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    print(f"[EXEC] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode not in allow_rc:
        raise RuntimeError(
            f"Command failed (rc={r.returncode}): {' '.join(cmd)}\nSTDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}"
        )
    return r


def verify_manifest_pin() -> str:
    manifest = V3_ROOT / "IMMUTABLE_MANIFEST_V3.json"
    if not manifest.exists() or not TRUST_PIN_FILE.exists():
        raise RuntimeError("Missing IMMUTABLE_MANIFEST_V3.json or external trust root pin")
    expected = TRUST_PIN_FILE.read_text(encoding="utf-8").strip()
    _run_checked([sys.executable, "-I", "-B", str(GUARD_PY), "verify-manifest", "--manifest", str(manifest), "--expected-root-sha256", expected])
    return expected


def eval_and_score_model(
    *,
    track: str,
    model_key: str,
    model_path: str,
    candidate_id: str,
    split_prompts_file: Path,
    split_truth_file: Path,
    out_scored_json: Path,
    k: int = 8,
    gpus: str = "0,1,2,3,4,5,6,7",
    seed: int = 20260325,
    protocol_id: str = "v3_governed",
    compare_baseline_json: Path | None = None,
) -> dict[str, Any]:
    if out_scored_json.exists():
        return json.loads(out_scored_json.read_text(encoding="utf-8"))

    spec = MODEL_REGISTRY_V3[model_key]
    thinking = spec.lean_thinking if track == "lean" else spec.math_thinking
    max_toks = spec.lean_max_tokens if track == "lean" else spec.math_max_tokens
    max_model_len = 10240 if (track == "lean" and thinking) else (4096 if track == "lean" else 16384)

    gen_path = out_scored_json.with_suffix(".gen.jsonl.gz")
    if not gen_path.exists():
        vcmd = [
            sys.executable, "-I", "-B", str(VLLM_PY),
            "--model-key", model_key,
            "--model-path", model_path,
            "--split-file", str(split_prompts_file),
            "--output", str(gen_path),
            "--k", str(k),
            "--gpus", gpus,
            "--temperature", "0.7",
            "--max-tokens", str(max_toks),
            "--max-model-len", str(max_model_len),
            "--seed", str(seed),
        ]
        if thinking:
            vcmd.append("--thinking")
        _run_checked(vcmd)

    scmd = [
        sys.executable, "-I", "-B", str(SCORER_PY),
        "--track", track,
        "--split-file", str(split_truth_file),
        "--generations", str(gen_path),
        "--expected-k", str(k),
        "--candidate-id", candidate_id,
        "--protocol-id", protocol_id,
        "--output", str(out_scored_json),
    ]
    if thinking:
        scmd.append("--thinking")
    if compare_baseline_json and compare_baseline_json.exists():
        scmd.extend(["--compare-baseline", str(compare_baseline_json)])
    _run_checked(scmd)
    return json.loads(out_scored_json.read_text(encoding="utf-8"))


def _score_one_lean_rollout_pair(task: tuple[dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    gr, sp = task
    res = verify_lean_candidate(
        gr["text"],
        sp["formal_statement"],
        lean_project=V1_ROOT / "lean/mathlib_env",
        elan_bin=V1_ROOT / "lean/elan/bin",
        finish_reason=str(gr.get("finish_reason", "unknown")),
        thinking=False,
        strict_format=False,
        timeout_s=25.0,
        lean_header=sp.get("lean_header"),
    )
    return gr, bool(res.get("ok", False))


def generate_and_verify_shared_rlvr_rollouts(
    *,
    track: str,
    model_key: str,
    sft_star_path: str,
    rl_pool_file: Path,
    rl_subset_n: int,
    out_rollouts_jsonl: Path,
    gpus: str,
    seed: int = 20260325,
) -> Path:
    if out_rollouts_jsonl.exists():
        return out_rollouts_jsonl

    pool_rows = _read_jsonl(rl_pool_file)[:rl_subset_n]
    sub_prompts = out_rollouts_jsonl.with_suffix(".prompts.jsonl")
    _write_jsonl(sub_prompts, pool_rows)
    spec_by_id = {str(r["id"]): r for r in pool_rows}

    raw_gen = out_rollouts_jsonl.with_suffix(".raw_gen.jsonl.gz")
    max_toks = 2048 if track == "lean" else 10240
    max_mlen = 4096 if track == "lean" else 13312
    if not raw_gen.exists():
        _run_checked(
            [
                sys.executable, "-I", "-B", str(VLLM_PY),
                "--model-key", model_key,
                "--model-path", sft_star_path,
                "--split-file", str(sub_prompts),
                "--output", str(raw_gen),
                "--k", "8",
                "--gpus", gpus,
                "--temperature", "0.8",
                "--max-tokens", str(max_toks),
                "--max-model-len", str(max_mlen),
                "--seed", str(seed + 77),
            ]
        )

    gen_rows = _read_jsonl(raw_gen)
    verified_flags: dict[tuple[str, int], bool] = {}
    if track == "math":
        tasks_m = [(gr, spec_by_id[str(gr["id"])], False) for gr in gen_rows]
        scored_m = score_math_tasks_safely(tasks_m, workers=32)
        for r in scored_m:
            verified_flags[(str(r["id"]), int(r["sample_idx"]))] = bool(r.get("lenient_correct", False))
    else:
        tasks_l = [(gr, spec_by_id[str(gr["id"])]) for gr in gen_rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:
            for gr, ok in ex.map(_score_one_lean_rollout_pair, tasks_l, chunksize=8):
                verified_flags[(str(gr["id"]), int(gr["sample_idx"]))] = ok

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for gr in gen_rows:
        pid = str(gr["id"])
        sidx = int(gr["sample_idx"])
        sp = spec_by_id[pid]
        by_pid[pid].append(
            {
                "id": pid,
                "sample_idx": sidx,
                "prompt_messages": sp["messages"],
                "response_text": gr["text"],
                "finish_reason": gr["finish_reason"],
                "verified_ok": verified_flags[(pid, sidx)],
                "is_off_policy": False,
            }
        )

    out_rows: list[dict[str, Any]] = []
    for pid, prows in sorted(by_pid.items()):
        out_rows.extend(prows)
        if not any(x["verified_ok"] for x in prows):
            sp = spec_by_id[pid]
            ref_txt = sp.get("reference_solution") if track == "math" else sp.get("reference_proof")
            if ref_txt:
                out_rows.append(
                    {
                        "id": pid,
                        "sample_idx": 99,
                        "prompt_messages": sp["messages"],
                        "response_text": str(ref_txt),
                        "finish_reason": "stop",
                        "verified_ok": True,
                        "is_off_policy": True,
                    }
                )

    _write_jsonl(out_rollouts_jsonl, out_rows)
    return out_rollouts_jsonl


def run_plane1(args: argparse.Namespace) -> int:
    model_key = args.model_key
    spec = MODEL_REGISTRY_V3[model_key]
    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p1_dir.mkdir(parents=True, exist_ok=True)

    for track, split_fn, proto in [
        ("math", "v3_math_search160.jsonl", "v3_math_search160_k8"),
        ("lean", "v3_lean_search140.jsonl", "v3_lean_search140_k8"),
    ]:
        sp_path = V3_ROOT / "data/splits_v3" / split_fn
        b0_out = p1_dir / f"{track}_b0_{model_key}_search.json"
        eval_and_score_model(
            track=track,
            model_key=model_key,
            model_path=spec.local_path,
            candidate_id=f"b0_{model_key}",
            split_prompts_file=sp_path,
            split_truth_file=sp_path,
            out_scored_json=b0_out,
            k=8,
            gpus=args.gpus,
            seed=20260325,
            protocol_id=proto,
        )

        # Empirical A/A Null Gate Verification (`seed2 = 20269999` vs `seed1 = 20260325` MUST return `ROLLBACK`!)
        aa_out = p1_dir / f"{track}_b0_{model_key}_aa_seed2_search.json"
        eval_and_score_model(
            track=track,
            model_key=model_key,
            model_path=spec.local_path,
            candidate_id=f"b0_{model_key}_aa_seed2",
            split_prompts_file=sp_path,
            split_truth_file=sp_path,
            out_scored_json=aa_out,
            k=8,
            gpus=args.gpus,
            seed=20269999,
            protocol_id=proto,
            compare_baseline_json=b0_out,
        )
        aa_ledger = p1_dir / f"{track}_aa_null_ledger.jsonl"
        if not aa_ledger.exists():
            rules_p = V3_ROOT / "configs_v3" / f"gate_rules_v3_{track}_search.json"
            r_aa = _run_checked(
                [
                    sys.executable, "-I", "-B", str(GUARD_PY), "gate-decision",
                    "--candidate", str(aa_out),
                    "--incumbent", str(b0_out),
                    "--baseline", str(b0_out),
                    "--rules", str(rules_p),
                    "--ledger", str(aa_ledger),
                    "--trial-id", f"aa_null_{track}_{model_key}",
                    "--hypothesis", "A/A independent rollout seed null check must not promote",
                ],
                allow_rc=(0, 3),
            )
            aa_dec = json.loads(r_aa.stdout)
            print(f"[Plane-1 A/A Null] {track} {model_key}: decision={aa_dec.get('decision')}", flush=True)

        # Evaluate prior promoted champion if applicable (`v2_m_astar` / `v1_l_astar`)
        if (track, model_key) in PRIOR_CHAMPION_PATHS:
            champ_id, champ_path = PRIOR_CHAMPION_PATHS[(track, model_key)]
            if Path(champ_path).exists():
                eval_and_score_model(
                    track=track,
                    model_key=model_key,
                    model_path=champ_path,
                    candidate_id=champ_id,
                    split_prompts_file=sp_path,
                    split_truth_file=sp_path,
                    out_scored_json=p1_dir / f"{track}_{champ_id}_search.json",
                    k=8,
                    gpus=args.gpus,
                    seed=20260325,
                    protocol_id=proto,
                    compare_baseline_json=b0_out,
                )

    # Model-aware `B_0` `k=4` ZPD frontier probes on Stage-II pools
    for track, pool_fn in [
        ("math", "v3_math_stage2_frontier_pool.jsonl"),
        ("lean", "v3_lean_stage2_frontier_pool.jsonl"),
    ]:
        pool_p = V3_ROOT / "data/corpora_v3" / pool_fn
        probe_out = p1_dir / f"{track}_b0_{model_key}_stage2_probe_k4.json"
        if not probe_out.exists():
            # Project prompt-only view (`messages[:1]`) for Stage-II pool items that already carry assistant target
            raw_pool = _read_jsonl(pool_p)
            prompt_proj = p1_dir / f".{track}_stage2_probe_prompts.jsonl"
            _write_jsonl(
                prompt_proj,
                [{"id": str(r["id"]), "messages": [r["messages"][0]]} for r in raw_pool],
            )
            eval_and_score_model(
                track=track,
                model_key=model_key,
                model_path=spec.local_path,
                candidate_id=f"probe_{model_key}",
                split_prompts_file=prompt_proj,
                split_truth_file=pool_p,
                out_scored_json=probe_out,
                k=4,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=f"v3_{track}_stage2_probe_k4",
            )

    # On `node-0` (`qwen38_27b`), also run the Cross-Family-Audited `Axion` Math Problem Strengthening once!
    if model_key == "qwen38_27b":
        ax_out = V3_ROOT / "data/corpora_v3/v3_math_axion_strengthened.jsonl"
        if not ax_out.exists():
            _run_checked(
                [
                    sys.executable, "-I", "-B", str(AXION_PY), "strengthen-math",
                    "--stage1-file", str(V3_ROOT / "data/corpora_v3/v3_math_stage1_sft.jsonl"),
                    "--work-dir", str(V3_ROOT / "results_v3/axion_strengthen_work"),
                    "--output-file", str(ax_out),
                    "--num-candidates", "360",
                    "--gpus", args.gpus,
                ]
            )

    return 0


def run_plane2_track(args: argparse.Namespace, track: str) -> int:
    verify_manifest_pin()
    model_key = args.model_key
    spec = MODEL_REGISTRY_V3[model_key]
    hp = STAGE_HP[(track, model_key)]
    nproc = len([g for g in args.gpus.split(",") if g.strip()])

    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"{track}_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"{track}_{model_key}"
    p2_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    split_fn = "v3_math_search160.jsonl" if track == "math" else "v3_lean_search140.jsonl"
    proto = "v3_math_search160_k8" if track == "math" else "v3_lean_search140_k8"
    sp_path = V3_ROOT / "data/splits_v3" / split_fn
    rules_search = V3_ROOT / "configs_v3" / f"gate_rules_v3_{track}_search.json"
    ledger_path = p2_dir / f"trial_ledger_v3_{track}_{model_key}.jsonl"

    b0_json = p1_dir / f"{track}_b0_{model_key}_search.json"
    inc_json = b0_json
    if (track, model_key) in PRIOR_CHAMPION_PATHS:
        champ_id, _ = PRIOR_CHAMPION_PATHS[(track, model_key)]
        cand_champ_json = p1_dir / f"{track}_{champ_id}_search.json"
        if cand_champ_json.exists():
            inc_json = cand_champ_json

    max_seq_sft = 8192 if track == "math" else 2048
    max_tgt_sft = 7680 if track == "math" else 1536

    # ------------------------------------------------------------------
    # STAGE I: Multi-Source Breadth SFT (`constant_with_warmup`)
    # ------------------------------------------------------------------
    s1_id = f"s1_{track}_{model_key}_breadth_sft"
    s1_dir = ckpt_root / s1_id
    s1_factors = ckpt_root / f"{s1_id}_factors.safetensors"
    s1_sum = p2_dir / f"{s1_id}_train_summary.json"
    if not s1_sum.exists():
        _run_checked(
            [
                "torchrun", f"--nproc_per_node={nproc}", str(SFT_PY),
                "--model-key", model_key,
                "--train-file", str(V3_ROOT / "data/corpora_v3" / f"v3_{track}_stage1_sft.jsonl"),
                "--output-dir", str(s1_dir),
                "--save-fp32-deltas", str(s1_factors),
                "--summary-output", str(s1_sum),
                "--max-seq-len", str(max_seq_sft),
                "--max-target-tokens", str(max_tgt_sft),
                "--epochs", str(hp["s1_epochs"]),
                "--max-steps", str(hp["s1_max_steps"]),
                "--lr", str(hp["s1_lr"]),
                "--lr-schedule", "constant_with_warmup",
                "--global-batch-size", "32",
            ]
        )

    s1_scored = p2_dir / f"{s1_id}_search.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=str(s1_dir),
        candidate_id=s1_id,
        split_prompts_file=sp_path,
        split_truth_file=sp_path,
        out_scored_json=s1_scored,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=proto,
        compare_baseline_json=b0_json,
    )

    # ------------------------------------------------------------------
    # STAGE II: Model-Aware ZPD Hard-Frontier SFT (`cosine` + 15% Replay)
    # ------------------------------------------------------------------
    verify_manifest_pin()
    s2_data = p2_dir / f"stage2_sft_{track}_{model_key}.jsonl"
    if not s2_data.exists():
        ax_file = str(V3_ROOT / "data/corpora_v3/v3_math_axion_strengthened.jsonl") if track == "math" else ""
        _run_checked(
            [
                sys.executable, "-I", "-B", str(AXION_PY), "build-stage2-sft",
                "--track", track,
                "--model-key", model_key,
                "--probe-scored-json", str(p1_dir / f"{track}_b0_{model_key}_stage2_probe_k4.json"),
                "--stage1-file", str(V3_ROOT / "data/corpora_v3" / f"v3_{track}_stage1_sft.jsonl"),
                "--stage2-pool-file", str(V3_ROOT / "data/corpora_v3" / f"v3_{track}_stage2_frontier_pool.jsonl"),
                "--axion-strengthened-file", ax_file,
                "--output-file", str(s2_data),
            ]
        )

    s2_id = f"s2_{track}_{model_key}_zpd_sft"
    s2_dir = ckpt_root / s2_id
    s2_factors = ckpt_root / f"{s2_id}_factors.safetensors"
    s2_sum = p2_dir / f"{s2_id}_train_summary.json"
    if not s2_sum.exists():
        _run_checked(
            [
                "torchrun", f"--nproc_per_node={nproc}", str(SFT_PY),
                "--model-key", model_key,
                "--prior-fp32-deltas", str(s1_factors),
                "--train-file", str(s2_data),
                "--output-dir", str(s2_dir),
                "--save-fp32-deltas", str(s2_factors),
                "--summary-output", str(s2_sum),
                "--max-seq-len", str(max_seq_sft),
                "--max-target-tokens", str(max_tgt_sft),
                "--epochs", str(hp["s2_epochs"]),
                "--max-steps", str(hp["s2_max_steps"]),
                "--lr", str(hp["s2_lr"]),
                "--lr-schedule", "cosine",
                "--warmup-ratio", "0.08",
                "--global-batch-size", "32",
            ]
        )

    s2_scored = p2_dir / f"{s2_id}_search.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=str(s2_dir),
        candidate_id=s2_id,
        split_prompts_file=sp_path,
        split_truth_file=sp_path,
        out_scored_json=s2_scored,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=proto,
        compare_baseline_json=b0_json,
    )

    # Select SFT winner (`s1` vs `s2`) order-invariantly on Search
    sft_win_file = p2_dir / "sft_round_winner.json"
    if not sft_win_file.exists():
        r_sft = _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
                "--candidates", str(s1_scored), str(s2_scored),
                "--incumbent", str(b0_json),
                "--baseline", str(b0_json),
                "--rules", str(rules_search),
                "--ledger", str(ledger_path),
                "--round-id", f"round_sft_{track}_{model_key}",
            ],
            allow_rc=(0, 3),
        )
        sft_win_file.write_text(r_sft.stdout, encoding="utf-8")

    # Determine `SFT*` path & factor chain for Stage-III RLVR initialization
    s1_m = json.loads(s1_scored.read_text(encoding="utf-8"))["metrics"][f"{track}_equal_stratum_avg_at_8"]
    s2_m = json.loads(s2_scored.read_text(encoding="utf-8"))["metrics"][f"{track}_equal_stratum_avg_at_8"]
    if s2_m >= s1_m:
        sft_star_id, sft_star_dir, sft_star_scored, sft_prior_factors = (
            s2_id, s2_dir, s2_scored, f"{s1_factors},{s2_factors}"
        )
    else:
        sft_star_id, sft_star_dir, sft_star_scored, sft_prior_factors = (
            s1_id, s1_dir, s1_scored, str(s1_factors)
        )

    # ------------------------------------------------------------------
    # STAGE III: 3-Proposal Scale-Matched RLVR Tournament (`R1`, `R2`, `R3`)
    # ------------------------------------------------------------------
    verify_manifest_pin()
    rl_pool_fn = "v3_math_rl_pool640.jsonl" if track == "math" else "v3_lean_rl_pool500.jsonl"
    shared_rollouts = generate_and_verify_shared_rlvr_rollouts(
        track=track,
        model_key=model_key,
        sft_star_path=str(sft_star_dir),
        rl_pool_file=V3_ROOT / "data/corpora_v3" / rl_pool_fn,
        rl_subset_n=int(hp["rl_pool_n"]),
        out_rollouts_jsonl=p2_dir / f"shared_rlvr_rollouts_{track}_{model_key}.jsonl",
        gpus=args.gpus,
    )

    max_seq_rl = 11264 if track == "math" else 2560
    max_gen_rl = 10240 if track == "math" else 2048
    overlong_win = 2048 if track == "math" else 384

    rl_candidate_jsons: list[str] = []
    cand_id_to_ckpt: dict[str, str] = {
        s1_id: str(s1_dir),
        s2_id: str(s2_dir),
    }

    for algo in ("dapo_asymclip", "drgrpo_cispo", "gspo_luffy"):
        run_tag = f"s3_{track}_{model_key}_{algo}"
        out_100 = ckpt_root / f"{run_tag}_pct100"
        mroot = ckpt_root / f"{run_tag}_milestones"
        rl_sum = p2_dir / f"{run_tag}_train_summary.json"
        if not rl_sum.exists():
            _run_checked(
                [
                    "torchrun", f"--nproc_per_node={nproc}", str(RLVR_PY),
                    "--algo", algo,
                    "--model-key", model_key,
                    "--prior-fp32-deltas", sft_prior_factors,
                    "--rollouts-file", str(shared_rollouts),
                    "--output-dir", str(out_100),
                    "--milestones-root", str(mroot),
                    "--save-milestone-fractions", "0.50",
                    "--summary-output", str(rl_sum),
                    "--max-seq-len", str(max_seq_rl),
                    "--max-gen-tokens", str(max_gen_rl),
                    "--overlong-window", str(overlong_win),
                    "--ppo-epochs", "2",
                    "--max-steps", str(hp["rl_max_steps"]),
                    "--lr", str(hp["rl_lr"]),
                    "--global-batch-size", "32",
                ]
            )

        for pct_tag, cpath in [("pct50", mroot / "pct50"), ("pct100", out_100)]:
            if not cpath.exists():
                continue
            cid = f"{run_tag}_{pct_tag}"
            cand_id_to_ckpt[cid] = str(cpath)
            sc_out = p2_dir / f"{cid}_search.json"
            eval_and_score_model(
                track=track,
                model_key=model_key,
                model_path=str(cpath),
                candidate_id=cid,
                split_prompts_file=sp_path,
                split_truth_file=sp_path,
                out_scored_json=sc_out,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=proto,
                compare_baseline_json=b0_json,
            )
            rl_candidate_jsons.append(str(sc_out))

    # Order-invariant Search Winner Selection across ALL SFT + RLVR candidates against round incumbent (`inc_json`)!
    all_search_cands = [str(s1_scored), str(s2_scored), *rl_candidate_jsons]
    final_win_file = p2_dir / "search_tournament_winner.json"
    if not final_win_file.exists():
        r_win = _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
                "--candidates", *all_search_cands,
                "--incumbent", str(inc_json),
                "--baseline", str(b0_json),
                "--rules", str(rules_search),
                "--ledger", str(ledger_path),
                "--round-id", f"round_final_tournament_{track}_{model_key}",
            ],
            allow_rc=(0, 3),
        )
        win_payload = json.loads(r_win.stdout)
        winner_id = win_payload.get("winner_candidate_id")
        if not winner_id:
            # If none beat the prior V2 champion by +0.8pp LCB on Search, pick top-LCB tripwire-clean candidate for Confirm test
            ranked = win_payload.get("ranked_candidates", [])
            clean_ranked = [c for c in ranked if not c.get("tripwires")]
            if clean_ranked:
                winner_id = clean_ranked[0]["candidate_id"]
            else:
                winner_id = sft_star_id
        win_payload["selected_for_confirm_id"] = winner_id
        win_payload["selected_for_confirm_ckpt"] = cand_id_to_ckpt[winner_id]
        final_win_file.write_text(json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Clean up non-selected 54GB/18GB intermediate RLVR checkpoints to keep NVMe tidy
    win_info = json.loads(final_win_file.read_text(encoding="utf-8"))
    keep_ckpts = {win_info["selected_for_confirm_ckpt"], str(s1_dir), str(s2_dir)}
    for cid, cpath_s in cand_id_to_ckpt.items():
        if cpath_s not in keep_ckpts and Path(cpath_s).exists():
            shutil.rmtree(cpath_s, ignore_errors=True)

    print(f"[Plane-2 DONE] {track} {model_key}: winner={win_info['selected_for_confirm_id']}", flush=True)
    return 0


def run_confirm_and_holdout_track(args: argparse.Namespace, track: str) -> int:
    expected_root = verify_manifest_pin()
    model_key = args.model_key
    spec = MODEL_REGISTRY_V3[model_key]

    p2_dir = V3_ROOT / "results_v3/plane2" / f"{track}_{model_key}"
    ch_dir = V3_ROOT / "results_v3/confirm_holdout" / f"{track}_{model_key}"
    ch_dir.mkdir(parents=True, exist_ok=True)

    win_info = json.loads((p2_dir / "search_tournament_winner.json").read_text(encoding="utf-8"))
    astar_id = str(win_info["selected_for_confirm_id"])
    astar_ckpt = str(win_info["selected_for_confirm_ckpt"])

    confirm_fn = "v3_math_confirm300.jsonl" if track == "math" else "v3_lean_confirm180.jsonl"
    confirm_proto = "v3_math_confirm300_k8" if track == "math" else "v3_lean_confirm180_k8"
    confirm_sp = V3_ROOT / "data/splits_v3" / confirm_fn

    hold_prompts = V3_ROOT / "data/splits_v3" / f"v3_{track}_holdout_prompts.jsonl"
    hold_truth = V3_ROOT / "eval_v3/sealed_holdout" / f"v3_{track}_holdout.jsonl"
    hold_proto = f"v3_{track}_holdout_k8"

    rules_confirm = V3_ROOT / "configs_v3" / f"gate_rules_v3_{track}_confirm_{model_key}.json"
    rules_obj = json.loads(rules_confirm.read_text(encoding="utf-8"))
    req_comps: list[str] = [str(x) for x in rules_obj["required_comparators"]]
    confirm_ledger = ch_dir / f"confirm_ledger_v3_{track}_{model_key}.jsonl"

    # Build comparator -> model_path map
    comp_paths: dict[str, str] = {f"b0_{model_key}": spec.local_path}
    if (track, model_key) in PRIOR_CHAMPION_PATHS:
        cid_p, cpath_p = PRIOR_CHAMPION_PATHS[(track, model_key)]
        comp_paths[cid_p] = cpath_p

    # 1. Evaluate `B_0`, any required prior champion, and `A*_V3` on `Confirm_V3`
    b0_conf_json = ch_dir / f"confirm_b0_{model_key}.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=spec.local_path,
        candidate_id=f"b0_{model_key}",
        split_prompts_file=confirm_sp,
        split_truth_file=confirm_sp,
        out_scored_json=b0_conf_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=confirm_proto,
    )

    comp_conf_jsons: dict[str, Path] = {f"b0_{model_key}": b0_conf_json}
    for comp_id in req_comps:
        if comp_id not in comp_conf_jsons:
            c_out = ch_dir / f"confirm_{comp_id}.json"
            eval_and_score_model(
                track=track,
                model_key=model_key,
                model_path=comp_paths[comp_id],
                candidate_id=comp_id,
                split_prompts_file=confirm_sp,
                split_truth_file=confirm_sp,
                out_scored_json=c_out,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=confirm_proto,
                compare_baseline_json=b0_conf_json,
            )
            comp_conf_jsons[comp_id] = c_out

    astar_conf_json = ch_dir / f"confirm_{astar_id}.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=astar_ckpt,
        candidate_id=astar_id,
        split_prompts_file=confirm_sp,
        split_truth_file=confirm_sp,
        out_scored_json=astar_conf_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=confirm_proto,
        compare_baseline_json=b0_conf_json,
    )

    # Pin candidate artifact metadata JSON (`.json` safe file) for `gate-decision` + `gate-holdout`
    cand_pin_json = ch_dir / f"candidate_pin_{astar_id}.json"
    if not cand_pin_json.exists():
        safetensor_shas = sorted(p.name for p in Path(astar_ckpt).glob("*.safetensors"))
        cand_pin_json.write_text(
            json.dumps({"candidate_id": astar_id, "ckpt_dir": astar_ckpt, "shards": safetensor_shas}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # 2. Run IUT Confirmation Gate Decisions against ALL `required_comparators`
    verdicts_summary_path = ch_dir / "confirm_verdicts.json"
    if not verdicts_summary_path.exists():
        verdicts: dict[str, Any] = {}
        for comp_id in req_comps:
            r_gd = _run_checked(
                [
                    sys.executable, "-I", "-B", str(GUARD_PY), "gate-decision",
                    "--candidate", str(astar_conf_json),
                    "--incumbent", str(comp_conf_jsons[comp_id]),
                    "--baseline", str(b0_conf_json),
                    "--rules", str(rules_confirm),
                    "--ledger", str(confirm_ledger),
                    "--trial-id", f"confirm_{astar_id}_vs_{comp_id}",
                    "--comparator-id", comp_id,
                    "--candidate-id", astar_id,
                    "--candidate-artifacts", str(cand_pin_json),
                    "--hypothesis", f"Phase V3 2-SFT+RLVR winner {astar_id} beats {comp_id} on Confirm_V3",
                ],
                allow_rc=(0, 3),
            )
            verdicts[comp_id] = json.loads(r_gd.stdout)
        verdicts_summary_path.write_text(json.dumps(verdicts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        verdicts = json.loads(verdicts_summary_path.read_text(encoding="utf-8"))

    all_promoted = all(v.get("decision") == "PROMOTE" for v in verdicts.values())
    print(f"[Confirm IUT] {track} {model_key} ({astar_id}): all_promoted={all_promoted}", flush=True)

    # Always evaluate `B_0` + required comparators on Holdout_V3 once (for baseline reference table)
    b0_hold_json = ch_dir / f"holdout_b0_{model_key}.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=spec.local_path,
        candidate_id=f"b0_{model_key}",
        split_prompts_file=hold_prompts,
        split_truth_file=hold_truth,
        out_scored_json=b0_hold_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=hold_proto,
    )
    comp_hold_jsons: dict[str, Path] = {f"b0_{model_key}": b0_hold_json}
    for comp_id in req_comps:
        if comp_id not in comp_hold_jsons:
            ch_out = ch_dir / f"holdout_{comp_id}.json"
            eval_and_score_model(
                track=track,
                model_key=model_key,
                model_path=comp_paths[comp_id],
                candidate_id=comp_id,
                split_prompts_file=hold_prompts,
                split_truth_file=hold_truth,
                out_scored_json=ch_out,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=hold_proto,
                compare_baseline_json=b0_hold_json,
            )
            comp_hold_jsons[comp_id] = ch_out

    # On `node-0` Lean, also evaluate external `Goedel-Prover-V2-8B` zero-shot reference on `Holdout_V3`
    if track == "lean" and model_key == "qwen38_27b":
        g_spec = MODEL_REGISTRY_V3["goedel_prover_v2_8b"]
        if Path(g_spec.local_path).exists():
            eval_and_score_model(
                track="lean",
                model_key="goedel_prover_v2_8b",
                model_path=g_spec.local_path,
                candidate_id="ref_goedel_prover_v2_8b",
                split_prompts_file=hold_prompts,
                split_truth_file=hold_truth,
                out_scored_json=ch_dir / "holdout_ref_goedel_prover_v2_8b.json",
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=hold_proto,
            )

    if not all_promoted:
        (ch_dir / "holdout_status.json").write_text(
            json.dumps(
                {
                    "status": "CONFIRMED_ROLLBACK_NO_HOLDOUT_TOUCH",
                    "track": track,
                    "model_key": model_key,
                    "candidate_id": astar_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    # 3. Atomic Single-Touch Protected Holdout (`gate-holdout --phase reserve` -> Eval -> `--phase finalize`)
    seal_file = ch_dir / f"HOLDOUT_SEAL_v3_{track}_{model_key}.json"
    final_hold_status = ch_dir / "holdout_status.json"
    if final_hold_status.exists():
        return 0

    if not seal_file.exists():
        _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "gate-holdout",
                "--phase", "reserve",
                "--seal-file", str(seal_file),
                "--manifest", str(V3_ROOT / "IMMUTABLE_MANIFEST_V3.json"),
                "--expected-root-sha256", expected_root,
                "--ledger", str(confirm_ledger),
                "--rules", str(rules_confirm),
                "--candidate-id", astar_id,
                "--candidate-artifacts", str(cand_pin_json),
            ]
        )

    astar_hold_json = ch_dir / f"holdout_{astar_id}.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=astar_ckpt,
        candidate_id=astar_id,
        split_prompts_file=hold_prompts,
        split_truth_file=hold_truth,
        out_scored_json=astar_hold_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=hold_proto,
        compare_baseline_json=b0_hold_json,
    )

    comp_args: list[str] = []
    for cid, cpath_j in comp_hold_jsons.items():
        comp_args.extend(["--comparator-holdout", f"{cid}={cpath_j}"])

    r_fin = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "gate-holdout",
            "--phase", "finalize",
            "--seal-file", str(seal_file),
            "--manifest", str(V3_ROOT / "IMMUTABLE_MANIFEST_V3.json"),
            "--expected-root-sha256", expected_root,
            "--ledger", str(confirm_ledger),
            "--rules", str(rules_confirm),
            "--candidate-id", astar_id,
            "--candidate-artifacts", str(cand_pin_json),
            "--candidate-holdout", str(astar_hold_json),
            *comp_args,
        ],
        allow_rc=(0, 3),
    )
    final_hold_status.write_text(r_fin.stdout, encoding="utf-8")
    return 0


def run_longitudinal(args: argparse.Namespace) -> int:
    """
    Post-lock longitudinal benchmark evaluation on V2's 2025 4-Contest Math Suite
    (`BeyondAIME`, `CMIMC 2025`, `BRUMO 2025`, `HMMT Nov 2025`) and `MiniF2F-test` / `ProofNet-test`
    across `4B`, `9B`, and tuned `27B`!
    """
    model_key = args.model_key
    spec = MODEL_REGISTRY_V3[model_key]
    long_dir = V3_ROOT / "results_v3/longitudinal" / model_key
    long_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate V3 Math Search Winner (`A*_V3`) vs `B_0` on V2's 2025 4-Contest Suite (`N=200`)
    m_win_file = V3_ROOT / "results_v3/plane2" / f"math_{model_key}" / "search_tournament_winner.json"
    if m_win_file.exists():
        m_win = json.loads(m_win_file.read_text(encoding="utf-8"))
        m_cid = str(m_win["selected_for_confirm_id"])
        m_ckpt = str(m_win["selected_for_confirm_ckpt"])
        for bench_tag, p_fn, t_fn in [
            ("beyondaime_100", "v2_math_holdout_beyondaime_prompts.jsonl", "v2_math_holdout_beyondaime.jsonl"),
            ("cmimc2025_40", "v2_math_holdout_cmimc2025_prompts.jsonl", "v2_math_holdout_cmimc2025.jsonl"),
            ("brumo2025_30", "v2_math_holdout_brumo2025_prompts.jsonl", "v2_math_holdout_brumo2025.jsonl"),
            ("hmmt_nov2025_30", "v2_math_holdout_hmmt_nov2025_prompts.jsonl", "v2_math_holdout_hmmt_nov2025.jsonl"),
        ]:
            p_path = V2_ROOT / "data/splits_v2" / p_fn
            t_path = V2_ROOT / "eval_v2/sealed_holdout" / t_fn
            if not (p_path.exists() and t_path.exists()):
                continue
            b0_out = long_dir / f"v2suite_{bench_tag}_b0_{model_key}.json"
            eval_and_score_model(
                track="math",
                model_key=model_key,
                model_path=spec.local_path,
                candidate_id=f"b0_{model_key}",
                split_prompts_file=p_path,
                split_truth_file=t_path,
                out_scored_json=b0_out,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=f"longitudinal_{bench_tag}_k8",
            )
            eval_and_score_model(
                track="math",
                model_key=model_key,
                model_path=m_ckpt,
                candidate_id=m_cid,
                split_prompts_file=p_path,
                split_truth_file=t_path,
                out_scored_json=long_dir / f"v2suite_{bench_tag}_{m_cid}.json",
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=f"longitudinal_{bench_tag}_k8",
                compare_baseline_json=b0_out,
            )

    # Evaluate V3 Lean Search Winner (`A*_V3`) vs `B_0` on `ProofNet-test` (`N=167`) & `MiniF2F-test` (`N=244`)
    l_win_file = V3_ROOT / "results_v3/plane2" / f"lean_{model_key}" / "search_tournament_winner.json"
    if l_win_file.exists():
        l_win = json.loads(l_win_file.read_text(encoding="utf-8"))
        l_cid = str(l_win["selected_for_confirm_id"])
        l_ckpt = str(l_win["selected_for_confirm_ckpt"])
        for bench_tag, p_path, t_path in [
            (
                "proofnet_test167",
                V2_ROOT / "data/splits_v2/v2_lean_holdout_proofnet174_prompts.jsonl",
                V2_ROOT / "eval_v2/sealed_holdout/v2_lean_holdout_proofnet174.jsonl",
            ),
            (
                "minif2f_test244",
                V1_ROOT / "data/splits/lean_holdout_minif2f_test244_prompts.jsonl",
                V1_ROOT / "eval/sealed_holdout/lean_holdout_minif2f_test244.jsonl",
            ),
        ]:
            if not (p_path.exists() and t_path.exists()):
                continue
            b0_out = long_dir / f"leansuite_{bench_tag}_b0_{model_key}.json"
            eval_and_score_model(
                track="lean",
                model_key=model_key,
                model_path=spec.local_path,
                candidate_id=f"b0_{model_key}",
                split_prompts_file=p_path,
                split_truth_file=t_path,
                out_scored_json=b0_out,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=f"longitudinal_{bench_tag}_k8",
            )
            eval_and_score_model(
                track="lean",
                model_key=model_key,
                model_path=l_ckpt,
                candidate_id=l_cid,
                split_prompts_file=p_path,
                split_truth_file=t_path,
                out_scored_json=long_dir / f"leansuite_{bench_tag}_{l_cid}.json",
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=f"longitudinal_{bench_tag}_k8",
                compare_baseline_json=b0_out,
            )

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V3 Per-Node Governed End-to-End Worker")
    ap.add_argument("--phase", required=True, choices=["plane1", "plane2", "confirm-and-holdout", "longitudinal", "full-post-freeze"])
    ap.add_argument("--model-key", required=True, choices=["qwen35_4b", "qwen35_9b", "qwen38_27b"])
    ap.add_argument("--tracks", default="lean,math", help="Comma-separated tracks (`lean,math`)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    track_list = [t.strip() for t in args.tracks.split(",") if t.strip()]
    if args.phase == "plane1":
        return run_plane1(args)
    if args.phase == "plane2":
        for tr in track_list:
            run_plane2_track(args, tr)
        return 0
    if args.phase == "confirm-and-holdout":
        for tr in track_list:
            run_confirm_and_holdout_track(args, tr)
        return 0
    if args.phase == "longitudinal":
        return run_longitudinal(args)
    if args.phase == "full-post-freeze":
        for tr in track_list:
            run_plane2_track(args, tr)
            run_confirm_and_holdout_track(args, tr)
        run_longitudinal(args)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Phase V4 Cross-Family Governed End-to-End Orchestrator (`plane1`, `plane2`, `confirm-and-holdout`, `longitudinal`, `e4b-reference`).

Covers:
  - `node-0` (`8x A100-80GB`): `inkling2_mini` (`hf_dp` x 8 + `FSDP`) & `muse_glimmer_30b` (`vLLM tp=4` x 2 + `FSDP`)
  - `node-1` (`8x A100-80GB`): `gemma4_26b_a4b` (`vLLM tp=2` x 4 + `FSDP`), `gemma4_31b` (`vLLM tp=4` x 2 + `FSDP`), & `gemma4_e4b` (`vLLM tp=1` x 8 reference)

Governance & Statistical Guarantees:
  1. Verifies `IMMUTABLE_MANIFEST_V4.json` against out-of-repo `/scratch/governed_crossfamily_v4_trust_root.sha256` before every post-freeze phase.
  2. Executes controlled Plane-2 Diagnostic Crossover (`Arm A: Pure-B0 -> On-Policy RLVR` vs `Arm B: Horizon-Matched 2-SFT -> 3-Proposal RLVR`) inside `Search_V3` (`0` Confirmation looks spent), followed by order-invariant `select-search-winner` (`aggregation_weighting = "equal_stratum"`).
  3. Binds full SHA-256 digests of every `.safetensors` shard + `config.json` in `candidate_pin_<id>.json`, runs `gate-confirm-iut` (`max_confirm_looks = 2`, `alpha = 0.025`, `+1.50pp` margin), and executes atomic write-once `gate-holdout` (`reserve -> eval -> finalize`) strictly on Confirmed winners.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

V4_ROOT = Path(os.environ.get("V4_ROOT", "/scratch/governed_crossfamily_v4"))
TRUST_PIN_FILE = Path(os.environ.get("V4_TRUST_PIN", "/scratch/governed_crossfamily_v4_trust_root.sha256"))

EVAL_V4_DIR = V4_ROOT / "eval_v4"
CODE_V4_DIR = V4_ROOT / "code/v4"
for p in (EVAL_V4_DIR, CODE_V4_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from channel_normalizer_v4 import normalize_reasoning_channel  # noqa: E402
from isolated_scorer_v4 import score_math_tasks_safely  # noqa: E402
from lean4_verifier_v4 import verify_lean_candidate  # noqa: E402
from model_registry_v4 import MODEL_REGISTRY_V4  # noqa: E402

GUARD_PY = EVAL_V4_DIR / "governed_loop_guard_v38.py"
SCORER_PY = EVAL_V4_DIR / "isolated_scorer_v4.py"
EVAL_PY = CODE_V4_DIR / "eval_engine_v4.py"
SFT_PY = CODE_V4_DIR / "train_sft_v4.py"
RLVR_PY = CODE_V4_DIR / "train_rlvr_v4.py"


STAGE_HP_V4: dict[tuple[str, str], dict[str, Any]] = {
    # Sparse MoEs (frozen routers & frozen 128 experts; adapting unconditionally active attention / shared projections)
    ("math", "inkling2_mini"): {
        "s1_lr": 2.4e-5, "s1_epochs": 2, "s1_max_steps": 88,
        "s2_lr": 1.1e-5, "s2_epochs": 2, "s2_max_steps": 44,
        "rl_pool_n": 128, "rl_lr": 8.5e-6, "rl_ppo_epochs": 1, "rl_max_steps": 24,
    },
    ("lean", "inkling2_mini"): {
        "s1_lr": 3.5e-5, "s1_epochs": 2, "s1_max_steps": 112,
        "s2_lr": 1.5e-5, "s2_epochs": 1, "s2_max_steps": 56,
        "rl_pool_n": 128, "rl_lr": 6.5e-6, "rl_ppo_epochs": 1, "rl_max_steps": 24,
    },
    ("math", "gemma4_26b_a4b"): {
        "s1_lr": 2.2e-5, "s1_epochs": 2, "s1_max_steps": 92,
        "s2_lr": 1.0e-5, "s2_epochs": 2, "s2_max_steps": 44,
        "rl_pool_n": 144, "rl_lr": 8.0e-6, "rl_ppo_epochs": 1, "rl_max_steps": 24,
    },
    ("lean", "gemma4_26b_a4b"): {
        "s1_lr": 3.2e-5, "s1_epochs": 2, "s1_max_steps": 112,
        "s2_lr": 1.4e-5, "s2_epochs": 1, "s2_max_steps": 56,
        "rl_pool_n": 256, "rl_lr": 6.0e-6, "rl_ppo_epochs": 1, "rl_max_steps": 24,
    },
    # ~30B Dense Models (full attention + MLP LoRA rank-32)
    ("math", "gemma4_31b"): {
        "s1_lr": 1.8e-5, "s1_epochs": 2, "s1_max_steps": 88,
        "s2_lr": 8.5e-6, "s2_epochs": 2, "s2_max_steps": 42,
        "rl_pool_n": 128, "rl_lr": 7.0e-6, "rl_ppo_epochs": 1, "rl_max_steps": 22,
    },
    ("lean", "gemma4_31b"): {
        "s1_lr": 2.8e-5, "s1_epochs": 2, "s1_max_steps": 108,
        "s2_lr": 1.2e-5, "s2_epochs": 1, "s2_max_steps": 52,
        "rl_pool_n": 240, "rl_lr": 5.5e-6, "rl_ppo_epochs": 1, "rl_max_steps": 22,
    },
    ("math", "muse_glimmer_30b"): {
        "s1_lr": 1.8e-5, "s1_epochs": 2, "s1_max_steps": 88,
        "s2_lr": 8.5e-6, "s2_epochs": 2, "s2_max_steps": 42,
        "rl_pool_n": 128, "rl_lr": 7.0e-6, "rl_ppo_epochs": 1, "rl_max_steps": 22,
    },
    ("lean", "muse_glimmer_30b"): {
        "s1_lr": 2.8e-5, "s1_epochs": 2, "s1_max_steps": 108,
        "s2_lr": 1.2e-5, "s2_epochs": 1, "s2_max_steps": 52,
        "rl_pool_n": 240, "rl_lr": 5.5e-6, "rl_ppo_epochs": 1, "rl_max_steps": 22,
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


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    lib_dir = str(V4_ROOT / "lib")
    cur_ld = env.get("LD_LIBRARY_PATH", "")
    if lib_dir not in cur_ld.split(":"):
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{cur_ld}" if cur_ld else lib_dir
    env["V4_ROOT"] = str(V4_ROOT)
    return env


def _run_checked(cmd: list[str], *, allow_rc: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    print(f"[EXEC] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, capture_output=True, env=_child_env())
    if r.returncode not in allow_rc:
        raise RuntimeError(
            f"Command failed (rc={r.returncode}): {' '.join(cmd)}\nSTDOUT:\n{r.stdout[-2500:]}\nSTDERR:\n{r.stderr[-2500:]}"
        )
    return r


def verify_manifest_pin() -> str:
    manifest = V4_ROOT / "IMMUTABLE_MANIFEST_V4.json"
    if not manifest.exists() or not TRUST_PIN_FILE.exists():
        raise RuntimeError("Missing IMMUTABLE_MANIFEST_V4.json or external trust root pin")
    expected = TRUST_PIN_FILE.read_text(encoding="utf-8").strip()
    _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "verify-manifest",
            "--manifest", str(manifest),
            "--expected-root-sha256", expected,
        ]
    )
    return expected


def _hash_one_file_sha256(fpath: Path) -> tuple[str, str, int]:
    h = hashlib.sha256()
    sz = 0
    with fpath.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            sz += len(chunk)
    return fpath.name, h.hexdigest(), sz


def build_cryptographic_candidate_pin(
    candidate_id: str,
    ckpt_dir: str | Path,
    out_pin_json: Path,
    *,
    verify_existing: bool = True,
) -> Path:
    """
    Computes (and re-verifies) full parallel SHA-256 digests of every `.safetensors` shard,
    `model.safetensors.index.json`, and `config.json` inside `ckpt_dir` for non-repudiable
    cryptographic weight binding before and after evaluation.
    """
    cdir = Path(ckpt_dir)
    raw_targets = [
        *cdir.glob("*.safetensors"),
        *cdir.glob("*.json"),
        *cdir.glob("*.jinja"),
        *cdir.glob("*.model"),
        *cdir.glob("*.py"),
        *(list((cdir / "tiktoken").glob("*")) if (cdir / "tiktoken").is_dir() else []),
        EVAL_PY,
        CODE_V4_DIR / "model_registry_v4.py",
    ]
    targets = sorted([p for p in raw_targets if p.is_file()], key=lambda p: p.as_posix())
    if not targets:
        raise RuntimeError(f"No checkpoint weight/config artifacts found in {cdir}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(targets))) as ex:
        records = list(ex.map(_hash_one_file_sha256, targets))
    shard_map = {name: {"sha256": digest, "bytes": sz} for name, digest, sz in records}
    tree_hash = hashlib.sha256(json.dumps(shard_map, sort_keys=True).encode("utf-8")).hexdigest()

    if out_pin_json.exists() and verify_existing:
        existing = json.loads(out_pin_json.read_text(encoding="utf-8"))
        if existing.get("weights_merkle_sha256") != tree_hash:
            raise RuntimeError(
                f"Cryptographic checkpoint drift detected for {candidate_id}: "
                f"live={tree_hash} != pinned={existing.get('weights_merkle_sha256')}"
            )
        return out_pin_json

    payload = {
        "candidate_id": candidate_id,
        "ckpt_dir": str(cdir),
        "weights_merkle_sha256": tree_hash,
        "artifacts": shard_map,
    }
    out_pin_json.parent.mkdir(parents=True, exist_ok=True)
    out_pin_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_pin_json


def _ensure_unique_split_ids(split_prompts_file: Path, split_truth_file: Path) -> tuple[Path, Path]:
    """
    Materializes a deterministic read-only disambiguated view (`__occ{k}`) when frozen split files
    contain colliding IDs across subject sub-domains (e.g. `v3_math_holdout.jsonl`), without
    touching a single byte of any frozen split file.
    """
    t_rows = _read_jsonl(split_truth_file)
    id_counts = collections.Counter(str(r["id"]) for r in t_rows)
    if len(id_counts) == len(t_rows):
        return split_prompts_file, split_truth_file

    dis_dir = V4_ROOT / "data/disambiguated_splits"
    dis_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(f"{split_prompts_file}:{split_truth_file}".encode("utf-8")).hexdigest()[:10]
    out_t = dis_dir / f"{split_truth_file.stem}_{tag}_truth.jsonl"
    out_p = out_t if split_prompts_file == split_truth_file else (dis_dir / f"{split_prompts_file.stem}_{tag}_prompts.jsonl")
    if out_t.exists() and out_p.exists():
        return out_p, out_t

    p_rows = t_rows if split_prompts_file == split_truth_file else _read_jsonl(split_prompts_file)
    if len(p_rows) != len(t_rows):
        raise RuntimeError(f"Prompt/truth length mismatch: {len(p_rows)} vs {len(t_rows)}")

    seen_occ: dict[str, int] = collections.defaultdict(int)
    new_p: list[dict[str, Any]] = []
    new_t: list[dict[str, Any]] = []
    for pr, tr in zip(p_rows, t_rows):
        raw_id = str(tr["id"])
        if str(pr["id"]) != raw_id:
            raise RuntimeError(f"Row ID alignment mismatch: {pr['id']} vs {raw_id}")
        p_sha = hashlib.sha256(json.dumps(pr["messages"], sort_keys=True).encode("utf-8")).hexdigest()
        t_sha = hashlib.sha256(json.dumps(tr["messages"], sort_keys=True).encode("utf-8")).hexdigest()
        if p_sha != t_sha:
            raise RuntimeError(f"Row messages SHA mismatch on {raw_id}")

        seen_occ[raw_id] += 1
        occ = seen_occ[raw_id]
        uid = raw_id if id_counts[raw_id] == 1 else f"{raw_id}__occ{occ}"

        pr_c = dict(pr)
        pr_c["id"] = uid
        if "cluster_id" in pr_c and str(pr_c["cluster_id"]) == raw_id:
            pr_c["cluster_id"] = uid
        new_p.append(pr_c)

        tr_c = dict(tr)
        tr_c["id"] = uid
        if "cluster_id" in tr_c and str(tr_c["cluster_id"]) == raw_id:
            tr_c["cluster_id"] = uid
        new_t.append(tr_c)

    _write_jsonl(out_t, new_t)
    if out_p != out_t:
        _write_jsonl(out_p, new_p)
    return out_p, out_t


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
    protocol_id: str = "v4_governed",
    compare_baseline_json: Path | None = None,
) -> dict[str, Any]:
    if out_scored_json.exists():
        return json.loads(out_scored_json.read_text(encoding="utf-8"))

    split_prompts_file, split_truth_file = _ensure_unique_split_ids(split_prompts_file, split_truth_file)
    spec = MODEL_REGISTRY_V4[model_key]
    max_toks = spec.lean_max_tokens if track == "lean" else spec.math_max_tokens
    max_model_len = 8192 if (track == "lean" and max_toks > 2048) else (4096 if track == "lean" else 16384)

    gen_path = out_scored_json.with_suffix(".gen.jsonl.gz")
    if not gen_path.exists():
        _run_checked(
            [
                sys.executable, "-B", str(EVAL_PY),
                "--model-key", model_key,
                "--model-path", model_path,
                "--track", track,
                "--split-file", str(split_prompts_file),
                "--output", str(gen_path),
                "--k", str(k),
                "--gpus", gpus,
                "--temperature", "0.7",
                "--max-tokens", str(max_toks),
                "--max-model-len", str(max_model_len),
                "--seed", str(seed),
            ]
        )

    thinking = spec.lean_thinking if track == "lean" else spec.math_thinking
    scmd = [
        sys.executable, "-I", "-B", str(SCORER_PY),
        "--track", track,
        "--channel-family", spec.channel_family,
        "--split-file", str(split_truth_file),
        "--generations", str(gen_path),
        "--expected-k", str(k),
        "--candidate-id", candidate_id,
        "--protocol-id", protocol_id,
        "--lean-project", str(V4_ROOT / "lean/mathlib_env"),
        "--elan-bin", str(V4_ROOT / "lean/elan/bin"),
        "--output", str(out_scored_json),
    ]
    if thinking:
        scmd.append("--thinking")
    if compare_baseline_json and compare_baseline_json.exists():
        scmd.extend(["--compare-baseline", str(compare_baseline_json)])
    _run_checked(scmd)
    return json.loads(out_scored_json.read_text(encoding="utf-8"))


def _score_one_lean_rollout_pair(task: tuple[dict[str, Any], dict[str, Any], str]) -> tuple[dict[str, Any], bool]:
    gr, sp, ch_fam = task
    res = verify_lean_candidate(
        gr["text"],
        sp["formal_statement"],
        lean_project=V4_ROOT / "lean/mathlib_env",
        elan_bin=V4_ROOT / "lean/elan/bin",
        finish_reason=str(gr.get("finish_reason", "unknown")),
        channel_family=ch_fam,
        strict_format=False,
        timeout_s=25.0,
        lean_header=sp.get("lean_header"),
    )
    return gr, bool(res.get("ok", False))


def generate_and_verify_rollouts(
    *,
    track: str,
    model_key: str,
    policy_path: str,
    rl_pool_file: Path,
    rl_subset_n: int,
    out_rollouts_jsonl: Path,
    gpus: str,
    inject_off_policy: bool = True,
    seed: int = 20260325,
) -> Path:
    if out_rollouts_jsonl.exists():
        return out_rollouts_jsonl

    spec = MODEL_REGISTRY_V4[model_key]
    pool_rows = _read_jsonl(rl_pool_file)[:rl_subset_n]
    # Project strictly to single user prompt message so `eval_engine_v4` never sees any target
    sub_prompts = out_rollouts_jsonl.with_suffix(".prompts.jsonl")
    _write_jsonl(
        sub_prompts,
        [{"id": str(r["id"]), "messages": [r["messages"][0]]} for r in pool_rows],
    )
    spec_by_id = {str(r["id"]): r for r in pool_rows}

    raw_gen = out_rollouts_jsonl.with_suffix(".raw_gen.jsonl.gz")
    max_toks = min(spec.lean_max_tokens, 3072) if track == "lean" else min(spec.math_max_tokens, 10240)
    max_mlen = 6144 if track == "lean" else 13312
    if not raw_gen.exists():
        _run_checked(
            [
                sys.executable, "-B", str(EVAL_PY),
                "--model-key", model_key,
                "--model-path", policy_path,
                "--track", track,
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
        thinking_flag = spec.math_thinking
        tasks_m = [(gr, spec_by_id[str(gr["id"])], thinking_flag, spec.channel_family) for gr in gen_rows]
        scored_m = score_math_tasks_safely(tasks_m, workers=32)
        for r in scored_m:
            verified_flags[(str(r["id"]), int(r["sample_idx"]))] = bool(r.get("lenient_correct", False))
    else:
        tasks_l = [(gr, spec_by_id[str(gr["id"])], spec.channel_family) for gr in gen_rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:
            for gr, ok in ex.map(_score_one_lean_rollout_pair, tasks_l, chunksize=8):
                verified_flags[(str(gr["id"]), int(gr["sample_idx"]))] = ok

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for gr in gen_rows:
        pid = str(gr["id"])
        sidx = int(gr["sample_idx"])
        by_pid[pid].append(
            {
                "sample_idx": sidx,
                "text": gr["text"],
                "token_ids": gr.get("token_ids"),
                "finish_reason": gr["finish_reason"],
                "completion_tokens": int(gr.get("completion_tokens", 0)),
                "verified": verified_flags[(pid, sidx)],
                "off_policy": False,
            }
        )

    out_groups: list[dict[str, Any]] = []
    for pid, psamples in sorted(by_pid.items()):
        sp = spec_by_id[pid]
        samples_list = list(psamples)
        if inject_off_policy and not any(x["verified"] for x in samples_list):
            ref_txt = sp.get("reference_solution") if track == "math" else sp.get("reference_proof")
            if not ref_txt and len(sp.get("messages", [])) >= 2 and sp["messages"][-1].get("role") == "assistant":
                ref_txt = sp["messages"][-1]["content"]
            if ref_txt:
                clean_ref = normalize_reasoning_channel(str(ref_txt), channel_family="auto").final_text or str(ref_txt).strip()
                # Verify rescue trace through the track verifier before marking `verified=True`
                rescue_ok = False
                if track == "math":
                    sc_ref = score_math_tasks_safely(
                        [({"id": pid, "sample_idx": 99, "text": clean_ref, "finish_reason": "stop", "completion_tokens": 0}, sp, False, "qwen")],
                        workers=1,
                    )[0]
                    rescue_ok = bool(sc_ref.get("lenient_correct", False))
                else:
                    v_ref = verify_lean_candidate(
                        clean_ref,
                        sp["formal_statement"],
                        lean_project=V4_ROOT / "lean/mathlib_env",
                        elan_bin=V4_ROOT / "lean/elan/bin",
                        finish_reason="stop",
                        channel_family="qwen",
                        strict_format=False,
                        timeout_s=25.0,
                        lean_header=sp.get("lean_header"),
                    )
                    rescue_ok = bool(v_ref.get("ok", False))
                if rescue_ok:
                    samples_list.append(
                        {
                            "sample_idx": 99,
                            "text": clean_ref,
                            "token_ids": None,
                            "finish_reason": "stop",
                            "completion_tokens": max(1, len(clean_ref) // 4),
                            "verified": True,
                            "off_policy": True,
                        }
                    )
        out_groups.append(
            {
                "id": pid,
                "messages": [sp["messages"][0]],
                "samples": samples_list,
            }
        )

    _write_jsonl(out_rollouts_jsonl, out_groups)
    return out_rollouts_jsonl


def build_model_aware_stage2_sft(
    *,
    track: str,
    model_key: str,
    probe_scored_json: Path,
    stage1_file: Path,
    stage2_pool_file: Path,
    s1_onpolicy_rollouts: Path | None,
    out_sft_jsonl: Path,
) -> Path:
    """
    Constructs a model-aware Stage-II ZPD Frontier SFT dataset:
      1. Retains `stage2_pool_file` items in the target model's ZPD/hard-frontier (`n_correct <= 2` out of `k=4`).
      2. For Lean, optionally adds `<= 2` shortest `#print axioms`-verified on-policy `s1` rollouts (`ReST^EM`).
      3. Mixes `25%` Stage-I anti-forgetting replay.
      4. Enforces zero overlap against all protected Search/Confirm/Holdout IDs.
    """
    if out_sft_jsonl.exists():
        return out_sft_jsonl

    forbidden_ids: set[str] = set()
    for sp_f in (V4_ROOT / "data/splits_v3").glob("*.jsonl"):
        for r in _read_jsonl(sp_f):
            forbidden_ids.add(str(r.get("id", "")))

    probe_obj = json.loads(probe_scored_json.read_text(encoding="utf-8"))
    hard_ids = {
        str(it["id"])
        for it in probe_obj.get("items", [])
        if float(it.get("value", 1.0)) <= 0.501
    }

    rng = random.Random(20261123 ^ int(hashlib.sha256(f"{track}:{model_key}".encode()).hexdigest()[:8], 16))
    pool_rows = _read_jsonl(stage2_pool_file)
    zpd_rows = [
        r for r in pool_rows
        if (str(r.get("problem_id") or r.get("id", "")) in hard_ids or str(r.get("id", "")) in hard_ids)
        and str(r.get("id", "")) not in forbidden_ids
    ]
    if len(zpd_rows) < 120:
        # Fallback if model is very strong on probe: take hardest 50%
        extra = [r for r in pool_rows if str(r.get("id", "")) not in forbidden_ids and r not in zpd_rows]
        rng.shuffle(extra)
        zpd_rows.extend(extra[: max(0, 240 - len(zpd_rows))])
    rng.shuffle(zpd_rows)
    core = zpd_rows[:750]

    # Add verified on-policy ReST^EM shortest proofs if available (grouped rollout JSONL schema)
    if s1_onpolicy_rollouts is not None and s1_onpolicy_rollouts.exists():
        for grp in _read_jsonl(s1_onpolicy_rollouts):
            pid = str(grp.get("id", ""))
            if pid in forbidden_ids:
                continue
            p_msgs = list(grp.get("messages", []))
            ok_list = [
                s for s in grp.get("samples", [])
                if bool(s.get("verified")) and not bool(s.get("off_policy")) and str(s.get("finish_reason")) == "stop"
            ]
            ok_sorted = sorted(ok_list, key=lambda x: len(str(x.get("text", ""))))
            for k_idx, item in enumerate(ok_sorted[:2]):
                norm = normalize_reasoning_channel(str(item["text"]), channel_family="auto")
                clean_ans = norm.final_text.strip()
                if clean_ans and not norm.has_template_leak and not norm.has_unclosed_thinking:
                    core.append(
                        {
                            "id": f"{pid}:restem_{k_idx}",
                            "source": "onpolicy_verified_restem",
                            "messages": p_msgs + [{"role": "assistant", "content": clean_ans}],
                        }
                    )

    s1_rows = [r for r in _read_jsonl(stage1_file) if str(r.get("id", "")) not in forbidden_ids]
    rng.shuffle(s1_rows)
    n_replay = max(160, int(round(0.30 * len(core))))
    combined = core + s1_rows[:n_replay]
    rng.shuffle(combined)
    _write_jsonl(out_sft_jsonl, combined)
    print(f"[Stage-II ZPD Data] {track} {model_key}: total={len(combined)} (core={len(core)}, replay={n_replay})", flush=True)
    return out_sft_jsonl


def run_plane1(args: argparse.Namespace) -> int:
    model_key = args.model_key
    spec = MODEL_REGISTRY_V4[model_key]
    p1_dir = V4_ROOT / "results_v4/plane1" / model_key
    p1_dir.mkdir(parents=True, exist_ok=True)
    req_tracks = {t.strip() for t in getattr(args, "tracks", "lean,math").split(",") if t.strip()}

    hygiene_path = p1_dir / "verifier_hygiene_audit.json"
    hygiene_summary: dict[str, Any] = (
        json.loads(hygiene_path.read_text(encoding="utf-8")) if hygiene_path.exists() else {}
    )

    for track, split_fn, proto in [
        ("math", "v3_math_search160.jsonl", "v4_math_search160_k8"),
        ("lean", "v3_lean_search140.jsonl", "v4_lean_search140_k8"),
    ]:
        if track not in req_tracks:
            continue
        sp_path = V4_ROOT / "data/splits_v3" / split_fn
        b0_out = p1_dir / f"{track}_b0_{model_key}_search.json"
        b0_pay = eval_and_score_model(
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

        # 40-sample Plane-1 Verifier Hygiene Audit on `B_0` generations
        gen_gz = b0_out.with_suffix(".gen.jsonl.gz")
        gen_rows = _read_jsonl(gen_gz)
        sample_40 = gen_rows[:: max(1, len(gen_rows) // 40)][:40]
        leaks = 0
        unclosed = 0
        for gr in sample_40:
            nm = normalize_reasoning_channel(str(gr.get("text", "")), channel_family=spec.channel_family)
            if nm.has_template_leak:
                leaks += 1
            if nm.has_unclosed_thinking:
                unclosed += 1
        hygiene_summary[track] = {
            "audited_samples": len(sample_40),
            "template_leak_count": leaks,
            "unclosed_thinking_count": unclosed,
            "b0_metrics": b0_pay.get("metrics", {}),
        }

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
            rules_p = V4_ROOT / "configs_v4" / f"gate_rules_v4_{track}_search.json"
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
                allow_rc=(0, 1, 3),
            )
            aa_dec = json.loads(r_aa.stdout)
            print(f"[Plane-1 A/A Null] {track} {model_key}: decision={aa_dec.get('decision')}", flush=True)
            if aa_dec.get("decision") != "ROLLBACK":
                raise RuntimeError(f"Plane-1 A/A null check failed to rollback on {track} {model_key}: {aa_dec}")

    # Model-aware `B_0` `k=4` ZPD frontier probes on unique Stage-II frontier problems (up to 240 unique problems)
    for track, pool_fn in [
        ("math", "v4_math_stage2_frontier_pool.jsonl"),
        ("lean", "v4_lean_stage2_frontier_pool.jsonl"),
    ]:
        if track not in req_tracks:
            continue
        pool_p = V4_ROOT / "data/corpora_v4" / pool_fn
        probe_out = p1_dir / f"{track}_b0_{model_key}_stage2_probe_k4.json"
        if pool_p.exists() and not probe_out.exists():
            raw_pool = _read_jsonl(pool_p)
            seen_pids: set[str] = set()
            dedup_truth: list[dict[str, Any]] = []
            for r in raw_pool:
                pid = str(r.get("problem_id") or r["id"])
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                rc = dict(r)
                rc["id"] = pid
                dedup_truth.append(rc)
                if len(dedup_truth) >= 240:
                    break
            prompt_proj = p1_dir / f".{track}_stage2_probe_prompts.jsonl"
            truth_proj = p1_dir / f".{track}_stage2_probe_truth.jsonl"
            _write_jsonl(
                prompt_proj,
                [{"id": str(r["id"]), "messages": [r["messages"][0]]} for r in dedup_truth],
            )
            _write_jsonl(truth_proj, dedup_truth)
            eval_and_score_model(
                track=track,
                model_key=model_key,
                model_path=spec.local_path,
                candidate_id=f"probe_{model_key}",
                split_prompts_file=prompt_proj,
                split_truth_file=truth_proj,
                out_scored_json=probe_out,
                k=4,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=f"v4_{track}_stage2_probe_k4",
            )

    (p1_dir / "verifier_hygiene_audit.json").write_text(
        json.dumps(hygiene_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def run_plane2_track(args: argparse.Namespace, track: str) -> int:
    verify_manifest_pin()
    model_key = args.model_key
    spec = MODEL_REGISTRY_V4[model_key]
    hp = STAGE_HP_V4[(track, model_key)]
    nproc = len([g for g in args.gpus.split(",") if g.strip()])

    p1_dir = V4_ROOT / "results_v4/plane1" / model_key
    p2_dir = V4_ROOT / "results_v4/plane2" / f"{track}_{model_key}"
    ckpt_root = V4_ROOT / "checkpoints_v4" / f"{track}_{model_key}"
    p2_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    split_fn = "v3_math_search160.jsonl" if track == "math" else "v3_lean_search140.jsonl"
    proto = f"v4_{track}_search160_k8" if track == "math" else f"v4_{track}_search140_k8"
    sp_path = V4_ROOT / "data/splits_v3" / split_fn
    rules_search = V4_ROOT / "configs_v4" / f"gate_rules_v4_{track}_search.json"
    ledger_path = p2_dir / f"trial_ledger_v4_{track}_{model_key}.jsonl"
    b0_json = p1_dir / f"{track}_b0_{model_key}_search.json"

    rl_pool_file = V4_ROOT / "data/corpora_v4" / f"v4_{track}_rl_pool.jsonl"
    max_seq_rl = 11264 if track == "math" else 3584
    max_gen_rl = min(spec.math_max_tokens, 10240) if track == "math" else min(spec.lean_max_tokens, 3072)
    overlong_win = 2048 if track == "math" else 512

    cand_id_to_ckpt: dict[str, str] = {}
    arm_a_jsons: list[str] = []
    arm_b_jsons: list[str] = []

    # ==================================================================
    # ARM A: Zero-SFT On-Policy RLVR (`Pure-B0 -> {DAPO, Dr.GRPO+CISPO}`)
    # ==================================================================
    verify_manifest_pin()
    pure_b0_rollouts = generate_and_verify_rollouts(
        track=track,
        model_key=model_key,
        policy_path=spec.local_path,
        rl_pool_file=rl_pool_file,
        rl_subset_n=int(hp["rl_pool_n"]),
        out_rollouts_jsonl=p2_dir / f"pure_b0_rollouts_{track}_{model_key}.jsonl",
        gpus=args.gpus,
        inject_off_policy=False,
    )

    for algo in ("dapo_asymclip", "drgrpo_cispo"):
        run_tag = f"armA_pureb0_{track}_{model_key}_{algo}"
        out_prefix = ckpt_root / run_tag
        rl_sum = p2_dir / f"{run_tag}_train_summary.json"
        if not rl_sum.exists():
            _run_checked(
                [
                    sys.executable, "-m", "torch.distributed.run", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                    "--track", track,
                    "--algo", algo,
                    "--model-key", model_key,
                    "--rollouts-file", str(pure_b0_rollouts),
                    "--output-dir", str(out_prefix),
                    "--save-milestone-fractions", "1.00",
                    "--summary-output", str(rl_sum),
                    "--max-seq-len", str(max_seq_rl),
                    "--max-gen-tokens", str(max_gen_rl),
                    "--overlong-window", str(overlong_win),
                    "--ppo-epochs", str(hp["rl_ppo_epochs"]),
                    "--groups-per-step", "4",
                    "--max-steps", str(hp["rl_max_steps"]),
                    "--lr", str(hp["rl_lr"]),
                ]
            )
        rl_sum_obj = json.loads(rl_sum.read_text(encoding="utf-8"))
        m_ckpts = rl_sum_obj.get("milestone_checkpoints", {})
        for pct_tag in ("pct50", "pct100"):
            cpath_str = m_ckpts.get(pct_tag) or str(ckpt_root / f"{run_tag}_{pct_tag}")
            cpath = Path(cpath_str)
            if cpath.exists():
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
                arm_a_jsons.append(str(sc_out))

    # ==================================================================
    # ARM B: Horizon-Matched `2-SFT + 3-Proposal RLVR`
    # ==================================================================
    verify_manifest_pin()
    max_seq_sft = 8192 if track == "math" else 2048
    max_tgt_sft = 7680 if track == "math" else 1536
    s1_corpus = V4_ROOT / "data/corpora_v4" / f"v4_{track}_stage1_sft.jsonl"

    s1_id = f"armB_s1_{track}_{model_key}_breadth_sft"
    s1_dir = ckpt_root / s1_id
    s1_factors = ckpt_root / f"{s1_id}_factors.safetensors"
    s1_sum = p2_dir / f"{s1_id}_train_summary.json"
    if not s1_sum.exists():
        _run_checked(
            [
                sys.executable, "-m", "torch.distributed.run", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(SFT_PY),
                "--track", track,
                "--model-key", model_key,
                "--train-file", str(s1_corpus),
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
    cand_id_to_ckpt[s1_id] = str(s1_dir)
    arm_b_jsons.append(str(s1_scored))

    # Generate verified rollouts from `s1` on RL pool (serves both Lean ReST^EM and RLVR if `s1` wins SFT)
    s1_rollouts = generate_and_verify_rollouts(
        track=track,
        model_key=model_key,
        policy_path=str(s1_dir),
        rl_pool_file=rl_pool_file,
        rl_subset_n=int(hp["rl_pool_n"]),
        out_rollouts_jsonl=p2_dir / f"s1_shared_rollouts_{track}_{model_key}.jsonl",
        gpus=args.gpus,
        inject_off_policy=True,
    )

    # Stage II: Model-Aware ZPD Hard-Frontier SFT (+ verified on-policy ReST^EM + 30% S1 replay)
    verify_manifest_pin()
    s2_data = build_model_aware_stage2_sft(
        track=track,
        model_key=model_key,
        probe_scored_json=p1_dir / f"{track}_b0_{model_key}_stage2_probe_k4.json",
        stage1_file=s1_corpus,
        stage2_pool_file=V4_ROOT / "data/corpora_v4" / f"v4_{track}_stage2_frontier_pool.jsonl",
        s1_onpolicy_rollouts=s1_rollouts if track == "lean" else None,
        out_sft_jsonl=p2_dir / f"stage2_zpd_sft_{track}_{model_key}.jsonl",
    )

    s2_id = f"armB_s2_{track}_{model_key}_zpd_sft"
    s2_dir = ckpt_root / s2_id
    s2_factors = ckpt_root / f"{s2_id}_factors.safetensors"
    s2_sum = p2_dir / f"{s2_id}_train_summary.json"
    if not s2_sum.exists():
        _run_checked(
            [
                sys.executable, "-m", "torch.distributed.run", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(SFT_PY),
                "--track", track,
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
    cand_id_to_ckpt[s2_id] = str(s2_dir)
    arm_b_jsons.append(str(s2_scored))

    # Pick SFT* anchor (`s1` vs `s2`)
    pm_key = f"{track}_equal_stratum_avg_at_8"
    s1_m = float(json.loads(s1_scored.read_text(encoding="utf-8"))["metrics"][pm_key])
    s2_m = float(json.loads(s2_scored.read_text(encoding="utf-8"))["metrics"][pm_key])
    if s2_m > s1_m:
        sft_star_id, sft_star_dir, sft_prior_factors = s2_id, s2_dir, f"{s1_factors},{s2_factors}"
        shared_sft_rollouts = generate_and_verify_rollouts(
            track=track,
            model_key=model_key,
            policy_path=str(s2_dir),
            rl_pool_file=rl_pool_file,
            rl_subset_n=int(hp["rl_pool_n"]),
            out_rollouts_jsonl=p2_dir / f"s2_shared_rollouts_{track}_{model_key}.jsonl",
            gpus=args.gpus,
            inject_off_policy=True,
        )
    else:
        sft_star_id, sft_star_dir, sft_prior_factors = s1_id, s1_dir, str(s1_factors)
        shared_sft_rollouts = s1_rollouts

    # Stage III: 3-Proposal RLVR Tournament (`dapo_asymclip`, `drgrpo_cispo`, `gspo_luffy`) from `SFT*`
    verify_manifest_pin()
    for algo in ("dapo_asymclip", "drgrpo_cispo", "gspo_luffy"):
        run_tag = f"armB_s3_{track}_{model_key}_{algo}"
        out_prefix = ckpt_root / run_tag
        rl_sum = p2_dir / f"{run_tag}_train_summary.json"
        m_fracs = "0.50,1.00" if algo == "dapo_asymclip" else "1.00"
        if not rl_sum.exists():
            _run_checked(
                [
                    sys.executable, "-m", "torch.distributed.run", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                    "--track", track,
                    "--algo", algo,
                    "--model-key", model_key,
                    "--prior-fp32-deltas", sft_prior_factors,
                    "--rollouts-file", str(shared_sft_rollouts),
                    "--output-dir", str(out_prefix),
                    "--save-milestone-fractions", m_fracs,
                    "--summary-output", str(rl_sum),
                    "--max-seq-len", str(max_seq_rl),
                    "--max-gen-tokens", str(max_gen_rl),
                    "--overlong-window", str(overlong_win),
                    "--ppo-epochs", str(hp["rl_ppo_epochs"]),
                    "--groups-per-step", "4",
                    "--max-steps", str(hp["rl_max_steps"]),
                    "--lr", str(hp["rl_lr"]),
                ]
            )
        rl_sum_obj = json.loads(rl_sum.read_text(encoding="utf-8"))
        m_ckpts = rl_sum_obj.get("milestone_checkpoints", {})
        for pct_tag in ("pct50", "pct100"):
            cpath_str = m_ckpts.get(pct_tag) or str(ckpt_root / f"{run_tag}_{pct_tag}")
            cpath = Path(cpath_str)
            if cpath.exists():
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
                arm_b_jsons.append(str(sc_out))

    # ==================================================================
    # UNIFIED ORDER-INVARIANT SEARCH TOURNAMENT (`Arm A` + `Arm B`)
    # ==================================================================
    all_cands = [*arm_a_jsons, *arm_b_jsons]
    final_win_file = p2_dir / "search_tournament_winner.json"
    if not final_win_file.exists():
        r_win = _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
                "--candidates", *all_cands,
                "--incumbent", str(b0_json),
                "--baseline", str(b0_json),
                "--rules", str(rules_search),
                "--ledger", str(ledger_path),
                "--round-id", f"round_v4_crossover_{track}_{model_key}",
            ],
            allow_rc=(0, 1, 3),
        )
        win_payload = json.loads(r_win.stdout)
        # Strictly respect Search ROLLBACK: never spend a Confirmation look if no candidate passed Search!
        if win_payload.get("decision") == "PROVISIONAL_KEEP" and win_payload.get("selected_candidate_id"):
            winner_id: str | None = str(win_payload["selected_candidate_id"])
            winner_ckpt: str | None = cand_id_to_ckpt[winner_id]
        else:
            winner_id = None
            winner_ckpt = None

        if arm_a_jsons:
            arm_a_best = max((json.loads(Path(p).read_text(encoding="utf-8")) for p in arm_a_jsons), key=lambda x: float(x["metrics"][pm_key]))
            arm_a_summary: dict[str, Any] = {
                "candidate_id": arm_a_best["candidate_id"],
                "equal_stratum_avg_at_8": arm_a_best["metrics"][pm_key],
                "stop_rate": arm_a_best["metrics"]["stop_rate"],
            }
        else:
            arm_a_summary = {
                "candidate_id": None,
                "status": "SKIPPED_ZERO_ACTIVE_GROUPS",
                "equal_stratum_avg_at_8": None,
                "stop_rate": None,
            }
        arm_b_best = max((json.loads(Path(p).read_text(encoding="utf-8")) for p in arm_b_jsons), key=lambda x: float(x["metrics"][pm_key]))
        win_payload["selected_for_confirm_id"] = winner_id
        win_payload["selected_for_confirm_ckpt"] = winner_ckpt
        win_payload["arm_a_pure_b0_rlvr_best"] = arm_a_summary
        win_payload["arm_b_2sft_rlvr_best"] = {
            "candidate_id": arm_b_best["candidate_id"],
            "equal_stratum_avg_at_8": arm_b_best["metrics"][pm_key],
            "stop_rate": arm_b_best["metrics"]["stop_rate"],
        }
        final_win_file.write_text(json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Prune non-winning intermediate 55GB merged checkpoints while retaining all JSONs, summaries, and LoRA FP32 factors!
    win_info = json.loads(final_win_file.read_text(encoding="utf-8"))
    keep_ckpts = {win_info["selected_for_confirm_ckpt"]} if win_info.get("selected_for_confirm_ckpt") else set()
    for cid, cpath_s in cand_id_to_ckpt.items():
        if cpath_s not in keep_ckpts and Path(cpath_s).exists():
            shutil.rmtree(cpath_s, ignore_errors=True)

    print(f"[Plane-2 DONE] {track} {model_key}: winner={win_info['selected_for_confirm_id']}", flush=True)
    return 0


def run_confirm_and_holdout_track(args: argparse.Namespace, track: str) -> int:
    expected_root = verify_manifest_pin()
    model_key = args.model_key
    spec = MODEL_REGISTRY_V4[model_key]

    p2_dir = V4_ROOT / "results_v4/plane2" / f"{track}_{model_key}"
    ch_dir = V4_ROOT / "results_v4/confirm_holdout" / f"{track}_{model_key}"
    ch_dir.mkdir(parents=True, exist_ok=True)
    final_hold_status = ch_dir / "holdout_status.json"

    win_info = json.loads((p2_dir / "search_tournament_winner.json").read_text(encoding="utf-8"))
    astar_id_opt = win_info.get("selected_for_confirm_id")
    astar_ckpt_opt = win_info.get("selected_for_confirm_ckpt")
    if not astar_id_opt or not astar_ckpt_opt:
        final_hold_status.write_text(
            json.dumps(
                {
                    "status": "SEARCH_ROLLBACK_NO_CONFIRM_OR_HOLDOUT_TOUCH",
                    "track": track,
                    "model_key": model_key,
                    "search_decision": win_info.get("decision"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[Search Rollback] {track} {model_key}: 0 Confirm looks & 0 Holdout touches.", flush=True)
        return 0

    astar_id = str(astar_id_opt)
    astar_ckpt = str(astar_ckpt_opt)

    # Pin cryptographic `.safetensors` + config SHA-256 digests BEFORE Confirmation evaluation!
    cand_pin_json = build_cryptographic_candidate_pin(
        candidate_id=astar_id,
        ckpt_dir=astar_ckpt,
        out_pin_json=ch_dir / f"candidate_pin_{astar_id}.json",
        verify_existing=True,
    )

    confirm_fn = "v3_math_confirm300.jsonl" if track == "math" else "v3_lean_confirm180.jsonl"
    confirm_proto = f"v4_{track}_confirm_k8"
    confirm_sp = V4_ROOT / "data/splits_v3" / confirm_fn

    hold_prompts = V4_ROOT / "data/splits_v3" / f"v3_{track}_holdout_prompts.jsonl"
    hold_truth = V4_ROOT / "eval_v3/sealed_holdout" / f"v3_{track}_holdout.jsonl"
    hold_proto = f"v4_{track}_holdout_k8"

    rules_confirm = V4_ROOT / "configs_v4" / f"gate_rules_v4_{track}_confirm_{model_key}.json"
    confirm_ledger = ch_dir / f"confirm_ledger_v4_{track}_{model_key}.jsonl"

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

    # Re-verify live checkpoint hashes against `cand_pin_json` immediately before `gate-confirm-iut`
    build_cryptographic_candidate_pin(
        candidate_id=astar_id,
        ckpt_dir=astar_ckpt,
        out_pin_json=cand_pin_json,
        verify_existing=True,
    )

    # Run atomic `gate-confirm-iut` (guard output key is `iut_decision`)
    verdict_path = ch_dir / "confirm_iut_verdict.json"
    if not verdict_path.exists():
        r_iut = _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "gate-confirm-iut",
                "--candidate", str(astar_conf_json),
                "--baseline", str(b0_conf_json),
                "--comparator", f"b0_{model_key}={b0_conf_json}",
                "--rules", str(rules_confirm),
                "--ledger", str(confirm_ledger),
                "--trial-prefix", f"confirm_{astar_id}",
                "--candidate-artifacts", str(cand_pin_json),
                "--hypothesis", f"Phase V4 crossover winner {astar_id} beats b0_{model_key} on Confirm_V3",
            ],
            allow_rc=(0, 1, 3),
        )
        verdict_path.write_text(r_iut.stdout, encoding="utf-8")
    verdict_obj = json.loads(verdict_path.read_text(encoding="utf-8"))
    if "iut_decision" not in verdict_obj:
        raise RuntimeError(f"Malformed gate-confirm-iut output missing 'iut_decision': {verdict_obj}")
    promoted = verdict_obj["iut_decision"] == "PROMOTE"
    print(f"[Confirm IUT] {track} {model_key} ({astar_id}): iut_decision={verdict_obj['iut_decision']}", flush=True)

    if not promoted:
        final_hold_status.write_text(
            json.dumps(
                {
                    "status": "CONFIRMED_ROLLBACK_NO_HOLDOUT_TOUCH",
                    "track": track,
                    "model_key": model_key,
                    "candidate_id": astar_id,
                    "confirm_verdict": verdict_obj,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    if final_hold_status.exists():
        return 0

    # Re-verify live checkpoint hashes against `cand_pin_json` immediately before reserving Holdout seal!
    build_cryptographic_candidate_pin(
        candidate_id=astar_id,
        ckpt_dir=astar_ckpt,
        out_pin_json=cand_pin_json,
        verify_existing=True,
    )

    seal_file = ch_dir / f"HOLDOUT_SEAL_v4_{track}_{model_key}.json"
    if seal_file.exists():
        seal_existing = json.loads(seal_file.read_text(encoding="utf-8"))
        if seal_existing.get("status") != "RESERVED" or seal_existing.get("candidate_id") != astar_id:
            raise RuntimeError(f"Refusing Holdout execution on pre-existing non-matching/finalized seal: {seal_existing}")
    else:
        _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "gate-holdout",
                "--phase", "reserve",
                "--seal-file", str(seal_file),
                "--manifest", str(V4_ROOT / "IMMUTABLE_MANIFEST_V4.json"),
                "--expected-root-sha256", expected_root,
                "--ledger", str(confirm_ledger),
                "--rules", str(rules_confirm),
                "--candidate-id", astar_id,
                "--candidate-artifacts", str(cand_pin_json),
            ]
        )

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

    # Re-verify live checkpoint hashes AFTER Holdout scoring and BEFORE finalization!
    build_cryptographic_candidate_pin(
        candidate_id=astar_id,
        ckpt_dir=astar_ckpt,
        out_pin_json=cand_pin_json,
        verify_existing=True,
    )

    r_fin = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "gate-holdout",
            "--phase", "finalize",
            "--seal-file", str(seal_file),
            "--manifest", str(V4_ROOT / "IMMUTABLE_MANIFEST_V4.json"),
            "--expected-root-sha256", expected_root,
            "--ledger", str(confirm_ledger),
            "--rules", str(rules_confirm),
            "--candidate-id", astar_id,
            "--candidate-artifacts", str(cand_pin_json),
            "--candidate-holdout", str(astar_hold_json),
            "--comparator-holdout", f"b0_{model_key}={b0_hold_json}",
        ],
        allow_rc=(0, 1, 3),
    )
    final_hold_status.write_text(r_fin.stdout, encoding="utf-8")
    return 0


def _ensure_dedup_proofnet() -> tuple[Path, Path]:
    pn_raw_p = V4_ROOT / "data/splits_v2/v2_lean_holdout_proofnet174_prompts.jsonl"
    pn_raw_t = V4_ROOT / "eval_v2/sealed_holdout/v2_lean_holdout_proofnet174.jsonl"
    pn_dedup_p = V4_ROOT / "data/longitudinal_proofnet163_prompts.jsonl"
    pn_dedup_t = V4_ROOT / "data/longitudinal_proofnet163_truth.jsonl"
    if pn_raw_p.exists() and pn_raw_t.exists() and not (pn_dedup_p.exists() and pn_dedup_t.exists()):
        for src_f, dst_f in [(pn_raw_p, pn_dedup_p), (pn_raw_t, pn_dedup_t)]:
            seen_ids: set[str] = set()
            kept_lines: list[str] = []
            for ln in src_f.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                rid = str(json.loads(ln)["id"])
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    kept_lines.append(ln)
            dst_f.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return pn_dedup_p, pn_dedup_t


def run_longitudinal(args: argparse.Namespace) -> int:
    model_key = args.model_key
    spec = MODEL_REGISTRY_V4[model_key]
    long_dir = V4_ROOT / "results_v4/longitudinal" / model_key
    long_dir.mkdir(parents=True, exist_ok=True)
    req_tracks = {t.strip() for t in getattr(args, "tracks", "lean,math").split(",") if t.strip()}

    if "math" in req_tracks:
        m_win_file = V4_ROOT / "results_v4/plane2" / f"math_{model_key}" / "search_tournament_winner.json"
        m_verd_file = V4_ROOT / "results_v4/confirm_holdout" / f"math_{model_key}" / "confirm_iut_verdict.json"
        m_cid = m_ckpt = None
        if m_win_file.exists() and m_verd_file.exists():
            m_verd = json.loads(m_verd_file.read_text(encoding="utf-8"))
            if m_verd.get("iut_decision") == "PROMOTE":
                m_win = json.loads(m_win_file.read_text(encoding="utf-8"))
                m_cid = m_win.get("selected_for_confirm_id")
                m_ckpt = m_win.get("selected_for_confirm_ckpt")
        for bench_tag, p_fn, t_fn in [
            ("beyondaime_100", "v2_math_holdout_beyondaime_prompts.jsonl", "v2_math_holdout_beyondaime.jsonl"),
            ("cmimc2025_40", "v2_math_holdout_cmimc2025_prompts.jsonl", "v2_math_holdout_cmimc2025.jsonl"),
            ("brumo2025_30", "v2_math_holdout_brumo2025_prompts.jsonl", "v2_math_holdout_brumo2025.jsonl"),
            ("hmmt_nov2025_30", "v2_math_holdout_hmmt_nov2025_prompts.jsonl", "v2_math_holdout_hmmt_nov2025.jsonl"),
        ]:
            p_path = V4_ROOT / "data/splits_v2" / p_fn
            t_path = V4_ROOT / "eval_v2/sealed_holdout" / t_fn
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
            if m_cid and m_ckpt and Path(m_ckpt).exists():
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

    if "lean" in req_tracks:
        l_win_file = V4_ROOT / "results_v4/plane2" / f"lean_{model_key}" / "search_tournament_winner.json"
        l_verd_file = V4_ROOT / "results_v4/confirm_holdout" / f"lean_{model_key}" / "confirm_iut_verdict.json"
        l_cid = l_ckpt = None
        if l_win_file.exists() and l_verd_file.exists():
            l_verd = json.loads(l_verd_file.read_text(encoding="utf-8"))
            if l_verd.get("iut_decision") == "PROMOTE":
                l_win = json.loads(l_win_file.read_text(encoding="utf-8"))
                l_cid = l_win.get("selected_for_confirm_id")
                l_ckpt = l_win.get("selected_for_confirm_ckpt")
        pn_dedup_p, pn_dedup_t = _ensure_dedup_proofnet()
        for bench_tag, p_path, t_path in [
            ("proofnet_test163", pn_dedup_p, pn_dedup_t),
            (
                "minif2f_test244",
                V4_ROOT / "data/splits/lean_holdout_minif2f_test244_prompts.jsonl",
                V4_ROOT / "eval/sealed_holdout/lean_holdout_minif2f_test244.jsonl",
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
            if l_cid and l_ckpt and Path(l_ckpt).exists():
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


def run_e4b_reference(args: argparse.Namespace) -> int:
    """Evaluates compact active-FLOPs-matched `google/gemma-4-E4B-it` zero-shot on `Holdout_V3` and Longitudinal."""
    verify_manifest_pin()
    spec = MODEL_REGISTRY_V4["gemma4_e4b"]
    for track in ("math", "lean"):
        ch_dir = V4_ROOT / "results_v4/confirm_holdout" / f"{track}_gemma4_e4b"
        ch_dir.mkdir(parents=True, exist_ok=True)
        eval_and_score_model(
            track=track,
            model_key="gemma4_e4b",
            model_path=spec.local_path,
            candidate_id="b0_gemma4_e4b",
            split_prompts_file=V4_ROOT / "data/splits_v3" / f"v3_{track}_holdout_prompts.jsonl",
            split_truth_file=V4_ROOT / "eval_v3/sealed_holdout" / f"v3_{track}_holdout.jsonl",
            out_scored_json=ch_dir / "holdout_b0_gemma4_e4b.json",
            k=8,
            gpus=args.gpus,
            seed=20260325,
            protocol_id=f"v4_{track}_holdout_k8",
        )
    ns = argparse.Namespace(model_key="gemma4_e4b", tracks="lean,math", gpus=args.gpus)
    run_longitudinal(ns)
    return 0


def run_holdout_b0_reference(args: argparse.Namespace, track: str) -> int:
    """
    Post-campaign baseline reference evaluation (`B_0` only) executed strictly AFTER `holdout_status.json`
    is sealed for `(track, model_key)`, so unpromoted cells still report their zero-shot `B_0` Holdout_V3
    baseline row without ever touching Holdout on any rolled-back candidate checkpoint.
    """
    verify_manifest_pin()
    model_key = args.model_key
    spec = MODEL_REGISTRY_V4[model_key]
    ch_dir = V4_ROOT / "results_v4/confirm_holdout" / f"{track}_{model_key}"
    status_file = ch_dir / "holdout_status.json"
    if not status_file.exists():
        raise RuntimeError(f"Cannot run post-lock B0 Holdout reference before {status_file} is sealed!")
    b0_hold_json = ch_dir / f"holdout_b0_{model_key}.json"
    eval_and_score_model(
        track=track,
        model_key=model_key,
        model_path=spec.local_path,
        candidate_id=f"b0_{model_key}",
        split_prompts_file=V4_ROOT / "data/splits_v3" / f"v3_{track}_holdout_prompts.jsonl",
        split_truth_file=V4_ROOT / "eval_v3/sealed_holdout" / f"v3_{track}_holdout.jsonl",
        out_scored_json=b0_hold_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=f"v4_{track}_holdout_k8",
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V4 Cross-Family Governed End-to-End Orchestrator")
    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "plane1",
            "plane2",
            "confirm-and-holdout",
            "holdout-b0-reference",
            "longitudinal",
            "e4b-reference",
            "full-post-freeze",
        ],
    )
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V4.keys()))
    ap.add_argument("--tracks", default="lean,math")
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
    if args.phase == "holdout-b0-reference":
        for tr in track_list:
            run_holdout_b0_reference(args, tr)
        return 0
    if args.phase == "longitudinal":
        return run_longitudinal(args)
    if args.phase == "e4b-reference":
        return run_e4b_reference(args)
    if args.phase == "full-post-freeze":
        for tr in track_list:
            run_plane2_track(args, tr)
            run_confirm_and_holdout_track(args, tr)
            run_holdout_b0_reference(args, tr)
        run_longitudinal(args)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

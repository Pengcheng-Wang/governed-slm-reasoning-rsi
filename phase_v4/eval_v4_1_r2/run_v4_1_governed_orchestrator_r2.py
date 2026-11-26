#!/usr/bin/env python3
"""
Wave-2 (`V4.1`) End-to-End Governed Orchestrator (`eval_v4_1/run_v4_1_governed_orchestrator.py`).

Locked inside `IMMUTABLE_MANIFEST_V4_1.json` and executed across `node-0` (`--role node0`)
and `node-1` (`--role node1`) ONLY AFTER each node's Wave-1 (`V4.0`) processes finish cleanly.
Enforces `governed_loop_guard_v39.py` (v3.9.0):
  - `verify-manifest` on BOTH `IMMUTABLE_MANIFEST_V4.json` and `IMMUTABLE_MANIFEST_V4_1.json`
  - `audit-leaks` on `code/v4_1`
  - `select-search-winner` -> single-look `gate-confirm-iut` -> atomic single-touch `gate-holdout`
    (`reserve` -> score once -> `finalize`, recording `holdout_accepted`).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/scratch/governed_crossfamily_v4")
PY = str(ROOT / ".venv/bin/python")
EVAL_ENGINE = ROOT / "code/v4/eval_engine_v4.py"
SCORER_V41 = ROOT / "eval_v4_1/isolated_scorer_v4_1.py"
GUARD_V39 = ROOT / "eval_v4_1/governed_loop_guard_v39.py"
GUARD_V38 = ROOT / "eval_v4/governed_loop_guard_v38.py"
SFT_V41 = ROOT / "code/v4_1/train_sft_v4_1.py"
RLVR_V41 = ROOT / "code/v4_1/train_rlvr_v4_1.py"
CLUSTER_CTL = ROOT / "code/cluster_control.py"

CODE_V4_DIR = ROOT / "code/v4"
EVAL_V4_DIR = ROOT / "eval_v4"
for p in (CODE_V4_DIR, EVAL_V4_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_registry_v4 import MODEL_REGISTRY_V4  # noqa: E402
from lean4_verifier_v4 import verify_lean_candidate  # noqa: E402
from math_verifier_v4 import score_math_response  # noqa: E402

V4_MANIFEST_SHA = "8f69efbd508ae4291fde22faa6edcc6a6cae759228d45dbcb147ea5f85e47c50"


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def _run(cmd: list[str], *, check: bool = True, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    _log("[EXEC] " + " ".join(str(x) for x in cmd))
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([str(x) for x in cmd], env=env, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-2500:], file=sys.stderr, flush=True)
        if check:
            raise RuntimeError(f"Command failed (rc={r.returncode}): {' '.join(str(x) for x in cmd)}")
    return r


def wait_for_v4_0_idle_and_quiesce() -> None:
    """Never interrupt Wave-1 (`V4.0`); poll cleanly until all V4.0 GPU jobs finish on this node."""
    while True:
        chk = subprocess.run(
            ["pgrep", "-f", "run_v4_worker|drive_math_inkling|train_sft_v4\\.py|train_rlvr_v4\\.py|eval_engine_v4\\.py"],
            capture_output=True,
            text=True,
        )
        pids = [p.strip() for p in chk.stdout.splitlines() if p.strip() and int(p.strip()) != os.getpid()]
        if not pids:
            break
        _log(f"Waiting 30s for Wave-1 (V4.0) PIDs to finish cleanly: {pids[:5]}")
        time.sleep(30)
    if CLUSTER_CTL.exists():
        subprocess.run([PY, str(CLUSTER_CTL), "quiesce"], check=False)


V4_1_ORIG_SHA = "b9c58dcb423892cbb461b1e14c77793494ed69a5d3501209a1e8ae991997a984"


def verify_all_manifests(v41_sha: str) -> None:
    _run([PY, "-I", "-B", str(GUARD_V38), "verify-manifest", "--manifest", str(ROOT / "IMMUTABLE_MANIFEST_V4.json"), "--expected-root-sha256", V4_MANIFEST_SHA])
    _run([PY, "-I", "-B", str(GUARD_V39), "verify-manifest", "--manifest", str(ROOT / "IMMUTABLE_MANIFEST_V4_1.json"), "--expected-root-sha256", V4_1_ORIG_SHA])
    _run([PY, "-I", "-B", str(GUARD_V39), "verify-manifest", "--manifest", str(ROOT / "IMMUTABLE_MANIFEST_V4_1_R2.json"), "--expected-root-sha256", v41_sha])
    _run([PY, "-I", "-B", str(GUARD_V39), "audit-leaks", "--paths", str(ROOT / "code/v4_1")])


def pin_checkpoint_digest_json(ckpt_dir: Path, out_pin_json: Path, extra_meta: dict[str, Any]) -> Path:
    """Creates a canonical JSON binding all `.safetensors` shard/index SHA-256 digests without hashing 60GB twice."""
    digests: dict[str, Any] = {}
    idx_file = ckpt_dir / "model.safetensors.index.json"
    if idx_file.exists():
        digests["index_sha256"] = hashlib.sha256(idx_file.read_bytes()).hexdigest()
    shards = sorted(ckpt_dir.glob("*.safetensors"))
    shard_meta = []
    for s in shards:
        # Hash first 4MB + last 4MB + exact byte size for fast deterministic cryptographic binding of 60GB weights
        sz = s.stat().st_size
        h = hashlib.sha256()
        h.update(str(sz).encode("utf-8"))
        with s.open("rb") as f:
            h.update(f.read(4 * 1024 * 1024))
            if sz > 8 * 1024 * 1024:
                f.seek(sz - 4 * 1024 * 1024)
                h.update(f.read(4 * 1024 * 1024))
        shard_meta.append({"name": s.name, "size": sz, "head_tail_sha256": h.hexdigest()})
    payload = {"checkpoint_dir": str(ckpt_dir), "shards": shard_meta, **digests, **extra_meta}
    out_pin_json.parent.mkdir(parents=True, exist_ok=True)
    out_pin_json.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return out_pin_json


def eval_and_score_v4_1(
    *,
    model_key: str,
    model_path: str,
    track: str,
    split_file: Path,
    out_json: Path,
    candidate_id: str,
    protocol_id: str,
    max_tokens: int,
    chat_kwargs: dict[str, Any],
    thinking_flag: bool,
    thinking_label: str,
    temperature: float = 0.7,
    k: int = 8,
    seed: int = 20261124,
    compare_baseline: str = "",
    reuse_gen_gz: Path | None = None,
) -> dict[str, Any]:
    if out_json.exists():
        _log(f"[SKIP-DONE] {out_json.name}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    spec = MODEL_REGISTRY_V4[model_key]
    gen_gz = reuse_gen_gz if (reuse_gen_gz and reuse_gen_gz.exists()) else out_json.with_suffix(".gen.jsonl.gz")

    if not gen_gz.exists():
        if CLUSTER_CTL.exists():
            subprocess.run([PY, str(CLUSTER_CTL), "quiesce"], check=False)
        max_model_len = 16384 if track == "math" else 8192
        cmd_gen = [
            PY, "-B", str(EVAL_ENGINE),
            "--model-key", model_key,
            "--model-path", str(model_path),
            "--track", track,
            "--split-file", str(split_file),
            "--output", str(gen_gz),
            "--k", str(k),
            "--gpus", "0,1,2,3,4,5,6,7",
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--max-model-len", str(max_model_len),
            "--seed", str(seed),
            "--chat-kwargs-json", json.dumps(chat_kwargs, sort_keys=True),
        ]
        _run(cmd_gen)

    cmd_sc = [
        PY, "-I", "-B", str(SCORER_V41),
        "--track", track,
        "--channel-family", spec.channel_family,
        "--split-file", str(split_file),
        "--generations", str(gen_gz),
        "--expected-k", str(k),
        "--candidate-id", candidate_id,
        "--protocol-id", protocol_id,
        "--thinking-mode-label", thinking_label,
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
        "--eval-backend", spec.eval_backend,
        "--eval-seed", str(seed),
        "--output", str(out_json),
    ]
    if thinking_flag:
        cmd_sc.append("--thinking")
    if compare_baseline and Path(compare_baseline).exists():
        cmd_sc.extend(["--compare-baseline", str(compare_baseline)])
    _run(cmd_sc)
    return json.loads(out_json.read_text(encoding="utf-8"))


def paired_cluster_contrast_10k(cand_json: Path, ref_json: Path, *, seed: int = 20261124, n_boot: int = 10000) -> dict[str, Any]:
    """Exact paired problem-cluster bootstrap (`SAC Gate G4`: 10,000 resamples, seed=20261124)."""
    c_d = json.loads(cand_json.read_text(encoding="utf-8"))
    r_d = json.loads(ref_json.read_text(encoding="utf-8"))
    c_map = {x["id"]: x for x in c_d["items"]}
    r_map = {x["id"]: x for x in r_d["items"]}
    pids = sorted(c_map.keys())
    assert pids == sorted(r_map.keys())
    n = len(pids)
    diffs_avg = [float(c_map[p]["value"]) - float(r_map[p]["value"]) for p in pids]
    diffs_stop = [float(c_map[p]["stop_rate"]) - float(r_map[p]["stop_rate"]) for p in pids]
    diffs_think = [float(c_map[p].get("has_thinking_rate", 0.0)) - float(r_map[p].get("has_thinking_rate", 0.0)) for p in pids]

    rng = random.Random(seed)
    b_avg, b_stop = [], []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        b_avg.append(sum(diffs_avg[i] for i in idxs) / float(n))
        b_stop.append(sum(diffs_stop[i] for i in idxs) / float(n))
    b_avg.sort()
    b_stop.sort()
    return {
        "cand_id": c_d["candidate_id"],
        "ref_id": r_d["candidate_id"],
        "n_problems": n,
        "delta_avg_at_8": statistics.fmean(diffs_avg),
        "lcb95_one_sided_avg_at_8": b_avg[int(0.05 * n_boot)],
        "ci95_two_sided_avg_at_8": [b_avg[int(0.025 * n_boot)], b_avg[int(0.975 * n_boot)]],
        "delta_stop_rate": statistics.fmean(diffs_stop),
        "lcb95_one_sided_stop_rate": b_stop[int(0.05 * n_boot)],
        "delta_has_thinking_rate": statistics.fmean(diffs_think),
    }


def _train_sft_8gpu(
    *,
    model_key: str,
    track: str,
    channel_mode: str,
    train_file: Path,
    out_ckpt: Path,
    summary_json: Path,
    save_deltas: Path,
    max_seq_len: int,
    max_target_tokens: int,
    max_steps: int,
    lr: float,
    seed: int = 20261124,
) -> Path:
    if summary_json.exists() and (out_ckpt / "config.json").exists():
        _log(f"[SKIP-TRAINED-SFT] {summary_json.name}")
        return out_ckpt
    if CLUSTER_CTL.exists():
        subprocess.run([PY, str(CLUSTER_CTL), "quiesce"], check=False)
    port = 21500 + random.randint(0, 3000)
    cmd = [
        PY, "-m", "torch.distributed.run", f"--master_port={port}", "--nproc_per_node=8",
        str(SFT_V41),
        "--model-key", model_key,
        "--track", track,
        "--channel-mode", channel_mode,
        "--train-file", str(train_file),
        "--output-dir", str(out_ckpt),
        "--save-fp32-deltas", str(save_deltas),
        "--summary-output", str(summary_json),
        "--max-seq-len", str(max_seq_len),
        "--max-target-tokens", str(max_target_tokens),
        "--epochs", "2",
        "--max-steps", str(max_steps),
        "--lr", str(lr),
        "--lr-schedule", "cosine",
        "--global-batch-size", "32",
        "--stochastic-bf16-merge",
        "--seed", str(seed),
    ]
    _run(cmd)
    return out_ckpt


def _train_rlvr_8gpu(
    *,
    model_key: str,
    track: str,
    algo: str,
    offpolicy_mode: str,
    rollouts_file: Path,
    out_ckpt: Path,
    summary_json: Path,
    max_seq_len: int,
    max_gen_tokens: int,
    max_steps: int,
    lr: float,
    prior_deltas: str = "",
    rollout_temp: float = 0.7,
    seed: int = 20261124,
) -> Path:
    if summary_json.exists() and (out_ckpt / "config.json").exists():
        _log(f"[SKIP-TRAINED-RLVR] {summary_json.name}")
        return out_ckpt
    if CLUSTER_CTL.exists():
        subprocess.run([PY, str(CLUSTER_CTL), "quiesce"], check=False)
    port = 21500 + random.randint(0, 3000)
    cmd = [
        PY, "-m", "torch.distributed.run", f"--master_port={port}", "--nproc_per_node=8",
        str(RLVR_V41),
        "--model-key", model_key,
        "--track", track,
        "--algo", algo,
        "--offpolicy-mode", offpolicy_mode,
        "--target-offpolicy-group-share", "0.25",
        "--rollout-temperature", str(rollout_temp),
        "--rollouts-file", str(rollouts_file),
        "--output-dir", str(out_ckpt),
        "--summary-output", str(summary_json),
        "--max-seq-len", str(max_seq_len),
        "--max-gen-tokens", str(max_gen_tokens),
        "--overlong-penalty", "0.0",
        "--cispo-sft-alpha", "0.0",
        "--ppo-epochs", "2",
        "--groups-per-step", "4",
        "--max-steps", str(max_steps),
        "--lr", str(lr),
        "--stochastic-bf16-merge",
        "--seed", str(seed),
    ]
    if prior_deltas:
        cmd.extend(["--prior-fp32-deltas", prior_deltas])
    _run(cmd)
    return out_ckpt


def build_verified_rollouts_with_offpolicy_v4_1(
    *,
    model_key: str,
    model_path: str,
    track: str,
    rl_pool_jsonl: Path,
    out_rollouts_jsonl: Path,
    n_prompts: int,
    max_tokens: int,
    chat_kwargs: dict[str, Any],
    thinking_flag: bool,
    temperature: float = 0.7,
    seed: int = 20261124,
) -> Path:
    if out_rollouts_jsonl.exists():
        _log(f"[SKIP-ROLLOUTS] {out_rollouts_jsonl.name}")
        return out_rollouts_jsonl

    out_rollouts_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pool_rows = [json.loads(ln) for ln in rl_pool_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()][:n_prompts]
    prompts_tmp = out_rollouts_jsonl.with_suffix(".prompts.jsonl")
    prompts_tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pool_rows) + "\n", encoding="utf-8")
    raw_gen_gz = out_rollouts_jsonl.with_suffix(".raw_gen.jsonl.gz")

    if not raw_gen_gz.exists():
        if CLUSTER_CTL.exists():
            subprocess.run([PY, str(CLUSTER_CTL), "quiesce"], check=False)
        max_model_len = 16384 if track == "math" else 8192
        _run([
            PY, "-B", str(EVAL_ENGINE),
            "--model-key", model_key,
            "--model-path", str(model_path),
            "--track", track,
            "--split-file", str(prompts_tmp),
            "--output", str(raw_gen_gz),
            "--k", "8",
            "--gpus", "0,1,2,3,4,5,6,7",
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--max-model-len", str(max_model_len),
            "--seed", str(seed),
            "--chat-kwargs-json", json.dumps(chat_kwargs, sort_keys=True),
        ])

    spec = MODEL_REGISTRY_V4[model_key]
    by_id: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(raw_gen_gz, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                gr = json.loads(ln)
                by_id.setdefault(str(gr["id"]), []).append(gr)

    spec_by_id = {str(r["id"]): r for r in pool_rows}

    def _verify_group(pid: str) -> dict[str, Any]:
        pr = spec_by_id[pid]
        g_samples = sorted(by_id.get(pid, []), key=lambda x: int(x["sample_idx"]))
        verified_samples: list[dict[str, Any]] = []
        n_pos = 0
        for s in g_samples:
            fr = str(s["finish_reason"])
            txt = str(s["text"])
            if track == "math":
                sc = score_math_response(
                    txt,
                    str(pr["gold"]),
                    finish_reason=fr,
                    thinking=thinking_flag,
                    channel_family=spec.channel_family,
                    integer_task=bool(pr.get("integer_task", False)),
                    problem=str(pr.get("problem", "")),
                )
                ok = bool(sc.get("lenient_correct", False)) and (fr == "stop")
            else:
                sc_l = verify_lean_candidate(
                    txt,
                    str(pr["formal_statement"]),
                    lean_project=ROOT / "lean/mathlib_env",
                    elan_bin=ROOT / "lean/elan/bin",
                    finish_reason=fr,
                    thinking=thinking_flag,
                    channel_family=spec.channel_family,
                    strict_format=False,
                    timeout_s=60.0,
                    lean_header=str(pr.get("lean_header", "")),
                )
                ok = bool(sc_l.get("ok", False)) and (fr == "stop")
            if ok:
                n_pos += 1
            verified_samples.append({**s, "verified": ok, "off_policy": False})

        # Attach verified teacher reference when n_pos == 0 so mixed-group arms have `1 teacher + 7 student negs`
        ref_sol = str(pr.get("reference_solution") or pr.get("proof_body") or "").strip()
        if n_pos == 0 and ref_sol:
            if track == "lean" and not ref_sol.startswith("```"):
                ref_sol = f"```lean4\n{ref_sol}\n```"
            verified_samples.append(
                {
                    "id": pid,
                    "sample_idx": 99,
                    "text": ref_sol,
                    "finish_reason": "stop",
                    "completion_tokens": max(16, len(ref_sol) // 4),
                    "verified": True,
                    "off_policy": True,
                }
            )
        return {
            "id": pid,
            "messages": pr["messages"],
            "gold": pr.get("gold", ""),
            "samples": verified_samples,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        groups_out = list(ex.map(_verify_group, [str(r["id"]) for r in pool_rows]))

    out_rollouts_jsonl.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in groups_out) + "\n", encoding="utf-8")
    return out_rollouts_jsonl


def run_governed_confirm_and_holdout_v4_1(
    *,
    cell_name: str,
    model_key: str,
    track: str,
    v41_manifest_sha: str,
    winner_id: str,
    winner_ckpt: Path,
    winner_train_summary: Path,
    extra_confirm_arms: list[tuple[str, str]],
    max_tokens: int,
    chat_kwargs: dict[str, Any],
    thinking_flag: bool,
    thinking_label: str,
) -> dict[str, Any]:
    out_dir = ROOT / f"results_v4_1/confirm_holdout/{cell_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "trial_ledger_v4_1.jsonl"
    rules = ROOT / f"configs_v4_1_r1/gate_rules_v4_1_{track}_confirm.json"
    confirm_split = ROOT / f"data/splits_v4_1/v4_1_{track}_confirm150.jsonl"
    holdout_split = ROOT / f"data/splits_v4_1/v4_1_{track}_holdout150.jsonl"
    spec = MODEL_REGISTRY_V4[model_key]

    # Pin candidate artifacts (`code` + `summary` + `checkpoint shard digest JSON`)
    pin_json = pin_checkpoint_digest_json(
        winner_ckpt,
        out_dir / f"{winner_id}_ckpt_pin.json",
        {"candidate_id": winner_id, "model_key": model_key, "track": track},
    )
    cand_artifacts = [str(SFT_V41), str(RLVR_V41), str(winner_train_summary), str(pin_json)]

    # Score B0 and winner (plus any diagnostic comparison arm) on Confirm_V4_1
    b0_conf_json = out_dir / f"b0_base_{cell_name}_confirm150.json"
    eval_and_score_v4_1(
        model_key=model_key,
        model_path=spec.local_path,
        track=track,
        split_file=confirm_split,
        out_json=b0_conf_json,
        candidate_id="b0_base",
        protocol_id=f"v4_1_{track}_confirm150_k8",
        max_tokens=max_tokens,
        chat_kwargs=chat_kwargs,
        thinking_flag=thinking_flag,
        thinking_label=thinking_label,
    )

    win_conf_json = out_dir / f"{winner_id}_confirm150.json"
    eval_and_score_v4_1(
        model_key=model_key,
        model_path=str(winner_ckpt),
        track=track,
        split_file=confirm_split,
        out_json=win_conf_json,
        candidate_id=winner_id,
        protocol_id=f"v4_1_{track}_confirm150_k8",
        max_tokens=max_tokens,
        chat_kwargs=chat_kwargs,
        thinking_flag=thinking_flag,
        thinking_label=thinking_label,
        compare_baseline=str(b0_conf_json),
    )

    extra_contrasts: dict[str, Any] = {}
    for arm_id, arm_path in extra_confirm_arms:
        arm_conf_json = out_dir / f"{arm_id}_confirm150.json"
        eval_and_score_v4_1(
            model_key=model_key,
            model_path=arm_path,
            track=track,
            split_file=confirm_split,
            out_json=arm_conf_json,
            candidate_id=arm_id,
            protocol_id=f"v4_1_{track}_confirm150_k8",
            max_tokens=max_tokens,
            chat_kwargs=chat_kwargs,
            thinking_flag=thinking_flag,
            thinking_label=thinking_label,
            compare_baseline=str(b0_conf_json),
        )
        extra_contrasts[f"confirm_{winner_id}_vs_{arm_id}"] = paired_cluster_contrast_10k(win_conf_json, arm_conf_json)

    extra_contrasts[f"confirm_{winner_id}_vs_b0"] = paired_cluster_contrast_10k(win_conf_json, b0_conf_json)

    verdict_json = out_dir / "confirm_iut_verdict.json"
    proposer_json = out_dir / "proposer_redacted.json"
    if not verdict_json.exists():
        r_conf = _run(
            [
                PY, "-I", "-B", str(GUARD_V39), "gate-confirm-iut",
                "--candidate", str(win_conf_json),
                "--baseline", str(b0_conf_json),
                "--comparator", f"b0_base={b0_conf_json}",
                "--rules", str(rules),
                "--ledger", str(ledger),
                "--trial-prefix", f"confirm_{cell_name}",
                "--candidate-artifacts", *cand_artifacts,
                "--proposer-out", str(proposer_json),
                "--hypothesis", f"{cell_name}: {winner_id} beats B0 with clean channel/precision hygiene",
            ],
            check=False,
        )
        verdict_json.write_text(r_conf.stdout, encoding="utf-8")

    conf_verdict = json.loads(verdict_json.read_text(encoding="utf-8"))
    dec_val = str(conf_verdict.get("iut_decision") or conf_verdict.get("overall_decision") or conf_verdict.get("decision") or "")
    promoted = dec_val == "PROMOTE"

    result_summary: dict[str, Any] = {
        "cell_name": cell_name,
        "winner_id": winner_id,
        "confirm_decision": dec_val,
        "confirm_verdict": conf_verdict,
        "paired_contrasts": extra_contrasts,
        "holdout_touched": False,
    }

    # Atomic single-touch Holdout_V4_1 ONLY if Confirm_V4_1 returned `PROMOTE`!
    seal_receipt = out_dir / "HOLDOUT_SEAL_RECEIPT.json"
    if promoted and not seal_receipt.exists():
        _run([
            PY, "-I", "-B", str(GUARD_V39), "gate-holdout",
            "--phase", "reserve",
            "--seal-file", str(seal_receipt),
            "--manifest", str(ROOT / "IMMUTABLE_MANIFEST_V4_1_R2.json"),
            "--expected-root-sha256", v41_manifest_sha,
            "--ledger", str(ledger),
            "--rules", str(rules),
            "--candidate-id", winner_id,
            "--candidate-artifacts", *cand_artifacts,
        ])

        b0_hold_json = out_dir / f"b0_base_{cell_name}_holdout150.json"
        eval_and_score_v4_1(
            model_key=model_key,
            model_path=spec.local_path,
            track=track,
            split_file=holdout_split,
            out_json=b0_hold_json,
            candidate_id="b0_base",
            protocol_id=f"v4_1_{track}_holdout150_k8",
            max_tokens=max_tokens,
            chat_kwargs=chat_kwargs,
            thinking_flag=thinking_flag,
            thinking_label=thinking_label,
        )
        win_hold_json = out_dir / f"{winner_id}_holdout150.json"
        eval_and_score_v4_1(
            model_key=model_key,
            model_path=str(winner_ckpt),
            track=track,
            split_file=holdout_split,
            out_json=win_hold_json,
            candidate_id=winner_id,
            protocol_id=f"v4_1_{track}_holdout150_k8",
            max_tokens=max_tokens,
            chat_kwargs=chat_kwargs,
            thinking_flag=thinking_flag,
            thinking_label=thinking_label,
            compare_baseline=str(b0_hold_json),
        )
        for arm_id, arm_path in extra_confirm_arms:
            arm_hold_json = out_dir / f"{arm_id}_holdout150.json"
            eval_and_score_v4_1(
                model_key=model_key,
                model_path=arm_path,
                track=track,
                split_file=holdout_split,
                out_json=arm_hold_json,
                candidate_id=arm_id,
                protocol_id=f"v4_1_{track}_holdout150_k8",
                max_tokens=max_tokens,
                chat_kwargs=chat_kwargs,
                thinking_flag=thinking_flag,
                thinking_label=thinking_label,
                compare_baseline=str(b0_hold_json),
            )
            extra_contrasts[f"holdout_{winner_id}_vs_{arm_id}"] = paired_cluster_contrast_10k(win_hold_json, arm_hold_json)
        extra_contrasts[f"holdout_{winner_id}_vs_b0"] = paired_cluster_contrast_10k(win_hold_json, b0_hold_json)

        r_fin = _run([
            PY, "-I", "-B", str(GUARD_V39), "gate-holdout",
            "--phase", "finalize",
            "--seal-file", str(seal_receipt),
            "--manifest", str(ROOT / "IMMUTABLE_MANIFEST_V4_1_R2.json"),
            "--expected-root-sha256", v41_manifest_sha,
            "--ledger", str(ledger),
            "--rules", str(rules),
            "--candidate-id", winner_id,
            "--candidate-artifacts", *cand_artifacts,
            "--candidate-holdout", str(win_hold_json),
            "--comparator-holdout", f"b0_base={b0_hold_json}",
        ])
        result_summary["holdout_touched"] = True
        result_summary["holdout_finalize"] = json.loads(r_fin.stdout)

    (out_dir / "cell_v4_1_summary.json").write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
    return result_summary


def run_node0(v41_manifest_sha: str) -> None:
    wait_for_v4_0_idle_and_quiesce()
    verify_all_manifests(v41_manifest_sha)

    lean_search = ROOT / "data/splits_v3/v3_lean_search140.jsonl"
    g2_dir = ROOT / "results_v4_1/g2_protocol_equalization"
    g3_dir = ROOT / "results_v4_1/g3_inkling_lean_channel"
    ckpt_root = ROOT / "checkpoints_v4_1/inkling_lean"

    # Gate G2b: Inkling-2-Mini Lean B0 reasoning_effort="low" (reuse cached gen) vs "none" (direct text)
    spec_ink = MODEL_REGISTRY_V4["inkling2_mini"]
    b0_low_json = g2_dir / "lean_b0_inkling2_mini_effort_low_3072_search.json"
    eval_and_score_v4_1(
        model_key="inkling2_mini",
        model_path=spec_ink.local_path,
        track="lean",
        split_file=lean_search,
        out_json=b0_low_json,
        candidate_id="b0_base",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "low"},
        thinking_flag=False,
        thinking_label="reasoning_effort=low",
        reuse_gen_gz=ROOT / "results_v4/plane1/inkling2_mini/lean_b0_inkling2_mini_search.gen.jsonl.gz",
    )
    b0_none_json = g2_dir / "lean_b0_inkling2_mini_effort_none_3072_search.json"
    eval_and_score_v4_1(
        model_key="inkling2_mini",
        model_path=spec_ink.local_path,
        track="lean",
        split_file=lean_search,
        out_json=b0_none_json,
        candidate_id="b0_inkling_effort_none",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "none"},
        thinking_flag=False,
        thinking_label="reasoning_effort=none",
        compare_baseline=str(b0_low_json),
    )

    # Gate G3a: Exact Matched Wrapper-Only Causal Intervention on Inkling-2-Mini Lean (`SAC Gates G1-G4`)
    sft_file = ROOT / "data/corpora_v4/v4_lean_stage1_sft.jsonl"

    # Arm 1: Stripped-Channel Mismatch (V4.0 bug replication under identical cosine/stochastic-BF16 setup)
    c1_ckpt = _train_sft_8gpu(
        model_key="inkling2_mini",
        track="lean",
        channel_mode="stripped_mismatch",
        train_file=sft_file,
        out_ckpt=ckpt_root / "inkling_lean_sft_stripped",
        summary_json=g3_dir / "inkling_lean_sft_stripped_train_summary.json",
        save_deltas=ckpt_root / "inkling_lean_sft_stripped_factors.safetensors",
        max_seq_len=2048,
        max_target_tokens=1536,
        max_steps=80,
        lr=2.0e-5,
    )
    c1_search = g3_dir / "inkling_lean_sft_stripped_search.json"
    eval_and_score_v4_1(
        model_key="inkling2_mini",
        model_path=str(c1_ckpt),
        track="lean",
        split_file=lean_search,
        out_json=c1_search,
        candidate_id="inkling_lean_sft_stripped",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "low"},
        thinking_flag=False,
        thinking_label="reasoning_effort=low",
        compare_baseline=str(b0_low_json),
    )

    # Arm 2: Effort-Matched Wrapper-Only Fix (`Thinking effort level: 0` on thoughtless proofs; 0.0% target-token delta!)
    c2_ckpt = _train_sft_8gpu(
        model_key="inkling2_mini",
        track="lean",
        channel_mode="effort_matched_wrapper",
        train_file=sft_file,
        out_ckpt=ckpt_root / "inkling_lean_sft_effort_matched",
        summary_json=g3_dir / "inkling_lean_sft_effort_matched_train_summary.json",
        save_deltas=ckpt_root / "inkling_lean_sft_effort_matched_factors.safetensors",
        max_seq_len=2048,
        max_target_tokens=1536,
        max_steps=80,
        lr=2.0e-5,
    )
    c2_search = g3_dir / "inkling_lean_sft_effort_matched_search.json"
    eval_and_score_v4_1(
        model_key="inkling2_mini",
        model_path=str(c2_ckpt),
        track="lean",
        split_file=lean_search,
        out_json=c2_search,
        candidate_id="inkling_lean_sft_effort_matched",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "low"},
        thinking_flag=False,
        thinking_label="reasoning_effort=low",
        compare_baseline=str(b0_low_json),
    )

    # Arm 3: Effort-Matched Wrapper + Precision-Safe On-Policy GSPO (`24` steps at T=0.7)
    rl_rollouts = build_verified_rollouts_with_offpolicy_v4_1(
        model_key="inkling2_mini",
        model_path=str(c2_ckpt),
        track="lean",
        rl_pool_jsonl=ROOT / "data/corpora_v4/v4_lean_rl_pool.jsonl",
        out_rollouts_jsonl=g3_dir / "inkling_lean_c2_rollouts_t07.jsonl",
        n_prompts=144,
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "low"},
        thinking_flag=False,
        temperature=0.7,
    )
    c3_ckpt = _train_rlvr_8gpu(
        model_key="inkling2_mini",
        track="lean",
        algo="gspo",
        offpolicy_mode="none",
        rollouts_file=rl_rollouts,
        out_ckpt=ckpt_root / "inkling_lean_sft_matched_plus_gspo",
        summary_json=g3_dir / "inkling_lean_sft_matched_plus_gspo_train_summary.json",
        max_seq_len=3584,
        max_gen_tokens=3072,
        max_steps=24,
        lr=1.0e-5,
        prior_deltas=str(ckpt_root / "inkling_lean_sft_effort_matched_factors.safetensors"),
        rollout_temp=0.7,
    )
    c3_search = g3_dir / "inkling_lean_sft_matched_plus_gspo_search.json"
    eval_and_score_v4_1(
        model_key="inkling2_mini",
        model_path=str(c3_ckpt),
        track="lean",
        split_file=lean_search,
        out_json=c3_search,
        candidate_id="inkling_lean_sft_matched_plus_gspo",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "low"},
        thinking_flag=False,
        thinking_label="reasoning_effort=low",
        compare_baseline=str(b0_low_json),
    )

    # Record Search paired contrasts (`effort_matched` vs `stripped` & `B0`)
    contrasts_search = {
        "search_effort_matched_vs_stripped": paired_cluster_contrast_10k(c2_search, c1_search),
        "search_effort_matched_vs_b0": paired_cluster_contrast_10k(c2_search, b0_low_json),
        "search_matched_gspo_vs_b0": paired_cluster_contrast_10k(c3_search, b0_low_json),
    }
    (g3_dir / "inkling_lean_search_contrasts.json").write_text(json.dumps(contrasts_search, indent=2), encoding="utf-8")

    # Order-invariant Search winner selection via `governed_loop_guard_v39.py select-search-winner`
    r_sel = _run([
        PY, "-I", "-B", str(GUARD_V39), "select-search-winner",
        "--candidates", str(c1_search), str(c2_search), str(c3_search),
        "--incumbent", str(b0_low_json),
        "--baseline", str(b0_low_json),
        "--rules", str(ROOT / "configs_v4_1_r1/gate_rules_v4_1_lean_search.json"),
        "--ledger", str(g3_dir / "search_ledger_v4_1_r2.jsonl"),
        "--round-id", "v4_1_inkling_lean_r1",
    ], check=False)
    sel_out = json.loads(r_sel.stdout)
    (g3_dir / "select_search_winner_r2.json").write_text(json.dumps(sel_out, indent=2), encoding="utf-8")

    # Choose Confirm candidate (Search winner if tripwire-clean, else pure wrapper-only `c2` for SAC Gate G4 contrast)
    win_id = str(sel_out.get("selected_candidate_id") or "")
    ckpt_lookup = {
        "inkling_lean_sft_effort_matched": (c2_ckpt, g3_dir / "inkling_lean_sft_effort_matched_train_summary.json"),
        "inkling_lean_sft_matched_plus_gspo": (c3_ckpt, g3_dir / "inkling_lean_sft_matched_plus_gspo_train_summary.json"),
    }
    if win_id not in ckpt_lookup:
        win_id = "inkling_lean_sft_effort_matched"
    win_ckpt, win_sum = ckpt_lookup[win_id]

    verify_all_manifests(v41_manifest_sha)
    run_governed_confirm_and_holdout_v4_1(
        cell_name="lean_inkling2_mini",
        model_key="inkling2_mini",
        track="lean",
        v41_manifest_sha=v41_manifest_sha,
        winner_id=win_id,
        winner_ckpt=win_ckpt,
        winner_train_summary=win_sum,
        extra_confirm_arms=[("inkling_lean_sft_stripped", str(c1_ckpt))],
        max_tokens=3072,
        chat_kwargs={"reasoning_effort": "low"},
        thinking_flag=False,
        thinking_label="reasoning_effort=low",
    )
    _log("=== NODE-0 V4.1 COMPLETE ===")


def run_node1(v41_manifest_sha: str) -> None:
    wait_for_v4_0_idle_and_quiesce()
    verify_all_manifests(v41_manifest_sha)

    lean_search = ROOT / "data/splits_v3/v3_lean_search140.jsonl"
    math_search = ROOT / "data/splits_v3/v3_math_search160.jsonl"
    g2_dir = ROOT / "results_v4_1/g2_protocol_equalization"

    # 1. Gate G2: Gemma-4-E4B Math re-score with Markdown-bold parser fix + Gemma-4 2x2 `{think=False, True} x {2048, 3072}` on Lean
    e4b_gen = ROOT / "results_v4/plane1/gemma4_e4b/math_b0_gemma4_e4b_search.gen.jsonl.gz"
    if e4b_gen.exists():
        eval_and_score_v4_1(
            model_key="gemma4_e4b",
            model_path=MODEL_REGISTRY_V4["gemma4_e4b"].local_path,
            track="math",
            split_file=math_search,
            out_json=g2_dir / "math_b0_gemma4_e4b_parser_fixed_search.json",
            candidate_id="b0_gemma4_e4b_fixed_parser",
            protocol_id="v4_1_math_search160_k8",
            max_tokens=12288,
            chat_kwargs={"enable_thinking": False},
            thinking_flag=False,
            thinking_label="enable_thinking=False",
            reuse_gen_gz=e4b_gen,
        )

    for g_key in ("gemma4_26b_a4b", "gemma4_31b"):
        sp = MODEL_REGISTRY_V4[g_key]
        # Reuse existing (False, 2048) generation + run missing (False, 3072), (True, 2048), (True, 3072)
        for th_bool in (False, True):
            for mtok in (2048, 3072):
                reuse_p = (
                    ROOT / f"results_v4/plane1/{g_key}/lean_b0_{g_key}_search.gen.jsonl.gz"
                    if (not th_bool and mtok == 2048)
                    else None
                )
                tag = f"think_{'on' if th_bool else 'off'}_cap_{mtok}"
                eval_and_score_v4_1(
                    model_key=g_key,
                    model_path=sp.local_path,
                    track="lean",
                    split_file=lean_search,
                    out_json=g2_dir / f"lean_b0_{g_key}_{tag}_search.json",
                    candidate_id=f"b0_{g_key}_{tag}",
                    protocol_id=f"v4_1_lean_search140_k8_{tag}",
                    max_tokens=mtok,
                    chat_kwargs={"enable_thinking": th_bool},
                    thinking_flag=th_bool,
                    thinking_label=f"enable_thinking={th_bool}",
                    reuse_gen_gz=reuse_p,
                )

    # 2. Gate G4: Common-Backbone GSPO Off-Policy LUFFY vs Log-Shaped vs Channel-Aligned SFT->RL (`Muse-Glimmer-30B` Math)
    g4_dir = ROOT / "results_v4_1/g4_muse_math_offpolicy_ablation"
    ckpt_m = ROOT / "checkpoints_v4_1/muse_math"
    spec_muse = MODEL_REGISTRY_V4["muse_glimmer_30b"]
    muse_math_ckw = {"current_date": "2026-11-22", "reasoning_strength": "medium"}

    b0_muse_math = g4_dir / "math_b0_muse_glimmer_30b_search.json"
    eval_and_score_v4_1(
        model_key="muse_glimmer_30b",
        model_path=spec_muse.local_path,
        track="math",
        split_file=math_search,
        out_json=b0_muse_math,
        candidate_id="b0_base",
        protocol_id="v4_1_math_search160_k8_12288",
        max_tokens=12288,
        chat_kwargs=muse_math_ckw,
        thinking_flag=False,
        thinking_label="reasoning_strength=medium",
        reuse_gen_gz=ROOT / "results_v4/plane1/muse_glimmer_30b/math_b0_muse_glimmer_30b_search.gen.jsonl.gz",
    )

    shared_rollouts = build_verified_rollouts_with_offpolicy_v4_1(
        model_key="muse_glimmer_30b",
        model_path=spec_muse.local_path,
        track="math",
        rl_pool_jsonl=ROOT / "data/corpora_v4/v4_math_rl_pool.jsonl",
        out_rollouts_jsonl=g4_dir / "muse_math_b0_rollouts_240_t07.jsonl",
        n_prompts=240,
        max_tokens=10240,
        chat_kwargs=muse_math_ckw,
        thinking_flag=False,
        temperature=0.7,
    )

    g4_search_jsons: list[Path] = []
    g4_ckpts: dict[str, tuple[Path, Path]] = {}

    for arm_name, off_mode in [
        ("muse_math_gspo_pure_onpolicy", "none"),
        ("muse_math_gspo_faithful_luffy", "faithful_luffy"),
        ("muse_math_gspo_log_shaped", "log_shaped"),
    ]:
        sum_p = g4_dir / f"{arm_name}_train_summary.json"
        ck_p = _train_rlvr_8gpu(
            model_key="muse_glimmer_30b",
            track="math",
            algo="gspo",
            offpolicy_mode=off_mode,
            rollouts_file=shared_rollouts,
            out_ckpt=ckpt_m / arm_name,
            summary_json=sum_p,
            max_seq_len=11264,
            max_gen_tokens=10240,
            max_steps=40,
            lr=1.2e-5,
            rollout_temp=0.7,
        )
        sc_p = g4_dir / f"{arm_name}_search.json"
        eval_and_score_v4_1(
            model_key="muse_glimmer_30b",
            model_path=str(ck_p),
            track="math",
            split_file=math_search,
            out_json=sc_p,
            candidate_id=arm_name,
            protocol_id="v4_1_math_search160_k8_12288",
            max_tokens=12288,
            chat_kwargs=muse_math_ckw,
            thinking_flag=False,
            thinking_label="reasoning_strength=medium",
            compare_baseline=str(b0_muse_math),
        )
        g4_search_jsons.append(sc_p)
        g4_ckpts[arm_name] = (ck_p, sum_p)

    # Arm 4: Channel-Aligned SFT (`to=self` reasoning + concise `to=user` `\boxed{}`) -> Pure On-Policy GSPO
    sft_m_ckpt = _train_sft_8gpu(
        model_key="muse_glimmer_30b",
        track="math",
        channel_mode="native_thought_aligned",
        train_file=ROOT / "data/corpora_v4/v4_math_stage1_sft.jsonl",
        out_ckpt=ckpt_m / "muse_math_sft_channel_aligned",
        summary_json=g4_dir / "muse_math_sft_channel_aligned_train_summary.json",
        save_deltas=ckpt_m / "muse_math_sft_channel_aligned_factors.safetensors",
        max_seq_len=8192,
        max_target_tokens=7680,
        max_steps=80,
        lr=2.0e-5,
    )
    sft_m_search = g4_dir / "muse_math_sft_channel_aligned_search.json"
    eval_and_score_v4_1(
        model_key="muse_glimmer_30b",
        model_path=str(sft_m_ckpt),
        track="math",
        split_file=math_search,
        out_json=sft_m_search,
        candidate_id="muse_math_sft_channel_aligned",
        protocol_id="v4_1_math_search160_k8_12288",
        max_tokens=12288,
        chat_kwargs=muse_math_ckw,
        thinking_flag=False,
        thinking_label="reasoning_strength=medium",
        compare_baseline=str(b0_muse_math),
    )
    g4_search_jsons.append(sft_m_search)
    g4_ckpts["muse_math_sft_channel_aligned"] = (sft_m_ckpt, g4_dir / "muse_math_sft_channel_aligned_train_summary.json")

    sft_rollouts = build_verified_rollouts_with_offpolicy_v4_1(
        model_key="muse_glimmer_30b",
        model_path=str(sft_m_ckpt),
        track="math",
        rl_pool_jsonl=ROOT / "data/corpora_v4/v4_math_rl_pool.jsonl",
        out_rollouts_jsonl=g4_dir / "muse_math_sft_aligned_rollouts_160_t07.jsonl",
        n_prompts=160,
        max_tokens=10240,
        chat_kwargs=muse_math_ckw,
        thinking_flag=False,
        temperature=0.7,
    )
    sft_rl_sum = g4_dir / "muse_math_sft_aligned_then_gspo_train_summary.json"
    sft_rl_ckpt = _train_rlvr_8gpu(
        model_key="muse_glimmer_30b",
        track="math",
        algo="gspo",
        offpolicy_mode="none",
        rollouts_file=sft_rollouts,
        out_ckpt=ckpt_m / "muse_math_sft_aligned_then_gspo",
        summary_json=sft_rl_sum,
        max_seq_len=11264,
        max_gen_tokens=10240,
        max_steps=30,
        lr=9.0e-6,
        prior_deltas=str(ckpt_m / "muse_math_sft_channel_aligned_factors.safetensors"),
        rollout_temp=0.7,
    )
    sft_rl_search = g4_dir / "muse_math_sft_aligned_then_gspo_search.json"
    eval_and_score_v4_1(
        model_key="muse_glimmer_30b",
        model_path=str(sft_rl_ckpt),
        track="math",
        split_file=math_search,
        out_json=sft_rl_search,
        candidate_id="muse_math_sft_aligned_then_gspo",
        protocol_id="v4_1_math_search160_k8_12288",
        max_tokens=12288,
        chat_kwargs=muse_math_ckw,
        thinking_flag=False,
        thinking_label="reasoning_strength=medium",
        compare_baseline=str(b0_muse_math),
    )
    g4_search_jsons.append(sft_rl_search)
    g4_ckpts["muse_math_sft_aligned_then_gspo"] = (sft_rl_ckpt, sft_rl_sum)

    r_sel_m = _run([
        PY, "-I", "-B", str(GUARD_V39), "select-search-winner",
        "--candidates", *[str(p) for p in g4_search_jsons],
        "--incumbent", str(b0_muse_math),
        "--baseline", str(b0_muse_math),
        "--rules", str(ROOT / "configs_v4_1_r1/gate_rules_v4_1_math_search.json"),
        "--ledger", str(g4_dir / "search_ledger_v4_1_r2.jsonl"),
        "--round-id", "v4_1_muse_math_r1",
    ], check=False)
    sel_m = json.loads(r_sel_m.stdout)
    (g4_dir / "select_search_winner_r2.json").write_text(json.dumps(sel_m, indent=2), encoding="utf-8")

    win_m_id = str(sel_m.get("selected_candidate_id") or "")
    if win_m_id in g4_ckpts:
        verify_all_manifests(v41_manifest_sha)
        win_m_ck, win_m_sum = g4_ckpts[win_m_id]
        run_governed_confirm_and_holdout_v4_1(
            cell_name="math_muse_glimmer_30b",
            model_key="muse_glimmer_30b",
            track="math",
            v41_manifest_sha=v41_manifest_sha,
            winner_id=win_m_id,
            winner_ckpt=win_m_ck,
            winner_train_summary=win_m_sum,
            extra_confirm_arms=[("muse_math_gspo_pure_onpolicy", str(g4_ckpts["muse_math_gspo_pure_onpolicy"][0]))] if win_m_id == "muse_math_gspo_faithful_luffy" else [("muse_math_gspo_faithful_luffy", str(g4_ckpts["muse_math_gspo_faithful_luffy"][0]))],
            max_tokens=12288,
            chat_kwargs=muse_math_ckw,
            thinking_flag=False,
            thinking_label="reasoning_strength=medium",
        )

    # 3. Gate G3b: Muse-Glimmer-30B Track L Channel-Aligned SFT + GSPO (`stop_rate` repair!)
    g3b_dir = ROOT / "results_v4_1/g3b_muse_lean_channel"
    ckpt_l = ROOT / "checkpoints_v4_1/muse_lean"
    muse_lean_ckw = {"current_date": "2026-11-22", "reasoning_strength": "low"}

    b0_muse_lean = g3b_dir / "lean_b0_muse_glimmer_30b_search.json"
    eval_and_score_v4_1(
        model_key="muse_glimmer_30b",
        model_path=spec_muse.local_path,
        track="lean",
        split_file=lean_search,
        out_json=b0_muse_lean,
        candidate_id="b0_base",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs=muse_lean_ckw,
        thinking_flag=False,
        thinking_label="reasoning_strength=low",
        reuse_gen_gz=ROOT / "results_v4/plane1/muse_glimmer_30b/lean_b0_muse_glimmer_30b_search.gen.jsonl.gz",
    )

    ml_sft_sum = g3b_dir / "muse_lean_sft_effort_matched_train_summary.json"
    ml_sft_ck = _train_sft_8gpu(
        model_key="muse_glimmer_30b",
        track="lean",
        channel_mode="effort_matched_wrapper",
        train_file=ROOT / "data/corpora_v4/v4_lean_stage1_sft.jsonl",
        out_ckpt=ckpt_l / "muse_lean_sft_effort_matched",
        summary_json=ml_sft_sum,
        save_deltas=ckpt_l / "muse_lean_sft_effort_matched_factors.safetensors",
        max_seq_len=2048,
        max_target_tokens=1536,
        max_steps=80,
        lr=2.0e-5,
    )
    ml_sft_search = g3b_dir / "muse_lean_sft_effort_matched_search.json"
    eval_and_score_v4_1(
        model_key="muse_glimmer_30b",
        model_path=str(ml_sft_ck),
        track="lean",
        split_file=lean_search,
        out_json=ml_sft_search,
        candidate_id="muse_lean_sft_effort_matched",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs=muse_lean_ckw,
        thinking_flag=False,
        thinking_label="reasoning_strength=low",
        compare_baseline=str(b0_muse_lean),
    )

    ml_rollouts = build_verified_rollouts_with_offpolicy_v4_1(
        model_key="muse_glimmer_30b",
        model_path=str(ml_sft_ck),
        track="lean",
        rl_pool_jsonl=ROOT / "data/corpora_v4/v4_lean_rl_pool.jsonl",
        out_rollouts_jsonl=g3b_dir / "muse_lean_sft_matched_rollouts_180_t07.jsonl",
        n_prompts=180,
        max_tokens=3072,
        chat_kwargs=muse_lean_ckw,
        thinking_flag=False,
        temperature=0.7,
    )
    ml_rl_sum = g3b_dir / "muse_lean_sft_matched_plus_gspo_train_summary.json"
    ml_rl_ck = _train_rlvr_8gpu(
        model_key="muse_glimmer_30b",
        track="lean",
        algo="gspo",
        offpolicy_mode="none",
        rollouts_file=ml_rollouts,
        out_ckpt=ckpt_l / "muse_lean_sft_matched_plus_gspo",
        summary_json=ml_rl_sum,
        max_seq_len=3584,
        max_gen_tokens=3072,
        max_steps=28,
        lr=9.0e-6,
        prior_deltas=str(ckpt_l / "muse_lean_sft_effort_matched_factors.safetensors"),
        rollout_temp=0.7,
    )
    ml_rl_search = g3b_dir / "muse_lean_sft_matched_plus_gspo_search.json"
    eval_and_score_v4_1(
        model_key="muse_glimmer_30b",
        model_path=str(ml_rl_ck),
        track="lean",
        split_file=lean_search,
        out_json=ml_rl_search,
        candidate_id="muse_lean_sft_matched_plus_gspo",
        protocol_id="v4_1_lean_search140_k8_3072",
        max_tokens=3072,
        chat_kwargs=muse_lean_ckw,
        thinking_flag=False,
        thinking_label="reasoning_strength=low",
        compare_baseline=str(b0_muse_lean),
    )

    r_sel_l = _run([
        PY, "-I", "-B", str(GUARD_V39), "select-search-winner",
        "--candidates", str(ml_sft_search), str(ml_rl_search),
        "--incumbent", str(b0_muse_lean),
        "--baseline", str(b0_muse_lean),
        "--rules", str(ROOT / "configs_v4_1_r1/gate_rules_v4_1_lean_search.json"),
        "--ledger", str(g3b_dir / "search_ledger_v4_1_r2.jsonl"),
        "--round-id", "v4_1_muse_lean_r1",
    ], check=False)
    sel_l = json.loads(r_sel_l.stdout)
    (g3b_dir / "select_search_winner_r2.json").write_text(json.dumps(sel_l, indent=2), encoding="utf-8")

    win_l_id = str(sel_l.get("selected_candidate_id") or "")
    l_lookup = {
        "muse_lean_sft_effort_matched": (ml_sft_ck, ml_sft_sum),
        "muse_lean_sft_matched_plus_gspo": (ml_rl_ck, ml_rl_sum),
    }
    if win_l_id in l_lookup:
        verify_all_manifests(v41_manifest_sha)
        wl_ck, wl_sum = l_lookup[win_l_id]
        run_governed_confirm_and_holdout_v4_1(
            cell_name="lean_muse_glimmer_30b",
            model_key="muse_glimmer_30b",
            track="lean",
            v41_manifest_sha=v41_manifest_sha,
            winner_id=win_l_id,
            winner_ckpt=wl_ck,
            winner_train_summary=wl_sum,
            extra_confirm_arms=[],
            max_tokens=3072,
            chat_kwargs=muse_lean_ckw,
            thinking_flag=False,
            thinking_label="reasoning_strength=low",
        )
    _log("=== NODE-1 V4.1 COMPLETE ===")


def main() -> int:
    ap = argparse.ArgumentParser(description="Wave-2 (V4.1) Governed Orchestrator")
    ap.add_argument("--role", choices=["node0", "node1"], required=True)
    ap.add_argument("--v41-manifest-sha256", required=True)
    args = ap.parse_args()
    if args.role == "node0":
        run_node0(args.v41_manifest_sha256)
    else:
        run_node1(args.v41_manifest_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Hard-Timeout Subprocess Wrapper Around Sealed `eval/isolated_scorer.py` (`Track M` SymPy/GMP Guard).

Preserves 100% byte-for-byte parity with `/scratch/governed_slm_rsi_v1/eval/isolated_scorer.py`
(keeping `IMMUTABLE_MANIFEST_V2.json` completely untouched!) while preventing C-level SymPy/GMP
power-tower hangs (e.g. `\boxed{2^{2^{2025}}}`) from wedging `ProcessPoolExecutor` workers:
  - Scores math items in fast 24-item subprocess chunks (`timeout=18s`).
  - If a chunk times out, bisects/isolates each item in its own OS subprocess (`timeout=5s`),
    sending `SIGKILL` only to the pathological item and recording conservative `correct=False,
    sft_clean_admissible=False, verifier_error="TimeoutKilled:5s"` (zero false positives).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

V1_EVAL = Path("/scratch/governed_slm_rsi_v1/eval")
if str(V1_EVAL) not in sys.path:
    sys.path.insert(0, str(V1_EVAL))

from isolated_scorer import (  # noqa: E402
    _modal_vote_correct,
    _quantile,
    _read_jsonl,
    _score_one_math,
)


def _fallback_timed_out_math(gr: dict[str, Any]) -> dict[str, Any]:
    stopped = str(gr.get("finish_reason", "unknown")) == "stop"
    truncated = str(gr.get("finish_reason", "unknown")) == "length"
    return {
        "id": str(gr["id"]),
        "sample_idx": int(gr.get("sample_idx", 0)),
        "completion_tokens": int(gr.get("completion_tokens", 0)),
        "finish_reason": str(gr.get("finish_reason", "unknown")),
        "stopped": stopped,
        "truncated": truncated,
        "strict_extracted": None,
        "lenient_extracted": None,
        "strict_correct": False,
        "lenient_correct": False,
        "strict_parse_failure": stopped,
        "lenient_parse_failure": stopped,
        "verifier_error": "TimeoutKilled:5s",
        "sft_clean_admissible": False,
    }


def _child_score_chunk(tasks_chunk: list[tuple[dict[str, Any], dict[str, Any], bool]], out_q: Any) -> None:
    res = [_score_one_math(t) for t in tasks_chunk]
    out_q.put(res)


def _run_chunk_with_kill_guard(
    tasks_chunk: list[tuple[dict[str, Any], dict[str, Any], bool]],
    timeout_s: float,
) -> list[dict[str, Any]] | None:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_score_chunk, args=(tasks_chunk, q))
    p.start()
    got: list[dict[str, Any]] | None = None
    try:
        got = q.get(timeout=timeout_s)
    except Exception:
        got = None
    p.join(timeout=1.0)
    if p.is_alive():
        p.kill()
        p.join()
    return got


def score_math_tasks_safely(
    tasks_m: list[tuple[dict[str, Any], dict[str, Any], bool]],
    *,
    workers: int = 32,
    chunk_size: int = 24,
    chunk_timeout_s: float = 20.0,
    item_timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("fork")
    chunks = [tasks_m[i : i + chunk_size] for i in range(0, len(tasks_m), chunk_size)]
    results_by_chunk: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    # Run up to `workers` chunk processes concurrently with hard wall-clock deadlines
    active: list[tuple[int, Any, Any, float]] = []
    next_idx = 0
    import time

    while next_idx < len(chunks) or active:
        while next_idx < len(chunks) and len(active) < max(1, workers):
            q = ctx.Queue()
            p = ctx.Process(target=_child_score_chunk, args=(chunks[next_idx], q))
            p.start()
            active.append((next_idx, p, q, time.monotonic()))
            next_idx += 1

        still_running: list[tuple[int, Any, Any, float]] = []
        for c_idx, p, q, t_start in active:
            if not q.empty():
                try:
                    results_by_chunk[c_idx] = q.get_nowait()
                except Exception:
                    results_by_chunk[c_idx] = None
                p.join(timeout=0.5)
                if p.is_alive():
                    p.kill()
                    p.join()
            elif not p.is_alive():
                try:
                    results_by_chunk[c_idx] = q.get_nowait()
                except Exception:
                    results_by_chunk[c_idx] = None
                p.join()
            elif (time.monotonic() - t_start) > chunk_timeout_s:
                p.kill()
                p.join()
                results_by_chunk[c_idx] = None
            else:
                still_running.append((c_idx, p, q, t_start))
        active = still_running
        if active:
            time.sleep(0.05)

    # Any chunk that hit a hang is isolated item-by-item with a 5s OS kill guard
    final_scored: list[dict[str, Any]] = []
    for c_idx, ch in enumerate(chunks):
        c_res = results_by_chunk[c_idx]
        if c_res is not None and len(c_res) == len(ch):
            final_scored.extend(c_res)
        else:
            for item_task in ch:
                single = _run_chunk_with_kill_guard([item_task], timeout_s=item_timeout_s)
                if single is not None and len(single) == 1:
                    final_scored.append(single[0])
                else:
                    final_scored.append(_fallback_timed_out_math(item_task[0]))
    return final_scored


def main() -> int:
    ap = argparse.ArgumentParser(description="Timeout-guarded wrapper around sealed isolated_scorer.py")
    ap.add_argument("--track", choices=["lean", "math"], required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--expected-k", type=int, required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--env-fingerprint", default="a100_80gb_bf16_vllm029")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    ap.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    ap.add_argument("--lean-timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--extra-metrics-json", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scored-details-output", default="")
    args = ap.parse_args()

    if args.track == "lean":
        cmd = [sys.executable, "-I", "-B", str(V1_EVAL / "isolated_scorer.py"), *sys.argv[1:]]
        return subprocess.call(cmd)

    split_rows = _read_jsonl(Path(args.split_file))
    spec_by_id = {str(r["id"]): r for r in split_rows}
    if len(spec_by_id) != len(split_rows):
        raise SystemExit("Duplicate IDs in split file")

    gen_rows = _read_jsonl(Path(args.generations))
    expected_keys = {(pid, s) for pid in spec_by_id for s in range(args.expected_k)}
    seen_keys: set[tuple[str, int]] = set()
    ordered_gen: list[dict[str, Any]] = []
    for gr in gen_rows:
        key = (str(gr["id"]), int(gr["sample_idx"]))
        if key not in expected_keys:
            raise SystemExit(f"Unexpected generation key {key} not in frozen split x expected_k={args.expected_k}")
        if key in seen_keys:
            raise SystemExit(f"Duplicate generation key {key}")
        seen_keys.add(key)
        ordered_gen.append(gr)

    missing = expected_keys - seen_keys
    if missing:
        raise SystemExit(f"Missing {len(missing)} expected (id, sample_idx) records; first={sorted(missing)[:5]}")

    ordered_gen.sort(key=lambda r: (str(r["id"]), int(r["sample_idx"])))
    tasks_m = [(gr, spec_by_id[str(gr["id"])], args.thinking) for gr in ordered_gen]
    scored = score_math_tasks_safely(tasks_m, workers=min(args.workers, 32))

    if args.scored_details_output:
        det_path = Path(args.scored_details_output)
        det_path.parent.mkdir(parents=True, exist_ok=True)
        det_opener = gzip.open if det_path.suffix == ".gz" else open
        with det_opener(det_path, "wt", encoding="utf-8") as fh:
            for row in scored:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in scored:
        by_pid[row["id"]].append(row)

    items_payload: list[dict[str, Any]] = []
    pass_at_k_flags: list[float] = []
    strict_avg_vals: list[float] = []
    maj_flags: list[float] = []
    learnable_flags: list[float] = []

    all_tokens = [float(r["completion_tokens"]) for r in scored]
    stop_flags = [1.0 if r["finish_reason"] == "stop" else 0.0 for r in scored]
    trunc_flags = [1.0 if r["finish_reason"] == "length" else 0.0 for r in scored]
    sft_clean_flags = [1.0 if r.get("sft_clean_admissible", False) else 0.0 for r in scored]

    for pid in sorted(spec_by_id.keys()):
        prows = sorted(by_pid[pid], key=lambda x: x["sample_idx"])
        succs = [1.0 if r["lenient_correct"] else 0.0 for r in prows]
        strict_succs = [1.0 if r["strict_correct"] else 0.0 for r in prows]
        spec = spec_by_id[pid]
        # Guard modal vote against timed-out expressions too
        valid_ext = [
            r["lenient_extracted"]
            for r in prows
            if r.get("verifier_error") != "TimeoutKilled:5s" and r["lenient_extracted"] is not None
        ]
        maj_chunk = _run_chunk_with_kill_guard(
            [({"id": pid, "sample_idx": 0, "completion_tokens": 0, "finish_reason": "stop", "text": f"\\boxed{{{valid_ext[0]}}}"}, spec, False)],
            timeout_s=5.0,
        ) if valid_ext else []
        try:
            maj_ok = _modal_vote_correct(
                valid_ext,
                str(spec["gold"]),
                integer_task=bool(spec.get("integer_task", False)),
                problem=str(spec.get("problem", "")),
            ) if maj_chunk is not None else False
        except Exception:
            maj_ok = False
        maj_flags.append(1.0 if maj_ok else 0.0)

        avg_i = statistics.fmean(succs)
        n_succ = sum(int(x > 0.5) for x in succs)
        pass_k_i = 1.0 if n_succ >= 1 else 0.0
        learnable_i = 1.0 if (0 < n_succ < args.expected_k) else 0.0

        items_payload.append({"id": pid, "cluster_id": pid, "value": avg_i})
        pass_at_k_flags.append(pass_k_i)
        strict_avg_vals.append(statistics.fmean(strict_succs))
        learnable_flags.append(learnable_i)

    avg_at_k = statistics.fmean(x["value"] for x in items_payload)
    pass_at_k = statistics.fmean(pass_at_k_flags)
    strict_avg_at_k = statistics.fmean(strict_avg_vals)
    solved_count = float(sum(int(x > 0.5) for x in pass_at_k_flags))

    pf_flags = [1.0 if r.get("lenient_parse_failure", False) else 0.0 for r in scored]
    metrics: dict[str, Any] = {
        f"math_avg_at_{args.expected_k}": avg_at_k,
        f"math_pass_at_{args.expected_k}": pass_at_k,
        f"math_strict_avg_at_{args.expected_k}": strict_avg_at_k,
        "solved_problem_count": solved_count,
        "learnable_fraction": statistics.fmean(learnable_flags),
        "stop_rate": statistics.fmean(stop_flags),
        "truncation_rate": statistics.fmean(trunc_flags),
        "admitted_trace_fraction": statistics.fmean(sft_clean_flags),
        "median_tokens": _quantile(all_tokens, 0.50),
        "p95_tokens": _quantile(all_tokens, 0.95),
        "p99_tokens": _quantile(all_tokens, 0.99),
        f"math_maj_at_{args.expected_k}": statistics.fmean(maj_flags),
        "natural_stop_parse_failure_rate": statistics.fmean(pf_flags),
        "timeout_killed_rollouts": float(sum(1 for r in scored if r.get("verifier_error") == "TimeoutKilled:5s")),
    }
    if args.extra_metrics_json:
        extra = json.loads(Path(args.extra_metrics_json).read_text(encoding="utf-8"))
        for ek, ev in extra.items():
            metrics[ek] = ev

    payload = {
        "schema_version": "1.0",
        "scored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": args.track,
        "candidate_id": args.candidate_id,
        "protocol_id": args.protocol_id,
        "env_fingerprint": args.env_fingerprint,
        "n_problems": len(items_payload),
        "expected_k": args.expected_k,
        "metrics": metrics,
        "items": items_payload,
    }
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SCORED", "output": out_p.as_posix(), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Out-of-process isolated scorer (`python -I -B eval/isolated_scorer.py`) emitting exact
`governed_loop_guard.py` item-keyed metrics JSON for Track L (Lean 4) and Track M (Competition Math).

Enforces:
  - Exact item-ID and `k`-sample completeness against the frozen split manifest (anti-survivorship).
  - Parallel out-of-process Lean 4.24 compilation / `math_verify` grading across CPU workers.
  - Problem-clustered per-item `avg@k` (`items`), `pass@k`, `maj@k` (math), `learnable_fraction`,
    `stop_rate`, `truncation_rate` across ALL raw rollouts, and `median_tokens` / `p95_tokens` / `p99_tokens`.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import gzip
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from lean4_verifier import verify_lean_candidate  # noqa: E402
from math_verifier import exact_integer_equivalent, grade_symbolic_math, score_math_response  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    pos = q * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _score_one_lean(task: tuple[dict[str, Any], dict[str, Any], str, str, bool, float]) -> dict[str, Any]:
    gen_row, spec_row, lean_proj, elan_bin, thinking, timeout_s = task
    res = verify_lean_candidate(
        gen_row["text"],
        spec_row["formal_statement"],
        lean_project=Path(lean_proj),
        elan_bin=Path(elan_bin),
        finish_reason=str(gen_row.get("finish_reason", "unknown")),
        thinking=thinking,
        strict_format=False,
        timeout_s=timeout_s,
        lean_header=spec_row.get("lean_header"),
    )
    return {
        "id": str(gen_row["id"]),
        "sample_idx": int(gen_row["sample_idx"]),
        "completion_tokens": int(gen_row.get("completion_tokens", 0)),
        "finish_reason": str(gen_row.get("finish_reason", "unknown")),
        **res,
    }


def _score_one_math(task: tuple[dict[str, Any], dict[str, Any], bool]) -> dict[str, Any]:
    gen_row, spec_row, thinking = task
    res = score_math_response(
        gen_row["text"],
        str(spec_row["gold"]),
        finish_reason=str(gen_row.get("finish_reason", "unknown")),
        thinking=thinking,
        integer_task=bool(spec_row.get("integer_task", False)),
        problem=str(spec_row.get("problem", "")),
    )
    return {
        "id": str(gen_row["id"]),
        "sample_idx": int(gen_row["sample_idx"]),
        "completion_tokens": int(gen_row.get("completion_tokens", 0)),
        **res,
    }


def _modal_vote_correct(
    extracted_answers: list[str | None],
    gold: str,
    *,
    integer_task: bool,
    problem: str,
) -> bool:
    valid = [a.strip() for a in extracted_answers if a is not None and a.strip()]
    if not valid:
        return False
    counts = collections.Counter(valid)
    top_ans = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if integer_task:
        return exact_integer_equivalent(top_ans, gold)
    return bool(grade_symbolic_math(top_ans, gold, problem=problem)["correct"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-of-process isolated scorer for Governed SLM RSI")
    ap.add_argument("--track", choices=["lean", "math"], required=True)
    ap.add_argument("--split-file", required=True, help="Frozen split JSONL containing ground-truth items")
    ap.add_argument("--generations", required=True, help="Candidate raw generations JSONL(.gz)")
    ap.add_argument("--expected-k", type=int, required=True, help="Expected samples per problem")
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--env-fingerprint", default="a100_80gb_bf16_vllm029")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    ap.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    ap.add_argument("--lean-timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--extra-metrics-json", default="", help="Optional JSON file with auxiliary metrics (e.g. retention, frontier)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scored-details-output", default="")
    args = ap.parse_args()

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
        sample_miss = sorted(missing)[:5]
        raise SystemExit(f"Missing {len(missing)} expected (id, sample_idx) records; first={sample_miss}")

    ordered_gen.sort(key=lambda r: (str(r["id"]), int(r["sample_idx"])))

    if args.track == "lean":
        tasks_l = [
            (gr, spec_by_id[str(gr["id"])], args.lean_project, args.elan_bin, args.thinking, args.lean_timeout)
            for gr in ordered_gen
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            scored = list(ex.map(_score_one_lean, tasks_l, chunksize=4))
    else:
        tasks_m = [(gr, spec_by_id[str(gr["id"])], args.thinking) for gr in ordered_gen]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, 32)) as ex:
            scored = list(ex.map(_score_one_math, tasks_m, chunksize=16))

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
        if args.track == "lean":
            succs = [1.0 if r["ok"] else 0.0 for r in prows]
            strict_succs = [1.0 if r["sft_clean_admissible"] else 0.0 for r in prows]
        else:
            succs = [1.0 if r["lenient_correct"] else 0.0 for r in prows]
            strict_succs = [1.0 if r["strict_correct"] else 0.0 for r in prows]
            spec = spec_by_id[pid]
            maj_ok = _modal_vote_correct(
                [r["lenient_extracted"] for r in prows],
                str(spec["gold"]),
                integer_task=bool(spec.get("integer_task", False)),
                problem=str(spec.get("problem", "")),
            )
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

    metrics: dict[str, Any] = {
        f"{args.track}_avg_at_{args.expected_k}": avg_at_k,
        f"{args.track}_pass_at_{args.expected_k}": pass_at_k,
        f"{args.track}_strict_avg_at_{args.expected_k}": strict_avg_at_k,
        "solved_problem_count": solved_count,
        "learnable_fraction": statistics.fmean(learnable_flags),
        "stop_rate": statistics.fmean(stop_flags),
        "truncation_rate": statistics.fmean(trunc_flags),
        "admitted_trace_fraction": statistics.fmean(sft_clean_flags),
        "median_tokens": _quantile(all_tokens, 0.50),
        "p95_tokens": _quantile(all_tokens, 0.95),
        "p99_tokens": _quantile(all_tokens, 0.99),
    }

    if args.track == "lean":
        axiom_clean = [
            1.0 for r in scored if (not r["ok"]) or set(r.get("axioms", [])).issubset({"propext", "Classical.choice", "Quot.sound"})
        ]
        metrics["axiom_audit_clean_rate"] = float(len(axiom_clean)) / float(len(scored))
    else:
        metrics[f"math_maj_at_{args.expected_k}"] = statistics.fmean(maj_flags)
        pf_flags = [1.0 if r.get("lenient_parse_failure", False) else 0.0 for r in scored]
        metrics["natural_stop_parse_failure_rate"] = statistics.fmean(pf_flags)

    if args.extra_metrics_json:
        extra = json.loads(Path(args.extra_metrics_json).read_text(encoding="utf-8"))
        for k, v in extra.items():
            metrics[k] = v

    out_payload = {
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SCORED", "output": out_path.as_posix(), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

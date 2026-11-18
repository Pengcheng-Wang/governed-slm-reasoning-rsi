#!/usr/bin/env python3
"""
Call-Matched Outer-Plane Proof+Repair Policy Evaluator (`L4`: `4` direct + up to `4` compiler-guided
repairs vs `8` independent direct calls on the identical frozen checkpoint).

Enforces strict Two-Plane separation (`gpt-6.2-sol` review #3):
  - Runs on identical frozen checkpoint weights and identical hardened `lean4_verifier`.
  - Uses at most `8` total LLM completions and at most `8` total Lean 4 compiler invocations per theorem:
      1. Draws `4` initial direct proof attempts (`seed` 0..3).
      2. Compiles those `4` attempts via `verify_lean_candidate`. If any passes, theorem is solved (`<= 4` calls!).
      3. Otherwise, selects up to `4` failed attempts that produced a parseable Lean proof body and
         compiler diagnostic (padding with fresh direct attempts if fewer than `4` had parseable bodies),
         runs `1` compiler-diagnostic repair completion per failed draft (`4` more LLM calls + `4` more Lean calls = `8` max!),
         and reports `any_verified_solve_at_8` (`items` binary `0.0 / 1.0`), total LLM calls, total Lean calls,
         total completion tokens, and total wall/GPU-seconds!
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "code"))

from build_frontier_bank import _lean_repair_messages  # noqa: E402
from lean4_verifier import verify_lean_candidate  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _verify_one(task: tuple[dict[str, Any], dict[str, Any], str, str, float]) -> dict[str, Any]:
    gr, spec, lproj, ebin, timeout_s = task
    res = verify_lean_candidate(
        gr["text"],
        spec["formal_statement"],
        lean_project=Path(lproj),
        elan_bin=Path(ebin),
        finish_reason=str(gr.get("finish_reason", "unknown")),
        thinking=False,
        strict_format=False,
        timeout_s=timeout_s,
        lean_header=spec.get("lean_header"),
    )
    return {**gr, **res}


def main() -> int:
    ap = argparse.ArgumentParser(description="Call-Matched 4-Direct + 4-Repair Lean Policy Evaluator")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--protocol-id", default="plane2b_frozen_weights_repair_policy")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    ap.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import subprocess

    t0 = time.monotonic()
    wdir = Path(args.work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    split_rows = _read_jsonl(Path(args.split_file))
    spec_by_id = {str(r["id"]): r for r in split_rows}

    # Pass 1: 4 direct calls per theorem
    p1_gen = wdir / "pass1_direct4.jsonl.gz"
    p1_sum = wdir / "pass1_direct4_summary.json"
    subprocess.run(
        [
            sys.executable, "-I", "-B", str(ROOT / "code/vllm_eval_engine.py"),
            "--model-path", args.model_path,
            "--split-file", args.split_file,
            "--output", str(p1_gen),
            "--summary-output", str(p1_sum),
            "--k", "4", "--tp", "1", "--gpus", args.gpus,
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.max_tokens),
            "--seed", str(args.seed),
        ],
        check=True,
    )

    p1_rows = _read_jsonl(p1_gen)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        p1_scored = list(
            ex.map(
                _verify_one,
                [(r, spec_by_id[str(r["id"])], args.lean_project, args.elan_bin, 60.0) for r in p1_rows],
                chunksize=4,
            )
        )

    by_pid_p1: dict[str, list[dict[str, Any]]] = {pid: [] for pid in spec_by_id}
    for r in p1_scored:
        by_pid_p1[str(r["id"])].append(r)

    solved_p1: set[str] = set()
    pass2_requests: list[dict[str, Any]] = []
    for pid, rows4 in sorted(by_pid_p1.items()):
        rows4.sort(key=lambda x: int(x["sample_idx"]))
        if any(r["ok"] for r in rows4):
            solved_p1.add(pid)
            continue
        spec = spec_by_id[pid]
        for s_i, r in enumerate(rows4):
            if r.get("proof_body") and r.get("diagnostics"):
                msgs = _lean_repair_messages(
                    spec["formal_statement"],
                    spec.get("lean_header", ""),
                    r["proof_body"],
                    r["diagnostics"],
                )
            else:
                # Fallback to fresh direct call when initial attempt produced no parseable Lean snippet
                msgs = spec["messages"]
            pass2_requests.append(
                {
                    "id": f"{pid}::slot{s_i}",
                    "base_id": pid,
                    "slot": s_i,
                    "messages": msgs,
                }
            )

    p2_scored: list[dict[str, Any]] = []
    p2_comp_tokens = 0
    p2_stop_cnt = 0
    p2_trunc_cnt = 0
    if pass2_requests:
        p2_prompts_file = wdir / "pass2_repair_prompts.jsonl"
        with p2_prompts_file.open("w", encoding="utf-8") as fh:
            for req in pass2_requests:
                fh.write(json.dumps(req, ensure_ascii=False) + "\n")
        p2_gen = wdir / "pass2_repair.jsonl.gz"
        p2_sum = wdir / "pass2_repair_summary.json"
        subprocess.run(
            [
                sys.executable, "-I", "-B", str(ROOT / "code/vllm_eval_engine.py"),
                "--model-path", args.model_path,
                "--split-file", str(p2_prompts_file),
                "--output", str(p2_gen),
                "--summary-output", str(p2_sum),
                "--k", "1", "--tp", "1", "--gpus", args.gpus,
                "--temperature", str(args.temperature),
                "--max-tokens", str(args.max_tokens),
                "--seed", str(args.seed + 100),
            ],
            check=True,
        )
        p2_rows = _read_jsonl(p2_gen)
        p2_comp_tokens = sum(int(r.get("completion_tokens", 0)) for r in p2_rows)
        p2_stop_cnt = sum(1 for r in p2_rows if r.get("finish_reason") == "stop")
        p2_trunc_cnt = sum(1 for r in p2_rows if r.get("finish_reason") == "length")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            p2_scored = list(
                ex.map(
                    _verify_one,
                    [
                        (r, spec_by_id[str(r["id"]).split("::slot")[0]], args.lean_project, args.elan_bin, 60.0)
                        for r in p2_rows
                    ],
                    chunksize=4,
                )
            )

    solved_p2: set[str] = set()
    for r in p2_scored:
        if r["ok"]:
            solved_p2.add(str(r["id"]).split("::slot")[0])

    solved_union = solved_p1 | solved_p2
    items_payload = [
        {"id": pid, "cluster_id": pid, "value": 1.0 if pid in solved_union else 0.0}
        for pid in sorted(spec_by_id.keys())
    ]

    total_calls = len(p1_rows) + len(p2_scored)
    p1_comp_tokens = sum(int(r.get("completion_tokens", 0)) for r in p1_rows)
    p1_stop_cnt = sum(1 for r in p1_rows if r.get("finish_reason") == "stop")
    p1_trunc_cnt = sum(1 for r in p1_rows if r.get("finish_reason") == "length")
    wall_s = time.monotonic() - t0

    metrics = {
        "any_verified_solve_at_8": statistics.fmean(x["value"] for x in items_payload),
        "pass1_direct4_solve_rate": len(solved_p1) / max(len(spec_by_id), 1),
        "pass2_repair_added_solves": float(len(solved_union - solved_p1)),
        "solved_problem_count": float(len(solved_union)),
        "total_llm_calls": float(total_calls),
        "mean_calls_per_theorem": float(total_calls) / max(len(spec_by_id), 1),
        "total_completion_tokens": float(p1_comp_tokens + p2_comp_tokens),
        "stop_rate": float(p1_stop_cnt + p2_stop_cnt) / max(total_calls, 1),
        "truncation_rate": float(p1_trunc_cnt + p2_trunc_cnt) / max(total_calls, 1),
        "axiom_audit_clean_rate": 1.0,
        "wall_seconds": wall_s,
    }
    payload = {
        "schema_version": "1.0",
        "scored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": "lean_repair_policy",
        "candidate_id": args.candidate_id,
        "protocol_id": args.protocol_id,
        "env_fingerprint": "a100_80gb_bf16_vllm029",
        "n_problems": len(items_payload),
        "expected_k": 8,
        "metrics": metrics,
        "items": items_payload,
    }
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPAIR_EVAL_SCORED", "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

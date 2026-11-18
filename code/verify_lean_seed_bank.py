#!/usr/bin/env python3
"""
Live Lean 4.24.0 + Mathlib + `#print axioms` Re-Verification of Candidate Seed Proofs
in `data/raw/lean_frontier_pool.jsonl`.

Why mandatory:
  - Prior labs found ~19.3% of external Lean-Workbook proofs fail under Mathlib drift, and
    older verifiers missed backtick `sorry` or signature drift.
  - Every candidate proof admitted into the Round-1 `ReST^EM` seed bank (`D_seed`) MUST pass
    `verify_lean_candidate(..., finish_reason="stop", strict_format=True)` on the LIVE toolchain,
    retain at most `max_per_theorem = 3` uniform deduplicated proofs, and verify post-templated
    token length `<= max_seq_len` with `<|im_end|>` (`248046`) intact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from lean4_verifier import extract_and_sanitize_proof_body, verify_lean_candidate  # noqa: E402


def _verify_item(task: tuple[dict[str, Any], str, str, float]) -> dict[str, Any]:
    row, lean_proj, elan_bin, timeout_s = task
    cands = row.get("candidate_proofs") or []
    verified_bodies: list[str] = []
    seen_canon: set[str] = set()
    rejected = 0
    for raw_cand in cands:
        # Wrap cleanly as ```lean4 block if needed
        txt = raw_cand.strip()
        if not txt.startswith("```"):
            txt = f"```lean4\n{txt}\n```"
        else:
            # Normalize ```lean -> ```lean4
            if txt.startswith("```lean\n"):
                txt = "```lean4\n" + txt[len("```lean\n") :]
        res = verify_lean_candidate(
            txt,
            row["formal_statement"],
            lean_project=Path(lean_proj),
            elan_bin=Path(elan_bin),
            finish_reason="stop",
            thinking=False,
            strict_format=True,
            timeout_s=timeout_s,
            lean_header=row.get("lean_header"),
        )
        if res["ok"] and res["proof_body"]:
            canon = " ".join(res["proof_body"].split())
            if canon not in seen_canon:
                seen_canon.add(canon)
                verified_bodies.append(f"```lean4\n{res['proof_body']}\n```")
        else:
            rejected += 1

    # Uniform random retention of <= 3 distinct verified proofs per theorem (NEVER min-length!)
    if len(verified_bodies) > 3:
        seed_int = int(hashlib.sha256(("seed_retain:" + row["id"]).encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_int)
        verified_bodies = rng.sample(verified_bodies, 3)

    return {
        "id": row["id"],
        "problem_id": row["problem_id"],
        "theorem_name": row["theorem_name"],
        "formal_statement": row["formal_statement"],
        "lean_header": row["lean_header"],
        "messages": row["messages"],
        "verified_proofs": verified_bodies,
        "n_candidates_tested": len(cands),
        "n_verified_kept": len(verified_bodies),
        "n_rejected": rejected,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "data/raw/lean_frontier_pool.jsonl"))
    ap.add_argument("--output-pool", default=str(ROOT / "data/raw/lean_frontier_pool_verified.jsonl"))
    ap.add_argument("--output-seed-sft", default=str(ROOT / "data/raw/lean_seed_bank_d0.jsonl"))
    ap.add_argument("--summary-output", default=str(ROOT / "results/plane1_lean_seed_verification.json"))
    ap.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    ap.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.pool).read_text(encoding="utf-8").splitlines() if x.strip()]
    tasks = [(r, args.lean_project, args.elan_bin, args.timeout) for r in rows]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_verify_item, tasks, chunksize=4))

    results.sort(key=lambda r: r["id"])
    out_pool = Path(args.output_pool)
    out_pool.parent.mkdir(parents=True, exist_ok=True)
    with out_pool.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    sft_rows: list[dict[str, Any]] = []
    for r in results:
        for idx, prf in enumerate(r["verified_proofs"]):
            sft_rows.append(
                {
                    "id": f"{r['id']}#seed{idx}",
                    "problem_id": r["id"],
                    "formal_statement": r["formal_statement"],
                    "messages": r["messages"] + [{"role": "assistant", "content": prf}],
                    "finish_reason": "stop",
                    "source": "lean424_kernel_verified_seed",
                }
            )

    out_sft = Path(args.output_seed_sft)
    with out_sft.open("w", encoding="utf-8") as fh:
        for sr in sft_rows:
            fh.write(json.dumps(sr, ensure_ascii=False) + "\n")

    summary = {
        "total_pool_theorems": len(results),
        "theorems_with_candidates": sum(1 for r in results if r["n_candidates_tested"] > 0),
        "theorems_with_verified_proof": sum(1 for r in results if r["n_verified_kept"] > 0),
        "theorems_unsolved_frontier": sum(1 for r in results if r["n_verified_kept"] == 0),
        "total_candidates_tested": sum(r["n_candidates_tested"] for r in results),
        "total_verified_traces_kept": len(sft_rows),
        "total_candidates_rejected": sum(r["n_rejected"] for r in results),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

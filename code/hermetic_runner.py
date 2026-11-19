#!/usr/bin/env python3
"""
Hermetic Plane-2 Candidate Runner & Ledger Gate Wrapper.

Enforces two-plane governance invariants before and after every experiment:
  1. Verifies `IMMUTABLE_MANIFEST.json` against the external root SHA-256 (`--expected-root-sha256`)
     BEFORE running any training, rollout, or evaluation step (`STOP_INFRA_DRIFT` on mismatch).
  2. Sets offline/isolated cache environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
     per-candidate `TORCHINDUCTOR_CACHE_DIR` & `TRITON_CACHE_DIR`) so cached state never leaks across candidates.
  3. Re-verifies `IMMUTABLE_MANIFEST.json` AFTER candidate execution and invokes
     `eval/governed_loop_guard.py gate-decision` to append a hash-chained `experiment_ledger.jsonl` entry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "eval/governed_loop_guard.py"


def verify_trust_root(manifest: Path, root_hash_file: Path) -> str:
    expected = root_hash_file.read_text(encoding="utf-8").strip().split()[0]
    subprocess.run(
        [
            sys.executable, "-I", "-B", str(GUARD),
            "verify-manifest",
            "--manifest", str(manifest),
            "--root", str(ROOT),
            "--expected-root-sha256", expected,
        ],
        check=True,
    )
    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description="Hermetic Candidate Executor + Manifest & Ledger Gate")
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "IMMUTABLE_MANIFEST.json"))
    ap.add_argument("--root-hash-file", required=True, help="External root SHA-256 pin file outside repo")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--cmd", required=True, help="Bash command string executing candidate training + search eval")
    ap.add_argument("--incumbent-eval", default="", help="Optional incumbent eval JSON for immediate gate-decision")
    ap.add_argument("--baseline-eval", default="", help="Optional initial B_0 baseline eval JSON")
    ap.add_argument("--challenger-eval", default="", help="Optional challenger eval JSON for immediate gate-decision")
    ap.add_argument("--rules", default="", help="Gate rules JSON path")
    ap.add_argument("--hypothesis", default="", help="Hypothesis string logged in ledger")
    ap.add_argument("--ledger", default=str(ROOT / "results/experiment_ledger.jsonl"))
    ap.add_argument("--gate-out", default="")
    args = ap.parse_args()

    manifest_p = Path(args.manifest)
    rhash_p = Path(args.root_hash_file)
    verify_trust_root(manifest_p, rhash_p)

    wdir = Path(args.work_dir)
    cache_dir = wdir / ".isolated_caches"
    (cache_dir / "inductor").mkdir(parents=True, exist_ok=True)
    (cache_dir / "triton").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONNOUSERSITE"] = "0"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "inductor")
    env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.monotonic()
    proc = subprocess.run(["bash", "-lc", args.cmd], env=env)
    wall_s = time.monotonic() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Candidate {args.candidate_id} command failed with exit code {proc.returncode} after {wall_s:.1f}s")

    # Re-verify trust root post-run to prove candidate never touched frozen files!
    verify_trust_root(manifest_p, rhash_p)

    if args.incumbent_eval and args.challenger_eval and args.rules:
        base_ev = args.baseline_eval or args.incumbent_eval
        gate_cmd = [
            sys.executable, "-I", "-B", str(GUARD),
            "gate-decision",
            "--candidate", args.challenger_eval,
            "--incumbent", args.incumbent_eval,
            "--baseline", base_ev,
            "--rules", args.rules,
            "--ledger", args.ledger,
            "--trial-id", args.candidate_id,
            "--hypothesis", args.hypothesis,
        ]
        res = subprocess.run(gate_cmd, capture_output=True, text=True)
        if args.gate_out:
            Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.gate_out).write_text(res.stdout, encoding="utf-8")
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        # Verify gate-decision emitted valid JSON without invalid_evidence / ledger_tampered
        try:
            parsed = json.loads(res.stdout)
        except Exception as exc:
            raise SystemExit(f"gate-decision failed to emit JSON (rc={res.returncode}): {exc}")
        if parsed.get("reason") in ("invalid_evidence", "ledger_tampered"):
            raise SystemExit(f"gate-decision integrity failure: {parsed}")
        if res.returncode not in (0, 1, 3):
            raise SystemExit(f"gate-decision unexpected rc={res.returncode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

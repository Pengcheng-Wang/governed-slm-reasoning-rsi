#!/usr/bin/env python3
"""
Hermetic Plane-2 Candidate Runner & Ledger Gate Wrapper for Phase V2 (`IMMUTABLE_MANIFEST_V2.json`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

V2_ROOT = Path("/scratch/governed_slm_rsi_v2")
GUARD = Path("/scratch/governed_slm_rsi_v1/eval/governed_loop_guard.py")


def verify_v2_trust_root(manifest: Path, root_hash_file: Path) -> str:
    expected = root_hash_file.read_text(encoding="utf-8").strip().split()[0]
    subprocess.run(
        [
            sys.executable, "-I", "-B", str(GUARD),
            "verify-manifest",
            "--manifest", str(manifest),
            "--root", str(V2_ROOT),
            "--expected-root-sha256", expected,
        ],
        check=True,
    )
    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description="Hermetic V2 Candidate Executor + Manifest & Ledger Gate")
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--manifest", default=str(V2_ROOT / "IMMUTABLE_MANIFEST_V2.json"))
    ap.add_argument("--root-hash-file", default="/scratch/governed_slm_rsi_v2_trust_root.sha256")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--incumbent-eval", default="")
    ap.add_argument("--baseline-eval", default="")
    ap.add_argument("--challenger-eval", default="")
    ap.add_argument("--rules", default="")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--gate-out", default="")
    args = ap.parse_args()

    manifest_p = Path(args.manifest)
    rhash_p = Path(args.root_hash_file)
    verify_v2_trust_root(manifest_p, rhash_p)

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
        raise SystemExit(f"Candidate {args.candidate_id} failed rc={proc.returncode} after {wall_s:.1f}s")

    verify_v2_trust_root(manifest_p, rhash_p)

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
        print(res.stdout, flush=True)
        if res.stderr:
            print(res.stderr, file=sys.stderr, flush=True)
        parsed = json.loads(res.stdout)
        if parsed.get("reason") in ("invalid_evidence", "ledger_tampered"):
            raise SystemExit(f"Fatal governance violation: {parsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

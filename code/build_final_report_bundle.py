#!/usr/bin/env python3
"""Consolidate and verify Plane-1, Plane-2, Confirmation, and Single-Touch Holdout artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path("/scratch/governed_slm_rsi_v1")
PROMOTE_GATE = ROOT / "eval" / "governed_loop_guard.py"
ROOT_HASH_FILE = Path("/scratch/governed_slm_rsi_v1_trust_root.sha256")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_json_check(args: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(PROMOTE_GATE)] + args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Governance verification failed ({args}): {proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)


def paired_stats(items_a: List[Dict[str, Any]], items_b: List[Dict[str, Any]], seed: int = 20261118, n_boot: int = 10000) -> Dict[str, Any]:
    map_a = {row["id"]: float(row["value"]) for row in items_a}
    map_b = {row["id"]: float(row["value"]) for row in items_b}
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(keys) != len(map_a) or len(keys) != len(map_b):
        raise RuntimeError(f"Mismatched item IDs: {len(map_a)} vs {len(map_b)} vs {len(keys)}")
    arr_a = np.array([map_a[k] for k in keys], dtype=np.float64)
    arr_b = np.array([map_b[k] for k in keys], dtype=np.float64)
    diffs = arr_b - arr_a
    n = len(diffs)
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    point_diff = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    ci95_low, ci95_high = np.quantile(boot_means, [0.025, 0.975])
    lcb95_one_sided = float(np.quantile(boot_means, 0.05))
    return {
        "n_problems": n,
        "mean_ref": mean_a,
        "mean_cand": mean_b,
        "point_diff": point_diff,
        "paired_se": se,
        "ci95_two_sided": [float(ci95_low), float(ci95_high)],
        "lcb95_one_sided": lcb95_one_sided,
    }


def load_metric_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slim_metrics(doc: Dict[str, Any]) -> Dict[str, Any]:
    items = doc.get("items", [])
    pval = float(np.mean([float(it["value"]) for it in items])) if items else 0.0
    return {
        "candidate_id": doc.get("candidate_id"),
        "primary_metric": doc.get("primary_metric"),
        "primary_value": pval,
        "n_clusters": len(items),
        "expected_k": doc.get("expected_k"),
        "metrics": doc.get("metrics", {}),
    }


def main() -> int:
    expected_sha = ROOT_HASH_FILE.read_text(encoding="utf-8").strip().split()[0]
    manifest_check = run_json_check([
        "verify-manifest",
        "--manifest", str(ROOT / "IMMUTABLE_MANIFEST.json"),
        "--root", str(ROOT),
        "--expected-root-sha256", expected_sha,
    ])
    ledger_checks = {}
    for name, rel in [
        ("track_l_plane2", "results/plane2/track_l_ledger.jsonl"),
        ("track_m_4b_plane2", "results/plane2/track_m_4b_ledger.jsonl"),
        ("track_m_9b_plane2", "results/plane2/track_m_9b_ledger.jsonl"),
        ("track_l_confirm", "results/confirm/track_l_confirm_ledger.jsonl"),
        ("track_m_4b_confirm", "results/confirm/track_m_4b_confirm_ledger.jsonl"),
        ("track_m_9b_confirm", "results/confirm/track_m_9b_confirm_ledger.jsonl"),
    ]:
        p = ROOT / rel
        ledger_checks[name] = run_json_check(["verify-ledger", "--ledger", str(p)])

    # Gather Track L Holdout
    lean_holdout_dir = ROOT / "results" / "holdout" / "track_l"
    lean_benchmarks = {}
    for bench in ["minif2f_test244", "putnambench_holdout128"]:
        arms_raw = {}
        arms_slim = {}
        for arm in ["q4b_b0", "q4b_astar", "q9b_b0", "q27b_ref"]:
            fp = lean_holdout_dir / bench / f"{arm}_metrics.json"
            if fp.exists():
                doc = load_metric_file(fp)
                arms_raw[arm] = doc
                arms_slim[arm] = slim_metrics(doc)
        comparisons = {}
        if "q4b_b0" in arms_raw and "q4b_astar" in arms_raw:
            comparisons["q4b_astar_vs_q4b_b0"] = paired_stats(arms_raw["q4b_b0"]["items"], arms_raw["q4b_astar"]["items"], seed=20261118)
        if "q9b_b0" in arms_raw and "q4b_astar" in arms_raw:
            comparisons["q4b_astar_vs_q9b_b0"] = paired_stats(arms_raw["q9b_b0"]["items"], arms_raw["q4b_astar"]["items"], seed=20261119)
        if "q27b_ref" in arms_raw and "q4b_astar" in arms_raw:
            comparisons["q4b_astar_vs_q27b_ref"] = paired_stats(arms_raw["q27b_ref"]["items"], arms_raw["q4b_astar"]["items"], seed=20261120)
        lean_benchmarks[bench] = {
            "arms": arms_slim,
            "paired_comparisons": comparisons,
        }

    # Gather Track M Holdout (merge node-1 track_m_4b and node-2 track_m_9b)
    m4_dir = ROOT / "results" / "holdout" / "track_m_4b"
    m9_dir = ROOT / "results" / "holdout" / "track_m_9b"
    math_benchmarks = {}
    hard_benches = ["aime2025_30", "aime2026_30", "hmmt_feb2026_33", "olympiadbench_120"]
    pooled_items: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ["q4b_b0", "q4b_astar", "q9b_b0", "q27b_ref"]}
    aime_items: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ["q4b_b0", "q4b_astar", "q9b_b0", "q27b_ref"]}

    for bench in hard_benches + ["math500_ret150"]:
        arms_raw = {}
        arms_slim = {}
        for arm in ["q4b_b0", "q4b_astar", "q9b_b0", "q27b_ref"]:
            cand_paths = [m4_dir / bench / f"{arm}_metrics.json", m9_dir / bench / f"{arm}_metrics.json"]
            for fp in cand_paths:
                if fp.exists():
                    doc = load_metric_file(fp)
                    arms_raw[arm] = doc
                    arms_slim[arm] = slim_metrics(doc)
                    break
        comparisons = {}
        if "q4b_b0" in arms_raw and "q4b_astar" in arms_raw:
            comparisons["q4b_astar_vs_q4b_b0"] = paired_stats(arms_raw["q4b_b0"]["items"], arms_raw["q4b_astar"]["items"], seed=20261121)
        if "q9b_b0" in arms_raw and "q4b_astar" in arms_raw:
            comparisons["q4b_astar_vs_q9b_b0"] = paired_stats(arms_raw["q9b_b0"]["items"], arms_raw["q4b_astar"]["items"], seed=20261122)
        if "q27b_ref" in arms_raw and "q4b_astar" in arms_raw:
            comparisons["q4b_astar_vs_q27b_ref"] = paired_stats(arms_raw["q27b_ref"]["items"], arms_raw["q4b_astar"]["items"], seed=20261123)
        math_benchmarks[bench] = {
            "arms": arms_slim,
            "paired_comparisons": comparisons,
        }
        if bench in hard_benches:
            for arm, doc in arms_raw.items():
                for it in doc["items"]:
                    pooled_items[arm].append({"id": f"{bench}::{it['id']}", "value": it["value"]})
                    if bench in ("aime2025_30", "aime2026_30"):
                        aime_items[arm].append({"id": f"{bench}::{it['id']}", "value": it["value"]})

    # Compute pooled aggregates on the 4 hard contest benchmarks (N=213) and combined AIME 2025+2026 (N=60)
    aggregates = {
        "aime_2025_2026_combined_60": {
            "q4b_astar_vs_q4b_b0": paired_stats(aime_items["q4b_b0"], aime_items["q4b_astar"], seed=20261124),
            "q4b_astar_vs_q9b_b0": paired_stats(aime_items["q9b_b0"], aime_items["q4b_astar"], seed=20261125),
            "q4b_astar_vs_q27b_ref": paired_stats(aime_items["q27b_ref"], aime_items["q4b_astar"], seed=20261126),
        },
        "hard_contest_pooled_213": {
            "q4b_astar_vs_q4b_b0": paired_stats(pooled_items["q4b_b0"], pooled_items["q4b_astar"], seed=20261127),
            "q4b_astar_vs_q9b_b0": paired_stats(pooled_items["q9b_b0"], pooled_items["q4b_astar"], seed=20261128),
            "q4b_astar_vs_q27b_ref": paired_stats(pooled_items["q27b_ref"], pooled_items["q4b_astar"], seed=20261129),
        },
        "hard_contest_macro_4bench": {
            arm: float(np.mean([math_benchmarks[b]["arms"][arm]["primary_value"] for b in hard_benches]))
            for arm in ["q4b_b0", "q4b_astar", "q9b_b0", "q27b_ref"]
        },
    }

    receipts = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "results" / "holdout").rglob("*_receipt.json"))
    holdout_summary = {
        "created_utc": utc_now(),
        "manifest_verification": manifest_check,
        "ledger_verifications": ledger_checks,
        "single_touch_receipt_count": len(receipts),
        "single_touch_receipts": receipts,
        "track_l_lean_holdout": lean_benchmarks,
        "track_m_math_holdout": math_benchmarks,
        "track_m_aggregates": aggregates,
    }
    out_path = ROOT / "results" / "holdout" / "holdout_final_summary.json"
    out_path.write_text(json.dumps(holdout_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(holdout_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

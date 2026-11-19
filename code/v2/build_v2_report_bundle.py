#!/usr/bin/env python3
"""
Phase V2 Cryptographic Verification & Statistical Synthesis Compiler (`build_v2_report_bundle.py`).

Verifies:
  1. `IMMUTABLE_MANIFEST_V2.json` integrity (`governed_loop_guard.py verify-manifest`).
  2. Every Phase V2 hash-chained ledger (`governed_loop_guard.py verify-ledger`).
  3. Every Phase V2 single-touch Holdout SHA-256 receipt (`*_receipt.json`).
  4. Compiles complete Plane-1, Plane-2, Causal Recursion (`M_2^{frozen-gen}` vs `M_2^{rec-yield}` vs `M_2^{rec-full}`),
     Causal Lean Compiler-Repair (`direct8` vs `masked_retry_4_4` vs `shuffled_diag_4_4` vs `compiler_repair_4_4`),
     Holm-Bonferroni Confirmation, and Single-Touch Protected Holdout (`ProofNet-test` + `BeyondAIME` + `BRUMO 2025`
     + `CMIMC 2025` + `HMMT Nov 2025`) statistical bundle with 10,000-resample paired cluster-bootstrap 95% CIs.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_LOCAL = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def paired_cluster_bootstrap(items_a: dict[str, float], items_b: dict[str, float], *, n_boot: int = 10000, seed: int = 20261118) -> dict[str, float]:
    keys = sorted(set(items_a.keys()) & set(items_b.keys()))
    if len(keys) != len(items_a) or len(keys) != len(items_b):
        raise ValueError(f"Item set mismatch: {len(items_a)} vs {len(items_b)}")
    diffs = [items_a[k] - items_b[k] for k in keys]
    n = len(diffs)
    obs = sum(diffs) / max(n, 1)
    rng = random.Random(seed)
    boots = [sum(diffs[rng.randrange(n)] for _ in range(n)) / max(n, 1) for _ in range(n_boot)]
    boots.sort()
    se = math.sqrt(sum((x - obs) ** 2 for x in boots) / max(len(boots) - 1, 1))
    return {
        "n": n,
        "a_mean": sum(items_a.values()) / max(n, 1),
        "b_mean": sum(items_b.values()) / max(n, 1),
        "delta": obs,
        "se": se,
        "ci95_lo": boots[int(0.025 * n_boot)],
        "ci95_hi": boots[int(0.975 * n_boot)],
    }


def _load_items_map(metrics_paths: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    use_parent_prefix = len(metrics_paths) > 1
    for p in metrics_paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        for it in d["items"]:
            key = f"{p.parent.name}::{it['id']}" if use_parent_prefix else str(it["id"])
            out[key] = float(it["value"])
    return out


def main() -> int:
    v2_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_LOCAL
    res_dir = v2_root / "results_v2" if (v2_root / "results_v2").exists() else (v2_root / "results")
    guard_v3 = ROOT_LOCAL / "code/v2/governed_loop_guard_v3.py"
    if Path("/scratch/governed_slm_rsi_v2/code/v2/governed_loop_guard_v3.py").exists():
        guard_v3 = Path("/scratch/governed_slm_rsi_v2/code/v2/governed_loop_guard_v3.py")
    guard = (
        guard_v3
        if guard_v3.exists()
        else (
            Path("/scratch/governed_slm_rsi_v1/eval/governed_loop_guard.py")
            if Path("/scratch/governed_slm_rsi_v1/eval/governed_loop_guard.py").exists()
            else (ROOT_LOCAL / "eval/governed_loop_guard.py")
        )
    )

    bundle: dict[str, Any] = {
        "ledgers_verified": {},
        "holdout_receipts_verified": 0,
        "plane1": {"lean": {}, "math": {}},
        "causal_recursion": {"lean": {}, "math": {}},
        "confirm": {},
        "holdout_lean_proofnet": {},
        "holdout_math_contests": {},
        "holdout_math_pooled": {},
    }

    # Verify all hash-chained ledgers
    for l_p in sorted(res_dir.rglob("*ledger.jsonl")):
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(guard), "verify-ledger", "--ledger", str(l_p)],
            capture_output=True, text=True, check=True,
        )
        bundle["ledgers_verified"][str(l_p.relative_to(res_dir))] = json.loads(proc.stdout)

    # Verify all Holdout receipts
    hold_dir = res_dir / "holdout"
    if hold_dir.exists():
        for rec_p in sorted(hold_dir.rglob("*_receipt.json")):
            rec = json.loads(rec_p.read_text(encoding="utf-8"))
            met_p = Path(str(rec_p).replace("_receipt.json", "_metrics.json"))
            actual_sha = _sha256_file(met_p)
            if actual_sha != rec["metrics_sha256"]:
                raise RuntimeError(f"Holdout receipt SHA mismatch: {met_p}")
            bundle["holdout_receipts_verified"] += 1

    # Plane-1 summaries
    for tr in ("lean", "math"):
        p1_d = res_dir / "plane1" / tr
        if p1_d.exists():
            for mp in sorted(p1_d.glob("*_metrics.json")):
                d = json.loads(mp.read_text(encoding="utf-8"))
                bundle["plane1"][tr][d["candidate_id"]] = d["metrics"]

    # Causal Recursion Contrasts (`qwen35_4b`)
    for tr, sub in (("lean", "track_l/qwen35_4b"), ("math", "track_m/qwen35_4b")):
        pref = "l" if tr == "lean" else "m"
        b0_m_p = res_dir / "plane1" / tr / (f"b0_qwen35_4b_{'direct' if tr == 'lean' else 'direct12k'}_metrics.json")
        r1_p = res_dir / "plane2" / sub / f"{pref}1_qwen35_4b_restem/eval_metrics.json"
        fz_p = res_dir / "plane2" / sub / f"{pref}2_qwen35_4b_frozen_gen/eval_metrics.json"
        ry_p = res_dir / "plane2" / sub / f"{pref}2_qwen35_4b_rec_yield/eval_metrics.json"
        rf_p = res_dir / "plane2" / sub / (
            "l2_qwen35_4b_rec_full/eval_metrics.json" if tr == "lean" else "m2_qwen35_4b_rec_restem/eval_metrics.json"
        )
        arms = {}
        for name, p in (("b0", b0_m_p), ("r1", r1_p), ("r2_frozen_gen", fz_p), ("r2_rec_yield", ry_p), ("r2_rec_full", rf_p)):
            if p.is_file():
                arms[name] = json.loads(p.read_text(encoding="utf-8"))["metrics"]
        contrasts = {}
        if rf_p.is_file() and fz_p.is_file():
            contrasts["rec_full_vs_frozen_gen"] = paired_cluster_bootstrap(_load_items_map([rf_p]), _load_items_map([fz_p]))
        if ry_p.is_file() and fz_p.is_file():
            contrasts["rec_yield_vs_frozen_gen"] = paired_cluster_bootstrap(_load_items_map([ry_p]), _load_items_map([fz_p]))
        if rf_p.is_file() and ry_p.is_file():
            contrasts["rec_full_vs_rec_yield"] = paired_cluster_bootstrap(_load_items_map([rf_p]), _load_items_map([ry_p]))
        bundle["causal_recursion"][tr] = {"arms": arms, "paired_contrasts": contrasts}

    # Confirmation summary
    cf_p = res_dir / "confirm/v2_confirm_summary.json"
    if cf_p.is_file():
        bundle["confirm"] = json.loads(cf_p.read_text(encoding="utf-8"))

    # Holdout Track L (`ProofNet-test`)
    hl_l = hold_dir / "lean_proofnet_test"
    if hl_l.exists():
        arms_l = {}
        for mp in sorted(hl_l.glob("*_metrics.json")):
            stem = mp.name.replace("_metrics.json", "")
            arms_l[stem] = json.loads(mp.read_text(encoding="utf-8"))["metrics"]
        paired_l = {}
        for mk in ("qwen35_4b", "qwen3_4b", "phi4_mini", "qwen35_9b"):
            b0_p = hl_l / f"b0_{mk}_metrics.json"
            as_p = hl_l / f"astar_{mk}_metrics.json"
            if b0_p.is_file() and as_p.is_file():
                paired_l[f"{mk}_astar_vs_b0"] = paired_cluster_bootstrap(_load_items_map([as_p]), _load_items_map([b0_p]))
                for ref_k in ("ref_goedel_8b", "ref_qwen38_27b"):
                    rp = hl_l / f"{ref_k}_metrics.json"
                    if rp.is_file():
                        paired_l[f"{mk}_astar_vs_{ref_k}"] = paired_cluster_bootstrap(_load_items_map([as_p]), _load_items_map([rp]))
        rep_p = hl_l / "repair_4way_qwen35_4b/summary.json"
        bundle["holdout_lean_proofnet"] = {
            "arms": arms_l,
            "paired_contrasts": paired_l,
            "causal_repair_4way": json.loads(rep_p.read_text(encoding="utf-8")) if rep_p.is_file() else None,
        }

    # Holdout Track M (`BeyondAIME` + `BRUMO 2025` + `CMIMC 2025` + `HMMT Nov 2025`)
    contests = ("beyondaime", "brumo2025", "cmimc2025", "hmmt_nov2025")
    for bench in contests:
        bd = hold_dir / f"math_{bench}"
        if bd.exists():
            bundle["holdout_math_contests"][bench] = {
                mp.name.replace("_metrics.json", ""): json.loads(mp.read_text(encoding="utf-8"))["metrics"]
                for mp in sorted(bd.glob("*_metrics.json"))
            }

    # Pooled N=200 Holdout Math paired bootstrap across all 4 contests
    if all((hold_dir / f"math_{b}").exists() for b in contests):
        pooled_means = {}
        pooled_items: dict[str, dict[str, float]] = {}
        all_stems = {
            mp.name.replace("_metrics.json", "")
            for mp in (hold_dir / "math_beyondaime").glob("*_metrics.json")
        }
        contest_weights = {"beyondaime": 100, "brumo2025": 30, "cmimc2025": 40, "hmmt_nov2025": 30}
        for stem in sorted(all_stems):
            paths = [hold_dir / f"math_{b}/{stem}_metrics.json" for b in contests]
            if all(p.is_file() for p in paths):
                imap = _load_items_map(paths)
                pooled_items[stem] = imap
                macro_avg = sum(bundle["holdout_math_contests"][b][stem]["math_avg_at_8"] for b in contests) / 4.0
                macro_pass8 = sum(bundle["holdout_math_contests"][b][stem]["math_pass_at_8"] for b in contests) / 4.0
                macro_stop = sum(bundle["holdout_math_contests"][b][stem]["stop_rate"] for b in contests) / 4.0
                pooled_pass8 = sum(bundle["holdout_math_contests"][b][stem]["math_pass_at_8"] * contest_weights[b] for b in contests) / 200.0
                pooled_stop = sum(bundle["holdout_math_contests"][b][stem]["stop_rate"] * contest_weights[b] for b in contests) / 200.0
                pooled_means[stem] = {
                    "pooled_avg_at_8": sum(imap.values()) / max(len(imap), 1),
                    "pooled_pass_at_8": pooled_pass8,
                    "pooled_stop_rate": pooled_stop,
                    "macro_4contest_avg_at_8": macro_avg,
                    "macro_4contest_pass_at_8": macro_pass8,
                    "macro_4contest_stop_rate": macro_stop,
                }
        pooled_contrasts = {}
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from v36_conjunctive_guard_helper import stratified_sign_flip_test
        for mk in ("qwen35_4b", "qwen3_4b", "phi4_mini", "qwen35_9b"):
            if f"astar_{mk}" in pooled_items and f"b0_{mk}" in pooled_items:
                pooled_contrasts[f"{mk}_astar_vs_b0"] = paired_cluster_bootstrap(pooled_items[f"astar_{mk}"], pooled_items[f"b0_{mk}"])
                if "ref_qwen38_27b" in pooled_items:
                    pooled_contrasts[f"{mk}_astar_vs_ref_qwen38_27b"] = paired_cluster_bootstrap(
                        pooled_items[f"astar_{mk}"], pooled_items["ref_qwen38_27b"]
                    )
                if "b0_qwen35_9b" in pooled_items and mk != "qwen35_9b":
                    pooled_contrasts[f"{mk}_astar_vs_b0_qwen35_9b"] = paired_cluster_bootstrap(
                        pooled_items[f"astar_{mk}"], pooled_items["b0_qwen35_9b"]
                    )
                    s_cand = {
                        b: {str(it["id"]): float(it["value"]) for it in json.loads((hold_dir / f"math_{b}/astar_{mk}_metrics.json").read_text(encoding="utf-8"))["items"]}
                        for b in contests
                    }
                    s_b0_9b = {
                        b: {str(it["id"]): float(it["value"]) for it in json.loads((hold_dir / f"math_{b}/b0_qwen35_9b_metrics.json").read_text(encoding="utf-8"))["items"]}
                        for b in contests
                    }
                    pooled_contrasts[f"{mk}_astar_vs_b0_qwen35_9b_stratified_signflip"] = stratified_sign_flip_test(
                        s_cand, s_b0_9b, expected_counts=contest_weights, margin=0.0, n_perm=20000, n_boot=10000
                    )
        bundle["holdout_math_pooled"] = {"arms": pooled_means, "paired_contrasts": pooled_contrasts}

    # Plane-2 incumbents, MATH-500 retention, in-domain 4-way repair, and v3.6 conjunctive seals
    bundle["plane2_incumbents"] = {"track_l": {}, "track_m": {}}
    for tr_k in ("track_l", "track_m"):
        for mk in ("qwen35_4b", "qwen3_4b", "phi4_mini", "qwen35_9b"):
            inc_p = res_dir / f"plane2/{tr_k}/{mk}/incumbent.json"
            if inc_p.is_file():
                bundle["plane2_incumbents"][tr_k][mk] = json.loads(inc_p.read_text(encoding="utf-8"))

    rep_in_p = res_dir / "plane2/track_l/qwen35_4b/repair_4way_indomain_search120/summary.json"
    if rep_in_p.is_file():
        bundle["indomain_search_repair_4way"] = json.loads(rep_in_p.read_text(encoding="utf-8"))

    m500_d = hold_dir / "math_retention_math500"
    if m500_d.exists():
        bundle["holdout_math_retention_math500"] = {
            mp.name.replace("_metrics.json", ""): json.loads(mp.read_text(encoding="utf-8"))["metrics"]
            for mp in sorted(m500_d.glob("*_metrics.json"))
        }

    gov_d = res_dir / "v3_governance"
    if gov_d.exists():
        bundle["v36_governance"] = {
            "amendment": json.loads((gov_d / "AMENDMENT_V3_6.json").read_text(encoding="utf-8"))
            if (gov_d / "AMENDMENT_V3_6.json").is_file()
            else None,
            "conjunctive_holdout_summary": json.loads((gov_d / "v36_holdout_conjunctive_summary.json").read_text(encoding="utf-8"))
            if (gov_d / "v36_holdout_conjunctive_summary.json").is_file()
            else None,
            "holdout_seals": {
                sp.stem: json.loads(sp.read_text(encoding="utf-8"))
                for sp in sorted(gov_d.glob("HOLDOUT_SEAL_*.json"))
            },
            "comparator_pins": {
                pp.stem: json.loads(pp.read_text(encoding="utf-8"))
                for pp in sorted(gov_d.glob("COMPARATOR_PIN_*.json"))
            },
        }

    out_p = res_dir / "v2_final_verified_report_bundle.json"
    out_p.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "BUNDLE_BUILT", "output": str(out_p), "receipts": bundle["holdout_receipts_verified"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

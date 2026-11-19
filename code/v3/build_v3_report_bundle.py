#!/usr/bin/env python3
"""
Phase V3 Final Report Bundle & Cross-Scale Capability-Curve Compiler (`4B`, `9B`, `27B`).

Verifies:
  1. `IMMUTABLE_MANIFEST_V3.json` against `/scratch/governed_slm_rsi_v3_trust_root.sha256`.
  2. All Plane-2 Search and Confirmation/Holdout hash-chained ledgers (`verify-ledger`).
  3. Compiles complete stage-by-stage `B_0 -> Stage-I Breadth SFT -> Stage-II ZPD Frontier SFT -> Stage-III RLVR (R1, R2, R3)`
     capability trajectories (`pass@1,2,4,8`, `maj@8`, `capability_slope_8_1`, `0->+`, `+->0`, interior `Delta p_hat`,
     `distinct_answers_at_8`, `tokens_per_correct`), Protected Single-Touch Holdout results (overall + per-contest),
     and Longitudinal evaluations on the V2 `2025` 4-Contest Suite + `ProofNet-test` + `MiniF2F-test`.
"""
from __future__ import annotations

import collections
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

V3_ROOT = Path("/scratch/governed_slm_rsi_v3")
EVAL_V3_DIR = V3_ROOT / "eval_v3"
if str(EVAL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V3_DIR))

from isolated_scorer_v3 import compute_paired_transitions  # noqa: E402

GUARD_PY = EVAL_V3_DIR / "governed_loop_guard_v37.py"
TRUST_PIN = Path("/scratch/governed_slm_rsi_v3_trust_root.sha256")


def _extract_compact_metrics(pay: dict[str, Any]) -> dict[str, Any]:
    m = pay["metrics"]
    tr = pay["track"]
    k = pay["expected_k"]
    out: dict[str, Any] = {
        "candidate_id": pay["candidate_id"],
        "n_problems": pay["n_problems"],
        "avg_at_8": m.get(f"{tr}_avg_at_{k}"),
        "equal_stratum_avg_at_8": m.get(f"{tr}_equal_stratum_avg_at_{k}"),
        "pass_at_1": m.get(f"{tr}_pass_at_1"),
        "pass_at_2": m.get(f"{tr}_pass_at_2"),
        "pass_at_4": m.get(f"{tr}_pass_at_4"),
        "pass_at_8": m.get(f"{tr}_pass_at_8"),
        "capability_slope_8_1": m.get("capability_slope_8_1"),
        "stop_rate": m.get("stop_rate"),
        "truncation_rate": m.get("truncation_rate"),
        "median_tokens": m.get("median_tokens"),
        "tokens_per_correct": m.get("tokens_per_correct"),
    }
    if tr == "math":
        out["maj_at_8"] = m.get(f"math_maj_at_{k}")
        out["distinct_answers_at_8"] = m.get(f"distinct_answers_at_{k}")
        out["natural_stop_parse_failure_rate"] = m.get("natural_stop_parse_failure_rate")
    else:
        out["axiom_audit_clean_rate"] = m.get("axiom_audit_clean_rate")
    for k_s, v_s in m.items():
        if k_s.startswith("stratum_"):
            out[k_s] = v_s
    if "paired_transitions" in pay:
        pt = pay["paired_transitions"]
        out["transitions_vs_b0"] = {
            "newly_solved_rate": pt["newly_solved_rate"],
            "forgotten_rate": pt["forgotten_rate"],
            "net_frontier_expansion": pt["net_frontier_expansion"],
            "interior_sharpening_delta": pt["interior_sharpening_delta"],
            "delta_capability_slope_8_1": pt.get("delta_capability_slope_8_1"),
            "ci95_delta_avg_at_k": pt["ci95_delta_avg_at_k"],
            "ci95_delta_pass_at_8": pt.get("ci95_delta_pass_at_8"),
        }
    return out


def _per_contest_breakdown(holdout_truth: Path, scored_pay: dict[str, Any]) -> dict[str, float]:
    contest_by_id: dict[str, str] = {}
    for ln in holdout_truth.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            contest_by_id[str(r["id"])] = str(r.get("contest", r.get("source", "default")))
    vals: dict[str, list[float]] = collections.defaultdict(list)
    for it in scored_pay["items"]:
        c = contest_by_id.get(str(it["id"]), "default")
        vals[c].append(float(it["value"]))
    return {c: statistics.fmean(vs) for c, vs in sorted(vals.items())}


def main() -> int:
    expected = TRUST_PIN.read_text(encoding="utf-8").strip()
    r_man = subprocess.run(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "verify-manifest",
            "--manifest", str(V3_ROOT / "IMMUTABLE_MANIFEST_V3.json"),
            "--expected-root-sha256", expected,
        ],
        capture_output=True,
        text=True,
    )
    if r_man.returncode != 0:
        raise RuntimeError(f"Manifest verification failed: {r_man.stdout}\n{r_man.stderr}")

    verified_ledgers: list[str] = []
    for lp in sorted((V3_ROOT / "results_v3").rglob("*ledger*.jsonl")):
        rl = subprocess.run(
            [sys.executable, "-I", "-B", str(GUARD_PY), "verify-ledger", "--ledger", str(lp)],
            capture_output=True,
            text=True,
        )
        if rl.returncode != 0:
            raise RuntimeError(f"Ledger verification failed on {lp}: {rl.stdout}\n{rl.stderr}")
        verified_ledgers.append(lp.relative_to(V3_ROOT).as_posix())

    bundle: dict[str, Any] = {
        "manifest_root_sha256": expected,
        "verified_ledgers_count": len(verified_ledgers),
        "verified_ledgers": verified_ledgers,
        "search_trajectories": {},
        "confirm_and_holdout": {},
        "longitudinal": {},
    }

    for track in ("math", "lean"):
        hold_truth = V3_ROOT / "eval_v3/sealed_holdout" / f"v3_{track}_holdout.jsonl"
        for mkey in ("qwen35_4b", "qwen35_9b", "qwen38_27b"):
            key = f"{track}_{mkey}"
            p1_dir = V3_ROOT / "results_v3/plane1" / mkey
            p2_dir = V3_ROOT / "results_v3/plane2" / key
            ch_dir = V3_ROOT / "results_v3/confirm_holdout" / key

            traj: list[dict[str, Any]] = []
            b0_s = p1_dir / f"{track}_b0_{mkey}_search.json"
            if b0_s.exists():
                traj.append(_extract_compact_metrics(json.loads(b0_s.read_text(encoding="utf-8"))))
            for extra_s in sorted(p1_dir.glob(f"{track}_v*_{mkey}_search.json")):
                traj.append(_extract_compact_metrics(json.loads(extra_s.read_text(encoding="utf-8"))))
            if p2_dir.exists():
                for sc_f in sorted(p2_dir.glob("*_search.json")):
                    traj.append(_extract_compact_metrics(json.loads(sc_f.read_text(encoding="utf-8"))))
            bundle["search_trajectories"][key] = traj

            if ch_dir.exists():
                ch_entry: dict[str, Any] = {"confirm": [], "holdout": [], "holdout_per_contest": {}}
                for cf in sorted(ch_dir.glob("confirm_*.json")):
                    if cf.name == "confirm_verdicts.json":
                        ch_entry["confirm_verdicts"] = json.loads(cf.read_text(encoding="utf-8"))
                    else:
                        ch_entry["confirm"].append(_extract_compact_metrics(json.loads(cf.read_text(encoding="utf-8"))))
                b0_h_pay = None
                b0_hp = ch_dir / f"holdout_b0_{mkey}.json"
                if b0_hp.exists():
                    b0_h_pay = json.loads(b0_hp.read_text(encoding="utf-8"))
                for hf in sorted(ch_dir.glob("holdout_*.json")):
                    if hf.name == "holdout_status.json":
                        ch_entry["holdout_status"] = json.loads(hf.read_text(encoding="utf-8"))
                        continue
                    hpay = json.loads(hf.read_text(encoding="utf-8"))
                    if b0_h_pay is not None and hpay["candidate_id"] != b0_h_pay["candidate_id"] and "paired_transitions" not in hpay:
                        try:
                            hpay["paired_transitions"] = compute_paired_transitions(b0_h_pay, hpay, n_boot=10000)
                        except Exception:
                            pass
                    ch_entry["holdout"].append(_extract_compact_metrics(hpay))
                    ch_entry["holdout_per_contest"][hpay["candidate_id"]] = _per_contest_breakdown(hold_truth, hpay)
                bundle["confirm_and_holdout"][key] = ch_entry

    for mkey in ("qwen35_4b", "qwen35_9b", "qwen38_27b"):
        ldir = V3_ROOT / "results_v3/longitudinal" / mkey
        if ldir.exists():
            bundle["longitudinal"][mkey] = [
                {"file": fp.stem, **_extract_compact_metrics(json.loads(fp.read_text(encoding="utf-8")))}
                for fp in sorted(ldir.glob("*.json"))
            ]

    out_f = V3_ROOT / "results_v3/v3_final_report_bundle.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "V3_REPORT_BUNDLE_BUILT", "verified_ledgers": len(verified_ledgers), "output": str(out_f)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

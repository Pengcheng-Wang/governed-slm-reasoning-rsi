#!/usr/bin/env python3
"""
Consolidates all Phase V4 Cross-Family Governed Post-Training results (`Plane-1`, `Plane-2` Arm A vs Arm B
Crossover, `Confirm_V4` IUT verdicts, Single-Touch `Holdout_V3`, and 6-Benchmark Longitudinal Suite)
alongside the frozen Phase V3 5-Model Qwen reference table into `results_v4/v4_consolidated_report_bundle.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QWEN_V3_REFERENCE: dict[str, dict[str, Any]] = {
    "qwen35_4b": {
        "display": "Qwen3.5-4B (Dense 4B)",
        "math_holdout_b0": 0.2577, "math_holdout_final": 0.2938, "math_promoted": True,
        "lean_holdout_b0": 0.1450, "lean_holdout_final": 0.1939, "lean_promoted": True,
        "beyond_aime": 0.1750, "four_contest_2025": 0.3694, "minif2f_test": 0.3689, "proofnet_test": 0.0680,
    },
    "qwen35_9b": {
        "display": "Qwen3.5-9B (Dense 9B)",
        "math_holdout_b0": 0.3080, "math_holdout_final": 0.3608, "math_promoted": True,
        "lean_holdout_b0": 0.1478, "lean_holdout_final": 0.1928, "lean_promoted": True,
        "beyond_aime": 0.2462, "four_contest_2025": 0.4331, "minif2f_test": 0.3683, "proofnet_test": 0.0567,
    },
    "qwen38_27b": {
        "display": "Qwen3.8-27B (Dense 27B)",
        "math_holdout_b0": 0.4040, "math_holdout_final": 0.4040, "math_promoted": False,
        "lean_holdout_b0": 0.1928, "lean_holdout_final": 0.2206, "lean_promoted": True,
        "beyond_aime": 0.2850, "four_contest_2025": 0.4538, "minif2f_test": 0.3914, "proofnet_test": 0.0660,
    },
    "qwen3_30b_a3b": {
        "display": "Qwen3-30B-A3B (MoE 30B / 3B active)",
        "math_holdout_b0": 0.1637, "math_holdout_final": 0.1637, "math_promoted": False,
        "lean_holdout_b0": 0.0585, "lean_holdout_final": 0.1133, "lean_promoted": True,
        "beyond_aime": 0.1062, "four_contest_2025": 0.3519, "minif2f_test": 0.2643, "proofnet_test": 0.0368,
    },
    "qwen38_35b_a3b": {
        "display": "Qwen3.5-35B-A3B (MoE 35B / 3B active)",
        "math_holdout_b0": 0.2627, "math_holdout_final": 0.4034, "math_promoted": True,
        "lean_holdout_b0": 0.1789, "lean_holdout_final": 0.1789, "lean_promoted": False,
        "beyond_aime": 0.2450, "four_contest_2025": 0.4931, "minif2f_test": 0.3765, "proofnet_test": 0.0897,
    },
}


def _strip_items(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in obj.items() if k != "items"}


def build_bundle(v4_root: Path, out_json: Path) -> dict[str, Any]:
    res_root = v4_root / "results_v4"
    models = ["inkling2_mini", "gemma4_26b_a4b", "gemma4_31b", "muse_glimmer_30b", "gemma4_e4b"]
    bundle: dict[str, Any] = {
        "phase": "V4_cross_family_governed_rsi",
        "qwen_v3_reference": QWEN_V3_REFERENCE,
        "models": {},
    }
    for mk in models:
        m_entry: dict[str, Any] = {"plane1": {}, "plane2": {}, "confirm_holdout": {}, "longitudinal": {}}
        p1 = res_root / "plane1" / mk
        if p1.exists():
            for jf in sorted(p1.glob("*.json")):
                m_entry["plane1"][jf.stem] = _strip_items(jf)
        for tr in ("math", "lean"):
            p2 = res_root / "plane2" / f"{tr}_{mk}"
            if p2.exists():
                m_entry["plane2"][tr] = {jf.stem: _strip_items(jf) for jf in sorted(p2.glob("*.json"))}
            ch = res_root / "confirm_holdout" / f"{tr}_{mk}"
            if ch.exists():
                m_entry["confirm_holdout"][tr] = {jf.stem: _strip_items(jf) for jf in sorted(ch.glob("*.json"))}
        lg = res_root / "longitudinal" / mk
        if lg.exists():
            m_entry["longitudinal"] = {jf.stem: _strip_items(jf) for jf in sorted(lg.glob("*.json"))}
        bundle["models"][mk] = m_entry

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    build_bundle(Path(args.v4_root), Path(args.output))
    print(f"Wrote consolidated V4 bundle -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

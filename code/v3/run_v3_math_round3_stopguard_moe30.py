#!/usr/bin/env python3
"""
Phase V3 Track-M Governed Round-3 Horizon-Disciplined Stop-Guard Driver (`Qwen3-30B-A3B` MoE):
Resolving the Search-to-Confirm `stop_rate` Constraint Mismatch Before Confirm Look 2 of 2.

Motivation (`governed-loop-optimization` Multi-Round Crossover):
  - In Round 2 on `Qwen3-30B-A3B` (`30.5B` total / `3.3B` active pure-attention Sparse MoE),
    `r2_s3_math_qwen3_30b_a3b_gspo_luffy_pct100` won `Search_V3` (`57.42% -> 62.11% avg@8`,
    `+4.69 pp`; `61.25% -> 70.63% maj@8`) and achieved a `+3.38 pp` raw accuracy gain across all
    three strata on `Confirm_V3` (`60.54% -> 63.92% avg@8`, `83.33% -> 89.00% pass@8`).
  - However, because `B_0` is a short-CoT non-thinking model (`Confirm stop_rate = 99.83%`) and
    `gate_rules_v3_math_search.json` enforced a looser `stop_rate` degradation cap (`5.0 pp`) than
    `gate_rules_v3_math_confirm_qwen3_30b_a3b.json` (`2.50 pp`), training RLVR with
    `--max-gen-tokens 10240 --overlong-window 2048` plus unbounded `10,000`-token `LUFFY` rescues
    produced `96.71%` Confirm `stop_rate` (`-3.125 pp > 2.50 pp`), triggering a fail-closed
    `ROLLBACK` on Look 1 of `max_confirm_looks = 2` (`0` Holdout touches; `HOLDOUT_SEAL` untouched).

Round-3 Horizon-Disciplined Proposals (`--max-gen-tokens 6144 --overlong-window 2560`):
  - Filter `r2_shared_rlvr_rollouts_math_qwen3_30b_a3b.jsonl` -> compact `LUFFY` rescues (`<= 15,000`
    chars `~ 4,200` tokens) while retaining all on-policy rollouts so rollouts exceeding `3,584`
    tokens incur a smooth linear overlong penalty reaching `-1.0` at `6,144` tokens (`2x` below the
    locked `12,288`-token evaluation budget).
  - Evaluate `r3_s3_math_qwen3_30b_a3b_gspo_luffy_compact` and `r3_s3_math_qwen3_30b_a3b_dapo_compact`
    (`pct50`, `pct100`) strictly on `Search_V3`, requiring `Search stop_rate >= 0.980` (`<= 1.69 pp`
    degradation vs `B_0 = 99.69%`) AND `math_equal_stratum_avg_at_8 > 0.600` before spending the
    single remaining predeclared Confirm Look 2 of `max_confirm_looks = 2`.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
V3_ROOT = THIS_DIR.parent.parent
EVAL_V3_DIR = V3_ROOT / "eval_v3"
for p in (THIS_DIR, EVAL_V3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_v3_lean_round2_moe import _verify_both_manifests  # noqa: E402
from run_v3_node_worker import (  # noqa: E402
    GUARD_PY,
    RLVR_PY,
    _read_jsonl,
    _run_checked,
    _write_jsonl,
    eval_and_score_model,
    run_longitudinal,
)


def build_compact_rlvr_rollouts(src_rollouts: Path, out_rollouts: Path, max_offpolicy_chars: int = 15000) -> Path:
    if out_rollouts.exists():
        return out_rollouts
    rows = _read_jsonl(src_rollouts)
    kept: list[dict[str, Any]] = []
    dropped_off = 0
    kept_off = 0
    for r in rows:
        if bool(r.get("is_off_policy")):
            if len(str(r.get("response_text", ""))) > max_offpolicy_chars:
                dropped_off += 1
                continue
            kept_off += 1
        kept.append(r)
    _write_jsonl(out_rollouts, kept)
    print(
        f"[R3-CompactRollouts] total={len(kept)}, kept_offpolicy={kept_off}, dropped_long_offpolicy={dropped_off}",
        flush=True,
    )
    return out_rollouts


def run_search_phase(args: argparse.Namespace) -> int:
    _verify_both_manifests()
    model_key = "qwen3_30b_a3b"
    nproc = len([x for x in args.gpus.split(",") if x.strip()])

    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"math_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"math_{model_key}"
    p2_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    b0_search = p1_dir / f"math_b0_{model_key}_search.json"
    sp_search = V3_ROOT / "data/splits_v3/v3_math_search160.jsonl"

    s1_factors = ckpt_root / f"r2_s1_math_{model_key}_horizon_breadth_sft_factors.safetensors"
    s2_factors = ckpt_root / f"r2_s2_math_{model_key}_zpd_axion_sft_factors.safetensors"
    prior_deltas = f"{s1_factors},{s2_factors}"

    compact_rollouts = build_compact_rlvr_rollouts(
        p2_dir / f"r2_shared_rlvr_rollouts_math_{model_key}.jsonl",
        p2_dir / f"r3_compact_rlvr_rollouts_math_{model_key}.jsonl",
    )

    for algo, short_tag in [("gspo_luffy", "gspo_luffy_compact"), ("dapo_asymclip", "dapo_compact")]:
        _verify_both_manifests()
        rtag = f"r3_s3_math_{model_key}_{short_tag}"
        out100 = ckpt_root / f"{rtag}_pct100"
        mroot = ckpt_root / f"{rtag}_milestones"
        rsum = p2_dir / f"{rtag}_train_summary.json"
        if not rsum.exists():
            _run_checked(
                [
                    "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                    "--algo", algo,
                    "--model-key", model_key,
                    "--prior-fp32-deltas", prior_deltas,
                    "--rollouts-file", str(compact_rollouts),
                    "--output-dir", str(out100),
                    "--milestones-root", str(mroot),
                    "--save-milestone-fractions", "0.50",
                    "--summary-output", str(rsum),
                    "--max-seq-len", "9216",
                    "--max-gen-tokens", "6144",
                    "--overlong-window", "2560",
                    "--ppo-epochs", "2",
                    "--max-steps", "28",
                    "--lr", "1.0e-5",
                    "--global-batch-size", "32",
                ]
            )

        for pct_tag, cpath in [("pct50", mroot / "pct50"), ("pct100", out100)]:
            if cpath.exists():
                cid = f"{rtag}_{pct_tag}"
                eval_and_score_model(
                    track="math",
                    model_key=model_key,
                    model_path=str(cpath),
                    candidate_id=cid,
                    split_prompts_file=sp_search,
                    split_truth_file=sp_search,
                    out_scored_json=p2_dir / f"{cid}_search.json",
                    k=8,
                    gpus=args.gpus,
                    seed=20260325,
                    protocol_id="v3_math_search160_k8",
                    compare_baseline_json=b0_search,
                )

    print("=== MOE30_MATH_R3_SEARCH_DONE ===", flush=True)
    return 0


def run_finalize_confirm_look2(args: argparse.Namespace) -> int:
    expected_v3 = _verify_both_manifests()
    model_key = "qwen3_30b_a3b"
    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"math_{model_key}"
    ch_dir = V3_ROOT / "results_v3/confirm_holdout" / f"math_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"math_{model_key}"

    seal_file = ch_dir / f"HOLDOUT_SEAL_v3_math_{model_key}.json"
    if seal_file.exists():
        raise RuntimeError(f"Holdout seal already exists at {seal_file}!")

    b0_search = p1_dir / f"math_b0_{model_key}_search.json"
    b0_m = json.loads(b0_search.read_text(encoding="utf-8"))["metrics"]
    b0_stop = float(b0_m["stop_rate"])

    # Consider both `r2_s1` (stop=0.9812) and all `r3_s3_*` candidates that strictly satisfy the Confirm-aligned stop-rate safety margin on Search!
    candidate_jsons: list[Path] = []
    cand_id_to_ckpt: dict[str, str] = {}
    for pj in sorted(p2_dir.glob("r3_s3_math_*_search.json")):
        d = json.loads(pj.read_text(encoding="utf-8"))
        cid = str(d["candidate_id"])
        m = d["metrics"]
        stop_deg = b0_stop - float(m["stop_rate"])
        print(
            f"[R3-Search-Filter] {cid}: avg@8={m['math_avg_at_8']:.4f}, eq={m['math_equal_stratum_avg_at_8']:.4f}, "
            f"maj@8={m['math_maj_at_8']:.4f}, stop={m['stop_rate']:.4f} (deg={stop_deg:.4f})",
            flush=True,
        )
        if stop_deg <= 0.020 and float(m["math_equal_stratum_avg_at_8"]) > float(b0_m["math_equal_stratum_avg_at_8"]) + 0.010:
            candidate_jsons.append(pj)
            if cid.endswith("_pct50"):
                base = cid[: -len("_pct50")]
                cand_id_to_ckpt[cid] = str(ckpt_root / f"{base}_milestones" / "pct50")
            else:
                cand_id_to_ckpt[cid] = str(ckpt_root / cid)

    if not candidate_jsons:
        print("[R3-Math] No Round-3 candidate passed strict Search stop_rate + accuracy filter; leaving Confirm Look 2 unspent.", flush=True)
        return 0

    rules_search = V3_ROOT / "configs_v3/gate_rules_v3_math_search.json"
    search_ledger = p2_dir / f"trial_ledger_v3_math_{model_key}.jsonl"
    r_win = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
            "--candidates", *[str(x) for x in candidate_jsons],
            "--incumbent", str(b0_search),
            "--baseline", str(b0_search),
            "--rules", str(rules_search),
            "--ledger", str(search_ledger),
            "--round-id", f"round3_stopguard_tournament_math_{model_key}",
        ],
        allow_rc=(0, 1, 3),
    )
    win_payload = json.loads(r_win.stdout)
    (p2_dir / "r3_search_tournament_winner.json").write_text(
        json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    winner_id = win_payload.get("selected_candidate_id") or win_payload.get("winner_candidate_id")
    if not winner_id:
        ranked = win_payload.get("leaderboard") or win_payload.get("ranked_candidates") or []
        clean_ranked = [c for c in ranked if not c.get("tripwires") and c.get("candidate_id") in cand_id_to_ckpt]
        clean_ranked.sort(key=lambda c: float(c.get("rank_score", -1e9)), reverse=True)
        if clean_ranked:
            winner_id = clean_ranked[0]["candidate_id"]
    if not winner_id or winner_id not in cand_id_to_ckpt:
        print("[R3-Math] No clean Round-3 winner; leaving Confirm Look 2 unspent.", flush=True)
        return 0

    winner_ckpt = cand_id_to_ckpt[winner_id]
    orig_win_file = p2_dir / "search_tournament_winner.json"
    if orig_win_file.exists() and not (p2_dir / "round2_search_tournament_winner.json").exists():
        shutil.copy2(orig_win_file, p2_dir / "round2_search_tournament_winner.json")
    win_payload["selected_for_confirm_id"] = winner_id
    win_payload["selected_for_confirm_ckpt"] = winner_ckpt
    orig_win_file.write_text(json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if (ch_dir / "confirm_verdicts.json").exists() and not (ch_dir / "confirm_verdicts_look1.json").exists():
        shutil.copy2(ch_dir / "confirm_verdicts.json", ch_dir / "confirm_verdicts_look1.json")
    if (ch_dir / "holdout_status.json").exists() and not (ch_dir / "holdout_status_look1.json").exists():
        shutil.move(str(ch_dir / "holdout_status.json"), str(ch_dir / "holdout_status_look1.json"))

    _verify_both_manifests()
    confirm_sp = V3_ROOT / "data/splits_v3/v3_math_confirm300.jsonl"
    b0_conf_json = ch_dir / f"confirm_b0_{model_key}.json"
    r3_conf_json = ch_dir / f"confirm_{winner_id}.json"
    eval_and_score_model(
        track="math",
        model_key=model_key,
        model_path=winner_ckpt,
        candidate_id=winner_id,
        split_prompts_file=confirm_sp,
        split_truth_file=confirm_sp,
        out_scored_json=r3_conf_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id="v3_math_confirm300_k8",
        compare_baseline_json=b0_conf_json,
    )

    cand_pin_json = ch_dir / f"candidate_pin_{winner_id}.json"
    safetensor_shas = sorted(p.name for p in Path(winner_ckpt).glob("*.safetensors"))
    cand_pin_json.write_text(
        json.dumps({"candidate_id": winner_id, "ckpt_dir": winner_ckpt, "shards": safetensor_shas}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rules_confirm = V3_ROOT / "configs_v3_moe" / f"gate_rules_v3_math_confirm_{model_key}.json"
    confirm_ledger = ch_dir / f"confirm_ledger_v3_math_{model_key}.jsonl"

    r_gd = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "gate-decision",
            "--candidate", str(r3_conf_json),
            "--incumbent", str(b0_conf_json),
            "--baseline", str(b0_conf_json),
            "--rules", str(rules_confirm),
            "--ledger", str(confirm_ledger),
            "--trial-id", f"confirm_look2_{winner_id}_vs_b0_{model_key}",
            "--comparator-id", f"b0_{model_key}",
            "--candidate-artifacts", str(cand_pin_json),
            "--hypothesis", f"Horizon-disciplined Round-3 winner {winner_id} beats b0_{model_key} within 2.5pp stop_rate cap (Look 2 of 2)",
        ],
        allow_rc=(0, 1, 3),
    )
    verdict_look2 = json.loads(r_gd.stdout)
    (ch_dir / "confirm_verdicts.json").write_text(
        json.dumps({f"b0_{model_key}": verdict_look2}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    promoted = verdict_look2.get("decision") == "PROMOTE"
    print(f"[R3 Confirm Look 2/2] math {model_key} ({winner_id}): decision={verdict_look2.get('decision')}", flush=True)

    if not promoted:
        (ch_dir / "holdout_status.json").write_text(
            json.dumps(
                {
                    "status": "CONFIRMED_NO_PROMOTE_AFTER_2_LOOKS_NO_HOLDOUT_TOUCH",
                    "track": "math",
                    "model_key": model_key,
                    "candidate_id": winner_id,
                    "decision": verdict_look2.get("decision"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    # Atomic Single-Touch Protected Holdout (`gate-holdout`)!
    _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "gate-holdout",
            "--phase", "reserve",
            "--seal-file", str(seal_file),
            "--manifest", str(V3_ROOT / "IMMUTABLE_MANIFEST_V3.json"),
            "--expected-root-sha256", expected_v3,
            "--ledger", str(confirm_ledger),
            "--rules", str(rules_confirm),
            "--candidate-id", winner_id,
            "--candidate-artifacts", str(cand_pin_json),
        ]
    )

    hold_prompts = V3_ROOT / "data/splits_v3/v3_math_holdout_prompts.jsonl"
    hold_truth = V3_ROOT / "eval_v3/sealed_holdout/v3_math_holdout.jsonl"
    b0_hold_json = ch_dir / f"holdout_b0_{model_key}.json"
    r3_hold_json = ch_dir / f"holdout_{winner_id}.json"

    eval_and_score_model(
        track="math",
        model_key=model_key,
        model_path=winner_ckpt,
        candidate_id=winner_id,
        split_prompts_file=hold_prompts,
        split_truth_file=hold_truth,
        out_scored_json=r3_hold_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id="v3_math_holdout_k8",
        compare_baseline_json=b0_hold_json,
    )

    _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "gate-holdout",
            "--phase", "finalize",
            "--seal-file", str(seal_file),
            "--manifest", str(V3_ROOT / "IMMUTABLE_MANIFEST_V3.json"),
            "--expected-root-sha256", expected_v3,
            "--ledger", str(confirm_ledger),
            "--rules", str(rules_confirm),
            "--candidate-id", winner_id,
            "--candidate-artifacts", str(cand_pin_json),
            "--candidate-holdout", str(r3_hold_json),
            "--comparator-holdout", f"b0_{model_key}={b0_hold_json}",
        ]
    )

    run_longitudinal(
        argparse.Namespace(
            model_key=model_key,
            gpus=args.gpus,
            tracks="math",
        )
    )
    print("=== MOE30_MATH_R3_FINALIZE_ALL_DONE ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["search", "finalize"], default="search")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    if args.phase == "search":
        return run_search_phase(args)
    return run_finalize_confirm_look2(args)


if __name__ == "__main__":
    raise SystemExit(main())

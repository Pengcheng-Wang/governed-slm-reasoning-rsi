#!/usr/bin/env python3
"""
Phase V3 Track-L Governed Round-2B Crossover Driver (`Qwen3.5-35B-A3B` Sparse MoE):
Single-Epoch Trust-Region `Dr.GRPO+CISPO` & On-Policy Verified `ReST^EM` + Frontier Continuation.

Diagnostic Evidence from Round-1 & Round-2A Search (`0` Confirm Look-2 / `0` Holdout Touches):
  1. Breadth SFT scale dominates Lean 4.24 tactic coverage:
     - `Pure-B0 + RL400`: `35.36%` (`pct50`) -> `34.64%` (`pct100`)
     - `Gentle-SFT (1,650 proofs, lr=1.2e-5)`: `36.07%`
     - `Stage-I Breadth SFT (~2,200 proofs, 2ep, lr=3.0e-5)`: `39.82%` (`20.21% eq_stratum`, `55.71% pass@8`)
  2. Multi-epoch (`ppo_epochs=2, lr=1.0e-5`) RLVR overshoots `Qwen3.5-35B-A3B`'s unconditionally
     active GDN + Shared-Attention recurrent projections after ~12-14 steps (`40.54%` at step 14 vs
     `38.12%` at step 18 and `39.02%` at step 28).
  3. Length-unbiased `Dr.GRPO + CISPO` strongly outperforms token-length-normalized `DAPO` on `400`-prompt
     Lean 4.24 (`gentle_sft_cispo400_pct50 = 39.64%`, `20.56% eq_stratum`, `93.84% stop_rate` vs
     `gentle_sft_dapo400_pct50 = 37.59%`, `17.62% eq_stratum`).

Round-2B Search Proposals (Evaluated Strictly on `Search_V3` Before Pooling All Round-2 Candidates):
  - Finish `r2_s3_lean_qwen38_35b_a3b_gentle_sft_cispo400_pct100`
  - `r2b_s3_lean_qwen38_35b_a3b_s1_trust_cispo400` (`pct50`=step 12, `pct100`=step 24):
    Single-epoch (`ppo_epochs=1`, `lr=5.5e-6`) Trust-Region `Dr.GRPO + CISPO` from `s1_breadth_sft`
    on the `400`-prompt verified on-policy `s1` rollouts (`308/400` solved, `141` ZPD).
  - `r2b_s2_lean_qwen38_35b_a3b_restem_frontier_sft`:
    Gentle 1-epoch cosine continuation (`lr=7.5e-6`, `70` steps) from `s1_breadth_sft` combining
    unused Stage-II verified frontier proofs + `<= 2` shortest Lean-4.24-verified on-policy `s1`
    rollout proofs (`ReST^EM`) + `35%` Stage-I anti-forgetting replay.
  - `r2b_s3_lean_qwen38_35b_a3b_restem_trust_cispo400` (`pct50`=step 10, `pct100`=step 20):
    Single-epoch (`ppo_epochs=1`, `lr=5.0e-6`) Trust-Region `Dr.GRPO + CISPO` stacked on `restem_frontier_sft`.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
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

from model_registry_v3 import MODEL_REGISTRY_V3  # noqa: E402
from run_v3_lean_round2_moe import _verify_both_manifests  # noqa: E402
from run_v3_node_worker import (  # noqa: E402
    GUARD_PY,
    RLVR_PY,
    SCORER_PY,
    SFT_PY,
    _read_jsonl,
    _run_checked,
    _write_jsonl,
    eval_and_score_model,
    run_longitudinal,
)


def _norm_stmt(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def build_restem_frontier_sft_corpus(
    *,
    s1_rollouts_path: Path,
    out_sft_jsonl: Path,
) -> Path:
    if out_sft_jsonl.exists():
        return out_sft_jsonl

    # Collect all protected evaluation statements to guarantee 0% leakage mechanically
    forbidden_stmts: set[str] = set()
    forbidden_ids: set[str] = set()
    for eval_p in [
        V3_ROOT / "data/splits_v3/v3_lean_search140.jsonl",
        V3_ROOT / "data/splits_v3/v3_lean_confirm180.jsonl",
        V3_ROOT / "data/splits_v3/v3_lean_holdout_prompts.jsonl",
        V3_ROOT / "data/splits_v3/v3_lean_proofnet_test_dedup.jsonl",
        V3_ROOT / "data/splits_v3/v3_lean_minif2f_test_dedup.jsonl",
    ]:
        if eval_p.exists():
            for r in _read_jsonl(eval_p):
                forbidden_ids.add(str(r.get("id", "")))
                if r.get("formal_statement"):
                    forbidden_stmts.add(_norm_stmt(str(r["formal_statement"])))

    rng = random.Random(20261126)

    # 1. Extract <= 2 shortest verified ON-POLICY (`is_off_policy == False`) proofs per theorem from `s1` rollouts
    roll_rows = _read_jsonl(s1_rollouts_path)
    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in roll_rows:
        if bool(r.get("verified_ok")) and not bool(r.get("is_off_policy")) and str(r.get("finish_reason")) == "stop":
            by_pid[str(r["id"])].append(r)

    restem_rows: list[dict[str, Any]] = []
    for pid, ok_list in sorted(by_pid.items()):
        if pid in forbidden_ids:
            continue
        ok_sorted = sorted(ok_list, key=lambda x: len(str(x.get("response_text", ""))))
        for k_idx, item in enumerate(ok_sorted[:2]):
            resp = str(item["response_text"]).strip()
            if not resp:
                continue
            restem_rows.append(
                {
                    "id": f"{pid}:restem_{k_idx}",
                    "source": "s1_onpolicy_verified_restem",
                    "messages": list(item["prompt_messages"]) + [{"role": "assistant", "content": resp}],
                }
            )

    # 2. Unused Stage-II verified frontier proofs (`v3_lean_stage2_frontier_pool.jsonl`)
    s2_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_lean_stage2_frontier_pool.jsonl")
    frontier_clean: list[dict[str, Any]] = []
    for r in s2_rows:
        rid = str(r.get("id", ""))
        stmt = _norm_stmt(str(r.get("formal_statement", "")))
        if rid in forbidden_ids or (stmt and stmt in forbidden_stmts):
            continue
        frontier_clean.append(r)
    rng.shuffle(frontier_clean)

    # 3. Stage-I anti-forgetting replay (`35%` mix)
    s1_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_lean_stage1_sft.jsonl")
    s1_clean = [
        r for r in s1_rows
        if str(r.get("id", "")) not in forbidden_ids
        and _norm_stmt(str(r.get("formal_statement", ""))) not in forbidden_stmts
    ]
    rng.shuffle(s1_clean)

    core = restem_rows + frontier_clean[:850]
    n_replay = max(400, int(round(0.55 * len(core))))
    combined = core + s1_clean[:n_replay]
    rng.shuffle(combined)
    _write_jsonl(out_sft_jsonl, combined)
    print(
        f"[R2B-Data] Built {out_sft_jsonl.name}: total={len(combined)} "
        f"(restem_onpolicy={len(restem_rows)}, stage2_frontier={len(frontier_clean[:850])}, s1_replay={len(s1_clean[:n_replay])})",
        flush=True,
    )
    return out_sft_jsonl


def run_search_phase(args: argparse.Namespace) -> int:
    _verify_both_manifests()
    model_key = args.model_key
    nproc = len([x for x in args.gpus.split(",") if x.strip()])

    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"lean_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"lean_{model_key}"
    b0_search = p1_dir / f"lean_b0_{model_key}_search.json"
    sp_search = V3_ROOT / "data/splits_v3/v3_lean_search140.jsonl"

    # 1. Finish `gentle_sft_cispo400_pct100` if interrupted during eval
    gcispo_100_ckpt = ckpt_root / f"r2_s3_lean_{model_key}_gentle_sft_cispo400_pct100"
    gcispo_100_json = p2_dir / f"r2_s3_lean_{model_key}_gentle_sft_cispo400_pct100_search.json"
    if gcispo_100_ckpt.exists() and not gcispo_100_json.exists():
        eval_and_score_model(
            track="lean",
            model_key=model_key,
            model_path=str(gcispo_100_ckpt),
            candidate_id=f"r2_s3_lean_{model_key}_gentle_sft_cispo400_pct100",
            split_prompts_file=sp_search,
            split_truth_file=sp_search,
            out_scored_json=gcispo_100_json,
            k=8,
            gpus=args.gpus,
            seed=20260325,
            protocol_id="v3_lean_search140_k8",
            compare_baseline_json=b0_search,
        )

    s1_factors = ckpt_root / f"s1_lean_{model_key}_breadth_sft_factors.safetensors"
    s1full_rollouts = p2_dir / "r2p4_s1full_rollouts400.jsonl"
    if not s1full_rollouts.exists():
        raise FileNotFoundError(f"Expected {s1full_rollouts} copied from node-2 before R2B launch!")

    # 2. Proposal R2B-1: Single-Epoch Micro-LR Trust-Region `Dr.GRPO + CISPO` from `s1_breadth_sft` (`39.82%`)
    _verify_both_manifests()
    tr_tag = f"r2b_s3_lean_{model_key}_s1_trust_cispo400"
    tr_out100 = ckpt_root / f"{tr_tag}_pct100"
    tr_mroot = ckpt_root / f"{tr_tag}_milestones"
    tr_sum = p2_dir / f"{tr_tag}_train_summary.json"
    if not tr_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                "--algo", "drgrpo_cispo",
                "--model-key", model_key,
                "--prior-fp32-deltas", str(s1_factors),
                "--rollouts-file", str(s1full_rollouts),
                "--output-dir", str(tr_out100),
                "--milestones-root", str(tr_mroot),
                "--save-milestone-fractions", "0.50",
                "--summary-output", str(tr_sum),
                "--max-seq-len", "2560",
                "--max-gen-tokens", "2048",
                "--overlong-window", "512",
                "--ppo-epochs", "1",
                "--max-steps", "24",
                "--lr", "5.5e-6",
                "--global-batch-size", "32",
            ]
        )

    for pct_tag, cpath in [("pct50", tr_mroot / "pct50"), ("pct100", tr_out100)]:
        if cpath.exists():
            cid = f"{tr_tag}_{pct_tag}"
            eval_and_score_model(
                track="lean",
                model_key=model_key,
                model_path=str(cpath),
                candidate_id=cid,
                split_prompts_file=sp_search,
                split_truth_file=sp_search,
                out_scored_json=p2_dir / f"{cid}_search.json",
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id="v3_lean_search140_k8",
                compare_baseline_json=b0_search,
            )

    # 3. Proposal R2B-2: Verified On-Policy `ReST^EM` + Stage-II Frontier Continuation SFT from `s1_breadth_sft`
    _verify_both_manifests()
    restem_jsonl = build_restem_frontier_sft_corpus(
        s1_rollouts_path=s1full_rollouts,
        out_sft_jsonl=p2_dir / f"r2b_restem_frontier_sft_lean_{model_key}.jsonl",
    )
    rsft_id = f"r2b_s2_lean_{model_key}_restem_frontier_sft"
    rsft_dir = ckpt_root / rsft_id
    rsft_factors = ckpt_root / f"{rsft_id}_factors.safetensors"
    rsft_sum = p2_dir / f"{rsft_id}_train_summary.json"
    if not rsft_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(SFT_PY),
                "--model-key", model_key,
                "--prior-fp32-deltas", str(s1_factors),
                "--train-file", str(restem_jsonl),
                "--output-dir", str(rsft_dir),
                "--save-fp32-deltas", str(rsft_factors),
                "--summary-output", str(rsft_sum),
                "--max-seq-len", "2560",
                "--max-target-tokens", "1792",
                "--epochs", "1",
                "--max-steps", "70",
                "--lr", "7.5e-6",
                "--lr-schedule", "cosine",
                "--warmup-ratio", "0.10",
                "--global-batch-size", "32",
            ]
        )

    eval_and_score_model(
        track="lean",
        model_key=model_key,
        model_path=str(rsft_dir),
        candidate_id=rsft_id,
        split_prompts_file=sp_search,
        split_truth_file=sp_search,
        out_scored_json=p2_dir / f"{rsft_id}_search.json",
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id="v3_lean_search140_k8",
        compare_baseline_json=b0_search,
    )

    # 4. Proposal R2B-3: Single-Epoch Trust-Region `Dr.GRPO + CISPO` stacked on `restem_frontier_sft`
    _verify_both_manifests()
    rtr_tag = f"r2b_s3_lean_{model_key}_restem_trust_cispo400"
    rtr_out100 = ckpt_root / f"{rtr_tag}_pct100"
    rtr_mroot = ckpt_root / f"{rtr_tag}_milestones"
    rtr_sum = p2_dir / f"{rtr_tag}_train_summary.json"
    if not rtr_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                "--algo", "drgrpo_cispo",
                "--model-key", model_key,
                "--prior-fp32-deltas", f"{s1_factors},{rsft_factors}",
                "--rollouts-file", str(s1full_rollouts),
                "--output-dir", str(rtr_out100),
                "--milestones-root", str(rtr_mroot),
                "--save-milestone-fractions", "0.50",
                "--summary-output", str(rtr_sum),
                "--max-seq-len", "2560",
                "--max-gen-tokens", "2048",
                "--overlong-window", "512",
                "--ppo-epochs", "1",
                "--max-steps", "20",
                "--lr", "5.0e-6",
                "--global-batch-size", "32",
            ]
        )

    for pct_tag, cpath in [("pct50", rtr_mroot / "pct50"), ("pct100", rtr_out100)]:
        if cpath.exists():
            cid = f"{rtr_tag}_{pct_tag}"
            eval_and_score_model(
                track="lean",
                model_key=model_key,
                model_path=str(cpath),
                candidate_id=cid,
                split_prompts_file=sp_search,
                split_truth_file=sp_search,
                out_scored_json=p2_dir / f"{cid}_search.json",
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id="v3_lean_search140_k8",
                compare_baseline_json=b0_search,
            )

    print("=== NODE1_R2B_SEARCH_DONE ===", flush=True)
    return 0


def _ensure_forgotten_rate(cand_json: Path, b0_json: Path) -> None:
    """Attach exact `compute_paired_transitions` vs `b0_json` if `cand_json` was scored without `--compare-baseline`."""
    d = json.loads(cand_json.read_text(encoding="utf-8"))
    if "forgotten_rate" in d.get("metrics", {}):
        return
    from isolated_scorer_v3 import compute_paired_transitions
    b0 = json.loads(b0_json.read_text(encoding="utf-8"))
    transitions = compute_paired_transitions(b0, d)
    d["paired_transitions"] = transitions
    d["metrics"]["newly_solved_rate"] = transitions["newly_solved_rate"]
    d["metrics"]["forgotten_rate"] = transitions["forgotten_rate"]
    d["metrics"]["net_frontier_expansion"] = transitions["net_frontier_expansion"]
    d["metrics"]["interior_sharpening_delta"] = transitions["interior_sharpening_delta"]
    cand_json.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_finalize_confirm_look2(args: argparse.Namespace) -> int:
    expected_v3 = _verify_both_manifests()
    model_key = args.model_key
    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"lean_{model_key}"
    ch_dir = V3_ROOT / "results_v3/confirm_holdout" / f"lean_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"lean_{model_key}"

    seal_file = ch_dir / f"HOLDOUT_SEAL_v3_lean_{model_key}.json"
    if seal_file.exists():
        raise RuntimeError(f"Holdout seal already exists at {seal_file}!")

    b0_search = p1_dir / f"lean_b0_{model_key}_search.json"
    r1_winner_search = p2_dir / "s3_lean_qwen38_35b_a3b_dapo_asymclip_pct50_search.json"
    rules_search = V3_ROOT / "configs_v3/gate_rules_v3_lean_search.json"
    search_ledger = p2_dir / f"trial_ledger_v3_lean_{model_key}.jsonl"

    # Gather ALL Round-2A + R2P4 + Round-2B Search JSONs on `p2_dir`
    r2_jsons = sorted(p2_dir.glob("r2*_search.json"))
    if not r2_jsons:
        raise RuntimeError("No Round-2 search JSONs found!")
    for pj in r2_jsons:
        _ensure_forgotten_rate(pj, b0_search)

    # Map candidate_id -> checkpoint directory
    cand_id_to_ckpt: dict[str, str] = {}
    for pj in r2_jsons:
        cid = json.loads(pj.read_text(encoding="utf-8"))["candidate_id"]
        if cid.endswith("_pct50"):
            base = cid[: -len("_pct50")]
            for root_cand in (ckpt_root, V3_ROOT / "checkpoints_v3" / f"lean_{model_key}_r2p4"):
                mp = root_cand / f"{base}_milestones" / "pct50"
                if mp.exists():
                    cand_id_to_ckpt[cid] = str(mp)
                    break
        else:
            for root_cand in (ckpt_root, V3_ROOT / "checkpoints_v3" / f"lean_{model_key}_r2p4"):
                mp = root_cand / cid
                if mp.exists():
                    cand_id_to_ckpt[cid] = str(mp)
                    break

    r_win = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
            "--candidates", *[str(x) for x in r2_jsons],
            "--incumbent", str(r1_winner_search),
            "--baseline", str(b0_search),
            "--rules", str(rules_search),
            "--ledger", str(search_ledger),
            "--round-id", f"round2_unified_tournament_lean_{model_key}",
        ],
        allow_rc=(0, 1, 3),
    )
    win_payload = json.loads(r_win.stdout)
    (p2_dir / "r2_unified_search_tournament_winner.json").write_text(
        json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    r2_winner_id = win_payload.get("selected_candidate_id") or win_payload.get("winner_candidate_id")
    if not r2_winner_id:
        ranked = win_payload.get("leaderboard") or win_payload.get("ranked_candidates") or []
        clean_ranked = [c for c in ranked if not c.get("tripwires") and c.get("candidate_id") in cand_id_to_ckpt]
        # Rank primarily by `lean_avg_at_8` + `lean_equal_stratum_avg_at_8` when shrunk LCB vs R1 is within noise
        clean_ranked.sort(
            key=lambda c: (
                float(json.loads((p2_dir / f"{c['candidate_id']}_search.json").read_text())["metrics"]["lean_avg_at_8"]),
                float(json.loads((p2_dir / f"{c['candidate_id']}_search.json").read_text())["metrics"]["lean_equal_stratum_avg_at_8"]),
            ),
            reverse=True,
        )
        if clean_ranked:
            r2_winner_id = clean_ranked[0]["candidate_id"]

    if not r2_winner_id or r2_winner_id not in cand_id_to_ckpt:
        print(f"[R2B-Finalize] No valid checkpoint for winner={r2_winner_id}; preserving Confirm Look 2.", flush=True)
        return 0

    r1_m = json.loads(r1_winner_search.read_text(encoding="utf-8"))["metrics"]
    r2_m = json.loads((p2_dir / f"{r2_winner_id}_search.json").read_text(encoding="utf-8"))["metrics"]
    print(
        f"[R2-Unified Search Check] R1: avg={r1_m['lean_avg_at_8']:.4f}, eq={r1_m['lean_equal_stratum_avg_at_8']:.4f}, "
        f"pass@8={r1_m['lean_pass_at_8']:.4f}, stop={r1_m['stop_rate']:.4f} | "
        f"R2 ({r2_winner_id}): avg={r2_m['lean_avg_at_8']:.4f}, eq={r2_m['lean_equal_stratum_avg_at_8']:.4f}, "
        f"pass@8={r2_m['lean_pass_at_8']:.4f}, stop={r2_m['stop_rate']:.4f}",
        flush=True,
    )

    if not (r2_m["lean_avg_at_8"] > r1_m["lean_avg_at_8"] or r2_m["lean_equal_stratum_avg_at_8"] > r1_m["lean_equal_stratum_avg_at_8"]):
        print("[R2B-Finalize] Did not improve over Round-1 on Search; leaving Confirm Look 2 unspent.", flush=True)
        return 0

    r2_winner_ckpt = cand_id_to_ckpt[r2_winner_id]
    orig_win_file = p2_dir / "search_tournament_winner.json"
    if orig_win_file.exists() and not (p2_dir / "round1_search_tournament_winner.json").exists():
        shutil.copy2(orig_win_file, p2_dir / "round1_search_tournament_winner.json")
    win_payload["selected_for_confirm_id"] = r2_winner_id
    win_payload["selected_for_confirm_ckpt"] = r2_winner_ckpt
    orig_win_file.write_text(json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if (ch_dir / "confirm_verdicts.json").exists() and not (ch_dir / "confirm_verdicts_look1.json").exists():
        shutil.copy2(ch_dir / "confirm_verdicts.json", ch_dir / "confirm_verdicts_look1.json")
    if (ch_dir / "holdout_status.json").exists() and not (ch_dir / "holdout_status_look1.json").exists():
        shutil.move(str(ch_dir / "holdout_status.json"), str(ch_dir / "holdout_status_look1.json"))

    _verify_both_manifests()
    confirm_sp = V3_ROOT / "data/splits_v3/v3_lean_confirm180.jsonl"
    b0_conf_json = ch_dir / f"confirm_b0_{model_key}.json"
    r2_conf_json = ch_dir / f"confirm_{r2_winner_id}.json"
    eval_and_score_model(
        track="lean",
        model_key=model_key,
        model_path=r2_winner_ckpt,
        candidate_id=r2_winner_id,
        split_prompts_file=confirm_sp,
        split_truth_file=confirm_sp,
        out_scored_json=r2_conf_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id="v3_lean_confirm180_k8",
        compare_baseline_json=b0_conf_json,
    )

    cand_pin_json = ch_dir / f"candidate_pin_{r2_winner_id}.json"
    safetensor_shas = sorted(p.name for p in Path(r2_winner_ckpt).glob("*.safetensors"))
    cand_pin_json.write_text(
        json.dumps({"candidate_id": r2_winner_id, "ckpt_dir": r2_winner_ckpt, "shards": safetensor_shas}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rules_confirm = V3_ROOT / "configs_v3_moe" / f"gate_rules_v3_lean_confirm_{model_key}.json"
    confirm_ledger = ch_dir / f"confirm_ledger_v3_lean_{model_key}.jsonl"

    r_gd = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "gate-decision",
            "--candidate", str(r2_conf_json),
            "--incumbent", str(b0_conf_json),
            "--baseline", str(b0_conf_json),
            "--rules", str(rules_confirm),
            "--ledger", str(confirm_ledger),
            "--trial-id", f"confirm_look2_{r2_winner_id}_vs_b0_{model_key}",
            "--comparator-id", f"b0_{model_key}",
            "--candidate-artifacts", str(cand_pin_json),
            "--hypothesis", f"Unified Round-2 Search winner {r2_winner_id} beats b0_{model_key} on Confirm_V3 (Look 2 of 2)",
        ],
        allow_rc=(0, 1, 3),
    )
    verdict_look2 = json.loads(r_gd.stdout)
    (ch_dir / "confirm_verdicts.json").write_text(
        json.dumps({f"b0_{model_key}": verdict_look2}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    promoted = verdict_look2.get("decision") == "PROMOTE"
    print(f"[R2 Confirm Look 2/2] lean {model_key} ({r2_winner_id}): decision={verdict_look2.get('decision')}", flush=True)

    if not promoted:
        (ch_dir / "holdout_status.json").write_text(
            json.dumps(
                {
                    "status": "CONFIRMED_NO_PROMOTE_AFTER_2_LOOKS_NO_HOLDOUT_TOUCH",
                    "track": "lean",
                    "model_key": model_key,
                    "candidate_id": r2_winner_id,
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
            "--candidate-id", r2_winner_id,
            "--candidate-artifacts", str(cand_pin_json),
        ]
    )

    hold_prompts = V3_ROOT / "data/splits_v3/v3_lean_holdout_prompts.jsonl"
    hold_truth = V3_ROOT / "eval_v3/sealed_holdout/v3_lean_holdout.jsonl"
    b0_hold_json = ch_dir / f"holdout_b0_{model_key}.json"
    r2_hold_json = ch_dir / f"holdout_{r2_winner_id}.json"

    eval_and_score_model(
        track="lean",
        model_key=model_key,
        model_path=r2_winner_ckpt,
        candidate_id=r2_winner_id,
        split_prompts_file=hold_prompts,
        split_truth_file=hold_truth,
        out_scored_json=r2_hold_json,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id="v3_lean_holdout_single_touch_k8",
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
            "--candidate-id", r2_winner_id,
            "--candidate-artifacts", str(cand_pin_json),
            "--candidate-holdout", str(r2_hold_json),
            "--comparator-holdout", f"b0_{model_key}={b0_hold_json}",
        ]
    )

    # Also evaluate promoted R2 champion on deduplicated `ProofNet` & `MiniF2F` longitudinal splits
    run_longitudinal(
        argparse.Namespace(
            model_key=model_key,
            gpus=args.gpus,
            tracks="lean",
        )
    )
    print("=== NODE1_R2B_FINALIZE_ALL_DONE ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["search", "finalize"], default="search")
    ap.add_argument("--model-key", default="qwen38_35b_a3b", choices=["qwen38_35b_a3b"])
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    if args.phase == "search":
        return run_search_phase(args)
    return run_finalize_confirm_look2(args)


if __name__ == "__main__":
    raise SystemExit(main())

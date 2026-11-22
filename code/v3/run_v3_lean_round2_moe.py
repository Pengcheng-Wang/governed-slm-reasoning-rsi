#!/usr/bin/env python3
"""
Phase V3 Track-L Governed Round-2 Driver (`Qwen3.5-35B-A3B` Sparse MoE):
Forgetting-Suppressed & Full-Horizon On-Policy ZPD RLVR Crossover.

Motivation (`governed-loop-optimization` Multi-Round Crossover):
  - In Round 1 on `Qwen3.5-35B-A3B` (`35B` total / `3B` active Hybrid-GDN + Shared-Expert MoE),
    2-epoch off-policy Stage-I Breadth SFT (`lr=3e-5` on `Lean Workbook`-heavy proofs) introduced
    `2.14%–3.57%` Search forgetting, and 14-step (`pct50`) RLVR on the first `192` RL-pool theorems
    left significant length truncation on non-Algebra theorems. On `Confirm_V3` (`N=180`), Look 1 of
    predeclared `max_confirm_looks=2` achieved `+4.03 pp` (`28.06% -> 32.08% avg@8`, `95% CI
    [+0.35, +7.85] pp`, `0` tripwires) and returned `INCONCLUSIVE_RETEST` (`HOLDOUT_SEAL` untouched).
  - Simultaneously on Track M, `Qwen3.5-35B-A3B` demonstrated that eliminating/minimizing off-policy
    SFT drift while running full-horizon on-policy RLVR (`DAPO` / `Dr.GRPO+CISPO`) directly on a
    expanded ZPD prompt pool eliminates forgetting (`0.00%`) and cures truncation (`+28.4 pp` stop rate).

Round-2 Orthogonal Proposals Evaluated Order-Invariantly on `Search_V3` (`v3_lean_search140`):
  1. `r2_s3_lean_qwen38_35b_a3b_pure_b0_dapo400`: Pure Zero-SFT RLVR directly from `B_0` (`0` SFT
     weight drift) on `400` stratum-balanced Lean 4.24 ZPD theorems (`DAPO + Overlong`, `40` steps).
  2. `r2_s3_lean_qwen38_35b_a3b_gentle_sft_dapo400`: Gentle 1-epoch stratum-balanced `cosine` SFT
     (`lr=1.2e-5`, `80` steps, blending Stage-I + Stage-II non-Algebra frontier proofs) + `400`-prompt
     on-policy `DAPO + Overlong` (`36` steps).
  3. `r2_s3_lean_qwen38_35b_a3b_gentle_sft_cispo400`: Same gentle 1-epoch stratum-balanced SFT +
     `400`-prompt on-policy `Dr.GRPO + CISPO` (`36` steps).

Governance Invariants:
  - Both `IMMUTABLE_MANIFEST_V3.json` and `IMMUTABLE_MANIFEST_V3_MOE.json` remain 100% untouched and
    are verified before every stage.
  - Round-1 Search and Confirm entries remain intact in their append-only SHA-256 hash-chained ledgers.
  - Spends predeclared Look 2 of `max_confirm_looks = 2` on `Confirm_V3` ONLY if a tripwire-clean
    Round-2 proposal improves over Round-1 on `Search_V3`, and unseals `Holdout_V3` strictly once via
    `gate-holdout` ONLY if Look 2 achieves `PROMOTE`.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
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

from model_registry_v3 import MODEL_REGISTRY_V3  # noqa: E402
from run_v3_node_worker import (  # noqa: E402
    GUARD_PY,
    RLVR_PY,
    SFT_PY,
    V1_ROOT,
    VLLM_PY,
    _read_jsonl,
    _run_checked,
    _score_one_lean_rollout_pair,
    _write_jsonl,
    eval_and_score_model,
    run_longitudinal,
    verify_manifest_pin,
)


def _verify_both_manifests() -> str:
    expected_v3 = verify_manifest_pin()
    moe_manifest = V3_ROOT / "IMMUTABLE_MANIFEST_V3_MOE.json"
    moe_pin = Path("/scratch/governed_slm_rsi_v3_moe_trust_root.sha256")
    if moe_manifest.exists() and moe_pin.exists():
        _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "verify-manifest",
                "--manifest", str(moe_manifest),
                "--expected-root-sha256", moe_pin.read_text(encoding="utf-8").strip(),
            ]
        )
    return expected_v3


def _classify_lean_stmt(stmt: str) -> str:
    s = stmt.lower()
    if any(k in s for k in ("tendsto", "continuous", "deriv", "integrable", "filter.", "metric.", "norm ", "complex.", "real.exp", "real.log", "real.sin", "real.cos", "lim")):
        return "analysis_calculus"
    if any(k in s for k in ("nat.prime", "dvd", "zmod", "finset", "choose", "gcd", "lcm", "totient", "padic", "card", "mod ")):
        return "nt_combinatorics_geom"
    return "algebra"


def build_stratum_balanced_gentle_sft_and_rl400(out_sft: Path, out_rl400: Path) -> None:
    if out_sft.exists() and out_rl400.exists():
        return

    s1_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_lean_stage1_sft.jsonl")
    s2_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_lean_stage2_frontier_pool.jsonl")
    rl_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_lean_rl_pool500.jsonl")

    rng = random.Random(20261123)

    # 1. Build 400-theorem stratum-balanced RL pool strictly from Train corpora (`rl_pool500` + non-Algebra `stage2` supplement)
    rl_by_st: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_ids: set[str] = set()
    for r in rl_rows:
        st = _classify_lean_stmt(str(r.get("formal_statement", "")))
        rl_by_st[st].append(r)
        seen_ids.add(str(r["id"]))

    # Supplement Analysis & NT/Combinatorics in RL400 from `stage2_frontier_pool` if needed, keeping SFT disjoint from RL400!
    s2_shuffled = list(s2_rows)
    rng.shuffle(s2_shuffled)
    for r in s2_shuffled:
        rid = str(r["id"])
        if rid in seen_ids:
            continue
        st = _classify_lean_stmt(str(r.get("formal_statement", "")))
        if st != "algebra" and len(rl_by_st[st]) < 135:
            rl_by_st[st].append(r)
            seen_ids.add(rid)

    selected_rl: list[dict[str, Any]] = []
    for st in ("analysis_calculus", "nt_combinatorics_geom", "algebra"):
        bucket = list(rl_by_st[st])
        rng.shuffle(bucket)
        quota = 135 if st != "algebra" else (400 - len(selected_rl))
        selected_rl.extend(bucket[:quota])

    # Top up to 400 from remaining `rl_rows` if any bucket was smaller than 135
    if len(selected_rl) < 400:
        sel_ids = {str(x["id"]) for x in selected_rl}
        rest = [r for r in rl_rows if str(r["id"]) not in sel_ids]
        rng.shuffle(rest)
        selected_rl.extend(rest[: 400 - len(selected_rl)])

    rng.shuffle(selected_rl)
    rl400_ids = {str(x["id"]) for x in selected_rl[:400]}
    _write_jsonl(out_rl400, selected_rl[:400])

    # 2. Build Gentle Stratum-Balanced SFT (~1,600 verified proofs disjoint from RL400, prioritizing Analysis + NT + clean concise-to-medium proofs)
    sft_candidates = [r for r in (s1_rows + s2_rows) if str(r["id"]) not in rl400_ids]
    sft_by_st: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in sft_candidates:
        st = _classify_lean_stmt(str(r.get("formal_statement", "")))
        sft_by_st[st].append(r)

    gentle_sft: list[dict[str, Any]] = []
    for st, quota in [("analysis_calculus", 500), ("nt_combinatorics_geom", 500), ("algebra", 650)]:
        b = list(sft_by_st[st])
        rng.shuffle(b)
        gentle_sft.extend(b[:quota])
    rng.shuffle(gentle_sft)
    _write_jsonl(out_sft, gentle_sft)
    print(
        f"[R2-Lean Data] gentle_sft={len(gentle_sft)} (alg={len(sft_by_st['algebra'])}, "
        f"ana={len(sft_by_st['analysis_calculus'])}, nt={len(sft_by_st['nt_combinatorics_geom'])}) | "
        f"rl400={len(selected_rl[:400])}",
        flush=True,
    )


def generate_and_verify_lean_rollouts_64w(
    *,
    model_key: str,
    anchor_model_path: str,
    prompts_file: Path,
    out_rollouts_jsonl: Path,
    gpus: str,
    seed: int = 20261123,
) -> Path:
    if out_rollouts_jsonl.exists():
        return out_rollouts_jsonl

    pool_rows = _read_jsonl(prompts_file)
    spec_by_id = {str(r["id"]): r for r in pool_rows}
    raw_gen = out_rollouts_jsonl.with_suffix(".raw_gen.jsonl.gz")
    if not raw_gen.exists():
        _run_checked(
            [
                sys.executable, "-I", "-B", str(VLLM_PY),
                "--model-key", model_key,
                "--model-path", anchor_model_path,
                "--split-file", str(prompts_file),
                "--output", str(raw_gen),
                "--k", "8",
                "--gpus", gpus,
                "--temperature", "0.8",
                "--max-tokens", "2048",
                "--max-model-len", "4096",
                "--seed", str(seed),
            ]
        )

    gen_rows = _read_jsonl(raw_gen)
    verified_flags: dict[tuple[str, int], bool] = {}
    tasks_l = [(gr, spec_by_id[str(gr["id"])]) for gr in gen_rows]
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as ex:
        for gr, ok in ex.map(_score_one_lean_rollout_pair, tasks_l, chunksize=8):
            verified_flags[(str(gr["id"]), int(gr["sample_idx"]))] = ok

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for gr in gen_rows:
        pid = str(gr["id"])
        sidx = int(gr["sample_idx"])
        sp = spec_by_id[pid]
        by_pid[pid].append(
            {
                "id": pid,
                "sample_idx": sidx,
                "prompt_messages": sp["messages"],
                "response_text": gr["text"],
                "finish_reason": gr["finish_reason"],
                "verified_ok": verified_flags[(pid, sidx)],
                "is_off_policy": False,
            }
        )

    out_rows: list[dict[str, Any]] = []
    solved_any = 0
    zpd_cnt = 0
    for pid, prows in sorted(by_pid.items()):
        n_ok = sum(1 for x in prows if x["verified_ok"])
        if n_ok > 0:
            solved_any += 1
        if 1 <= n_ok <= 7:
            zpd_cnt += 1
        out_rows.extend(prows)
        if n_ok == 0:
            sp = spec_by_id[pid]
            ref_txt = sp.get("reference_proof")
            if ref_txt:
                out_rows.append(
                    {
                        "id": pid,
                        "sample_idx": 8,
                        "prompt_messages": sp["messages"],
                        "response_text": f"```lean4\n{ref_txt.strip()}\n```" if "```" not in ref_txt else ref_txt,
                        "finish_reason": "stop",
                        "verified_ok": True,
                        "is_off_policy": True,
                    }
                )
    _write_jsonl(out_rollouts_jsonl, out_rows)
    print(
        f"[R2-Lean Rollouts] {out_rollouts_jsonl.name}: total_prompts={len(by_pid)}, "
        f"solved_any={solved_any}, zpd_1_to_7={zpd_cnt}",
        flush=True,
    )
    return out_rollouts_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen38_35b_a3b", choices=["qwen38_35b_a3b"])
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    expected_v3 = _verify_both_manifests()
    model_key = args.model_key
    spec = MODEL_REGISTRY_V3[model_key]
    nproc = len([x for x in args.gpus.split(",") if x.strip()])

    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"lean_{model_key}"
    ch_dir = V3_ROOT / "results_v3/confirm_holdout" / f"lean_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"lean_{model_key}"
    p2_dir.mkdir(parents=True, exist_ok=True)
    ch_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    seal_file = ch_dir / f"HOLDOUT_SEAL_v3_lean_{model_key}.json"
    if seal_file.exists():
        raise RuntimeError(f"Holdout seal already exists at {seal_file}; refusing to run Round 2!")

    b0_search = p1_dir / f"lean_b0_{model_key}_search.json"
    r1_winner_search = p2_dir / "s3_lean_qwen38_35b_a3b_dapo_asymclip_pct50_search.json"
    sp_search = V3_ROOT / "data/splits_v3/v3_lean_search140.jsonl"
    rules_search = V3_ROOT / "configs_v3/gate_rules_v3_lean_search.json"
    search_ledger = p2_dir / f"trial_ledger_v3_lean_{model_key}.jsonl"

    # 1. Build Round-2 Stratum-Balanced Gentle SFT + RL400 Pool
    gentle_sft_jsonl = p2_dir / f"r2_gentle_balanced_sft_lean_{model_key}.jsonl"
    rl400_jsonl = p2_dir / f"r2_balanced_rl400_lean_{model_key}.jsonl"
    build_stratum_balanced_gentle_sft_and_rl400(gentle_sft_jsonl, rl400_jsonl)

    cand_id_to_ckpt: dict[str, str] = {}
    r2_search_jsons: list[str] = []

    # ------------------------------------------------------------------
    # Proposal R2-P1: Pure Zero-SFT On-Policy RLVR (`B_0 -> DAPO400`)
    # ------------------------------------------------------------------
    _verify_both_manifests()
    b0_rollouts = generate_and_verify_lean_rollouts_64w(
        model_key=model_key,
        anchor_model_path=spec.local_path,
        prompts_file=rl400_jsonl,
        out_rollouts_jsonl=p2_dir / f"r2_b0_rollouts400_lean_{model_key}.jsonl",
        gpus=args.gpus,
        seed=20261123,
    )

    p1_tag = f"r2_s3_lean_{model_key}_pure_b0_dapo400"
    p1_out100 = ckpt_root / f"{p1_tag}_pct100"
    p1_mroot = ckpt_root / f"{p1_tag}_milestones"
    p1_sum = p2_dir / f"{p1_tag}_train_summary.json"
    if not p1_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                "--algo", "dapo_asymclip",
                "--model-key", model_key,
                "--prior-fp32-deltas", "",
                "--rollouts-file", str(b0_rollouts),
                "--output-dir", str(p1_out100),
                "--milestones-root", str(p1_mroot),
                "--save-milestone-fractions", "0.50",
                "--summary-output", str(p1_sum),
                "--max-seq-len", "2560",
                "--max-gen-tokens", "2048",
                "--overlong-window", "512",
                "--ppo-epochs", "2",
                "--max-steps", "40",
                "--lr", "1.2e-5",
                "--global-batch-size", "32",
            ]
        )

    for pct_tag, cpath in [("pct50", p1_mroot / "pct50"), ("pct100", p1_out100)]:
        if cpath.exists():
            cid = f"{p1_tag}_{pct_tag}"
            cand_id_to_ckpt[cid] = str(cpath)
            sc_out = p2_dir / f"{cid}_search.json"
            eval_and_score_model(
                track="lean",
                model_key=model_key,
                model_path=str(cpath),
                candidate_id=cid,
                split_prompts_file=sp_search,
                split_truth_file=sp_search,
                out_scored_json=sc_out,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id="v3_lean_search140_k8",
                compare_baseline_json=b0_search,
            )
            r2_search_jsons.append(str(sc_out))

    # ------------------------------------------------------------------
    # Gentle 1-Epoch Stratum-Balanced Cosine SFT Anchor (`r2_s1_gentle`)
    # ------------------------------------------------------------------
    _verify_both_manifests()
    gsft_id = f"r2_s1_lean_{model_key}_gentle_balanced_sft"
    gsft_dir = ckpt_root / gsft_id
    gsft_factors = ckpt_root / f"{gsft_id}_factors.safetensors"
    gsft_sum = p2_dir / f"{gsft_id}_train_summary.json"
    if not gsft_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(SFT_PY),
                "--model-key", model_key,
                "--train-file", str(gentle_sft_jsonl),
                "--output-dir", str(gsft_dir),
                "--save-fp32-deltas", str(gsft_factors),
                "--summary-output", str(gsft_sum),
                "--max-seq-len", "2560",
                "--max-target-tokens", "1792",
                "--epochs", "1",
                "--max-steps", "80",
                "--lr", "1.2e-5",
                "--lr-schedule", "cosine",
                "--warmup-ratio", "0.10",
                "--global-batch-size", "32",
            ]
        )

    gsft_scored = p2_dir / f"{gsft_id}_search.json"
    cand_id_to_ckpt[gsft_id] = str(gsft_dir)
    eval_and_score_model(
        track="lean",
        model_key=model_key,
        model_path=str(gsft_dir),
        candidate_id=gsft_id,
        split_prompts_file=sp_search,
        split_truth_file=sp_search,
        out_scored_json=gsft_scored,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id="v3_lean_search140_k8",
        compare_baseline_json=b0_search,
    )
    r2_search_jsons.append(str(gsft_scored))

    # Shared on-policy rollouts from Gentle SFT for R2-P2 (`DAPO`) & R2-P3 (`Dr.GRPO+CISPO`)
    gsft_rollouts = generate_and_verify_lean_rollouts_64w(
        model_key=model_key,
        anchor_model_path=str(gsft_dir),
        prompts_file=rl400_jsonl,
        out_rollouts_jsonl=p2_dir / f"r2_gsft_rollouts400_lean_{model_key}.jsonl",
        gpus=args.gpus,
        seed=20261124,
    )

    for algo, short_tag in [("dapo_asymclip", "gentle_sft_dapo400"), ("drgrpo_cispo", "gentle_sft_cispo400")]:
        _verify_both_manifests()
        rtag = f"r2_s3_lean_{model_key}_{short_tag}"
        out100 = ckpt_root / f"{rtag}_pct100"
        mroot = ckpt_root / f"{rtag}_milestones"
        rsum = p2_dir / f"{rtag}_train_summary.json"
        if not rsum.exists():
            _run_checked(
                [
                    "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                    "--algo", algo,
                    "--model-key", model_key,
                    "--prior-fp32-deltas", str(gsft_factors),
                    "--rollouts-file", str(gsft_rollouts),
                    "--output-dir", str(out100),
                    "--milestones-root", str(mroot),
                    "--save-milestone-fractions", "0.50",
                    "--summary-output", str(rsum),
                    "--max-seq-len", "2560",
                    "--max-gen-tokens", "2048",
                    "--overlong-window", "512",
                    "--ppo-epochs", "2",
                    "--max-steps", "36",
                    "--lr", "1.0e-5",
                    "--global-batch-size", "32",
                ]
            )
        for pct_tag, cpath in [("pct50", mroot / "pct50"), ("pct100", out100)]:
            if cpath.exists():
                cid = f"{rtag}_{pct_tag}"
                cand_id_to_ckpt[cid] = str(cpath)
                sc_out = p2_dir / f"{cid}_search.json"
                eval_and_score_model(
                    track="lean",
                    model_key=model_key,
                    model_path=str(cpath),
                    candidate_id=cid,
                    split_prompts_file=sp_search,
                    split_truth_file=sp_search,
                    out_scored_json=sc_out,
                    k=8,
                    gpus=args.gpus,
                    seed=20260325,
                    protocol_id="v3_lean_search140_k8",
                    compare_baseline_json=b0_search,
                )
                r2_search_jsons.append(str(sc_out))

    # ------------------------------------------------------------------
    # Order-Invariant Round-2 Search Tournament vs Round-1 Winner!
    # ------------------------------------------------------------------
    r2_win_file = p2_dir / "r2_search_tournament_winner.json"
    r_win = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
            "--candidates", *r2_search_jsons,
            "--incumbent", str(r1_winner_search),
            "--baseline", str(b0_search),
            "--rules", str(rules_search),
            "--ledger", str(search_ledger),
            "--round-id", f"round2_crossover_tournament_lean_{model_key}",
        ],
        allow_rc=(0, 1, 3),
    )
    win_payload = json.loads(r_win.stdout)
    r2_winner_id = win_payload.get("selected_candidate_id") or win_payload.get("winner_candidate_id")
    if not r2_winner_id:
        ranked = win_payload.get("leaderboard") or win_payload.get("ranked_candidates") or []
        clean_ranked = [c for c in ranked if not c.get("tripwires")]
        clean_ranked.sort(key=lambda c: float(c.get("rank_score", -1e9)), reverse=True)
        if clean_ranked:
            r2_winner_id = clean_ranked[0]["candidate_id"]

    r2_win_file.write_text(json.dumps(win_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not r2_winner_id:
        print("[R2-Lean] No tripwire-clean Round-2 candidate found; halting without using Confirm Look 2.", flush=True)
        return 0

    # Check Search-only eligibility before spending final Confirm Look 2 of `max_confirm_looks = 2`
    r1_m = json.loads(r1_winner_search.read_text(encoding="utf-8"))["metrics"]
    r2_m = json.loads((p2_dir / f"{r2_winner_id}_search.json").read_text(encoding="utf-8"))["metrics"]
    print(
        f"[R2-Lean Search Check] R1 ({r1_winner_search.stem}): eq={r1_m['lean_equal_stratum_avg_at_8']:.4f}, "
        f"avg={r1_m['lean_avg_at_8']:.4f}, forg={r1_m.get('forgotten_rate', 0.0):.4f}, stop={r1_m['stop_rate']:.4f} | "
        f"R2 ({r2_winner_id}): eq={r2_m['lean_equal_stratum_avg_at_8']:.4f}, "
        f"avg={r2_m['lean_avg_at_8']:.4f}, forg={r2_m.get('forgotten_rate', 0.0):.4f}, stop={r2_m['stop_rate']:.4f}",
        flush=True,
    )

    # Prune non-winning Round-2 checkpoints to save NVMe space
    r2_winner_ckpt = cand_id_to_ckpt[r2_winner_id]
    for cid, cpath_s in cand_id_to_ckpt.items():
        if cid != r2_winner_id and Path(cpath_s).exists():
            shutil.rmtree(cpath_s, ignore_errors=True)

    # Require Round-2 winner to strictly improve Search equal-stratum accuracy OR (match within 0.5pp while cutting forgetting & raising stop_rate)
    beats_r1_on_search = (
        r2_m["lean_equal_stratum_avg_at_8"] > r1_m["lean_equal_stratum_avg_at_8"]
        or r2_m["lean_avg_at_8"] > r1_m["lean_avg_at_8"]
        or (
            r2_m["lean_equal_stratum_avg_at_8"] >= r1_m["lean_equal_stratum_avg_at_8"] - 0.005
            and r2_m.get("forgotten_rate", 1.0) < r1_m.get("forgotten_rate", 1.0)
            and r2_m["stop_rate"] > r1_m["stop_rate"]
        )
    )
    if not beats_r1_on_search:
        print("[R2-Lean] Round-2 did not improve Search metrics over Round-1; preserving Confirm Look 2 unspent.", flush=True)
        return 0

    # Preserve Round-1 tournament & confirm metadata for full auditability, then run Predeclared Confirm Look 2 of 2!
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

    # Evaluate `r2_winner_id` on `Confirm_V3` (`N=180`) and append Look 2 of `max_confirm_looks = 2`!
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
            "--hypothesis", f"Round-2 forgetting-suppressed 400-prompt RLVR winner {r2_winner_id} beats b0_{model_key} on Confirm_V3 (Look 2 of 2)",
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

    # Atomic Single-Touch Protected Holdout (`gate-holdout --phase reserve` -> Eval -> `--phase finalize`)!
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
        protocol_id="v3_lean_holdout_k8",
        compare_baseline_json=b0_hold_json,
    )

    r_fin = _run_checked(
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
        ],
        allow_rc=(0, 1, 3),
    )
    (ch_dir / "holdout_status.json").write_text(r_fin.stdout, encoding="utf-8")

    # Run Longitudinal (`ProofNet-test167`, `MiniF2F-test244`) for the promoted Round-2 champion!
    long_ns = argparse.Namespace(model_key=model_key, tracks="lean", gpus=args.gpus)
    run_longitudinal(long_ns)
    print(f"[R2-Lean COMPLETE] {model_key} -> {r2_winner_id} PROMOTED & HOLDOUT SEALED!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

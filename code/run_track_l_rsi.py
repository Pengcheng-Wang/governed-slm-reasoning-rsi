#!/usr/bin/env python3
"""
Autonomous Plane-2 Governed RSI Driver for Track L (Formal Lean 4.24 Theorem Proving on Qwen3.5-4B & 9B).

Executes on `node-0` (`8x A100-80GB` + `96-core Lean 4.24` verifier) with full resume-safety and
strict `hermetic_runner.py` + `governed_loop_guard.py` manifest verification & hash-chained ledger gating:
  - Step 0: 16-row 8-GPU DDP trainer smoke check (`train_slm_rsi.py` + `FP32` LoRA merge parity).
  - Round L1a (`4B` Hybrid Gated-DeltaNet LoRA `r=32`, `lr=1e-4`, `3` epochs from `M_0` on `1,402` live-verified proofs).
  - Round L1b (`4B` `full_ft_lm`, `lr=8e-6`, `2` epochs from `M_0`) — direct empirical comparison vs `L1a`.
  - Round L1_9b (`9B` winning paradigm from `M_0` on `1,402` live-verified proofs).
  - Round L2 (`ReST^EM` Round 2 Frontier Self-Play on `197` unsolved theorems + Lean 4.24 Compiler-Diagnostic
    Repair Trajectory Discovery -> cumulative verified bank -> **reset to `M_0`** on both `4B` and `9B`).
  - Round L3 (Frontier-Curated `DAPO + LUFFY` Clipped Policy-Gradient RLVR on active frontier theorems on `4B` & `9B`).
  - Plane 2B (`L4` Call-Matched `4 direct + 4 compiler-repair` vs `8 direct` on frozen winner weights on `lean_repair_reserve97`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/governed_slm_rsi_v1")
MANIFEST = ROOT / "IMMUTABLE_MANIFEST.json"
ROOT_HASH_FILE = Path("/scratch/governed_slm_rsi_v1_trust_root.sha256")
LEDGER = ROOT / "results/plane2/track_l_ledger.jsonl"
RULES_SEARCH = ROOT / "configs/gate_rules_lean_search.json"
RULES_REPAIR = ROOT / "configs/gate_rules_lean_repair_reserve.json"


def _run_hermetic(
    cand_id: str,
    work_dir: Path,
    cmd: str,
    incumbent_eval: Path | None = None,
    challenger_eval: Path | None = None,
    baseline_eval: Path | None = None,
    rules: Path = RULES_SEARCH,
) -> dict | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    gate_out = work_dir / f"{cand_id}_gate.json"
    if challenger_eval and challenger_eval.is_file() and gate_out.is_file():
        return json.loads(gate_out.read_text(encoding="utf-8"))

    hcmd = [
        sys.executable, "-I", "-B", str(ROOT / "code/hermetic_runner.py"),
        "--candidate-id", cand_id,
        "--manifest", str(MANIFEST),
        "--root-hash-file", str(ROOT_HASH_FILE),
        "--work-dir", str(work_dir),
        "--cmd", cmd,
        "--ledger", str(LEDGER),
    ]
    if incumbent_eval and challenger_eval:
        hcmd.extend(
            [
                "--incumbent-eval", str(incumbent_eval),
                "--baseline-eval", str(baseline_eval or incumbent_eval),
                "--challenger-eval", str(challenger_eval),
                "--rules", str(rules),
                "--gate-out", str(gate_out),
            ]
        )
    subprocess.run(hcmd, check=True)
    return json.loads(gate_out.read_text(encoding="utf-8")) if gate_out.is_file() else None


def _promoted(gate: dict | None) -> bool:
    return bool(gate and str(gate.get("decision", "")) in ("PROMOTE", "PROVISIONAL_KEEP"))


def _eval_lean_cmd(model_dir: Path, out_prefix: Path, cand_id: str, split_file: Path | None = None) -> str:
    sp = split_file or (ROOT / "data/splits/lean_search128.jsonl")
    return (
        f"( [ -s {out_prefix}.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
        f"--model-path {model_dir} --split-file {sp} --output {out_prefix}.jsonl.gz "
        f"--summary-output {out_prefix}_gen.json --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 "
        f"--temperature 0.8 --max-tokens 2048 --seed 20261118 ) && "
        f"( [ -s {out_prefix}_metrics.json ] || python -I -B {ROOT}/eval/isolated_scorer.py "
        f"--track lean --split-file {sp} --generations {out_prefix}.jsonl.gz "
        f"--expected-k 8 --candidate-id {cand_id} --protocol-id plane2_lean_search "
        f"--workers 36 --output {out_prefix}_metrics.json --scored-details-output {out_prefix}_details.jsonl.gz )"
    )


def main() -> int:
    p2 = ROOT / "results/plane2/track_l"
    p2.mkdir(parents=True, exist_ok=True)
    seed_bank = ROOT / "data/raw/lean_seed_bank_d0.jsonl"
    lean_pool = ROOT / "data/raw/lean_frontier_pool.jsonl"
    if not seed_bank.is_file():
        raise SystemExit(f"Missing {seed_bank}")

    # Step 0: 16-row 8-GPU DDP smoke check
    smoke_dir = p2 / "smoke_ddp"
    if not (smoke_dir / "smoke_summary.json").is_file():
        smoke_cmd = (
            f"mkdir -p {smoke_dir} && head -n 16 {seed_bank} > {smoke_dir}/smoke16.jsonl && "
            f"torchrun --nproc_per_node=8 --master_port=29501 {ROOT}/code/train_slm_rsi.py "
            f"--base-model {ROOT}/models/qwen35_4b --train-file {smoke_dir}/smoke16.jsonl "
            f"--output-dir {smoke_dir}/merged_ckpt --summary-output {smoke_dir}/smoke_summary.json "
            f"--mode lora --max-seq-len 2048 --epochs 1 --lr 1e-4 --lora-r 16 --lora-alpha 32 "
            f"--global-batch-size 16 && rm -rf {smoke_dir}/merged_ckpt"
        )
        _run_hermetic("track_l_smoke_ddp", smoke_dir, smoke_cmd)

    # Track 4B & 9B incumbents (starting at Plane-1 B_0 direct)
    b0_4b_eval = ROOT / "results/plane1/lean/q4b_direct_b0_metrics.json"
    b0_9b_eval = ROOT / "results/plane1/lean/q9b_direct_b0_metrics.json"
    inc_4b_eval = b0_4b_eval
    inc_4b_ckpt = ROOT / "models/qwen35_4b"
    inc_9b_eval = b0_9b_eval
    inc_9b_ckpt = ROOT / "models/qwen35_9b"

    # Round L1a: 4B Hybrid Gated-DeltaNet LoRA (r=32, alpha=64, lr=1.2e-4, 3 epochs from M_0)
    l1a_dir = p2 / "l1a_q4b_hybrid_lora"
    l1a_ckpt = ROOT / "checkpoints/track_l/l1a_q4b_hybrid_lora"
    l1a_eval = l1a_dir / "eval_metrics.json"
    l1a_cmd = (
        f"( [ -s {l1a_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port=29502 {ROOT}/code/train_slm_rsi.py "
        f"--base-model {ROOT}/models/qwen35_4b --train-file {seed_bank} "
        f"--output-dir {l1a_ckpt} --summary-output {l1a_dir}/train_summary.json "
        f"--mode lora --max-seq-len 4096 --epochs 3 --lr 1.2e-4 --lora-r 32 --lora-alpha 64 --global-batch-size 32 ) && "
        + _eval_lean_cmd(l1a_ckpt, l1a_dir / "eval", "l1a_q4b_hybrid_lora")
    )
    g_l1a = _run_hermetic("l1a_q4b_hybrid_lora", l1a_dir, l1a_cmd, inc_4b_eval, l1a_eval, b0_4b_eval)
    if _promoted(g_l1a):
        inc_4b_eval, inc_4b_ckpt = l1a_eval, l1a_ckpt

    # Round L1b: 4B Full Language-Model Fine-Tuning (lr=8e-6, 2 epochs from M_0) — direct architecture ablation!
    l1b_dir = p2 / "l1b_q4b_full_ft_lm"
    l1b_ckpt = ROOT / "checkpoints/track_l/l1b_q4b_full_ft_lm"
    l1b_eval = l1b_dir / "eval_metrics.json"
    l1b_cmd = (
        f"( [ -s {l1b_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port=29503 {ROOT}/code/train_slm_rsi.py "
        f"--base-model {ROOT}/models/qwen35_4b --train-file {seed_bank} "
        f"--output-dir {l1b_ckpt} --summary-output {l1b_dir}/train_summary.json "
        f"--mode full_ft_lm --max-seq-len 4096 --epochs 2 --lr 8e-6 --global-batch-size 32 ) && "
        + _eval_lean_cmd(l1b_ckpt, l1b_dir / "eval", "l1b_q4b_full_ft_lm")
    )
    g_l1b = _run_hermetic("l1b_q4b_full_ft_lm", l1b_dir, l1b_cmd, inc_4b_eval, l1b_eval, b0_4b_eval)
    best_mode = "lora"
    if _promoted(g_l1b):
        inc_4b_eval, inc_4b_ckpt = l1b_eval, l1b_ckpt
        best_mode = "full_ft_lm"

    # Round L1_9b: Qwen3.5-9B Round 1 ReST^EM from M_0 using winning paradigm
    l1_9b_dir = p2 / "l1_q9b_restem"
    l1_9b_ckpt = ROOT / "checkpoints/track_l/l1_q9b_restem"
    l1_9b_eval = l1_9b_dir / "eval_metrics.json"
    mode_args_9b = (
        "--mode lora --epochs 3 --lr 1.0e-4 --lora-r 32 --lora-alpha 64"
        if best_mode == "lora"
        else "--mode full_ft_lm --epochs 2 --lr 6e-6"
    )
    l1_9b_cmd = (
        f"( [ -s {l1_9b_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port=29504 {ROOT}/code/train_slm_rsi.py "
        f"--base-model {ROOT}/models/qwen35_9b --train-file {seed_bank} "
        f"--output-dir {l1_9b_ckpt} --summary-output {l1_9b_dir}/train_summary.json "
        f"{mode_args_9b} --max-seq-len 4096 --global-batch-size 32 ) && "
        + _eval_lean_cmd(l1_9b_ckpt, l1_9b_dir / "eval", "l1_q9b_restem")
    )
    g_l1_9b = _run_hermetic("l1_q9b_restem", l1_9b_dir, l1_9b_cmd, inc_9b_eval, l1_9b_eval, b0_9b_eval)
    if _promoted(g_l1_9b):
        inc_9b_eval, inc_9b_ckpt = l1_9b_eval, l1_9b_ckpt

    # Round L2: Frontier Self-Play on Unsolved 197 + Compiler-Diagnostic Repair Pair Synthesis -> Reset to M_0!
    l2_bank_dir = p2 / "l2_frontier_bank"
    l2_bank = ROOT / "data/verified_banks/lean_r2_cumulative_verified.jsonl"
    if not l2_bank.is_file():
        # Subsample unsolved 197 + 250 seed theorems to harvest BOTH new frontier proofs AND compiler-repair trajectories
        prep_frontier_py = (
            "import json, random; "
            f"pool=[json.loads(l) for l in open('{lean_pool}') if l.strip()]; "
            f"seed_ids={{json.loads(l)['problem_id'] for l in open('{seed_bank}') if l.strip()}}; "
            "unsolved=[r for r in pool if r['id'] not in seed_ids]; "
            "solved=[r for r in pool if r['id'] in seed_ids]; "
            "random.Random(20261118).shuffle(solved); "
            "sub=unsolved + solved[:220]; "
            f"open('{l2_bank_dir}/r2_rollout_prompts.jsonl','w').writelines(json.dumps(x)+'\\n' for x in sub)"
        )
        l2_harvest_cmd = (
            f"mkdir -p {l2_bank_dir} {l2_bank.parent} && python -I -B -c \"{prep_frontier_py}\" && "
            f"( [ -s {l2_bank_dir}/r2_direct_gen.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
            f"--model-path {inc_9b_ckpt} --split-file {l2_bank_dir}/r2_rollout_prompts.jsonl "
            f"--output {l2_bank_dir}/r2_direct_gen.jsonl.gz --k 6 --tp 1 --gpus 0,1,2,3,4,5,6,7 "
            f"--temperature 0.85 --max-tokens 2048 --seed 20261120 ) && "
            f"python -I -B {ROOT}/code/build_frontier_bank.py merge-restem "
            f"--track lean --pool {lean_pool} "
            f"--prev-bank {seed_bank} --generations {l2_bank_dir}/r2_direct_gen.jsonl.gz "
            f"--tokenizer-path {ROOT}/models/qwen35_4b --max-seq-len 4096 --max-per-problem 3 "
            f"--round-tag l2_direct --workers 36 --output-bank {l2_bank_dir}/bank_after_direct.jsonl "
            f"--output-repair-prompts {l2_bank_dir}/repair_prompts.jsonl "
            f"--summary-output {l2_bank_dir}/direct_merge_summary.json && "
            f"if [ -s {l2_bank_dir}/repair_prompts.jsonl ]; then "
            f"  python -I -B {ROOT}/code/vllm_eval_engine.py --model-path {inc_9b_ckpt} "
            f"  --split-file {l2_bank_dir}/repair_prompts.jsonl --output {l2_bank_dir}/r2_repair_gen.jsonl.gz "
            f"  --k 4 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.8 --max-tokens 2048 --seed 20261121 && "
            f"  python -I -B {ROOT}/code/build_frontier_bank.py merge-restem "
            f"  --track lean --pool {lean_pool} "
            f"  --prev-bank {l2_bank_dir}/bank_after_direct.jsonl --generations {l2_bank_dir}/r2_repair_gen.jsonl.gz "
            f"  --prompt-override-file {l2_bank_dir}/repair_prompts.jsonl "
            f"  --tokenizer-path {ROOT}/models/qwen35_4b --max-seq-len 4096 --max-per-problem 2 "
            f"  --round-tag l2_repair --workers 36 --output-bank {l2_bank} "
            f"  --summary-output {l2_bank_dir}/repair_merge_summary.json; "
            f"else cp {l2_bank_dir}/bank_after_direct.jsonl {l2_bank}; fi"
        )
        _run_hermetic("l2_frontier_harvest", l2_bank_dir, l2_harvest_cmd)

    # Train L2_4b and L2_9b fresh from M_0 on cumulative direct + compiler-repair verified bank!
    l2_4b_dir = p2 / "l2_q4b_restem_repair"
    l2_4b_ckpt = ROOT / "checkpoints/track_l/l2_q4b_restem_repair"
    l2_4b_eval = l2_4b_dir / "eval_metrics.json"
    mode_args_4b = (
        "--mode lora --epochs 3 --lr 1.2e-4 --lora-r 32 --lora-alpha 64"
        if best_mode == "lora"
        else "--mode full_ft_lm --epochs 2 --lr 8e-6"
    )
    l2_4b_cmd = (
        f"( [ -s {l2_4b_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port=29505 {ROOT}/code/train_slm_rsi.py "
        f"--base-model {ROOT}/models/qwen35_4b --train-file {l2_bank} "
        f"--output-dir {l2_4b_ckpt} --summary-output {l2_4b_dir}/train_summary.json "
        f"{mode_args_4b} --max-seq-len 4096 --global-batch-size 32 ) && "
        + _eval_lean_cmd(l2_4b_ckpt, l2_4b_dir / "eval", "l2_q4b_restem_repair")
    )
    g_l2_4b = _run_hermetic("l2_q4b_restem_repair", l2_4b_dir, l2_4b_cmd, inc_4b_eval, l2_4b_eval, b0_4b_eval)
    if _promoted(g_l2_4b):
        inc_4b_eval, inc_4b_ckpt = l2_4b_eval, l2_4b_ckpt

    l2_9b_dir = p2 / "l2_q9b_restem_repair"
    l2_9b_ckpt = ROOT / "checkpoints/track_l/l2_q9b_restem_repair"
    l2_9b_eval = l2_9b_dir / "eval_metrics.json"
    l2_9b_cmd = (
        f"( [ -s {l2_9b_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port=29506 {ROOT}/code/train_slm_rsi.py "
        f"--base-model {ROOT}/models/qwen35_9b --train-file {l2_bank} "
        f"--output-dir {l2_9b_ckpt} --summary-output {l2_9b_dir}/train_summary.json "
        f"{mode_args_9b} --max-seq-len 4096 --global-batch-size 32 ) && "
        + _eval_lean_cmd(l2_9b_ckpt, l2_9b_dir / "eval", "l2_q9b_restem_repair")
    )
    g_l2_9b = _run_hermetic("l2_q9b_restem_repair", l2_9b_dir, l2_9b_cmd, inc_9b_eval, l2_9b_eval, b0_9b_eval)
    if _promoted(g_l2_9b):
        inc_9b_eval, inc_9b_ckpt = l2_9b_eval, l2_9b_ckpt

    # Round L3: Frontier-Curated DAPO + LUFFY RLVR on 4B and 9B
    for tag, msize, cur_ckpt, cur_eval, b0_ev, port in [
        ("l3_q4b_dapo_luffy", "4b", inc_4b_ckpt, inc_4b_eval, b0_4b_eval, 29507),
        ("l3_q9b_dapo_luffy", "9b", inc_9b_ckpt, inc_9b_eval, b0_9b_eval, 29508),
    ]:
        l3_dir = p2 / tag
        l3_ckpt = ROOT / f"checkpoints/track_l/{tag}"
        l3_eval = l3_dir / "eval_metrics.json"
        rl_prompts_py = (
            "import json, random; "
            f"pool=[json.loads(l) for l in open('{lean_pool}') if l.strip()]; "
            "random.Random(20261122).shuffle(pool); "
            f"open('{l3_dir}/rl_frontier192.jsonl','w').writelines(json.dumps(x)+'\\n' for x in pool[:192])"
        )
        l3_cmd = (
            f"mkdir -p {l3_dir} && python -I -B -c \"{rl_prompts_py}\" && "
            f"( [ -s {l3_dir}/rl_gen.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
            f"--model-path {cur_ckpt} --split-file {l3_dir}/rl_frontier192.jsonl "
            f"--output {l3_dir}/rl_gen.jsonl.gz --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 "
            f"--temperature 0.85 --max-tokens 2048 --seed 20261122 ) && "
            f"( [ -s {l3_dir}/rl_groups.jsonl ] || python -I -B {ROOT}/code/build_frontier_bank.py build-rl-groups "
            f"--track lean --pool {lean_pool} "
            f"--generations {l3_dir}/rl_gen.jsonl.gz --verified-bank {l2_bank} "
            f"--workers 36 --output {l3_dir}/rl_groups.jsonl --summary-output {l3_dir}/rl_groups_summary.json ) && "
            f"( [ -s {l3_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port} {ROOT}/code/train_dapo_luffy.py "
            f"--init-model {cur_ckpt} --rollouts-file {l3_dir}/rl_groups.jsonl "
            f"--output-dir {l3_ckpt} --summary-output {l3_dir}/train_summary.json "
            f"--max-seq-len 3072 --max-gen-tokens 2048 --overlong-window 384 --overlong-penalty 0.3 "
            f"--ppo-epochs 1 --lr 2.5e-5 --lora-r 32 --lora-alpha 64 --global-batch-size 32 ) && "
            + _eval_lean_cmd(l3_ckpt, l3_dir / "eval", tag)
        )
        g_l3 = _run_hermetic(tag, l3_dir, l3_cmd, cur_eval, l3_eval, b0_ev)
        if _promoted(g_l3):
            if msize == "4b":
                inc_4b_eval, inc_4b_ckpt = l3_eval, l3_ckpt
            else:
                inc_9b_eval, inc_9b_ckpt = l3_eval, l3_ckpt

    # Plane 2B (L4): Call-Matched Outer Compiler-Repair Policy (4 direct + 4 repair <= 8 calls) on lean_repair_reserve97!
    reserve_split = ROOT / "data/splits/lean_repair_reserve97.jsonl"
    for msize, win_ckpt in [("4b", inc_4b_ckpt), ("9b", inc_9b_ckpt)]:
        l4_dir = p2 / f"l4_q{msize}_outer_repair_reserve97"
        direct8_eval = l4_dir / "direct8_metrics.json"
        repair8_eval = l4_dir / "repair4plus4_metrics.json"
        # Convert isolated_scorer `lean_pass_at_8` per-item binary into `any_verified_solve_at_8` item values for paired comparison
        conv_py = (
            "import json; "
            f"d=json.load(open('{l4_dir}/direct8_raw_metrics.json')); "
            "items=[{'id':x['id'],'cluster_id':x['cluster_id'],'value':1.0 if x['value']>0.0 else 0.0} for x in d['items']]; "
            "d['items']=items; d['metrics']['any_verified_solve_at_8']=d['metrics']['lean_pass_at_8']; "
            "d['protocol_id']='plane2b_frozen_weights_repair_policy'; "
            f"json.dump(d,open('{direct8_eval}','w'),indent=2)"
        )
        l4_cmd = (
            f"mkdir -p {l4_dir} && "
            + _eval_lean_cmd(win_ckpt, l4_dir / "direct8_raw", f"q{msize}_reserve97_direct8", reserve_split)
            + f" && python -I -B -c \"{conv_py}\" && "
            f"( [ -s {repair8_eval} ] || python -I -B {ROOT}/code/repair_policy_eval.py "
            f"--model-path {win_ckpt} --split-file {reserve_split} "
            f"--candidate-id q{msize}_reserve97_repair4p4 --work-dir {l4_dir}/repair_work "
            f"--workers 36 --output {repair8_eval} )"
        )
        _run_hermetic(f"l4_q{msize}_repair_vs_direct8", l4_dir, l4_cmd, direct8_eval, repair8_eval, direct8_eval, RULES_REPAIR)

    winners = {
        "track": "lean",
        "winner_4b_ckpt": str(inc_4b_ckpt),
        "winner_4b_search_eval": str(inc_4b_eval),
        "winner_9b_ckpt": str(inc_9b_ckpt),
        "winner_9b_search_eval": str(inc_9b_eval),
    }
    (p2 / "track_l_winners.json").write_text(json.dumps(winners, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "TRACK_L_RSI_COMPLETE", **winners}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

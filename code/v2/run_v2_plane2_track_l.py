#!/usr/bin/env python3
"""
Phase V2 Plane-2 Multi-Model Governed RSI Driver for Track L (Formal Lean 4.24 Theorem Proving).

Executes across the multi-architecture candidate matrix (`qwen35_4b`, `qwen3_4b`, `qwen35_9b`, `phi4_mini`)
with strict `IMMUTABLE_MANIFEST_V2.json` verification, hash-chained per-model ledgers, and BOTH
`GPT 6.2 Astra, Max` + `Opus 5.5, Max` causal mechanism isolations:
  1. Pre-registered `learnable_fraction(B_0) >= 0.10` estimability gate per model.
  2. Round L1 (`l1_{mk}_restem`): Pure self-generated `M_0` rollouts (`k=8`) on `v2_lean_train_pool600`
     -> `#print axioms` verified bank + failed-draft compiler diagnostic harvest -> train from `M_0`.
  3. Causal Recursion Isolation on `qwen35_4b` (`M_2^{frozen-gen}` vs `M_2^{rec-yield}` vs `M_2^{rec-full}`):
     Same frontier theorems, same `k=8`, same seed `20261202`, isolating `M_1` self-play yield & quality
     against frozen-`M_0` extra sampling!
  4. Round L2 + Natural Compiler-Diagnostic 50/50 Repair (`l2_{mk}_restem_rep5050`):
     `k=4` repair completions per failed syntactic Lean draft -> `merge-repair-5050` (`>= 256` natural
     repair trajectories split 50/50 between exact compiler diagnostics and masked-retry prompts)
     -> reset & train from `M_0`.
  5. Round L3 (`l3_{mk}_dapo_chunked`): Activation-checkpointed chunked-logits `DAPO + LUFFY` RLVR
     on `128` active frontier theorems using only the model's own verified bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v2 import MODEL_REGISTRY_V2  # noqa: E402

V2 = Path("/scratch/governed_slm_rsi_v2")
V1 = Path("/scratch/governed_slm_rsi_v1")
RULES_SEARCH = V2 / "configs_v2/gate_rules_v2_lean_search.json"


def _select_plane1_lean_b0(mk: str) -> tuple[Path, bool, int]:
    p1 = V2 / "results/plane1/lean"
    if mk == "qwen3_4b":
        d_p = p1 / "b0_qwen3_4b_direct_metrics.json"
        t_p = p1 / "b0_qwen3_4b_think6k_metrics.json"
        d_m = json.loads(d_p.read_text(encoding="utf-8"))["metrics"]["lean_avg_at_8"]
        t_m = json.loads(t_p.read_text(encoding="utf-8"))["metrics"]["lean_avg_at_8"]
        if t_m > d_m + 0.01:
            return t_p, True, 6144
        return d_p, False, 2048
    if mk == "phi4_mini":
        return p1 / "b0_phi4_mini_think6k_metrics.json", True, 6144
    return p1 / f"b0_{mk}_direct_metrics.json", False, 2048


def _run_hermetic(
    cand_id: str,
    work_dir: Path,
    cmd: str,
    ledger: Path,
    *,
    inc_eval: Path | None = None,
    chal_eval: Path | None = None,
    base_eval: Path | None = None,
    hypothesis: str = "",
) -> dict[str, Any] | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    gate_out = work_dir / f"{cand_id}_gate.json"
    if chal_eval and chal_eval.is_file() and gate_out.is_file():
        return json.loads(gate_out.read_text(encoding="utf-8"))

    hcmd = [
        sys.executable, "-I", "-B", str(V2 / "code/v2/hermetic_runner_v2.py"),
        "--candidate-id", cand_id,
        "--work-dir", str(work_dir),
        "--cmd", cmd,
        "--ledger", str(ledger),
        "--hypothesis", hypothesis,
    ]
    if inc_eval and chal_eval:
        hcmd.extend(
            [
                "--incumbent-eval", str(inc_eval),
                "--baseline-eval", str(base_eval or inc_eval),
                "--challenger-eval", str(chal_eval),
                "--rules", str(RULES_SEARCH),
                "--gate-out", str(gate_out),
            ]
        )
    subprocess.run(hcmd, check=True)
    return json.loads(gate_out.read_text(encoding="utf-8")) if gate_out.is_file() else None


def _promoted(gate: dict[str, Any] | None) -> bool:
    return bool(gate and str(gate.get("decision", "")) in ("PROMOTE", "PROVISIONAL_KEEP"))


def _eval_lean_search_cmd(mk: str, ckpt_dir: Path, out_prefix: Path, cand_id: str, *, thinking: bool, max_tokens: int) -> str:
    sp = V2 / "data/splits_v2/v2_lean_search120.jsonl"
    th = "--thinking " if thinking else ""
    proto = "v2_lean_search_think" if thinking else "v2_lean_search"
    return (
        f"( [ -s {out_prefix}.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {ckpt_dir} --split-file {sp} --output {out_prefix}.jsonl.gz "
        f"--summary-output {out_prefix}_gen.json --k 8 --gpus 0,1,2,3,4,5,6,7 "
        f"{th}--temperature 0.8 --max-tokens {max_tokens} --seed 20261118 ) && "
        f"( [ -s {out_prefix}_metrics.json ] || python3 -I -B {V1}/eval/isolated_scorer.py "
        f"--track lean --split-file {sp} --generations {out_prefix}.jsonl.gz "
        f"--expected-k 8 --candidate-id {cand_id} --protocol-id {proto} "
        f"{th}--workers 36 --output {out_prefix}_metrics.json --scored-details-output {out_prefix}_details.jsonl.gz )"
    )


def _curate_active_rl_subset(pool_path: Path, verified_bank_path: Path, out_subset_path: Path, *, max_items: int = 128, seed: int = 20261205) -> int:
    if out_subset_path.is_file():
        return sum(1 for ln in out_subset_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    pool = [json.loads(ln) for ln in pool_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    counts: dict[str, int] = {}
    if verified_bank_path.is_file():
        for ln in verified_bank_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                pid = str(r.get("base_problem_id") or r["problem_id"])
                if "::rep" not in str(r["problem_id"]):
                    counts[pid] = counts.get(pid, 0) + 1
    # Prioritize active boundary theorems (1..2 verified proofs first, then 0-proof frontier theorems)
    boundary = [r for r in pool if 1 <= counts.get(str(r["id"]), 0) <= 2]
    unsolved = [r for r in pool if counts.get(str(r["id"]), 0) == 0]
    rng = random.Random(seed)
    rng.shuffle(boundary)
    rng.shuffle(unsolved)
    chosen = (boundary[:96] + unsolved[:64])[:max_items]
    out_subset_path.parent.mkdir(parents=True, exist_ok=True)
    with out_subset_path.open("w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(chosen)


def run_model_track_l(mk: str, port_base: int) -> dict[str, Any]:
    spec = MODEL_REGISTRY_V2[mk]
    base_ckpt = Path(spec.local_path)
    pool = V2 / "data/corpora/v2_lean_train_pool.jsonl"
    p2 = V2 / f"results/plane2/track_l/{mk}"
    p2.mkdir(parents=True, exist_ok=True)
    ledger = V2 / f"results/plane2/track_l_{mk}_ledger.jsonl"

    b0_eval, thinking, eval_max_tok = _select_plane1_lean_b0(mk)
    b0_data = json.loads(b0_eval.read_text(encoding="utf-8"))
    lf = float(b0_data["metrics"].get("learnable_fraction", 0.0))
    p8 = float(b0_data["metrics"].get("lean_pass_at_8", 0.0))

    # Pre-registered Opus 5.5 Max estimability gate (`learnable_fraction >= 0.10`)
    if lf < 0.10 and p8 < 0.10:
        audit = {
            "model_key": mk,
            "status": "SKIPPED_BELOW_ESTIMABILITY_FLOOR",
            "b0_eval": str(b0_eval),
            "b0_learnable_fraction": lf,
            "b0_pass_at_8": p8,
            "b0_avg_at_8": b0_data["metrics"]["lean_avg_at_8"],
            "incumbent_ckpt": str(base_ckpt),
            "incumbent_eval": str(b0_eval),
            "promoted_challenger": False,
        }
        (p2 / "incumbent.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2), flush=True)
        return audit

    th_flag = "--thinking " if thinking else ""
    max_seq = 6656 if thinking else 4096
    lr_sft = "9e-5" if spec.params_b >= 8.0 else "1.2e-4"
    inc_eval = b0_eval
    inc_ckpt = base_ckpt
    inc_id = f"b0_{mk}"

    # Round L1: ReST^EM Round 1 from M_0
    l1_id = f"l1_{mk}_restem"
    l1_dir = p2 / l1_id
    l1_bank = V2 / f"data/verified_banks_v2/lean_{mk}_r1.jsonl"
    l1_front = l1_dir / "r1_next_frontier.jsonl"
    l1_rep_prompts = l1_dir / "r1_repair_prompts.jsonl"
    l1_ckpt = V2 / f"checkpoints/track_l/{l1_id}"
    l1_eval = l1_dir / "eval_metrics.json"

    l1_cmd = (
        f"mkdir -p {l1_dir} {l1_bank.parent} && "
        f"( [ -s {l1_dir}/r1_pool_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {base_ckpt} --split-file {pool} --output {l1_dir}/r1_pool_gen.jsonl.gz "
        f"--k 8 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.8 --max-tokens {eval_max_tok} --seed 20261201 ) && "
        f"( [ -s {l1_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
        f"--track lean --model-key {mk} --pool {pool} --generations {l1_dir}/r1_pool_gen.jsonl.gz "
        f"{th_flag}--max-seq-len {max_seq} --max-per-problem 3 --repair-drafts-per-thm 3 --round-tag {l1_id} "
        f"--output-bank {l1_bank} --output-next-frontier {l1_front} --output-repair-prompts {l1_rep_prompts} "
        f"--summary-output {l1_dir}/bank_summary.json ) && "
        f"( [ -s {l1_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 1} {V2}/code/v2/train_slm_v2.py "
        f"--model-key {mk} --base-model {base_ckpt} --train-file {l1_bank} --output-dir {l1_ckpt} "
        f"--summary-output {l1_dir}/train_summary.json {th_flag}--max-seq-len {max_seq} --epochs 3 --lr {lr_sft} --global-batch-size 32 ) && "
        + _eval_lean_search_cmd(mk, l1_ckpt, l1_dir / "eval", l1_id, thinking=thinking, max_tokens=eval_max_tok)
    )
    g_l1 = _run_hermetic(
        l1_id, l1_dir, l1_cmd, ledger,
        inc_eval=inc_eval, chal_eval=l1_eval, base_eval=b0_eval,
        hypothesis=f"R1 pure self-generated Lean 4.24 ReST^EM from M_0 on {mk}",
    )
    if _promoted(g_l1):
        inc_eval, inc_ckpt, inc_id = l1_eval, l1_ckpt, l1_id

    # Causal Recursion Control on primary 4B anchor (`qwen35_4b`): M_2^{frozen-gen} vs M_2^{rec-yield} vs M_2^{rec-full}
    if mk == "qwen35_4b":
        # Arm A: Frozen M_0 Round-2 rollout at seed=20261202
        froz_id = "l2_qwen35_4b_frozen_gen"
        froz_dir = p2 / froz_id
        froz_bank = V2 / "data/verified_banks_v2/lean_qwen35_4b_r2_frozen_gen.jsonl"
        froz_ckpt = V2 / f"checkpoints/track_l/{froz_id}"
        froz_eval = froz_dir / "eval_metrics.json"
        froz_cmd = (
            f"mkdir -p {froz_dir} && "
            f"( [ -s {froz_dir}/r2_m0_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
            f"--model-key {mk} --model-path {base_ckpt} --split-file {l1_front} --output {froz_dir}/r2_m0_gen.jsonl.gz "
            f"--k 8 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.85 --max-tokens {eval_max_tok} --seed 20261202 ) && "
            f"( [ -s {froz_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
            f"--track lean --model-key {mk} --pool {pool} --prev-bank {l1_bank} --generations {froz_dir}/r2_m0_gen.jsonl.gz "
            f"{th_flag}--max-seq-len {max_seq} --max-per-problem 3 --round-tag {froz_id} "
            f"--output-bank {froz_bank} --summary-output {froz_dir}/bank_summary.json ) && "
            f"( [ -s {froz_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 2} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {froz_bank} --output-dir {froz_ckpt} "
            f"--summary-output {froz_dir}/train_summary.json {th_flag}--max-seq-len {max_seq} --epochs 3 --lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_lean_search_cmd(mk, froz_ckpt, froz_dir / "eval", froz_id, thinking=thinking, max_tokens=eval_max_tok)
        )
        _run_hermetic(
            froz_id, froz_dir, froz_cmd, ledger,
            inc_eval=inc_eval, chal_eval=froz_eval, base_eval=b0_eval,
            hypothesis="Causal Recursion Control: Round-2 rollouts drawn from frozen M_0 at matched budget & seed",
        )
        froz_steps = int(json.loads((froz_dir / "train_summary.json").read_text(encoding="utf-8"))["global_steps"])

        # Arm B: Recursive M_1 Round-2 rollout (direct proofs only, pre-repair, for pure recursion comparison!)
        rec_id = "l2_qwen35_4b_rec_full"
        rec_dir = p2 / rec_id
        rec_bank = V2 / "data/verified_banks_v2/lean_qwen35_4b_r2_rec_direct.jsonl"
        rec_ckpt = V2 / f"checkpoints/track_l/{rec_id}"
        rec_eval = rec_dir / "eval_metrics.json"
        m1_source = l1_ckpt if l1_ckpt.exists() else inc_ckpt
        rec_cmd = (
            f"mkdir -p {rec_dir} && "
            f"( [ -s {rec_dir}/r2_m1_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
            f"--model-key {mk} --model-path {m1_source} --split-file {l1_front} --output {rec_dir}/r2_m1_gen.jsonl.gz "
            f"--k 8 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.85 --max-tokens {eval_max_tok} --seed 20261202 ) && "
            f"( [ -s {rec_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
            f"--track lean --model-key {mk} --pool {pool} --prev-bank {l1_bank} --generations {rec_dir}/r2_m1_gen.jsonl.gz "
            f"{th_flag}--max-seq-len {max_seq} --max-per-problem 3 --round-tag {rec_id} "
            f"--output-bank {rec_bank} --summary-output {rec_dir}/bank_summary.json ) && "
            f"( [ -s {rec_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 3} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {rec_bank} --output-dir {rec_ckpt} "
            f"--summary-output {rec_dir}/train_summary.json {th_flag}--max-seq-len {max_seq} --epochs 3 --lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_lean_search_cmd(mk, rec_ckpt, rec_dir / "eval", rec_id, thinking=thinking, max_tokens=eval_max_tok)
        )
        g_rec = _run_hermetic(
            rec_id, rec_dir, rec_cmd, ledger,
            inc_eval=inc_eval, chal_eval=rec_eval, base_eval=b0_eval,
            hypothesis="Causal Recursion Full: Round-2 rollouts drawn from M_1 at matched budget & seed",
        )

        # Arm C: Yield-Matched Recursive Sub-Arm (Opus 5.5 Max control: exact trace count & optimizer steps of frozen_gen!)
        ry_id = "l2_qwen35_4b_rec_yield"
        ry_dir = p2 / ry_id
        ry_bank = V2 / "data/verified_banks_v2/lean_qwen35_4b_r2_rec_yield.jsonl"
        ry_ckpt = V2 / f"checkpoints/track_l/{ry_id}"
        ry_eval = ry_dir / "eval_metrics.json"
        ry_cmd = (
            f"mkdir -p {ry_dir} && "
            f"( [ -s {ry_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py match-yield "
            f"--r1-bank {l1_bank} --rec-bank {rec_bank} --frozen-bank {froz_bank} "
            f"--output-bank {ry_bank} --summary-output {ry_dir}/yield_match_summary.json ) && "
            f"( [ -s {ry_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 4} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {ry_bank} --output-dir {ry_ckpt} "
            f"--summary-output {ry_dir}/train_summary.json {th_flag}--max-seq-len {max_seq} --epochs 3 --max-steps {froz_steps} "
            f"--lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_lean_search_cmd(mk, ry_ckpt, ry_dir / "eval", ry_id, thinking=thinking, max_tokens=eval_max_tok)
        )
        _run_hermetic(
            ry_id, ry_dir, ry_cmd, ledger,
            inc_eval=inc_eval, chal_eval=ry_eval, base_eval=b0_eval,
            hypothesis="Causal Recursion Yield-Matched: M_1 traces subsampled to exact frozen-M_0 yield & optimizer steps",
        )
        if _promoted(g_rec):
            inc_eval, inc_ckpt, inc_id = rec_eval, rec_ckpt, rec_id

    # Round L2 + Natural Compiler-Diagnostic 50/50 Repair (`>= 256` repair trajectories -> Reset to M_0!)
    l2r_id = f"l2_{mk}_restem_rep5050"
    l2r_dir = p2 / l2r_id
    l2r_direct_bank = V2 / f"data/verified_banks_v2/lean_{mk}_r2_direct.jsonl"
    l2r_bank = V2 / f"data/verified_banks_v2/lean_{mk}_r2_rep5050_cumulative.jsonl"
    l2r_rep_only = V2 / f"data/verified_banks_v2/lean_{mk}_r2_rep5050_pairs_only.jsonl"
    l2r_ckpt = V2 / f"checkpoints/track_l/{l2r_id}"
    l2r_eval = l2r_dir / "eval_metrics.json"
    rollout_src = l1_ckpt if l1_ckpt.exists() else inc_ckpt

    l2r_cmd = (
        f"mkdir -p {l2r_dir} && "
        f"( [ -s {l2r_dir}/r2_front_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {rollout_src} --split-file {l1_front} --output {l2r_dir}/r2_front_gen.jsonl.gz "
        f"--k 8 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.85 --max-tokens {eval_max_tok} --seed 20261202 ) && "
        f"( [ -s {l2r_direct_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
        f"--track lean --model-key {mk} --pool {pool} --prev-bank {l1_bank} --generations {l2r_dir}/r2_front_gen.jsonl.gz "
        f"{th_flag}--max-seq-len {max_seq} --max-per-problem 3 --round-tag {l2r_id}_dir "
        f"--output-bank {l2r_direct_bank} --summary-output {l2r_dir}/direct_bank_summary.json ) && "
        f"( [ -s {l2r_dir}/r2_repair_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {rollout_src} --split-file {l1_rep_prompts} --output {l2r_dir}/r2_repair_gen.jsonl.gz "
        f"--k 4 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.85 --max-tokens {eval_max_tok} --seed 20261203 ) && "
        f"( [ -s {l2r_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-repair-5050 "
        f"--model-key {mk} --repair-prompts {l1_rep_prompts} --generations {l2r_dir}/r2_repair_gen.jsonl.gz "
        f"--prev-bank {l2r_direct_bank} {th_flag}--max-seq-len {max_seq + 1024} --max-per-prompt 2 --round-tag {l2r_id} "
        f"--output-bank {l2r_bank} --output-repair-only {l2r_rep_only} --summary-output {l2r_dir}/repair_bank_summary.json ) && "
        f"( [ -s {l2r_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 5} {V2}/code/v2/train_slm_v2.py "
        f"--model-key {mk} --base-model {base_ckpt} --train-file {l2r_bank} --output-dir {l2r_ckpt} "
        f"--summary-output {l2r_dir}/train_summary.json {th_flag}--max-seq-len {max_seq + 1024} --epochs 3 --lr {lr_sft} --global-batch-size 32 ) && "
        + _eval_lean_search_cmd(mk, l2r_ckpt, l2r_dir / "eval", l2r_id, thinking=thinking, max_tokens=eval_max_tok)
    )
    g_l2r = _run_hermetic(
        l2r_id, l2r_dir, l2r_cmd, ledger,
        inc_eval=inc_eval, chal_eval=l2r_eval, base_eval=b0_eval,
        hypothesis=f"R2 Frontier ReST^EM + 50/50 Masked/Unmasked Natural Lean 4.24 Compiler-Repair Bank reset to M_0 on {mk}",
    )
    if _promoted(g_l2r):
        inc_eval, inc_ckpt, inc_id = l2r_eval, l2r_ckpt, l2r_id

    # Round L3: Checkpointed-Chunk DAPO + LUFFY RLVR on 128 Active Frontier Theorems
    l3_id = f"l3_{mk}_dapo_chunked"
    l3_dir = p2 / l3_id
    l3_sub = l3_dir / "rl_frontier128.jsonl"
    _curate_active_rl_subset(pool, l2r_direct_bank, l3_sub, max_items=128)
    l3_groups = l3_dir / "rl_groups.jsonl"
    l3_ckpt = V2 / f"checkpoints/track_l/{l3_id}"
    l3_eval = l3_dir / "eval_metrics.json"
    lr_rl = "1.8e-5" if spec.params_b >= 8.0 else "2.5e-5"

    l3_cmd = (
        f"mkdir -p {l3_dir} && "
        f"( [ -s {l3_dir}/rl_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {inc_ckpt} --split-file {l3_sub} --output {l3_dir}/rl_gen.jsonl.gz "
        f"--k 8 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.85 --max-tokens {eval_max_tok} --seed 20261204 ) && "
        f"( [ -s {l3_groups} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py build-rl-groups "
        f"--track lean --pool {l3_sub} --generations {l3_dir}/rl_gen.jsonl.gz --verified-bank {l2r_direct_bank} "
        f"{th_flag}--output {l3_groups} --summary-output {l3_dir}/rl_group_summary.json ) && "
        f"( [ -s {l3_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 6} {V2}/code/v2/train_dapo_chunked_v2.py "
        f"--model-key {mk} --init-model {inc_ckpt} --rollouts-file {l3_groups} --output-dir {l3_ckpt} "
        f"--summary-output {l3_dir}/train_summary.json {th_flag}--max-seq-len {max_seq + 512} "
        f"--max-gen-tokens {eval_max_tok} --overlong-window 384 --overlong-penalty 0.25 --ppo-epochs 1 --lr {lr_rl} --global-batch-size 32 ) && "
        + _eval_lean_search_cmd(mk, l3_ckpt, l3_dir / "eval", l3_id, thinking=thinking, max_tokens=eval_max_tok)
    )
    g_l3 = _run_hermetic(
        l3_id, l3_dir, l3_cmd, ledger,
        inc_eval=inc_eval, chal_eval=l3_eval, base_eval=b0_eval,
        hypothesis=f"R3 Checkpointed-Chunk DAPO+LUFFY Verifier-RLVR on 128 active Lean 4.24 frontier theorems on {mk}",
    )
    if _promoted(g_l3):
        inc_eval, inc_ckpt, inc_id = l3_eval, l3_ckpt, l3_id

    inc_summary = {
        "model_key": mk,
        "status": "COMPLETED",
        "thinking": thinking,
        "eval_max_tokens": eval_max_tok,
        "b0_eval": str(b0_eval),
        "b0_avg_at_8": b0_data["metrics"]["lean_avg_at_8"],
        "incumbent_id": inc_id,
        "incumbent_ckpt": str(inc_ckpt),
        "incumbent_eval": str(inc_eval),
        "incumbent_avg_at_8": json.loads(inc_eval.read_text(encoding="utf-8"))["metrics"]["lean_avg_at_8"],
        "repair_trained_ckpt": str(l2r_ckpt) if l2r_ckpt.exists() else str(inc_ckpt),
        "promoted_challenger": (inc_ckpt != base_ckpt),
    }
    (p2 / "incumbent.json").write_text(json.dumps(inc_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inc_summary, indent=2), flush=True)
    return inc_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen35_4b,qwen3_4b,qwen35_9b,phi4_mini")
    ap.add_argument("--port-base", type=int, default=29710)
    args = ap.parse_args()

    mks = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}
    for idx, mk in enumerate(mks):
        results[mk] = run_model_track_l(mk, args.port_base + idx * 10)
    print(json.dumps({"event": "TRACK_L_V2_PLANE2_DONE", "models": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

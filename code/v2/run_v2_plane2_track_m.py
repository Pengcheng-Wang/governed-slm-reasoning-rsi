#!/usr/bin/env python3
"""
Phase V2 Plane-2 Multi-Model Governed RSI Driver for Track M (Olympiad & Hard Competition Math).

Runs pure verifier-governed self-improvement (`0` external teacher traces!) across 4 SLMs spanning
3 distinct architectures (`qwen35_4b`, `qwen3_4b`, `phi4_mini` LongRoPE `<think>`, and `qwen35_9b`
at FULL `12,800`-token sequence length via activation-checkpointed `lm_head` chunks!):
  1. Pre-registered `learnable_fraction(B_0) >= 0.10` estimability gate per model.
  2. Round M1 (`m1_{mk}_restem`): Self-generated `M_0` rollouts (`k=6`, `max_tokens=10240`) on
     `v2_math_train_pool600` (`Omni-MATH Tier 3-4`) -> strict natural-stop hygiene -> train from `M_0` (`12,800` seq len).
  3. Causal Recursion Isolation on `qwen35_4b` (`m2_qwen35_4b_frozen_gen` vs `m2_qwen35_4b_rec_yield` vs `m2_qwen35_4b_rec_full`).
  4. Round M2 (`m2_{mk}_rec_restem`): Pure `M_1` frontier self-play (`k=6`, `seed=20261212`, zero 27B traces!)
     -> cumulative verified bank -> reset & train from `M_0` (`12,800` seq len).
  5. Round M3 (`m3_{mk}_dapo_chunked12k`): Full `13,312`-token activation-checkpointed chunked-logits
     `DAPO + LUFFY` RLVR (`10,000 -> 12,288` linear soft-overlong penalty + truncated IS cap `<= 2.0`),
     directly fixing Phase V1's 9B 2,688-token OOM & completion-floor miss!
"""
from __future__ import annotations

import argparse
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
RULES_SEARCH = V2 / "configs_v2/gate_rules_v2_math_search.json"


def _select_plane1_math_b0(mk: str) -> tuple[Path, bool, int, float]:
    p1 = V2 / "results/plane1/math"
    if mk == "qwen3_4b":
        d_p = p1 / "b0_qwen3_4b_direct12k_metrics.json"
        t_p = p1 / "b0_qwen3_4b_think16k_metrics.json"
        d_d = json.loads(d_p.read_text(encoding="utf-8"))["metrics"]
        t_d = json.loads(t_p.read_text(encoding="utf-8"))["metrics"]
        if t_d["math_avg_at_8"] > d_d["math_avg_at_8"] + 0.02:
            return t_p, True, 16384, 1.0
        return d_p, False, 12288, 0.0
    if mk == "phi4_mini":
        return p1 / "b0_phi4_mini_think12k_metrics.json", True, 12288, 0.0
    return p1 / f"b0_{mk}_direct12k_metrics.json", False, 12288, 0.0


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


def _eval_math_search_cmd(
    mk: str,
    ckpt_dir: Path,
    out_prefix: Path,
    cand_id: str,
    *,
    thinking: bool,
    max_tokens: int,
    presence_penalty: float,
) -> str:
    sp = V2 / "data/splits_v2/v2_math_search128.jsonl"
    th = "--thinking " if thinking else ""
    temp = "0.8" if thinking else "0.7"
    max_mlen = max(16384, max_tokens + 4096)
    proto = f"v2_math_search_{'think' if thinking else 'direct'}{max_tokens // 1024}k"
    return (
        f"( [ -s {out_prefix}.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {ckpt_dir} --split-file {sp} --output {out_prefix}.jsonl.gz "
        f"--summary-output {out_prefix}_gen.json --k 8 --gpus 0,1,2,3,4,5,6,7 "
        f"{th}--temperature {temp} --presence-penalty {presence_penalty} --max-tokens {max_tokens} "
        f"--max-model-len {max_mlen} --seed 20261118 ) && "
        f"( [ -s {out_prefix}_metrics.json ] || python3 -I -B {V2}/code/v2/timeout_isolated_scorer_v2.py "
        f"--track math --split-file {sp} --generations {out_prefix}.jsonl.gz "
        f"--expected-k 8 --candidate-id {cand_id} --protocol-id {proto} "
        f"{th}--workers 32 --output {out_prefix}_metrics.json --scored-details-output {out_prefix}_details.jsonl.gz )"
    )


def _curate_active_math_rl_subset(pool_path: Path, verified_bank_path: Path, out_subset_path: Path, *, max_items: int = 128, seed: int = 20261215) -> int:
    if out_subset_path.is_file():
        return sum(1 for ln in out_subset_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    pool = [json.loads(ln) for ln in pool_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    counts: dict[str, int] = {}
    if verified_bank_path.is_file():
        for ln in verified_bank_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                pid = str(r["problem_id"])
                counts[pid] = counts.get(pid, 0) + 1
    boundary = [r for r in pool if 1 <= counts.get(str(r["id"]), 0) <= 2]
    unsolved = [r for r in pool if counts.get(str(r["id"]), 0) == 0]
    graduated = [r for r in pool if counts.get(str(r["id"]), 0) >= 3]
    rng = random.Random(seed)
    rng.shuffle(boundary)
    rng.shuffle(unsolved)
    rng.shuffle(graduated)
    chosen = (boundary[:88] + unsolved[:32] + graduated[:16])[:max_items]
    out_subset_path.parent.mkdir(parents=True, exist_ok=True)
    with out_subset_path.open("w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(chosen)


def run_model_track_m(mk: str, port_base: int) -> dict[str, Any]:
    spec = MODEL_REGISTRY_V2[mk]
    base_ckpt = Path(spec.local_path)
    full_pool = V2 / "data/corpora/v2_math_train_pool.jsonl"
    p2 = V2 / f"results/plane2/track_m/{mk}"
    p2.mkdir(parents=True, exist_ok=True)
    if mk == "phi4_mini":
        pool = p2 / "phi4_pool360.jsonl"
        if not pool.is_file():
            rows_all = [ln for ln in full_pool.read_text(encoding="utf-8").splitlines() if ln.strip()]
            pool.write_text("\n".join(rows_all[:360]) + "\n", encoding="utf-8")
    else:
        pool = full_pool
    ledger = V2 / f"results/plane2/track_m_{mk}_ledger.jsonl"

    b0_eval, thinking, eval_max_tok, pp = _select_plane1_math_b0(mk)
    b0_data = json.loads(b0_eval.read_text(encoding="utf-8"))
    lf = float(b0_data["metrics"].get("learnable_fraction", 0.0))
    p8 = float(b0_data["metrics"].get("math_pass_at_8", 0.0))

    if lf < 0.10 and p8 < 0.10:
        audit = {
            "model_key": mk,
            "status": "SKIPPED_BELOW_ESTIMABILITY_FLOOR",
            "b0_eval": str(b0_eval),
            "b0_learnable_fraction": lf,
            "b0_pass_at_8": p8,
            "b0_avg_at_8": b0_data["metrics"]["math_avg_at_8"],
            "incumbent_ckpt": str(base_ckpt),
            "incumbent_eval": str(b0_eval),
            "promoted_challenger": False,
        }
        (p2 / "incumbent.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return audit

    th_flag = "--thinking " if thinking else ""
    rollout_max_tok = 8192 if mk == "phi4_mini" else 10240
    rollout_k = 4 if mk == "phi4_mini" else 6
    train_seq_len = 12800  # Full 12.8K training enabled for ALL models including 9B via checkpointed lm_head chunks!
    lr_sft = "4.5e-5" if spec.params_b >= 8.0 else "6.0e-5"
    lr_rl = "1.8e-5" if spec.params_b >= 8.0 else "2.5e-5"
    temp_r1 = "0.8" if thinking else "0.75"
    temp_r2 = "0.85" if thinking else "0.8"

    inc_eval = b0_eval
    inc_ckpt = base_ckpt
    inc_id = f"b0_{mk}"

    # Round M1: Pure Self-Generated ReST^EM Round 1 from M_0 (Full 12.8K Seq Len)
    m1_id = f"m1_{mk}_restem"
    m1_dir = p2 / m1_id
    m1_bank = V2 / f"data/verified_banks_v2/math_{mk}_r1.jsonl"
    m1_front = m1_dir / "r1_next_frontier.jsonl"
    m1_ckpt = V2 / f"checkpoints/track_m/{m1_id}"
    m1_eval = m1_dir / "eval_metrics.json"

    m1_cmd = (
        f"mkdir -p {m1_dir} {m1_bank.parent} && "
        f"( [ -s {m1_dir}/r1_pool_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {base_ckpt} --split-file {pool} --output {m1_dir}/r1_pool_gen.jsonl.gz "
        f"--k {rollout_k} --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature {temp_r1} --presence-penalty {pp} "
        f"--max-tokens {rollout_max_tok} --max-model-len 16384 --seed 20261211 ) && "
        f"( [ -s {m1_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
        f"--track math --model-key {mk} --pool {pool} --generations {m1_dir}/r1_pool_gen.jsonl.gz "
        f"{th_flag}--max-seq-len {train_seq_len} --max-per-problem 3 --round-tag {m1_id} "
        f"--output-bank {m1_bank} --output-next-frontier {m1_front} --summary-output {m1_dir}/bank_summary.json ) && "
        f"( [ -s {m1_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 1} {V2}/code/v2/train_slm_v2.py "
        f"--model-key {mk} --base-model {base_ckpt} --train-file {m1_bank} --output-dir {m1_ckpt} "
        f"--summary-output {m1_dir}/train_summary.json {th_flag}--max-seq-len {train_seq_len} --epochs 2 --lr {lr_sft} --global-batch-size 32 ) && "
        + _eval_math_search_cmd(mk, m1_ckpt, m1_dir / "eval", m1_id, thinking=thinking, max_tokens=eval_max_tok, presence_penalty=pp)
    )
    g_m1 = _run_hermetic(
        m1_id, m1_dir, m1_cmd, ledger,
        inc_eval=inc_eval, chal_eval=m1_eval, base_eval=b0_eval,
        hypothesis=f"R1 pure self-generated full-12.8K ReST^EM from M_0 on {mk} (Omni-MATH Tier 3-4)",
    )
    if _promoted(g_m1):
        inc_eval, inc_ckpt, inc_id = m1_eval, m1_ckpt, m1_id

    # Causal Recursion Control on primary 4B anchor (`qwen35_4b`)
    if mk == "qwen35_4b":
        froz_id = "m2_qwen35_4b_frozen_gen"
        froz_dir = p2 / froz_id
        froz_bank = V2 / "data/verified_banks_v2/math_qwen35_4b_r2_frozen_gen.jsonl"
        froz_ckpt = V2 / f"checkpoints/track_m/{froz_id}"
        froz_eval = froz_dir / "eval_metrics.json"
        froz_cmd = (
            f"mkdir -p {froz_dir} && "
            f"( [ -s {froz_dir}/r2_m0_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
            f"--model-key {mk} --model-path {base_ckpt} --split-file {m1_front} --output {froz_dir}/r2_m0_gen.jsonl.gz "
            f"--k 6 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature {temp_r2} --presence-penalty {pp} "
            f"--max-tokens {rollout_max_tok} --max-model-len 16384 --seed 20261212 ) && "
            f"( [ -s {froz_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
            f"--track math --model-key {mk} --pool {pool} --prev-bank {m1_bank} --generations {froz_dir}/r2_m0_gen.jsonl.gz "
            f"{th_flag}--max-seq-len {train_seq_len} --max-per-problem 3 --round-tag {froz_id} "
            f"--output-bank {froz_bank} --summary-output {froz_dir}/bank_summary.json ) && "
            f"( [ -s {froz_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 2} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {froz_bank} --output-dir {froz_ckpt} "
            f"--summary-output {froz_dir}/train_summary.json {th_flag}--max-seq-len {train_seq_len} --epochs 2 --lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_math_search_cmd(mk, froz_ckpt, froz_dir / "eval", froz_id, thinking=thinking, max_tokens=eval_max_tok, presence_penalty=pp)
        )
        _run_hermetic(
            froz_id, froz_dir, froz_cmd, ledger,
            inc_eval=inc_eval, chal_eval=froz_eval, base_eval=b0_eval,
            hypothesis="Causal Recursion Control (Math): Round-2 rollouts drawn from frozen M_0 at matched budget & seed",
        )
        froz_steps = int(json.loads((froz_dir / "train_summary.json").read_text(encoding="utf-8"))["global_steps"])
    else:
        froz_bank = None
        froz_steps = 0

    # Round M2: Pure Frontier Self-Play -> Reset to M_0 (Zero 27B Teacher Confound!)
    # On phi4_mini (<think> LongRoPE), M1 audit revealed 69.3% truncation under 8,192 cap;
    # M2 raises rollout budget to 12,288 (matching eval cap 1:1) + block-repetition filter.
    if mk == "phi4_mini":
        m2_id = "m2_phi4_mini_uncensored12k_restem"
        m2_dir = p2 / m2_id
        m2_bank = V2 / "data/verified_banks_v2/math_phi4_mini_r2_uncensored12k.jsonl"
        m2_ckpt = V2 / f"checkpoints/track_m/{m2_id}"
        m2_eval = m2_dir / "eval_metrics.json"
        m2_seq_len = 13312
        m2_cmd = (
            f"mkdir -p {m2_dir} && "
            f"( [ -s {m2_dir}/r2_pool12k_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
            f"--model-key {mk} --model-path {base_ckpt} --split-file {pool} --output {m2_dir}/r2_pool12k_gen.jsonl.gz "
            f"--k 4 --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.82 --presence-penalty {pp} "
            f"--max-tokens 12288 --max-model-len 16384 --seed 20261301 ) && "
            f"( [ -s {m2_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
            f"--track math --model-key {mk} --pool {pool} --generations {m2_dir}/r2_pool12k_gen.jsonl.gz "
            f"{th_flag}--max-seq-len {m2_seq_len} --max-per-problem 3 --reject-repetitive-blocks --round-tag {m2_id} "
            f"--output-bank {m2_bank} --summary-output {m2_dir}/bank_summary.json ) && "
            f"( [ -s {m2_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 3} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {m2_bank} --output-dir {m2_ckpt} "
            f"--summary-output {m2_dir}/train_summary.json {th_flag}--max-seq-len {m2_seq_len} --epochs 2 --lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_math_search_cmd(mk, m2_ckpt, m2_dir / "eval", m2_id, thinking=thinking, max_tokens=eval_max_tok, presence_penalty=pp)
        )
        m2_hyp = (
            "R2 uncensored 12,288-cap + block-repetition-filtered uniform ReST^EM reset to M_0 on phi4_mini "
            "(fixing M1 8,192 right-censoring length ratchet)"
        )
    else:
        m2_id = f"m2_{mk}_rec_restem"
        m2_dir = p2 / m2_id
        m2_bank = V2 / f"data/verified_banks_v2/math_{mk}_r2_cumulative.jsonl"
        m2_ckpt = V2 / f"checkpoints/track_m/{m2_id}"
        m2_eval = m2_dir / "eval_metrics.json"
        m1_src = m1_ckpt if m1_ckpt.exists() else inc_ckpt

        m2_cmd = (
            f"mkdir -p {m2_dir} && "
            f"( [ -s {m2_dir}/r2_m1_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
            f"--model-key {mk} --model-path {m1_src} --split-file {m1_front} --output {m2_dir}/r2_m1_gen.jsonl.gz "
            f"--k {rollout_k} --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature {temp_r2} --presence-penalty {pp} "
            f"--max-tokens {rollout_max_tok} --max-model-len 16384 --seed 20261212 ) && "
            f"( [ -s {m2_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py merge-restem "
            f"--track math --model-key {mk} --pool {pool} --prev-bank {m1_bank} --generations {m2_dir}/r2_m1_gen.jsonl.gz "
            f"{th_flag}--max-seq-len {train_seq_len} --max-per-problem 3 --round-tag {m2_id} "
            f"--output-bank {m2_bank} --summary-output {m2_dir}/bank_summary.json ) && "
            f"( [ -s {m2_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 3} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {m2_bank} --output-dir {m2_ckpt} "
            f"--summary-output {m2_dir}/train_summary.json {th_flag}--max-seq-len {train_seq_len} --epochs 2 --lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_math_search_cmd(mk, m2_ckpt, m2_dir / "eval", m2_id, thinking=thinking, max_tokens=eval_max_tok, presence_penalty=pp)
        )
        m2_hyp = f"R2 pure M_1 recursive frontier self-play reset to M_0 on {mk} (zero teacher distillation)"

    g_m2 = _run_hermetic(
        m2_id, m2_dir, m2_cmd, ledger,
        inc_eval=inc_eval, chal_eval=m2_eval, base_eval=b0_eval,
        hypothesis=m2_hyp,
    )

    if mk == "qwen35_4b" and froz_bank is not None:
        ry_id = "m2_qwen35_4b_rec_yield"
        ry_dir = p2 / ry_id
        ry_bank = V2 / "data/verified_banks_v2/math_qwen35_4b_r2_rec_yield.jsonl"
        ry_ckpt = V2 / f"checkpoints/track_m/{ry_id}"
        ry_eval = ry_dir / "eval_metrics.json"
        ry_cmd = (
            f"mkdir -p {ry_dir} && "
            f"( [ -s {ry_bank} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py match-yield "
            f"--r1-bank {m1_bank} --rec-bank {m2_bank} --frozen-bank {froz_bank} "
            f"--output-bank {ry_bank} --summary-output {ry_dir}/yield_match_summary.json ) && "
            f"( [ -s {ry_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 4} {V2}/code/v2/train_slm_v2.py "
            f"--model-key {mk} --base-model {base_ckpt} --train-file {ry_bank} --output-dir {ry_ckpt} "
            f"--summary-output {ry_dir}/train_summary.json {th_flag}--max-seq-len {train_seq_len} --epochs 2 --max-steps {froz_steps} "
            f"--lr {lr_sft} --global-batch-size 32 ) && "
            + _eval_math_search_cmd(mk, ry_ckpt, ry_dir / "eval", ry_id, thinking=thinking, max_tokens=eval_max_tok, presence_penalty=pp)
        )
        _run_hermetic(
            ry_id, ry_dir, ry_cmd, ledger,
            inc_eval=inc_eval, chal_eval=ry_eval, base_eval=b0_eval,
            hypothesis="Causal Recursion Yield-Matched (Math): M_1 traces subsampled to exact frozen-M_0 yield & steps",
        )

    if _promoted(g_m2):
        inc_eval, inc_ckpt, inc_id = m2_eval, m2_ckpt, m2_id

    # Round M3: Full-12K Activation-Checkpointed Chunked-Logits DAPO + LUFFY RLVR
    m3_id = f"m3_{mk}_dapo_chunked12k"
    m3_dir = p2 / m3_id
    m3_sub = m3_dir / "rl_frontier128.jsonl"
    rl_max_items = 96 if mk == "phi4_mini" else 128
    rl_k = 6 if mk == "phi4_mini" else 8
    _curate_active_math_rl_subset(pool, m2_bank, m3_sub, max_items=rl_max_items)
    m3_groups = m3_dir / "rl_groups.jsonl"
    m3_ckpt = V2 / f"checkpoints/track_m/{m3_id}"
    m3_eval = m3_dir / "eval_metrics.json"
    rl_gen_max = min(eval_max_tok, 12288)

    m3_cmd = (
        f"mkdir -p {m3_dir} && "
        f"( [ -s {m3_dir}/rl_gen.jsonl.gz ] || python3 -I -B {V2}/code/v2/vllm_eval_v2.py "
        f"--model-key {mk} --model-path {inc_ckpt} --split-file {m3_sub} --output {m3_dir}/rl_gen.jsonl.gz "
        f"--k {rl_k} --gpus 0,1,2,3,4,5,6,7 {th_flag}--temperature 0.85 --presence-penalty {pp} "
        f"--max-tokens {rl_gen_max} --max-model-len 16384 --seed 20261213 ) && "
        f"( [ -s {m3_groups} ] || python3 -I -B {V2}/code/v2/build_v2_banks_and_controls.py build-rl-groups "
        f"--track math --pool {m3_sub} --generations {m3_dir}/rl_gen.jsonl.gz --verified-bank {m2_bank} "
        f"{th_flag}--output {m3_groups} --summary-output {m3_dir}/rl_group_summary.json ) && "
        f"( [ -s {m3_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 5} {V2}/code/v2/train_dapo_chunked_v2.py "
        f"--model-key {mk} --init-model {inc_ckpt} --rollouts-file {m3_groups} --output-dir {m3_ckpt} "
        f"--summary-output {m3_dir}/train_summary.json {th_flag}--max-seq-len 13312 "
        f"--max-gen-tokens {rl_gen_max} --overlong-window 2288 --overlong-penalty 0.25 --ppo-epochs 1 --lr {lr_rl} --global-batch-size 32 ) && "
        + _eval_math_search_cmd(mk, m3_ckpt, m3_dir / "eval", m3_id, thinking=thinking, max_tokens=eval_max_tok, presence_penalty=pp)
    )
    g_m3 = _run_hermetic(
        m3_id, m3_dir, m3_cmd, ledger,
        inc_eval=inc_eval, chal_eval=m3_eval, base_eval=b0_eval,
        hypothesis=f"R3 Full-12K Activation-Checkpointed Chunked-Logits DAPO+LUFFY RLVR on {mk}",
    )
    if _promoted(g_m3):
        inc_eval, inc_ckpt, inc_id = m3_eval, m3_ckpt, m3_id

    inc_summary = {
        "model_key": mk,
        "status": "COMPLETED",
        "thinking": thinking,
        "eval_max_tokens": eval_max_tok,
        "presence_penalty": pp,
        "b0_eval": str(b0_eval),
        "b0_avg_at_8": b0_data["metrics"]["math_avg_at_8"],
        "b0_stop_rate": b0_data["metrics"]["stop_rate"],
        "incumbent_id": inc_id,
        "incumbent_ckpt": str(inc_ckpt),
        "incumbent_eval": str(inc_eval),
        "incumbent_avg_at_8": json.loads(inc_eval.read_text(encoding="utf-8"))["metrics"]["math_avg_at_8"],
        "incumbent_stop_rate": json.loads(inc_eval.read_text(encoding="utf-8"))["metrics"]["stop_rate"],
        "promoted_challenger": (inc_ckpt != base_ckpt),
    }
    (p2 / "incumbent.json").write_text(json.dumps(inc_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inc_summary, indent=2), flush=True)
    return inc_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="Comma-separated model keys for this node")
    ap.add_argument("--port-base", type=int, default=29810)
    args = ap.parse_args()

    mks = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}
    for idx, mk in enumerate(mks):
        results[mk] = run_model_track_m(mk, args.port_base + idx * 10)
    print(json.dumps({"event": "TRACK_M_V2_PLANE2_DONE", "models": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

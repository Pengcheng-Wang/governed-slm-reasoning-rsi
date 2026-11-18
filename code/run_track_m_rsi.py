#!/usr/bin/env python3
"""
Autonomous Plane-2 Governed RSI Driver for Track M (Hard Competition & Olympiad Math on Qwen3.5-4B & 9B).

Runs `4B` (`--model-size 4b`, `node-1`) and `9B` (`--model-size 9b`, `node-2`) simultaneously across
16x A100-80GB GPUs with full resume-safety and strict `hermetic_runner.py` + `governed_loop_guard.py`
manifest verification & hash-chained ledger gating:
  - Round M1 (`ReST^EM` Round 1 Strict Native-Thinking Hygiene from `M_0`):
    Rolls out `M_0` (`k=6`, `--thinking`, `--presence-penalty 1.0`, `--max-tokens 12288`) on
    `math_frontier_pool.jsonl` (`600` hard competition problems), keeps ONLY `finish_reason == "stop"` +
    `sft_clean_admissible == True` (`0%` right-truncation, `<= 3` deduplicated traces/problem), trains
    Hybrid Gated-DeltaNet LoRA (`r=32, alpha=64, lr=5e-5, epochs=2`) fresh from `M_0`, and gates on `math_search128`.
  - Round M2 (`ReST^EM` Round 2 Frontier Expansion + Length-Bounded Verified `Qwen3.8-27B` Assist on `0`-Trace
    Unsolved Frontier Items -> **reset to `M_0`**):
    Rolls out `M1` incumbent (`k=6`) on unsolved/`1`-trace `next_frontier.jsonl`, PLUS uses `Qwen3.8-27B`
    (`k=4`, `--max-tokens 10240`, `--thinking`, `--presence-penalty 1.0`) strictly on `0`-trace unsolved problems
    (admitting ONLY verified, natural-stop `<= 10,240`-token solutions), merges cumulative bank, **resets to `M_0`**,
    trains `M2`, and gates on `math_search128`.
  - Round M3 (Frontier-Curated `DAPO + LUFFY` Soft-Overlong Clipped Policy-Gradient RLVR):
    Samples `G=8` on-policy rollouts on `128` active frontier problems (`1..6/8` pass + `LUFFY` 1-of-8 rescued `0/8`),
    applies linear soft-overlong penalty above `10,240` tokens (`max_gen_tokens=12288, overlong_window=2048`),
    trains via `train_dapo_luffy.py`, and gates on `math_search128`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/governed_slm_rsi_v1")
MANIFEST = ROOT / "IMMUTABLE_MANIFEST.json"
ROOT_HASH_FILE = Path("/scratch/governed_slm_rsi_v1_trust_root.sha256")
RULES_SEARCH = ROOT / "configs/gate_rules_math_search.json"


def _run_hermetic(
    cand_id: str,
    work_dir: Path,
    cmd: str,
    ledger: Path,
    incumbent_eval: Path | None = None,
    challenger_eval: Path | None = None,
    baseline_eval: Path | None = None,
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
        "--ledger", str(ledger),
    ]
    if incumbent_eval and challenger_eval:
        hcmd.extend(
            [
                "--incumbent-eval", str(incumbent_eval),
                "--baseline-eval", str(baseline_eval or incumbent_eval),
                "--challenger-eval", str(challenger_eval),
                "--rules", str(RULES_SEARCH),
                "--gate-out", str(gate_out),
            ]
        )
    subprocess.run(hcmd, check=True)
    return json.loads(gate_out.read_text(encoding="utf-8")) if gate_out.is_file() else None


def _promoted(gate: dict | None) -> bool:
    return bool(gate and str(gate.get("decision", "")) in ("PROMOTE", "PROVISIONAL_KEEP"))


def _eval_math_cmd(model_dir: Path, out_prefix: Path, cand_id: str, *, thinking: bool, max_tokens: int) -> str:
    sp = ROOT / "data/splits/math_search128.jsonl"
    th_gen = "--thinking --temperature 0.8 --presence-penalty 1.0" if thinking else "--temperature 0.7"
    th_sc = "--thinking " if thinking else ""
    proto = "plane2_math_search_think" if thinking else "plane2_math_search_direct12k"
    return (
        f"( [ -s {out_prefix}.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
        f"--model-path {model_dir} --split-file {sp} --output {out_prefix}.jsonl.gz "
        f"--summary-output {out_prefix}_gen.json --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 "
        f"{th_gen} --max-tokens {max_tokens} --max-model-len 32768 --seed 20261118 ) && "
        f"( [ -s {out_prefix}_metrics.json ] || python -I -B {ROOT}/eval/isolated_scorer.py "
        f"--track math --split-file {sp} --generations {out_prefix}.jsonl.gz "
        f"--expected-k 8 --candidate-id {cand_id} --protocol-id {proto} "
        f"{th_sc}--output {out_prefix}_metrics.json --scored-details-output {out_prefix}_details.jsonl.gz )"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=["4b", "9b"], required=True)
    ap.add_argument("--channel", choices=["direct12k", "think16k"], default="direct12k")
    args = ap.parse_args()

    ms = args.model_size
    use_think = args.channel == "think16k"
    eval_max_tok = 16384 if use_think else 12288
    rollout_max_tok = 8192
    train_seq_len = 2688 if ms == "9b" else 3456
    th_flag = "--thinking " if use_think else ""
    gen_th_r1 = "--thinking --temperature 0.8 --presence-penalty 1.0" if use_think else "--temperature 0.75"
    gen_th_r2 = "--thinking --temperature 0.85 --presence-penalty 1.0" if use_think else "--temperature 0.8"

    base_ckpt = ROOT / f"models/qwen35_{ms}"
    p2 = ROOT / f"results/plane2/track_m_{ms}"
    p2.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / f"results/plane2/track_m_{ms}_ledger.jsonl"

    b0_eval = (
        ROOT / f"results/plane1/math/q{ms}_think16k_pp1_metrics.json"
        if use_think
        else ROOT / f"results/plane1/math/q{ms}_direct12k_b0_metrics.json"
    )
    inc_eval = b0_eval
    inc_ckpt = base_ckpt
    pool = ROOT / "data/raw/math_frontier_pool600.jsonl"

    # Round M1: Strict Hygiene ReST^EM Round 1 from M_0 in calibrated winning channel
    m1_dir = p2 / f"m1_q{ms}_restem"
    m1_bank = ROOT / f"data/verified_banks/math_q{ms}_r1_verified.jsonl"
    m1_next = m1_dir / "next_frontier.jsonl"
    m1_ckpt = ROOT / f"checkpoints/track_m/m1_q{ms}_restem"
    m1_eval = m1_dir / "eval_metrics.json"
    lr_sft = "6e-5" if ms == "4b" else "4.5e-5"
    port_base = 29610 if ms == "4b" else 29620

    m1_cmd = (
        f"mkdir -p {m1_dir} {m1_bank.parent} && "
        f"( [ -s {m1_dir}/r1_pool_gen.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
        f"--model-path {base_ckpt} --split-file {pool} --output {m1_dir}/r1_pool_gen.jsonl.gz "
        f"--k 6 --tp 1 --gpus 0,1,2,3,4,5,6,7 {gen_th_r1} "
        f"--max-tokens {rollout_max_tok} --max-model-len 16384 --seed 20261120 ) && "
        f"( [ -s {m1_bank} ] || python -I -B {ROOT}/code/build_frontier_bank.py merge-restem "
        f"--track math --pool {pool} --generations {m1_dir}/r1_pool_gen.jsonl.gz "
        f"--tokenizer-path {base_ckpt} {th_flag}--max-seq-len {train_seq_len} --max-per-problem 3 "
        f"--round-tag m1_q{ms} --output-bank {m1_bank} --output-next-frontier {m1_next} "
        f"--summary-output {m1_dir}/bank_summary.json ) && "
        f"( [ -s {m1_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 1} {ROOT}/code/train_slm_rsi.py "
        f"--base-model {base_ckpt} --train-file {m1_bank} --output-dir {m1_ckpt} "
        f"--summary-output {m1_dir}/train_summary.json --mode lora {th_flag}--max-seq-len {train_seq_len} "
        f"--epochs 2 --lr {lr_sft} --lora-r 32 --lora-alpha 64 --global-batch-size 32 ) && "
        + _eval_math_cmd(m1_ckpt, m1_dir / "eval", f"m1_q{ms}_restem", thinking=use_think, max_tokens=eval_max_tok)
    )
    g_m1 = _run_hermetic(f"m1_q{ms}_restem", m1_dir, m1_cmd, ledger, inc_eval, m1_eval, b0_eval)
    if _promoted(g_m1):
        inc_eval, inc_ckpt = m1_eval, m1_ckpt

    # Round M2: Frontier Self-Play + Length-Bounded Verified 27B Assist on 0-trace Unsolved -> Reset to M_0!
    m2_dir = p2 / f"m2_q{ms}_frontier_restem"
    m2_bank = ROOT / f"data/verified_banks/math_q{ms}_r2_cumulative_verified.jsonl"
    m2_ckpt = ROOT / f"checkpoints/track_m/m2_q{ms}_frontier_restem"
    m2_eval = m2_dir / "eval_metrics.json"
    unsolved0_py = (
        "import json; "
        f"pool=[json.loads(l) for l in open('{pool}') if l.strip()]; "
        f"solved={{json.loads(l)['problem_id'] for l in open('{m2_dir}/bank_after_slm.jsonl') if l.strip()}}; "
        "u0=[r for r in pool if r['id'] not in solved]; "
        f"open('{m2_dir}/unsolved_0trace.jsonl','w').writelines(json.dumps(x)+'\\n' for x in u0)"
    )
    m2_cmd = (
        f"mkdir -p {m2_dir} && "
        f"( [ -s {m2_dir}/r2_slm_gen.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
        f"--model-path {inc_ckpt} --split-file {m1_next} --output {m2_dir}/r2_slm_gen.jsonl.gz "
        f"--k 6 --tp 1 --gpus 0,1,2,3,4,5,6,7 {gen_th_r2} "
        f"--max-tokens {rollout_max_tok} --max-model-len 16384 --seed 20261121 ) && "
        f"( [ -s {m2_dir}/bank_after_slm.jsonl ] || python -I -B {ROOT}/code/build_frontier_bank.py merge-restem "
        f"--track math --pool {pool} --prev-bank {m1_bank} --generations {m2_dir}/r2_slm_gen.jsonl.gz "
        f"--tokenizer-path {base_ckpt} {th_flag}--max-seq-len {train_seq_len} --max-per-problem 3 "
        f"--round-tag m2_slm_q{ms} --output-bank {m2_dir}/bank_after_slm.jsonl "
        f"--summary-output {m2_dir}/slm_bank_summary.json ) && "
        f"python -I -B -c \"{unsolved0_py}\" && "
        f"if [ -s {m2_dir}/unsolved_0trace.jsonl ] && [ -e {ROOT}/models/qwen38_27b/config.json ]; then "
        f"  ( [ -s {m2_dir}/r2_27b_gen.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
        f"  --model-path {ROOT}/models/qwen38_27b --split-file {m2_dir}/unsolved_0trace.jsonl "
        f"  --output {m2_dir}/r2_27b_gen.jsonl.gz --k 4 --tp 4 --gpus 0,1,2,3,4,5,6,7 "
        f"  {gen_th_r1} --max-tokens 4096 --max-model-len 16384 --seed 20261122 ) && "
        f"  ( [ -s {m2_bank} ] || python -I -B {ROOT}/code/build_frontier_bank.py merge-restem "
        f"  --track math --pool {pool} --prev-bank {m2_dir}/bank_after_slm.jsonl "
        f"  --generations {m2_dir}/r2_27b_gen.jsonl.gz --tokenizer-path {base_ckpt} {th_flag}"
        f"  --max-seq-len {train_seq_len} --max-per-problem 2 --round-tag m2_27b_assist_q{ms} "
        f"  --output-bank {m2_bank} --summary-output {m2_dir}/assist_bank_summary.json ); "
        f"else cp {m2_dir}/bank_after_slm.jsonl {m2_bank}; fi && "
        f"( [ -s {m2_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 2} {ROOT}/code/train_slm_rsi.py "
        f"--base-model {base_ckpt} --train-file {m2_bank} --output-dir {m2_ckpt} "
        f"--summary-output {m2_dir}/train_summary.json --mode lora {th_flag}--max-seq-len {train_seq_len} "
        f"--epochs 2 --lr {lr_sft} --lora-r 32 --lora-alpha 64 --lora-dropout 0.0 --global-batch-size 32 ) && "
        + _eval_math_cmd(m2_ckpt, m2_dir / "eval", f"m2_q{ms}_frontier_restem", thinking=use_think, max_tokens=eval_max_tok)
    )
    g_m2 = _run_hermetic(f"m2_q{ms}_frontier_restem", m2_dir, m2_cmd, ledger, inc_eval, m2_eval, b0_eval)
    if _promoted(g_m2):
        inc_eval, inc_ckpt = m2_eval, m2_ckpt

    # Round M3: Frontier-Curated DAPO + LUFFY Soft-Overlong RLVR
    m3_dir = p2 / f"m3_q{ms}_dapo_luffy"
    m3_ckpt = ROOT / f"checkpoints/track_m/m3_q{ms}_dapo_luffy"
    m3_eval = m3_dir / "eval_metrics.json"
    rl_sub_py = (
        "import json, random; "
        f"nxt=[json.loads(l) for l in open('{m1_next}') if l.strip()]; "
        "random.Random(20261123).shuffle(nxt); "
        f"open('{m3_dir}/rl_frontier128.jsonl','w').writelines(json.dumps(x)+'\\n' for x in nxt[:128])"
    )
    m3_rl_max_tok = 2432 if ms == "9b" else 3200
    m3_cmd = (
        f"mkdir -p {m3_dir} && python -I -B -c \"{rl_sub_py}\" && "
        f"( [ -s {m3_dir}/rl_gen.jsonl.gz ] || python -I -B {ROOT}/code/vllm_eval_engine.py "
        f"--model-path {inc_ckpt} --split-file {m3_dir}/rl_frontier128.jsonl "
        f"--output {m3_dir}/rl_gen.jsonl.gz --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 "
        f"{gen_th_r2} --max-tokens {m3_rl_max_tok} --max-model-len 16384 --seed 20261123 ) && "
        f"( [ -s {m3_dir}/rl_groups.jsonl ] || python -I -B {ROOT}/code/build_frontier_bank.py build-rl-groups "
        f"--track math --pool {pool} --generations {m3_dir}/rl_gen.jsonl.gz --verified-bank {m2_bank} "
        f"{th_flag}--output {m3_dir}/rl_groups.jsonl --summary-output {m3_dir}/rl_groups_summary.json ) && "
        f"( [ -s {m3_dir}/train_summary.json ] || torchrun --nproc_per_node=8 --master_port={port_base + 3} {ROOT}/code/train_dapo_luffy.py "
        f"--init-model {inc_ckpt} --rollouts-file {m3_dir}/rl_groups.jsonl "
        f"--output-dir {m3_ckpt} --summary-output {m3_dir}/train_summary.json "
        f"{th_flag}--max-seq-len {train_seq_len} --max-gen-tokens {m3_rl_max_tok} --overlong-window 1024 --overlong-penalty 0.5 "
        f"--ppo-epochs 1 --lr 1.8e-5 --lora-r 32 --lora-alpha 64 --global-batch-size 32 ) && "
        + _eval_math_cmd(m3_ckpt, m3_dir / "eval", f"m3_q{ms}_dapo_luffy", thinking=use_think, max_tokens=eval_max_tok)
    )
    g_m3 = _run_hermetic(f"m3_q{ms}_dapo_luffy", m3_dir, m3_cmd, ledger, inc_eval, m3_eval, b0_eval)
    if _promoted(g_m3):
        inc_eval, inc_ckpt = m3_eval, m3_ckpt

    winner = {
        "track": f"math_{ms}",
        "model_size": ms,
        "channel": args.channel,
        "winner_ckpt": str(inc_ckpt),
        "winner_search_eval": str(inc_eval),
    }
    (p2 / f"track_m_{ms}_winner.json").write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": f"TRACK_M_{ms.upper()}_RSI_COMPLETE", **winner}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

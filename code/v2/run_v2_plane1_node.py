#!/usr/bin/env python3
"""
Phase V2 Plane-1 Multi-Model Calibration & 9B Full-12K Checkpointed-Chunk Smoke Driver.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

V2 = Path("/scratch/governed_slm_rsi_v2")
V1 = Path("/scratch/governed_slm_rsi_v1")
MANIFEST_V2 = V2 / "IMMUTABLE_MANIFEST_V2.json"
TRUST_V2 = Path("/scratch/governed_slm_rsi_v2_trust_root.sha256")


def _verify_manifest() -> None:
    expected = TRUST_V2.read_text(encoding="utf-8").strip().split()[0]
    subprocess.run(
        [
            sys.executable, "-I", "-B", str(V1 / "eval/governed_loop_guard.py"),
            "verify-manifest",
            "--manifest", str(MANIFEST_V2),
            "--root", str(V2),
            "--expected-root-sha256", expected,
        ],
        check=True,
    )


def _eval_one(
    *,
    track: str,
    model_key: str,
    split_file: Path,
    out_dir: Path,
    cand_id: str,
    protocol_id: str,
    thinking: bool,
    max_tokens: int,
    temperature: float,
    presence_penalty: float = 0.0,
    max_model_len: int = 16384,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_p = out_dir / f"{cand_id}.jsonl.gz"
    gen_s = out_dir / f"{cand_id}_gen.json"
    met_p = out_dir / f"{cand_id}_metrics.json"
    det_p = out_dir / f"{cand_id}_details.jsonl.gz"

    if not gen_p.is_file():
        cmd = [
            sys.executable, "-I", "-B", str(V2 / "code/v2/vllm_eval_v2.py"),
            "--model-key", model_key,
            "--split-file", str(split_file),
            "--output", str(gen_p),
            "--summary-output", str(gen_s),
            "--k", "8",
            "--gpus", "0,1,2,3,4,5,6,7",
            "--temperature", str(temperature),
            "--presence-penalty", str(presence_penalty),
            "--max-tokens", str(max_tokens),
            "--max-model-len", str(max_model_len),
            "--seed", "20261118",
        ]
        if thinking:
            cmd.append("--thinking")
        subprocess.run(cmd, check=True)

    if not met_p.is_file():
        sc = [
            sys.executable, "-I", "-B", str(V1 / "eval/isolated_scorer.py"),
            "--track", track,
            "--split-file", str(split_file),
            "--generations", str(gen_p),
            "--expected-k", "8",
            "--candidate-id", cand_id,
            "--protocol-id", protocol_id,
            "--workers", "36",
            "--output", str(met_p),
            "--scored-details-output", str(det_p),
        ]
        if thinking:
            sc.append("--thinking")
        subprocess.run(sc, check=True)

    data = json.loads(met_p.read_text(encoding="utf-8"))
    print(json.dumps({"cand_id": cand_id, "metrics": data["metrics"]}), flush=True)
    return data


def _run_9b_12k_smoke() -> dict:
    """Proves Qwen3.5-9B trains full 12,800-token sequences without OOM via checkpointed lm_head chunks."""
    smoke_dir = V2 / "results/plane1/smoke_9b_12k"
    sum_p = smoke_dir / "smoke_9b_12k_summary.json"
    if sum_p.is_file():
        return json.loads(sum_p.read_text(encoding="utf-8"))

    smoke_dir.mkdir(parents=True, exist_ok=True)
    synth_rl = smoke_dir / "synth_12k_rollouts.jsonl"
    # Construct 2 synthetic ~11,500-token groups (16 full-length ~11.5K sequences!) to stress-test peak VRAM
    long_body = ("Let us verify step by step that k^2 + k + 1 > 0 for real k. " * 520) + "\nFinal result: \\boxed{42}"
    rows = []
    for g in range(2):
        for s in range(8):
            rows.append(
                {
                    "id": f"smoke_prob_{g}",
                    "sample_idx": s,
                    "prompt_messages": [{"role": "user", "content": "Compute 6 * 7 and place the final answer in \\boxed{}."}],
                    "response_text": long_body,
                    "finish_reason": "stop" if s < 4 else "length",
                    "verified_ok": (s < 4),
                    "is_off_policy": False,
                }
            )
    with synth_rl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    out_ckpt = smoke_dir / "tmp_merged_9b"
    subprocess.run(
        [
            "torchrun", "--nproc_per_node=8", "--master_port=29701",
            str(V2 / "code/v2/train_dapo_chunked_v2.py"),
            "--model-key", "qwen35_9b",
            "--init-model", "/scratch/governed_slm_rsi_v1/models/qwen35_9b",
            "--rollouts-file", str(synth_rl),
            "--output-dir", str(out_ckpt),
            "--summary-output", str(sum_p),
            "--max-seq-len", "12800",
            "--max-gen-tokens", "12288",
            "--ppo-epochs", "1",
            "--max-steps", "1",
            "--global-batch-size", "16",
        ],
        check=True,
    )
    import shutil
    shutil.rmtree(out_ckpt, ignore_errors=True)
    return json.loads(sum_p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["node0_lean", "node1_math_small", "node2_math_large"], required=True)
    args = ap.parse_args()

    _verify_manifest()
    lean_sp = V2 / "data/splits_v2/v2_lean_search120.jsonl"
    math_sp = V2 / "data/splits_v2/v2_math_search128.jsonl"

    if args.role == "node0_lean":
        od = V2 / "results/plane1/lean"
        _eval_one(track="lean", model_key="qwen35_4b", split_file=lean_sp, out_dir=od, cand_id="b0_qwen35_4b_direct", protocol_id="v2_lean_search", thinking=False, max_tokens=2048, temperature=0.8)
        _eval_one(track="lean", model_key="qwen3_4b", split_file=lean_sp, out_dir=od, cand_id="b0_qwen3_4b_direct", protocol_id="v2_lean_search", thinking=False, max_tokens=2048, temperature=0.8)
        _eval_one(track="lean", model_key="qwen3_4b", split_file=lean_sp, out_dir=od, cand_id="b0_qwen3_4b_think6k", protocol_id="v2_lean_search_think", thinking=True, max_tokens=6144, temperature=0.8, presence_penalty=1.0)
        _eval_one(track="lean", model_key="phi4_mini", split_file=lean_sp, out_dir=od, cand_id="b0_phi4_mini_think6k", protocol_id="v2_lean_search_think", thinking=True, max_tokens=6144, temperature=0.8)
        _eval_one(track="lean", model_key="qwen35_9b", split_file=lean_sp, out_dir=od, cand_id="b0_qwen35_9b_direct", protocol_id="v2_lean_search", thinking=False, max_tokens=2048, temperature=0.8)
        _eval_one(track="lean", model_key="goedel_8b", split_file=lean_sp, out_dir=od, cand_id="ref_goedel_8b_direct", protocol_id="v2_lean_search", thinking=False, max_tokens=2048, temperature=0.8)
        _eval_one(track="lean", model_key="goedel_8b", split_file=lean_sp, out_dir=od, cand_id="ref_goedel_8b_think8k", protocol_id="v2_lean_search_think", thinking=True, max_tokens=8192, temperature=0.8)
        _eval_one(track="lean", model_key="qwen38_27b", split_file=lean_sp, out_dir=od, cand_id="ref_qwen38_27b_direct", protocol_id="v2_lean_search", thinking=False, max_tokens=2048, temperature=0.8)

    elif args.role == "node1_math_small":
        od = V2 / "results/plane1/math"
        _eval_one(track="math", model_key="qwen35_4b", split_file=math_sp, out_dir=od, cand_id="b0_qwen35_4b_direct12k", protocol_id="v2_math_search_direct12k", thinking=False, max_tokens=12288, temperature=0.7)
        _eval_one(track="math", model_key="qwen3_4b", split_file=math_sp, out_dir=od, cand_id="b0_qwen3_4b_direct12k", protocol_id="v2_math_search_direct12k", thinking=False, max_tokens=12288, temperature=0.7)
        _eval_one(track="math", model_key="qwen3_4b", split_file=math_sp, out_dir=od, cand_id="b0_qwen3_4b_think16k", protocol_id="v2_math_search_think16k", thinking=True, max_tokens=16384, temperature=0.8, presence_penalty=1.0, max_model_len=24576)
        _eval_one(track="math", model_key="phi4_mini", split_file=math_sp, out_dir=od, cand_id="b0_phi4_mini_think12k", protocol_id="v2_math_search_think12k", thinking=True, max_tokens=12288, temperature=0.8)

    elif args.role == "node2_math_large":
        smoke_res = _run_9b_12k_smoke()
        print(json.dumps({"event": "SMOKE_9B_12K_PASSED", "peak_gib": smoke_res.get("peak_allocated_gib")}), flush=True)
        od = V2 / "results/plane1/math"
        _eval_one(track="math", model_key="qwen35_9b", split_file=math_sp, out_dir=od, cand_id="b0_qwen35_9b_direct12k", protocol_id="v2_math_search_direct12k", thinking=False, max_tokens=12288, temperature=0.7)
        _eval_one(track="math", model_key="qwen38_27b", split_file=math_sp, out_dir=od, cand_id="ref_qwen38_27b_direct12k", protocol_id="v2_math_search_direct12k", thinking=False, max_tokens=12288, temperature=0.7)

    _verify_manifest()
    print(json.dumps({"event": "PLANE1_NODE_COMPLETE", "role": args.role}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

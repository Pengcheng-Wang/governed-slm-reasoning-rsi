#!/usr/bin/env python3
"""
Pure Zero-SFT Native-Horizon RLVR (`B_0 -> DAPO` & `B_0 -> Dr.GRPO+CISPO`) on `Qwen3-30B-A3B` Math.

Tests the symmetric Zero-SFT RLVR hypothesis directly on `Qwen3-30B-A3B` (`3B`-active pure-attention MoE,
native `99.69%` Search / `99.83%` Confirm `stop_rate`) to eliminate Stage-II long-CoT verbosity inflation
while preserving `stop_rate` well inside the locked `<= 2.50 pp` Confirm degradation guardrail.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
V3_ROOT = THIS_DIR.parent.parent
for p in (THIS_DIR, V3_ROOT / "eval_v3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_registry_v3 import MODEL_REGISTRY_V3  # noqa: E402
from run_v3_lean_round2_moe import _verify_both_manifests  # noqa: E402
from run_v3_node_worker import (  # noqa: E402
    RLVR_PY,
    _run_checked,
    eval_and_score_model,
    generate_and_verify_shared_rlvr_rollouts,
)


def main() -> int:
    _verify_both_manifests()
    mk = "qwen3_30b_a3b"
    spec = MODEL_REGISTRY_V3[mk]
    p1_dir = V3_ROOT / "results_v3/plane1" / mk
    p2_dir = V3_ROOT / "results_v3/plane2" / f"math_{mk}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"math_{mk}"
    b0_search = p1_dir / f"math_b0_{mk}_search.json"
    sp_search = V3_ROOT / "data/splits_v3/v3_math_search160.jsonl"
    rl_pool = V3_ROOT / "data/corpora_v3" / f"v3_r2_math_rl_pool_{mk}.jsonl"
    if not rl_pool.exists():
        rl_pool = V3_ROOT / "data/corpora_v3/v3_math_rl_pool640.jsonl"

    b0_rollouts = generate_and_verify_shared_rlvr_rollouts(
        track="math",
        model_key=mk,
        sft_star_path=spec.local_path,
        rl_pool_file=rl_pool,
        rl_subset_n=320,
        out_rollouts_jsonl=p2_dir / f"r3_pure_b0_rollouts_math_{mk}.jsonl",
        gpus="0,1,2,3,4,5,6,7",
        seed=20261128,
    )

    for algo, short_tag in [("dapo_asymclip", "pure_b0_dapo"), ("drgrpo_cispo", "pure_b0_cispo")]:
        _verify_both_manifests()
        rtag = f"r3_s3_math_{mk}_{short_tag}"
        out100 = ckpt_root / f"{rtag}_pct100"
        mroot = ckpt_root / f"{rtag}_milestones"
        rsum = p2_dir / f"{rtag}_train_summary.json"
        if not rsum.exists():
            _run_checked(
                [
                    "torchrun", "--master_port", str(random.randint(21000, 28999)), "--nproc_per_node=8", str(RLVR_PY),
                    "--algo", algo,
                    "--model-key", mk,
                    "--prior-fp32-deltas", "",
                    "--rollouts-file", str(b0_rollouts),
                    "--output-dir", str(out100),
                    "--milestones-root", str(mroot),
                    "--save-milestone-fractions", "0.50",
                    "--summary-output", str(rsum),
                    "--max-seq-len", "6144",
                    "--max-gen-tokens", "5120",
                    "--overlong-window", "1536",
                    "--ppo-epochs", "2",
                    "--max-steps", "32",
                    "--lr", "1.2e-5",
                    "--global-batch-size", "32",
                ]
            )
        for pct_tag, cpath in [("pct50", mroot / "pct50"), ("pct100", out100)]:
            if cpath.exists():
                cid = f"{rtag}_{pct_tag}"
                eval_and_score_model(
                    track="math",
                    model_key=mk,
                    model_path=str(cpath),
                    candidate_id=cid,
                    split_prompts_file=sp_search,
                    split_truth_file=sp_search,
                    out_scored_json=p2_dir / f"{cid}_search.json",
                    k=8,
                    gpus="0,1,2,3,4,5,6,7",
                    seed=20260325,
                    protocol_id="v3_math_search160_k8",
                    compare_baseline_json=b0_search,
                )
    print("=== NODE0_PURE_B0_MOE30_DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

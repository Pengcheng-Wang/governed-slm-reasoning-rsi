#!/usr/bin/env python3
"""
Phase V3 Model Registry (`Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B`, and `Goedel-Prover-V2-8B` reference).

Supports both DDP (`4B`, `9B`) and 8-GPU PyTorch FSDP `FULL_SHARD` (`27B`), with verified LoRA target
coverage across all hybrid Gated-DeltaNet + full-attention + SwiGLU linear projections (`496` on `27B`,
`248` on `4B`/`9B`).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List


HYBRID_QWEN35_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

HYBRID_QWEN35_MOE_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))$"
)

PURE_QWEN3_MOE_LORA_REGEX = (
    r"^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
)

DENSE_CAUSAL_LORA_REGEX = (
    r"^model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


@dataclass(frozen=True)
class ModelSpecV3:
    key: str
    hf_repo: str
    local_path: str
    family: str
    params_b: float
    loader_kind: str          # "image_text" | "causal_lm"
    dist_strategy: str        # "ddp" | "fsdp"
    vllm_lm_only: bool
    supports_enable_thinking_kwarg: bool
    lean_thinking: bool
    lean_max_tokens: int
    math_thinking: bool
    math_max_tokens: int
    stop_token_ids: List[int]
    sup_eos_id: int
    lora_regex: str
    lora_r: int
    lora_alpha: int
    expected_min_lora_modules: int
    tp_eval: int = 1


MODEL_REGISTRY_V3: dict[str, ModelSpecV3] = {
    "qwen35_4b": ModelSpecV3(
        key="qwen35_4b",
        hf_repo="Qwen/Qwen3.5-4B",
        local_path="/scratch/governed_slm_rsi_v1/models/qwen35_4b",
        family="Qwen3.5-Hybrid-GDN",
        params_b=4.0,
        loader_kind="image_text",
        dist_strategy="ddp",
        vllm_lm_only=True,
        supports_enable_thinking_kwarg=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=12288,
        stop_token_ids=[248044, 248046],
        sup_eos_id=248046,
        lora_regex=HYBRID_QWEN35_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=200,
        tp_eval=1,
    ),
    "qwen35_9b": ModelSpecV3(
        key="qwen35_9b",
        hf_repo="Qwen/Qwen3.5-9B",
        local_path="/scratch/governed_slm_rsi_v1/models/qwen35_9b",
        family="Qwen3.5-Hybrid-GDN",
        params_b=9.0,
        loader_kind="image_text",
        dist_strategy="ddp",
        vllm_lm_only=True,
        supports_enable_thinking_kwarg=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=12288,
        stop_token_ids=[248044, 248046],
        sup_eos_id=248046,
        lora_regex=HYBRID_QWEN35_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=200,
        tp_eval=1,
    ),
    "qwen38_27b": ModelSpecV3(
        key="qwen38_27b",
        hf_repo="Qwen/Qwen3.8-27B",
        local_path="/scratch/governed_slm_rsi_v1/models/qwen38_27b",
        family="Qwen3.8-Hybrid-GDN-27B",
        params_b=27.0,
        loader_kind="image_text",
        dist_strategy="fsdp",
        vllm_lm_only=True,
        supports_enable_thinking_kwarg=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=12288,
        stop_token_ids=[248044, 248046],
        sup_eos_id=248046,
        lora_regex=HYBRID_QWEN35_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=480,  # 64 layers -> 496 adapted linear projections
        tp_eval=4,
    ),
    "qwen38_35b_a3b": ModelSpecV3(
        key="qwen38_35b_a3b",
        hf_repo="Qwen/Qwen3.5-35B-A3B",
        local_path=os.environ.get("SLM_RSI_MOE35_PATH", "/scratch/governed_slm_rsi_v3/models/qwen38_35b_a3b"),
        family="Qwen3.8-MoE-35B-A3B",
        params_b=35.0,
        loader_kind="image_text",
        dist_strategy="fsdp",
        vllm_lm_only=True,
        supports_enable_thinking_kwarg=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=12288,
        stop_token_ids=[248044, 248046],
        sup_eos_id=248046,
        lora_regex=HYBRID_QWEN35_MOE_LORA_REGEX,
        lora_r=64,
        lora_alpha=128,
        expected_min_lora_modules=300,  # 40 MoE layers -> 310 unconditionally active attention + shared-expert Linear projections
        tp_eval=4,
    ),
    "qwen3_30b_a3b": ModelSpecV3(
        key="qwen3_30b_a3b",
        hf_repo="Qwen/Qwen3-30B-A3B",
        local_path=os.environ.get("SLM_RSI_MOE30_PATH", "/scratch/governed_slm_rsi_v3/models/qwen3_30b_a3b"),
        family="Qwen3-MoE-30B-A3B",
        params_b=30.0,
        loader_kind="causal_lm",
        dist_strategy="fsdp",
        vllm_lm_only=False,
        supports_enable_thinking_kwarg=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=12288,
        stop_token_ids=[151643, 151645],
        sup_eos_id=151645,
        lora_regex=PURE_QWEN3_MOE_LORA_REGEX,
        lora_r=64,
        lora_alpha=128,
        expected_min_lora_modules=190,  # 48 MoE layers -> 192 unconditionally active self-attention Linear projections
        tp_eval=4,
    ),
    "goedel_prover_v2_8b": ModelSpecV3(
        key="goedel_prover_v2_8b",
        hf_repo="Goedel-LM/Goedel-Prover-V2-8B",
        local_path="/scratch/governed_slm_rsi_v2/models/Goedel-Prover-V2-8B",
        family="Goedel-Prover-V2",
        params_b=8.0,
        loader_kind="causal_lm",
        dist_strategy="ddp",
        vllm_lm_only=False,
        supports_enable_thinking_kwarg=False,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=4096,
        stop_token_ids=[151643, 151645],
        sup_eos_id=151645,
        lora_regex=DENSE_CAUSAL_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=200,
        tp_eval=1,
    ),
    "phi4_mini_reasoning": ModelSpecV3(
        key="phi4_mini_reasoning",
        hf_repo="microsoft/Phi-4-mini-reasoning",
        local_path="/scratch/governed_slm_rsi_v2/models/Phi-4-mini-reasoning",
        family="Phi-4-Mini-Reasoning",
        params_b=3.8,
        loader_kind="causal_lm",
        dist_strategy="ddp",
        vllm_lm_only=False,
        supports_enable_thinking_kwarg=False,
        lean_thinking=True,
        lean_max_tokens=4096,
        math_thinking=True,
        math_max_tokens=8192,
        stop_token_ids=[],
        sup_eos_id=200020,
        lora_regex=DENSE_CAUSAL_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=60,
        tp_eval=1,
    ),
}

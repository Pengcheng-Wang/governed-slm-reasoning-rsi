#!/usr/bin/env python3
"""
Phase V2 Multi-Architecture SLM Registry (`Qwen3.5` Hybrid Gated-DeltaNet, `Qwen3` Pure Dense,
`Phi-4-mini-reasoning` Fused LongRoPE, `Goedel-Prover-V2-8B`, and `Qwen3.8-27B`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


HYBRID_QWEN35_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

DENSE_QWEN3_LORA_REGEX = (
    r"^model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

FUSED_PHI4_LORA_REGEX = (
    r"^model\.layers\.\d+\.(?:"
    r"self_attn\.(?:qkv_proj|o_proj)|"
    r"mlp\.(?:gate_up_proj|down_proj))$"
)


@dataclass(frozen=True)
class ModelSpecV2:
    key: str
    hf_repo: str
    local_path: str
    family: str
    params_b: float
    loader_kind: str  # "image_text" | "causal_lm"
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


MODEL_REGISTRY_V2: dict[str, ModelSpecV2] = {
    "qwen35_4b": ModelSpecV2(
        key="qwen35_4b",
        hf_repo="Qwen/Qwen3.5-4B",
        local_path="/scratch/governed_slm_rsi_v1/models/qwen35_4b",
        family="Qwen3.5-Hybrid-GDN",
        params_b=4.0,
        loader_kind="image_text",
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
    "qwen35_9b": ModelSpecV2(
        key="qwen35_9b",
        hf_repo="Qwen/Qwen3.5-9B",
        local_path="/scratch/governed_slm_rsi_v1/models/qwen35_9b",
        family="Qwen3.5-Hybrid-GDN",
        params_b=9.0,
        loader_kind="image_text",
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
    "qwen3_4b": ModelSpecV2(
        key="qwen3_4b",
        hf_repo="Qwen/Qwen3-4B",
        local_path="/scratch/governed_slm_rsi_v2/models/Qwen3-4B",
        family="Qwen3-Dense-Transformer",
        params_b=4.0,
        loader_kind="causal_lm",
        vllm_lm_only=False,
        supports_enable_thinking_kwarg=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        math_thinking=False,
        math_max_tokens=12288,
        stop_token_ids=[151643, 151645],
        sup_eos_id=151645,
        lora_regex=DENSE_QWEN3_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=240,
        tp_eval=1,
    ),
    "phi4_mini": ModelSpecV2(
        key="phi4_mini",
        hf_repo="microsoft/Phi-4-mini-reasoning",
        local_path="/scratch/governed_slm_rsi_v2/models/Phi-4-mini-reasoning",
        family="Phi-4-Mini-Reasoning",
        params_b=3.8,
        loader_kind="causal_lm",
        vllm_lm_only=False,
        supports_enable_thinking_kwarg=False,
        # Opus 5.5 Max safeguard: Phi-4-mini-reasoning is a pure long-CoT <think> distillation;
        # always run and grade in its native <think>...</think> channel (post-</think> answer only).
        lean_thinking=True,
        lean_max_tokens=6144,
        math_thinking=True,
        math_max_tokens=12288,
        stop_token_ids=[199999, 200020],
        sup_eos_id=200020,  # <|end|> turn terminator supervised inside loss mask
        lora_regex=FUSED_PHI4_LORA_REGEX,
        lora_r=48,  # Parameter-matched to unfused r=32 across fused qkv_proj & gate_up_proj
        lora_alpha=96,
        expected_min_lora_modules=120,
        tp_eval=1,
    ),
    "goedel_8b": ModelSpecV2(
        key="goedel_8b",
        hf_repo="Goedel-LM/Goedel-Prover-V2-8B",
        local_path="/scratch/governed_slm_rsi_v2/models/Goedel-Prover-V2-8B",
        family="Goedel-Prover-V2-Specialist",
        params_b=8.0,
        loader_kind="causal_lm",
        vllm_lm_only=False,
        supports_enable_thinking_kwarg=True,
        lean_thinking=True,
        lean_max_tokens=8192,
        math_thinking=False,
        math_max_tokens=8192,
        stop_token_ids=[151643, 151645],
        sup_eos_id=151645,
        lora_regex=DENSE_QWEN3_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=200,
        tp_eval=1,
    ),
    "qwen38_27b": ModelSpecV2(
        key="qwen38_27b",
        hf_repo="Qwen/Qwen3.8-27B",
        local_path="/scratch/governed_slm_rsi_v1/models/qwen38_27b",
        family="Qwen3.8-Frontier-Ref",
        params_b=27.0,
        loader_kind="image_text",
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
        tp_eval=4,
    ),
}

#!/usr/bin/env python3
"""
Phase V4 Cross-Family Model Registry (`Inkling-2-Mini`, `Gemma-4-26B-A4B`, `Gemma-4-31B`,
`Muse-Glimmer-30B`, plus `Gemma-4-E4B` active-compute compact Dense comparator).

Covers four distinct open-weight frontier families:
  1. Thinking Machines (`Inkling2LoopForConditionalGeneration`, looped 24->48 visit Sparse MoE)
  2. Google DeepMind (`Gemma4ForConditionalGeneration`, both `26B-A4B` Sparse MoE and `31B` / `E4B` Dense)
  3. Meta (`MuseGlimmerForConditionalGeneration`, `30B` Dense hybrid local/global gated attention)
  4. Direct 1-to-1 split & verifier comparability against Phase V3 Qwen Dense (`4B`, `9B`, `27B`)
     and Qwen `3B`-Active Sparse MoEs (`30B-A3B`, `35B-A3B`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List


INKLING2_LOOP_MOE_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj\.linear|k_proj\.linear|v_proj\.linear|o_proj\.gate_proj|o_proj\.linear|sink_proj)|"
    r"(?:q|k|v|mlp)_message_gate\.down_proj)$"
)

GEMMA4_UNCONDITIONAL_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)

MUSE_GLIMMER_DENSE_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


@dataclass(frozen=True)
class ModelSpecV4:
    key: str
    hf_repo: str
    local_path: str
    family: str
    arch_kind: str                  # "looped_moe" | "sparse_moe" | "dense"
    total_params_b: float
    active_params_b: float
    channel_family: str             # "inkling2" | "gemma4" | "muse_glimmer" | "qwen"
    eval_backend: str               # "vllm" | "hf_dp"
    loader_kind: str                # "inkling2_loop" | "gemma4" | "muse_glimmer"
    dist_strategy: str              # "fsdp" | "ddp"
    vllm_lm_only: bool
    lean_thinking: bool
    lean_max_tokens: int
    lean_chat_kwargs: Dict[str, Any]
    math_thinking: bool
    math_max_tokens: int
    math_chat_kwargs: Dict[str, Any]
    default_temperature: float
    default_top_p: float
    default_top_k: int
    stop_token_ids: List[int]
    pad_token_id: int
    sft_prefix: str                 # Exact assistant prefix after `add_generation_prompt=True` for SFT targets
    sft_suffix: str                 # Exact assistant turn-closing suffix supervised at EOS
    lora_regex: str
    lora_r: int
    lora_alpha: int
    expected_min_lora_modules: int
    tp_eval: int = 1
    hf_batch_groups: int = 1        # Number of equal-length prompt groups (`x k`) per GPU batch for `hf_dp`


ROOT_DEFAULT = os.environ.get("CROSSFAMILY_V4_ROOT", "/scratch/governed_crossfamily_v4")


MODEL_REGISTRY_V4: dict[str, ModelSpecV4] = {
    "inkling2_mini": ModelSpecV4(
        key="inkling2_mini",
        hf_repo="thinkingmachines/Inkling-2-Mini",
        local_path=os.environ.get("V4_INKLING2_PATH", f"{ROOT_DEFAULT}/models/inkling2_mini_bf16"),
        family="ThinkingMachines-Inkling2-Loop",
        arch_kind="looped_moe",
        total_params_b=26.4,
        active_params_b=3.4,   # 3.4B physical active (5.4B effective across 3 core passes)
        channel_family="inkling2",
        eval_backend="hf_dp",
        loader_kind="inkling2_loop",
        dist_strategy="fsdp",
        vllm_lm_only=False,
        lean_thinking=False,
        lean_max_tokens=3072,
        lean_chat_kwargs={"reasoning_effort": "low"},
        math_thinking=False,
        math_max_tokens=8192,
        math_chat_kwargs={"reasoning_effort": "medium"},
        default_temperature=0.8,
        default_top_p=0.95,
        default_top_k=64,
        stop_token_ids=[200006],
        pad_token_id=199999,
        # Prompt ends with `<|end_message|>`; direct model turn is `<|message_model|><|content_text|>{ans}<|end_message|><|content_model_end_sampling|>`
        sft_prefix="<|message_model|><|content_text|>",
        sft_suffix="<|end_message|><|content_model_end_sampling|>",
        lora_regex=INKLING2_LOOP_MOE_LORA_REGEX,
        lora_r=64,
        lora_alpha=128,
        expected_min_lora_modules=190,  # 24 * 6 attn + 12 * 4 loop message down_proj = 192 BF16 Linears
        tp_eval=1,
        hf_batch_groups=2,
    ),
    "gemma4_26b_a4b": ModelSpecV4(
        key="gemma4_26b_a4b",
        hf_repo="google/gemma-4-26B-A4B-it",
        local_path=os.environ.get("V4_GEMMA4_26B_PATH", f"{ROOT_DEFAULT}/models/gemma4_26b_a4b"),
        family="Google-Gemma4-MoE-26B-A4B",
        arch_kind="sparse_moe",
        total_params_b=25.2,
        active_params_b=3.8,
        channel_family="gemma4",
        eval_backend="vllm",
        loader_kind="gemma4",
        dist_strategy="fsdp",
        vllm_lm_only=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        lean_chat_kwargs={"enable_thinking": False},
        math_thinking=False,
        math_max_tokens=12288,
        math_chat_kwargs={"enable_thinking": False},
        default_temperature=0.8,
        default_top_p=0.95,
        default_top_k=64,
        stop_token_ids=[1, 106, 50],
        pad_token_id=0,
        # With `enable_thinking=False`, prompt already ends with `<|turn>model\n<|channel>thought\n<channel|>`
        sft_prefix="",
        sft_suffix="<turn|>",
        lora_regex=GEMMA4_UNCONDITIONAL_LORA_REGEX,
        lora_r=64,
        lora_alpha=128,
        expected_min_lora_modules=150,  # 30 layers unconditionally active self_attn + shared dense mlp (0 routed experts touched)
        tp_eval=2,
    ),
    "gemma4_31b": ModelSpecV4(
        key="gemma4_31b",
        hf_repo="google/gemma-4-31B-it",
        local_path=os.environ.get("V4_GEMMA4_31B_PATH", f"{ROOT_DEFAULT}/models/gemma4_31b"),
        family="Google-Gemma4-Dense-31B",
        arch_kind="dense",
        total_params_b=30.7,
        active_params_b=30.7,
        channel_family="gemma4",
        eval_backend="vllm",
        loader_kind="gemma4",
        dist_strategy="fsdp",
        vllm_lm_only=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        lean_chat_kwargs={"enable_thinking": False},
        math_thinking=False,
        math_max_tokens=12288,
        math_chat_kwargs={"enable_thinking": False},
        default_temperature=0.8,
        default_top_p=0.95,
        default_top_k=64,
        stop_token_ids=[1, 106, 50],
        pad_token_id=0,
        sft_prefix="",
        sft_suffix="<turn|>",
        lora_regex=GEMMA4_UNCONDITIONAL_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=300,  # 60 layers (with KV-shared layers omitting redundant k/v)
        tp_eval=4,
    ),
    "muse_glimmer_30b": ModelSpecV4(
        key="muse_glimmer_30b",
        hf_repo="meta-models/Muse-Glimmer-30B",
        local_path=os.environ.get("V4_MUSE_30B_PATH", f"{ROOT_DEFAULT}/models/muse_glimmer_30b"),
        family="Meta-MuseGlimmer-Dense-30B",
        arch_kind="dense",
        total_params_b=29.6,
        active_params_b=29.6,
        channel_family="muse_glimmer",
        eval_backend="vllm",
        loader_kind="muse_glimmer",
        dist_strategy="fsdp",
        vllm_lm_only=True,
        lean_thinking=False,
        lean_max_tokens=3072,
        lean_chat_kwargs={"current_date": "2026-11-22", "reasoning_strength": "low"},
        math_thinking=False,
        math_max_tokens=12288,
        math_chat_kwargs={"current_date": "2026-11-22", "reasoning_strength": "medium"},
        default_temperature=0.8,
        default_top_p=0.95,
        default_top_k=64,
        stop_token_ids=[200001, 200008],  # `<|end_of_text|>` and `<|eot|>`; NEVER `<|eom|>` (200007)!
        pad_token_id=200018,
        # Prompt ends with `<|start|>assistant`; direct user-channel derivation is ` to=user<|message|>{ans}<|eot|>`
        sft_prefix=" to=user<|message|>",
        sft_suffix="<|eot|>",
        lora_regex=MUSE_GLIMMER_DENSE_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=400,  # 52 layers * 8 projections = 416 Linears
        tp_eval=4,
    ),
    "gemma4_e4b": ModelSpecV4(
        key="gemma4_e4b",
        hf_repo="google/gemma-4-E4B-it",
        local_path=os.environ.get("V4_GEMMA4_E4B_PATH", f"{ROOT_DEFAULT}/models/gemma4_e4b"),
        family="Google-Gemma4-Dense-E4B",
        arch_kind="dense",
        total_params_b=8.0,
        active_params_b=4.5,
        channel_family="gemma4",
        eval_backend="vllm",
        loader_kind="gemma4",
        dist_strategy="ddp",
        vllm_lm_only=True,
        lean_thinking=False,
        lean_max_tokens=2048,
        lean_chat_kwargs={"enable_thinking": False},
        math_thinking=False,
        math_max_tokens=12288,
        math_chat_kwargs={"enable_thinking": False},
        default_temperature=0.8,
        default_top_p=0.95,
        default_top_k=64,
        stop_token_ids=[1, 106, 50],
        pad_token_id=0,
        sft_prefix="",
        sft_suffix="<turn|>",
        lora_regex=GEMMA4_UNCONDITIONAL_LORA_REGEX,
        lora_r=32,
        lora_alpha=64,
        expected_min_lora_modules=200,
        tp_eval=1,
    ),
}

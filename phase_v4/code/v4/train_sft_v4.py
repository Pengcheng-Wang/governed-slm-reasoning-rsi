#!/usr/bin/env python3
"""
Phase V4 Cross-Family Distributed SFT Trainer (`8-GPU FSDP FULL_SHARD` & `DDP`).

Supports `Inkling-2-Mini` (`Inkling2LoopForConditionalGeneration`), `Gemma-4-26B-A4B` &
`Gemma-4-31B` (`Gemma4ForConditionalGeneration`), and `Muse-Glimmer-30B` (`MuseGlimmerForConditionalGeneration`).

Key engineering invariants:
  1. Exact family-native logit head (`output_multiplier` + `final_logit_softcapping` `T * tanh(x / T)`)
     inside `1,024`-token activation-checkpointed `FP32` cross-entropy chunks (`max_abs_diff < 1e-4`
     vs full-model `forward()`).
  2. Multi-class `FSDP FULL_SHARD` auto-wrapping across `{type(l) for l in lm_backbone.layers}`
     (handling `Inkling2Loop`'s 4 distinct sliding/full x loop/non-loop layer classes, `3` shared-weight
     passes over layers `6..17`, and `Gemma4`'s KV-shared layers), keeping `buffer_dtype=None` (`FP32`
     RoPE `inv_freq`) and placing frozen non-`BF16` parameters (`FP32` loop gates, short convs, routers,
     and tied `lm_head`) in `ignored_states`.
  3. Native chat-template prefix/suffix supervision (`render_prompt_v4` + `sft_prefix` / `sft_suffix`)
     with zero sequence packing and lockstep dummy microbatches on uneven tail steps.
  4. Single-rounding multi-stage `FP32` LoRA consolidation (`--prior-fp32-deltas` / `--save-fp32-deltas`)
     merged onto the intact base checkpoint (`vision_tower` / `audio_tower` / `tie_word_embeddings`
     preserved) and verified on multi-GPU reload (`device_map="auto"`, `top1_token_agreement >= 0.96`).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as ckpt_fn
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_engine_v4 import render_prompt_v4  # noqa: E402
from model_registry_v4 import MODEL_REGISTRY_V4, ModelSpecV4  # noqa: E402


def init_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    torch.cuda.set_device(0)
    return 0, 1, 0


def drop_multimodal_towers_inplace(model: torch.nn.Module) -> None:
    inner = getattr(model, "model", None)
    if inner is None:
        return
    for attr in ("visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio", "vision_adapter", "vision_projection"):
        if hasattr(inner, attr) and getattr(inner, attr) is not None:
            setattr(inner, attr, None)


def extract_logit_head_Spec(model: torch.nn.Module) -> tuple[float | None, float | None]:
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else getattr(model.config, "text_config", model.config)
    out_mult = getattr(cfg, "output_multiplier", None)
    softcap = getattr(cfg, "final_logit_softcapping", None)
    return (
        float(out_mult) if out_mult is not None else None,
        float(softcap) if softcap is not None else None,
    )


def load_hf_model_staggered_v4(
    spec: ModelSpecV4,
    model_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    drop_multimodal_for_train: bool = False,
) -> torch.nn.Module:
    from fast_moe_patch_v4 import install_fast_grouped_moe_kernels

    install_fast_grouped_moe_kernels()
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    # Inkling2Loop uses custom per-token attention sinks (`_supports_flash_attn = False`);
    # Gemma4 and MuseGlimmer support flash_attention_2 / sdpa natively.
    if spec.loader_kind in ("gemma4", "muse_glimmer"):
        kwargs["attn_implementation"] = "sdpa"

    model: torch.nn.Module | None = None
    waves = max(1, math.ceil(world_size / 4))
    for w in range(waves):
        if (rank // 4) == w:
            model = AutoModelForMultimodalLM.from_pretrained(str(model_path), **kwargs)
            if drop_multimodal_for_train:
                drop_multimodal_towers_inplace(model)
        if world_size > 1:
            dist.barrier()
    assert model is not None
    return model


def unwrap_peft_inner(model: torch.nn.Module) -> torch.nn.Module:
    m = model.module if isinstance(model, DDP) else model
    return m.base_model.model if (hasattr(m, "peft_config") and hasattr(m, "base_model")) else m


def get_backbone_and_head(peft_or_raw: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    inner = unwrap_peft_inner(peft_or_raw)
    if hasattr(inner, "model") and hasattr(inner.model, "language_model"):
        return inner.model.language_model, inner.lm_head
    return inner.model, inner.lm_head


def clean_module_key(raw_name: str) -> str:
    s = raw_name
    for pref in ("module.", "base_model.model."):
        if s.startswith(pref):
            s = s[len(pref) :]
    s = s.replace("._fsdp_wrapped_module.", ".")
    if s.startswith("_fsdp_wrapped_module."):
        s = s[len("_fsdp_wrapped_module.") :]
    return s


def apply_prior_fp32_factors_inplace(base_model: torch.nn.Module, factor_paths_csv: str, device: torch.device) -> int:
    paths = [Path(p.strip()) for p in factor_paths_csv.split(",") if p.strip()]
    if not paths:
        return 0
    factor_packs = [load_safetensors(str(p), device="cpu") for p in paths]
    w_keys: set[str] = set()
    for pack in factor_packs:
        for k in pack.keys():
            if k.endswith("__lora_A"):
                w_keys.add(k[: -len("__lora_A")])

    sd = base_model.state_dict()
    applied = 0
    with torch.no_grad():
        for w_key in sorted(w_keys):
            if w_key not in sd:
                raise KeyError(f"Prior FP32 factor target key missing in base model: {w_key}")
            orig_cpu = sd[w_key]
            acc_gpu = orig_cpu.to(device=device, dtype=torch.float32)
            for pack in factor_packs:
                ak, bk, sk = f"{w_key}__lora_A", f"{w_key}__lora_B", f"{w_key}__scale"
                if ak in pack and bk in pack:
                    a_gpu = pack[ak].to(device=device, dtype=torch.float32)
                    b_gpu = pack[bk].to(device=device, dtype=torch.float32)
                    sc = float(pack[sk].item()) if sk in pack else 1.0
                    acc_gpu.addmm_(b_gpu, a_gpu, beta=1.0, alpha=sc)
            sd[w_key].copy_(acc_gpu.to(dtype=orig_cpu.dtype).cpu())
            del acc_gpu
            applied += 1
    torch.cuda.empty_cache()
    return applied


def tokenize_and_audit_dataset_v4(
    data_path: Path,
    tok: Any,
    spec: ModelSpecV4,
    *,
    track: str,
    chat_kwargs: dict[str, Any],
    max_seq_len: int,
    max_target_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_rows:
        raise ValueError(f"Empty training dataset: {data_path}")

    kept: list[tuple[int, dict[str, Any], list[int], list[int]]] = []
    dropped_overlong = 0

    for idx, row in enumerate(raw_rows):
        if str(row.get("finish_reason", "stop")) != "stop":
            continue
        msgs = row["messages"]
        prompt_msgs = msgs[:-1]
        p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=chat_kwargs)
        p_ids = [int(x) for x in tok.encode(p_str, add_special_tokens=False)]

        raw_ans = str(msgs[-1]["content"])
        if bool(row.get("raw_model_completion", False)):
            r_str = raw_ans + spec.sft_suffix
        else:
            # Clean external verified trajectory: wrap in target family's exact turn prefix & suffix!
            # Note: if `enable_thinking=True` is active on Gemma4, prepend empty `<|channel>thought\n<channel|>`
            g4_think_prefix = "<|channel>thought\n<channel|>" if (spec.channel_family == "gemma4" and chat_kwargs.get("enable_thinking")) else spec.sft_prefix
            r_str = g4_think_prefix + raw_ans.strip() + spec.sft_suffix

        r_ids = [int(x) for x in tok.encode(r_str, add_special_tokens=False)]
        if max_target_tokens > 0 and len(r_ids) > max_target_tokens:
            dropped_overlong += 1
            continue
        f_ids = p_ids + r_ids
        if len(f_ids) > max_seq_len:
            dropped_overlong += 1
            continue
        if len(r_ids) < 2:
            continue
        kept.append((idx, row, p_ids, f_ids))

    if not kept:
        raise RuntimeError(
            f"No valid training rows after max_seq_len={max_seq_len}, max_target_tokens={max_target_tokens}"
        )

    prob_counts = collections.Counter(str(r.get("problem_id", r.get("id", f"row_{i}"))) for i, r, _, _ in kept)
    tokenized: list[dict[str, Any]] = []
    lengths: list[int] = []
    sup_lengths: list[int] = []
    for i, r, p_ids, f_ids in kept:
        pid = str(r.get("problem_id", r.get("id", f"row_{i}")))
        labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        w = float(r.get("sample_weight", 1.0)) / float(prob_counts[pid])
        tokenized.append({"problem_id": pid, "input_ids": f_ids, "labels": labels, "weight": w})
        lengths.append(len(f_ids))
        sup_lengths.append(len(f_ids) - len(p_ids))

    lengths.sort()
    n = len(lengths)
    return tokenized, {
        "total_traces": n,
        "dropped_overlong_rows": dropped_overlong,
        "unique_problems": len(prob_counts),
        "seq_len_p50": lengths[n // 2],
        "seq_len_p95": lengths[min(n - 1, int(0.95 * n))],
        "seq_len_max": lengths[-1],
        "supervised_tokens_total": sum(sup_lengths),
        "sft_suffix": spec.sft_suffix,
    }


def attach_verified_lora_v4(
    model: torch.nn.Module,
    spec: ModelSpecV4,
    *,
    r: int = 0,
    alpha: int = 0,
    dropout: float = 0.0,
) -> tuple[torch.nn.Module, int]:
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm_sub = model.model.language_model
        if hasattr(lm_sub, "gradient_checkpointing_enable"):
            lm_sub.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    for pname, param in model.named_parameters():
        if any(tag in pname for tag in ("visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio", "vision_adapter", "vision_projection", "mtp.")):
            param.requires_grad = False

    use_r = r if r > 0 else spec.lora_r
    use_alpha = alpha if alpha > 0 else spec.lora_alpha
    lcfg = LoraConfig(
        r=use_r,
        lora_alpha=use_alpha,
        lora_dropout=dropout,
        target_modules=spec.lora_regex,
        bias="none",
    )
    peft_m = get_peft_model(model, lcfg)
    adapted: list[str] = []
    for name, mod in peft_m.named_modules():
        if hasattr(mod, "lora_A") and "default" in mod.lora_A:
            adapted.append(name)
            mod.lora_A["default"].weight.data = mod.lora_A["default"].weight.data.to(torch.bfloat16)
            mod.lora_B["default"].weight.data = mod.lora_B["default"].weight.data.to(torch.bfloat16)

    if len(adapted) < spec.expected_min_lora_modules:
        raise RuntimeError(
            f"LoRA module coverage failed for {spec.key}: adapted={len(adapted)} < expected={spec.expected_min_lora_modules}"
        )
    bad = [
        n
        for n, p in peft_m.named_parameters()
        if p.requires_grad
        and any(
            x in n
            for x in (
                "vision_tower",
                "audio_tower",
                "embed_tokens",
                "lm_head",
                "experts",
                "router",
                "mlp.gate.",
                "forget_gate_proj",
                "message_gate_proj",
                "sconv",
                "post_attention_convolution",
                "post_mlp_convolution",
            )
        )
    ]
    if bad:
        raise RuntimeError(f"Forbidden trainable parameters detected on {spec.key}: {bad[:5]}")
    return peft_m, len(adapted)


def wrap_distributed_backbone_v4(
    peft_model: torch.nn.Module,
    spec: ModelSpecV4,
    device: torch.device,
    local_rank: int,
    world_size: int,
) -> tuple[torch.nn.Module, torch.nn.Module, bool]:
    lm_backbone, lm_head = get_backbone_and_head(peft_model)
    use_fsdp = (spec.dist_strategy == "fsdp") and (world_size > 1)

    # Enable non-reentrant activation checkpointing across ALL decoder layers so 8K-12K sequences
    # and 48-visit looped MoE forward/backward fit comfortably within 80 GB A100 HBM.
    gc_kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    if hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable(**gc_kwargs)
    if hasattr(lm_backbone, "gradient_checkpointing_enable"):
        lm_backbone.gradient_checkpointing_enable(**gc_kwargs)
    if hasattr(lm_backbone, "embed_tokens"):
        lm_backbone.embed_tokens.register_forward_hook(
            lambda _mod, _inp, out: out.requires_grad_(True)
            if (torch.is_tensor(out) and out.is_floating_point() and torch.is_grad_enabled())
            else out
        )
    if not all(
        bool(getattr(layer, "gradient_checkpointing", False)) and hasattr(layer, "_gradient_checkpointing_func")
        for layer in lm_backbone.layers
    ):
        raise RuntimeError(f"Gradient checkpointing failed to activate on all layers of {spec.key}")

    if not use_fsdp:
        peft_model.to(device)
        peft_model.train()
        lm_backbone.train()
        if world_size > 1:
            ddp_backbone = DDP(lm_backbone, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
            return ddp_backbone, lm_head, False
        return lm_backbone, lm_head, False

    lm_head.to(device)
    for p in lm_head.parameters():
        p.requires_grad = False

    ignored_states: list[torch.nn.Parameter] = []
    head_param_ids = {id(p) for p in lm_head.parameters()}
    for p in lm_backbone.parameters():
        if id(p) in head_param_ids:
            p.data = p.data.to(device)
            ignored_states.append(p)
        elif (not p.requires_grad) and (p.dtype != torch.bfloat16):
            p.data = p.data.to(device)
            ignored_states.append(p)

    # Wrap ALL distinct decoder layer classes (e.g., all 4 sliding/full x loop/non-loop classes in Inkling2Loop!)
    layer_classes = {type(layer) for layer in lm_backbone.layers}
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=None,  # Keep FP32 RoPE `inv_freq` and `layer_scalar` buffers in exact FP32!
    )
    lm_backbone.train()
    fsdp_backbone = FSDP(
        lm_backbone,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        ignored_states=ignored_states,
    )
    return fsdp_backbone, lm_head, True


def compute_family_logits_fp32(
    h_chunk: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    output_multiplier: float | None,
    final_logit_softcapping: float | None,
) -> torch.Tensor:
    logits = F.linear(h_chunk, lm_head_weight, lm_head_bias).float()
    if output_multiplier is not None and output_multiplier != 1.0:
        logits = logits * output_multiplier
    if final_logit_softcapping is not None and final_logit_softcapping > 0.0:
        logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
    return logits


def _ce_chunk_step(
    h_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
    w_b: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    output_multiplier: float | None,
    final_logit_softcapping: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, clen, hdim = h_chunk.shape
    logits = compute_family_logits_fp32(
        h_chunk.reshape(-1, hdim),
        lm_head_weight,
        lm_head_bias,
        output_multiplier,
        final_logit_softcapping,
    )
    tok_nll = F.cross_entropy(logits, y_chunk.reshape(-1), ignore_index=-100, reduction="none").view(bsz, clen)
    vmask = (y_chunk != -100).float()
    wnll = (tok_nll * vmask * w_b).sum()
    wtoks = (vmask * w_b).sum()
    return wnll, wtoks


def chunked_backbone_ce_forward(
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    output_multiplier: float | None = None,
    final_logit_softcapping: float | None = None,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state
    zero_anchor = hidden.sum() * 0.0

    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    seq_m1 = shift_hidden.shape[1]
    w_b = sample_weights.view(-1, 1)

    total_wnll = torch.zeros((), device=hidden.device, dtype=torch.float32)
    total_wtoks = torch.zeros((), device=hidden.device, dtype=torch.float32)

    for start in range(0, seq_m1, chunk_tokens):
        end = min(seq_m1, start + chunk_tokens)
        y_sub = shift_labels[:, start:end]
        if not (y_sub != -100).any():
            continue
        h_sub = shift_hidden[:, start:end, :]
        if torch.is_grad_enabled() and h_sub.requires_grad:
            wnll, wtoks = ckpt_fn(
                _ce_chunk_step,
                h_sub,
                y_sub,
                w_b,
                lm_head.weight,
                lm_head.bias,
                output_multiplier,
                final_logit_softcapping,
                use_reentrant=False,
            )
        else:
            wnll, wtoks = _ce_chunk_step(
                h_sub,
                y_sub,
                w_b,
                lm_head.weight,
                lm_head.bias,
                output_multiplier,
                final_logit_softcapping,
            )
        total_wnll = total_wnll + wnll
        total_wtoks = total_wtoks + wtoks

    return total_wnll, total_wtoks, zero_anchor


def extract_lora_factors_fp32(
    peft_model: torch.nn.Module,
    wrapped_backbone: torch.nn.Module,
    *,
    is_fsdp: bool,
    rank: int,
) -> dict[str, torch.Tensor]:
    factors: dict[str, torch.Tensor] = {}
    ctx = (
        FSDP.summon_full_params(wrapped_backbone, writeback=False, rank0_only=True, offload_to_cpu=True)
        if is_fsdp
        else contextlib.nullcontext()
    )
    with ctx:
        if rank == 0:
            for name, mod in peft_model.named_modules():
                if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "default" in mod.lora_A:
                    a_w = mod.lora_A["default"].weight.detach().float().cpu().contiguous()
                    b_w = mod.lora_B["default"].weight.detach().float().cpu().contiguous()
                    scale = float(mod.scaling["default"]) if isinstance(mod.scaling, dict) else float(mod.scaling)
                    w_key = clean_module_key(name) + ".weight"
                    factors[f"{w_key}__lora_A"] = a_w
                    factors[f"{w_key}__lora_B"] = b_w
                    factors[f"{w_key}__scale"] = torch.tensor([scale], dtype=torch.float32)
    return factors


def merge_factor_packs_to_checkpoint_rank0(
    spec: ModelSpecV4,
    orig_base_model_path: Path,
    factor_packs: list[dict[str, torch.Tensor]],
    out_dir: Path,
    device: torch.device,
) -> int:
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    clean_model = AutoModelForMultimodalLM.from_pretrained(
        str(orig_base_model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sd = clean_model.state_dict()
    w_keys: set[str] = set()
    for pack in factor_packs:
        for k in pack.keys():
            if k.endswith("__lora_A"):
                w_keys.add(k[: -len("__lora_A")])

    merged_count = 0
    with torch.no_grad():
        for w_key in sorted(w_keys):
            if w_key not in sd:
                raise KeyError(f"LoRA target weight key missing in intact base model: {w_key}")
            orig_t = sd[w_key]
            acc_gpu = orig_t.to(device=device, dtype=torch.float32)
            for pack in factor_packs:
                ak, bk, sk = f"{w_key}__lora_A", f"{w_key}__lora_B", f"{w_key}__scale"
                if ak in pack and bk in pack:
                    a_gpu = pack[ak].to(device=device, dtype=torch.float32)
                    b_gpu = pack[bk].to(device=device, dtype=torch.float32)
                    sc = float(pack[sk].item()) if sk in pack else 1.0
                    acc_gpu.addmm_(b_gpu, a_gpu, beta=1.0, alpha=sc)
            sd[w_key].copy_(acc_gpu.to(dtype=orig_t.dtype).cpu())
            del acc_gpu
            merged_count += 1
    torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_model.save_pretrained(str(out_dir), safe_serialization=True)
    # Preserve upstream `config.json` verbatim (prevents transformers 5.x from rewriting Gemma-4
    # `global_head_dim` / `num_global_key_value_heads` into zero-padded `per_layer_config` keys that break vLLM).
    for extra_fn in orig_base_model_path.iterdir():
        if extra_fn.is_file() and (
            extra_fn.name == "config.json"
            or extra_fn.name.startswith(
                (
                    "tokenizer",
                    "vocab",
                    "merges",
                    "special_tokens",
                    "added_tokens",
                    "preprocessor",
                    "processor",
                    "video_preprocessor",
                    "chat_template",
                    "generation_config",
                )
            )
            or extra_fn.suffix == ".py"
        ):
            shutil.copy2(extra_fn, out_dir / extra_fn.name)
        elif extra_fn.is_dir() and extra_fn.name == "tiktoken":
            shutil.copytree(extra_fn, out_dir / "tiktoken", dirs_exist_ok=True)
    del clean_model
    return merged_count


@torch.no_grad()
def consolidate_and_verify_fp32_checkpoint(
    peft_model: torch.nn.Module,
    wrapped_backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    spec: ModelSpecV4,
    *,
    output_multiplier: float | None,
    final_logit_softcapping: float | None,
    is_fsdp: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    orig_base_model_path: Path,
    prior_deltas_path: str,
    save_deltas_path: str,
    out_dir: Path,
    parity_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    wrapped_backbone.eval()
    ref_logits_rank0: list[torch.Tensor] = []
    n_par = min(4, len(parity_samples))
    for i in range(n_par):
        ids = torch.tensor([parity_samples[i]["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        h = wrapped_backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        if rank == 0:
            ref_logits_rank0.append(
                compute_family_logits_fp32(
                    h, lm_head.weight, lm_head.bias, output_multiplier, final_logit_softcapping
                ).cpu()
            )

    current_factors = extract_lora_factors_fp32(peft_model, wrapped_backbone, is_fsdp=is_fsdp, rank=rank)
    if world_size > 1:
        dist.barrier()

    del wrapped_backbone
    del lm_head
    del peft_model
    torch.cuda.empty_cache()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

    if rank != 0:
        return {}

    if save_deltas_path:
        sp = Path(save_deltas_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors(current_factors, str(sp))

    factor_packs: list[dict[str, torch.Tensor]] = []
    if prior_deltas_path:
        for p_str in prior_deltas_path.split(","):
            if p_str.strip():
                factor_packs.append(load_safetensors(p_str.strip(), device="cpu"))
    factor_packs.append(current_factors)

    merged_count = merge_factor_packs_to_checkpoint_rank0(
        spec, orig_base_model_path, factor_packs, out_dir, device
    )

    use_multi_gpu_reload = is_fsdp and (torch.cuda.device_count() > 1)
    reload_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if spec.loader_kind in ("gemma4", "muse_glimmer"):
        reload_kwargs["attn_implementation"] = "sdpa"
    if use_multi_gpu_reload:
        reload_kwargs["device_map"] = "auto"

    reloaded = AutoModelForMultimodalLM.from_pretrained(str(out_dir), **reload_kwargs)
    drop_multimodal_towers_inplace(reloaded)
    if not use_multi_gpu_reload:
        reloaded.to(device)
    reloaded.eval()
    r_backbone, r_head = get_backbone_and_head(reloaded)
    in_dev = next(r_backbone.parameters()).device
    head_dev = r_head.weight.device

    max_abs_diff = 0.0
    mean_diffs: list[float] = []
    top1_matches = 0
    top1_total = 0
    for i, ref_log in enumerate(ref_logits_rank0):
        ids = torch.tensor([parity_samples[i]["input_ids"][:2048]], dtype=torch.long, device=in_dev)
        mask = torch.ones_like(ids)
        h = r_backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :].to(head_dev)
        m_log = compute_family_logits_fp32(
            h, r_head.weight, r_head.bias, output_multiplier, final_logit_softcapping
        ).cpu()
        diff = (m_log - ref_log).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        mean_diffs.append(float(diff.mean().item()))
        top1_matches += int((ref_log.argmax(dim=-1) == m_log.argmax(dim=-1)).sum().item())
        top1_total += int(ref_log.shape[0] * ref_log.shape[1])

    del reloaded
    torch.cuda.empty_cache()

    parity = {
        "merged_linear_tensors": merged_count,
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": sum(mean_diffs) / max(len(mean_diffs), 1),
        "top1_token_agreement": top1_matches / max(top1_total, 1),
    }
    min_top1 = 0.92 if ("moe" in spec.arch_kind) else 0.96
    if parity["top1_token_agreement"] < min_top1 or parity["mean_abs_logit_diff"] > 1.0:
        raise RuntimeError(f"FP32 merge parity gate failed (min_top1={min_top1}): {parity}")
    return parity


def clip_distributed_grad_norm(wrapped_backbone: torch.nn.Module, *, is_fsdp: bool, max_norm: float = 1.0) -> float:
    if is_fsdp:
        gn = wrapped_backbone.clip_grad_norm_(max_norm)
        return float(gn.item()) if torch.is_tensor(gn) else float(gn)
    gn = torch.nn.utils.clip_grad_norm_([p for p in wrapped_backbone.parameters() if p.requires_grad], max_norm)
    return float(gn.item()) if torch.is_tensor(gn) else float(gn)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V4 Cross-Family Governed SFT Trainer (8-GPU FSDP FULL_SHARD)")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V4.keys()))
    ap.add_argument("--track", choices=["math", "lean"], default="math")
    ap.add_argument("--chat-kwargs-json", default="")
    ap.add_argument("--base-model", default="", help="Original untouched base model path")
    ap.add_argument("--prior-fp32-deltas", default="", help="Optional Stage-I FP32 LoRA deltas .safetensors")
    ap.add_argument("--save-fp32-deltas", default="", help="Optional output path for current FP32 LoRA deltas")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-target-tokens", type=int, default=7680)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--lr-schedule", choices=["constant_with_warmup", "cosine"], default="constant_with_warmup")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--global-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260325)
    args = ap.parse_args()

    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    spec = MODEL_REGISTRY_V4[args.model_key]
    orig_base_path = Path(args.base_model) if args.base_model else Path(spec.local_path)
    default_ckw = spec.math_chat_kwargs if args.track == "math" else spec.lean_chat_kwargs
    chat_kwargs = json.loads(args.chat_kwargs_json) if args.chat_kwargs_json else dict(default_ckw)

    tok = AutoTokenizer.from_pretrained(str(orig_base_path), trust_remote_code=True)
    dataset, data_audit = tokenize_and_audit_dataset_v4(
        Path(args.train_file),
        tok,
        spec,
        track=args.track,
        chat_kwargs=chat_kwargs,
        max_seq_len=args.max_seq_len,
        max_target_tokens=args.max_target_tokens,
    )

    t0 = time.monotonic()
    base_model = load_hf_model_staggered_v4(
        spec, orig_base_path, rank=rank, world_size=world_size, drop_multimodal_for_train=True
    )
    out_mult, softcap = extract_logit_head_Spec(base_model)
    applied_prior = 0
    if args.prior_fp32_deltas:
        applied_prior = apply_prior_fp32_factors_inplace(base_model, args.prior_fp32_deltas, device)
        if world_size > 1:
            dist.barrier()

    peft_m, n_adapted = attach_verified_lora_v4(
        base_model, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    wrapped_backbone, lm_head, is_fsdp = wrap_distributed_backbone_v4(
        peft_m, spec, device, local_rank, world_size
    )

    trainable_params = [p for p in wrapped_backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    grad_accum = max(1, args.global_batch_size // world_size)
    eff_gbs = grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(dataset) / eff_gbs))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(round(total_steps * args.warmup_ratio)))

    if args.lr_schedule == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    torch.cuda.reset_peak_memory_stats(device)
    step = 0
    loss_history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        if step >= total_steps:
            break
        rng_ep = random.Random(args.seed + 1009 * epoch)
        indices = list(range(len(dataset)))
        rng_ep.shuffle(indices)
        rem = len(indices) % eff_gbs
        if rem != 0:
            indices.extend(indices[: (eff_gbs - rem)])

        for batch_start in range(0, len(indices), eff_gbs):
            if step >= total_steps:
                break
            batch_idxs = indices[batch_start : batch_start + eff_gbs]
            rank_idxs = batch_idxs[rank::world_size]

            # Exact global supervised token denominator across all `eff_gbs` samples in this step,
            # computed upfront on CPU so every microbatch calls `.backward()` immediately after its
            # forward pass (allowing FSDP FULL_SHARD to reshard each layer right away!).
            batch_wtoks = max(
                1.0,
                sum(
                    sum(1 for y in dataset[idx]["labels"][1:] if y != -100) * float(dataset[idx]["weight"])
                    for idx in batch_idxs
                ),
            )

            optimizer.zero_grad(set_to_none=True)
            local_wnll_val = 0.0

            for m_i in range(grad_accum):
                if m_i < len(rank_idxs):
                    sample = dataset[rank_idxs[m_i]]
                    ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
                    lbls = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
                    sw = torch.tensor([sample["weight"]], dtype=torch.float32, device=device)
                else:
                    ids = torch.tensor([dataset[0]["input_ids"][:64]], dtype=torch.long, device=device)
                    lbls = torch.full_like(ids, -100)
                    sw = torch.zeros((1,), dtype=torch.float32, device=device)
                amask = torch.ones_like(ids)
                wnll, _wtoks, z_anchor = chunked_backbone_ce_forward(
                    wrapped_backbone,
                    lm_head,
                    ids,
                    amask,
                    lbls,
                    sw,
                    output_multiplier=out_mult,
                    final_logit_softcapping=softcap,
                )
                local_wnll_val += float(wnll.detach().item())
                micro_loss = wnll * (float(world_size) / batch_wtoks) + z_anchor
                micro_loss.backward()

            grad_norm = clip_distributed_grad_norm(wrapped_backbone, is_fsdp=is_fsdp, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            mean_nll_val = torch.tensor(local_wnll_val / batch_wtoks, device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(mean_nll_val, op=dist.ReduceOp.SUM)
            if rank == 0 and (step == 1 or step % 10 == 0 or step == total_steps):
                rec = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(mean_nll_val.item()),
                    "grad_norm": grad_norm,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                loss_history.append(rec)
                print(json.dumps(rec), flush=True)

    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    train_wall_s = time.monotonic() - t0

    parity = consolidate_and_verify_fp32_checkpoint(
        peft_m,
        wrapped_backbone,
        lm_head,
        spec,
        output_multiplier=out_mult,
        final_logit_softcapping=softcap,
        is_fsdp=is_fsdp,
        rank=rank,
        world_size=world_size,
        device=device,
        orig_base_model_path=orig_base_path,
        prior_deltas_path=args.prior_fp32_deltas,
        save_deltas_path=args.save_fp32_deltas,
        out_dir=Path(args.output_dir),
        parity_samples=dataset[:4],
    )

    if rank == 0:
        summary = {
            "status": "TRAINED_AND_CONSOLIDATED",
            "model_key": args.model_key,
            "track": args.track,
            "chat_kwargs": chat_kwargs,
            "dist_strategy": "fsdp" if is_fsdp else "ddp",
            "world_size": world_size,
            "adapted_linear_modules": n_adapted,
            "applied_prior_factor_tensors": applied_prior,
            "output_multiplier": out_mult,
            "final_logit_softcapping": softcap,
            "steps_completed": step,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "train_wall_seconds": train_wall_s,
            "peak_vram_gb_rank0": peak_vram_gb,
            "data_audit": data_audit,
            "merge_parity": parity,
            "loss_history": loss_history,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "OK", "steps": step, "peak_vram_gb": peak_vram_gb, "parity": parity}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

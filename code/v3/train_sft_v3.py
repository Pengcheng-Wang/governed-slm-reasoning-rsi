#!/usr/bin/env python3
"""
Phase V3 Distributed SFT Trainer (`DDP` for `4B`/`9B` + `FSDP FULL_SHARD` for `27B`).

Key engineering invariants:
  1. Native dtype preservation (`dtype="auto"`): keeps Gated-DeltaNet `A_log` in exact `FP32` where
     stored as `FP32`, with staggered 2-wave CPU loading across 8 ranks to avoid host RAM spikes.
  2. `FSDP FULL_SHARD` (`use_orig_params=True`, `reduce_dtype=torch.float32`, `limit_all_gathers=True`)
     wrapping `lm_backbone` decoder layers while keeping frozen `lm_head` (`2.54 GB`) local per GPU,
     eliminating variable-chunk-count collective deadlocks across unpadded sequences.
  3. Unpadded `micro_batch_size=1` per rank (zero sequence packing or pad-token GDN state bleed),
     with lockstep dummy microbatches (`0.0 * hidden.sum()`) on partial tail steps.
  4. Globally shifted `1,024`-token activation-checkpointed `FP32` cross-entropy head normalized by
     exact `all_reduce(SUM)` global token weight (`loss_rank = local_wnll * world_size / global_wtoks`).
  5. Single-rounding multi-stage `FP32` LoRA consolidation (`--prior-fp32-deltas` / `--save-fp32-deltas`)
     merged onto an intact base checkpoint (`visual` and `tie_word_embeddings` preserved) with `>= 0.98`
     top-1 parity verification.
"""
from __future__ import annotations

import argparse
import collections
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
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v3 import MODEL_REGISTRY_V3, ModelSpecV3  # noqa: E402


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


def load_hf_model_staggered(
    spec: ModelSpecV3,
    model_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    flash_attn: bool = True,
    drop_visual_for_train: bool = False,
) -> torch.nn.Module:
    cls = AutoModelForImageTextToText if spec.loader_kind == "image_text" else AutoModelForCausalLM
    kwargs: dict[str, Any] = {
        "dtype": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model: torch.nn.Module | None = None
    waves = max(1, math.ceil(world_size / 4))
    for w in range(waves):
        if (rank // 4) == w:
            model = cls.from_pretrained(str(model_path), **kwargs)
            if drop_visual_for_train and hasattr(model, "model") and hasattr(model.model, "visual"):
                model.model.visual = None
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
    """
    Applies one or more comma-separated `.safetensors` low-rank FP32 factor files
    (`{w_key}__lora_A`, `{w_key}__lora_B`, `{w_key}__scale`) onto `base_model` in exact FP32 on GPU.
    """
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
                ak = f"{w_key}__lora_A"
                bk = f"{w_key}__lora_B"
                sk = f"{w_key}__scale"
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


def chat_prompt_ids(tok: Any, spec: ModelSpecV3, prompt_msgs: list[dict[str, str]], *, thinking: bool) -> list[int]:
    kwargs: dict[str, Any] = {"tokenize": True, "return_dict": False, "add_generation_prompt": True}
    if spec.supports_enable_thinking_kwarg:
        kwargs["enable_thinking"] = thinking
    try:
        out = tok.apply_chat_template(prompt_msgs, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        out = tok.apply_chat_template(prompt_msgs, **kwargs)
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def tokenize_and_audit_dataset_v3(
    data_path: Path,
    tok: Any,
    spec: ModelSpecV3,
    *,
    thinking: bool,
    max_seq_len: int,
    max_target_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_rows:
        raise ValueError(f"Empty training dataset: {data_path}")

    kept: list[tuple[int, dict[str, Any], list[int], list[int]]] = []
    dropped_overlong = 0
    stop_set = set(spec.stop_token_ids)

    for idx, row in enumerate(raw_rows):
        if str(row.get("finish_reason", "stop")) != "stop":
            continue
        msgs = row["messages"]
        prompt_msgs = msgs[:-1]
        p_ids = chat_prompt_ids(tok, spec, prompt_msgs, thinking=thinking)
        r_ids = [int(x) for x in tok.encode(str(msgs[-1]["content"]), add_special_tokens=False)]
        while r_ids and r_ids[-1] in stop_set:
            r_ids.pop()
        r_ids_eos = r_ids + [spec.sup_eos_id]
        if max_target_tokens > 0 and len(r_ids_eos) > max_target_tokens:
            dropped_overlong += 1
            continue
        f_ids = p_ids + r_ids_eos
        if len(f_ids) > max_seq_len:
            dropped_overlong += 1
            continue
        if len(r_ids_eos) < 2:
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
        "sup_eos_id": spec.sup_eos_id,
    }


def attach_verified_lora_v3(
    model: torch.nn.Module,
    spec: ModelSpecV3,
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
        if "visual" in pname or pname.startswith("mtp."):
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
            # Cast LoRA adapter weights to BF16 so FSDP FlatParameters have uniform BF16 dtype
            # while reduce_dtype=FP32 and AdamW master states maintain full FP32 precision.
            mod.lora_A["default"].weight.data = mod.lora_A["default"].weight.data.to(torch.bfloat16)
            mod.lora_B["default"].weight.data = mod.lora_B["default"].weight.data.to(torch.bfloat16)

    if len(adapted) < spec.expected_min_lora_modules:
        raise RuntimeError(
            f"LoRA module coverage failed for {spec.key}: adapted={len(adapted)} < expected={spec.expected_min_lora_modules}"
        )
    bad = [
        n
        for n, p in peft_m.named_parameters()
        if p.requires_grad and any(x in n for x in ("visual", "mtp.", "embed_tokens", "lm_head", "A_log", "dt_bias"))
    ]
    if bad:
        raise RuntimeError(f"Forbidden trainable parameters detected: {bad[:5]}")
    return peft_m, len(adapted)


def wrap_distributed_backbone(
    peft_model: torch.nn.Module,
    spec: ModelSpecV3,
    device: torch.device,
    local_rank: int,
    world_size: int,
) -> tuple[torch.nn.Module, torch.nn.Module, bool]:
    """
    Returns `(wrapped_backbone, local_lm_head, is_fsdp)`.
    For FSDP (`27B`), wraps `lm_backbone` with `FULL_SHARD` (`reduce_dtype=FP32`) and keeps frozen
    `lm_head` (`2.54 GB`) local on each GPU so chunked `FP32` CE/logprob never runs variable collectives.
    """
    lm_backbone, lm_head = get_backbone_and_head(peft_model)
    use_fsdp = (spec.dist_strategy == "fsdp") and (world_size > 1)

    if not use_fsdp:
        peft_model.to(device)
        peft_model.train()
        lm_backbone.train()
        if world_size > 1:
            ddp_backbone = DDP(lm_backbone, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
            return ddp_backbone, lm_head, False
        return lm_backbone, lm_head, False

    # Move frozen local head and any non-BF16 frozen tensors (e.g. FP32 GDN A_log) to GPU prior to FSDP
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

    layer_cls = type(lm_backbone.layers[0])
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
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


def _ce_chunk_step(
    h_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
    w_b: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, clen, hdim = h_chunk.shape
    logits = F.linear(h_chunk.reshape(-1, hdim), lm_head_weight, lm_head_bias).float()
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
                _ce_chunk_step, h_sub, y_sub, w_b, lm_head.weight, lm_head.bias, use_reentrant=False
            )
        else:
            wnll, wtoks = _ce_chunk_step(h_sub, y_sub, w_b, lm_head.weight, lm_head.bias)
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
    """
    Extracts compact low-rank FP32 `(lora_A, lora_B, scale)` tensors (`< 1 GB` even on `27B`!)
    with `_fsdp_wrapped_module.` prefixes stripped cleanly.
    """
    factors: dict[str, torch.Tensor] = {}
    if is_fsdp:
        ctx = FSDP.summon_full_params(wrapped_backbone, writeback=False, rank0_only=True, offload_to_cpu=True)
    else:
        import contextlib
        ctx = contextlib.nullcontext()

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
    spec: ModelSpecV3,
    orig_base_model_path: Path,
    factor_packs: list[dict[str, torch.Tensor]],
    out_dir: Path,
    device: torch.device,
) -> int:
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    cls = AutoModelForImageTextToText if spec.loader_kind == "image_text" else AutoModelForCausalLM
    clean_model = cls.from_pretrained(
        str(orig_base_model_path),
        dtype="auto",
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
    for extra_fn in orig_base_model_path.iterdir():
        if extra_fn.is_file() and (
            extra_fn.name.startswith(
                (
                    "tokenizer",
                    "vocab",
                    "merges",
                    "special_tokens",
                    "added_tokens",
                    "preprocessor",
                    "video_preprocessor",
                    "chat_template",
                    "generation_config",
                )
            )
            or extra_fn.suffix == ".py"
        ):
            shutil.copy2(extra_fn, out_dir / extra_fn.name)
    del clean_model
    return merged_count


@torch.no_grad()
def consolidate_and_verify_fp32_checkpoint(
    peft_model: torch.nn.Module,
    wrapped_backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    spec: ModelSpecV3,
    *,
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
            ref_logits_rank0.append(F.linear(h, lm_head.weight, lm_head.bias).float().cpu())

    current_factors = extract_lora_factors_fp32(peft_model, wrapped_backbone, is_fsdp=is_fsdp, rank=rank)
    if world_size > 1:
        dist.barrier()

    # Destroy process group BEFORE Rank-0 disk export & reload so Ranks 1..7 exit cleanly with 0 spin-wait!
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

    cls = AutoModelForImageTextToText if spec.loader_kind == "image_text" else AutoModelForCausalLM
    reloaded = cls.from_pretrained(
        str(out_dir),
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if hasattr(reloaded, "model") and hasattr(reloaded.model, "visual"):
        reloaded.model.visual = None
    reloaded.to(device)
    reloaded.eval()
    r_backbone, r_head = get_backbone_and_head(reloaded)

    max_abs_diff = 0.0
    mean_diffs: list[float] = []
    top1_matches = 0
    top1_total = 0
    for i, ref_log in enumerate(ref_logits_rank0):
        ids = torch.tensor([parity_samples[i]["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        h = r_backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        m_log = F.linear(h, r_head.weight, r_head.bias).float().cpu()
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
    if parity["top1_token_agreement"] < 0.96:
        raise RuntimeError(f"FP32 merge parity gate failed: {parity}")
    return parity


def clip_distributed_grad_norm(wrapped_backbone: torch.nn.Module, *, is_fsdp: bool, max_norm: float = 1.0) -> float:
    if is_fsdp:
        gn = wrapped_backbone.clip_grad_norm_(max_norm)
        return float(gn.item()) if torch.is_tensor(gn) else float(gn)
    gn = torch.nn.utils.clip_grad_norm_([p for p in wrapped_backbone.parameters() if p.requires_grad], max_norm)
    return float(gn.item()) if torch.is_tensor(gn) else float(gn)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V3 Governed SFT Trainer (DDP + 27B FSDP FULL_SHARD)")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V3.keys()))
    ap.add_argument("--base-model", default="", help="Original untouched base model path")
    ap.add_argument("--prior-fp32-deltas", default="", help="Optional Stage-I FP32 LoRA deltas .safetensors")
    ap.add_argument("--save-fp32-deltas", default="", help="Optional output path for cumulative FP32 LoRA deltas")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=9216)
    ap.add_argument("--max-target-tokens", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1.0e-4)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    t0 = time.monotonic()
    spec = MODEL_REGISTRY_V3[args.model_key]
    orig_base_path = Path(args.base_model or spec.local_path)
    tok = AutoTokenizer.from_pretrained(str(orig_base_path), trust_remote_code=True)
    dataset, data_audit = tokenize_and_audit_dataset_v3(
        Path(args.train_file),
        tok,
        spec,
        thinking=args.thinking,
        max_seq_len=args.max_seq_len,
        max_target_tokens=args.max_target_tokens,
    )
    if rank == 0:
        print(json.dumps({"event": "DATA_AUDIT_PASSED", "model_key": spec.key, **data_audit}), flush=True)

    base_m = load_hf_model_staggered(
        spec, orig_base_path, rank=rank, world_size=world_size, flash_attn=True, drop_visual_for_train=True
    )
    applied_prior = 0
    if args.prior_fp32_deltas:
        applied_prior = apply_prior_fp32_factors_inplace(base_m, args.prior_fp32_deltas, device)
        if world_size > 1:
            dist.barrier()

    peft_m, n_adapted = attach_verified_lora_v3(
        base_m, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    trainable_params = sum(p.numel() for p in peft_m.parameters() if p.requires_grad)

    wrapped_backbone, lm_head, is_fsdp = wrap_distributed_backbone(
        peft_m, spec, device, local_rank, world_size
    )

    grad_accum = max(1, args.global_batch_size // world_size)
    microbatches_per_step = grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(dataset) / microbatches_per_step))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in wrapped_backbone.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    if args.lr_schedule == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    pad_id = spec.stop_token_ids[0]
    dummy_sample = {"input_ids": [pad_id] * 16, "labels": [-100] * 16, "weight": 0.0, "is_dummy": True}

    step_losses: list[float] = []
    grad_norms: list[float] = []
    global_step = 0
    peak_mem_gb = 0.0

    for epoch in range(args.epochs):
        if global_step >= total_steps:
            break
        rng = random.Random(args.seed + 1009 * epoch)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        # Pad epoch stream to exact multiple of `microbatches_per_step` using zero-weight dummy items
        rem = len(indices) % microbatches_per_step
        padded_items: list[dict[str, Any]] = [dataset[idx] for idx in indices]
        if rem != 0:
            padded_items.extend([dummy_sample] * (microbatches_per_step - rem))

        for step_base in range(0, len(padded_items), microbatches_per_step):
            if global_step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            step_loss_accum = 0.0

            for ga in range(grad_accum):
                item = padded_items[step_base + ga * world_size + rank]
                ids_t = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                mask_t = torch.ones_like(ids_t)
                lbl_t = torch.tensor([item["labels"]], dtype=torch.long, device=device)
                w_t = torch.tensor([float(item["weight"])], dtype=torch.float32, device=device)

                local_wnll, local_wtoks, zero_anchor = chunked_backbone_ce_forward(
                    wrapped_backbone, lm_head, ids_t, mask_t, lbl_t, w_t
                )
                global_stats = torch.stack([local_wnll.detach(), local_wtoks.detach()])
                if world_size > 1:
                    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
                g_wnll, g_wtoks = float(global_stats[0].item()), float(global_stats[1].item())

                if g_wtoks > 1e-8:
                    # Multiply by `world_size` because DDP/FSDP averages gradients by `1 / world_size`
                    loss = (local_wnll * float(world_size)) / (g_wtoks * float(grad_accum)) + zero_anchor
                    step_loss_accum += (g_wnll / g_wtoks) / float(grad_accum)
                else:
                    loss = zero_anchor

                loss.backward()

            gnorm = clip_distributed_grad_norm(wrapped_backbone, is_fsdp=is_fsdp, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            step_losses.append(step_loss_accum)
            grad_norms.append(gnorm)
            peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated(device) / (1024.0 ** 3))

            if rank == 0 and (global_step == 1 or global_step % 10 == 0 or global_step == total_steps):
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "epoch": epoch + 1,
                            "loss": round(step_loss_accum, 5),
                            "grad_norm": round(gnorm, 4),
                            "lr": scheduler.get_last_lr()[0],
                            "peak_hbm_gb": round(peak_mem_gb, 2),
                        }
                    ),
                    flush=True,
                )

    parity_samples = dataset[: min(6, len(dataset))]
    parity = consolidate_and_verify_fp32_checkpoint(
        peft_m,
        wrapped_backbone,
        lm_head,
        spec,
        is_fsdp=is_fsdp,
        rank=rank,
        world_size=world_size,
        device=device,
        orig_base_model_path=orig_base_path,
        prior_deltas_path=args.prior_fp32_deltas,
        save_deltas_path=args.save_fp32_deltas,
        out_dir=Path(args.output_dir),
        parity_samples=parity_samples,
    )

    if rank == 0:
        summary = {
            "status": "TRAINED_AND_MERGED_V3",
            "model_key": spec.key,
            "dist_strategy": "fsdp_full_shard" if is_fsdp else "ddp",
            "world_size": world_size,
            "adapted_linear_modules": n_adapted,
            "applied_prior_fp32_deltas": applied_prior,
            "trainable_params": trainable_params,
            "lr_schedule": args.lr_schedule,
            "lr": args.lr,
            "epochs": args.epochs,
            "steps_executed": global_step,
            "initial_loss": step_losses[0] if step_losses else None,
            "final_loss": step_losses[-1] if step_losses else None,
            "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
            "peak_hbm_gb_rank0": round(peak_mem_gb, 3),
            "wall_seconds": round(time.monotonic() - t0, 2),
            "data_audit": data_audit,
            "merge_parity": parity,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

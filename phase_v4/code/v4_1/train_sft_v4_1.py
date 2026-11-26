#!/usr/bin/env python3
"""
Phase V4.1 Channel-Aligned & Precision-Safe SFT Trainer (`code/v4_1/train_sft_v4_1.py`).

Key design upgrades over `code/v4/train_sft_v4.py` (which remains frozen for V4.0):
  1. Native Reasoning-Channel Alignment (`--channel-mode`):
     - `stripped_mismatch`: Reproduces V4.0 `Arm B` channel bypass (negative control).
     - `effort_matched_wrapper`: Wrapper-only causal control. Renders thoughtless
       tactic/answer targets under zero-effort prompt metadata (`reasoning_effort="none"` ->
       `Thinking effort level: 0` on Inkling-2-Mini; `reasoning_strength="none"` ->
       `Reasoning strength: none.` on Muse-Glimmer-30B), keeping 100.0% identical supervised
       target tokens while avoiding direct suppression of active thinking prompt headers.
     - `native_thought_aligned`: Supervises full `(thought, final_answer)` pairs inside each
       family's exact native thinking channel (`<|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>`
       for Inkling-2-Mini; ` to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>`
       for Muse-Glimmer-30B; `{thought}<channel|>{final}` under `enable_thinking=True` for Gemma-4).
  2. Unbiased Stochastic `FP32 -> BF16` Consolidation (`--stochastic-bf16-merge`) + Post-Merge
     Retention Telemetry (`changed_entry_fraction`, `delta_energy_retention`, `inner_product_retention`,
     `trainable_numel`) eliminating deterministic BF16 rounding underflow.
  3. Default `cosine` LR decay preventing end-of-stage gradient norm spikes.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]
CODE_V4_DIR = ROOT / "code/v4"
EVAL_V4_DIR = ROOT / "eval_v4"
for p in (CODE_V4_DIR, EVAL_V4_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_engine_v4 import render_prompt_v4  # noqa: E402
from model_registry_v4 import MODEL_REGISTRY_V4, ModelSpecV4  # noqa: E402
from math_verifier_v4 import last_boxed  # noqa: E402
from train_sft_v4 import (  # noqa: E402
    apply_prior_fp32_factors_inplace,
    attach_verified_lora_v4,
    chunked_backbone_ce_forward,
    clip_distributed_grad_norm,
    compute_family_logits_fp32,
    drop_multimodal_towers_inplace,
    extract_logit_head_Spec,
    extract_lora_factors_fp32,
    get_backbone_and_head,
    init_dist,
    load_hf_model_staggered_v4,
    wrap_distributed_backbone_v4,
)


def stochastic_round_fp32_to_bf16(x_fp32: torch.Tensor, *, seed: int = 20261124) -> torch.Tensor:
    """Unbiased stochastic rounding from IEEE-754 FP32 to BF16: E[SR(x)] == x."""
    assert x_fp32.dtype == torch.float32
    gen = torch.Generator(device=x_fp32.device)
    gen.manual_seed(int(seed))
    noise = torch.randint(0, 65536, x_fp32.shape, dtype=torch.int32, device=x_fp32.device, generator=gen)
    bits = x_fp32.view(torch.int32)
    # Preserve exact zeros & non-finite values; apply unbiased 16-bit dither to finite non-zero magnitudes
    finite_nz = torch.isfinite(x_fp32) & (x_fp32 != 0.0)
    rounded_bits = torch.where(finite_nz, (bits + noise) & (-65536), bits & (-65536))
    return rounded_bits.view(torch.float32).to(torch.bfloat16)


def _split_derivation_and_final_answer(raw_ans: str, track: str, gold: str = "") -> tuple[str, str]:
    """Splits a clean step-by-step math solution into (internal_thought, concise_final_response)."""
    txt = raw_ans.strip()
    if track == "math":
        bx = last_boxed(txt) or str(gold).strip()
        if bx:
            return txt, f"\\boxed{{{bx}}}"
    return "", txt


def format_channel_aligned_pair_v4_1(
    tok: Any,
    spec: ModelSpecV4,
    prompt_msgs: list[dict[str, str]],
    raw_ans: str,
    *,
    row: dict[str, Any],
    track: str,
    eval_chat_kwargs: dict[str, Any],
    channel_mode: str,
) -> tuple[str, str]:
    """Returns `(prompt_str, target_completion_str)` under the requested `channel_mode`."""
    if bool(row.get("raw_model_completion", False)):
        p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=eval_chat_kwargs)
        suf = spec.sft_suffix
        if spec.channel_family == "inkling2" and raw_ans.rstrip().endswith("<|end_message|>"):
            suf = "<|content_model_end_sampling|>"
        return p_str, raw_ans + suf

    explicit_thought = str(row.get("thinking_text", "")).strip()

    if channel_mode == "stripped_mismatch":
        p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=eval_chat_kwargs)
        g4_pre = "<|channel>thought\n<channel|>" if (spec.channel_family == "gemma4" and eval_chat_kwargs.get("enable_thinking")) else spec.sft_prefix
        return p_str, g4_pre + raw_ans.strip() + spec.sft_suffix

    if channel_mode == "effort_matched_wrapper":
        # Pure wrapper-only fix: bind thoughtless supervision to native zero-effort mode so
        # supervised target tokens remain 100.0% identical while active thinking mode is uncorrupted!
        ckw = dict(eval_chat_kwargs)
        if not explicit_thought:
            if spec.channel_family == "inkling2":
                ckw["reasoning_effort"] = "none"
                p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=ckw)
                return p_str, f"<|message_model|><|content_text|>{raw_ans.strip()}<|end_message|><|content_model_end_sampling|>"
            if spec.channel_family == "muse_glimmer":
                ckw["reasoning_strength"] = "none"
                p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=ckw)
                return p_str, f" to=user<|message|>{raw_ans.strip()}<|eot|>"
            if spec.channel_family == "gemma4":
                ckw["enable_thinking"] = False
                p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=ckw)
                return p_str, f"{raw_ans.strip()}<turn|>"

    if channel_mode in ("native_thought_aligned", "effort_matched_wrapper"):
        thought_txt = explicit_thought
        final_txt = raw_ans.strip()
        if not thought_txt and track == "math":
            thought_txt, final_txt = _split_derivation_and_final_answer(raw_ans, track, str(row.get("gold", "")))

        if thought_txt:
            p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=eval_chat_kwargs)
            if spec.channel_family == "inkling2":
                r_str = (
                    f"<|message_model|><|content_thinking|>{thought_txt}<|end_message|>"
                    f"<|message_model|><|content_text|>{final_txt}<|end_message|><|content_model_end_sampling|>"
                )
                return p_str, r_str
            if spec.channel_family == "muse_glimmer":
                r_str = (
                    f" to=self<|message|>{thought_txt}<|eom|>"
                    f"<|start|>assistant to=user<|message|>{final_txt}<|eot|>"
                )
                return p_str, r_str
            if spec.channel_family == "gemma4":
                ckw = dict(eval_chat_kwargs)
                ckw["enable_thinking"] = True
                p_str = render_prompt_v4(tok, spec, prompt_msgs, chat_kwargs=ckw)
                return p_str, f"{thought_txt}<channel|>{final_txt}<turn|>"

        # Fallback for thoughtless rows under `native_thought_aligned`: route cleanly to effort=0
        return format_channel_aligned_pair_v4_1(
            tok,
            spec,
            prompt_msgs,
            raw_ans,
            row=row,
            track=track,
            eval_chat_kwargs=eval_chat_kwargs,
            channel_mode="effort_matched_wrapper",
        )

    raise ValueError(f"Unsupported channel_mode={channel_mode!r}")


def tokenize_and_audit_dataset_v4_1(
    data_path: Path,
    tok: Any,
    spec: ModelSpecV4,
    *,
    track: str,
    chat_kwargs: dict[str, Any],
    channel_mode: str,
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
        p_str, r_str = format_channel_aligned_pair_v4_1(
            tok,
            spec,
            msgs[:-1],
            str(msgs[-1]["content"]),
            row=row,
            track=track,
            eval_chat_kwargs=chat_kwargs,
            channel_mode=channel_mode,
        )
        p_ids = [int(x) for x in tok.encode(p_str, add_special_tokens=False)]
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
        raise RuntimeError(f"No valid training rows after max_seq_len={max_seq_len}")

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
        "channel_mode": channel_mode,
        "total_traces": n,
        "dropped_overlong_rows": dropped_overlong,
        "unique_problems": len(prob_counts),
        "seq_len_p50": lengths[n // 2],
        "seq_len_p95": lengths[min(n - 1, int(0.95 * n))],
        "seq_len_max": lengths[-1],
        "supervised_tokens_total": sum(sup_lengths),
    }


@torch.no_grad()
def merge_factor_packs_with_retention_rank0(
    spec: ModelSpecV4,
    orig_base_model_path: Path,
    factor_packs: list[dict[str, torch.Tensor]],
    out_dir: Path,
    device: torch.device,
    *,
    stochastic_bf16: bool = True,
    seed: int = 20261124,
) -> dict[str, Any]:
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
    total_entries = 0
    changed_entries = 0
    fp32_delta_sq_sum = 0.0
    bf16_delta_sq_sum = 0.0
    inner_prod_sum = 0.0

    for idx_k, w_key in enumerate(sorted(w_keys)):
        if w_key not in sd:
            raise KeyError(f"LoRA target weight key missing in base model: {w_key}")
        orig_t = sd[w_key]
        w0_fp32 = orig_t.to(device=device, dtype=torch.float32)
        dw_fp32 = torch.zeros_like(w0_fp32)
        for pack in factor_packs:
            ak, bk, sk = f"{w_key}__lora_A", f"{w_key}__lora_B", f"{w_key}__scale"
            if ak in pack and bk in pack:
                a_gpu = pack[ak].to(device=device, dtype=torch.float32)
                b_gpu = pack[bk].to(device=device, dtype=torch.float32)
                sc = float(pack[sk].item()) if sk in pack else 1.0
                dw_fp32.addmm_(b_gpu, a_gpu, beta=1.0, alpha=sc)

        acc_fp32 = w0_fp32 + dw_fp32
        if stochastic_bf16 and orig_t.dtype == torch.bfloat16:
            w_merged = stochastic_round_fp32_to_bf16(acc_fp32, seed=seed + idx_k)
        else:
            w_merged = acc_fp32.to(dtype=orig_t.dtype)

        eff_dw = w_merged.to(torch.float32) - w0_fp32
        total_entries += int(orig_t.numel())
        changed_entries += int((w_merged != orig_t.to(device=device)).sum().item())
        fp32_delta_sq_sum += float((dw_fp32 * dw_fp32).sum().item())
        bf16_delta_sq_sum += float((eff_dw * eff_dw).sum().item())
        inner_prod_sum += float((eff_dw * dw_fp32).sum().item())

        sd[w_key].copy_(w_merged.cpu())
        del w0_fp32, dw_fp32, acc_fp32, w_merged, eff_dw
        merged_count += 1

    torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_model.save_pretrained(str(out_dir), safe_serialization=True)
    for extra_fn in orig_base_model_path.iterdir():
        if extra_fn.is_file() and (
            extra_fn.name == "config.json"
            or extra_fn.name.startswith(
                ("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "preprocessor", "processor", "video_preprocessor", "chat_template", "generation_config")
            )
            or extra_fn.suffix == ".py"
        ):
            shutil.copy2(extra_fn, out_dir / extra_fn.name)
        elif extra_fn.is_dir() and extra_fn.name == "tiktoken":
            shutil.copytree(extra_fn, out_dir / "tiktoken", dirs_exist_ok=True)
    del clean_model

    denom_sq = max(fp32_delta_sq_sum, 1e-30)
    return {
        "merged_linear_tensors": merged_count,
        "stochastic_bf16_rounding": bool(stochastic_bf16),
        "total_merged_entries": total_entries,
        "changed_entry_fraction": changed_entries / float(max(1, total_entries)),
        "fp32_delta_rms": math.sqrt(fp32_delta_sq_sum / float(max(1, total_entries))),
        "delta_energy_retention_ratio": bf16_delta_sq_sum / denom_sq,
        "inner_product_retention_ratio": inner_prod_sum / denom_sq,
    }


@torch.no_grad()
def consolidate_and_verify_v4_1(
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
    stochastic_bf16: bool = True,
    seed: int = 20261124,
) -> dict[str, Any]:
    wrapped_backbone.eval()
    ref_logits_rank0: list[torch.Tensor] = []
    for i in range(min(4, len(parity_samples))):
        ids = torch.tensor([parity_samples[i]["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        h = wrapped_backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        if rank == 0:
            ref_logits_rank0.append(compute_family_logits_fp32(h, lm_head.weight, lm_head.bias, output_multiplier, final_logit_softcapping).cpu())

    current_factors = extract_lora_factors_fp32(peft_model, wrapped_backbone, is_fsdp=is_fsdp, rank=rank)
    if world_size > 1:
        dist.barrier()

    del wrapped_backbone, lm_head, peft_model
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

    retention = merge_factor_packs_with_retention_rank0(
        spec, orig_base_model_path, factor_packs, out_dir, device, stochastic_bf16=stochastic_bf16, seed=seed
    )

    use_multi_gpu_reload = is_fsdp and (torch.cuda.device_count() > 1)
    reload_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True, "trust_remote_code": True}
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
        m_log = compute_family_logits_fp32(h, r_head.weight, r_head.bias, output_multiplier, final_logit_softcapping).cpu()
        diff = (m_log - ref_log).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        mean_diffs.append(float(diff.mean().item()))
        top1_matches += int((ref_log.argmax(dim=-1) == m_log.argmax(dim=-1)).sum().item())
        top1_total += int(ref_log.shape[0] * ref_log.shape[1])

    del reloaded
    torch.cuda.empty_cache()

    parity = {
        **retention,
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": sum(mean_diffs) / max(len(mean_diffs), 1),
        "top1_token_agreement": top1_matches / max(top1_total, 1),
    }
    min_top1 = 0.92 if ("moe" in spec.arch_kind) else 0.96
    if parity["top1_token_agreement"] < min_top1 or parity["mean_abs_logit_diff"] > 1.0:
        raise RuntimeError(f"Merge parity gate failed (min_top1={min_top1}): {parity}")
    return parity


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V4.1 Channel-Aligned & Precision-Safe SFT Trainer")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V4.keys()))
    ap.add_argument("--track", choices=["math", "lean"], default="lean")
    ap.add_argument("--channel-mode", choices=["stripped_mismatch", "effort_matched_wrapper", "native_thought_aligned"], default="effort_matched_wrapper")
    ap.add_argument("--chat-kwargs-json", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--prior-fp32-deltas", default="")
    ap.add_argument("--save-fp32-deltas", default="")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-target-tokens", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2.2e-5)
    ap.add_argument("--lr-schedule", choices=["constant_with_warmup", "cosine"], default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--global-batch-size", type=int, default=32)
    ap.add_argument("--stochastic-bf16-merge", action="store_true", default=True)
    ap.add_argument("--no-stochastic-bf16-merge", dest="stochastic_bf16_merge", action="store_false")
    ap.add_argument("--seed", type=int, default=20261124)
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
    dataset, data_audit = tokenize_and_audit_dataset_v4_1(
        Path(args.train_file),
        tok,
        spec,
        track=args.track,
        chat_kwargs=chat_kwargs,
        channel_mode=args.channel_mode,
        max_seq_len=args.max_seq_len,
        max_target_tokens=args.max_target_tokens,
    )

    t0 = time.monotonic()
    base_model = load_hf_model_staggered_v4(spec, orig_base_path, rank=rank, world_size=world_size, drop_multimodal_for_train=True)
    out_mult, softcap = extract_logit_head_Spec(base_model)
    if args.prior_fp32_deltas:
        apply_prior_fp32_factors_inplace(base_model, args.prior_fp32_deltas, device)
        if world_size > 1:
            dist.barrier()

    peft_m, n_adapted = attach_verified_lora_v4(base_model, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    trainable_numel = sum(int(p.numel()) for p in peft_m.parameters() if p.requires_grad)
    wrapped_backbone, lm_head, is_fsdp = wrap_distributed_backbone_v4(peft_m, spec, device, local_rank, world_size)

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
                    am = torch.ones_like(ids)
                    sw = torch.tensor([float(sample["weight"])], dtype=torch.float32, device=device)
                    wnll, _wtoks, z_anc = chunked_backbone_ce_forward(
                        wrapped_backbone,
                        lm_head,
                        ids,
                        am,
                        lbls,
                        sw,
                        output_multiplier=out_mult,
                        final_logit_softcapping=softcap,
                    )
                    local_wnll_val += float(wnll.detach().item())
                    micro_loss = wnll * (float(world_size) / batch_wtoks) + z_anc
                    micro_loss.backward()
                else:
                    ids_d = torch.tensor([dataset[0]["input_ids"][:64]], dtype=torch.long, device=device)
                    lbl_d = torch.full_like(ids_d, -100)
                    am_d = torch.ones_like(ids_d)
                    sw_d = torch.zeros((1,), dtype=torch.float32, device=device)
                    _wnll_d, _wt_d, z_anc = chunked_backbone_ce_forward(
                        wrapped_backbone,
                        lm_head,
                        ids_d,
                        am_d,
                        lbl_d,
                        sw_d,
                        output_multiplier=out_mult,
                        final_logit_softcapping=softcap,
                    )
                    z_anc.backward()

            gn = clip_distributed_grad_norm(wrapped_backbone, is_fsdp=is_fsdp, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            loss_t = torch.tensor(local_wnll_val / batch_wtoks, device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            if rank == 0 and (step == 1 or step % 10 == 0 or step == total_steps):
                rec = {"step": step, "loss": float(loss_t.item()), "grad_norm": gn, "lr": float(scheduler.get_last_lr()[0])}
                loss_history.append(rec)
                print(json.dumps(rec), flush=True)

    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    train_wall_s = time.monotonic() - t0

    parity = consolidate_and_verify_v4_1(
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
        stochastic_bf16=args.stochastic_bf16_merge,
        seed=args.seed,
    )

    if rank == 0:
        summary = {
            "status": "TRAINED_AND_CONSOLIDATED_V4_1",
            "model_key": args.model_key,
            "track": args.track,
            "channel_mode": args.channel_mode,
            "adapted_lora_modules": n_adapted,
            "trainable_numel": trainable_numel,
            "steps_completed": step,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "seed": args.seed,
            "train_wall_sec": train_wall_s,
            "peak_vram_gb": peak_vram_gb,
            "data_audit": data_audit,
            "parity": parity,
            "loss_history": loss_history,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

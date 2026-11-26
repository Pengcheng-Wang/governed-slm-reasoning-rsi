#!/usr/bin/env python3
"""
Phase V4 Unified 3-Proposal Multi-Epoch Cross-Family Distributed RLVR Trainer (`8-GPU FSDP FULL_SHARD`).

Implements `dapo_asymclip`, `drgrpo_cispo`, and `gspo_luffy` across `Inkling-2-Mini`,
`Gemma-4-26B-A4B`, `Gemma-4-31B`, and `Muse-Glimmer-30B`:
  1. Uses exact family logit scaling & softcapping (`compute_family_logits_fp32`) inside `1,024`-token
     activation-checkpointed log-probability chunks so `pi_old` and `pi_theta` match inference logits!
  2. Appends stop EOS tokens ONLY on `finish_reason == "stop"` rollouts (never penalizing EOS on
     `finish_reason == "length"` truncated rollouts!).
  3. Supports `--save-milestone-fractions "0.50,1.00"` with multi-GPU `device_map="auto"` `FP32`
     merge parity verification (`top1_token_agreement >= 0.96`).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from torch.utils.checkpoint import checkpoint as ckpt_fn
from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_engine_v4 import render_prompt_v4  # noqa: E402
from model_registry_v4 import MODEL_REGISTRY_V4, ModelSpecV4  # noqa: E402
from train_sft_v4 import (  # noqa: E402
    apply_prior_fp32_factors_inplace,
    attach_verified_lora_v4,
    clip_distributed_grad_norm,
    compute_family_logits_fp32,
    consolidate_and_verify_fp32_checkpoint,
    extract_logit_head_Spec,
    extract_lora_factors_fp32,
    init_dist,
    load_hf_model_staggered_v4,
    merge_factor_packs_to_checkpoint_rank0,
    wrap_distributed_backbone_v4,
)

RLVR_ALGOS = ("dapo_asymclip", "drgrpo_cispo", "gspo_luffy")


def soft_overlong_penalty(comp_tokens: int, max_tokens: int, cache_window: int, max_pen: float) -> float:
    if cache_window <= 0 or max_pen <= 0.0:
        return 0.0
    thresh = max(1, max_tokens - cache_window)
    if comp_tokens <= thresh:
        return 0.0
    if comp_tokens >= max_tokens:
        return -max_pen
    return -max_pen * float(comp_tokens - thresh) / float(cache_window)


def compute_sample_rlvr_loss(
    algo: str,
    tok_logp: torch.Tensor,
    old_logp: torch.Tensor,
    shift_mask: torch.Tensor,
    *,
    advantage: float,
    verified_positive: bool,
    is_off_policy: bool,
    norm_tokens: float,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
    gspo_eps_low: float = 3e-4,
    gspo_eps_high: float = 4e-4,
    cispo_sft_alpha: float = 0.05,
    luffy_gamma: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    m = shift_mask.float()
    comp_len = m.sum().clamp_min(1.0)
    adv_t = torch.tensor(float(advantage), dtype=torch.float32, device=tok_logp.device)
    denom = max(float(norm_tokens), 1.0)

    if is_off_policy:
        pi_prob = torch.exp(tok_logp.detach().clamp(min=-20.0, max=0.0))
        w_luffy = (pi_prob / (pi_prob + luffy_gamma)).detach()
        seq_loss = -((w_luffy * adv_t * tok_logp) * m).sum() / comp_len
        return seq_loss, {
            "clipped_frac": 0.0,
            "mean_ratio": float(w_luffy[shift_mask.bool()].mean().item()) if shift_mask.any() else 0.0,
        }

    log_ratio = (tok_logp - old_logp.detach()) * m

    if algo == "dapo_asymclip":
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        clipped_r = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
        surr1 = ratio * adv_t
        surr2 = clipped_r * adv_t
        tok_obj = torch.minimum(surr1, surr2) * m
        seq_loss = -tok_obj.sum() / denom
        with torch.no_grad():
            is_clipped = ((adv_t > 0) & (ratio > 1.0 + eps_high)) | ((adv_t < 0) & (ratio < 1.0 - eps_low))
            cfrac = float((is_clipped.float() * m).sum().item() / float(comp_len.item()))
            mr = float((ratio * m).sum().item() / float(comp_len.item()))
        return seq_loss, {"clipped_frac": cfrac, "mean_ratio": mr}

    if algo == "drgrpo_cispo":
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        w_cispo = torch.clamp(ratio.detach(), 1.0 - eps_low, 1.0 + eps_high)
        pg_loss = -((w_cispo * adv_t * tok_logp) * m).sum() / denom
        if verified_positive and cispo_sft_alpha > 0.0:
            nll_anchor = -(tok_logp * m).sum() / denom
            seq_loss = pg_loss + cispo_sft_alpha * nll_anchor
        else:
            seq_loss = pg_loss
        with torch.no_grad():
            is_clipped = (ratio > 1.0 + eps_high) | (ratio < 1.0 - eps_low)
            cfrac = float((is_clipped.float() * m).sum().item() / float(comp_len.item()))
            mr = float((ratio * m).sum().item() / float(comp_len.item()))
        return seq_loss, {"clipped_frac": cfrac, "mean_ratio": mr}

    if algo == "gspo_luffy":
        seq_log_ratio = log_ratio.sum() / comp_len
        s_i = torch.exp(seq_log_ratio.clamp(-0.5, 0.5))
        s_clip = torch.clamp(s_i, 1.0 - gspo_eps_low, 1.0 + gspo_eps_high)
        seq_loss = -torch.minimum(s_i * adv_t, s_clip * adv_t)
        with torch.no_grad():
            clipped = bool(
                (advantage > 0 and float(s_i.item()) > 1.0 + gspo_eps_high)
                or (advantage < 0 and float(s_i.item()) < 1.0 - gspo_eps_low)
            )
        return seq_loss, {"clipped_frac": 1.0 if clipped else 0.0, "mean_ratio": float(s_i.item())}

    raise ValueError(f"Unknown RLVR algorithm: {algo!r}")


def _logp_chunk_step(
    hc: torch.Tensor,
    yc: torch.Tensor,
    m_sub: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    output_multiplier: float | None,
    final_logit_softcapping: float | None,
) -> torch.Tensor:
    logits = compute_family_logits_fp32(
        hc, lm_head_weight, lm_head_bias, output_multiplier, final_logit_softcapping
    )
    lse = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(dim=-1, index=yc.unsqueeze(-1)).squeeze(-1)
    return (chosen - lse) * m_sub.float()


def compute_backbone_sequence_logprobs(
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    resp_mask: torch.Tensor,
    *,
    output_multiplier: float | None = None,
    final_logit_softcapping: float | None = None,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state
    zero_anchor = hidden.sum() * 0.0

    shift_hidden = hidden[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = resp_mask[:, 1:].bool()
    seq_m1 = shift_hidden.shape[1]

    tok_logp_chunks: list[torch.Tensor] = []
    for start in range(0, seq_m1, chunk_tokens):
        end = min(seq_m1, start + chunk_tokens)
        m_sub = shift_mask[:, start:end]
        if not m_sub.any():
            tok_logp_chunks.append(torch.zeros_like(m_sub, dtype=torch.float32))
            continue
        hc = shift_hidden[:, start:end, :]
        yc = shift_targets[:, start:end]
        if torch.is_grad_enabled() and hc.requires_grad:
            lp_chunk = ckpt_fn(
                _logp_chunk_step,
                hc,
                yc,
                m_sub,
                lm_head.weight,
                lm_head.bias,
                output_multiplier,
                final_logit_softcapping,
                use_reentrant=False,
            )
        else:
            lp_chunk = _logp_chunk_step(
                hc,
                yc,
                m_sub,
                lm_head.weight,
                lm_head.bias,
                output_multiplier,
                final_logit_softcapping,
            )
        tok_logp_chunks.append(lp_chunk)

    tok_logp = torch.cat(tok_logp_chunks, dim=1) if tok_logp_chunks else torch.zeros((1, 0), device=hidden.device)
    return tok_logp, shift_mask, zero_anchor


def prepare_rlvr_groups_v4(
    rollouts_path: Path,
    tok: Any,
    spec: ModelSpecV4,
    *,
    algo: str,
    chat_kwargs: dict[str, Any],
    max_seq_len: int,
    max_gen_tokens: int,
    overlong_window: int,
    overlong_penalty: float,
    max_offpolicy_fraction: float = 0.20,
    seed: int = 20260325,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    raw_groups = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mixed_groups: list[list[dict[str, Any]]] = []
    offpolicy_rescue_groups: list[list[dict[str, Any]]] = []
    skipped_saturated = 0
    skipped_zero = 0
    all_comp_lens: list[int] = []

    for g in raw_groups:
        p_str = render_prompt_v4(tok, spec, g["messages"], chat_kwargs=chat_kwargs)
        p_ids = [int(x) for x in tok.encode(p_str, add_special_tokens=False)]
        samples = g["samples"]

        n_pos = sum(1 for s in samples if bool(s.get("verified", False)))
        has_off = any(bool(s.get("off_policy", False)) for s in samples)

        if n_pos == len(samples):
            skipped_saturated += 1
            continue
        if n_pos == 0:
            skipped_zero += 1
            continue
        if has_off and algo != "gspo_luffy":
            skipped_zero += 1
            continue

        raw_rewards: list[float] = []
        valid_entries: list[dict[str, Any]] = []
        for s in samples:
            is_off = bool(s.get("off_policy", False))
            fr = str(s.get("finish_reason", "stop"))
            raw_txt = str(s["text"])
            if not is_off and s.get("token_ids"):
                # Lossless exact generated token sequence from `eval_engine_v4` (including sampled stop token when fr=="stop")
                r_ids = [int(x) for x in s["token_ids"]]
            elif is_off:
                # Off-policy verified rescue trace: wrap cleanly in target family's turn framing
                g4_think_prefix = "<|channel>thought\n<channel|>" if (spec.channel_family == "gemma4" and chat_kwargs.get("enable_thinking")) else spec.sft_prefix
                r_str = g4_think_prefix + raw_txt.strip() + spec.sft_suffix
                r_ids = [int(x) for x in tok.encode(r_str, add_special_tokens=False)]
            elif fr == "stop":
                # Fallback string re-encoding without duplicating trailing `<|end_message|>`
                suf = spec.sft_suffix
                if spec.channel_family == "inkling2" and raw_txt.rstrip().endswith("<|end_message|>"):
                    suf = "<|content_model_end_sampling|>"
                r_str = raw_txt + suf
                r_ids = [int(x) for x in tok.encode(r_str, add_special_tokens=False)]
            else:
                # Truncated rollout (`finish_reason == "length"`): do NOT append fake stop token!
                r_ids = [int(x) for x in tok.encode(raw_txt, add_special_tokens=False)]

            f_ids = p_ids + r_ids
            was_cropped = False
            if len(f_ids) > max_seq_len:
                f_ids = f_ids[:max_seq_len]
                was_cropped = True
            comp_len = len(f_ids) - len(p_ids)
            if comp_len < 2:
                continue

            # Never assign positive verified reward to a rollout whose tail was cropped by `max_seq_len`
            ver_ok = bool(s.get("verified", False)) and (not was_cropped)
            base_r = 1.0 if ver_ok else 0.0
            pen = soft_overlong_penalty(
                int(s.get("completion_tokens", comp_len)),
                max_gen_tokens,
                overlong_window,
                overlong_penalty,
            )
            tot_r = base_r + pen
            raw_rewards.append(tot_r)
            all_comp_lens.append(comp_len)
            valid_entries.append(
                {
                    "input_ids": f_ids,
                    "prompt_len": len(p_ids),
                    "comp_len": comp_len,
                    "raw_reward": tot_r,
                    "verified": ver_ok,
                    "off_policy": is_off,
                }
            )

        if len(valid_entries) < 2:
            continue

        if has_off and algo == "gspo_luffy":
            # Keep ONLY the LUFFY off-policy positive trace + 1 negative trace for rescue
            off_items = [e for e in valid_entries if e["off_policy"] and e["verified"]]
            if not off_items:
                continue
            for e in off_items:
                e["advantage"] = 1.0
            offpolicy_rescue_groups.append(off_items[:1])
            continue

        mu = sum(raw_rewards) / float(len(raw_rewards))
        var = sum((r - mu) ** 2 for r in raw_rewards) / float(len(raw_rewards))
        std = math.sqrt(var)
        if std < 1e-6:
            continue

        for e in valid_entries:
            if algo == "drgrpo_cispo":
                # Unbiased Dr.GRPO centered advantage `2 * (R_i - mu_g)` without `1/std` inflation
                e["advantage"] = 2.0 * (e["raw_reward"] - mu)
            else:
                e["advantage"] = (e["raw_reward"] - mu) / (std + 1e-6)
        mixed_groups.append(valid_entries)

    rng = random.Random(seed)
    rng.shuffle(mixed_groups)
    rng.shuffle(offpolicy_rescue_groups)
    if algo == "gspo_luffy" and offpolicy_rescue_groups:
        if mixed_groups:
            max_off = max(1, int(round(len(mixed_groups) * max_offpolicy_fraction)))
        else:
            max_off = min(len(offpolicy_rescue_groups), 24)
        selected_off = offpolicy_rescue_groups[:max_off]
        active_groups = mixed_groups + selected_off
        rng.shuffle(active_groups)
    else:
        selected_off = []
        active_groups = mixed_groups

    mean_comp = sum(all_comp_lens) / float(max(1, len(all_comp_lens)))
    stats = {
        "algo": algo,
        "raw_groups_total": len(raw_groups),
        "mixed_onpolicy_groups": len(mixed_groups),
        "luffy_offpolicy_groups_included": len(selected_off),
        "active_groups_total": len(active_groups),
        "skipped_saturated_groups": skipped_saturated,
        "skipped_zero_groups": skipped_zero,
        "mean_comp_tokens": mean_comp,
    }
    return active_groups, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V4 Cross-Family 3-Proposal Distributed RLVR Trainer")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V4.keys()))
    ap.add_argument("--track", choices=["math", "lean"], default="math")
    ap.add_argument("--algo", required=True, choices=RLVR_ALGOS)
    ap.add_argument("--chat-kwargs-json", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--prior-fp32-deltas", default="", help="Comma-separated prior SFT FP32 factor files")
    ap.add_argument("--rollouts-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--save-fp32-deltas", default="")
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--save-milestone-fractions", default="0.50,1.00")
    ap.add_argument("--max-seq-len", type=int, default=10240)
    ap.add_argument("--max-gen-tokens", type=int, default=8192)
    ap.add_argument("--overlong-window", type=int, default=2048)
    ap.add_argument("--overlong-penalty", type=float, default=0.5)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--groups-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=28)
    ap.add_argument("--lr", type=float, default=1.0e-5)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
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
    groups, group_stats = prepare_rlvr_groups_v4(
        Path(args.rollouts_file),
        tok,
        spec,
        algo=args.algo,
        chat_kwargs=chat_kwargs,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_gen_tokens,
        overlong_window=args.overlong_window,
        overlong_penalty=args.overlong_penalty,
        seed=args.seed,
    )
    if not groups:
        if rank == 0:
            summary = {
                "status": "SKIPPED_ZERO_ACTIVE_GROUPS",
                "model_key": args.model_key,
                "track": args.track,
                "algo": args.algo,
                "chat_kwargs": chat_kwargs,
                "steps_completed": 0,
                "lr": args.lr,
                "group_stats": group_stats,
                "milestone_checkpoints": {},
            }
            sp = Path(args.summary_output)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"status": "SKIPPED_ZERO_ACTIVE_GROUPS", "algo": args.algo}), flush=True)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return 0

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
        base_model, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0
    )
    wrapped_backbone, lm_head, is_fsdp = wrap_distributed_backbone_v4(
        peft_m, spec, device, local_rank, world_size
    )

    # Flatten active samples & compute exact FP32 `old_logp` under initial policy `pi_old`!
    flat_samples: list[dict[str, Any]] = [s for g in groups for s in g]
    wrapped_backbone.eval()
    with torch.no_grad():
        for idx_s in range(rank, len(flat_samples), world_size):
            sm = flat_samples[idx_s]
            ids = torch.tensor([sm["input_ids"]], dtype=torch.long, device=device)
            am = torch.ones_like(ids)
            rm = torch.zeros_like(ids)
            rm[:, sm["prompt_len"] :] = 1
            lp, _, _ = compute_backbone_sequence_logprobs(
                wrapped_backbone,
                lm_head,
                ids,
                am,
                rm,
                output_multiplier=out_mult,
                final_logit_softcapping=softcap,
            )
            sm["old_logp_cpu"] = lp.squeeze(0).cpu()
        # Ensure dummy collective parity if len(flat_samples) % world_size != 0
        rem_s = len(flat_samples) % world_size
        if rem_s != 0 and rank >= rem_s:
            ids_d = torch.tensor([flat_samples[0]["input_ids"][:64]], dtype=torch.long, device=device)
            am_d = torch.ones_like(ids_d)
            rm_d = torch.zeros_like(ids_d)
            compute_backbone_sequence_logprobs(
                wrapped_backbone, lm_head, ids_d, am_d, rm_d, output_multiplier=out_mult, final_logit_softcapping=softcap
            )

    # Broadcast `old_logp_cpu` across ranks so shuffled PPO epochs can assign any sample to any rank
    if world_size > 1:
        local_map = {idx_s: flat_samples[idx_s]["old_logp_cpu"] for idx_s in range(rank, len(flat_samples), world_size)}
        gathered: list[dict[int, torch.Tensor]] = [{} for _ in range(world_size)]
        dist.all_gather_object(gathered, local_map)
        for gmap in gathered:
            for idx_s, t_cpu in gmap.items():
                flat_samples[idx_s]["old_logp_cpu"] = t_cpu

    wrapped_backbone.train()
    trainable_params = [p for p in wrapped_backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    steps_per_pass = max(1, math.ceil(len(groups) / max(1, args.groups_per_step)))
    total_steps = min(args.max_steps, steps_per_pass * args.ppo_epochs)
    fracs = [float(x.strip()) for x in args.save_milestone_fractions.split(",") if x.strip()]
    milestone_steps: dict[int, str] = {}
    for fr in fracs:
        if fr < 0.999:
            ms = max(1, min(total_steps - 1, int(round(total_steps * fr))))
            milestone_steps[ms] = f"pct{int(round(fr * 100))}"

    saved_intermediate_factors: list[tuple[str, Path]] = []
    norm_tokens = float(group_stats["mean_comp_tokens"])
    torch.cuda.reset_peak_memory_stats(device)
    step = 0
    step_logs: list[dict[str, Any]] = []

    for ep in range(args.ppo_epochs):
        if step >= total_steps:
            break
        rng_ep = random.Random(args.seed + 7919 * ep)
        g_order = list(range(len(groups)))
        rng_ep.shuffle(g_order)

        for g_start in range(0, len(g_order), args.groups_per_step):
            if step >= total_steps:
                break
            batch_groups = [groups[gi] for gi in g_order[g_start : g_start + args.groups_per_step]]
            batch_samples = [s for bg in batch_groups for s in bg]
            n_active = len(batch_samples)
            if n_active == 0:
                continue

            max_micro = math.ceil(n_active / float(world_size))
            rank_samples = batch_samples[rank::world_size]
            optimizer.zero_grad(set_to_none=True)

            local_loss_sum_val = 0.0
            clip_fracs: list[float] = []
            ratios: list[float] = []

            for m_i in range(max_micro):
                if m_i < len(rank_samples):
                    sm = rank_samples[m_i]
                    ids = torch.tensor([sm["input_ids"]], dtype=torch.long, device=device)
                    am = torch.ones_like(ids)
                    rm = torch.zeros_like(ids)
                    rm[:, sm["prompt_len"] :] = 1
                    tok_lp, s_mask, z_anc = compute_backbone_sequence_logprobs(
                        wrapped_backbone,
                        lm_head,
                        ids,
                        am,
                        rm,
                        output_multiplier=out_mult,
                        final_logit_softcapping=softcap,
                    )
                    old_lp = sm["old_logp_cpu"].to(device=device).unsqueeze(0)
                    seq_l, st = compute_sample_rlvr_loss(
                        args.algo,
                        tok_lp,
                        old_lp,
                        s_mask,
                        advantage=float(sm["advantage"]),
                        verified_positive=bool(sm["verified"]),
                        is_off_policy=bool(sm["off_policy"]),
                        norm_tokens=norm_tokens,
                    )
                    local_loss_sum_val += float(seq_l.detach().item())
                    clip_fracs.append(st["clipped_frac"])
                    ratios.append(st["mean_ratio"])
                    micro_loss = seq_l * (float(world_size) / float(n_active)) + z_anc
                    micro_loss.backward()
                else:
                    ids_d = torch.tensor([flat_samples[0]["input_ids"][:64]], dtype=torch.long, device=device)
                    am_d = torch.ones_like(ids_d)
                    rm_d = torch.zeros_like(ids_d)
                    _, _, z_anc = compute_backbone_sequence_logprobs(
                        wrapped_backbone,
                        lm_head,
                        ids_d,
                        am_d,
                        rm_d,
                        output_multiplier=out_mult,
                        final_logit_softcapping=softcap,
                    )
                    z_anc.backward()

            gn = clip_distributed_grad_norm(wrapped_backbone, is_fsdp=is_fsdp, max_norm=1.0)
            optimizer.step()
            step += 1

            mean_loss_t = torch.tensor(local_loss_sum_val / float(n_active), device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(mean_loss_t, op=dist.ReduceOp.SUM)
            if rank == 0 and (step == 1 or step % 4 == 0 or step == total_steps):
                rec = {
                    "step": step,
                    "ppo_epoch": ep + 1,
                    "loss": float(mean_loss_t.item()),
                    "grad_norm": gn,
                    "clipped_frac_rank0": sum(clip_fracs) / max(1, len(clip_fracs)),
                    "mean_ratio_rank0": sum(ratios) / max(1, len(ratios)),
                }
                step_logs.append(rec)
                print(json.dumps(rec), flush=True)

            if step in milestone_steps:
                tag = milestone_steps[step]
                fac_mid = extract_lora_factors_fp32(peft_m, wrapped_backbone, is_fsdp=is_fsdp, rank=rank)
                if rank == 0:
                    mid_fac_path = Path(args.output_dir).parent / f"{Path(args.output_dir).name}_{tag}_factors.safetensors"
                    save_safetensors(fac_mid, str(mid_fac_path))
                    saved_intermediate_factors.append((tag, mid_fac_path))
                if world_size > 1:
                    dist.barrier()

    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    train_wall_s = time.monotonic() - t0

    parity_samples = [{"input_ids": s["input_ids"]} for s in flat_samples[:4]]
    out_dir_final = Path(f"{args.output_dir}_pct100")
    parity_final = consolidate_and_verify_fp32_checkpoint(
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
        out_dir=out_dir_final,
        parity_samples=parity_samples,
    )

    if rank == 0:
        prior_packs = [
            load_safetensors(p.strip(), device="cpu")
            for p in args.prior_fp32_deltas.split(",")
            if p.strip()
        ]
        milestone_dirs: dict[str, str] = {"pct100": str(out_dir_final)}
        for tag, mid_fac_p in saved_intermediate_factors:
            mid_dir = Path(f"{args.output_dir}_{tag}")
            packs_mid = list(prior_packs) + [load_safetensors(str(mid_fac_p), device="cpu")]
            merge_factor_packs_to_checkpoint_rank0(spec, orig_base_path, packs_mid, mid_dir, device)
            milestone_dirs[tag] = str(mid_dir)

        summary = {
            "status": "RLVR_TRAINED_AND_CONSOLIDATED",
            "model_key": args.model_key,
            "track": args.track,
            "algo": args.algo,
            "chat_kwargs": chat_kwargs,
            "steps_completed": step,
            "lr": args.lr,
            "adapted_linear_modules": n_adapted,
            "applied_prior_factor_tensors": applied_prior,
            "group_stats": group_stats,
            "milestone_checkpoints": milestone_dirs,
            "merge_parity_pct100": parity_final,
            "peak_vram_gb_rank0": peak_vram_gb,
            "train_wall_seconds": train_wall_s,
            "step_logs": step_logs,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "OK", "algo": args.algo, "milestones": milestone_dirs, "parity": parity_final}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Phase V3 Unified 3-Proposal Multi-Epoch Distributed RLVR Trainer (`DDP` + `27B` `FSDP FULL_SHARD`).

Implements three scale-matched post-SFT RLVR algorithms under identical rollout budgets and
exact trainer-computed `FP32` `pi_old` across `ppo_epochs = 2`:
  - `dapo_asymclip` (`R1`): Dynamic variance-filtered (`1 <= c <= 7`) token-level asymmetric PPO clip
    `[1 - 0.20, 1 + 0.28]` with group-std normalized advantage and soft-overlong length shaping.
  - `drgrpo_cispo` (`R2`): Unbiased `Dr.GRPO` centered advantage `2 * (R_i - mu_g)` (eliminating `1/sigma_g`
    near-saturated outlier inflation) + constant batch `L_norm` denominator (eliminating `1/|y_i|` length bias)
    + `CISPO` detached clipped-IS weight `sg(clip(r_{i,t}, 1 - 0.20, 1 + 0.28))` + `0.05` verified-positive
    NLL anchor strictly on `1 <= c <= 7` mixed groups.
  - `gspo_luffy` (`R3`): Sequence-level geometric importance ratio `s_i = exp(mean_t log(pi / pi_old))`
    with sequence-scale asymmetric clip `[1 - 3e-4, 1 + 4e-4]` (`GSPO`) on `1 <= c <= 7` groups +
    corpus-reference `LUFFY` `sg(pi / (pi + 0.10))` off-policy shaping (`A = +1.0`, capped at `<= 20%`
    of active groups) on `0/8` unsolved groups.

Supports `--save-milestone-fractions "0.50,0.75,1.00"` so order-invariant `select-search-winner`
evaluates rollout-matched checkpoints on disjoint `Search_V3`.
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
from torch.utils.checkpoint import checkpoint as ckpt_fn
from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v3 import MODEL_REGISTRY_V3, ModelSpecV3  # noqa: E402
from train_sft_v3 import (  # noqa: E402
    apply_prior_fp32_factors_inplace,
    attach_verified_lora_v3,
    chat_prompt_ids,
    clip_distributed_grad_norm,
    consolidate_and_verify_fp32_checkpoint,
    extract_lora_factors_fp32,
    get_backbone_and_head,
    init_dist,
    load_hf_model_staggered,
    merge_factor_packs_to_checkpoint_rank0,
    wrap_distributed_backbone,
)
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText  # noqa: E402
import shutil  # noqa: E402


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
    """
    Pure per-sequence RLVR loss computation (differentiable w.r.t. `tok_logp`), normalized so
    every arm has matched O(1) per-sequence gradient magnitude at `p = 0.5`.

    Returns `(seq_loss, stats_dict)` where summing `seq_loss` over a batch of `B_eff` active
    sequences and dividing by `B_eff` produces the batch objective.
    """
    m = shift_mask.float()
    comp_len = m.sum().clamp_min(1.0)
    adv_t = torch.tensor(float(advantage), dtype=torch.float32, device=tok_logp.device)
    denom = max(float(norm_tokens), 1.0)

    if is_off_policy:
        # LUFFY off-policy rescue (`sg(pi / (pi + gamma)) * A_off * log pi`), outside on-policy ratio
        pi_prob = torch.exp(tok_logp.detach().clamp(min=-20.0, max=0.0))
        w_luffy = (pi_prob / (pi_prob + luffy_gamma)).detach()
        seq_loss = -((w_luffy * adv_t * tok_logp) * m).sum() / comp_len
        return seq_loss, {"clipped_frac": 0.0, "mean_ratio": float(w_luffy[shift_mask.bool()].mean().item()) if shift_mask.any() else 0.0}

    log_ratio = (tok_logp - old_logp.detach()) * m

    if algo == "dapo_asymclip":
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        clipped_r = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
        surr1 = ratio * adv_t
        surr2 = clipped_r * adv_t
        tok_obj = torch.minimum(surr1, surr2) * m
        # Normalized by batch mean completion length `denom` so total batch denominator == active tokens
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
        # Sequence-level geometric importance ratio `s_i = exp((1/|y_i|) sum_t log(pi / pi_old))`
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
) -> torch.Tensor:
    logits = F.linear(hc, lm_head_weight, lm_head_bias).float()
    lse = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(dim=-1, index=yc.unsqueeze(-1)).squeeze(-1)
    return (chosen - lse) * m_sub.float()


def compute_backbone_sequence_logprobs(
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    resp_mask: torch.Tensor,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state
    zero_anchor = hidden.sum() * 0.0

    shift_h = hidden[:, :-1, :]
    shift_y = input_ids[:, 1:]
    shift_m = resp_mask[:, 1:]
    bsz, seq_m1, _ = shift_h.shape

    out_logp: list[torch.Tensor] = []
    for start in range(0, seq_m1, chunk_tokens):
        end = min(seq_m1, start + chunk_tokens)
        m_sub = shift_m[:, start:end]
        if not m_sub.any():
            out_logp.append(torch.zeros((bsz, end - start), device=hidden.device, dtype=torch.float32))
            continue
        hc = shift_h[:, start:end, :]
        yc = shift_y[:, start:end]
        if torch.is_grad_enabled() and hc.requires_grad:
            lp = ckpt_fn(_logp_chunk_step, hc, yc, m_sub, lm_head.weight, lm_head.bias, use_reentrant=False)
        else:
            lp = _logp_chunk_step(hc, yc, m_sub, lm_head.weight, lm_head.bias)
        out_logp.append(lp)

    return torch.cat(out_logp, dim=1), shift_m, zero_anchor


def prepare_rlvr_samples_v3(
    rollouts_path: Path,
    tok: Any,
    spec: ModelSpecV3,
    *,
    algo: str,
    thinking: bool,
    max_seq_len: int,
    max_gen_tokens: int,
    overlong_window: int,
    overlong_penalty: float,
    max_offpolicy_group_frac: float = 0.20,
    seed: int = 20260325,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in raw:
        by_pid[str(r["id"])].append(r)

    onpolicy_groups: list[list[dict[str, Any]]] = []
    offpolicy_candidates: list[dict[str, Any]] = []
    n_groups_total = len(by_pid)
    n_all_zero = 0
    n_all_solved = 0
    stop_set = set(spec.stop_token_ids)
    comp_lengths: list[int] = []
    shaped_rewards_all: list[float] = []

    for pid, grows in sorted(by_pid.items()):
        on_rows = [r for r in grows if not bool(r.get("is_off_policy", False))]
        off_rows = [r for r in grows if bool(r.get("is_off_policy", False))]

        tok_on: list[dict[str, Any]] = []
        n_verified_on = 0
        for r in on_rows:
            p_ids = chat_prompt_ids(tok, spec, r["prompt_messages"], thinking=thinking)
            r_ids = [int(x) for x in tok.encode(str(r["response_text"]), add_special_tokens=False)]
            stopped = str(r.get("finish_reason", "stop")) == "stop"
            while r_ids and r_ids[-1] in stop_set:
                r_ids.pop()
            if stopped:
                r_ids = r_ids + [spec.sup_eos_id]
            full_ids = p_ids + r_ids
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
                stopped = False
                verified = False
            else:
                verified = bool(r.get("verified_ok", False)) and stopped

            if verified:
                n_verified_on += 1
            comp_len = max(1, len(full_ids) - len(p_ids))
            comp_lengths.append(comp_len)
            resp_mask = [0] * len(p_ids) + [1] * comp_len
            base_rew = 1.0 if verified else 0.0
            pen = (
                -overlong_penalty
                if not stopped
                else soft_overlong_penalty(comp_len, max_gen_tokens, overlong_window, overlong_penalty)
            )
            rew = base_rew + pen
            shaped_rewards_all.append(rew)
            tok_on.append(
                {
                    "problem_id": pid,
                    "input_ids": full_ids,
                    "resp_mask": resp_mask,
                    "comp_len": comp_len,
                    "reward": rew,
                    "verified_positive": verified,
                    "is_off_policy": False,
                }
            )

        k_on = len(tok_on)
        if k_on == 0:
            continue
        if n_verified_on == k_on:
            n_all_solved += 1
            continue
        if n_verified_on == 0:
            n_all_zero += 1
            # Only `gspo_luffy` uses corpus-reference off-policy 0/8 rescue
            if algo == "gspo_luffy" and off_rows:
                r_off = off_rows[0]
                p_ids = chat_prompt_ids(tok, spec, r_off["prompt_messages"], thinking=thinking)
                r_ids = [int(x) for x in tok.encode(str(r_off["response_text"]), add_special_tokens=False)]
                while r_ids and r_ids[-1] in stop_set:
                    r_ids.pop()
                r_ids = r_ids + [spec.sup_eos_id]
                full_ids = p_ids + r_ids
                if len(full_ids) <= max_seq_len and len(r_ids) >= 2:
                    comp_len = len(r_ids)
                    offpolicy_candidates.append(
                        {
                            "problem_id": pid,
                            "input_ids": full_ids,
                            "resp_mask": [0] * len(p_ids) + [1] * comp_len,
                            "comp_len": comp_len,
                            "advantage": 1.0,  # Capped unit advantage (prevents +2.65 std-inflation)
                            "verified_positive": True,
                            "is_off_policy": True,
                        }
                    )
            continue

        # Mixed group (`1 <= n_verified_on <= k_on - 1`)
        rews = [x["reward"] for x in tok_on]
        mean_r = sum(rews) / float(k_on)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in rews) / float(k_on))
        if std_r < 1e-6:
            continue

        group_out: list[dict[str, Any]] = []
        for x in tok_on:
            if algo == "drgrpo_cispo":
                # Scale-matched centered advantage: equals +/-1.0 at p=0.5 without 1/sigma_g outlier blowup
                adv = 2.0 * (x["reward"] - mean_r)
            else:
                adv = (x["reward"] - mean_r) / (std_r + 1e-6)
            item_out = dict(x)
            item_out["advantage"] = float(adv)
            group_out.append(item_out)
        onpolicy_groups.append(group_out)

    samples: list[dict[str, Any]] = [s for g in onpolicy_groups for s in g]
    n_off_kept = 0
    if algo == "gspo_luffy" and offpolicy_candidates and onpolicy_groups:
        max_off = max(1, int(round(max_offpolicy_group_frac * len(onpolicy_groups))))
        rng = random.Random(seed)
        shuffled_off = list(offpolicy_candidates)
        rng.shuffle(shuffled_off)
        kept_off = shuffled_off[:max_off]
        samples.extend(kept_off)
        n_off_kept = len(kept_off)

    mean_comp_len = (sum(s["comp_len"] for s in samples) / float(len(samples))) if samples else 1024.0
    return samples, {
        "algo": algo,
        "groups_total": n_groups_total,
        "groups_mixed_active": len(onpolicy_groups),
        "groups_all_zero": n_all_zero,
        "groups_all_solved": n_all_solved,
        "off_policy_rescued_groups": n_off_kept,
        "active_rl_samples": len(samples),
        "mean_comp_len_norm": round(mean_comp_len, 2),
        "mean_shaped_reward": round(sum(shaped_rewards_all) / max(len(shaped_rewards_all), 1), 4),
    }


@torch.no_grad()
def precompute_trainer_old_logprobs(
    samples: list[dict[str, Any]],
    wrapped_backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    *,
    pad_id: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """
    Computes exact `FP32` `old_logp` inside the trainer before optimization begins,
    avoiding vLLM-vs-FSDP numerical drift on step 1 while enabling genuine `ppo_epochs >= 2`
    importance ratios on subsequent updates.
    """
    was_training = wrapped_backbone.training
    wrapped_backbone.eval()

    n = len(samples)
    rem = n % world_size
    total_padded = n if rem == 0 else (n + world_size - rem)

    local_results: dict[int, torch.Tensor] = {}
    for base in range(0, total_padded, world_size):
        idx = base + rank
        if idx < n and not samples[idx]["is_off_policy"]:
            s = samples[idx]
            ids_t = torch.tensor([s["input_ids"]], dtype=torch.long, device=device)
            mask_t = torch.ones_like(ids_t)
            rmask_t = torch.tensor([s["resp_mask"]], dtype=torch.long, device=device)
            lp, _, _ = compute_backbone_sequence_logprobs(wrapped_backbone, lm_head, ids_t, mask_t, rmask_t)
            local_results[idx] = lp[0].detach().cpu()
        else:
            ids_t = torch.tensor([[pad_id] * 16], dtype=torch.long, device=device)
            mask_t = torch.ones_like(ids_t)
            rmask_t = torch.zeros_like(ids_t)
            _, _, _ = compute_backbone_sequence_logprobs(wrapped_backbone, lm_head, ids_t, mask_t, rmask_t)

    gathered: list[dict[int, torch.Tensor]] = [{} for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]

    merged: dict[int, torch.Tensor] = {}
    for d in gathered:
        merged.update(d)

    for idx, s in enumerate(samples):
        if s["is_off_policy"]:
            s["old_logp_cpu"] = torch.zeros(len(s["input_ids"]) - 1, dtype=torch.float32)
        else:
            s["old_logp_cpu"] = merged[idx]

    if was_training:
        wrapped_backbone.train()


@torch.no_grad()
def export_intermediate_milestone_checkpoint(
    peft_model: torch.nn.Module,
    wrapped_backbone: torch.nn.Module,
    spec: ModelSpecV3,
    *,
    is_fsdp: bool,
    rank: int,
    world_size: int,
    orig_base_model_path: Path,
    prior_deltas_path: str,
    milestone_dir: Path,
    save_deltas_path: str = "",
) -> None:
    """Exports an intermediate milestone checkpoint without destroying the live distributed training graph."""
    current_deltas = extract_lora_deltas_fp32(peft_model, wrapped_backbone, is_fsdp=is_fsdp, rank=rank)
    if rank == 0:
        total_deltas: dict[str, torch.Tensor] = {}
        if prior_deltas_path:
            prior = load_safetensors(prior_deltas_path, device="cpu")
            for k, v in prior.items():
                total_deltas[k] = v.float().clone()
        for k, v in current_deltas.items():
            total_deltas[k] = (total_deltas[k] + v) if k in total_deltas else v

        if save_deltas_path:
            sp = Path(save_deltas_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            save_safetensors(total_deltas, str(sp))

        cls = AutoModelForImageTextToText if spec.loader_kind == "image_text" else AutoModelForCausalLM
        clean_model = cls.from_pretrained(
            str(orig_base_model_path),
            dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        sd = clean_model.state_dict()
        for w_key, delta_fp32 in total_deltas.items():
            orig_t = sd[w_key]
            sd[w_key].copy_((orig_t.float() + delta_fp32.float()).to(orig_t.dtype))

        milestone_dir.mkdir(parents=True, exist_ok=True)
        clean_model.save_pretrained(str(milestone_dir), safe_serialization=True)
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
                shutil.copy2(extra_fn, milestone_dir / extra_fn.name)
        del clean_model

    if world_size > 1:
        dist.barrier()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V3 Unified 3-Proposal Distributed RLVR Trainer")
    ap.add_argument("--algo", required=True, choices=RLVR_ALGOS)
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V3.keys()))
    ap.add_argument("--base-model", default="", help="Original untouched base model path")
    ap.add_argument("--prior-fp32-deltas", default="", help="Stage-II cumulative FP32 LoRA deltas .safetensors")
    ap.add_argument("--save-fp32-deltas", default="")
    ap.add_argument("--rollouts-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--milestones-root", default="", help="Optional root dir for 50%/75% intermediate checkpoints")
    ap.add_argument("--save-milestone-fractions", default="0.50,0.75")
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=13312)
    ap.add_argument("--max-gen-tokens", type=int, default=12288)
    ap.add_argument("--overlong-window", type=int, default=2288)
    ap.add_argument("--overlong-penalty", type=float, default=0.25)
    ap.add_argument("--eps-low", type=float, default=0.20)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--gspo-eps-low", type=float, default=3e-4)
    ap.add_argument("--gspo-eps-high", type=float, default=4e-4)
    ap.add_argument("--cispo-sft-alpha", type=float, default=0.05)
    ap.add_argument("--luffy-gamma", type=float, default=0.10)
    ap.add_argument("--max-offpolicy-group-frac", type=float, default=0.20)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
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

    samples, rl_audit = prepare_rlvr_samples_v3(
        Path(args.rollouts_file),
        tok,
        spec,
        algo=args.algo,
        thinking=args.thinking,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_gen_tokens,
        overlong_window=args.overlong_window,
        overlong_penalty=args.overlong_penalty,
        max_offpolicy_group_frac=args.max_offpolicy_group_frac,
        seed=args.seed,
    )
    if rank == 0:
        print(json.dumps({"event": "RLVR_AUDIT", "model_key": spec.key, **rl_audit}), flush=True)
    if not samples:
        raise RuntimeError(f"No active RLVR samples after variance filtering: {rl_audit}")

    base_m = load_hf_model_staggered(
        spec, orig_base_path, rank=rank, world_size=world_size, flash_attn=True, drop_visual_for_train=True
    )
    applied_prior = 0
    if args.prior_fp32_deltas:
        applied_prior = apply_prior_fp32_factors_inplace(base_m, args.prior_fp32_deltas, device)
        if world_size > 1:
            dist.barrier()

    peft_m, n_adapted = attach_verified_lora_v3(
        base_m, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0
    )
    wrapped_backbone, lm_head, is_fsdp = wrap_distributed_backbone(
        peft_m, spec, device, local_rank, world_size
    )

    pad_id = spec.stop_token_ids[0]
    precompute_trainer_old_logprobs(
        samples, wrapped_backbone, lm_head, pad_id=pad_id, device=device, rank=rank, world_size=world_size
    )

    grad_accum = max(1, args.global_batch_size // world_size)
    microbatches_per_step = grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(samples) / microbatches_per_step))
    total_steps = steps_per_epoch * args.ppo_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    milestone_step_to_tag: dict[int, str] = {}
    if args.milestones_root and args.save_milestone_fractions.strip():
        for frac_s in args.save_milestone_fractions.split(","):
            frac_s = frac_s.strip()
            if not frac_s:
                continue
            fval = float(frac_s)
            if 0.0 < fval < 1.0:
                mstep = max(1, min(total_steps - 1, int(round(fval * total_steps))))
                milestone_step_to_tag[mstep] = f"pct{int(round(fval * 100)):02d}"

    optimizer = torch.optim.AdamW(
        [p for p in wrapped_backbone.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    dummy_sample = {
        "input_ids": [pad_id] * 16,
        "resp_mask": [0] * 16,
        "old_logp_cpu": torch.zeros(15, dtype=torch.float32),
        "advantage": 0.0,
        "verified_positive": False,
        "is_off_policy": False,
        "is_dummy": True,
    }

    norm_tokens = float(rl_audit["mean_comp_len_norm"])
    step_losses: list[float] = []
    clip_pos_hist: list[float] = []
    clip_neg_hist: list[float] = []
    saved_milestones: dict[str, str] = {}
    global_step = 0
    peak_mem_gb = 0.0

    for ep in range(args.ppo_epochs):
        if global_step >= total_steps:
            break
        rng = random.Random(args.seed + 7919 * ep)
        order = list(range(len(samples)))
        rng.shuffle(order)
        padded: list[dict[str, Any]] = [samples[idx] for idx in order]
        rem = len(padded) % microbatches_per_step
        if rem != 0:
            padded.extend([dummy_sample] * (microbatches_per_step - rem))

        for step_base in range(0, len(padded), microbatches_per_step):
            if global_step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)

            # Count real active sequences across all ranks in this optimizer step
            active_in_step = sum(
                1
                for idx_chk in range(step_base, step_base + microbatches_per_step)
                if not padded[idx_chk].get("is_dummy", False)
            )
            denom_sequences = float(max(1, active_in_step))
            step_loss_local_sum = 0.0
            pos_clip_sum = 0.0
            pos_cnt = 0.0
            neg_clip_sum = 0.0
            neg_cnt = 0.0

            for ga in range(grad_accum):
                item = padded[step_base + ga * world_size + rank]
                ids_t = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                mask_t = torch.ones_like(ids_t)
                rmask_t = torch.tensor([item["resp_mask"]], dtype=torch.long, device=device)

                tok_logp, shift_m, zero_anchor = compute_backbone_sequence_logprobs(
                    wrapped_backbone, lm_head, ids_t, mask_t, rmask_t
                )
                if item.get("is_dummy", False):
                    loss = zero_anchor
                else:
                    old_lp = item["old_logp_cpu"].to(device)
                    seq_loss, st = compute_sample_rlvr_loss(
                        args.algo,
                        tok_logp[0],
                        old_lp,
                        shift_m[0],
                        advantage=float(item["advantage"]),
                        verified_positive=bool(item["verified_positive"]),
                        is_off_policy=bool(item["is_off_policy"]),
                        norm_tokens=norm_tokens,
                        eps_low=args.eps_low,
                        eps_high=args.eps_high,
                        gspo_eps_low=args.gspo_eps_low,
                        gspo_eps_high=args.gspo_eps_high,
                        cispo_sft_alpha=args.cispo_sft_alpha,
                        luffy_gamma=args.luffy_gamma,
                    )
                    loss = (seq_loss * float(world_size)) / denom_sequences + zero_anchor
                    step_loss_local_sum += float(seq_loss.detach().item())
                    if float(item["advantage"]) > 0:
                        pos_clip_sum += st["clipped_frac"]
                        pos_cnt += 1.0
                    elif float(item["advantage"]) < 0:
                        neg_clip_sum += st["clipped_frac"]
                        neg_cnt += 1.0

                loss.backward()

            gnorm = clip_distributed_grad_norm(wrapped_backbone, is_fsdp=is_fsdp, max_norm=1.0)
            optimizer.step()
            global_step += 1

            agg_t = torch.tensor(
                [step_loss_local_sum, pos_clip_sum, pos_cnt, neg_clip_sum, neg_cnt],
                dtype=torch.float32,
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(agg_t, op=dist.ReduceOp.SUM)
            mean_step_loss = float(agg_t[0].item()) / denom_sequences
            c_pos = float(agg_t[1].item()) / max(1.0, float(agg_t[2].item()))
            c_neg = float(agg_t[3].item()) / max(1.0, float(agg_t[4].item()))
            step_losses.append(mean_step_loss)
            clip_pos_hist.append(c_pos)
            clip_neg_hist.append(c_neg)
            peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated(device) / (1024.0 ** 3))

            if rank == 0 and (global_step == 1 or global_step % 5 == 0 or global_step == total_steps):
                print(
                    json.dumps(
                        {
                            "rlvr_step": global_step,
                            "total_steps": total_steps,
                            "ppo_epoch": ep + 1,
                            "algo": args.algo,
                            "loss": round(mean_step_loss, 5),
                            "grad_norm": round(gnorm, 4),
                            "clip_frac_pos": round(c_pos, 4),
                            "clip_frac_neg": round(c_neg, 4),
                            "peak_hbm_gb": round(peak_mem_gb, 2),
                        }
                    ),
                    flush=True,
                )

            if global_step in milestone_step_to_tag:
                tag = milestone_step_to_tag[global_step]
                # Save compact <1GB FP32 low-rank factors during training; materialize merged model after destroy_process_group!
                fac_mid = extract_lora_factors_fp32(
                    peft_m, wrapped_backbone, is_fsdp=is_fsdp, rank=rank
                )
                if rank == 0:
                    mdir = Path(args.milestones_root) / tag
                    mdir.mkdir(parents=True, exist_ok=True)
                    save_safetensors(fac_mid, str(mdir / "stage_fp32_factors.safetensors"))
                    saved_milestones[tag] = mdir.as_posix()
                if world_size > 1:
                    dist.barrier()

    parity_samples = [{"input_ids": s["input_ids"]} for s in samples[: min(6, len(samples))]]
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
    if rank == 0 and saved_milestones:
        prior_packs = [
            load_safetensors(p_str.strip(), device="cpu")
            for p_str in args.prior_fp32_deltas.split(",")
            if p_str.strip()
        ]
        for tag, mdir_s in list(saved_milestones.items()):
            mdir_p = Path(mdir_s)
            fac_file = mdir_p / "stage_fp32_factors.safetensors"
            if fac_file.exists():
                cur_pack = load_safetensors(str(fac_file), device="cpu")
                merge_factor_packs_to_checkpoint_rank0(
                    spec, orig_base_path, [*prior_packs, cur_pack], mdir_p, device
                )
                fac_file.unlink(missing_ok=True)
    saved_milestones["pct100"] = Path(args.output_dir).as_posix()

    if rank == 0:
        summary = {
            "status": "RLVR_TRAINED_AND_MERGED_V3",
            "algo": args.algo,
            "model_key": spec.key,
            "dist_strategy": "fsdp_full_shard" if is_fsdp else "ddp",
            "world_size": world_size,
            "adapted_linear_modules": n_adapted,
            "applied_prior_fp32_deltas": applied_prior,
            "ppo_epochs": args.ppo_epochs,
            "steps_executed": global_step,
            "lr": args.lr,
            "mean_loss": sum(step_losses) / max(len(step_losses), 1),
            "mean_clip_frac_pos": sum(clip_pos_hist) / max(len(clip_pos_hist), 1),
            "mean_clip_frac_neg": sum(clip_neg_hist) / max(len(clip_neg_hist), 1),
            "peak_hbm_gb_rank0": round(peak_mem_gb, 3),
            "wall_seconds": round(time.monotonic() - t0, 2),
            "saved_milestones": saved_milestones,
            "rl_audit": rl_audit,
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

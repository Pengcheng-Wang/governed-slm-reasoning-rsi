#!/usr/bin/env python3
"""
Multi-Architecture Full-12K Activation-Checkpointed Chunked-Logits `DAPO + LUFFY` RLVR Trainer V2.

Fixes 9B/27B long-sequence logit OOMs using activation-checkpointed 1,024-token
`lm_head` log-softmax chunks (`< 30 GB` peak HBM on A100-80GB at full 12,800-token sequences)
with a linear 10,000->12,288 soft-overlong termination penalty + truncated IS cap.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as ckpt_fn
from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v2 import MODEL_REGISTRY_V2, ModelSpecV2  # noqa: E402
from train_slm_v2 import (  # noqa: E402
    _init_dist,
    attach_verified_lora,
    chat_prompt_ids,
    load_hf_model,
    merge_lora_fp32_and_verify_parity_v2,
    unwrap_backbone_and_head,
)


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


def compute_sequence_token_logprobs_v2(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    resp_mask: torch.Tensor,
    chunk_tokens: int = 1024,
) -> torch.Tensor:
    lm_backbone, lm_head = unwrap_backbone_and_head(model)
    hidden = lm_backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
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

    return torch.cat(out_logp, dim=1)


class _ChunkedLogprobForwardV2(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        resp_mask: torch.Tensor,
    ) -> torch.Tensor:
        return compute_sequence_token_logprobs_v2(self.model, input_ids, attention_mask, resp_mask)


def soft_overlong_penalty(comp_tokens: int, max_tokens: int, cache_window: int, max_pen: float) -> float:
    if cache_window <= 0 or max_pen <= 0.0:
        return 0.0
    thresh = max(1, max_tokens - cache_window)
    if comp_tokens <= thresh:
        return 0.0
    if comp_tokens >= max_tokens:
        return -max_pen
    return -max_pen * float(comp_tokens - thresh) / float(cache_window)


def prepare_rl_groups_v2(
    rollouts_path: Path,
    tok: Any,
    spec: ModelSpecV2,
    *,
    thinking: bool,
    max_seq_len: int,
    max_gen_tokens: int,
    overlong_window: int,
    overlong_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in raw:
        by_pid[str(r["id"])].append(r)

    samples: list[dict[str, Any]] = []
    n_groups_total = len(by_pid)
    n_dead_groups = 0
    n_offpolicy_injected = 0
    rewards_all: list[float] = []
    stop_set = set(spec.stop_token_ids)

    for pid, grows in sorted(by_pid.items()):
        g_rewards: list[float] = []
        tokenized_group: list[tuple[list[int], list[int], bool]] = []

        for r in grows:
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

            comp_len = max(1, len(full_ids) - len(p_ids))
            resp_mask = [0] * len(p_ids) + [1] * comp_len
            is_off = bool(r.get("is_off_policy", False))

            base_rew = 1.0 if verified else 0.0
            pen = -overlong_penalty if not stopped else soft_overlong_penalty(comp_len, max_gen_tokens, overlong_window, overlong_penalty)
            rew = base_rew + pen

            g_rewards.append(rew)
            rewards_all.append(rew)
            tokenized_group.append((full_ids, resp_mask, is_off))

        mean_r = sum(g_rewards) / max(len(g_rewards), 1)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in g_rewards) / max(len(g_rewards), 1))
        if std_r < 1e-5:
            n_dead_groups += 1
            continue

        for (f_ids, r_mask, is_off), rew in zip(tokenized_group, g_rewards):
            adv = (rew - mean_r) / (std_r + 1e-6)
            if is_off:
                if adv <= 0.0:
                    continue
                n_offpolicy_injected += 1
            samples.append(
                {
                    "problem_id": pid,
                    "input_ids": f_ids,
                    "resp_mask": r_mask,
                    "advantage": float(adv),
                    "is_off_policy": is_off,
                }
            )

    return samples, {
        "groups_total": n_groups_total,
        "dead_groups_skipped": n_dead_groups,
        "dead_group_fraction": n_dead_groups / max(n_groups_total, 1),
        "active_rl_samples": len(samples),
        "off_policy_positive_samples": n_offpolicy_injected,
        "mean_shaped_reward": sum(rewards_all) / max(len(rewards_all), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-Architecture Full-12K Checkpointed Chunked DAPO+LUFFY V2")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V2.keys()))
    ap.add_argument("--init-model", required=True)
    ap.add_argument("--rollouts-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=13312)
    ap.add_argument("--max-gen-tokens", type=int, default=12288)
    ap.add_argument("--overlong-window", type=int, default=2288)
    ap.add_argument("--overlong-penalty", type=float, default=0.25)
    ap.add_argument("--luffy-gamma", type=float, default=0.1)
    ap.add_argument("--eps-low", type=float, default=0.20)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--global-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261118)
    args = ap.parse_args()

    rank, world_size, local_rank = _init_dist()
    device = torch.device(f"cuda:{local_rank}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.monotonic()
    spec = MODEL_REGISTRY_V2[args.model_key]
    init_path = Path(args.init_model)
    tok = AutoTokenizer.from_pretrained(str(init_path), trust_remote_code=True)
    samples, group_stats = prepare_rl_groups_v2(
        Path(args.rollouts_file),
        tok,
        spec,
        thinking=args.thinking,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_gen_tokens,
        overlong_window=args.overlong_window,
        overlong_penalty=args.overlong_penalty,
    )
    if len(samples) < world_size:
        raise RuntimeError(f"Insufficient active RL samples ({len(samples)}): {group_stats}")
    if rank == 0:
        print(json.dumps({"event": "RL_GROUPS_READY", "model_key": spec.key, **group_stats}), flush=True)

    base_m = load_hf_model(spec, init_path, flash_attn=True)
    model, n_adapted = attach_verified_lora(base_m, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0)
    model.to(device)

    # Precompute exact old_logp in-trainer under identical RoPE/LongRoPE scaling (Opus 5.5 Max safeguard)
    model.eval()
    with torch.no_grad():
        for idx in range(rank, len(samples), world_size):
            s = samples[idx]
            ids_t = torch.tensor([s["input_ids"]], dtype=torch.long, device=device)
            mask_t = torch.ones_like(ids_t)
            rmask_t = torch.tensor([s["resp_mask"]], dtype=torch.long, device=device)
            lp = compute_sequence_token_logprobs_v2(model, ids_t, mask_t, rmask_t)
            s["old_logp"] = lp[0].half().cpu()
            del ids_t, mask_t, rmask_t, lp

    if world_size > 1:
        local_pairs = [(idx, samples[idx]["old_logp"]) for idx in range(rank, len(samples), world_size)]
        gathered: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, local_pairs)
        for plist in gathered:
            for idx, olp in plist:
                samples[idx]["old_logp"] = olp

    torch.cuda.empty_cache()
    model.train()
    fwd_mod = _ChunkedLogprobForwardV2(model)
    ddp_model: torch.nn.Module = (
        DDP(fwd_mod, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
        if world_size > 1
        else fwd_mod
    )
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    grad_accum = max(1, args.global_batch_size // world_size)
    per_step = world_size * grad_accum
    global_step = 0
    step_losses: list[float] = []

    for ep in range(args.ppo_epochs):
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
        rng = random.Random(args.seed + 7919 * ep)
        ep_rows = list(samples)
        rng.shuffle(ep_rows)
        rem = len(ep_rows) % per_step
        if rem != 0:
            ep_rows.extend(ep_rows[: (per_step - rem)])

        for s_idx in range(0, len(ep_rows), per_step):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            grp = ep_rows[s_idx : s_idx + per_step]
            grp.sort(key=lambda x: len(x["input_ids"]), reverse=True)
            r_slice = grp[rank::world_size]

            local_tok_cnt = sum(sum(x["resp_mask"][1:]) for x in r_slice)
            denom_t = torch.tensor(float(local_tok_cnt), device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(denom_t, op=dist.ReduceOp.SUM)
            global_toks = torch.clamp(denom_t, min=1.0)

            optimizer.zero_grad(set_to_none=True)
            acc_loss = torch.zeros((), device=device, dtype=torch.float32)

            for m_i, item in enumerate(r_slice):
                ids_t = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                mask_t = torch.ones_like(ids_t)
                rmask_t = torch.tensor([item["resp_mask"]], dtype=torch.long, device=device)
                old_lp = item["old_logp"].to(device=device, dtype=torch.float32).unsqueeze(0)
                adv = float(item["advantage"])
                m_valid = rmask_t[:, 1:].float()

                sync_ctx = (
                    ddp_model.no_sync()
                    if (isinstance(ddp_model, DDP) and m_i < len(r_slice) - 1)
                    else torch.enable_grad()
                )
                with sync_ctx:
                    new_lp = ddp_model(ids_t, mask_t, rmask_t)
                    if item["is_off_policy"]:
                        pi_prob = torch.exp(torch.clamp(new_lp, min=-20.0, max=0.0))
                        shaped = pi_prob / (pi_prob + args.luffy_gamma)
                        surr = shaped * adv * m_valid
                    else:
                        log_ratio = torch.clamp(new_lp - old_lp, min=-5.0, max=5.0)
                        # Truncated importance sampling cap <= 2.0 (Opus 5.5 Max safeguard)
                        ratio = torch.clamp(torch.exp(log_ratio), max=2.0)
                        clipped = torch.clamp(ratio, 1.0 - args.eps_low, 1.0 + args.eps_high)
                        surr = torch.minimum(ratio * adv, clipped * adv) * m_valid

                    loss_sum = -surr.sum()
                    scaled = (loss_sum * world_size) / global_toks
                    scaled.backward()
                    acc_loss += loss_sum.detach()
                del ids_t, mask_t, rmask_t, old_lp, m_valid, new_lp, surr, loss_sum, scaled

            gnorm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0).item())
            optimizer.step()
            global_step += 1

            if world_size > 1:
                dist.all_reduce(acc_loss, op=dist.ReduceOp.SUM)
            mean_l = float((acc_loss / global_toks).item())
            step_losses.append(mean_l)
            if rank == 0 and (global_step == 1 or global_step % 4 == 0):
                print(
                    json.dumps(
                        {
                            "rl_step": global_step,
                            "pg_loss": round(mean_l, 5),
                            "grad_norm": round(gnorm, 4),
                            "peak_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
                        }
                    ),
                    flush=True,
                )

    wall_s = time.monotonic() - t0
    peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank != 0:
        return 0

    out_dir = Path(args.output_dir)
    raw_model = (ddp_model.module if isinstance(ddp_model, DDP) else ddp_model).model
    parity = merge_lora_fp32_and_verify_parity_v2(raw_model, spec, init_path, out_dir, samples, device)
    summary = {
        "model_key": spec.key,
        "init_model": str(init_path),
        "rollouts_file": str(args.rollouts_file),
        "output_dir": str(out_dir),
        "adapted_linear_modules": n_adapted,
        "rl_steps": global_step,
        "wall_seconds": wall_s,
        "gpu_hours": (wall_s * world_size) / 3600.0,
        "peak_allocated_gib": peak_gib,
        "group_stats": group_stats,
        "merge_parity": parity,
    }
    sp = Path(args.summary_output)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "DAPO_V2_COMPLETE", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

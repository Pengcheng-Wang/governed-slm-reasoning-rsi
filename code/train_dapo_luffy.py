#!/usr/bin/env python3
"""
Phased Online `DAPO + LUFFY` Verifier-RLVR Policy-Gradient Updater (`8x A100-80GB DDP`).

Combines the three strongest validated RLVR mechanisms from prior `AILab*` studies:
  1. **DAPO Clip-Higher** (`eps_low = 0.20`, `eps_high = 0.28`, `beta = 0.0` no KL drift penalty).
  2. **DAPO Soft-Overlong Termination Penalty + Unmasked Truncated Completions**:
     Rambling / truncated completions (`finish_reason == "length"`) are NOT masked out; they
     receive `reward = -overlong_penalty`, and stopped completions exceeding `L_max - L_cache`
     receive a smooth linear penalty ramp so the model explicitly learns efficient termination.
  3. **LUFFY Advantage-Gated Off-Policy Trace Shaping (`f(r) = r / (r + gamma)`, `gamma = 0.1`)**:
     On frontier prompts where all `G` on-policy rollouts fail (`0/G` dead group), `1` verified
     trace from the cumulative verified bank / teacher is injected into the group (`is_off_policy = True`).
     On off-policy tokens, the surrogate uses `f(pi / pi_behav) * A` (with `pi_behav = 1.0` -> `pi / (pi + gamma)`)
     without upper PPO clipping, eliminating zero-variance dead groups (`frac_reward_zero_std -> 0`).
  4. **Chunked 248K-Vocab Token Log-Probs** (`1,024`-token slices through `lm_head`) so `16K–24K`
     rollouts never OOM on A100-80GB.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForImageTextToText, AutoTokenizer

from train_slm_rsi import (
    EOS_ENDOFTEXT_ID,
    EOS_IM_END_ID,
    HYBRID_LM_LINEAR_REGEX,
    _init_dist,
    _merge_lora_fp32_and_verify_parity,
    _unwrap_backbone_and_head,
)


def _compute_sequence_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    resp_mask: torch.Tensor,
    chunk_tokens: int = 1024,
) -> torch.Tensor:
    """
    Returns `[B, T-1]` log-probabilities `log pi(y_{t+1} | y_{<=t})` masked by `resp_mask[:, 1:]`,
    computed in 1,024-token vocabulary chunks to keep peak HBM minimal.
    """
    lm_backbone, lm_head = _unwrap_backbone_and_head(model)
    outputs = lm_backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = outputs.last_hidden_state  # [B, T, H]
    shift_h = hidden[:, :-1, :]
    shift_y = input_ids[:, 1:]
    shift_m = resp_mask[:, 1:]
    bsz, seq_m1, hdim = shift_h.shape

    out_logp: list[torch.Tensor] = []
    for start in range(0, seq_m1, chunk_tokens):
        end = min(seq_m1, start + chunk_tokens)
        m_sub = shift_m[:, start:end]
        if not m_sub.any():
            out_logp.append(torch.zeros((bsz, end - start), device=hidden.device, dtype=torch.float32))
            continue
        hc = shift_h[:, start:end, :]
        yc = shift_y[:, start:end]
        logits = lm_head(hc).float()  # [B, chunk, V]
        lse = torch.logsumexp(logits, dim=-1)
        chosen = logits.gather(dim=-1, index=yc.unsqueeze(-1)).squeeze(-1)
        lp = (chosen - lse) * m_sub.float()
        out_logp.append(lp)

    return torch.cat(out_logp, dim=1)


class _ChunkedLogprobForward(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        resp_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _compute_sequence_token_logprobs(self.model, input_ids, attention_mask, resp_mask)


def _soft_overlong_penalty(comp_tokens: int, max_tokens: int, cache_window: int, max_pen: float) -> float:
    if cache_window <= 0 or max_pen <= 0.0:
        return 0.0
    thresh = max(1, max_tokens - cache_window)
    if comp_tokens <= thresh:
        return 0.0
    if comp_tokens >= max_tokens:
        return -max_pen
    return -max_pen * float(comp_tokens - thresh) / float(cache_window)


def _prepare_rl_groups(
    rollouts_path: Path,
    tok: Any,
    *,
    thinking: bool,
    max_seq_len: int,
    max_gen_tokens: int,
    overlong_window: int,
    overlong_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Reads pre-scored rollout records `{id, prompt_messages, response_text, finish_reason, verified_ok, is_off_policy}`,
    computes DAPO soft-overlong shaped rewards, normalizes group advantages `(R - mean) / (std + 1e-6)`,
    drops zero-variance groups (after recording `dead_group_fraction`), and tokenizes sequences.
    """
    raw = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in raw:
        by_pid[str(r["id"])].append(r)

    samples: list[dict[str, Any]] = []
    n_groups_total = len(by_pid)
    n_dead_groups = 0
    n_offpolicy_injected = 0
    rewards_all: list[float] = []

    for pid, grows in sorted(by_pid.items()):
        g_rewards: list[float] = []
        tokenized_group: list[tuple[list[int], list[int], bool]] = []

        for r in grows:
            p_msgs = r["prompt_messages"]
            p_out = tok.apply_chat_template(
                p_msgs, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking
            )
            if hasattr(p_out, "input_ids"):
                p_out = p_out.input_ids
            elif isinstance(p_out, dict) and "input_ids" in p_out:
                p_out = p_out["input_ids"]
            if p_out and isinstance(p_out[0], list):
                p_out = p_out[0]
            p_ids = [int(x) for x in p_out]
            # Tokenize response directly so even truncated (`finish_reason == "length"`) rollouts can receive negative advantage!
            r_ids = tok.encode(str(r["response_text"]), add_special_tokens=False)
            stopped = str(r.get("finish_reason", "stop")) == "stop"
            if stopped and (not r_ids or r_ids[-1] != EOS_IM_END_ID):
                r_ids = r_ids + [EOS_IM_END_ID]

            full_ids = p_ids + r_ids
            if len(full_ids) > max_seq_len:
                # Cap strictly at max_seq_len for memory safety; truncated tail is penalized, never rewarded
                full_ids = full_ids[:max_seq_len]
                stopped = False
                verified = False
            else:
                verified = bool(r.get("verified_ok", False)) and stopped

            comp_len = max(1, len(full_ids) - len(p_ids))
            resp_mask = [0] * len(p_ids) + [1] * comp_len
            is_off = bool(r.get("is_off_policy", False))

            base_rew = 1.0 if verified else 0.0
            if not stopped:
                pen = -overlong_penalty
            else:
                pen = _soft_overlong_penalty(comp_len, max_gen_tokens, overlong_window, overlong_penalty)
            rew = base_rew + pen

            g_rewards.append(rew)
            rewards_all.append(rew)
            tokenized_group.append((full_ids, resp_mask, is_off))

        mean_r = sum(g_rewards) / max(len(g_rewards), 1)
        var_r = sum((x - mean_r) ** 2 for x in g_rewards) / max(len(g_rewards), 1)
        std_r = math.sqrt(var_r)
        if std_r < 1e-5:
            n_dead_groups += 1
            continue

        for (f_ids, r_mask, is_off), rew in zip(tokenized_group, g_rewards):
            adv = (rew - mean_r) / (std_r + 1e-6)
            if is_off:
                # Only apply positive off-policy guidance
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
                    "labels": [-100 if m == 0 else tid for tid, m in zip(f_ids, r_mask)],
                }
            )

    stats = {
        "groups_total": n_groups_total,
        "dead_groups_skipped": n_dead_groups,
        "dead_group_fraction": n_dead_groups / max(n_groups_total, 1),
        "active_rl_samples": len(samples),
        "off_policy_positive_samples": n_offpolicy_injected,
        "mean_shaped_reward": sum(rewards_all) / max(len(rewards_all), 1),
    }
    return samples, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="DAPO + LUFFY Clipped Policy-Gradient Trainer")
    ap.add_argument("--init-model", required=True, help="Starting checkpoint for this RLVR wave")
    ap.add_argument("--rollouts-file", required=True, help="Scored group rollouts JSONL")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=18432)
    ap.add_argument("--max-gen-tokens", type=int, default=16384)
    ap.add_argument("--overlong-window", type=int, default=3072)
    ap.add_argument("--overlong-penalty", type=float, default=0.5)
    ap.add_argument("--luffy-gamma", type=float, default=0.1)
    ap.add_argument("--eps-low", type=float, default=0.20)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--global-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261118)
    args = ap.parse_args()

    rank, world_size, local_rank = _init_dist()
    device = torch.device(f"cuda:{local_rank}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.monotonic()
    init_path = Path(args.init_model)
    tok = AutoTokenizer.from_pretrained(str(init_path), trust_remote_code=True)
    samples, group_stats = _prepare_rl_groups(
        Path(args.rollouts_file),
        tok,
        thinking=args.thinking,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_gen_tokens,
        overlong_window=args.overlong_window,
        overlong_penalty=args.overlong_penalty,
    )
    if len(samples) < world_size:
        raise RuntimeError(f"Insufficient active RL samples ({len(samples)}): {group_stats}")
    if rank == 0:
        print(json.dumps({"event": "RL_GROUPS_READY", **group_stats}), flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        str(init_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
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

    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=HYBRID_LM_LINEAR_REGEX,
        bias="none",
    )
    model = get_peft_model(model, lcfg)
    model.to(device)

    # Precompute exact `old_logp` (`pi_old` == `init_model` before LoRA updates, since LoRA B=0 at init!)
    model.eval()
    pad_id = EOS_ENDOFTEXT_ID
    with torch.no_grad():
        for idx in range(rank, len(samples), world_size):
            s = samples[idx]
            ids_t = torch.tensor([s["input_ids"]], dtype=torch.long, device=device)
            mask_t = torch.ones_like(ids_t)
            rmask_t = torch.tensor([s["resp_mask"]], dtype=torch.long, device=device)
            lp = _compute_sequence_token_logprobs(model, ids_t, mask_t, rmask_t)
            s["old_logp"] = lp[0].cpu()

    # Gather `old_logp` across all ranks so shuffling works cleanly
    if world_size > 1:
        local_pairs = [(idx, samples[idx]["old_logp"]) for idx in range(rank, len(samples), world_size)]
        gathered: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, local_pairs)
        for plist in gathered:
            for idx, olp in plist:
                samples[idx]["old_logp"] = olp

    model.train()
    fwd_mod = _ChunkedLogprobForward(model)
    ddp_model: torch.nn.Module = DDP(fwd_mod, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else fwd_mod
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    grad_accum = max(1, args.global_batch_size // world_size)
    per_step = world_size * grad_accum
    global_step = 0
    step_losses: list[float] = []

    for ep in range(args.ppo_epochs):
        rng = random.Random(args.seed + 7919 * ep)
        ep_rows = list(samples)
        rng.shuffle(ep_rows)
        rem = len(ep_rows) % per_step
        if rem != 0:
            ep_rows.extend(ep_rows[: (per_step - rem)])

        for s_idx in range(0, len(ep_rows), per_step):
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
                old_lp = item["old_logp"].to(device).unsqueeze(0)  # [1, T-1]
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
                        # LUFFY shaped off-policy surrogate: f(pi) = pi / (pi + gamma), where pi = exp(new_lp)
                        pi_prob = torch.exp(torch.clamp(new_lp, min=-20.0, max=0.0))
                        shaped = pi_prob / (pi_prob + args.luffy_gamma)
                        surr = shaped * adv * m_valid
                    else:
                        log_ratio = torch.clamp(new_lp - old_lp, min=-10.0, max=10.0)
                        ratio = torch.exp(log_ratio)
                        clipped = torch.clamp(ratio, 1.0 - args.eps_low, 1.0 + args.eps_high)
                        surr1 = ratio * adv
                        surr2 = clipped * adv
                        surr = torch.minimum(surr1, surr2) * m_valid

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
                            "epoch": ep + 1,
                            "pg_loss": round(mean_l, 5),
                            "grad_norm": round(gnorm, 4),
                            "peak_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
                        }
                    ),
                    flush=True,
                )

    wall_s = time.monotonic() - t0
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        out_dir = Path(args.output_dir)
        unwrapped_fwd = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
        raw_model = unwrapped_fwd.model
        parity = _merge_lora_fp32_and_verify_parity(raw_model, init_path, out_dir, samples, device)
        summary = {
            "init_model": str(init_path),
            "rollouts_file": str(args.rollouts_file),
            "output_dir": str(out_dir),
            "rl_steps": global_step,
            "wall_seconds": wall_s,
            "gpu_hours": (wall_s * world_size) / 3600.0,
            "group_stats": group_stats,
            "merge_parity": parity,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "DAPO_LUFFY_COMPLETE", **summary}, indent=2), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

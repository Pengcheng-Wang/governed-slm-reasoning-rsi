#!/usr/bin/env python3
"""
Phase V4.1 Precision-Safe, Temperature-Matched & Common-Backbone Off-Policy RLVR Trainer.

Key design upgrades over `code/v4/train_rlvr_v4.py` (which remains frozen for V4.0):
  1. Exact Temperature-Matched Logprobs (`--rollout-temperature 0.7`): divides `logits / T`
     inside chunked FP32 `log_softmax` so `T_train == T_rollout == T_eval`.
  2. Clean Objective Isolation (`--cispo-sft-alpha 0.0`, `--overlong-penalty 0.0` default):
     removes auxiliary NLL anchoring and cross-algorithm penalty terms.
  3. Common-Backbone `GSPO` Off-Policy Ablation (`--offpolicy-mode`):
     - `none`: Pure On-Policy GSPO (`0%` off-policy traces).
     - `faithful_luffy`: Yan et al. (2025) Mixed-Policy LUFFY — keeps `1` channel-aligned
       verified teacher trace + `7` on-policy student negative rollouts per rescue prompt with
       joint group-normalized advantages `A_i = (R_i - mu_g)/std_g`, and DIFFERENTIATES through
       `f(p) = p / (p + gamma)` (`d/d(log p) f(p) = gamma * p / (p + gamma)^2`), logging exact
       empirical `p01, p10, p50, p90` quantiles of teacher token probabilities.
     - `log_shaped`: Identical `1+7` mixed groups and joint `A_i`, replacing `d/d(log p) f(p)` with
       rare-token-preserving detached weight `w(p) = gamma / (p + gamma)`.
     - `unshaped_is`: Identical `1+7` mixed groups and joint `A_i` with unit weight `w(p) = 1.0`
       on verified teacher tokens.
  4. Unbiased Stochastic `FP32 -> BF16` Consolidation + `delta_energy_retention` telemetry.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as grad_ckpt
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CODE_V4_DIR = ROOT / "code/v4"
CODE_V4_1_DIR = ROOT / "code/v4_1"
EVAL_V4_DIR = ROOT / "eval_v4"
for p in (CODE_V4_1_DIR, CODE_V4_DIR, EVAL_V4_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_engine_v4 import render_prompt_v4  # noqa: E402
from model_registry_v4 import MODEL_REGISTRY_V4, ModelSpecV4  # noqa: E402
from train_rlvr_v4 import soft_overlong_penalty  # noqa: E402
from train_sft_v4 import (  # noqa: E402
    apply_prior_fp32_factors_inplace,
    attach_verified_lora_v4,
    clip_distributed_grad_norm,
    compute_family_logits_fp32,
    extract_logit_head_Spec,
    init_dist,
    load_hf_model_staggered_v4,
    wrap_distributed_backbone_v4,
)
from train_sft_v4_1 import (  # noqa: E402
    consolidate_and_verify_v4_1,
    format_channel_aligned_pair_v4_1,
)


def _logp_chunk_step_temp(
    hc: torch.Tensor,
    yc: torch.Tensor,
    m_sub: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    output_multiplier: float | None,
    final_logit_softcapping: float | None,
    inv_temperature: float,
) -> torch.Tensor:
    logits = compute_family_logits_fp32(hc, lm_head_weight, lm_head_bias, output_multiplier, final_logit_softcapping)
    if abs(inv_temperature - 1.0) > 1e-6:
        logits = logits * float(inv_temperature)
    lse = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(dim=-1, index=yc.unsqueeze(-1)).squeeze(-1)
    return (chosen - lse) * m_sub.float()


def compute_backbone_sequence_logprobs_v4_1(
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    output_multiplier: float | None,
    final_logit_softcapping: float | None,
    rollout_temperature: float = 0.7,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state
    zero_anchor = hidden[:, :1, :1].sum() * 0.0

    shift_h = hidden[:, :-1, :]
    shift_y = input_ids[:, 1:]
    shift_m = response_mask[:, 1:].bool()
    seq_m1 = shift_h.shape[1]
    w_head = lm_head.weight
    b_head = getattr(lm_head, "bias", None)
    inv_t = 1.0 / max(float(rollout_temperature), 1e-4)

    pieces: list[torch.Tensor] = []
    for start in range(0, seq_m1, chunk_tokens):
        end = min(seq_m1, start + chunk_tokens)
        m_sub = shift_m[:, start:end]
        if not bool(m_sub.any().item()):
            pieces.append(torch.zeros((shift_h.shape[0], end - start), dtype=torch.float32, device=shift_h.device))
            continue
        hc = shift_h[:, start:end, :]
        yc = shift_y[:, start:end]
        if torch.is_grad_enabled() and hc.requires_grad:
            lp_c = grad_ckpt(
                _logp_chunk_step_temp,
                hc,
                yc,
                m_sub,
                w_head,
                b_head,
                output_multiplier,
                final_logit_softcapping,
                inv_t,
                use_reentrant=False,
            )
        else:
            lp_c = _logp_chunk_step_temp(hc, yc, m_sub, w_head, b_head, output_multiplier, final_logit_softcapping, inv_t)
        pieces.append(lp_c)
    return torch.cat(pieces, dim=1), shift_m, zero_anchor


def compute_sample_rlvr_loss_v4_1(
    algo: str,
    offpolicy_mode: str,
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
    cispo_sft_alpha: float = 0.0,
    luffy_gamma: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    m = shift_mask.float()
    comp_len = m.sum().clamp_min(1.0)
    adv_t = torch.tensor(float(advantage), dtype=torch.float32, device=tok_logp.device)
    denom = max(float(norm_tokens), 1.0)

    if is_off_policy:
        with torch.no_grad():
            p_vals = torch.exp(tok_logp.detach()[shift_mask.bool()].clamp(min=-20.0, max=0.0))
            p_quants = (
                {
                    "p01": float(torch.quantile(p_vals, 0.01).item()),
                    "p10": float(torch.quantile(p_vals, 0.10).item()),
                    "p50": float(torch.quantile(p_vals, 0.50).item()),
                    "p90": float(torch.quantile(p_vals, 0.90).item()),
                    "frac_lt_1e3": float((p_vals < 1e-3).float().mean().item()),
                }
                if p_vals.numel() > 0
                else {}
            )

        if offpolicy_mode == "faithful_luffy":
            # Differentiated Yan et al. (2025) policy shaping: d/d(log p) [p/(p+gamma)] = gamma*p/(p+gamma)^2
            pi_prob = torch.exp(tok_logp.clamp(min=-20.0, max=0.0))
            f_luffy = pi_prob / (pi_prob + luffy_gamma)
            seq_loss = -((f_luffy * adv_t) * m).sum() / comp_len
            mean_w = float(((luffy_gamma * pi_prob.detach()) / ((pi_prob.detach() + luffy_gamma) ** 2))[shift_mask.bool()].mean().item()) if shift_mask.any() else 0.0
            return seq_loss, {"clipped_frac": 0.0, "mean_ratio": mean_w, "offpolicy_p_quantiles": p_quants}

        if offpolicy_mode == "log_shaped":
            # Rare-token-preserving weight w(p) = gamma / (p + gamma): ~1.0 at p=1e-5, 0.50 at p=gamma, 0.091 at p=1
            pi_det = torch.exp(tok_logp.detach().clamp(min=-20.0, max=0.0))
            w_log = (luffy_gamma / (pi_det + luffy_gamma)).detach()
            seq_loss = -((w_log * adv_t * tok_logp) * m).sum() / comp_len
            mean_w = float(w_log[shift_mask.bool()].mean().item()) if shift_mask.any() else 0.0
            return seq_loss, {"clipped_frac": 0.0, "mean_ratio": mean_w, "offpolicy_p_quantiles": p_quants}

        if offpolicy_mode == "unshaped_is":
            seq_loss = -((adv_t * tok_logp) * m).sum() / comp_len
            return seq_loss, {"clipped_frac": 0.0, "mean_ratio": 1.0, "offpolicy_p_quantiles": p_quants}

        raise ValueError(f"Off-policy sample encountered when offpolicy_mode={offpolicy_mode!r}")

    log_ratio = (tok_logp - old_logp.detach()) * m

    if algo == "gspo":
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

    if algo == "drgrpo":
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        w_cispo = torch.clamp(ratio.detach(), 1.0 - eps_low, 1.0 + eps_high)
        pg_loss = -((w_cispo * adv_t * tok_logp) * m).sum() / denom
        if verified_positive and cispo_sft_alpha > 0.0:
            pg_loss = pg_loss + cispo_sft_alpha * (-(tok_logp * m).sum() / denom)
        with torch.no_grad():
            is_clipped = (ratio > 1.0 + eps_high) | (ratio < 1.0 - eps_low)
            cfrac = float((is_clipped.float() * m).sum().item() / float(comp_len.item()))
            mr = float((ratio * m).sum().item() / float(comp_len.item()))
        return pg_loss, {"clipped_frac": cfrac, "mean_ratio": mr}

    if algo == "dapo":
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        clipped_r = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
        tok_obj = torch.minimum(ratio * adv_t, clipped_r * adv_t) * m
        seq_loss = -tok_obj.sum() / denom
        with torch.no_grad():
            is_clipped = ((adv_t > 0) & (ratio > 1.0 + eps_high)) | ((adv_t < 0) & (ratio < 1.0 - eps_low))
            cfrac = float((is_clipped.float() * m).sum().item() / float(comp_len.item()))
            mr = float((ratio * m).sum().item() / float(comp_len.item()))
        return seq_loss, {"clipped_frac": cfrac, "mean_ratio": mr}

    raise ValueError(f"Unknown RLVR algorithm: {algo!r}")


def prepare_rlvr_groups_v4_1(
    rollouts_path: Path,
    tok: Any,
    spec: ModelSpecV4,
    *,
    algo: str,
    offpolicy_mode: str,
    track: str,
    chat_kwargs: dict[str, Any],
    max_seq_len: int,
    max_gen_tokens: int,
    overlong_window: int,
    overlong_penalty: float,
    target_offpolicy_group_share: float = 0.25,
    seed: int = 20261124,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    raw_groups = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mixed_onpolicy_groups: list[list[dict[str, Any]]] = []
    mixed_offpolicy_groups: list[list[dict[str, Any]]] = []
    skipped_saturated = 0
    skipped_zero = 0
    all_comp_lens: list[int] = []

    for g in raw_groups:
        samples = g["samples"]
        on_samples = [s for s in samples if not bool(s.get("off_policy", False))]
        off_samples = [s for s in samples if bool(s.get("off_policy", False)) and bool(s.get("verified", False))]

        n_on_pos = sum(1 for s in on_samples if bool(s.get("verified", False)))
        if len(on_samples) > 0 and n_on_pos == len(on_samples):
            skipped_saturated += 1
            continue

        is_mixed_rescue = False
        if n_on_pos == 0:
            if offpolicy_mode != "none" and off_samples and len(on_samples) >= 1:
                # Faithful Mixed-Group Construction: 1 verified teacher trace + up to 7 student negatives!
                chosen_samples = [off_samples[0]] + on_samples[:7]
                is_mixed_rescue = True
            else:
                skipped_zero += 1
                continue
        else:
            chosen_samples = on_samples

        raw_rewards: list[float] = []
        valid_entries: list[dict[str, Any]] = []
        for s in chosen_samples:
            is_off = bool(s.get("off_policy", False))
            fr = str(s.get("finish_reason", "stop"))
            raw_txt = str(s["text"])

            if not is_off:
                p_str = render_prompt_v4(tok, spec, g["messages"], chat_kwargs=chat_kwargs)
                p_ids = [int(x) for x in tok.encode(p_str, add_special_tokens=False)]
                if s.get("token_ids"):
                    r_ids = [int(x) for x in s["token_ids"]]
                elif fr == "stop":
                    suf = spec.sft_suffix
                    if spec.channel_family == "inkling2" and raw_txt.rstrip().endswith("<|end_message|>"):
                        suf = "<|content_model_end_sampling|>"
                    r_ids = [int(x) for x in tok.encode(raw_txt + suf, add_special_tokens=False)]
                else:
                    r_ids = [int(x) for x in tok.encode(raw_txt, add_special_tokens=False)]
            else:
                # Channel-aligned teacher formatting (fixes V4.0 N9 bug inside off-policy traces!)
                p_str, r_str = format_channel_aligned_pair_v4_1(
                    tok,
                    spec,
                    g["messages"],
                    raw_txt,
                    row={"gold": g.get("gold", ""), "thinking_text": s.get("thinking_text", "")},
                    track=track,
                    eval_chat_kwargs=chat_kwargs,
                    channel_mode="native_thought_aligned",
                )
                p_ids = [int(x) for x in tok.encode(p_str, add_special_tokens=False)]
                r_ids = [int(x) for x in tok.encode(r_str, add_special_tokens=False)]

            f_ids = p_ids + r_ids
            was_cropped = False
            if len(f_ids) > max_seq_len:
                f_ids = f_ids[:max_seq_len]
                was_cropped = True
            comp_len = len(f_ids) - len(p_ids)
            if comp_len < 2:
                continue

            ver_ok = bool(s.get("verified", False)) and (not was_cropped)
            base_r = 1.0 if ver_ok else 0.0
            pen = (
                soft_overlong_penalty(int(s.get("completion_tokens", comp_len)), max_gen_tokens, overlong_window, overlong_penalty)
                if overlong_penalty > 0.0
                else 0.0
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
        mu = sum(raw_rewards) / float(len(raw_rewards))
        var = sum((r - mu) ** 2 for r in raw_rewards) / float(len(raw_rewards))
        std = math.sqrt(var)
        if std < 1e-6:
            continue

        for e in valid_entries:
            if algo == "drgrpo":
                e["advantage"] = 2.0 * (e["raw_reward"] - mu)
            else:
                e["advantage"] = (e["raw_reward"] - mu) / (std + 1e-6)

        if is_mixed_rescue:
            mixed_offpolicy_groups.append(valid_entries)
        else:
            mixed_onpolicy_groups.append(valid_entries)

    rng = random.Random(seed)
    rng.shuffle(mixed_onpolicy_groups)
    rng.shuffle(mixed_offpolicy_groups)

    if offpolicy_mode != "none" and mixed_offpolicy_groups:
        if mixed_onpolicy_groups and target_offpolicy_group_share < 0.999:
            # Solve n_off / (n_on + n_off) == share -> n_off = round(n_on * share / (1 - share))
            want_off = max(1, int(round(len(mixed_onpolicy_groups) * target_offpolicy_group_share / max(1e-6, 1.0 - target_offpolicy_group_share))))
            selected_off = mixed_offpolicy_groups[:want_off]
        else:
            selected_off = mixed_offpolicy_groups[:32]
        active_groups = mixed_onpolicy_groups + selected_off
        rng.shuffle(active_groups)
    else:
        selected_off = []
        active_groups = mixed_onpolicy_groups

    total_samples = sum(len(g) for g in active_groups)
    off_samples_count = sum(1 for g in active_groups for e in g if e["off_policy"])
    mean_comp = sum(all_comp_lens) / float(max(1, len(all_comp_lens)))
    stats = {
        "algo": algo,
        "offpolicy_mode": offpolicy_mode,
        "raw_groups_total": len(raw_groups),
        "mixed_onpolicy_groups": len(mixed_onpolicy_groups),
        "mixed_offpolicy_groups_available": len(mixed_offpolicy_groups),
        "mixed_offpolicy_groups_included": len(selected_off),
        "active_groups_total": len(active_groups),
        "total_training_samples": total_samples,
        "offpolicy_positive_samples": off_samples_count,
        "realized_offpolicy_group_share": len(selected_off) / float(max(1, len(active_groups))),
        "realized_offpolicy_sample_share": off_samples_count / float(max(1, total_samples)),
        "skipped_saturated_groups": skipped_saturated,
        "skipped_zero_groups": skipped_zero,
        "mean_comp_tokens": mean_comp,
    }
    return active_groups, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V4.1 Precision-Safe & Common-Backbone RLVR Trainer")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V4.keys()))
    ap.add_argument("--track", choices=["math", "lean"], required=True)
    ap.add_argument("--algo", choices=["gspo", "drgrpo", "dapo"], default="gspo")
    ap.add_argument("--offpolicy-mode", choices=["none", "faithful_luffy", "log_shaped", "unshaped_is"], default="none")
    ap.add_argument("--target-offpolicy-group-share", type=float, default=0.25)
    ap.add_argument("--luffy-gamma", type=float, default=0.10)
    ap.add_argument("--rollout-temperature", type=float, default=0.7)
    ap.add_argument("--chat-kwargs-json", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--prior-fp32-deltas", default="")
    ap.add_argument("--save-fp32-deltas", default="")
    ap.add_argument("--rollouts-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--max-seq-len", type=int, default=11264)
    ap.add_argument("--max-gen-tokens", type=int, default=10240)
    ap.add_argument("--overlong-window", type=int, default=2048)
    ap.add_argument("--overlong-penalty", type=float, default=0.0)
    ap.add_argument("--cispo-sft-alpha", type=float, default=0.0)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--groups-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1.2e-5)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--stochastic-bf16-merge", action="store_true", default=True)
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
    groups, group_stats = prepare_rlvr_groups_v4_1(
        Path(args.rollouts_file),
        tok,
        spec,
        algo=args.algo,
        offpolicy_mode=args.offpolicy_mode,
        track=args.track,
        chat_kwargs=chat_kwargs,
        max_seq_len=args.max_seq_len,
        max_gen_tokens=args.max_gen_tokens,
        overlong_window=args.overlong_window,
        overlong_penalty=args.overlong_penalty,
        target_offpolicy_group_share=args.target_offpolicy_group_share,
        seed=args.seed,
    )
    if not groups:
        raise RuntimeError(f"Zero active RLVR groups prepared: {group_stats}")

    t0 = time.monotonic()
    base_model = load_hf_model_staggered_v4(spec, orig_base_path, rank=rank, world_size=world_size, drop_multimodal_for_train=True)
    out_mult, softcap = extract_logit_head_Spec(base_model)
    if args.prior_fp32_deltas:
        apply_prior_fp32_factors_inplace(base_model, args.prior_fp32_deltas, device)
        if world_size > 1:
            dist.barrier()

    peft_m, n_adapted = attach_verified_lora_v4(base_model, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0)
    trainable_numel = sum(int(p.numel()) for p in peft_m.parameters() if p.requires_grad)
    wrapped_backbone, lm_head, is_fsdp = wrap_distributed_backbone_v4(peft_m, spec, device, local_rank, world_size)

    flat_samples = [s for g in groups for s in g]
    wrapped_backbone.eval()
    with torch.no_grad():
        for idx_s in range(rank, len(flat_samples), world_size):
            sm = flat_samples[idx_s]
            ids = torch.tensor([sm["input_ids"]], dtype=torch.long, device=device)
            am = torch.ones_like(ids)
            rm = torch.zeros_like(ids)
            rm[:, sm["prompt_len"] :] = 1
            lp, _, _ = compute_backbone_sequence_logprobs_v4_1(
                wrapped_backbone,
                lm_head,
                ids,
                am,
                rm,
                output_multiplier=out_mult,
                final_logit_softcapping=softcap,
                rollout_temperature=args.rollout_temperature,
            )
            sm["old_logp_cpu"] = lp.squeeze(0).cpu()
        rem_s = len(flat_samples) % world_size
        if rem_s != 0 and rank >= rem_s:
            ids_d = torch.tensor([flat_samples[0]["input_ids"][:64]], dtype=torch.long, device=device)
            compute_backbone_sequence_logprobs_v4_1(
                wrapped_backbone, lm_head, ids_d, torch.ones_like(ids_d), torch.zeros_like(ids_d),
                output_multiplier=out_mult, final_logit_softcapping=softcap, rollout_temperature=args.rollout_temperature,
            )

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
    # Allow multiple passes over active groups when needed to reach `args.max_steps`
    eff_epochs = max(args.ppo_epochs, math.ceil(args.max_steps / float(steps_per_pass)))
    total_steps = args.max_steps

    norm_tokens = float(group_stats["mean_comp_tokens"])
    torch.cuda.reset_peak_memory_stats(device)
    step = 0
    step_logs: list[dict[str, Any]] = []
    offpolicy_quant_logs: list[dict[str, float]] = []

    for ep in range(eff_epochs):
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
                    tok_lp, s_mask, z_anc = compute_backbone_sequence_logprobs_v4_1(
                        wrapped_backbone,
                        lm_head,
                        ids,
                        am,
                        rm,
                        output_multiplier=out_mult,
                        final_logit_softcapping=softcap,
                        rollout_temperature=args.rollout_temperature,
                    )
                    old_lp = sm["old_logp_cpu"].to(device=device).unsqueeze(0)
                    seq_l, st = compute_sample_rlvr_loss_v4_1(
                        args.algo,
                        args.offpolicy_mode,
                        tok_lp,
                        old_lp,
                        s_mask,
                        advantage=float(sm["advantage"]),
                        verified_positive=bool(sm["verified"]),
                        is_off_policy=bool(sm["off_policy"]),
                        norm_tokens=norm_tokens,
                        cispo_sft_alpha=args.cispo_sft_alpha,
                        luffy_gamma=args.luffy_gamma,
                    )
                    local_loss_sum_val += float(seq_l.detach().item())
                    clip_fracs.append(float(st["clipped_frac"]))
                    ratios.append(float(st["mean_ratio"]))
                    if st.get("offpolicy_p_quantiles") and len(offpolicy_quant_logs) < 32:
                        offpolicy_quant_logs.append(st["offpolicy_p_quantiles"])
                    micro_loss = seq_l * (float(world_size) / float(n_active)) + z_anc
                    micro_loss.backward()
                else:
                    ids_d = torch.tensor([flat_samples[0]["input_ids"][:64]], dtype=torch.long, device=device)
                    _, _, z_anc = compute_backbone_sequence_logprobs_v4_1(
                        wrapped_backbone, lm_head, ids_d, torch.ones_like(ids_d), torch.zeros_like(ids_d),
                        output_multiplier=out_mult, final_logit_softcapping=softcap, rollout_temperature=args.rollout_temperature,
                    )
                    z_anc.backward()

            gn = clip_distributed_grad_norm(wrapped_backbone, is_fsdp=is_fsdp, max_norm=1.0)
            optimizer.step()
            step += 1

            mean_loss_t = torch.tensor(local_loss_sum_val / float(n_active), device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(mean_loss_t, op=dist.ReduceOp.SUM)
            if rank == 0 and (step == 1 or step % 5 == 0 or step == total_steps):
                rec = {
                    "step": step,
                    "epoch": ep + 1,
                    "loss": float(mean_loss_t.item()),
                    "grad_norm": gn,
                    "clipped_frac_rank0": sum(clip_fracs) / max(1, len(clip_fracs)),
                    "mean_ratio_rank0": sum(ratios) / max(1, len(ratios)),
                }
                step_logs.append(rec)
                print(json.dumps(rec), flush=True)

    peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    train_wall_s = time.monotonic() - t0

    out_dir_final = Path(args.output_dir)
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
        out_dir=out_dir_final,
        parity_samples=[{"input_ids": s["input_ids"]} for s in flat_samples[:4]],
        stochastic_bf16=args.stochastic_bf16_merge,
        seed=args.seed,
    )

    if rank == 0:
        summary = {
            "status": "RLVR_TRAINED_AND_CONSOLIDATED_V4_1",
            "model_key": args.model_key,
            "track": args.track,
            "algo": args.algo,
            "offpolicy_mode": args.offpolicy_mode,
            "rollout_temperature": args.rollout_temperature,
            "cispo_sft_alpha": args.cispo_sft_alpha,
            "adapted_lora_modules": n_adapted,
            "trainable_numel": trainable_numel,
            "steps_completed": step,
            "lr": args.lr,
            "seed": args.seed,
            "train_wall_sec": train_wall_s,
            "peak_vram_gb": peak_vram_gb,
            "group_stats": group_stats,
            "offpolicy_p_quantile_samples": offpolicy_quant_logs,
            "parity": parity,
            "step_logs": step_logs,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

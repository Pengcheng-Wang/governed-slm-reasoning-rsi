#!/usr/bin/env python3
"""
Governed Weight-Space `ReST^EM` Trainer for `Qwen3_5ForConditionalGeneration` (4B & 9B).

Implements all Cross-Lab & Frontier-Review Hygiene Invariants:
  1. Always initializes fresh from pinned base `M_0` (`--base-model`).
  2. Explicit hybrid `language_model` linear projection allowlist covering BOTH
     Gated-DeltaNet (`in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`),
     Full Self-Attention (`q_proj, k_proj, v_proj, o_proj`), and MLP (`gate_proj, up_proj, down_proj`),
     strictly excluding `visual`, `mtp`, `embed_tokens`, and `lm_head` (plus `--mode full_ft_lm` ablation).
  3. Zero sequence packing (`packing=False`) so Gated-DeltaNet `conv1d` and recurrent states
     never bleed across sample boundaries.
  4. Strict pre-flight token audit: exact prompt-prefix match, supervised `<|im_end|>` (`248046`),
     `0%` right-truncation (overlength traces fail closed rather than slicing).
  5. Chunked `lm_head` Cross-Entropy (1,024-token chunks) avoiding 248,320-vocab 32 GB logit OOMs
     on 16K-24K long-CoT reasoning traces.
  6. Problem-balanced sample weights (`1 / count(problem_id)`) so easy multi-trace items never
     dominate hard 1-trace frontier items.
  7. Exact `FP32` LoRA delta merge (`W_base.float() + scaling * B.float() @ A.float() -> BF16`)
     followed by attached-PEFT vs merged-BF16 logit parity verification in the identical HF runtime.
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
from transformers import AutoModelForImageTextToText, AutoTokenizer, get_cosine_schedule_with_warmup

EOS_IM_END_ID = 248046
EOS_ENDOFTEXT_ID = 248044
HYBRID_LM_LINEAR_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def _init_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    torch.cuda.set_device(0)
    return 0, 1, 0


def _chat_prompt_ids(tok: Any, prompt_msgs: list[dict[str, str]], *, thinking: bool) -> list[int]:
    out = tok.apply_chat_template(
        prompt_msgs,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def _tokenize_and_audit_dataset(
    data_path: Path,
    tok: Any,
    *,
    thinking: bool,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_rows:
        raise ValueError(f"Empty training dataset: {data_path}")

    kept_rows: list[tuple[int, dict[str, Any], list[int], list[int]]] = []
    dropped_overlong = 0

    for idx, row in enumerate(raw_rows):
        if str(row.get("finish_reason", "stop")) != "stop":
            raise ValueError(f"Row {idx} ({row.get('id')}) has non-stop finish_reason={row.get('finish_reason')}")
        msgs = row["messages"]
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            raise ValueError(f"Row {idx} must end with assistant message")

        prompt_msgs = msgs[:-1]
        p_ids = _chat_prompt_ids(tok, prompt_msgs, thinking=thinking)
        r_ids = [int(x) for x in tok.encode(str(msgs[-1]["content"]), add_special_tokens=False)]
        while r_ids and r_ids[-1] == EOS_IM_END_ID:
            r_ids.pop()
        f_ids = p_ids + r_ids + [EOS_IM_END_ID]

        if len(f_ids) > max_seq_len:
            dropped_overlong += 1
            continue

        n_sup = len(f_ids) - len(p_ids)
        if n_sup < 2:
            raise ValueError(f"Row {idx} ({row.get('id')}) has insufficient supervised tokens ({n_sup})")
        kept_rows.append((idx, row, p_ids, f_ids))

    if not kept_rows or dropped_overlong > 0.45 * len(raw_rows):
        raise ValueError(
            f"Overlong filter rejected too many rows: dropped={dropped_overlong}/{len(raw_rows)} at max_seq_len={max_seq_len}"
        )

    prob_counts = collections.Counter(str(r["problem_id"]) for _, r, _, _ in kept_rows)
    tokenized: list[dict[str, Any]] = []
    lengths: list[int] = []
    sup_lengths: list[int] = []

    for idx, row, p_ids, f_ids in kept_rows:
        labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        assert labels[-1] == EOS_IM_END_ID
        pid = str(row["problem_id"])
        w = 1.0 / float(prob_counts[pid])

        tokenized.append(
            {
                "id": str(row.get("id", f"row_{idx}")),
                "problem_id": pid,
                "input_ids": f_ids,
                "labels": labels,
                "weight": w,
            }
        )
        lengths.append(len(f_ids))
        sup_lengths.append(len(f_ids) - len(p_ids))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    audit = {
        "total_traces": n,
        "dropped_whole_overlong_rows": dropped_overlong,
        "unique_problems": len(prob_counts),
        "seq_len_p50": lengths_sorted[n // 2],
        "seq_len_p95": lengths_sorted[min(n - 1, int(0.95 * n))],
        "seq_len_max": lengths_sorted[-1],
        "supervised_tokens_total": sum(sup_lengths),
        "overlength_truncated_count": 0,
        "eos_supervised_rate": 1.0,
    }
    return tokenized, audit


def _unwrap_backbone_and_head(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Returns `(language_model_backbone, lm_head)` through DDP and/or PEFT wrappers."""
    m = model.module if isinstance(model, DDP) else model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
    else:
        inner = m
    lm_backbone = inner.model.language_model
    lm_head = inner.lm_head
    return lm_backbone, lm_head


def _chunked_weighted_ce_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs `language_model` backbone to get `last_hidden_state` `[B, T, H]`, then computes token CE
    in 1,024-token chunks through `lm_head` so 248,320-vocab FP32 logits never OOM on long CoT.
    Returns `(weighted_loss_sum, weight_token_denom)`.
    """
    lm_backbone, lm_head = _unwrap_backbone_and_head(model)
    outputs = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    hidden = outputs.last_hidden_state  # [B, T, H]

    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    bsz, seq_m1, hdim = shift_hidden.shape

    total_weighted_nll = torch.zeros((), device=hidden.device, dtype=torch.float32)
    total_weight_toks = torch.zeros((), device=hidden.device, dtype=torch.float32)

    for start in range(0, seq_m1, chunk_tokens):
        end = min(seq_m1, start + chunk_tokens)
        h_chunk = shift_hidden[:, start:end, :].reshape(-1, hdim)
        y_chunk = shift_labels[:, start:end].reshape(-1)
        valid_mask = y_chunk != -100
        if not valid_mask.any():
            continue
        logits_chunk = lm_head(h_chunk)  # [B * chunk, V]
        tok_nll = F.cross_entropy(logits_chunk.float(), y_chunk, ignore_index=-100, reduction="none")
        tok_nll = tok_nll.view(bsz, end - start)
        vmask = (shift_labels[:, start:end] != -100).float()
        w_b = sample_weights.view(bsz, 1)
        total_weighted_nll = total_weighted_nll + (tok_nll * vmask * w_b).sum()
        total_weight_toks = total_weight_toks + (vmask * w_b).sum()

    return total_weighted_nll, total_weight_toks


class _ChunkedCEForward(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _chunked_weighted_ce_loss(self.model, input_ids, attention_mask, labels, sample_weights)


@torch.no_grad()
def _merge_lora_fp32_and_verify_parity(
    peft_model: torch.nn.Module,
    base_model_path: Path,
    out_dir: Path,
    parity_samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """
    1. Computes reference logits from the trained attached-PEFT model on `parity_samples`.
    2. Loads a clean BF16 copy of `base_model_path` on CPU, merges every LoRA adapter delta in
       exact `FP32` (`W_base.float() + scaling * (B.float() @ A.float()) -> BF16`), saves complete
       HF checkpoint + tokenizer/template files to `out_dir`.
    3. Reloads the merged model on `device` in BF16 and verifies attached-vs-merged logit equivalence.
    """
    peft_model.eval()
    ref_logits_list: list[torch.Tensor] = []
    for s in parity_samples[:6]:
        ids = torch.tensor([s["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        lm_b, lm_h = _unwrap_backbone_and_head(peft_model)
        h = lm_b(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        ref_logits_list.append(lm_h(h).float().cpu())

    # Extract LoRA state tensors to CPU FP32
    lora_deltas: dict[str, torch.Tensor] = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "default" in mod.lora_A:
            a_w = mod.lora_A["default"].weight.detach().float().cpu()
            b_w = mod.lora_B["default"].weight.detach().float().cpu()
            scale = float(mod.scaling["default"]) if isinstance(mod.scaling, dict) else float(mod.scaling)
            # Strip PEFT `base_model.model.` prefix
            clean_name = name
            if clean_name.startswith("base_model.model."):
                clean_name = clean_name[len("base_model.model.") :]
            lora_deltas[clean_name + ".weight"] = scale * (b_w @ a_w)

    # Free GPU memory before reloading clean base & merged verification
    del peft_model
    torch.cuda.empty_cache()

    clean_model = AutoModelForImageTextToText.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sd = clean_model.state_dict()
    merged_count = 0
    for w_key, delta_fp32 in lora_deltas.items():
        if w_key not in sd:
            raise KeyError(f"LoRA target weight key missing in clean base model: {w_key}")
        merged_w = (sd[w_key].float() + delta_fp32).to(torch.bfloat16)
        sd[w_key].copy_(merged_w)
        merged_count += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_model.save_pretrained(str(out_dir), safe_serialization=True)
    # Copy tokenizer, processor, chat template, generation_config files verbatim from base_model_path
    for extra_fn in base_model_path.iterdir():
        if extra_fn.is_file() and (
            extra_fn.name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "preprocessor", "video_preprocessor", "chat_template", "generation_config"))
        ):
            shutil.copy2(extra_fn, out_dir / extra_fn.name)

    del clean_model
    # Reload merged model on GPU to verify parity in identical HF runtime
    reloaded = AutoModelForImageTextToText.from_pretrained(
        str(out_dir),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    reloaded.eval()

    max_abs_diff = 0.0
    mean_abs_diff_acc = []
    top1_matches = 0
    top1_total = 0
    for s, ref_log in zip(parity_samples[:6], ref_logits_list):
        ids = torch.tensor([s["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        h = reloaded.model.language_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        m_log = reloaded.lm_head(h).float().cpu()
        diff = (m_log - ref_log).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        mean_abs_diff_acc.append(float(diff.mean().item()))
        ref_top1 = ref_log.argmax(dim=-1)
        m_top1 = m_log.argmax(dim=-1)
        top1_matches += int((ref_top1 == m_top1).sum().item())
        top1_total += int(ref_top1.numel())

    del reloaded
    torch.cuda.empty_cache()

    parity = {
        "merged_linear_tensors": merged_count,
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": sum(mean_abs_diff_acc) / max(len(mean_abs_diff_acc), 1),
        "top1_token_agreement": top1_matches / max(top1_total, 1),
    }
    if parity["top1_token_agreement"] < 0.95 or parity["mean_abs_logit_diff"] > 0.20:
        raise RuntimeError(f"FP32 merge parity gate failed: {parity}")
    return parity


def main() -> int:
    ap = argparse.ArgumentParser(description="Governed Weight-Space ReST^EM Trainer for Qwen3.5 SLMs")
    ap.add_argument("--base-model", required=True, help="Path to pinned base M_0")
    ap.add_argument("--train-file", required=True, help="Verified cumulative trace bank JSONL")
    ap.add_argument("--output-dir", required=True, help="Output standalone merged BF16 checkpoint dir")
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--mode", choices=["lora", "full_ft_lm"], default="lora")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--global-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261118)
    args = ap.parse_args()

    rank, world_size, local_rank = _init_dist()
    device = torch.device(f"cuda:{local_rank}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    t0 = time.monotonic()
    base_path = Path(args.base_model)
    tok = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    dataset, data_audit = _tokenize_and_audit_dataset(
        Path(args.train_file),
        tok,
        thinking=args.thinking,
        max_seq_len=args.max_seq_len,
    )
    if rank == 0:
        print(json.dumps({"event": "DATA_AUDIT_PASSED", **data_audit}), flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        str(base_path),
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

    # Unconditionally freeze vision tower & any MTP parameters
    for pname, param in model.named_parameters():
        if "visual" in pname or pname.startswith("mtp."):
            param.requires_grad = False

    if args.mode == "lora":
        lcfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=HYBRID_LM_LINEAR_REGEX,
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        # Verify architecture coverage across Gated-DeltaNet, Self-Attn, and MLP
        t_names = [n for n, p in model.named_parameters() if p.requires_grad]
        n_lin_attn = sum(1 for n in t_names if ".linear_attn." in n)
        n_self_attn = sum(1 for n in t_names if ".self_attn." in n)
        n_mlp = sum(1 for n in t_names if ".mlp." in n)
        n_bad = sum(1 for n in t_names if any(x in n for x in ("visual", "mtp.", "embed_tokens", "lm_head")))
        if not (n_lin_attn > 0 and n_self_attn > 0 and n_mlp > 0 and n_bad == 0):
            raise RuntimeError(
                f"Hybrid LoRA coverage check failed: lin_attn={n_lin_attn}, self_attn={n_self_attn}, mlp={n_mlp}, bad={n_bad}"
            )
    else:
        # Full-FT of language_model layers + final norm ONLY (keep embeddings/lm_head/visual/mtp frozen)
        for pname, param in model.named_parameters():
            if pname.startswith("model.language_model.layers.") or pname.startswith("model.language_model.norm."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    fwd_mod = _ChunkedCEForward(model)
    ddp_model: torch.nn.Module = (
        DDP(fwd_mod, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else fwd_mod
    )

    grad_accum = max(1, args.global_batch_size // (args.micro_batch_size * world_size))
    steps_per_epoch = max(1, math.ceil(len(dataset) / (args.micro_batch_size * world_size * grad_accum)))
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(args.warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    pad_id = EOS_ENDOFTEXT_ID
    step_losses: list[float] = []
    global_step = 0

    for ep in range(args.epochs):
        rng = random.Random(args.seed + ep * 1009)
        ep_rows = list(dataset)
        rng.shuffle(ep_rows)
        # Pad ep_rows cleanly so every rank runs identical micro-steps
        per_step_items = args.micro_batch_size * world_size * grad_accum
        rem = len(ep_rows) % per_step_items
        if rem != 0:
            ep_rows.extend(ep_rows[: (per_step_items - rem)])

        for s_idx in range(0, len(ep_rows), per_step_items):
            group = ep_rows[s_idx : s_idx + per_step_items]
            # Sort within global update group by length so DDP ranks have balanced sequence lengths
            group.sort(key=lambda x: len(x["input_ids"]), reverse=True)
            rank_slice = group[rank::world_size]

            optimizer.zero_grad(set_to_none=True)
            accum_nll = torch.zeros((), device=device, dtype=torch.float32)
            accum_wtoks = torch.zeros((), device=device, dtype=torch.float32)

            # Precompute local denominator & run micro-batches
            micro_batches = [
                rank_slice[m : m + args.micro_batch_size]
                for m in range(0, len(rank_slice), args.micro_batch_size)
            ]
            # Compute exact global weighted token denominator across ranks first for unbiased loss scaling
            local_wtok_est = sum(
                sum(1.0 for y in r["labels"][1:] if y != -100) * r["weight"]
                for r in rank_slice
            )
            denom_t = torch.tensor(local_wtok_est, device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(denom_t, op=dist.ReduceOp.SUM)
            global_denom = torch.clamp(denom_t, min=1.0)

            for mb_idx, mb in enumerate(micro_batches):
                max_l = max(len(x["input_ids"]) for x in mb)
                ids_t = torch.full((len(mb), max_l), pad_id, dtype=torch.long, device=device)
                mask_t = torch.zeros((len(mb), max_l), dtype=torch.long, device=device)
                lbl_t = torch.full((len(mb), max_l), -100, dtype=torch.long, device=device)
                w_t = torch.zeros((len(mb),), dtype=torch.float32, device=device)
                for b_i, item in enumerate(mb):
                    L = len(item["input_ids"])
                    ids_t[b_i, :L] = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
                    mask_t[b_i, :L] = 1
                    lbl_t[b_i, :L] = torch.tensor(item["labels"], dtype=torch.long, device=device)
                    w_t[b_i] = float(item["weight"])

                sync_ctx = (
                    ddp_model.no_sync()
                    if (isinstance(ddp_model, DDP) and mb_idx < len(micro_batches) - 1)
                    else torch.enable_grad()
                )
                with sync_ctx:
                    wnll, wtoks = ddp_model(ids_t, mask_t, lbl_t, w_t)
                    # Multiply by world_size because DDP averages gradients across ranks
                    scaled_loss = (wnll * world_size) / global_denom
                    scaled_loss.backward()
                    accum_nll += wnll.detach()
                    accum_wtoks += wtoks.detach()
                del ids_t, mask_t, lbl_t, w_t, wnll, wtoks, scaled_loss

            grad_norm = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0).item())
            optimizer.step()
            scheduler.step()
            global_step += 1

            if world_size > 1:
                dist.all_reduce(accum_nll, op=dist.ReduceOp.SUM)
                dist.all_reduce(accum_wtoks, op=dist.ReduceOp.SUM)
            step_loss = float((accum_nll / torch.clamp(accum_wtoks, min=1.0)).item())
            step_losses.append(step_loss)
            if rank == 0 and (global_step == 1 or global_step % 5 == 0 or global_step == total_steps):
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "epoch": ep + 1,
                            "loss": round(step_loss, 5),
                            "grad_norm": round(grad_norm, 4),
                            "lr": scheduler.get_last_lr()[0],
                            "peak_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
                        }
                    ),
                    flush=True,
                )

    train_wall_s = time.monotonic() - t0
    peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))

    if world_size > 1:
        dist.barrier()

    parity_info: dict[str, Any] = {}
    if rank == 0:
        out_dir = Path(args.output_dir)
        unwrapped_fwd = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
        raw_model = unwrapped_fwd.model
        if args.mode == "lora":
            parity_info = _merge_lora_fp32_and_verify_parity(
                raw_model,
                base_path,
                out_dir,
                dataset,
                device,
            )
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_model.config.use_cache = True
            raw_model.save_pretrained(str(out_dir), safe_serialization=True)
            for extra_fn in base_path.iterdir():
                if extra_fn.is_file() and extra_fn.name.startswith(
                    ("tokenizer", "vocab", "merges", "special_tokens", "preprocessor", "video_preprocessor", "chat_template", "generation_config")
                ):
                    shutil.copy2(extra_fn, out_dir / extra_fn.name)
            parity_info = {"mode": "full_ft_lm_direct_bf16_save"}

        summary = {
            "base_model": str(base_path),
            "train_file": str(args.train_file),
            "output_dir": str(out_dir),
            "mode": args.mode,
            "thinking": args.thinking,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "epochs": args.epochs,
            "lr": args.lr,
            "global_steps": global_step,
            "initial_loss": step_losses[0] if step_losses else None,
            "final_loss": step_losses[-1] if step_losses else None,
            "min_loss": min(step_losses) if step_losses else None,
            "train_wall_seconds": train_wall_s,
            "gpu_hours": (train_wall_s * world_size) / 3600.0,
            "peak_allocated_gib": peak_gib,
            "data_audit": data_audit,
            "merge_parity": parity_info,
        }
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "TRAIN_COMPLETE", **summary}, indent=2), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

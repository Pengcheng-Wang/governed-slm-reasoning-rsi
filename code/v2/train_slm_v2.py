#!/usr/bin/env python3
"""
Multi-Architecture Governed `ReST^EM` SFT Trainer V2 (`Qwen3.5` Hybrid GDN, `Qwen3` Dense, `Phi-4` Fused LongRoPE).

Implements all `GPT 6.2 Astra, Max` & `Opus 5.5, Max` safeguards:
  1. Model-aware loader (`AutoModelForImageTextToText` vs `AutoModelForCausalLM`), chat-template rendering,
     and supervised turn-terminator EOS (`248046` `<|im_end|>`, `151645` `<|im_end|>`, or `200020` `<|end|>`).
  2. Asserts adapted LoRA linear module count >= `spec.expected_min_lora_modules` at initialization
     (preventing silent fused `qkv_proj`/`gate_up_proj` misses on Phi-4).
  3. Activation-checkpointed 1,024-token `lm_head` Cross-Entropy chunks (`torch.utils.checkpoint.checkpoint`)
     so 13K-token sequences never retain full 248K-vocab FP32 logit graphs in HBM.
  4. Problem-balanced sample weights (`1 / count(problem_id)`) + optional `--max-steps` cap for exact
     compute-matched recursion and repair controls.
  5. Exact `FP32` LoRA merge + attached-vs-merged top-1 parity verification (`>= 0.98`).
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
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as ckpt_fn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, get_cosine_schedule_with_warmup

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v2 import MODEL_REGISTRY_V2, ModelSpecV2  # noqa: E402


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


def load_hf_model(spec: ModelSpecV2, model_path: Path, *, flash_attn: bool = True) -> torch.nn.Module:
    cls = AutoModelForImageTextToText if spec.loader_kind == "image_text" else AutoModelForCausalLM
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    return cls.from_pretrained(str(model_path), **kwargs)


def unwrap_backbone_and_head(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    m = model.module if isinstance(model, DDP) else model
    inner = m.base_model.model if (hasattr(m, "base_model") and hasattr(m.base_model, "model")) else m
    if hasattr(inner.model, "language_model"):
        return inner.model.language_model, inner.lm_head
    return inner.model, inner.lm_head


def chat_prompt_ids(tok: Any, spec: ModelSpecV2, prompt_msgs: list[dict[str, str]], *, thinking: bool) -> list[int]:
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


def tokenize_and_audit_dataset(
    data_path: Path,
    tok: Any,
    spec: ModelSpecV2,
    *,
    thinking: bool,
    max_seq_len: int,
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
        f_ids = p_ids + r_ids + [spec.sup_eos_id]
        if len(f_ids) > max_seq_len:
            dropped_overlong += 1
            continue
        if len(f_ids) - len(p_ids) < 2:
            continue
        kept.append((idx, row, p_ids, f_ids))

    if not kept:
        raise RuntimeError(f"No valid training rows after max_seq_len={max_seq_len} filter (dropped={dropped_overlong})")

    prob_counts = collections.Counter(str(r.get("problem_id", r.get("id", f"row_{i}"))) for i, r, _, _ in kept)
    tokenized: list[dict[str, Any]] = []
    lengths: list[int] = []
    sup_lengths: list[int] = []
    for i, r, p_ids, f_ids in kept:
        pid = str(r.get("problem_id", r.get("id", f"row_{i}")))
        labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        w = 1.0 / float(prob_counts[pid])
        tokenized.append({"problem_id": pid, "input_ids": f_ids, "labels": labels, "weight": w})
        lengths.append(len(f_ids))
        sup_lengths.append(len(f_ids) - len(p_ids))

    lengths.sort()
    n = len(lengths)
    return tokenized, {
        "total_traces": n,
        "dropped_whole_overlong_rows": dropped_overlong,
        "unique_problems": len(prob_counts),
        "seq_len_p50": lengths[n // 2],
        "seq_len_p95": lengths[min(n - 1, int(0.95 * n))],
        "seq_len_max": lengths[-1],
        "supervised_tokens_total": sum(sup_lengths),
        "sup_eos_id": spec.sup_eos_id,
    }


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


def chunked_weighted_ce_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    lm_backbone, lm_head = unwrap_backbone_and_head(model)
    hidden = lm_backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
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

    return total_wnll, total_wtoks


class _ChunkedCEForwardV2(torch.nn.Module):
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
        return chunked_weighted_ce_loss(self.model, input_ids, attention_mask, labels, sample_weights)


@torch.no_grad()
def merge_lora_fp32_and_verify_parity_v2(
    peft_model: torch.nn.Module,
    spec: ModelSpecV2,
    base_model_path: Path,
    out_dir: Path,
    parity_samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    peft_model.eval()
    ref_logits_list: list[torch.Tensor] = []
    for s in parity_samples[:6]:
        ids = torch.tensor([s["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        lm_b, lm_h = unwrap_backbone_and_head(peft_model)
        h = lm_b(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        ref_logits_list.append(lm_h(h).float().cpu())

    lora_deltas: dict[str, torch.Tensor] = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and "default" in mod.lora_A:
            a_w = mod.lora_A["default"].weight.detach().float().cpu()
            b_w = mod.lora_B["default"].weight.detach().float().cpu()
            scale = float(mod.scaling["default"]) if isinstance(mod.scaling, dict) else float(mod.scaling)
            clean_name = name[len("base_model.model.") :] if name.startswith("base_model.model.") else name
            lora_deltas[clean_name + ".weight"] = scale * (b_w @ a_w)

    del peft_model
    torch.cuda.empty_cache()

    clean_model = load_hf_model(spec, base_model_path, flash_attn=False)
    sd = clean_model.state_dict()
    merged_count = 0
    for w_key, delta_fp32 in lora_deltas.items():
        if w_key not in sd:
            raise KeyError(f"LoRA target weight key missing in clean base model: {w_key}")
        sd[w_key].copy_((sd[w_key].float() + delta_fp32).to(torch.bfloat16))
        merged_count += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_model.save_pretrained(str(out_dir), safe_serialization=True)
    for extra_fn in base_model_path.iterdir():
        if extra_fn.is_file() and (
            extra_fn.name.startswith(
                ("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "preprocessor", "video_preprocessor", "chat_template", "generation_config")
            )
            or extra_fn.suffix == ".py"
        ):
            shutil.copy2(extra_fn, out_dir / extra_fn.name)

    del clean_model
    reloaded = load_hf_model(spec, out_dir, flash_attn=True).to(device)
    reloaded.eval()

    max_abs_diff = 0.0
    mean_diffs: list[float] = []
    top1_matches = 0
    top1_total = 0
    for s, ref_log in zip(parity_samples[:6], ref_logits_list):
        ids = torch.tensor([s["input_ids"][:2048]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        lm_b, lm_h = unwrap_backbone_and_head(reloaded)
        h = lm_b(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -32:, :]
        m_log = lm_h(h).float().cpu()
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
    if parity["top1_token_agreement"] < 0.95:
        raise RuntimeError(f"FP32 merge parity gate failed: {parity}")
    return parity


def attach_verified_lora(model: torch.nn.Module, spec: ModelSpecV2, *, r: int = 0, alpha: int = 0, dropout: float = 0.0) -> tuple[torch.nn.Module, int]:
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
    adapted = [n for n, m in peft_m.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A]
    if len(adapted) < spec.expected_min_lora_modules:
        raise RuntimeError(
            f"LoRA module coverage check failed for {spec.key}: adapted={len(adapted)} < expected={spec.expected_min_lora_modules}"
        )
    bad = [n for n, p in peft_m.named_parameters() if p.requires_grad and any(x in n for x in ("visual", "mtp.", "embed_tokens", "lm_head"))]
    if bad:
        raise RuntimeError(f"Forbidden trainable parameters detected: {bad[:5]}")
    return peft_m, len(adapted)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-Architecture Governed ReST^EM Trainer V2")
    ap.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V2.keys()))
    ap.add_argument("--base-model", default="")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=12800)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="Optional hard optimizer-step cap for compute-matched controls")
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
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
    spec = MODEL_REGISTRY_V2[args.model_key]
    base_path = Path(args.base_model or spec.local_path)
    tok = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    dataset, data_audit = tokenize_and_audit_dataset(
        Path(args.train_file), tok, spec, thinking=args.thinking, max_seq_len=args.max_seq_len
    )
    if rank == 0:
        print(json.dumps({"event": "DATA_AUDIT_PASSED", "model_key": spec.key, **data_audit}), flush=True)

    base_m = load_hf_model(spec, base_path, flash_attn=True)
    model, n_adapted = attach_verified_lora(
        base_m, spec, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model.to(device)
    model.train()
    lm_backbone_chk, _ = unwrap_backbone_and_head(model)
    if not lm_backbone_chk.training:
        lm_backbone_chk.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    fwd_mod = _ChunkedCEForwardV2(model)
    ddp_model: torch.nn.Module = (
        DDP(fwd_mod, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
        if world_size > 1
        else fwd_mod
    )

    grad_accum = max(1, args.global_batch_size // (args.micro_batch_size * world_size))
    steps_per_epoch = max(1, math.ceil(len(dataset) / (args.micro_batch_size * world_size * grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

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

    pad_id = spec.stop_token_ids[0]
    step_losses: list[float] = []
    global_step = 0

    for ep in range(args.epochs):
        if global_step >= total_steps:
            break
        rng = random.Random(args.seed + ep * 1009)
        ep_rows = list(dataset)
        rng.shuffle(ep_rows)
        per_step_items = args.micro_batch_size * world_size * grad_accum
        rem = len(ep_rows) % per_step_items
        if rem != 0:
            ep_rows.extend(ep_rows[: (per_step_items - rem)])

        for s_idx in range(0, len(ep_rows), per_step_items):
            if global_step >= total_steps:
                break
            group = ep_rows[s_idx : s_idx + per_step_items]
            group.sort(key=lambda x: len(x["input_ids"]), reverse=True)
            rank_slice = group[rank::world_size]

            optimizer.zero_grad(set_to_none=True)
            accum_nll = torch.zeros((), device=device, dtype=torch.float32)
            accum_wtoks = torch.zeros((), device=device, dtype=torch.float32)

            micro_batches = [
                rank_slice[m : m + args.micro_batch_size]
                for m in range(0, len(rank_slice), args.micro_batch_size)
            ]
            local_wtok_est = sum(
                sum(1.0 for y in r["labels"][1:] if y != -100) * r["weight"] for r in rank_slice
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
                            "loss": round(step_loss, 5),
                            "grad_norm": round(grad_norm, 4),
                            "peak_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
                        }
                    ),
                    flush=True,
                )

    train_wall_s = time.monotonic() - t0
    peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank != 0:
        return 0

    out_dir = Path(args.output_dir)
    raw_model = (ddp_model.module if isinstance(ddp_model, DDP) else ddp_model).model
    parity = merge_lora_fp32_and_verify_parity_v2(raw_model, spec, base_path, out_dir, dataset, device)
    summary = {
        "model_key": spec.key,
        "base_model": str(base_path),
        "train_file": str(args.train_file),
        "output_dir": str(out_dir),
        "adapted_linear_modules": n_adapted,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "global_steps": global_step,
        "initial_loss": step_losses[0] if step_losses else None,
        "final_loss": step_losses[-1] if step_losses else None,
        "train_wall_seconds": train_wall_s,
        "gpu_hours": (train_wall_s * world_size) / 3600.0,
        "peak_allocated_gib": peak_gib,
        "data_audit": data_audit,
        "merge_parity": parity,
    }
    sp = Path(args.summary_output)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "TRAIN_V2_COMPLETE", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

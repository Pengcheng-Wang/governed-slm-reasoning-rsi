#!/usr/bin/env python3
"""
Phase V4 Parallel Inference & Rollout Engine (`vLLM` + 8-GPU Data-Parallel `HF BF16` for `Inkling2Loop`).

Invariants:
  1. Strict label-blind worker input projection (`{"id", "messages"}` only — zero `gold` or `formal_statement`
     passed to GPU workers).
  2. Deterministic per-`(item_id, sample_idx)` SHA-256 RNG seeding (`A/A` repeatable).
  3. Zero-padding group batching for `hf_dp` (`Inkling2LoopForConditionalGeneration`): groups prompts of
     identical token length (`k` rollouts per prompt share exact prompt length -> `0` pad tokens in prompt,
     eliminating any `position_ids` or `ShortConvolution` left-pad state drift!).
  4. Preserves internal reasoning channel delimiters (`skip_special_tokens=False` in `hf_dp` and `vLLM`
     `spaces_between_special_tokens=False`) up to the first `stop_token_id`, so `channel_normalizer_v4`
     deterministically separates `<|content_thinking|>`, `<|channel>thought`, and `to=self<|message|>`
     from the final user-facing response!
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gc
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v4 import MODEL_REGISTRY_V4, ModelSpecV4  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_prompt_v4(
    tok: Any,
    spec: ModelSpecV4,
    msgs: list[dict[str, str]],
    *,
    chat_kwargs: dict[str, Any],
) -> str:
    kwargs = dict(chat_kwargs)
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def _worker_run_vllm(cfg: dict[str, Any]) -> dict[str, Any]:
    wid = int(cfg["worker_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Isolate per-worker TorchInductor / Triton / vLLM compile caches & ports to prevent race conditions across TP shards
    w_cache = Path(f"/tmp/vllm_wcache_{os.getuid()}_w{wid}")
    w_cache.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(w_cache / "vllm")
    os.environ["TRITON_CACHE_DIR"] = str(w_cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(w_cache / "inductor")
    base_port = 29600 + wid * 37 + (os.getpid() % 29)
    os.environ["MASTER_PORT"] = str(base_port)
    os.environ["VLLM_PORT"] = str(base_port + 1000)
    if wid > 0:
        time.sleep(wid * 3.0)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    spec = MODEL_REGISTRY_V4[cfg["model_key"]]
    model_path = cfg["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": int(cfg["tp"]),
        "dtype": "bfloat16",
        "gpu_memory_utilization": float(cfg["gpu_mem_util"]),
        "max_model_len": int(cfg["max_model_len"]),
        "trust_remote_code": True,
        "enforce_eager": bool(cfg.get("enforce_eager", False)),
        "disable_custom_all_reduce": int(cfg["tp"]) > 1,
    }
    if spec.vllm_lm_only:
        llm_kwargs["language_model_only"] = True

    llm = LLM(**llm_kwargs)

    shard_items: list[dict[str, Any]] = cfg["items"]
    k = int(cfg["k"])
    chat_kwargs: dict[str, Any] = dict(cfg["chat_kwargs"])
    system_prompt = cfg.get("system_prompt") or ""

    prompts: list[str] = []
    sampling_params_list: list[SamplingParams] = []
    req_meta: list[tuple[str, int]] = []

    for item in shard_items:
        msgs = list(item["messages"])
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        rendered = render_prompt_v4(tok, spec, msgs, chat_kwargs=chat_kwargs)
        for s_idx in range(k):
            h_int = int(hashlib.sha256(f"{cfg['seed']}:{item['id']}:{s_idx}".encode("utf-8")).hexdigest()[:8], 16)
            sp = SamplingParams(
                n=1,
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
                top_k=int(cfg["top_k"]),
                presence_penalty=float(cfg["presence_penalty"]),
                max_tokens=int(cfg["max_tokens"]),
                stop_token_ids=list(spec.stop_token_ids) if spec.stop_token_ids else None,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                seed=h_int,
            )
            prompts.append(rendered)
            sampling_params_list.append(sp)
            req_meta.append((str(item["id"]), s_idx))

    t_gen0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
    gen_sec = time.monotonic() - t_gen0

    shard_out = Path(cfg["shard_out"])
    shard_out.parent.mkdir(parents=True, exist_ok=True)
    total_comp_toks = 0
    total_prompt_toks = 0
    stop_cnt = 0
    len_cnt = 0

    with gzip.open(shard_out, "wt", encoding="utf-8") as gz:
        for (item_id, s_idx), req_out in zip(req_meta, outputs):
            comp = req_out.outputs[0]
            fr = str(comp.finish_reason or "unknown")
            p_toks = len(req_out.prompt_token_ids) if req_out.prompt_token_ids else 0
            c_toks = len(comp.token_ids) if comp.token_ids else 0
            # Keep exact generated token_ids (including terminal stop token when fr=="stop") for lossless RLVR replay,
            # while decoding `text` without the trailing stop token so channel delimiters stay clean for verifiers.
            full_tids = [int(x) for x in comp.token_ids] if comp.token_ids else []
            out_tids = list(full_tids)
            if out_tids and spec.stop_token_ids and out_tids[-1] in set(spec.stop_token_ids):
                out_tids = out_tids[:-1]
            decoded_txt = tok.decode(out_tids, skip_special_tokens=False, clean_up_tokenization_spaces=False) if out_tids else comp.text
            total_prompt_toks += p_toks
            total_comp_toks += c_toks
            if fr == "stop":
                stop_cnt += 1
            elif fr == "length":
                len_cnt += 1
            row = {
                "id": item_id,
                "sample_idx": s_idx,
                "text": decoded_txt,
                "token_ids": full_tids,
                "finish_reason": fr,
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
            }
            gz.write(json.dumps(row, ensure_ascii=False) + "\n")

    del llm
    gc.collect()
    return {
        "worker_id": cfg["worker_id"],
        "backend": "vllm",
        "n_requests": len(req_meta),
        "gen_seconds": gen_sec,
        "wall_seconds": time.monotonic() - t0,
        "prompt_tokens": total_prompt_toks,
        "completion_tokens": total_comp_toks,
        "stop_count": stop_cnt,
        "length_count": len_cnt,
        "shard_out": str(shard_out),
    }


def _worker_run_hf_dp(cfg: dict[str, Any]) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    from fast_moe_patch_v4 import install_fast_grouped_moe_kernels

    install_fast_grouped_moe_kernels()
    t0 = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")

    spec = MODEL_REGISTRY_V4[cfg["model_key"]]
    model_path = cfg["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        model.model.vision_tower = None
    model.to(device)
    model.eval()

    lm_backbone = model.model.language_model if (hasattr(model, "model") and hasattr(model.model, "language_model")) else model.model
    lm_head = model.lm_head
    tcfg = getattr(model.config, "text_config", model.config)
    out_mult = getattr(tcfg, "output_multiplier", None)
    softcap = getattr(tcfg, "final_logit_softcapping", None)
    out_mult_f = float(out_mult) if out_mult is not None else None
    softcap_f = float(softcap) if softcap is not None else None

    def _compact_dynamic_cache(cache_obj: Any, keep_idx: torch.Tensor) -> None:
        for lyr in cache_obj.layers:
            if getattr(lyr, "keys", None) is not None and lyr.keys.numel() > 0:
                lyr.keys = lyr.keys.index_select(0, keep_idx).contiguous()
                lyr.values = lyr.values.index_select(0, keep_idx).contiguous()
            for c_k, c_v in list(getattr(lyr, "conv_states", {}).items()):
                if torch.is_tensor(c_v):
                    lyr.conv_states[c_k] = c_v.index_select(0, keep_idx).contiguous()

    def _extract_cache_slice_to_cpu(cache_obj: Any, idx_gpu: torch.Tensor) -> list[dict[str, Any]]:
        snaps: list[dict[str, Any]] = []
        for lyr in cache_obj.layers:
            has_kv = getattr(lyr, "keys", None) is not None and lyr.keys.numel() > 0
            snaps.append(
                {
                    "keys": lyr.keys.index_select(0, idx_gpu).cpu() if has_kv else None,
                    "values": lyr.values.index_select(0, idx_gpu).cpu() if has_kv else None,
                    "cum_len": getattr(lyr, "cumulative_length", None),
                    "conv": {
                        c_k: c_v.index_select(0, idx_gpu).cpu()
                        for c_k, c_v in getattr(lyr, "conv_states", {}).items()
                        if torch.is_tensor(c_v)
                    },
                }
            )
        return snaps

    def _restore_cache_slice_from_cpu(cache_obj: Any, snaps: list[dict[str, Any]]) -> None:
        for lyr, sn in zip(cache_obj.layers, snaps):
            if sn["keys"] is not None:
                lyr.keys = sn["keys"].to(device, non_blocking=True).contiguous()
                lyr.values = sn["values"].to(device, non_blocking=True).contiguous()
            if sn["cum_len"] is not None and hasattr(lyr, "cumulative_length"):
                lyr.cumulative_length = int(sn["cum_len"])
            if hasattr(lyr, "conv_states"):
                lyr.conv_states = {
                    c_k: c_v.to(device, non_blocking=True).contiguous() for c_k, c_v in sn["conv"].items()
                }

    def _sample_last_hidden(
        last_h: torch.Tensor,
        *,
        temp: float,
        top_p: float,
        top_k: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        logits = torch.nn.functional.linear(last_h, lm_head.weight, lm_head.bias).float()
        if out_mult_f is not None:
            logits = logits * out_mult_f
        if softcap_f is not None:
            logits = softcap_f * torch.tanh(logits / softcap_f)
        if temp <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temp
        if top_k > 0:
            k_eff = min(top_k, int(logits.shape[-1]))
            kth = torch.topk(logits, k_eff, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            s_log, s_idx = torch.sort(logits, descending=True, dim=-1)
            c_prob = torch.cumsum(torch.softmax(s_log, dim=-1), dim=-1)
            rem = c_prob > top_p
            rem[:, 1:] = rem[:, :-1].clone()
            rem[:, 0] = False
            s_log = s_log.masked_fill(rem, float("-inf"))
            probs = torch.softmax(s_log, dim=-1)
            pick = torch.multinomial(probs, num_samples=1, generator=gen)
            return s_idx.gather(-1, pick)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=gen)

    shard_items: list[dict[str, Any]] = cfg["items"]
    k = int(cfg["k"])
    max_tokens = int(cfg["max_tokens"])
    chat_kwargs: dict[str, Any] = dict(cfg["chat_kwargs"])
    system_prompt = cfg.get("system_prompt") or ""
    stop_set = set(spec.stop_token_ids)
    stop_ids_gpu = torch.tensor(sorted(stop_set), dtype=torch.long, device=device)
    temp_f = float(cfg["temperature"])
    top_p_f = float(cfg["top_p"])
    top_k_i = int(cfg["top_k"])

    # Sort prompts by token length and dynamically size left-padded batches within safe KV HBM budget (~20 GB)
    encoded_items: list[tuple[int, str, list[int]]] = []
    for item in shard_items:
        msgs = list(item["messages"])
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        rendered = render_prompt_v4(tok, spec, msgs, chat_kwargs=chat_kwargs)
        p_ids = [int(x) for x in tok.encode(rendered, add_special_tokens=False)]
        encoded_items.append((len(p_ids), str(item["id"]), p_ids))
    encoded_items.sort(key=lambda t: (t[0], t[1]))

    shard_out = Path(cfg["shard_out"])
    shard_out.parent.mkdir(parents=True, exist_ok=True)
    total_comp_toks = 0
    total_prompt_toks = 0
    stop_cnt = 0
    len_cnt = 0
    n_reqs = 0

    t_gen0 = time.monotonic()
    rows_out: list[dict[str, Any]] = []
    # Inkling-2-Mini weights occupy ~50.2 GiB BF16. Keep `memory_allocated <= 60 GiB` (~66 GiB physical HBM,
    # leaving 14 GiB safety headroom for CUDA context + allocator blocks + `torch.cat` KV growth) via 16-seq
    # chunked prefill and automatic 60 GiB CPU Park-and-Resume.
    max_wave_seqs = 128
    pf_chunk_seqs = 16
    hbm_park_threshold_bytes = int(62.5 * (1024**3))
    pad_id = int(spec.pad_token_id)

    cursor = 0
    while cursor < len(encoded_items):
        dyn_groups = max(1, max_wave_seqs // max(1, k))
        sub_items = encoded_items[cursor : cursor + dyn_groups]
        cursor += len(sub_items)

        max_plen = max(x[0] for x in sub_items)
        id_sig = "|".join(iid for _, iid, _ in sub_items)
        b_seed = int(hashlib.sha256(f"{cfg['seed']}:{id_sig}".encode("utf-8")).hexdigest()[:8], 16)
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(b_seed)

        expanded_ids: list[list[int]] = []
        expanded_mask: list[list[int]] = []
        meta_batch: list[tuple[str, int, int]] = []
        for orig_plen, iid, p_ids in sub_items:
            pad_n = max_plen - orig_plen
            padded_row = ([pad_id] * pad_n + p_ids) if pad_n > 0 else p_ids
            mask_row = ([0] * pad_n + [1] * orig_plen) if pad_n > 0 else ([1] * orig_plen)
            for s_idx in range(k):
                expanded_ids.append(padded_row)
                expanded_mask.append(mask_row)
                meta_batch.append((iid, s_idx, orig_plen))

        b_init = len(meta_batch)
        in_tensor = torch.tensor(expanded_ids, dtype=torch.long, device=device)
        cur_mask = torch.tensor(expanded_mask, dtype=torch.long, device=device)

        gen_buf = torch.full((b_init, max_tokens), pad_id, dtype=torch.long, device=device)
        stop_step_buf = torch.full((b_init,), max_tokens, dtype=torch.long, device=device)
        active_orig_idx = torch.arange(b_init, dtype=torch.long, device=device)

        with torch.inference_mode():
            pos0 = cur_mask.long().cumsum(-1) - 1
            pos0.masked_fill_(cur_mask == 0, 1)
            cache = None
            last_h_chunks: list[torch.Tensor] = []
            for pf_s in range(0, b_init, pf_chunk_seqs):
                pf_e = min(pf_s + pf_chunk_seqs, b_init)
                out_pf = lm_backbone(
                    input_ids=in_tensor[pf_s:pf_e],
                    attention_mask=cur_mask[pf_s:pf_e],
                    position_ids=pos0[pf_s:pf_e],
                    use_cache=True,
                )
                last_h_chunks.append(out_pf.last_hidden_state[:, -1, :].contiguous())
                if cache is None:
                    cache = out_pf.cache
                else:
                    for lyr0, lyr1 in zip(cache.layers, out_pf.cache.layers):
                        if getattr(lyr1, "keys", None) is not None and lyr1.keys.numel() > 0:
                            lyr0.keys = torch.cat([lyr0.keys, lyr1.keys], dim=0)
                            lyr0.values = torch.cat([lyr0.values, lyr1.values], dim=0)
                        for c_k, c_v in getattr(lyr1, "conv_states", {}).items():
                            if torch.is_tensor(c_v):
                                lyr0.conv_states[c_k] = torch.cat([lyr0.conv_states[c_k], c_v], dim=0)
                    del out_pf
            cur_pos = pos0[:, -1:]
            last_h_all = torch.cat(last_h_chunks, dim=0)
            del last_h_chunks, in_tensor, pos0
            torch.cuda.empty_cache()

            next_tok = _sample_last_hidden(
                last_h_all,
                temp=temp_f,
                top_p=top_p_f,
                top_k=top_k_i,
                gen=gen_rng,
            )
            del last_h_all

            parked_stack: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]] = []
            start_step = 0
            while True:
                for step_t in range(start_step, max_tokens):
                    nt_1d = next_tok.squeeze(-1)
                    gen_buf[active_orig_idx, step_t] = nt_1d
                    is_stop = torch.isin(nt_1d, stop_ids_gpu)
                    cand_step = torch.where(
                        is_stop,
                        torch.full_like(nt_1d, step_t),
                        torch.full_like(nt_1d, max_tokens),
                    )
                    stop_step_buf[active_orig_idx] = torch.minimum(stop_step_buf[active_orig_idx], cand_step)

                    if ((step_t + 1) % 16 == 0) or (step_t + 1 == max_tokens):
                        alive_mask = stop_step_buf[active_orig_idx] >= max_tokens
                        n_alive = int(alive_mask.sum().item())
                        if n_alive == 0:
                            break
                        if n_alive < int(active_orig_idx.shape[0]) and (step_t + 1 < max_tokens):
                            keep_local = alive_mask.nonzero(as_tuple=False).squeeze(-1)
                            _compact_dynamic_cache(cache, keep_local)
                            active_orig_idx = active_orig_idx.index_select(0, keep_local)
                            cur_mask = cur_mask.index_select(0, keep_local)
                            cur_pos = cur_pos.index_select(0, keep_local)
                            next_tok = next_tok.index_select(0, keep_local)

                        if (
                            (step_t + 1 < max_tokens)
                            and int(active_orig_idx.shape[0]) >= 4
                            and torch.cuda.memory_allocated(device) > hbm_park_threshold_bytes
                        ):
                            cur_n = int(active_orig_idx.shape[0])
                            half_n = cur_n // 2
                            keep_half = torch.arange(0, half_n, dtype=torch.long, device=device)
                            park_half = torch.arange(half_n, cur_n, dtype=torch.long, device=device)
                            park_snaps = _extract_cache_slice_to_cpu(cache, park_half)
                            parked_stack.append(
                                (
                                    step_t + 1,
                                    active_orig_idx.index_select(0, park_half).cpu(),
                                    cur_mask.index_select(0, park_half).cpu(),
                                    cur_pos.index_select(0, park_half).cpu(),
                                    next_tok.index_select(0, park_half).cpu(),
                                    park_snaps,
                                )
                            )
                            _compact_dynamic_cache(cache, keep_half)
                            active_orig_idx = active_orig_idx.index_select(0, keep_half)
                            cur_mask = cur_mask.index_select(0, keep_half)
                            cur_pos = cur_pos.index_select(0, keep_half)
                            next_tok = next_tok.index_select(0, keep_half)
                            torch.cuda.empty_cache()

                    if step_t + 1 < max_tokens:
                        cur_mask = torch.cat(
                            [cur_mask, torch.ones((next_tok.shape[0], 1), dtype=torch.long, device=device)],
                            dim=1,
                        )
                        cur_pos = cur_pos + 1
                        out_st = lm_backbone(
                            input_ids=next_tok,
                            attention_mask=cur_mask,
                            position_ids=cur_pos,
                            cache=cache,
                            use_cache=True,
                        )
                        next_tok = _sample_last_hidden(
                            out_st.last_hidden_state[:, -1, :],
                            temp=temp_f,
                            top_p=top_p_f,
                            top_k=top_k_i,
                            gen=gen_rng,
                        )

                if not parked_stack:
                    break
                start_step, p_orig, p_mask, p_pos, p_nt, p_snaps = parked_stack.pop()
                _restore_cache_slice_from_cpu(cache, p_snaps)
                active_orig_idx = p_orig.to(device, non_blocking=True)
                cur_mask = p_mask.to(device, non_blocking=True)
                cur_pos = p_pos.to(device, non_blocking=True)
                next_tok = p_nt.to(device, non_blocking=True)
                del p_snaps, p_orig, p_mask, p_pos, p_nt

        gen_cpu = gen_buf.cpu().tolist()
        stop_steps_cpu = stop_step_buf.cpu().tolist()
        del cache, gen_buf, stop_step_buf, cur_mask, cur_pos
        torch.cuda.empty_cache()

        for row_i, (iid, s_idx, orig_plen) in enumerate(meta_batch):
            raw_gen = gen_cpu[row_i]
            st_pos = int(stop_steps_cpu[row_i])
            if st_pos < max_tokens:
                fr = "stop"
                c_toks = st_pos + 1
                content_tids = raw_gen[:st_pos]
                full_tids = raw_gen[: st_pos + 1]
                stop_cnt += 1
            else:
                c_toks = max_tokens
                fr = "length"
                len_cnt += 1
                content_tids = raw_gen[:max_tokens]
                full_tids = raw_gen[:max_tokens]

            txt = tok.decode(content_tids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            total_prompt_toks += orig_plen
            total_comp_toks += c_toks
            n_reqs += 1
            rows_out.append(
                {
                    "id": iid,
                    "sample_idx": s_idx,
                    "text": txt,
                    "token_ids": full_tids,
                    "finish_reason": fr,
                    "prompt_tokens": orig_plen,
                    "completion_tokens": c_toks,
                }
            )

    gen_sec = time.monotonic() - t_gen0
    rows_out.sort(key=lambda r: (str(r["id"]), int(r["sample_idx"])))
    with gzip.open(shard_out, "wt", encoding="utf-8") as gz:
        for r in rows_out:
            gz.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "worker_id": cfg["worker_id"],
        "backend": "hf_dp",
        "n_requests": n_reqs,
        "gen_seconds": gen_sec,
        "wall_seconds": time.monotonic() - t0,
        "prompt_tokens": total_prompt_toks,
        "completion_tokens": total_comp_toks,
        "stop_count": stop_cnt,
        "length_count": len_cnt,
        "shard_out": str(shard_out),
    }


def _sync_remote_model_dir_if_missing(rhost: str, model_path: str) -> None:
    mp = Path(model_path).resolve()
    chk = subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", rhost, f"test -f {mp.as_posix()}/config.json"],
        capture_output=True,
        text=True,
    )
    if chk.returncode == 0:
        return
    cmd = (
        f"tar -C {mp.parent.as_posix()} --exclude=.cache -cf - {mp.name} "
        f"| ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com {rhost} "
        f"'mkdir -p {mp.parent.as_posix()} && tar -C {mp.parent.as_posix()} -xf -'"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)


def _probe_node1_completely_idle() -> bool:
    chk = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "node-1",
            "pgrep -f 'run_v4_worker|train_sft_v4|train_rlvr_v4|eval_engine_v4' >/dev/null 2>&1 && echo BUSY || nvidia-smi --query-compute-apps=pid --format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    return chk.returncode == 0 and chk.stdout.strip() == ""


def _spawn_worker_proc(w_cfg_path: Path) -> dict[str, Any]:
    wcfg = json.loads(w_cfg_path.read_text(encoding="utf-8"))
    rhost = str(wcfg.get("remote_host") or "").strip()
    if not rhost:
        r = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-config", str(w_cfg_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"Worker {w_cfg_path.name} failed (rc={r.returncode}):\nSTDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2500:]}"
            )
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
        return json.loads(lines[-1])

    subprocess.run(["scp", "-q", str(w_cfg_path), f"{rhost}:{w_cfg_path.as_posix()}"], check=True)
    r = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            rhost,
            f"{sys.executable} -B {Path(__file__).resolve().as_posix()} --worker-config {w_cfg_path.as_posix()}",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Remote worker {rhost}:{w_cfg_path.name} failed (rc={r.returncode}):\nSTDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2500:]}"
        )
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    res = json.loads(lines[-1])
    shard_p = Path(res["shard_out"])
    subprocess.run(["scp", "-q", f"{rhost}:{shard_p.as_posix()}", str(shard_p)], check=True)
    subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", rhost, f"rm -f {w_cfg_path.as_posix()} {shard_p.as_posix()}"],
        capture_output=True,
        text=True,
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V4 Parallel Inference & Rollout Engine")
    ap.add_argument("--worker-config", default="")
    ap.add_argument("--model-key", default="inkling2_mini", choices=sorted(MODEL_REGISTRY_V4.keys()))
    ap.add_argument("--model-path", default="")
    ap.add_argument("--backend-override", choices=["", "vllm", "hf_dp"], default="")
    ap.add_argument("--split-file", default="")
    ap.add_argument("--output", default="")
    ap.add_argument("--summary-output", default="")
    ap.add_argument("--track", choices=["math", "lean"], default="math")
    ap.add_argument("--chat-kwargs-json", default="", help="Optional JSON override for chat_template kwargs")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tp", type=int, default=0)
    ap.add_argument("--hf-batch-groups", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--temperature", type=float, default=-1.0)
    ap.add_argument("--top-p", type=float, default=-1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=20260325)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    if args.worker_config:
        wcfg = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
        res = _worker_run_hf_dp(wcfg) if wcfg["backend"] == "hf_dp" else _worker_run_vllm(wcfg)
        print(json.dumps(res))
        return 0

    spec = MODEL_REGISTRY_V4[args.model_key]
    model_path = args.model_path or spec.local_path
    backend = args.backend_override or spec.eval_backend
    tp = int(args.tp) if int(args.tp) > 0 else (1 if backend == "hf_dp" else spec.tp_eval)
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if backend == "hf_dp" and tp == 1:
        extra_gpus = [g.strip() for g in os.environ.get("V4_HF_DP_EXTRA_GPUS", "").split(",") if g.strip()]
        if not extra_gpus and os.environ.get("V4_HF_DP_AUTO_NODE1", "") == "1" and _probe_node1_completely_idle():
            extra_gpus = [f"node-1:{i}" for i in range(8)]
        for eg in extra_gpus:
            if eg not in gpu_list:
                gpu_list.append(eg)
    if len(gpu_list) % tp != 0:
        raise SystemExit(f"GPU count {len(gpu_list)} not divisible by tp={tp}")
    n_workers = len(gpu_list) // tp

    default_ckw = spec.math_chat_kwargs if args.track == "math" else spec.lean_chat_kwargs
    chat_kwargs = json.loads(args.chat_kwargs_json) if args.chat_kwargs_json else dict(default_ckw)
    max_tokens = int(args.max_tokens) if int(args.max_tokens) > 0 else (
        spec.math_max_tokens if args.track == "math" else spec.lean_max_tokens
    )
    temp = float(args.temperature) if float(args.temperature) >= 0.0 else spec.default_temperature
    top_p = float(args.top_p) if float(args.top_p) >= 0.0 else spec.default_top_p
    top_k = int(args.top_k) if int(args.top_k) >= 0 else spec.default_top_k

    # Strict label-blind projection: strip everything except `id` and `messages`!
    raw_items = _read_jsonl(Path(args.split_file))
    blind_items = [{"id": str(r["id"]), "messages": r["messages"]} for r in raw_items]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path.parent / f".shards_{out_path.stem}_{os.getpid()}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    remote_hosts_seen: set[str] = set()
    worker_cfg_paths: list[Path] = []
    for w_idx in range(n_workers):
        raw_slice = gpu_list[w_idx * tp : (w_idx + 1) * tp]
        rhost = ""
        loc_devs: list[str] = []
        for tok_g in raw_slice:
            if ":" in tok_g:
                rh, rd = tok_g.split(":", 1)
                rhost = rh.strip()
                loc_devs.append(rd.strip())
            else:
                loc_devs.append(tok_g.strip())
        if rhost and rhost not in remote_hosts_seen:
            _sync_remote_model_dir_if_missing(rhost, model_path)
            subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", rhost, f"mkdir -p {shard_dir.as_posix()}"],
                check=True,
            )
            remote_hosts_seen.add(rhost)
        w_gpus = ",".join(loc_devs)
        w_items = [it for idx, it in enumerate(blind_items) if (idx % n_workers) == w_idx]
        if not w_items:
            continue
        w_cfg = {
            "worker_id": w_idx,
            "remote_host": rhost,
            "backend": backend,
            "model_key": args.model_key,
            "model_path": model_path,
            "cuda_devices": w_gpus,
            "tp": tp,
            "hf_batch_groups": int(args.hf_batch_groups) if int(args.hf_batch_groups) > 0 else spec.hf_batch_groups,
            "items": w_items,
            "k": int(args.k),
            "chat_kwargs": chat_kwargs,
            "system_prompt": args.system_prompt,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": float(args.presence_penalty),
            "max_tokens": max_tokens,
            "max_model_len": max(int(args.max_model_len), max_tokens + 2048),
            "gpu_mem_util": float(args.gpu_mem_util),
            "seed": int(args.seed),
            "enforce_eager": bool(args.enforce_eager),
            "shard_out": str(shard_dir / f"shard_{w_idx:02d}.jsonl.gz"),
        }
        cfg_p = shard_dir / f"worker_{w_idx:02d}.json"
        cfg_p.write_text(json.dumps(w_cfg), encoding="utf-8")
        worker_cfg_paths.append(cfg_p)

    t_wall0 = time.monotonic()
    worker_summaries: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(worker_cfg_paths))) as ex:
        fut_map = {ex.submit(_spawn_worker_proc, p): p for p in worker_cfg_paths}
        for fut in concurrent.futures.as_completed(fut_map):
            worker_summaries.append(fut.result())
    worker_summaries.sort(key=lambda w: int(w["worker_id"]))

    merged_rows: list[dict[str, Any]] = []
    for ws in worker_summaries:
        merged_rows.extend(_read_jsonl(Path(ws["shard_out"])))
    merged_rows.sort(key=lambda r: (str(r["id"]), int(r["sample_idx"])))

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with gzip.open(tmp_out, "wt", encoding="utf-8") as gz:
        for r in merged_rows:
            gz.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_out.replace(out_path)
    shutil.rmtree(shard_dir, ignore_errors=True)
    for rh in remote_hosts_seen:
        subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", rh, f"rm -rf {shard_dir.as_posix()}"],
            capture_output=True,
            text=True,
        )

    wall_s = time.monotonic() - t_wall0
    tot_comp = sum(int(w["completion_tokens"]) for w in worker_summaries)
    tot_prompt = sum(int(w["prompt_tokens"]) for w in worker_summaries)
    summary = {
        "model_key": args.model_key,
        "model_path": model_path,
        "backend": backend,
        "track": args.track,
        "chat_kwargs": chat_kwargs,
        "n_problems": len(blind_items),
        "k": int(args.k),
        "n_requests": len(merged_rows),
        "wall_seconds": wall_s,
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "agg_tok_per_s": tot_comp / max(wall_s, 1e-6),
        "stop_rate": sum(int(w["stop_count"]) for w in worker_summaries) / float(max(1, len(merged_rows))),
        "workers": worker_summaries,
    }
    if args.summary_output:
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "GENERATED", "output": out_path.as_posix(), "agg_tok_per_s": summary["agg_tok_per_s"], "stop_rate": summary["stop_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

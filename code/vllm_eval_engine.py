#!/usr/bin/env python3
"""
Parallel Multi-GPU vLLM 0.29.0 Evaluation & Frontier Rollout Engine (`DP=8` for 4B/9B, `DP=4 x TP=2` for 27B).

Invariants:
  - Always passes `--language-model-only` (via `language_model_only=True`) for `Qwen3_5ForConditionalGeneration`.
  - Always pins explicit stop token IDs `[248044, 248046]` (`<|endoftext|>`, `<|im_end|>`).
  - Applies exact tokenizer `apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=...)`.
  - Uses deterministic per-(item_id, sample_idx) seeds (`seed_base + sample_idx`) for exact paired comparisons.
  - Emits complete `(id, sample_idx, text, finish_reason, prompt_tokens, completion_tokens)` records in `.jsonl.gz`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

MERGED_STOP_TOKEN_IDS = [248044, 248046]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _worker_run(cfg: dict[str, Any]) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    model_path = cfg["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": int(cfg["tp"]),
        "dtype": "bfloat16",
        "gpu_memory_utilization": float(cfg["gpu_mem_util"]),
        "max_model_len": int(cfg["max_model_len"]),
        "trust_remote_code": True,
        "language_model_only": True,
        "enforce_eager": bool(cfg.get("enforce_eager", False)),
        "disable_custom_all_reduce": int(cfg["tp"]) > 1,
    }
    llm = LLM(**llm_kwargs)

    shard_items: list[dict[str, Any]] = cfg["items"]
    k = int(cfg["k"])
    thinking = bool(cfg["thinking"])
    system_prompt = cfg.get("system_prompt") or ""

    prompts: list[str] = []
    sampling_params_list: list[SamplingParams] = []
    req_meta: list[tuple[str, int]] = []

    for item in shard_items:
        msgs = list(item["messages"])
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        rendered = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        for s_idx in range(k):
            # Deterministic item-paired seed
            h_int = int(hashlib.sha256(f"{cfg['seed']}:{item['id']}:{s_idx}".encode("utf-8")).hexdigest()[:8], 16)
            sp = SamplingParams(
                n=1,
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
                top_k=int(cfg["top_k"]),
                presence_penalty=float(cfg["presence_penalty"]),
                max_tokens=int(cfg["max_tokens"]),
                stop_token_ids=MERGED_STOP_TOKEN_IDS,
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
            total_prompt_toks += p_toks
            total_comp_toks += c_toks
            if fr == "stop":
                stop_cnt += 1
            elif fr == "length":
                len_cnt += 1
            row = {
                "id": item_id,
                "sample_idx": s_idx,
                "text": comp.text,
                "finish_reason": fr,
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
            }
            gz.write(json.dumps(row, ensure_ascii=False) + "\n")

    import gc
    del llm
    gc.collect()

    return {
        "worker_id": cfg["worker_id"],
        "n_requests": len(req_meta),
        "gen_seconds": gen_sec,
        "wall_seconds": time.monotonic() - t0,
        "prompt_tokens": total_prompt_toks,
        "completion_tokens": total_comp_toks,
        "stop_count": stop_cnt,
        "length_count": len_cnt,
        "shard_out": str(shard_out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel Multi-GPU vLLM Rollout & Evaluation Engine")
    ap.add_argument("--worker-config", default="", help="Internal single-shard worker JSON config path")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--split-file", default="")
    ap.add_argument("--output", default="", help="Output .jsonl.gz path")
    ap.add_argument("--summary-output", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem-util", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    if args.worker_config:
        cfg_p = Path(args.worker_config)
        w_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        st = _worker_run(w_cfg)
        Path(w_cfg["stats_out"]).write_text(json.dumps(st), encoding="utf-8")
        return 0

    if not (args.model_path and args.split_file and args.output):
        raise SystemExit("--model-path, --split-file, and --output are required")

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpu_list) % args.tp != 0:
        raise SystemExit(f"GPU count {len(gpu_list)} must be divisible by tp={args.tp}")
    n_workers = len(gpu_list) // args.tp

    items = _read_jsonl(Path(args.split_file))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f".shards_{out_path.stem}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    import sys

    procs: list[tuple[int, subprocess.Popen[str], Path, Path]] = []
    t_start = time.monotonic()
    for w in range(n_workers):
        w_gpus = ",".join(gpu_list[w * args.tp : (w + 1) * args.tp])
        w_items = [
            {"id": str(it["id"]), "messages": it["messages"]}
            for idx, it in enumerate(items)
            if (idx % n_workers) == w
        ]
        if not w_items:
            continue
        cfg_file = tmp_dir / f"worker_cfg_{w:02d}.json"
        stats_file = tmp_dir / f"worker_stats_{w:02d}.json"
        log_file = tmp_dir / f"worker_{w:02d}.log"
        wcfg = {
            "worker_id": w,
            "cuda_devices": w_gpus,
            "tp": args.tp,
            "model_path": args.model_path,
            "items": w_items,
            "k": args.k,
            "thinking": args.thinking,
            "system_prompt": args.system_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "gpu_mem_util": args.gpu_mem_util,
            "seed": args.seed,
            "enforce_eager": args.enforce_eager,
            "shard_out": str(tmp_dir / f"shard_{w:02d}.jsonl.gz"),
            "stats_out": str(stats_file),
        }
        cfg_file.write_text(json.dumps(wcfg, ensure_ascii=False), encoding="utf-8")
        lf = log_file.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker-config", str(cfg_file)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((w, proc, stats_file, log_file))

    w_stats: list[dict[str, Any]] = []
    failed_workers: list[str] = []
    for w, proc, stats_file, log_file in procs:
        rc = proc.wait()
        if rc != 0 or not stats_file.is_file():
            tail_txt = log_file.read_text(encoding="utf-8", errors="replace")[-1500:] if log_file.is_file() else ""
            failed_workers.append(f"worker {w} rc={rc}:\n{tail_txt}")
        else:
            w_stats.append(json.loads(stats_file.read_text(encoding="utf-8")))

    if failed_workers:
        raise SystemExit("vLLM shard worker failure:\n" + "\n---\n".join(failed_workers))

    merged_rows: list[dict[str, Any]] = []
    for st in w_stats:
        sp = Path(st["shard_out"])
        with gzip.open(sp, "rt", encoding="utf-8") as gz:
            for line in gz:
                if line.strip():
                    merged_rows.append(json.loads(line))

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    merged_rows.sort(key=lambda r: (str(r["id"]), int(r["sample_idx"])))
    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        for r in merged_rows:
            gz.write(json.dumps(r, ensure_ascii=False) + "\n")

    wall_total = time.monotonic() - t_start
    total_reqs = sum(s["n_requests"] for s in w_stats)
    total_comp = sum(s["completion_tokens"] for s in w_stats)
    total_stop = sum(s["stop_count"] for s in w_stats)
    total_len = sum(s["length_count"] for s in w_stats)

    summary = {
        "model_path": args.model_path,
        "split_file": args.split_file,
        "output": str(out_path),
        "n_problems": len(items),
        "k": args.k,
        "n_requests": total_reqs,
        "n_workers": len(procs),
        "tp": args.tp,
        "gpus_used": len(gpu_list),
        "thinking": args.thinking,
        "temperature": args.temperature,
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens,
        "wall_seconds": wall_total,
        "gpu_hours": (wall_total * len(gpu_list)) / 3600.0,
        "total_completion_tokens": total_comp,
        "toks_per_sec": total_comp / max(wall_total, 1e-6),
        "stop_rate": total_stop / max(total_reqs, 1),
        "truncation_rate": total_len / max(total_reqs, 1),
        "workers": w_stats,
    }
    if args.summary_output:
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

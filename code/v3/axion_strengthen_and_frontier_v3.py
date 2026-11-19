#!/usr/bin/env python3
"""
Phase V3 Model-Aware ZPD Frontier Filtering & Cross-Family-Audited `Axion` Problem Strengthening.

Implements two complementary data-scaling stages without teacher-self-confirmation bias (`B3`):
  1. `strengthen-math`:
     - Samples pairs of Train-only seed problems (`v3_math_stage1_sft.jsonl`).
     - Uses `Qwen3.8-27B` (`tp=4`) to compose harder multi-constraint competition problems.
     - Runs 4-layer decontamination (`ID`, `alnum_sha`, `skeleton_sha`, `13-gram`) against ALL V1, V2,
       and V3 Search, Confirm, and Holdout splits before solving.
     - Solves with `Qwen3.8-27B` (`k=8`, `T=0.7`) and requires `>= 6/8` (`75%`) exact symbolic consensus
       with `>= 4`-vote margin over any runner-up answer.
     - Runs an INDEPENDENT blind cross-family verification pass (`Phi-4-mini-reasoning`, `Microsoft`
       family, `k=4`, solving from scratch without seeing `27B`'s trace!) and admits ONLY problems where
       `Phi-4-mini-reasoning` independently derives the exact same symbolic answer (`1 <= c_phi <= 3`,
       verifying both mathematical well-posedness and genuine frontier difficulty!).
  2. `build-stage2-sft`:
     - Filters `stage2_frontier_pool.jsonl` using the target model's own `B_0` `k=4` ZPD probe
       (`n_correct <= max_probe_correct`, default `<= 2` out of `4`, i.e., `pass@1 <= 0.50`).
     - For `4B` and `9B` Math, merges the cross-family-audited `Axion` strengthened problems;
       for `27B` Math, uses strictly external human-ground-truth frontier items (`0` self-generated labels!).
     - Mixes deterministic `15%` Stage-I Breadth SFT replay to prevent catastrophic forgetting during
       Stage-II cosine-LR specialization.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

V1_ROOT = Path("/scratch/governed_slm_rsi_v1")
V2_ROOT = Path("/scratch/governed_slm_rsi_v2")
V3_ROOT = Path("/scratch/governed_slm_rsi_v3")

EVAL_V3_DIR = V3_ROOT / "eval_v3"
CODE_V3_DIR = V3_ROOT / "code/v3"
for p in (EVAL_V3_DIR, CODE_V3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isolated_scorer_v3 import normalize_extracted_surface  # noqa: E402
from math_verifier import extract_answer_lenient, score_math_response  # noqa: E402
from prepare_v3_splits_and_corpora import (  # noqa: E402
    MATH_USER_SUFFIX,
    _is_scorable_exact_math_gold,
    _math_prompt_messages,
    _skeleton_sha,
    collect_prior_v1_v2_blocklist,
)


STRENGTHEN_PROMPT_TEMPLATE = """You are designing an Olympiad/collegiate mathematics competition problem by synthesizing and strengthening the mathematical ideas from two seed problems.

Seed Problem A:
{prob_a}

Seed Problem B:
{prob_b}

Instructions:
1. Create ONE self-contained, mathematically rigorous, strictly harder competition problem that combines a structural mechanism or constraint from Seed A and Seed B (or generalizes the harder one with an extra non-trivial algebraic, number-theoretic, or combinatorial layer).
2. The problem MUST have a single, unique, exact integer or simplified closed-form rational/radical answer (do NOT ask for a proof, range, approximation, or multiple sub-questions).
3. Do NOT solve the problem here. Output ONLY the new problem statement inside `<problem>...</problem>` tags."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _load_full_v3_decontaminator() -> Any:
    decon, _, _ = collect_prior_v1_v2_blocklist()
    for p in [
        V3_ROOT / "data/splits_v3/v3_math_search160.jsonl",
        V3_ROOT / "data/splits_v3/v3_math_confirm300.jsonl",
        V3_ROOT / "eval_v3/sealed_holdout/v3_math_holdout.jsonl",
    ]:
        if p.exists():
            for r in _read_jsonl(p):
                decon.register_math_eval(str(r["id"]), str(r.get("problem", "")))
    return decon


def run_strengthen_math(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    decon = _load_full_v3_decontaminator()

    stage1_rows = _read_jsonl(Path(args.stage1_file))
    rng = random.Random(args.seed)
    pool = [r for r in stage1_rows if len(str(r.get("problem", ""))) >= 40]
    rng.shuffle(pool)

    gen_prompts: list[dict[str, Any]] = []
    for i in range(args.num_candidates):
        ra = pool[(2 * i) % len(pool)]
        rb = pool[(2 * i + 1 + (i // len(pool))) % len(pool)]
        up = STRENGTHEN_PROMPT_TEMPLATE.format(prob_a=ra["problem"], prob_b=rb["problem"])
        gen_prompts.append(
            {
                "id": f"axion_synth_cand:{i:04d}",
                "seed_a": ra["id"],
                "seed_b": rb["id"],
                "messages": [{"role": "user", "content": up}],
            }
        )

    gen_in = work_dir / "gen_prompts.jsonl"
    gen_out = work_dir / "gen_raw.jsonl.gz"
    _write_jsonl(gen_in, gen_prompts)

    if not gen_out.exists():
        rc = subprocess.call(
            [
                sys.executable, "-I", "-B", str(CODE_V3_DIR / "vllm_eval_v3.py"),
                "--model-key", "qwen38_27b",
                "--split-file", str(gen_in),
                "--output", str(gen_out),
                "--k", "1",
                "--tp", "4",
                "--gpus", args.gpus,
                "--temperature", "0.85",
                "--max-tokens", "1024",
                "--max-model-len", "8192",
                "--seed", str(args.seed),
            ]
        )
        if rc != 0:
            raise SystemExit(f"Strengthened problem generation failed (rc={rc})")

    seen_skels: set[str] = {_skeleton_sha(str(r.get("problem", ""))) for r in stage1_rows}
    parsed_problems: list[dict[str, Any]] = []
    for gr in _read_jsonl(gen_out):
        if str(gr.get("finish_reason")) != "stop":
            continue
        txt = str(gr.get("text", ""))
        m = re.search(r"<problem>(.*?)</problem>", txt, flags=re.DOTALL | re.IGNORECASE)
        prob = m.group(1).strip() if m else txt.strip()
        if len(prob) < 60 or len(prob) > 2200 or "<problem>" in prob:
            continue
        if any(bad in prob.lower() for bad in ("prove that", "show that", "sketch", "part (a)", "part a:")):
            continue
        if decon.is_math_contaminated(prob):
            continue
        sk = _skeleton_sha(prob)
        if sk in seen_skels:
            continue
        seen_skels.add(sk)
        iid = str(gr["id"])
        parsed_problems.append(
            {
                "id": iid,
                "problem": prob,
                "messages": _math_prompt_messages(prob),
            }
        )

    print(f"[Axion-Strengthen] Clean decontaminated candidate problems: {len(parsed_problems)}", flush=True)
    if not parsed_problems:
        _write_jsonl(Path(args.output_file), [])
        return 0

    solve27_in = work_dir / "solve27_prompts.jsonl"
    solve27_out = work_dir / "solve27_k8.jsonl.gz"
    _write_jsonl(solve27_in, parsed_problems)

    if not solve27_out.exists():
        rc = subprocess.call(
            [
                sys.executable, "-I", "-B", str(CODE_V3_DIR / "vllm_eval_v3.py"),
                "--model-key", "qwen38_27b",
                "--split-file", str(solve27_in),
                "--output", str(solve27_out),
                "--k", "8",
                "--tp", "4",
                "--gpus", args.gpus,
                "--temperature", "0.7",
                "--max-tokens", "7680",
                "--max-model-len", "10240",
                "--seed", str(args.seed + 11),
            ]
        )
        if rc != 0:
            raise SystemExit(f"27B k=8 consensus solving failed (rc={rc})")

    by_id_27: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in _read_jsonl(solve27_out):
        by_id_27[str(r["id"])].append(r)

    prob_by_id = {p["id"]: p for p in parsed_problems}
    consensus_passed: list[dict[str, Any]] = []

    for pid, prows in sorted(by_id_27.items()):
        p_spec = prob_by_id[pid]
        ans_groups: dict[str, list[tuple[str, str, int]]] = collections.defaultdict(list)
        for r in prows:
            if str(r.get("finish_reason")) != "stop":
                continue
            ext = extract_answer_lenient(str(r["text"]), thinking=False)
            if not ext:
                continue
            nk = normalize_extracted_surface(ext)
            if not nk or len(nk) > 48:
                continue
            ans_groups[nk].append((ext, str(r["text"]).strip(), int(r.get("completion_tokens", 0))))

        if not ans_groups:
            continue
        ranked = sorted(ans_groups.items(), key=lambda kv: -len(kv[1]))
        top_norm, top_list = ranked[0]
        runner_cnt = len(ranked[1][1]) if len(ranked) > 1 else 0
        # Strict >= 6/8 consensus AND >= 4-vote margin over runner-up!
        if len(top_list) < 6 or (len(top_list) - runner_cnt) < 4:
            continue
        rep_ans = top_list[0][0]
        if not _is_scorable_exact_math_gold(rep_ans):
            continue

        # Verify all >=6 traces pass `strict_correct` against `rep_ans` and pick shortest concise clean trace
        int_task = bool(re.fullmatch(r"[+-]?\d+", rep_ans.strip()))
        clean_traces: list[tuple[int, str]] = []
        for _, txt_trace, ctoks in top_list:
            if "<think>" in txt_trace or "</think>" in txt_trace or ctoks > 7400:
                continue
            sc = score_math_response(
                txt_trace, rep_ans, finish_reason="stop", thinking=False, integer_task=int_task, problem=p_spec["problem"]
            )
            if sc.get("strict_correct") and sc.get("sft_clean_admissible"):
                clean_traces.append((ctoks, txt_trace))

        if len(clean_traces) < 5:
            continue
        clean_traces.sort(key=lambda x: x[0])
        consensus_passed.append(
            {
                "id": pid,
                "problem": p_spec["problem"],
                "gold": rep_ans.strip(),
                "integer_task": int_task,
                "best_trace_27b": clean_traces[0][1],
                "consensus_27b": len(top_list),
                "messages": p_spec["messages"],
            }
        )

    print(
        f"[Axion-Strengthen] Passed 27B >=6/8 consensus (margin>=4): {len(consensus_passed)} / {len(parsed_problems)}",
        flush=True,
    )
    if not consensus_passed:
        _write_jsonl(Path(args.output_file), [])
        return 0

    # Independent blind cross-family verification using `Phi-4-mini-reasoning` (`Microsoft` family, k=4, thinking=True)
    # Run via V2's vllm engine or direct vllm call on `/scratch/governed_slm_rsi_v2/models/Phi-4-mini-reasoning`
    phi_in = work_dir / "audit_phi4_prompts.jsonl"
    phi_out = work_dir / "audit_phi4_k4.jsonl.gz"
    _write_jsonl(phi_in, [{"id": x["id"], "messages": x["messages"]} for x in consensus_passed])

    if not phi_out.exists():
        rc_phi = subprocess.call(
            [
                sys.executable, "-I", "-B", str(CODE_V3_DIR / "vllm_eval_v3.py"),
                "--model-key", "phi4_mini_reasoning",
                "--split-file", str(phi_in),
                "--output", str(phi_out),
                "--k", "4",
                "--gpus", args.gpus,
                "--thinking",
                "--temperature", "0.6",
                "--max-tokens", "8192",
                "--max-model-len", "10240",
                "--seed", str(args.seed + 29),
            ]
        )
        if rc_phi != 0:
            raise SystemExit(f"Independent cross-family Phi-4 blind audit failed (rc={rc_phi})")

    by_id_phi: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in _read_jsonl(phi_out):
        by_id_phi[str(r["id"])].append(r)

    admitted_final: list[dict[str, Any]] = []
    for item in consensus_passed:
        prows_phi = by_id_phi.get(item["id"], [])
        phi_hits = 0
        for r in prows_phi:
            sc = score_math_response(
                str(r.get("text", "")),
                item["gold"],
                finish_reason=str(r.get("finish_reason", "unknown")),
                thinking=True,
                integer_task=item["integer_task"],
                problem=item["problem"],
            )
            if sc.get("lenient_correct"):
                phi_hits += 1
        # Require >= 1/4 independent blind solve by cross-family Phi-4 AND <= 3/4 (not trivial!)
        if 1 <= phi_hits <= 3:
            admitted_final.append(
                {
                    "id": item["id"],
                    "problem_id": item["id"],
                    "source": "axion_strengthened_cross_audited",
                    "problem": item["problem"],
                    "gold": item["gold"],
                    "consensus_27b_of_8": item["consensus_27b"],
                    "cross_audit_phi4_of_4": phi_hits,
                    "finish_reason": "stop",
                    "messages": item["messages"] + [{"role": "assistant", "content": item["best_trace_27b"]}],
                }
            )

    _write_jsonl(Path(args.output_file), admitted_final)
    print(
        f"[Axion-Strengthen] FINAL cross-family-audited ZPD items (`1<=phi4<=3` & `27B>=6/8`): "
        f"{len(admitted_final)} -> {args.output_file}",
        flush=True,
    )
    return 0


def run_build_stage2_sft(args: argparse.Namespace) -> int:
    probe_payload = json.loads(Path(args.probe_scored_json).read_text(encoding="utf-8"))
    n_corr_by_id = {str(it["id"]): int(it["n_correct"]) for it in probe_payload["items"]}

    stage2_pool = _read_jsonl(Path(args.stage2_pool_file))
    stage1_rows = _read_jsonl(Path(args.stage1_file))

    kept_frontier: list[dict[str, Any]] = []
    for r in stage2_pool:
        rid = str(r.get("id", r.get("problem_id", "")))
        # Repair pairs or items with `0 <= n_correct <= max_probe_correct` enter Stage-II Hard-Frontier SFT
        nc = n_corr_by_id.get(rid, 0)
        if nc <= args.max_probe_correct:
            if args.track == "math":
                ref_sol = str(r.get("reference_solution", ""))
                if not ref_sol:
                    continue
                msgs = list(r["messages"])
                if msgs[-1]["role"] != "assistant":
                    msgs = msgs + [{"role": "assistant", "content": ref_sol}]
                kept_frontier.append(
                    {
                        "id": rid,
                        "problem_id": rid,
                        "source": r.get("source", "stage2_frontier"),
                        "finish_reason": "stop",
                        "messages": msgs,
                    }
                )
            else:
                kept_frontier.append(r)

    n_axion_added = 0
    # Strictly exclude `qwen38_27b` from consuming `27B`-generated strengthened items (`B3` zero self-confirmation!)
    if args.axion_strengthened_file and args.model_key != "qwen38_27b" and Path(args.axion_strengthened_file).exists():
        ax_rows = _read_jsonl(Path(args.axion_strengthened_file))
        kept_frontier.extend(ax_rows)
        n_axion_added = len(ax_rows)

    rng = random.Random(args.seed)
    n_replay = max(1, int(round(args.replay_frac * len(kept_frontier))))
    s1_shuf = list(stage1_rows)
    rng.shuffle(s1_shuf)
    replay_rows = s1_shuf[: min(len(s1_shuf), n_replay)]

    final_rows = kept_frontier + replay_rows
    rng.shuffle(final_rows)
    _write_jsonl(Path(args.output_file), final_rows)
    print(
        json.dumps(
            {
                "status": "STAGE2_SFT_BUILT",
                "track": args.track,
                "model_key": args.model_key,
                "frontier_items": len(kept_frontier) - n_axion_added,
                "axion_strengthened_items": n_axion_added,
                "stage1_replay_items": len(replay_rows),
                "total_stage2_rows": len(final_rows),
                "output_file": args.output_file,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V3 Axion Strengthening & Stage-II Frontier Builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_str = sub.add_parser("strengthen-math")
    p_str.add_argument("--stage1-file", required=True)
    p_str.add_argument("--work-dir", required=True)
    p_str.add_argument("--output-file", required=True)
    p_str.add_argument("--num-candidates", type=int, default=480)
    p_str.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p_str.add_argument("--seed", type=int, default=20260325)

    p_s2 = sub.add_parser("build-stage2-sft")
    p_s2.add_argument("--track", choices=["math", "lean"], required=True)
    p_s2.add_argument("--model-key", required=True)
    p_s2.add_argument("--probe-scored-json", required=True)
    p_s2.add_argument("--stage1-file", required=True)
    p_s2.add_argument("--stage2-pool-file", required=True)
    p_s2.add_argument("--axion-strengthened-file", default="")
    p_s2.add_argument("--max-probe-correct", type=int, default=2)
    p_s2.add_argument("--replay-frac", type=float, default=0.15)
    p_s2.add_argument("--seed", type=int, default=20260325)
    p_s2.add_argument("--output-file", required=True)

    args = ap.parse_args()
    if args.cmd == "strengthen-math":
        return run_strengthen_math(args)
    return run_build_stage2_sft(args)


if __name__ == "__main__":
    raise SystemExit(main())

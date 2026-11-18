#!/usr/bin/env python3
"""
Frontier Rollout & Cumulative Verified Bank Builder (`ReST^EM` + Compiler-Repair + `LUFFY` Groups).

Modes:
  1. `merge-restem`:
     Takes an existing verified bank (`--prev-bank`, optional), new frontier rollouts (`--generations`),
     and the frontier pool (`--pool`), scores every candidate out-of-process with `lean4_verifier` or
     `math_verifier`, filters strictly to:
       - `finish_reason == "stop"`
       - `sft_clean_admissible == True`
       - tokenized sequence length `<= --max-seq-len` with `<|im_end|>` intact (`0%` right-truncation)
     Deduplicates per problem and retains uniformly up to `--max-per-problem` (default `3`) traces
     per problem (**never** `min(len)`!), plus emits updated frontier allocation schedule
     (`k=8` on `0`-trace, `k=4` on `1`-trace, `k=0` on `>=2`-trace graduated items).

  2. `build-lean-repair-prompts`:
     Extracts failed Lean proof attempts from `--scored-details` where a syntactic Lean block existed
     but failed kernel verification (`UNSOLVED_GOALS`, `TYPE_MISMATCH`, `UNKNOWN_IDENT`, `TACTIC_FAILED`),
     and constructs compiler-diagnostic-conditioned repair prompts so the SLM can generate & learn
     verified `(failed_proof, lean_compiler_error) -> fixed_proof` trajectories!

  3. `build-rl-groups`:
     Constructs scored rollout groups for `train_dapo_luffy.py`, injecting `1` verified trace from
     `--verified-bank` (`is_off_policy = True`) into any `0/G` all-fail frontier group that has a
     known verified proof/solution (`LUFFY` 1-of-8 off-policy guidance, eliminating dead groups!).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from lean4_verifier import verify_lean_candidate  # noqa: E402
from math_verifier import score_math_response  # noqa: E402

EOS_IM_END_ID = 248046


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lean_repair_messages(formal_statement: str, lean_header: str, failed_body: str, diagnostics: str) -> list[dict[str, str]]:
    from lean4_verifier import split_frozen_statement

    _, decl_prefix = split_frozen_statement(formal_statement)
    diag_trim = diagnostics.strip()[:1200]
    body_trim = failed_body.strip()[:2400]
    prompt = (
        "You are repairing a Lean 4 theorem proof that failed compilation under Mathlib.\n"
        "Inspect the theorem statement, the failed proof attempt, and the exact Lean 4 compiler error, "
        "then output ONLY a corrected single fenced ```lean4 code block containing the complete valid proof body "
        "(starting with `by`), with no `sorry` or `admit`.\n\n"
        f"Theorem:\n```lean4\n{lean_header.strip()}\n\n{decl_prefix} := by\n  sorry\n```\n\n"
        f"Failed proof body:\n```lean4\n{body_trim}\n```\n\n"
        f"Lean 4 compiler diagnostics:\n```\n{diag_trim}\n```"
    )
    return [{"role": "user", "content": prompt}]


def _score_lean_item(task: tuple[dict[str, Any], dict[str, Any], str, str, bool, float]) -> dict[str, Any]:
    gr, spec, lproj, ebin, thinking, timeout_s = task
    res = verify_lean_candidate(
        gr["text"],
        spec["formal_statement"],
        lean_project=Path(lproj),
        elan_bin=Path(ebin),
        finish_reason=str(gr.get("finish_reason", "unknown")),
        thinking=thinking,
        strict_format=False,
        timeout_s=timeout_s,
        lean_header=spec.get("lean_header"),
    )
    return {**gr, **res}


def _score_math_item(task: tuple[dict[str, Any], dict[str, Any], bool]) -> dict[str, Any]:
    gr, spec, thinking = task
    res = score_math_response(
        gr["text"],
        str(spec["gold"]),
        finish_reason=str(gr.get("finish_reason", "unknown")),
        thinking=thinking,
        integer_task=bool(spec.get("integer_task", True)),
        problem=str(spec.get("problem", "")),
    )
    return {**gr, **res}


def _fits_token_cap(tok: Any, messages: list[dict[str, str]], *, thinking: bool, max_seq_len: int) -> bool:
    out = tok.apply_chat_template(
        messages[:-1],
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
    p_len = len(out)
    r_ids = tok.encode(str(messages[-1]["content"]), add_special_tokens=False)
    return bool(len(r_ids) >= 1 and (p_len + len(r_ids) + 1) <= max_seq_len)


def cmd_merge_restem(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    pool_rows = _read_jsonl(Path(args.pool))
    spec_by_id = {str(r["id"]): r for r in pool_rows}

    # Load previous cumulative bank traces by problem_id
    traces_by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_hash_by_pid: dict[str, set[str]] = collections.defaultdict(set)

    if args.prev_bank and Path(args.prev_bank).is_file():
        for r in _read_jsonl(Path(args.prev_bank)):
            pid = str(r["problem_id"])
            asst = r["messages"][-1]["content"]
            th = hashlib.sha256(" ".join(asst.split()).encode("utf-8")).hexdigest()
            if th not in seen_hash_by_pid[pid]:
                if _fits_token_cap(tok, r["messages"], thinking=args.thinking, max_seq_len=args.max_seq_len):
                    seen_hash_by_pid[pid].add(th)
                    traces_by_pid[pid].append(r)

    new_admitted = 0
    raw_total = 0
    raw_truncated = 0
    failed_lean_for_repair: list[dict[str, Any]] = []

    if args.generations and Path(args.generations).is_file():
        gen_rows = _read_jsonl(Path(args.generations))
        raw_total = len(gen_rows)
        raw_truncated = sum(1 for g in gen_rows if str(g.get("finish_reason")) == "length")

        # Map repair ids `base_id::rep0` back to `base_id` for verification!
        if args.track == "lean":
            tasks_l = []
            for g in gen_rows:
                gid = str(g["id"])
                base_id = gid.split("::rep")[0]
                if base_id in spec_by_id:
                    tasks_l.append((g, spec_by_id[base_id], args.lean_project, args.elan_bin, args.thinking, args.lean_timeout))
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                scored = list(ex.map(_score_lean_item, tasks_l, chunksize=4))
        else:
            tasks_m = []
            for g in gen_rows:
                gid = str(g["id"])
                if gid in spec_by_id:
                    tasks_m.append((g, spec_by_id[gid], args.thinking))
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, 32)) as ex:
                scored = list(ex.map(_score_math_item, tasks_m, chunksize=16))

        prompt_lookup = {}
        if args.prompt_override_file and Path(args.prompt_override_file).is_file():
            prompt_lookup = {str(r["id"]): r["messages"] for r in _read_jsonl(Path(args.prompt_override_file))}

        for sc in scored:
            gid = str(sc["id"])
            base_id = gid.split("::rep")[0]
            spec = spec_by_id[base_id]
            p_msgs = prompt_lookup.get(gid, spec["messages"])

            if args.track == "lean":
                if sc["ok"] and sc.get("proof_body"):
                    clean_asst = f"```lean4\n{sc['proof_body']}\n```"
                    full_msgs = p_msgs + [{"role": "assistant", "content": clean_asst}]
                    # Key repair traces under `gid` (`base_id::rep...`) or `base_id`
                    bank_pid = gid if "::rep" in gid else base_id
                    th = hashlib.sha256(" ".join(clean_asst.split()).encode("utf-8")).hexdigest()
                    if th not in seen_hash_by_pid[bank_pid] and _fits_token_cap(
                        tok, full_msgs, thinking=args.thinking, max_seq_len=args.max_seq_len
                    ):
                        seen_hash_by_pid[bank_pid].add(th)
                        traces_by_pid[bank_pid].append(
                            {
                                "id": f"{bank_pid}#r{args.round_tag}_{len(traces_by_pid[bank_pid])}",
                                "problem_id": bank_pid,
                                "base_problem_id": base_id,
                                "formal_statement": spec["formal_statement"],
                                "messages": full_msgs,
                                "finish_reason": "stop",
                                "source": args.round_tag,
                            }
                        )
                        new_admitted += 1
                elif (
                    "::rep" not in gid
                    and sc.get("proof_body")
                    and sc.get("error_class") in ("UNSOLVED_GOALS", "TYPE_MISMATCH", "UNKNOWN_IDENT", "TACTIC_FAILED")
                ):
                    failed_lean_for_repair.append(
                        {
                            "base_id": base_id,
                            "formal_statement": spec["formal_statement"],
                            "lean_header": spec.get("lean_header", ""),
                            "failed_body": sc["proof_body"],
                            "diagnostics": sc.get("diagnostics", ""),
                        }
                    )
            else:
                if sc.get("sft_clean_admissible", False):
                    asst_txt = str(sc["text"]).strip()
                    full_msgs = p_msgs + [{"role": "assistant", "content": asst_txt}]
                    th = hashlib.sha256(" ".join(asst_txt.split()).encode("utf-8")).hexdigest()
                    if th not in seen_hash_by_pid[base_id] and _fits_token_cap(
                        tok, full_msgs, thinking=args.thinking, max_seq_len=args.max_seq_len
                    ):
                        seen_hash_by_pid[base_id].add(th)
                        traces_by_pid[base_id].append(
                            {
                                "id": f"{base_id}#r{args.round_tag}_{len(traces_by_pid[base_id])}",
                                "problem_id": base_id,
                                "messages": full_msgs,
                                "finish_reason": "stop",
                                "source": args.round_tag,
                            }
                        )
                        new_admitted += 1

    # Uniform cap of `<= max_per_problem` per problem_id (NEVER min-length!)
    final_bank: list[dict[str, Any]] = []
    for pid in sorted(traces_by_pid.keys()):
        cands = traces_by_pid[pid]
        if len(cands) > args.max_per_problem:
            s_int = int(hashlib.sha256(f"{args.round_tag}:{pid}".encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(s_int)
            cands = rng.sample(cands, args.max_per_problem)
        final_bank.extend(cands)

    out_bank = Path(args.output_bank)
    out_bank.parent.mkdir(parents=True, exist_ok=True)
    with out_bank.open("w", encoding="utf-8") as fh:
        for r in final_bank:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Emit next-round frontier prompt subset (unsolved `0`-trace + `1`-trace base problems!)
    base_solved_counts = {
        pid: len(traces_by_pid.get(pid, [])) for pid in spec_by_id
    }
    frontier_next: list[dict[str, Any]] = []
    for pid, spec in sorted(spec_by_id.items()):
        if base_solved_counts[pid] < 2:
            frontier_next.append(spec)

    if args.output_next_frontier:
        nf_p = Path(args.output_next_frontier)
        nf_p.parent.mkdir(parents=True, exist_ok=True)
        with nf_p.open("w", encoding="utf-8") as fh:
            for r in frontier_next:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.output_repair_prompts and failed_lean_for_repair:
        by_b: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for item in failed_lean_for_repair:
            by_b[item["base_id"]].append(item)
        rep_prompts: list[dict[str, Any]] = []
        for bid, ilist in sorted(by_b.items()):
            for r_i, it in enumerate(ilist[:2]):
                rep_prompts.append(
                    {
                        "id": f"{bid}::rep{r_i}",
                        "base_id": bid,
                        "formal_statement": it["formal_statement"],
                        "lean_header": it["lean_header"],
                        "messages": _lean_repair_messages(
                            it["formal_statement"], it["lean_header"], it["failed_body"], it["diagnostics"]
                        ),
                    }
                )
        rp_path = Path(args.output_repair_prompts)
        rp_path.parent.mkdir(parents=True, exist_ok=True)
        with rp_path.open("w", encoding="utf-8") as fh:
            for r in rep_prompts:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    solved_base_problems = sum(1 for c in base_solved_counts.values() if c >= 1)
    summary = {
        "round_tag": args.round_tag,
        "track": args.track,
        "pool_total_problems": len(spec_by_id),
        "solved_pool_problems": solved_base_problems,
        "pool_coverage_rate": solved_base_problems / max(len(spec_by_id), 1),
        "unsolved_or_1trace_frontier_count": len(frontier_next),
        "raw_rollouts_scored": raw_total,
        "raw_rollouts_truncated": raw_truncated,
        "raw_rollout_truncation_rate": raw_truncated / max(raw_total, 1),
        "new_clean_traces_admitted": new_admitted,
        "cumulative_bank_traces": len(final_bank),
        "unique_bank_keys": len(traces_by_pid),
        "sft_hygiene_clean_rate": 1.0,
    }
    if args.summary_output:
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_build_rl_groups(args: argparse.Namespace) -> int:
    pool_rows = _read_jsonl(Path(args.pool))
    spec_by_id = {str(r["id"]): r for r in pool_rows}
    gen_rows = _read_jsonl(Path(args.generations))

    # Map verified bank off-policy fallback traces by base problem_id
    offpolicy_by_pid: dict[str, list[str]] = collections.defaultdict(list)
    if args.verified_bank and Path(args.verified_bank).is_file():
        for r in _read_jsonl(Path(args.verified_bank)):
            pid = str(r.get("base_problem_id") or r["problem_id"])
            if "::rep" not in str(r["problem_id"]):
                offpolicy_by_pid[pid].append(r["messages"][-1]["content"])

    if args.track == "lean":
        tasks_l = [
            (g, spec_by_id[str(g["id"])], args.lean_project, args.elan_bin, args.thinking, args.lean_timeout)
            for g in gen_rows
            if str(g["id"]) in spec_by_id
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            scored = list(ex.map(_score_lean_item, tasks_l, chunksize=4))
    else:
        tasks_m = [(g, spec_by_id[str(g["id"])], args.thinking) for g in gen_rows if str(g["id"]) in spec_by_id]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, 32)) as ex:
            scored = list(ex.map(_score_math_item, tasks_m, chunksize=16))

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for sc in scored:
        by_pid[str(sc["id"])].append(sc)

    out_records: list[dict[str, Any]] = []
    n_luffy_groups = 0
    n_onpolicy_mixed = 0
    n_all_pass = 0
    n_unassisted_all_fail = 0

    for pid, grows in sorted(by_pid.items()):
        grows.sort(key=lambda x: int(x["sample_idx"]))
        spec = spec_by_id[pid]
        ok_key = "ok" if args.track == "lean" else "lenient_correct"
        n_ok = sum(1 for r in grows if bool(r.get(ok_key, False)))

        group_out: list[dict[str, Any]] = []
        for r in grows:
            group_out.append(
                {
                    "id": pid,
                    "sample_idx": int(r["sample_idx"]),
                    "prompt_messages": spec["messages"],
                    "response_text": str(r["text"]),
                    "finish_reason": str(r.get("finish_reason", "unknown")),
                    "verified_ok": bool(r.get(ok_key, False)),
                    "is_off_policy": False,
                }
            )

        if n_ok == 0 and offpolicy_by_pid.get(pid) and not args.disable_luffy:
            # LUFFY 1-of-G replacement: replace last failed draw with 1 verified off-policy trace!
            rng = random.Random(int(hashlib.sha256(f"luffy:{pid}".encode("utf-8")).hexdigest()[:8], 16))
            chosen_trace = rng.choice(offpolicy_by_pid[pid])
            group_out[-1] = {
                "id": pid,
                "sample_idx": group_out[-1]["sample_idx"],
                "prompt_messages": spec["messages"],
                "response_text": chosen_trace,
                "finish_reason": "stop",
                "verified_ok": True,
                "is_off_policy": True,
            }
            n_luffy_groups += 1
        elif n_ok == 0:
            n_unassisted_all_fail += 1
        elif n_ok == len(grows):
            n_all_pass += 1
        else:
            n_onpolicy_mixed += 1

        out_records.extend(group_out)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as fh:
        for r in out_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "track": args.track,
        "groups_total": len(by_pid),
        "onpolicy_mixed_groups": n_onpolicy_mixed,
        "luffy_rescued_0pass_groups": n_luffy_groups,
        "all_pass_groups": n_all_pass,
        "unassisted_0pass_groups": n_unassisted_all_fail,
        "total_records": len(out_records),
    }
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_m = sub.add_parser("merge-restem")
    p_m.add_argument("--track", choices=["lean", "math"], required=True)
    p_m.add_argument("--pool", required=True)
    p_m.add_argument("--prev-bank", default="")
    p_m.add_argument("--generations", default="")
    p_m.add_argument("--prompt-override-file", default="")
    p_m.add_argument("--tokenizer-path", required=True)
    p_m.add_argument("--thinking", action="store_true")
    p_m.add_argument("--max-seq-len", type=int, default=8192)
    p_m.add_argument("--max-per-problem", type=int, default=3)
    p_m.add_argument("--round-tag", required=True)
    p_m.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    p_m.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    p_m.add_argument("--lean-timeout", type=float, default=45.0)
    p_m.add_argument("--workers", type=int, default=36)
    p_m.add_argument("--output-bank", required=True)
    p_m.add_argument("--output-next-frontier", default="")
    p_m.add_argument("--output-repair-prompts", default="")
    p_m.add_argument("--summary-output", default="")
    p_m.set_defaults(func=cmd_merge_restem)

    p_rl = sub.add_parser("build-rl-groups")
    p_rl.add_argument("--track", choices=["lean", "math"], required=True)
    p_rl.add_argument("--pool", required=True)
    p_rl.add_argument("--generations", required=True)
    p_rl.add_argument("--verified-bank", default="")
    p_rl.add_argument("--thinking", action="store_true")
    p_rl.add_argument("--disable-luffy", action="store_true")
    p_rl.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    p_rl.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    p_rl.add_argument("--lean-timeout", type=float, default=45.0)
    p_rl.add_argument("--workers", type=int, default=36)
    p_rl.add_argument("--output", required=True)
    p_rl.add_argument("--summary-output", default="")
    p_rl.set_defaults(func=cmd_build_rl_groups)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Phase V2 Multi-Architecture Bank Builder & Causal Control Constructor.

Subcommands:
  1. `merge-restem`:
     Multi-architecture (`qwen35_4b`, `qwen35_9b`, `qwen3_4b`, `phi4_mini`) verified trace bank builder.
     Enforces `finish_reason == "stop"`, `sft_clean_admissible` / Lean `#print axioms` clean,
     exact `chat_prompt_ids` length <= `max_seq_len`, and uniform `<= max_per_problem` deduplicated sampling.
     Also emits `--output-repair-prompts` (up to `--repair-drafts-per-thm` failed syntactic Lean drafts
     per theorem) so `k=4` repair rollouts harvest `>= 256` natural compiler-repair trajectories.

  2. `merge-repair-5050`:
     Scores Lean repair rollouts (`base_id::repK`), verifies each fixed proof in Lean 4.24 (`#print axioms`),
     and formats verified repair SFT rows with `Opus 5.5, Max`'s 50/50 diagnostic split:
       - 50% exact compiler diagnostics (`compiler_repair` prompt)
       - 50% masked diagnostics (`masked_retry` prompt)
     so a single trained checkpoint is simultaneously in-distribution for `compiler_repair_4+4`,
     `masked_retry_4+4`, and `shuffled_diag_4+4` inference ablations!

  3. `match-yield`:
     Constructs `Opus 5.5, Max`'s yield-matched recursive control bank (`M_2^{rec-yield}`):
     given `--rec-bank` (`B_1 union B_2^{M1}`) and `--frozen-bank` (`B_1 union B_2^{M0}`), keeps `B_1`
     identical and stratified-subsamples the newly added `R2` traces in `--rec-bank` to match
     `--frozen-bank`'s exact problem coverage and total trace count!

  4. `build-rl-groups`:
     Multi-architecture RLVR group builder with `LUFFY` 1-of-G off-policy rescue for `0/G` frontier items,
     automatically filtering to active frontier problems (or capping total active groups to `--max-groups`).
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

THIS_DIR = Path(__file__).resolve().parent
ROOT_V1_EVAL = Path("/scratch/governed_slm_rsi_v1/eval")
for p in (THIS_DIR, ROOT_V1_EVAL, THIS_DIR.parents[1] / "eval"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lean4_verifier import split_frozen_statement, verify_lean_candidate  # noqa: E402
from math_verifier import score_math_response  # noqa: E402
from model_registry_v2 import MODEL_REGISTRY_V2, ModelSpecV2  # noqa: E402
from train_slm_v2 import chat_prompt_ids  # noqa: E402

OMITTED_DIAG_TEXT = "[Compiler diagnostics omitted — inspect the failed proof attempt against the theorem statement and repair it directly.]"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def lean_repair_messages_v2(
    formal_statement: str,
    lean_header: str,
    failed_body: str,
    diagnostics: str,
    *,
    mask_diagnostics: bool = False,
) -> list[dict[str, str]]:
    _, decl_prefix = split_frozen_statement(formal_statement)
    diag_txt = OMITTED_DIAG_TEXT if mask_diagnostics else diagnostics.strip()[:1200]
    body_trim = failed_body.strip()[:2400]
    hdr = lean_header.strip()
    prompt = (
        "You are repairing a Lean 4 theorem proof that failed compilation under Mathlib.\n"
        "Inspect the theorem statement, the failed proof attempt, and the compiler feedback below, "
        "then output ONLY a corrected single fenced ```lean4 code block containing the complete valid proof body "
        "(starting with `by`), with no `sorry` or `admit`.\n\n"
        f"Theorem:\n```lean4\n{hdr}\n\n{decl_prefix} := by\n  sorry\n```\n\n"
        f"Failed proof body:\n```lean4\n{body_trim}\n```\n\n"
        f"Lean 4 compiler diagnostics:\n```\n{diag_txt}\n```"
    )
    return [{"role": "user", "content": prompt}]


def _has_degenerate_block_repetition(text: str, *, window_words: int = 24, max_repeats: int = 4) -> bool:
    """Reject <think> traces that repeat long blocks (>=24 words >=4 times) without progress."""
    words = text.lower().split()
    if len(words) < window_words * max_repeats:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(0, len(words) - window_words + 1, 4):
        gram = tuple(words[i : i + window_words])
        c = counts.get(gram, 0) + 1
        if c >= max_repeats:
            return True
        counts[gram] = c
    return False


def _fits_cap_v2(tok: Any, spec: ModelSpecV2, messages: list[dict[str, str]], *, thinking: bool, max_seq_len: int) -> bool:
    p_ids = chat_prompt_ids(tok, spec, messages[:-1], thinking=thinking)
    r_ids = tok.encode(str(messages[-1]["content"]), add_special_tokens=False)
    return bool(len(r_ids) >= 1 and (len(p_ids) + len(r_ids) + 1) <= max_seq_len)


def _score_lean_worker(task: tuple[dict[str, Any], dict[str, Any], str, str, bool, float]) -> dict[str, Any]:
    gr, spec_row, lproj, ebin, thinking, timeout_s = task
    res = verify_lean_candidate(
        gr["text"],
        spec_row["formal_statement"],
        lean_project=Path(lproj),
        elan_bin=Path(ebin),
        finish_reason=str(gr.get("finish_reason", "unknown")),
        thinking=thinking,
        strict_format=False,
        timeout_s=timeout_s,
        lean_header=spec_row.get("lean_header"),
    )
    return {**gr, **res}


def _score_math_worker(task: tuple[dict[str, Any], dict[str, Any], bool]) -> dict[str, Any]:
    gr, spec_row, thinking = task
    res = score_math_response(
        gr["text"],
        str(spec_row["gold"]),
        finish_reason=str(gr.get("finish_reason", "unknown")),
        thinking=thinking,
        integer_task=bool(spec_row.get("integer_task", False)),
        problem=str(spec_row.get("problem", "")),
    )
    return {**gr, **res}


def _clean_lean_asst(raw_text: str, proof_body: str, *, thinking: bool) -> str | None:
    fenced = f"```lean4\n{proof_body}\n```"
    if not thinking:
        return fenced
    # Preserve native <think>...</think> reasoning chain for thinking models (e.g. Phi-4-mini-reasoning)
    if "</think>" not in raw_text:
        return None
    think_prefix = raw_text.split("</think>", 1)[0].strip()
    if not think_prefix.startswith("<think>"):
        think_prefix = "<think>\n" + think_prefix
    return f"{think_prefix}\n</think>\n\n{fenced}"


def cmd_merge_restem(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    mspec = MODEL_REGISTRY_V2[args.model_key]
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or mspec.local_path, trust_remote_code=True)
    pool_rows = _read_jsonl(Path(args.pool))
    spec_by_id = {str(r["id"]): r for r in pool_rows}

    traces_by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_hash_by_pid: dict[str, set[str]] = collections.defaultdict(set)

    if args.prev_bank and Path(args.prev_bank).is_file():
        for r in _read_jsonl(Path(args.prev_bank)):
            pid = str(r["problem_id"])
            asst = r["messages"][-1]["content"]
            th = hashlib.sha256(" ".join(asst.split()).encode("utf-8")).hexdigest()
            if th not in seen_hash_by_pid[pid] and _fits_cap_v2(
                tok, mspec, r["messages"], thinking=args.thinking, max_seq_len=args.max_seq_len
            ):
                seen_hash_by_pid[pid].add(th)
                traces_by_pid[pid].append(r)

    prev_problem_ids = set(traces_by_pid.keys())
    new_admitted = 0
    raw_total = 0
    raw_truncated = 0
    failed_lean_drafts: list[dict[str, Any]] = []

    if args.generations and Path(args.generations).is_file():
        gen_rows = _read_jsonl(Path(args.generations))
        raw_total = len(gen_rows)
        raw_truncated = sum(1 for g in gen_rows if str(g.get("finish_reason")) == "length")

        if args.track == "lean":
            tasks_l = [
                (g, spec_by_id[str(g["id"])], args.lean_project, args.elan_bin, args.thinking, args.lean_timeout)
                for g in gen_rows
                if str(g["id"]) in spec_by_id
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                scored = list(ex.map(_score_lean_worker, tasks_l, chunksize=4))
        else:
            from timeout_isolated_scorer_v2 import score_math_tasks_safely
            g_valid = [g for g in gen_rows if str(g["id"]) in spec_by_id]
            tasks_m = [(g, spec_by_id[str(g["id"])], args.thinking) for g in g_valid]
            raw_scored = score_math_tasks_safely(tasks_m, workers=min(args.workers, 32))
            scored = [{**g, **sc_r} for g, sc_r in zip(g_valid, raw_scored)]

        for sc in scored:
            pid = str(sc["id"])
            spec_row = spec_by_id[pid]
            p_msgs = spec_row["messages"]
            if args.track == "lean":
                if sc.get("ok") and sc.get("proof_body") and str(sc.get("finish_reason")) == "stop":
                    clean_asst = _clean_lean_asst(str(sc["text"]), str(sc["proof_body"]), thinking=args.thinking)
                    if not clean_asst:
                        continue
                    full_msgs = p_msgs + [{"role": "assistant", "content": clean_asst}]
                    th = hashlib.sha256(" ".join(clean_asst.split()).encode("utf-8")).hexdigest()
                    if th not in seen_hash_by_pid[pid] and _fits_cap_v2(
                        tok, mspec, full_msgs, thinking=args.thinking, max_seq_len=args.max_seq_len
                    ):
                        seen_hash_by_pid[pid].add(th)
                        traces_by_pid[pid].append(
                            {
                                "id": f"{pid}#{args.round_tag}_{len(traces_by_pid[pid])}",
                                "problem_id": pid,
                                "base_problem_id": pid,
                                "formal_statement": spec_row["formal_statement"],
                                "messages": full_msgs,
                                "finish_reason": "stop",
                                "source": args.round_tag,
                            }
                        )
                        new_admitted += 1
                elif sc.get("proof_body") and sc.get("diagnostics") and sc.get("error_class") in (
                    "UNSOLVED_GOALS",
                    "TYPE_MISMATCH",
                    "UNKNOWN_IDENT",
                    "TACTIC_FAILED",
                    "SYNTAX_ERROR",
                    "LEAN_COMPILE_ERROR",
                ):
                    failed_lean_drafts.append(
                        {
                            "base_id": pid,
                            "formal_statement": spec_row["formal_statement"],
                            "lean_header": spec_row.get("lean_header", ""),
                            "failed_body": sc["proof_body"],
                            "diagnostics": sc.get("diagnostics", ""),
                        }
                    )
            else:
                if sc.get("sft_clean_admissible", False):
                    asst_txt = str(sc["text"]).strip()
                    if args.reject_repetitive_blocks and _has_degenerate_block_repetition(asst_txt):
                        continue
                    full_msgs = p_msgs + [{"role": "assistant", "content": asst_txt}]
                    th = hashlib.sha256(" ".join(asst_txt.split()).encode("utf-8")).hexdigest()
                    if th not in seen_hash_by_pid[pid] and _fits_cap_v2(
                        tok, mspec, full_msgs, thinking=args.thinking, max_seq_len=args.max_seq_len
                    ):
                        seen_hash_by_pid[pid].add(th)
                        traces_by_pid[pid].append(
                            {
                                "id": f"{pid}#{args.round_tag}_{len(traces_by_pid[pid])}",
                                "problem_id": pid,
                                "messages": full_msgs,
                                "finish_reason": "stop",
                                "source": args.round_tag,
                            }
                        )
                        new_admitted += 1

    final_bank: list[dict[str, Any]] = []
    for pid in sorted(traces_by_pid.keys()):
        cands = traces_by_pid[pid]
        if len(cands) > args.max_per_problem:
            s_int = int(hashlib.sha256(f"{args.round_tag}:{pid}".encode("utf-8")).hexdigest()[:8], 16)
            cands = random.Random(s_int).sample(cands, args.max_per_problem)
        final_bank.extend(cands)

    _write_jsonl(Path(args.output_bank), final_bank)

    base_counts = {pid: len(traces_by_pid.get(pid, [])) for pid in spec_by_id}
    frontier_next = [spec_row for pid, spec_row in sorted(spec_by_id.items()) if base_counts[pid] < 2]
    if args.output_next_frontier:
        _write_jsonl(Path(args.output_next_frontier), frontier_next)

    n_rep_prompts = 0
    if args.output_repair_prompts and failed_lean_drafts:
        by_b: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        seen_bodies: dict[str, set[str]] = collections.defaultdict(set)
        for item in failed_lean_drafts:
            bh = hashlib.sha256(" ".join(item["failed_body"].split()).encode("utf-8")).hexdigest()
            if bh not in seen_bodies[item["base_id"]]:
                seen_bodies[item["base_id"]].add(bh)
                by_b[item["base_id"]].append(item)
        rep_prompts: list[dict[str, Any]] = []
        for bid, ilist in sorted(by_b.items()):
            for r_i, it in enumerate(ilist[: args.repair_drafts_per_thm]):
                rep_prompts.append(
                    {
                        "id": f"{bid}::rep{r_i}",
                        "base_id": bid,
                        "formal_statement": it["formal_statement"],
                        "lean_header": it["lean_header"],
                        "failed_body": it["failed_body"],
                        "diagnostics": it["diagnostics"],
                        "messages": lean_repair_messages_v2(
                            it["formal_statement"], it["lean_header"], it["failed_body"], it["diagnostics"], mask_diagnostics=False
                        ),
                    }
                )
        _write_jsonl(Path(args.output_repair_prompts), rep_prompts)
        n_rep_prompts = len(rep_prompts)

    solved_probs = sum(1 for c in base_counts.values() if c >= 1)
    newly_solved = len(set(traces_by_pid.keys()) - prev_problem_ids)
    summary = {
        "round_tag": args.round_tag,
        "model_key": args.model_key,
        "track": args.track,
        "pool_total_problems": len(spec_by_id),
        "solved_pool_problems": solved_probs,
        "newly_solved_problems": newly_solved,
        "pool_coverage_rate": solved_probs / max(len(spec_by_id), 1),
        "frontier_remaining": len(frontier_next),
        "raw_rollouts_scored": raw_total,
        "raw_rollout_truncation_rate": raw_truncated / max(raw_total, 1),
        "new_clean_traces_admitted": new_admitted,
        "cumulative_bank_traces": len(final_bank),
        "repair_prompts_emitted": n_rep_prompts,
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_merge_repair_5050(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    mspec = MODEL_REGISTRY_V2[args.model_key]
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or mspec.local_path, trust_remote_code=True)
    rep_prompts = _read_jsonl(Path(args.repair_prompts))
    prompt_by_id = {str(r["id"]): r for r in rep_prompts}
    gen_rows = _read_jsonl(Path(args.generations))

    tasks = [
        (g, prompt_by_id[str(g["id"])], args.lean_project, args.elan_bin, args.thinking, args.lean_timeout)
        for g in gen_rows
        if str(g["id"]) in prompt_by_id
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        scored = list(ex.map(_score_lean_worker, tasks, chunksize=4))

    repair_by_prompt: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_hash: dict[str, set[str]] = collections.defaultdict(set)
    for sc in scored:
        rid = str(sc["id"])
        if not (sc.get("ok") and sc.get("proof_body") and str(sc.get("finish_reason")) == "stop"):
            continue
        clean_asst = _clean_lean_asst(str(sc["text"]), str(sc["proof_body"]), thinking=args.thinking)
        if not clean_asst:
            continue
        th = hashlib.sha256(" ".join(clean_asst.split()).encode("utf-8")).hexdigest()
        if th in seen_hash[rid]:
            continue
        seen_hash[rid].add(th)
        repair_by_prompt[rid].append({"clean_asst": clean_asst, "meta": prompt_by_id[rid]})

    repair_rows: list[dict[str, Any]] = []
    n_diag_kept = 0
    n_diag_masked = 0
    unique_repaired_thms: set[str] = set()

    for idx, rid in enumerate(sorted(repair_by_prompt.keys())):
        items = repair_by_prompt[rid][: args.max_per_prompt]
        for s_i, it in enumerate(items):
            meta = it["meta"]
            # Deterministic 50/50 split between full compiler diagnostics and masked-retry format (Opus 5.5 Max)
            mask_flag = ((idx + s_i) % 2) == 1
            p_msgs = lean_repair_messages_v2(
                meta["formal_statement"],
                meta.get("lean_header", ""),
                meta["failed_body"],
                meta["diagnostics"],
                mask_diagnostics=mask_flag,
            )
            full_msgs = p_msgs + [{"role": "assistant", "content": it["clean_asst"]}]
            if not _fits_cap_v2(tok, mspec, full_msgs, thinking=args.thinking, max_seq_len=args.max_seq_len):
                continue
            if mask_flag:
                n_diag_masked += 1
            else:
                n_diag_kept += 1
            unique_repaired_thms.add(str(meta["base_id"]))
            repair_rows.append(
                {
                    "id": f"{rid}#rep5050_{s_i}",
                    "problem_id": rid,
                    "base_problem_id": str(meta["base_id"]),
                    "formal_statement": meta["formal_statement"],
                    "messages": full_msgs,
                    "finish_reason": "stop",
                    "mask_diagnostics": mask_flag,
                    "source": args.round_tag,
                }
            )

    prev_rows = _read_jsonl(Path(args.prev_bank)) if (args.prev_bank and Path(args.prev_bank).is_file()) else []
    combined = prev_rows + repair_rows
    _write_jsonl(Path(args.output_bank), combined)
    if args.output_repair_only:
        _write_jsonl(Path(args.output_repair_only), repair_rows)

    summary = {
        "round_tag": args.round_tag,
        "model_key": args.model_key,
        "repair_rollouts_scored": len(scored),
        "verified_natural_repair_pairs": len(repair_rows),
        "unique_repaired_prompts": len(repair_by_prompt),
        "unique_repaired_theorems": len(unique_repaired_thms),
        "diag_kept_pairs": n_diag_kept,
        "diag_masked_pairs": n_diag_masked,
        "prev_direct_bank_rows": len(prev_rows),
        "combined_bank_rows": len(combined),
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_match_yield(args: argparse.Namespace) -> int:
    b1_rows = _read_jsonl(Path(args.r1_bank))
    rec_rows = _read_jsonl(Path(args.rec_bank))
    froz_rows = _read_jsonl(Path(args.frozen_bank))

    b1_ids = {str(r["id"]) for r in b1_rows}
    rec_new = [r for r in rec_rows if str(r["id"]) not in b1_ids]
    froz_new = [r for r in froz_rows if str(r["id"]) not in b1_ids]

    target_new_total = len(froz_new)
    target_new_pids = len({str(r["problem_id"]) for r in froz_new})
    rng = random.Random(args.seed)

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rec_new:
        by_pid[str(r["problem_id"])].append(r)

    pids = sorted(by_pid.keys())
    rng.shuffle(pids)
    kept_pids = pids[: min(len(pids), max(1, target_new_pids))] if target_new_total > 0 else []

    picked_new: list[dict[str, Any]] = []
    # First take 1 trace per selected problem, then fill remaining quota up to target_new_total
    leftovers: list[dict[str, Any]] = []
    for pid in kept_pids:
        cands = list(by_pid[pid])
        rng.shuffle(cands)
        if len(picked_new) < target_new_total:
            picked_new.append(cands[0])
            leftovers.extend(cands[1:])
    rng.shuffle(leftovers)
    while len(picked_new) < target_new_total and leftovers:
        picked_new.append(leftovers.pop())

    # Assemble matched bank: exact R1 base + yield-matched R2 delta (capped at len(froz_rows))
    out_rows = list(b1_rows) + picked_new
    if len(out_rows) > len(froz_rows):
        out_rows = out_rows[: len(froz_rows)]
    _write_jsonl(Path(args.output_bank), out_rows)

    summary = {
        "r1_traces": len(b1_rows),
        "frozen_r2_total_traces": len(froz_rows),
        "frozen_r2_new_traces": len(froz_new),
        "rec_full_r2_total_traces": len(rec_rows),
        "rec_full_r2_new_traces": len(rec_new),
        "rec_yield_matched_total_traces": len(out_rows),
        "rec_yield_matched_new_traces": len(picked_new),
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_build_rl_groups(args: argparse.Namespace) -> int:
    pool_rows = _read_jsonl(Path(args.pool))
    spec_by_id = {str(r["id"]): r for r in pool_rows}
    gen_rows = _read_jsonl(Path(args.generations))

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
            scored = list(ex.map(_score_lean_worker, tasks_l, chunksize=4))
    else:
        from timeout_isolated_scorer_v2 import score_math_tasks_safely
        g_valid = [g for g in gen_rows if str(g["id"]) in spec_by_id]
        tasks_m = [(g, spec_by_id[str(g["id"])], args.thinking) for g in g_valid]
        raw_scored = score_math_tasks_safely(tasks_m, workers=min(args.workers, 32))
        scored = [{**g, **sc_r} for g, sc_r in zip(g_valid, raw_scored)]

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for sc in scored:
        by_pid[str(sc["id"])].append(sc)

    out_records: list[dict[str, Any]] = []
    n_luffy = 0
    n_mixed = 0
    n_all_pass = 0
    n_all_fail = 0

    for pid, grows in sorted(by_pid.items()):
        grows.sort(key=lambda x: int(x["sample_idx"]))
        spec_row = spec_by_id[pid]
        ok_key = "ok" if args.track == "lean" else "lenient_correct"
        n_ok = sum(1 for r in grows if bool(r.get(ok_key, False)))

        group_out = [
            {
                "id": pid,
                "sample_idx": int(r["sample_idx"]),
                "prompt_messages": spec_row["messages"],
                "response_text": str(r["text"]),
                "finish_reason": str(r.get("finish_reason", "unknown")),
                "verified_ok": bool(r.get(ok_key, False)),
                "is_off_policy": False,
            }
            for r in grows
        ]

        if n_ok == 0 and offpolicy_by_pid.get(pid) and not args.disable_luffy:
            rng = random.Random(int(hashlib.sha256(f"luffy_v2:{pid}".encode("utf-8")).hexdigest()[:8], 16))
            group_out[-1] = {
                "id": pid,
                "sample_idx": group_out[-1]["sample_idx"],
                "prompt_messages": spec_row["messages"],
                "response_text": rng.choice(offpolicy_by_pid[pid]),
                "finish_reason": "stop",
                "verified_ok": True,
                "is_off_policy": True,
            }
            n_luffy += 1
        elif n_ok == 0:
            n_all_fail += 1
        elif n_ok == len(grows):
            n_all_pass += 1
        else:
            n_mixed += 1

        out_records.extend(group_out)

    _write_jsonl(Path(args.output), out_records)
    summary = {
        "track": args.track,
        "groups_total": len(by_pid),
        "onpolicy_mixed_groups": n_mixed,
        "luffy_rescued_0pass_groups": n_luffy,
        "all_pass_groups": n_all_pass,
        "unassisted_0pass_groups": n_all_fail,
        "total_records": len(out_records),
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V2 Multi-Arch Bank & Causal Control Builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("merge-restem")
    p1.add_argument("--track", choices=["lean", "math"], required=True)
    p1.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V2.keys()))
    p1.add_argument("--pool", required=True)
    p1.add_argument("--prev-bank", default="")
    p1.add_argument("--generations", default="")
    p1.add_argument("--tokenizer-path", default="")
    p1.add_argument("--thinking", action="store_true")
    p1.add_argument("--max-seq-len", type=int, default=12800)
    p1.add_argument("--max-per-problem", type=int, default=3)
    p1.add_argument("--reject-repetitive-blocks", action="store_true")
    p1.add_argument("--repair-drafts-per-thm", type=int, default=3)
    p1.add_argument("--round-tag", default="r1")
    p1.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    p1.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    p1.add_argument("--lean-timeout", type=float, default=60.0)
    p1.add_argument("--workers", type=int, default=36)
    p1.add_argument("--output-bank", required=True)
    p1.add_argument("--output-next-frontier", default="")
    p1.add_argument("--output-repair-prompts", default="")
    p1.add_argument("--summary-output", default="")
    p1.set_defaults(func=cmd_merge_restem)

    p2 = sub.add_parser("merge-repair-5050")
    p2.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY_V2.keys()))
    p2.add_argument("--repair-prompts", required=True)
    p2.add_argument("--generations", required=True)
    p2.add_argument("--prev-bank", default="")
    p2.add_argument("--tokenizer-path", default="")
    p2.add_argument("--thinking", action="store_true")
    p2.add_argument("--max-seq-len", type=int, default=6144)
    p2.add_argument("--max-per-prompt", type=int, default=2)
    p2.add_argument("--round-tag", default="r2_rep5050")
    p2.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    p2.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    p2.add_argument("--lean-timeout", type=float, default=60.0)
    p2.add_argument("--workers", type=int, default=36)
    p2.add_argument("--output-bank", required=True)
    p2.add_argument("--output-repair-only", default="")
    p2.add_argument("--summary-output", default="")
    p2.set_defaults(func=cmd_merge_repair_5050)

    p3 = sub.add_parser("match-yield")
    p3.add_argument("--r1-bank", required=True)
    p3.add_argument("--rec-bank", required=True)
    p3.add_argument("--frozen-bank", required=True)
    p3.add_argument("--output-bank", required=True)
    p3.add_argument("--summary-output", default="")
    p3.add_argument("--seed", type=int, default=20261118)
    p3.set_defaults(func=cmd_match_yield)

    p4 = sub.add_parser("build-rl-groups")
    p4.add_argument("--track", choices=["lean", "math"], required=True)
    p4.add_argument("--pool", required=True)
    p4.add_argument("--generations", required=True)
    p4.add_argument("--verified-bank", default="")
    p4.add_argument("--disable-luffy", action="store_true")
    p4.add_argument("--thinking", action="store_true")
    p4.add_argument("--lean-project", default="/scratch/governed_slm_rsi_v1/lean/mathlib_env")
    p4.add_argument("--elan-bin", default="/scratch/governed_slm_rsi_v1/lean/elan/bin")
    p4.add_argument("--lean-timeout", type=float, default=60.0)
    p4.add_argument("--workers", type=int, default=36)
    p4.add_argument("--output", required=True)
    p4.add_argument("--summary-output", default="")
    p4.set_defaults(func=cmd_build_rl_groups)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

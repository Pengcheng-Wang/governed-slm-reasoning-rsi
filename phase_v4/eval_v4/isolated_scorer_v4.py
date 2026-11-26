#!/usr/bin/env python3
"""
Out-of-process isolated scorer v4 (`python -I -B eval_v4/isolated_scorer_v4.py`)
for Cross-Family Governed Post-Training (`Inkling-2-Mini`, `Gemma 4 31B`, `Gemma 4 26B-A4B`, `Muse Glimmer 30B`).

Preserves 100% metric & verifier equivalence with Phase V3 (`isolated_scorer_v3.py`) while:
  1. Routing raw completions through `channel_normalizer_v4` (`--channel-family`).
  2. Emitting per-item `aux_metrics` (`stop_rate`, `pass_at_k`) alongside `stratum_id` and `cluster_id`
     for `governed_loop_guard_v38.py` (`aggregation_weighting = "equal_stratum"` + `gate-confirm-iut`).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import gzip
import json
import math
import multiprocessing as mp
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

EVAL_V4_DIR = Path(__file__).resolve().parent
if str(EVAL_V4_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V4_DIR))

from channel_normalizer_v4 import VALID_CHANNEL_FAMILIES  # noqa: E402
from lean4_verifier_v4 import verify_lean_candidate  # noqa: E402
from math_verifier_v4 import exact_integer_equivalent, grade_symbolic_math, score_math_response  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    pos = q * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (float(math.comb(n - c, k)) / float(math.comb(n, k)))


def normalize_extracted_surface(ans: str) -> str:
    import fractions

    s = ans.strip().strip("$").strip()
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s).strip()
    s = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", s)
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", s):
        s = str(int(s.replace(",", "")))
    elif re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        s = str(int(s.split(".", 1)[0]))
    else:
        m_frac = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s)
        if m_frac and int(m_frac.group(3)) != 0:
            sign = -1 if m_frac.group(1) == "-" else 1
            fr = sign * fractions.Fraction(int(m_frac.group(2)), int(m_frac.group(3)))
            return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
        m_slash = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", s)
        if m_slash and int(m_slash.group(2)) != 0:
            fr = fractions.Fraction(int(m_slash.group(1)), int(m_slash.group(2)))
            return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
        if re.fullmatch(r"[+-]?(?:\d+\.\d+|\.\d+)", s) and len(s) <= 18:
            fr = fractions.Fraction(s)
            return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
    return s


def _answers_equivalent(a: str, b: str, *, problem: str) -> bool:
    na, nb = normalize_extracted_surface(a), normalize_extracted_surface(b)
    if na == nb:
        return True
    if re.fullmatch(r"[+-]?\d+", nb):
        try:
            return exact_integer_equivalent(a, nb)
        except Exception:
            pass
    if re.fullmatch(r"[+-]?\d+", na):
        try:
            return exact_integer_equivalent(b, na)
        except Exception:
            pass
    try:
        return bool(grade_symbolic_math(a, b, problem=problem)["correct"])
    except Exception:
        return False


def conservative_majority_vote(
    extracted_answers: list[str | None],
    correct_flags: list[bool],
    *,
    problem: str = "",
) -> bool:
    valid_pairs = [
        (a.strip(), bool(ok))
        for a, ok in zip(extracted_answers, correct_flags)
        if a is not None and a.strip()
    ]
    if not valid_pairs:
        return False

    c_true = sum(1 for _, ok in valid_pairs if ok)
    if c_true == 0:
        return False

    wrong_raw = [a for a, ok in valid_pairs if not ok]
    if len(wrong_raw) < c_true:
        return True

    by_norm: dict[str, tuple[str, int]] = {}
    for w in wrong_raw:
        nk = normalize_extracted_surface(w)
        if nk in by_norm:
            rep, cnt = by_norm[nk]
            by_norm[nk] = (rep, cnt + 1)
        else:
            by_norm[nk] = (w, 1)
    surface_groups = sorted(by_norm.values(), key=lambda x: -x[1])

    if surface_groups[0][1] >= c_true:
        return False
    if len(surface_groups) == 1:
        return c_true > surface_groups[0][1]

    clusters: list[list[Any]] = []
    for rep, cnt in surface_groups:
        merged = False
        for cl in clusters:
            if _answers_equivalent(rep, cl[0], problem=problem):
                cl[1] += cnt
                if cl[1] >= c_true:
                    return False
                merged = True
                break
        if not merged:
            clusters.append([rep, cnt])

    max_wrong = max((cl[1] for cl in clusters), default=0)
    return c_true > max_wrong


def _score_one_lean(task: tuple[dict[str, Any], dict[str, Any], str, str, bool, str, float]) -> dict[str, Any]:
    gen_row, spec_row, lean_proj, elan_bin, thinking, channel_family, timeout_s = task
    res = verify_lean_candidate(
        gen_row["text"],
        spec_row["formal_statement"],
        lean_project=Path(lean_proj),
        elan_bin=Path(elan_bin),
        finish_reason=str(gen_row.get("finish_reason", "unknown")),
        thinking=thinking,
        channel_family=channel_family,
        strict_format=False,
        timeout_s=timeout_s,
        lean_header=spec_row.get("lean_header"),
    )
    return {
        "id": str(gen_row["id"]),
        "sample_idx": int(gen_row["sample_idx"]),
        "completion_tokens": int(gen_row.get("completion_tokens", 0)),
        "finish_reason": str(gen_row.get("finish_reason", "unknown")),
        **res,
    }


def _score_one_math(task: tuple[dict[str, Any], dict[str, Any], bool, str]) -> dict[str, Any]:
    gen_row, spec_row, thinking, channel_family = task
    res = score_math_response(
        gen_row["text"],
        str(spec_row["gold"]),
        finish_reason=str(gen_row.get("finish_reason", "unknown")),
        thinking=thinking,
        channel_family=channel_family,
        integer_task=bool(spec_row.get("integer_task", False)),
        problem=str(spec_row.get("problem", "")),
    )
    return {
        "id": str(gen_row["id"]),
        "sample_idx": int(gen_row["sample_idx"]),
        "completion_tokens": int(gen_row.get("completion_tokens", 0)),
        **res,
    }


def _fallback_timed_out_math(gr: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    stopped = str(gr.get("finish_reason", "unknown")) == "stop"
    truncated = str(gr.get("finish_reason", "unknown")) == "length"
    return {
        "id": str(gr["id"]),
        "sample_idx": int(gr.get("sample_idx", 0)),
        "completion_tokens": int(gr.get("completion_tokens", 0)),
        "finish_reason": str(gr.get("finish_reason", "unknown")),
        "stopped": stopped,
        "truncated": truncated,
        "strict_extracted": None,
        "lenient_extracted": None,
        "strict_correct": False,
        "lenient_correct": False,
        "strict_parse_failure": stopped,
        "lenient_parse_failure": stopped,
        "verifier_error": f"TimeoutKilled:{timeout_s:.1f}s",
        "sft_clean_admissible": False,
    }


def _child_score_math_chunk(tasks_chunk: list[tuple[dict[str, Any], dict[str, Any], bool, str]], out_q: Any) -> None:
    try:
        res = [_score_one_math(t) for t in tasks_chunk]
        out_q.put(res)
    except BaseException as exc:
        out_q.put({"__fatal_error__": repr(exc)})


def _mp_ctx() -> Any:
    return mp.get_context("fork" if sys.platform != "win32" else "spawn")


def _run_math_chunk_with_kill_guard(
    tasks_chunk: list[tuple[dict[str, Any], dict[str, Any], bool, str]],
    timeout_s: float,
) -> list[dict[str, Any]] | None:
    ctx = _mp_ctx()
    q = ctx.Queue()
    p = ctx.Process(target=_child_score_math_chunk, args=(tasks_chunk, q))
    p.start()
    got: Any = None
    timed_out = False
    try:
        got = q.get(timeout=timeout_s)
    except Exception:
        got = None
    p.join(timeout=1.0)
    if p.is_alive():
        timed_out = True
        p.kill()
        p.join()
    if isinstance(got, dict) and "__fatal_error__" in got:
        raise RuntimeError(f"Fatal internal error in math scorer worker: {got['__fatal_error__']}")
    if got is None and (not timed_out) and p.exitcode not in (0, None):
        raise RuntimeError(f"Math scorer worker exited abnormally without queue payload (exitcode={p.exitcode})")
    return got


def score_math_tasks_safely(
    tasks_m: list[tuple[dict[str, Any], dict[str, Any], bool, str]],
    *,
    workers: int = 32,
    chunk_size: int = 24,
    chunk_timeout_s: float = 20.0,
    item_timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    if sys.platform == "win32" and len(tasks_m) <= 64:
        res_fast = _run_math_chunk_with_kill_guard(tasks_m, timeout_s=chunk_timeout_s)
        if res_fast is not None and len(res_fast) == len(tasks_m):
            return res_fast

    ctx = _mp_ctx()
    chunks = [tasks_m[i : i + chunk_size] for i in range(0, len(tasks_m), chunk_size)]
    results_by_chunk: list[list[dict[str, Any]] | None] = [None] * len(chunks)

    active: list[tuple[int, Any, Any, float]] = []
    next_idx = 0
    while next_idx < len(chunks) or active:
        while next_idx < len(chunks) and len(active) < max(1, workers):
            q = ctx.Queue()
            p = ctx.Process(target=_child_score_math_chunk, args=(chunks[next_idx], q))
            p.start()
            active.append((next_idx, p, q, time.monotonic()))
            next_idx += 1

        still_running: list[tuple[int, Any, Any, float]] = []
        for c_idx, p, q, t_start in active:
            if not q.empty():
                try:
                    results_by_chunk[c_idx] = q.get_nowait()
                except Exception:
                    results_by_chunk[c_idx] = None
                p.join(timeout=0.5)
                if p.is_alive():
                    p.kill()
                    p.join()
            elif not p.is_alive():
                try:
                    results_by_chunk[c_idx] = q.get_nowait()
                except Exception:
                    results_by_chunk[c_idx] = None
                p.join()
                if results_by_chunk[c_idx] is None and p.exitcode not in (0, None):
                    raise RuntimeError(f"Math scorer chunk {c_idx} exited abnormally (exitcode={p.exitcode})")
            elif (time.monotonic() - t_start) > chunk_timeout_s:
                p.kill()
                p.join()
                results_by_chunk[c_idx] = None
            else:
                still_running.append((c_idx, p, q, t_start))
        active = still_running
        if active:
            time.sleep(0.05)

    final_scored: list[dict[str, Any]] = []
    for c_idx, ch in enumerate(chunks):
        c_res = results_by_chunk[c_idx]
        if isinstance(c_res, dict) and "__fatal_error__" in c_res:
            raise RuntimeError(f"Fatal internal error in math scorer chunk {c_idx}: {c_res['__fatal_error__']}")
        if c_res is not None and len(c_res) == len(ch):
            final_scored.extend(c_res)
        else:
            for item_task in ch:
                single = _run_math_chunk_with_kill_guard([item_task], timeout_s=item_timeout_s)
                if single is not None and len(single) == 1:
                    final_scored.append(single[0])
                else:
                    final_scored.append(_fallback_timed_out_math(item_task[0], item_timeout_s))
    return final_scored


def compute_paired_transitions(
    base_payload: dict[str, Any],
    cand_payload: dict[str, Any],
    *,
    n_boot: int = 10000,
    seed: int = 20260325,
) -> dict[str, Any]:
    if base_payload.get("track") != cand_payload.get("track"):
        raise ValueError("Mismatched tracks in compute_paired_transitions")
    if int(base_payload["expected_k"]) != int(cand_payload["expected_k"]):
        raise ValueError("Mismatched expected_k in compute_paired_transitions")

    track = str(base_payload["track"])
    k_max = int(base_payload["expected_k"])
    base_by_id = {str(it["id"]): it for it in base_payload["items"]}
    cand_by_id = {str(it["id"]): it for it in cand_payload["items"]}
    if set(base_by_id.keys()) != set(cand_by_id.keys()):
        raise ValueError("Item ID mismatch between baseline and candidate payloads")

    pids = sorted(base_by_id.keys())
    n_items = len(pids)
    if n_items == 0:
        raise ValueError("Empty items list")

    newly_solved = 0
    forgotten = 0
    both_solved = 0
    both_unsolved = 0
    interior_deltas: list[float] = []
    full_mastery_base = 0
    full_mastery_cand = 0

    by_stratum: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)

    for pid in pids:
        b_it = base_by_id[pid]
        c_it = cand_by_id[pid]
        cb = int(round(float(b_it.get("n_correct", float(b_it["value"]) * k_max))))
        cc = int(round(float(c_it.get("n_correct", float(c_it["value"]) * k_max))))
        if cb == 0 and cc > 0:
            newly_solved += 1
        elif cb > 0 and cc == 0:
            forgotten += 1
        elif cb > 0 and cc > 0:
            both_solved += 1
        else:
            both_unsolved += 1

        if 0 < cb < k_max:
            interior_deltas.append((cc - cb) / float(k_max))
        if cb == k_max:
            full_mastery_base += 1
        if cc == k_max:
            full_mastery_cand += 1

        strat = str(b_it.get("stratum", "default"))
        by_stratum[strat].append((b_it, c_it))

    newly_solved_rate = newly_solved / float(n_items)
    forgotten_rate = forgotten / float(n_items)
    net_frontier_expansion = newly_solved_rate - forgotten_rate
    unsolved_base = newly_solved + both_unsolved
    solved_base = forgotten + both_solved

    k_grid = [k for k in (1, 2, 4, 8) if k <= k_max]
    if k_max not in k_grid:
        k_grid.append(k_max)

    pass_curve_deltas: dict[str, float] = {}
    for k in k_grid:
        b_pk = statistics.fmean(
            unbiased_pass_at_k(k_max, int(round(float(base_by_id[p].get("n_correct", float(base_by_id[p]["value"]) * k_max)))), k)
            for p in pids
        )
        c_pk = statistics.fmean(
            unbiased_pass_at_k(k_max, int(round(float(cand_by_id[p].get("n_correct", float(cand_by_id[p]["value"]) * k_max)))), k)
            for p in pids
        )
        pass_curve_deltas[f"delta_pass_at_{k}"] = c_pk - b_pk

    delta_slope = pass_curve_deltas[f"delta_pass_at_{k_max}"] - pass_curve_deltas["delta_pass_at_1"]

    rng = random.Random(seed)
    boot_avg: list[float] = []
    boot_eq_avg: list[float] = []
    boot_passk: list[float] = []
    boot_slope: list[float] = []
    boot_maj: list[float] = []
    strata_keys = sorted(by_stratum.keys())
    has_maj = track == "math" and all(("maj_ok" in base_by_id[p] and "maj_ok" in cand_by_id[p]) for p in pids)

    strat_tuples: list[list[tuple[float, float, float, float]]] = []
    for sk in strata_keys:
        rows_s: list[tuple[float, float, float, float]] = []
        for b_it, c_it in by_stratum[sk]:
            cb = int(round(float(b_it.get("n_correct", float(b_it["value"]) * k_max))))
            cc = int(round(float(c_it.get("n_correct", float(c_it["value"]) * k_max))))
            d_avg = (cc - cb) / float(k_max)
            d_pk = (1.0 if cc > 0 else 0.0) - (1.0 if cb > 0 else 0.0)
            d_sl = d_pk - d_avg
            d_mj = (float(c_it.get("maj_ok", 0.0)) - float(b_it.get("maj_ok", 0.0))) if has_maj else 0.0
            rows_s.append((d_avg, d_pk, d_sl, d_mj))
        strat_tuples.append(rows_s)

    n_strata = max(1, len(strat_tuples))
    for _ in range(n_boot):
        s_avg = s_pk = s_sl = s_mj = 0.0
        s_eq = 0.0
        for rows_s in strat_tuples:
            ns = len(rows_s)
            st_sum = 0.0
            for _j in range(ns):
                da, dp, ds, dm = rows_s[rng.randrange(ns)]
                s_avg += da
                st_sum += da
                s_pk += dp
                s_sl += ds
                s_mj += dm
            s_eq += st_sum / float(ns)
        inv_n = 1.0 / float(n_items)
        boot_avg.append(s_avg * inv_n)
        boot_eq_avg.append(s_eq / float(n_strata))
        boot_passk.append(s_pk * inv_n)
        boot_slope.append(s_sl * inv_n)
        if has_maj:
            boot_maj.append(s_mj * inv_n)

    out: dict[str, Any] = {
        "baseline_id": base_payload.get("candidate_id", "baseline"),
        "candidate_id": cand_payload.get("candidate_id", "candidate"),
        "n_problems": n_items,
        "expected_k": k_max,
        "newly_solved_count": newly_solved,
        "forgotten_count": forgotten,
        "both_solved_count": both_solved,
        "both_unsolved_count": both_unsolved,
        "newly_solved_rate": newly_solved_rate,
        "forgotten_rate": forgotten_rate,
        "newly_solved_of_unsolved": newly_solved / float(max(1, unsolved_base)),
        "forgotten_of_solved": forgotten / float(max(1, solved_base)),
        "net_frontier_expansion": net_frontier_expansion,
        "interior_sharpening_delta": statistics.fmean(interior_deltas) if interior_deltas else 0.0,
        "interior_item_count": len(interior_deltas),
        "full_mastery_delta": (full_mastery_cand - full_mastery_base) / float(n_items),
        **pass_curve_deltas,
        f"delta_capability_slope_{k_max}_1": delta_slope,
        "ci95_delta_avg_at_k": [_quantile(boot_avg, 0.025), _quantile(boot_avg, 0.975)],
        "ci95_delta_equal_stratum_avg_at_k": [_quantile(boot_eq_avg, 0.025), _quantile(boot_eq_avg, 0.975)],
        f"ci95_delta_pass_at_{k_max}": [_quantile(boot_passk, 0.025), _quantile(boot_passk, 0.975)],
        f"ci95_delta_capability_slope_{k_max}_1": [_quantile(boot_slope, 0.025), _quantile(boot_slope, 0.975)],
    }
    if has_maj:
        d_maj_obs = statistics.fmean(
            float(cand_by_id[p]["maj_ok"]) - float(base_by_id[p]["maj_ok"]) for p in pids
        )
        out[f"delta_maj_at_{k_max}"] = d_maj_obs
        out[f"ci95_delta_maj_at_{k_max}"] = [_quantile(boot_maj, 0.025), _quantile(boot_maj, 0.975)]
    return out


def score_dataset(args: argparse.Namespace) -> int:
    split_rows = _read_jsonl(Path(args.split_file))
    spec_by_id = {str(r["id"]): r for r in split_rows}
    if len(spec_by_id) != len(split_rows):
        raise SystemExit("Duplicate IDs in split file; pass disambiguated view for multi-domain Holdout")

    gen_rows = _read_jsonl(Path(args.generations))
    expected_keys = {(pid, s) for pid in spec_by_id for s in range(args.expected_k)}
    seen_keys: set[tuple[str, int]] = set()
    ordered_gen: list[dict[str, Any]] = []
    for gr in gen_rows:
        key = (str(gr["id"]), int(gr["sample_idx"]))
        if key not in expected_keys:
            raise SystemExit(f"Unexpected generation key {key} not in frozen split x expected_k={args.expected_k}")
        if key in seen_keys:
            raise SystemExit(f"Duplicate generation key {key}")
        seen_keys.add(key)
        ordered_gen.append(gr)

    missing = expected_keys - seen_keys
    if missing:
        sample_miss = sorted(missing)[:5]
        raise SystemExit(f"Missing {len(missing)} expected (id, sample_idx) records; first={sample_miss}")

    ordered_gen.sort(key=lambda r: (str(r["id"]), int(r["sample_idx"])))

    if args.track == "lean":
        tasks_l = [
            (
                gr,
                spec_by_id[str(gr["id"])],
                args.lean_project,
                args.elan_bin,
                args.thinking,
                args.channel_family,
                args.lean_timeout,
            )
            for gr in ordered_gen
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            scored = list(ex.map(_score_one_lean, tasks_l, chunksize=4))
    else:
        tasks_m = [(gr, spec_by_id[str(gr["id"])], args.thinking, args.channel_family) for gr in ordered_gen]
        scored = score_math_tasks_safely(
            tasks_m,
            workers=min(args.workers, 32),
            chunk_timeout_s=args.math_chunk_timeout,
            item_timeout_s=args.math_item_timeout,
        )

    if args.scored_details_output:
        det_path = Path(args.scored_details_output)
        det_path.parent.mkdir(parents=True, exist_ok=True)
        det_opener = gzip.open if det_path.suffix == ".gz" else open
        with det_opener(det_path, "wt", encoding="utf-8") as fh:
            for row in scored:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in scored:
        by_pid[row["id"]].append(row)

    k_max = args.expected_k
    k_grid = [k for k in (1, 2, 4, 8) if k <= k_max]
    if k_max not in k_grid:
        k_grid.append(k_max)

    items_payload: list[dict[str, Any]] = []
    pass_curves: dict[int, list[float]] = {k: [] for k in k_grid}
    strict_avg_vals: list[float] = []
    maj_flags: list[float] = []
    distinct_ans_counts: list[float] = []
    learnable_flags: list[float] = []
    stratum_values: dict[str, list[float]] = collections.defaultdict(list)

    all_tokens = [float(r["completion_tokens"]) for r in scored]
    correct_tokens: list[float] = []
    incorrect_tokens: list[float] = []
    stop_flags = [1.0 if r["finish_reason"] == "stop" else 0.0 for r in scored]
    trunc_flags = [1.0 if r["finish_reason"] == "length" else 0.0 for r in scored]
    sft_clean_flags = [1.0 if r.get("sft_clean_admissible", False) else 0.0 for r in scored]
    total_correct_rollouts = 0

    for pid in sorted(spec_by_id.keys()):
        spec = spec_by_id[pid]
        stratum = str(spec.get("stratum") or spec.get("source") or "default")
        prows = sorted(by_pid[pid], key=lambda x: x["sample_idx"])
        if args.track == "lean":
            bool_succs = [bool(r["ok"]) for r in prows]
            strict_succs = [1.0 if r["sft_clean_admissible"] else 0.0 for r in prows]
            maj_val: float | None = None
        else:
            bool_succs = [bool(r["lenient_correct"]) for r in prows]
            strict_succs = [1.0 if r["strict_correct"] else 0.0 for r in prows]
            extracted_list = [r.get("lenient_extracted") for r in prows]
            maj_ok = conservative_majority_vote(
                extracted_list,
                bool_succs,
                problem=str(spec.get("problem", "")),
            )
            maj_val = 1.0 if maj_ok else 0.0
            maj_flags.append(maj_val)
            norm_nonempty = {
                normalize_extracted_surface(a)
                for a in extracted_list
                if a is not None and a.strip()
            }
            distinct_ans_counts.append(float(len(norm_nonempty)))

        for r, ok_flag in zip(prows, bool_succs):
            tok_f = float(r["completion_tokens"])
            if ok_flag:
                correct_tokens.append(tok_f)
                total_correct_rollouts += 1
            else:
                incorrect_tokens.append(tok_f)

        n_succ = sum(1 for ok_flag in bool_succs if ok_flag)
        avg_i = n_succ / float(k_max)
        stop_i = statistics.fmean(1.0 if r["finish_reason"] == "stop" else 0.0 for r in prows)
        for k in k_grid:
            pass_curves[k].append(unbiased_pass_at_k(k_max, n_succ, k))
        pass_k_i = 1.0 if n_succ >= 1 else 0.0
        learnable_i = 1.0 if (0 < n_succ < k_max) else 0.0

        item_entry: dict[str, Any] = {
            "id": pid,
            "cluster_id": pid,
            "stratum": stratum,
            "stratum_id": stratum,
            "value": avg_i,
            "n_correct": n_succ,
            "pass_at_k": pass_k_i,
            "stop_rate": stop_i,
            "aux_metrics": {
                "stop_rate": stop_i,
                f"{args.track}_pass_at_{k_max}": pass_k_i,
            },
        }
        if maj_val is not None:
            item_entry["maj_ok"] = maj_val
            item_entry["aux_metrics"][f"math_maj_at_{k_max}"] = maj_val
        items_payload.append(item_entry)
        stratum_values[stratum].append(avg_i)
        strict_avg_vals.append(statistics.fmean(strict_succs))
        learnable_flags.append(learnable_i)

    avg_at_k = statistics.fmean(x["value"] for x in items_payload)
    pass_at_k = pass_curves[k_max]
    pass_at_k_mean = statistics.fmean(pass_at_k)
    pass_at_1_mean = statistics.fmean(pass_curves[1])
    strict_avg_at_k = statistics.fmean(strict_avg_vals)
    solved_count = float(sum(int(x > 0.5) for x in pass_at_k))
    equal_stratum_avg = statistics.fmean(statistics.fmean(vs) for vs in stratum_values.values())

    metrics: dict[str, Any] = {
        f"{args.track}_avg_at_{k_max}": avg_at_k,
        f"{args.track}_equal_stratum_avg_at_{k_max}": equal_stratum_avg,
        f"{args.track}_strict_avg_at_{k_max}": strict_avg_at_k,
        "solved_problem_count": solved_count,
        "learnable_fraction": statistics.fmean(learnable_flags),
        "stop_rate": statistics.fmean(stop_flags),
        "truncation_rate": statistics.fmean(trunc_flags),
        "admitted_trace_fraction": statistics.fmean(sft_clean_flags),
        "median_tokens": _quantile(all_tokens, 0.50),
        "p95_tokens": _quantile(all_tokens, 0.95),
        "p99_tokens": _quantile(all_tokens, 0.99),
        "mean_tokens": statistics.fmean(all_tokens),
        "mean_correct_tokens": statistics.fmean(correct_tokens) if correct_tokens else 0.0,
        "mean_incorrect_tokens": statistics.fmean(incorrect_tokens) if incorrect_tokens else 0.0,
        "tokens_per_correct": sum(all_tokens) / float(max(1, total_correct_rollouts)),
        "total_correct_rollouts": float(total_correct_rollouts),
    }

    for k in k_grid:
        metrics[f"{args.track}_pass_at_{k}"] = statistics.fmean(pass_curves[k])
    metrics[f"capability_slope_{k_max}_1"] = pass_at_k_mean - pass_at_1_mean
    if 8 in pass_curves:
        metrics["capability_slope_8_1"] = statistics.fmean(pass_curves[8]) - pass_at_1_mean

    for strat_name, vs in sorted(stratum_values.items()):
        safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", strat_name).strip("_").lower()
        metrics[f"stratum_{safe_key}_avg_at_{k_max}"] = statistics.fmean(vs)

    if args.track == "lean":
        axiom_clean = [
            1.0
            for r in scored
            if (not r["ok"]) or set(r.get("axioms", [])).issubset({"propext", "Classical.choice", "Quot.sound"})
        ]
        metrics["axiom_audit_clean_rate"] = float(len(axiom_clean)) / float(len(scored))
    else:
        metrics[f"math_maj_at_{k_max}"] = statistics.fmean(maj_flags)
        metrics[f"distinct_answers_at_{k_max}"] = statistics.fmean(distinct_ans_counts)
        stopped_rows = [r for r in scored if r.get("stopped", r.get("finish_reason") == "stop")]
        if stopped_rows:
            metrics["natural_stop_parse_failure_rate"] = (
                sum(1.0 for r in stopped_rows if r.get("lenient_parse_failure", False)) / float(len(stopped_rows))
            )
        else:
            metrics["natural_stop_parse_failure_rate"] = 1.0
        metrics["overall_parse_failure_rate"] = statistics.fmean(
            1.0 if r.get("lenient_parse_failure", False) else 0.0 for r in scored
        )

    if args.extra_metrics_json:
        extra = json.loads(Path(args.extra_metrics_json).read_text(encoding="utf-8"))
        for k, v in extra.items():
            metrics[k] = v

    out_payload: dict[str, Any] = {
        "schema_version": "4.0",
        "scored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": args.track,
        "candidate_id": args.candidate_id,
        "protocol_id": args.protocol_id,
        "channel_family": args.channel_family,
        "env_fingerprint": args.env_fingerprint,
        "n_problems": len(items_payload),
        "expected_k": k_max,
        "metrics": metrics,
        "items": items_payload,
    }

    if args.compare_baseline:
        base_payload = json.loads(Path(args.compare_baseline).read_text(encoding="utf-8"))
        transitions = compute_paired_transitions(base_payload, out_payload)
        out_payload["paired_transitions"] = transitions
        metrics["newly_solved_rate"] = transitions["newly_solved_rate"]
        metrics["forgotten_rate"] = transitions["forgotten_rate"]
        metrics["net_frontier_expansion"] = transitions["net_frontier_expansion"]
        metrics["interior_sharpening_delta"] = transitions["interior_sharpening_delta"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SCORED", "output": out_path.as_posix(), "metrics": metrics}, indent=2))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        ap_cmp = argparse.ArgumentParser()
        ap_cmp.add_argument("mode", choices=["compare"])
        ap_cmp.add_argument("--baseline", required=True)
        ap_cmp.add_argument("--candidate", required=True)
        ap_cmp.add_argument("--n-boot", type=int, default=10000)
        ap_cmp.add_argument("--output", default="")
        cargs = ap_cmp.parse_args()
        b_pay = json.loads(Path(cargs.baseline).read_text(encoding="utf-8"))
        c_pay = json.loads(Path(cargs.candidate).read_text(encoding="utf-8"))
        res = compute_paired_transitions(b_pay, c_pay, n_boot=cargs.n_boot)
        if cargs.output:
            Path(cargs.output).parent.mkdir(parents=True, exist_ok=True)
            Path(cargs.output).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    ap = argparse.ArgumentParser(description="Out-of-process isolated scorer v4 for Cross-Family Governed Post-Training")
    ap.add_argument("--track", choices=["lean", "math"], required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--expected-k", type=int, required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--channel-family", choices=VALID_CHANNEL_FAMILIES, default="auto")
    ap.add_argument("--env-fingerprint", default="a100_80gb_bf16_v4")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--lean-project", default="/scratch/governed_crossfamily_v4/lean/mathlib_env")
    ap.add_argument("--elan-bin", default="/scratch/governed_crossfamily_v4/lean/elan/bin")
    ap.add_argument("--lean-timeout", type=float, default=60.0)
    ap.add_argument("--math-chunk-timeout", type=float, default=20.0)
    ap.add_argument("--math-item-timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--extra-metrics-json", default="")
    ap.add_argument("--compare-baseline", default="")
    ap.add_argument("--scored-details-output", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    return score_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())

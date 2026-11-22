#!/usr/bin/env python3
"""
Phase V3 Track-M Governed Round-2 Driver: Horizon-Aligned Long-CoT `2-SFT + 3-Proposal RLVR`.

Addresses the Round-1 Search diagnostic finding (`r1_concise_postthink_sft`):
  - In Round 1, stripping `<think>...</think>` into ~500-token post-think summaries compressed
    test-time reasoning horizon 10.8x (`5,577 -> 515` tokens on `9B`; `520` tokens on `4B`) and
    reduced `math_avg_at_8` (`66.25% -> 50.00%` on `9B`; `64.92% -> 43.59%` on `4B`), and Stage-II
    ZPD SFT (`50.39%`, `547` tokens) could not recover lost multi-step search depth.
  - Round 2 preserves Round-1 trials in the append-only Search ledger (`round1_concise_postthink_sft`)
    with ZERO Confirmation looks spent, then executes Horizon-Aligned `2-SFT + 3-Proposal RLVR` from `B_0`:
      1. Stage I (`r2_s1_math_<mk>_horizon_breadth_sft`): Scale-aligned natural-stop multi-domain
         verified Olympiad derivations (`<= 10,000` target tokens, strictly `< 12,288` eval cap)
         combined with <= 12% clean unwrapped `Axion 1.0` + `DeepMath` `0/4` frontier rescues.
      2. Stage II (`r2_s2_math_<mk>_zpd_axion_sft`): Model-aware ZPD (`1..2/4` boundary) + `0/4`
         rescues + full-horizon (`~2,800`-token) cross-family-audited `Axion` strengthened problems
         (`4B`/`9B` only; `27B` strictly excluded per `B3`) + 20% Stage-I replay.
      3. Stage III (`r2_s3_math_<mk>_{dapo_asymclip,drgrpo_cispo,gspo_luffy}`): Rollout-matched
         2-epoch RLVR tournament (`pct50`, `pct100`) seeded from the Search-winning SFT anchor,
         followed by order-invariant `select-search-winner`, `Confirm_V3` IUT, atomic single-touch
         `Holdout_V3`, and Longitudinal evaluation.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
V3_ROOT = THIS_DIR.parent.parent
EVAL_V3_DIR = V3_ROOT / "eval_v3"
for p in (THIS_DIR, EVAL_V3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from math_verifier import score_math_response  # noqa: E402
from model_registry_v3 import MODEL_REGISTRY_V3  # noqa: E402
from prepare_v3_splits_and_corpora import (  # noqa: E402
    FourLayerDecontaminator,
    _is_scorable_exact_math_gold,
    _math_prompt_messages,
    _skeleton_sha,
)
from run_v3_node_worker import (  # noqa: E402
    GUARD_PY,
    PRIOR_CHAMPION_PATHS,
    RLVR_PY,
    SFT_PY,
    V1_ROOT,
    V2_ROOT,
    _read_jsonl,
    _run_checked,
    _write_jsonl,
    eval_and_score_model,
    generate_and_verify_shared_rlvr_rollouts,
    verify_manifest_pin,
)

R2_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "qwen35_4b": {
        "s1_lr": 4.5e-5,
        "s1_epochs": 2,
        "s1_max_steps": 150,
        "s2_lr": 2.0e-5,
        "s2_epochs": 2,
        "s2_max_steps": 68,
        "rl_lr": 2.0e-5,
        "rl_max_steps": 36,
        "rl_subset_n": 160,
    },
    "qwen35_9b": {
        "s1_lr": 3.5e-5,
        "s1_epochs": 2,
        "s1_max_steps": 140,
        "s2_lr": 1.6e-5,
        "s2_epochs": 2,
        "s2_max_steps": 64,
        "rl_lr": 1.5e-5,
        "rl_max_steps": 36,
        "rl_subset_n": 160,
    },
    "qwen38_27b": {
        "s1_lr": 1.8e-5,
        "s1_epochs": 2,
        "s1_max_steps": 96,
        "s2_lr": 9.0e-6,
        "s2_epochs": 2,
        "s2_max_steps": 44,
        "rl_lr": 8.0e-6,
        "rl_max_steps": 24,
        "rl_subset_n": 112,
    },
    "qwen38_35b_a3b": {
        "s1_lr": 2.2e-5,
        "s1_epochs": 2,
        "s1_max_steps": 100,
        "s2_lr": 1.1e-5,
        "s2_epochs": 2,
        "s2_max_steps": 48,
        "rl_lr": 9.0e-6,
        "rl_max_steps": 28,
        "rl_subset_n": 128,
    },
    "qwen3_30b_a3b": {
        "s1_lr": 2.2e-5,
        "s1_epochs": 2,
        "s1_max_steps": 100,
        "s2_lr": 1.1e-5,
        "s2_epochs": 2,
        "s2_max_steps": 48,
        "rl_lr": 9.0e-6,
        "rl_max_steps": 28,
        "rl_subset_n": 128,
    },
}


def _build_full_eval_decontaminator() -> FourLayerDecontaminator:
    decon = FourLayerDecontaminator()
    eval_glob_roots = [
        V1_ROOT / "data/splits",
        V1_ROOT / "eval/sealed_holdout",
        V2_ROOT / "data/splits_v2",
        V2_ROOT / "eval_v2/sealed_holdout",
        V3_ROOT / "data/splits_v3",
        V3_ROOT / "eval_v3/sealed_holdout",
    ]
    for eroot in eval_glob_roots:
        if not eroot.exists():
            continue
        for jf in sorted(eroot.glob("*.jsonl")):
            if "lean" in jf.name.lower():
                continue
            for r in _read_jsonl(jf):
                prob = str(r.get("problem") or "")
                if not prob and r.get("messages"):
                    prob = str(r["messages"][0].get("content", ""))
                if prob:
                    decon.register_math_eval(f"{jf.name}:{r.get('id', '')}", prob)
    return decon


def _unwrap_clean_long_cot(raw_sol: str, gold: str, problem: str, *, min_chars: int = 1200, max_chars: int = 24000) -> str | None:
    """
    Unwraps `<think>...</think>` into a single coherent step-by-step derivation (`thinking=False`
    compatible: zero `<think>`/`</think>` tags, bounded R1 verbal-filler hesitation, exact verified
    `\boxed{gold}` at end).
    """
    if not raw_sol:
        return None
    txt = raw_sol
    if "<answer>" in txt and "</answer>" in txt:
        pre = txt.split("<answer>", 1)[0]
        ans_blk = txt.split("<answer>", 1)[1].rsplit("</answer>", 1)[0]
        txt = pre + "\n\n" + ans_blk
    txt = txt.replace("<think>", "").replace("</think>", "").strip()
    if not (min_chars <= len(txt) <= max_chars):
        return None
    if "<think>" in txt or "</think>" in txt:
        return None
    # Prevent foreign verbal-hesitation loops (`"Wait, let me double check..."` x 20) from degrading stop_rate
    if len(re.findall(r"\bwait\b", txt, flags=re.IGNORECASE)) > 6:
        return None
    int_task = bool(re.fullmatch(r"[+-]?\d+", str(gold).strip()))
    sc = score_math_response(
        txt,
        str(gold).strip(),
        finish_reason="stop",
        thinking=False,
        integer_task=int_task,
        problem=problem,
    )
    if sc.get("strict_correct") and sc.get("sft_clean_admissible"):
        return txt
    return None


def rebuild_full_horizon_axion_strengthened(decon: FourLayerDecontaminator) -> Path:
    """
    Upgrades `v3_math_axion_strengthened.jsonl` (`56` cross-family-audited items) on `node-0` (or uses
    already-synced file) so `assistant` holds the median-length full-horizon (`~2,500-4,500`-token)
    verified `27B` consensus derivation from `solve27_k8.jsonl.gz` rather than the shortest trace.
    """
    out_path = V3_ROOT / "data/corpora_v3/v3_r2_math_axion_strengthened_fullcot.jsonl"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    orig_ax = V3_ROOT / "data/corpora_v3/v3_math_axion_strengthened.jsonl"
    solve27_gz = V3_ROOT / "results_v3/plane1/axion_work/solve27_k8.jsonl.gz"
    if not orig_ax.exists():
        _write_jsonl(out_path, [])
        return out_path

    orig_rows = _read_jsonl(orig_ax)
    by_id_27: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    if solve27_gz.exists():
        with gzip.open(solve27_gz, "rt", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    by_id_27[str(r["id"])].append(r)

    upgraded: list[dict[str, Any]] = []
    for row in orig_rows:
        prob = str(row.get("problem", ""))
        if decon.is_math_contaminated(prob):
            continue
        gold = str(row["gold"]).strip()
        int_task = bool(re.fullmatch(r"[+-]?\d+", gold))
        pid = str(row["id"])
        best_txt = str(row["messages"][-1]["content"])

        if pid in by_id_27:
            cands: list[tuple[int, str]] = []
            for gr in by_id_27[pid]:
                if str(gr.get("finish_reason")) != "stop":
                    continue
                ctoks = int(gr.get("completion_tokens", 0))
                t_str = str(gr.get("text", "")).strip()
                if "<think>" in t_str or "</think>" in t_str or not (900 <= ctoks <= 7200):
                    continue
                sc = score_math_response(t_str, gold, finish_reason="stop", thinking=False, integer_task=int_task, problem=prob)
                if sc.get("strict_correct") and sc.get("sft_clean_admissible"):
                    cands.append((ctoks, t_str))
            if cands:
                cands.sort(key=lambda x: x[0])
                # Pick median-horizon verified derivation (~2,500-4,000 tokens)
                best_txt = cands[len(cands) // 2][1]

        new_row = dict(row)
        new_row["messages"] = _math_prompt_messages(prob) + [{"role": "assistant", "content": best_txt}]
        upgraded.append(new_row)

    _write_jsonl(out_path, upgraded)
    print(f"[R2-Axion] Full-horizon strengthened items ready: {len(upgraded)} -> {out_path}", flush=True)
    return out_path


def build_horizon_aligned_corpora_for_model(
    model_key: str,
    decon: FourLayerDecontaminator,
    axion_full_path: Path,
    seed: int = 20260410,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """
    Constructs:
      - `s1_path`: Stage-I Horizon-Aligned Multi-Source Breadth SFT (`<= 10,000` target tokens)
      - `s2_path`: Stage-II ZPD (`1..2/4`) + `0/4` Rescues + Cross-Family `Axion` Strengthened (`4B`/`9B`) + 20% Replay
      - `rl_pool_path`: Active ZPD RLVR Pool (`v3_math_rl_pool640` upgraded with long-CoT `reference_solution`)
    """
    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"math_{model_key}"
    p2_dir.mkdir(parents=True, exist_ok=True)

    s1_path = p2_dir / f"r2_stage1_sft_math_{model_key}.jsonl"
    s2_path = p2_dir / f"r2_stage2_sft_math_{model_key}.jsonl"
    rl_pool_path = p2_dir / f"r2_rl_pool_math_{model_key}.jsonl"
    summary_path = p2_dir / f"r2_corpus_build_summary_{model_key}.json"
    if s1_path.exists() and s2_path.exists() and rl_pool_path.exists() and summary_path.exists():
        return s1_path, s2_path, rl_pool_path, json.loads(summary_path.read_text(encoding="utf-8"))

    rng = random.Random(seed)
    probe_json = json.loads((p1_dir / f"math_b0_{model_key}_stage2_probe_k4.json").read_text(encoding="utf-8"))
    probe_gz = p1_dir / f"math_b0_{model_key}_stage2_probe_k4.gen.jsonl.gz"
    s2_pool_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_math_stage2_frontier_pool.jsonl")
    spec_by_id = {str(r["id"]): r for r in s2_pool_rows}
    ncorr_by_id = {str(it["id"]): int(it["n_correct"]) for it in probe_json["items"]}

    gen_by_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    with gzip.open(probe_gz, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                gr = json.loads(ln)
                gen_by_id[str(gr["id"])].append(gr)

    seen_skels: set[str] = set()
    stage1_rows: list[dict[str, Any]] = []
    zpd_frontier_rows: list[dict[str, Any]] = []
    unsolved_ids: list[str] = []

    # 1A. Model-native verified natural-stop derivations on `1,600` hard DeepMath problems (`k=4` probe)
    n_probe_problems = 0
    for pid, sp in sorted(spec_by_id.items()):
        prob = str(sp["problem"])
        if decon.is_math_contaminated(prob):
            continue
        sk = _skeleton_sha(prob)
        gold = str(sp["gold"]).strip()
        int_task = bool(sp.get("integer_task", re.fullmatch(r"[+-]?\d+", gold) is not None))
        nc = ncorr_by_id.get(pid, 0)
        if nc == 0:
            unsolved_ids.append(pid)
            continue

        valid_traces: list[tuple[int, str]] = []
        for gr in gen_by_id.get(pid, []):
            if str(gr.get("finish_reason")) != "stop":
                continue
            ctoks = int(gr.get("completion_tokens", 0))
            txt = str(gr.get("text", "")).strip()
            if ctoks < 250 or ctoks > 9200 or "<think>" in txt or "</think>" in txt:
                continue
            sc = score_math_response(txt, gold, finish_reason="stop", thinking=False, integer_task=int_task, problem=prob)
            if sc.get("strict_correct") and sc.get("sft_clean_admissible"):
                valid_traces.append((ctoks, txt))

        if not valid_traces:
            continue
        valid_traces.sort(key=lambda x: x[0])
        seen_skels.add(sk)
        n_probe_problems += 1

        # Keep up to 2 verified natural-stop traces per problem (`train_sft_v3.py` normalizes weight by `1/count(pid)`)
        chosen_s1 = valid_traces[:2]
        for t_idx, (_, txt) in enumerate(chosen_s1):
            row_obj = {
                "id": f"{pid}:t{t_idx}",
                "problem_id": pid,
                "source": "deepmath_model_verified_cot",
                "problem": prob,
                "gold": gold,
                "finish_reason": "stop",
                "messages": _math_prompt_messages(prob) + [{"role": "assistant", "content": txt}],
            }
            stage1_rows.append(row_obj)
            # Exact ZPD `1..2/4` hard boundary items enter Stage-II Hard-Frontier SFT!
            if 1 <= nc <= 2:
                zpd_frontier_rows.append(row_obj)

    # 1B. Include decontaminated V2 `Omni-MATH Tier 3-4` verified natural-stop derivations if present on node (`4B`/`9B`)
    n_omni_added = 0
    v2_bank = V2_ROOT / "data/verified_banks_v2" / f"math_{model_key}_r2_cumulative.jsonl"
    if v2_bank.exists():
        by_v2_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in _read_jsonl(v2_bank):
            msgs = r.get("messages", [])
            if len(msgs) < 2:
                continue
            prob = str(msgs[0].get("content", ""))
            ans_txt = str(msgs[-1].get("content", ""))
            if decon.is_math_contaminated(prob) or "<think>" in ans_txt or "</think>" in ans_txt:
                continue
            if len(ans_txt) > 34000 or len(ans_txt) < 400:
                continue
            sk = _skeleton_sha(prob)
            if sk in seen_skels:
                continue
            by_v2_pid[str(r.get("problem_id", sk))].append(r)

        for vpid, vrows in sorted(by_v2_pid.items()):
            prob0 = str(vrows[0]["messages"][0]["content"])
            sk0 = _skeleton_sha(prob0)
            if sk0 in seen_skels:
                continue
            seen_skels.add(sk0)
            n_omni_added += 1
            for t_idx, vr in enumerate(vrows[:2]):
                robj = {
                    "id": f"omni_v2:{vpid}:t{t_idx}",
                    "problem_id": f"omni_v2:{vpid}",
                    "source": "omnimath_verified_cot",
                    "finish_reason": "stop",
                    "messages": vr["messages"],
                }
                stage1_rows.append(robj)
                if len(vrows) == 1:
                    zpd_frontier_rows.append(robj)

    # 1C. Off-Policy `Axion 1.0` + `DeepMath` Unwrapped Long-CoT Frontier Rescues (`<= 12%` of Stage-I unique problems)
    max_offpolicy_s1 = max(80, int(0.12 * (n_probe_problems + n_omni_added)))
    offpolicy_rescues: list[dict[str, Any]] = []
    rl_longcot_by_id: dict[str, str] = {}

    shared_ax_cache = V3_ROOT / "data/corpora_v3/v3_r2_shared_axion_longcot.jsonl"
    shared_dm_s2_cache = V3_ROOT / "data/corpora_v3/v3_r2_shared_dm_s2_longcot.jsonl"
    shared_rl_cache = V3_ROOT / "data/corpora_v3/v3_r2_shared_rl_pool640_longcot.jsonl"

    if shared_ax_cache.exists():
        for robj in _read_jsonl(shared_ax_cache):
            if len(offpolicy_rescues) >= max_offpolicy_s1 // 2:
                break
            prob = str(robj.get("problem", ""))
            sk = _skeleton_sha(prob)
            if decon.is_math_contaminated(prob) or sk in seen_skels:
                continue
            seen_skels.add(sk)
            offpolicy_rescues.append(robj)

    if shared_dm_s2_cache.exists():
        dm_s2_map = {str(r["problem_id"]): r for r in _read_jsonl(shared_dm_s2_cache)}
        for pid in unsolved_ids:
            if len(offpolicy_rescues) >= max_offpolicy_s1:
                break
            if pid not in dm_s2_map:
                continue
            robj = dm_s2_map[pid]
            prob = str(robj.get("problem", ""))
            sk = _skeleton_sha(prob)
            if decon.is_math_contaminated(prob) or sk in seen_skels:
                continue
            seen_skels.add(sk)
            offpolicy_rescues.append(robj)

    if shared_rl_cache.exists():
        for r in _read_jsonl(shared_rl_cache):
            rl_longcot_by_id[str(r["id"])] = str(r["reference_solution"])

    stage1_all = stage1_rows + offpolicy_rescues
    rng.shuffle(stage1_all)
    _write_jsonl(s1_path, stage1_all)

    # 2. Build Stage-II ZPD + `0/4` Rescues + Cross-Family `Axion` Strengthened (`4B`/`9B` ONLY; `27B` & `35B-A3B` excluded!) + 20% Stage-I Replay
    s2_core = list(zpd_frontier_rows) + list(offpolicy_rescues)
    n_axion_str = 0
    if model_key not in ("qwen38_27b", "qwen38_35b_a3b", "qwen3_30b_a3b") and axion_full_path.exists():
        ax_rows = _read_jsonl(axion_full_path)
        for ar in ax_rows:
            if not decon.is_math_contaminated(str(ar.get("problem", ""))):
                ar_copy = dict(ar)
                ar_copy["sample_weight"] = 1.5
                s2_core.append(ar_copy)
                n_axion_str += 1

    n_replay = max(1, int(round(0.20 * len(s2_core))))
    s1_for_replay = list(stage1_rows)
    rng.shuffle(s1_for_replay)
    s2_all = s2_core + s1_for_replay[:n_replay]
    rng.shuffle(s2_all)
    _write_jsonl(s2_path, s2_all)

    # 3. Build upgraded RLVR Pool (`reference_solution` upgraded to full long-CoT where available, prioritized by difficulty)
    orig_rl = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_math_rl_pool640.jsonl")
    upgraded_rl: list[dict[str, Any]] = []
    for rr in orig_rl:
        if decon.is_math_contaminated(str(rr.get("problem", ""))):
            continue
        rcopy = dict(rr)
        rid = str(rr["id"])
        if rid in rl_longcot_by_id:
            rcopy["reference_solution"] = rl_longcot_by_id[rid]
        upgraded_rl.append(rcopy)
    # Sort harder `7.0 <= difficulty <= 8.8` items first so active subset hits peak ZPD variance (`1 <= c <= 7`)!
    upgraded_rl.sort(key=lambda r: abs(float(r.get("difficulty", 7.8)) - 7.9))
    _write_jsonl(rl_pool_path, upgraded_rl)

    stats = {
        "model_key": model_key,
        "stage1_total_rows": len(stage1_all),
        "stage1_probe_problems": n_probe_problems,
        "stage1_omnimath_problems": n_omni_added,
        "stage1_offpolicy_longcot_rescues": len(offpolicy_rescues),
        "stage2_total_rows": len(s2_all),
        "stage2_zpd_rows": len(zpd_frontier_rows),
        "stage2_axion_strengthened_items": n_axion_str,
        "stage2_replay_rows": n_replay,
        "rl_pool_rows": len(upgraded_rl),
        "rl_longcot_upgraded_refs": len(rl_longcot_by_id),
    }
    summary_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[R2-Corpora] {json.dumps(stats)}", flush=True)
    return s1_path, s2_path, rl_pool_path, stats


def build_shared_longcot_caches(decon: FourLayerDecontaminator) -> None:
    """
    Runs once on `node-0` (using local HF cache) to materialize shared decontaminated unwrapped
    long-CoT pools so `node-1` and `node-2` require zero HF network downloads.
    """
    rebuild_full_horizon_axion_strengthened(decon)
    shared_ax_cache = V3_ROOT / "data/corpora_v3/v3_r2_shared_axion_longcot.jsonl"
    shared_dm_s2_cache = V3_ROOT / "data/corpora_v3/v3_r2_shared_dm_s2_longcot.jsonl"
    shared_rl_cache = V3_ROOT / "data/corpora_v3/v3_r2_shared_rl_pool640_longcot.jsonl"
    if shared_ax_cache.exists() and shared_dm_s2_cache.exists() and shared_rl_cache.exists():
        print("[R2-Shared-Cache] Already built.", flush=True)
        return

    from datasets import load_dataset
    from math_verifier import last_boxed

    ax_rows: list[dict[str, Any]] = []
    seen_ax: set[str] = set()
    for ax_sp in ("stage1", "stage2"):
        ax_ds = load_dataset("Axion1-0/axion-training-data", split=ax_sp)
        kept_sp = 0
        for r in ax_ds:
            if kept_sp >= 120:
                break
            q = str(r.get("question", "")).strip()
            resp = str(r.get("response", "")).strip()
            if not q or not resp or "\\boxed{" not in resp or "</think>" not in resp:
                continue
            if any(bad in (q + " " + resp).lower() for bad in ("http://", "https://", "uniprot", "wikipedia", "search snippet", "webpage")):
                continue
            if decon.is_math_contaminated(q):
                continue
            sk = _skeleton_sha(q)
            if sk in seen_ax:
                continue
            boxed_g = last_boxed(resp.split("</think>", 1)[-1])
            if not boxed_g or not _is_scorable_exact_math_gold(boxed_g):
                continue
            unwrapped = _unwrap_clean_long_cot(resp, boxed_g, q, min_chars=2200, max_chars=22000)
            if not unwrapped:
                continue
            seen_ax.add(sk)
            iid = f"axion_longcot_{r['id']}"
            ax_rows.append(
                {
                    "id": iid,
                    "problem_id": iid,
                    "source": f"axion_{ax_sp}_longcot",
                    "problem": q,
                    "gold": boxed_g,
                    "finish_reason": "stop",
                    "messages": _math_prompt_messages(q) + [{"role": "assistant", "content": unwrapped}],
                }
            )
            kept_sp += 1
    _write_jsonl(shared_ax_cache, ax_rows)

    dm_ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    s2_pool_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_math_stage2_frontier_pool.jsonl")
    s2_dm_map = {int(str(r["id"]).split(":")[1]): str(r["id"]) for r in s2_pool_rows if str(r["id"]).startswith("v3_m_s2:")}
    orig_rl_rows = _read_jsonl(V3_ROOT / "data/corpora_v3/v3_math_rl_pool640.jsonl")
    rl_dm_map = {int(str(r["id"]).split(":")[1]): str(r["id"]) for r in orig_rl_rows if str(r["id"]).startswith("v3_m_rl:")}

    dm_s2_out: list[dict[str, Any]] = []
    rl_out: list[dict[str, Any]] = []
    for dm_idx in sorted(set(s2_dm_map.keys()) | set(rl_dm_map.keys())):
        r = dm_ds[dm_idx]
        prob = str(r.get("question", "")).strip()
        ans = str(r.get("final_answer", "")).strip()
        if decon.is_math_contaminated(prob):
            continue
        best_long: str | None = None
        for skey in ("r1_solution_1", "r1_solution_2", "r1_solution_3"):
            cand = _unwrap_clean_long_cot(str(r.get(skey, "")), ans, prob, min_chars=1800, max_chars=22000)
            if cand is not None:
                best_long = cand
                break
        if best_long is None:
            continue
        if dm_idx in s2_dm_map:
            pid = s2_dm_map[dm_idx]
            dm_s2_out.append(
                {
                    "id": f"{pid}:longcot_rescue",
                    "problem_id": pid,
                    "source": "deepmath_0of4_longcot_rescue",
                    "problem": prob,
                    "gold": ans,
                    "finish_reason": "stop",
                    "messages": _math_prompt_messages(prob) + [{"role": "assistant", "content": best_long}],
                }
            )
        if dm_idx in rl_dm_map:
            rl_out.append({"id": rl_dm_map[dm_idx], "reference_solution": best_long})

    _write_jsonl(shared_dm_s2_cache, dm_s2_out)
    _write_jsonl(shared_rl_cache, rl_out)
    print(
        json.dumps(
            {
                "status": "SHARED_LONGCOT_CACHES_BUILT",
                "axion_longcot": len(ax_rows),
                "dm_s2_longcot": len(dm_s2_out),
                "rl_longcot": len(rl_out),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase V3 Track-M Horizon-Aligned Round-2 Driver")
    ap.add_argument("--model-key", default="qwen35_4b", choices=["qwen35_4b", "qwen35_9b", "qwen38_27b", "qwen38_35b_a3b", "qwen3_30b_a3b"])
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--build-shared-cache", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    verify_manifest_pin()
    decon = _build_full_eval_decontaminator()
    if args.build_shared_cache:
        build_shared_longcot_caches(decon)
        return 0

    model_key = args.model_key
    spec = MODEL_REGISTRY_V3[model_key]
    hp = R2_HYPERPARAMS[model_key]
    nproc = len([g for g in args.gpus.split(",") if g.strip()])

    axion_full = rebuild_full_horizon_axion_strengthened(decon)
    s1_data, s2_data, rl_pool, _ = build_horizon_aligned_corpora_for_model(model_key, decon, axion_full)
    if args.prepare_only:
        return 0

    p1_dir = V3_ROOT / "results_v3/plane1" / model_key
    p2_dir = V3_ROOT / "results_v3/plane2" / f"math_{model_key}"
    ckpt_root = V3_ROOT / "checkpoints_v3" / f"math_{model_key}"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    sp_path = V3_ROOT / "data/splits_v3/v3_math_search160.jsonl"
    proto = "v3_math_search160_k8"
    rules_search = V3_ROOT / "configs_v3/gate_rules_v3_math_search.json"
    ledger = p2_dir / f"trial_ledger_v3_math_{model_key}.jsonl"

    b0_json = p1_dir / f"math_b0_{model_key}_search.json"
    inc_json = b0_json
    if ("math", model_key) in PRIOR_CHAMPION_PATHS:
        champ_id, _ = PRIOR_CHAMPION_PATHS[("math", model_key)]
        cand_champ = p1_dir / f"math_{champ_id}_search.json"
        if cand_champ.exists():
            inc_json = cand_champ

    # ------------------------------------------------------------------
    # STEP 0: Register Round-1 Concise-SFT Ablation (`s1_math_*`, `s2_math_*`) if present
    # ------------------------------------------------------------------
    r1_cands: list[str] = []
    for r1_fn in (f"s1_math_{model_key}_breadth_sft_search.json", f"s2_math_{model_key}_zpd_sft_search.json"):
        rp = p2_dir / r1_fn
        if rp.exists():
            r1_cands.append(str(rp))
    r1_logged_marker = p2_dir / "r1_concise_ablation_logged.json"
    if r1_cands and not r1_logged_marker.exists():
        res_r1 = _run_checked(
            [
                sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
                "--candidates", *r1_cands,
                "--incumbent", str(inc_json),
                "--baseline", str(b0_json),
                "--rules", str(rules_search),
                "--ledger", str(ledger),
                "--round-id", f"round1_concise_postthink_sft_{model_key}",
            ],
            allow_rc=(0, 1, 3),
        )
        r1_logged_marker.write_text(res_r1.stdout, encoding="utf-8")
        # Free disk space from Round-1 degraded checkpoints while keeping all `.json` Search metrics intact
        for old_d in (f"s1_math_{model_key}_breadth_sft", f"s2_math_{model_key}_zpd_sft"):
            if (ckpt_root / old_d).exists():
                shutil.rmtree(ckpt_root / old_d, ignore_errors=True)

    # ------------------------------------------------------------------
    # STAGE I (Round 2): Horizon-Aligned Multi-Source Breadth SFT (`<= 10,000` target tokens)
    # ------------------------------------------------------------------
    s1_id = f"r2_s1_math_{model_key}_horizon_breadth_sft"
    s1_dir = ckpt_root / s1_id
    s1_factors = ckpt_root / f"{s1_id}_factors.safetensors"
    s1_sum = p2_dir / f"{s1_id}_train_summary.json"
    if not s1_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(SFT_PY),
                "--model-key", model_key,
                "--train-file", str(s1_data),
                "--output-dir", str(s1_dir),
                "--save-fp32-deltas", str(s1_factors),
                "--summary-output", str(s1_sum),
                "--max-seq-len", "10752",
                "--max-target-tokens", "10000",
                "--epochs", str(hp["s1_epochs"]),
                "--max-steps", str(hp["s1_max_steps"]),
                "--lr", str(hp["s1_lr"]),
                "--lr-schedule", "constant_with_warmup",
                "--global-batch-size", "32",
            ]
        )

    s1_scored = p2_dir / f"{s1_id}_search.json"
    eval_and_score_model(
        track="math",
        model_key=model_key,
        model_path=str(s1_dir),
        candidate_id=s1_id,
        split_prompts_file=sp_path,
        split_truth_file=sp_path,
        out_scored_json=s1_scored,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=proto,
        compare_baseline_json=b0_json,
    )

    # ------------------------------------------------------------------
    # STAGE II (Round 2): Model-Aware ZPD + Cross-Family `Axion` Strengthened SFT
    # ------------------------------------------------------------------
    verify_manifest_pin()
    s2_id = f"r2_s2_math_{model_key}_zpd_axion_sft"
    s2_dir = ckpt_root / s2_id
    s2_factors = ckpt_root / f"{s2_id}_factors.safetensors"
    s2_sum = p2_dir / f"{s2_id}_train_summary.json"
    if not s2_sum.exists():
        _run_checked(
            [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(SFT_PY),
                "--model-key", model_key,
                "--prior-fp32-deltas", str(s1_factors),
                "--train-file", str(s2_data),
                "--output-dir", str(s2_dir),
                "--save-fp32-deltas", str(s2_factors),
                "--summary-output", str(s2_sum),
                "--max-seq-len", "10752",
                "--max-target-tokens", "10000",
                "--epochs", str(hp["s2_epochs"]),
                "--max-steps", str(hp["s2_max_steps"]),
                "--lr", str(hp["s2_lr"]),
                "--lr-schedule", "cosine",
                "--warmup-ratio", "0.08",
                "--global-batch-size", "32",
            ]
        )

    s2_scored = p2_dir / f"{s2_id}_search.json"
    eval_and_score_model(
        track="math",
        model_key=model_key,
        model_path=str(s2_dir),
        candidate_id=s2_id,
        split_prompts_file=sp_path,
        split_truth_file=sp_path,
        out_scored_json=s2_scored,
        k=8,
        gpus=args.gpus,
        seed=20260325,
        protocol_id=proto,
        compare_baseline_json=b0_json,
    )

    # Governed SFT Anchor Selection (`select-search-winner` vs `B_0`): never seed RLVR from a regressed SFT stage!
    r_sft_sel = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
            "--candidates", str(s1_scored), str(s2_scored),
            "--incumbent", str(b0_json),
            "--baseline", str(b0_json),
            "--rules", str(rules_search),
            "--ledger", str(ledger),
            "--round-id", f"round2_sft_math_{model_key}",
        ],
        allow_rc=(0, 1, 3),
    )
    sft_sel = json.loads(r_sft_sel.stdout)
    sel_sft_id = sft_sel.get("selected_candidate_id")
    if sel_sft_id == s1_id:
        sft_anchor_ckpt = str(s1_dir)
        prior_deltas_arg = str(s1_factors)
    elif sel_sft_id == s2_id:
        sft_anchor_ckpt = str(s2_dir)
        prior_deltas_arg = f"{s1_factors},{s2_factors}"
    else:
        # Compare raw `math_equal_stratum_avg_at_8` if above `B_0`, otherwise seed RLVR directly from `B_0`!
        b0_m = float(json.loads(b0_json.read_text(encoding="utf-8"))["metrics"]["math_equal_stratum_avg_at_8"])
        s1_m = float(json.loads(s1_scored.read_text(encoding="utf-8"))["metrics"]["math_equal_stratum_avg_at_8"])
        s2_m = float(json.loads(s2_scored.read_text(encoding="utf-8"))["metrics"]["math_equal_stratum_avg_at_8"])
        if max(s1_m, s2_m) >= b0_m:
            if s2_m >= s1_m:
                sft_anchor_ckpt, prior_deltas_arg = str(s2_dir), f"{s1_factors},{s2_factors}"
            else:
                sft_anchor_ckpt, prior_deltas_arg = str(s1_dir), str(s1_factors)
        else:
            sft_anchor_ckpt, prior_deltas_arg = spec.local_path, ""

    print(f"[R2-SFT-Anchor] {model_key}: anchor={sft_anchor_ckpt} (prior_deltas={prior_deltas_arg!r})", flush=True)

    # ------------------------------------------------------------------
    # STAGE III (Round 2): 3-Proposal Multi-Epoch (`ppo_epochs=2`) RLVR Tournament
    # ------------------------------------------------------------------
    verify_manifest_pin()
    shared_rollouts = p2_dir / f"r2_shared_rlvr_rollouts_math_{model_key}.jsonl"
    generate_and_verify_shared_rlvr_rollouts(
        track="math",
        model_key=model_key,
        sft_star_path=sft_anchor_ckpt,
        rl_pool_file=rl_pool,
        rl_subset_n=int(hp["rl_subset_n"]),
        out_rollouts_jsonl=shared_rollouts,
        gpus=args.gpus,
        seed=20260412,
    )

    all_r2_scored: list[Path] = [s1_scored, s2_scored]
    cand_id_to_ckpt: dict[str, str] = {s1_id: str(s1_dir), s2_id: str(s2_dir)}

    for algo in ("dapo_asymclip", "drgrpo_cispo", "gspo_luffy"):
        rl_base = f"r2_s3_math_{model_key}_{algo}"
        rl_final_dir = ckpt_root / f"{rl_base}_pct100"
        rl_mile_root = ckpt_root / f"{rl_base}_milestones"
        rl_sum = p2_dir / f"{rl_base}_train_summary.json"
        if not rl_sum.exists():
            rl_cmd = [
                "torchrun", "--master_port", str(random.randint(21000, 28999)), f"--nproc_per_node={nproc}", str(RLVR_PY),
                "--algo", algo,
                "--model-key", model_key,
                "--rollouts-file", str(shared_rollouts),
                "--output-dir", str(rl_final_dir),
                "--milestones-root", str(rl_mile_root),
                "--save-milestone-fractions", "0.50",
                "--summary-output", str(rl_sum),
                "--max-seq-len", "11264",
                "--max-gen-tokens", "10240",
                "--overlong-window", "2048",
                "--ppo-epochs", "2",
                "--max-steps", str(hp["rl_max_steps"]),
                "--lr", str(hp["rl_lr"]),
                "--global-batch-size", "32",
            ]
            if prior_deltas_arg:
                rl_cmd.extend(["--prior-fp32-deltas", prior_deltas_arg])
            _run_checked(rl_cmd)

        for tag, mdir in [("pct50", rl_mile_root / "pct50"), ("pct100", rl_final_dir)]:
            if not mdir.exists():
                continue
            cid = f"{rl_base}_{tag}"
            c_scored = p2_dir / f"{cid}_search.json"
            eval_and_score_model(
                track="math",
                model_key=model_key,
                model_path=str(mdir),
                candidate_id=cid,
                split_prompts_file=sp_path,
                split_truth_file=sp_path,
                out_scored_json=c_scored,
                k=8,
                gpus=args.gpus,
                seed=20260325,
                protocol_id=proto,
                compare_baseline_json=b0_json,
            )
            all_r2_scored.append(c_scored)
            cand_id_to_ckpt[cid] = str(mdir)

    # Order-invariant `select-search-winner` across ALL Round-1 + Round-2 candidates!
    all_cands_args = r1_cands + [str(p) for p in all_r2_scored]
    r_win = _run_checked(
        [
            sys.executable, "-I", "-B", str(GUARD_PY), "select-search-winner",
            "--candidates", *all_cands_args,
            "--incumbent", str(inc_json),
            "--baseline", str(b0_json),
            "--rules", str(rules_search),
            "--ledger", str(ledger),
            "--round-id", f"round2_final_tournament_math_{model_key}",
        ],
        allow_rc=(0, 1, 3),
    )
    win_info = json.loads(r_win.stdout)
    winner_id = win_info.get("selected_candidate_id")
    if not winner_id:
        ranked_lb = win_info.get("leaderboard", [])
        top_id = str(ranked_lb[0]["candidate_id"]) if ranked_lb else f"r2_s3_math_{model_key}_drgrpo_cispo_pct100"
        win_info["selected_for_confirm_id"] = top_id
        win_info["selected_for_confirm_ckpt"] = cand_id_to_ckpt.get(top_id, str(s2_dir))
    else:
        win_info["selected_for_confirm_id"] = str(winner_id)
        win_info["selected_for_confirm_ckpt"] = cand_id_to_ckpt[str(winner_id)]

    (p2_dir / "search_tournament_winner.json").write_text(
        json.dumps(win_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    keep_ckpts = {win_info["selected_for_confirm_ckpt"], str(s1_dir), str(s2_dir)}
    for cid, cpath_s in cand_id_to_ckpt.items():
        if cpath_s not in keep_ckpts and Path(cpath_s).exists():
            shutil.rmtree(cpath_s, ignore_errors=True)

    print(f"[R2-Math Plane-2 DONE] {model_key}: winner={win_info['selected_for_confirm_id']}", flush=True)

    # Immediately run Confirm_V3 IUT + Atomic Single-Touch Holdout_V3 + Deduplicated Longitudinal Suite!
    _run_checked(
        [
            sys.executable, "-I", "-B", str(THIS_DIR / "run_v3_node_worker.py"),
            "--phase", "confirm-and-holdout",
            "--model-key", model_key,
            "--tracks", "math",
            "--gpus", args.gpus,
        ]
    )
    _run_checked(
        [
            sys.executable, "-I", "-B", str(THIS_DIR / "run_v3_node_worker.py"),
            "--phase", "longitudinal",
            "--model-key", model_key,
            "--gpus", args.gpus,
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

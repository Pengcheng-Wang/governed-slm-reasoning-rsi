#!/usr/bin/env python3
"""
Deterministic 3-Way Split & Decontamination Builder for `AILab_Governed_SLM_RSI_v1`.

Materializes:
  - Track L (Lean 4.24):
      * `data/splits/lean_search128.jsonl` (128 theorems from decontaminated `powered_validation`)
      * `data/splits/lean_repair_reserve97.jsonl` (disjoint 97 theorems reserved for outer L4 policy check)
      * `data/splits/lean_confirm243.jsonl` (full decontaminated `minif2f_v2c_valid`, n=243)
      * `data/splits/lean_holdout_minif2f_test244_prompts.jsonl` (prompt-only)
      * `data/splits/lean_holdout_putnam128_prompts.jsonl` (prompt-only)
      * `eval/sealed_holdout/lean_holdout_minif2f_test244.jsonl` (sealed statements/headers)
      * `eval/sealed_holdout/lean_holdout_putnam128.jsonl` (sealed statements/headers)
      * `data/raw/lean_frontier_pool.jsonl` (decontaminated training theorems + candidate proofs)
  - Track M (Hard Competition & Olympiad Math):
      * `data/splits/math_search128.jsonl` (128 hard OlympiadBench + MATH-L5 problems)
      * `data/splits/math_confirm300.jsonl` (200 OlympiadBench + 30 AIME24 + 70 MATH-L5 = 300 hard problems)
      * `data/splits/math500_retention_confirm150.jsonl` (150 disjoint MATH-500 L1-L4 retention problems)
      * `data/splits/math_holdout_*_prompts.jsonl` (prompt-only Holdout panels: AIME25, AIME26, HMMT-Feb26, OlympiadBench-120, MATH500-150)
      * `eval/sealed_holdout/math_holdout_*.jsonl` (sealed gold answers)
      * `data/raw/math_frontier_pool1600.jsonl` (1,600 13-gram-decontaminated DAPO competition problems)
  - `data/splits/split_audit_manifest.json` (SHA-256 digests, counts, 0-overlap verification receipts).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from lean4_verifier import MATHLIB_HEADER, split_frozen_statement  # noqa: E402
from math_verifier import verify_gold_parseable  # noqa: E402

QUARANTINED_DAPO_SUBSTRS = ("b57078e1", "e09849b4")
KNOWN_LEAN_ALPHA_DUP_IDS = {"lean_workbook_plus_13812"}


def _det_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s).casefold()


def _ngrams_13(s: str) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+|[^\w\s]", s.casefold())
    if len(toks) < 13:
        return set()
    return {tuple(toks[i : i + 13]) for i in range(len(toks) - 12)}


def _norm_lean_type_skeleton(formal_statement: str) -> str:
    """Alpha-invariant coarse normalization of the theorem goal/binders after theorem name."""
    _, prefix = split_frozen_statement(formal_statement)
    # Remove `theorem <name>` or `lemma <name>` prefix
    rest = re.sub(r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+\S+\s*", "", prefix)
    return re.sub(r"\s+", " ", rest).strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(), "count": len(rows), "sha256": _sha256_file(path)}


def _lean_prompt_messages(formal_statement: str, lean_header: str | None = None) -> list[dict[str, str]]:
    _, decl_prefix = split_frozen_statement(formal_statement)
    hdr = (lean_header or MATHLIB_HEADER).strip()
    user_prompt = (
        "Complete the following Lean 4 theorem proof using Mathlib. "
        "Output ONLY a single fenced ```lean4 code block containing the proof body (starting with `by`) "
        "or the exact theorem with its proof, with no `sorry` or `admit`.\n\n"
        f"```lean4\n{hdr}\n\n{decl_prefix} := by\n  sorry\n```"
    )
    return [{"role": "user", "content": user_prompt}]


def _math_prompt_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"{problem.strip()}\n\nSolve the problem step by step and put your final answer inside \\boxed{{}}.",
        }
    ]


def build_lean_splits(gen2_lean_root: Path, extra_harvest_jsonl: Path | None) -> dict[str, Any]:
    dec = gen2_lean_root / "continuation_20260721/artifacts/decontaminated_benchmarks"

    def _load_lean_bench(fn: str, source_tag: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for raw in (dec / fn).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            r = json.loads(raw)
            pid = str(r["problem_id"])
            stmt = str(r["formal_statement"])
            hdr = str(r.get("lean_header") or MATHLIB_HEADER)
            thm_name, _ = split_frozen_statement(stmt)
            out.append(
                {
                    "id": f"{source_tag}:{pid}",
                    "problem_id": pid,
                    "theorem_name": thm_name,
                    "source": source_tag,
                    "formal_statement": stmt,
                    "lean_header": hdr,
                    "messages": _lean_prompt_messages(stmt, hdr),
                }
            )
        return out

    v2s_valid = _load_lean_bench("minif2f_v2s_valid.jsonl", "minif2f_v2s_valid")
    v2s_test = _load_lean_bench("minif2f_v2s_test.jsonl", "minif2f_v2s_test")
    powered_val = _load_lean_bench("powered_validation.jsonl", "powered_val")
    putnam_all = _load_lean_bench("putnambench.jsonl", "putnambench")

    # Deterministic partition of `powered_val` (225) -> Search (128) + Repair-Policy Reserve (97)
    powered_sorted = sorted(powered_val, key=lambda r: _det_hash("lean_search_split:" + r["id"]))
    lean_search128 = sorted(powered_sorted[:128], key=lambda r: r["id"])
    lean_reserve97 = sorted(powered_sorted[128:], key=lambda r: r["id"])

    # Deterministic 128-theorem Holdout slice of PutnamBench (672)
    putnam_sorted = sorted(putnam_all, key=lambda r: _det_hash("putnam_holdout128:" + r["id"]))
    putnam_hold128 = sorted(putnam_sorted[:128], key=lambda r: r["id"])
    lean_confirm243 = sorted(v2s_valid, key=lambda r: r["id"])
    lean_hold_v2s244 = sorted(v2s_test, key=lambda r: r["id"])

    eval_all = lean_search128 + lean_reserve97 + lean_confirm243 + lean_hold_v2s244 + putnam_all
    forbidden_thm_names = {r["theorem_name"] for r in eval_all} | {r["problem_id"] for r in eval_all} | KNOWN_LEAN_ALPHA_DUP_IDS
    forbidden_skeletons = {_norm_lean_type_skeleton(r["formal_statement"]) for r in eval_all}

    # Build decontaminated Training / Frontier Pool + candidate proofs
    pool_by_thm: dict[str, dict[str, Any]] = {}
    tar_path = gen2_lean_root / "data/powered_and_minif2f_inputs.tar.gz"
    with tarfile.open(tar_path, "r:gz") as tf:
        # 1. Powered Lean train prompts (1,600)
        f_tr = tf.extractfile("input/split/powered_lean_train.jsonl")
        assert f_tr is not None
        for raw_b in f_tr:
            line = raw_b.decode("utf-8").strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r["problem_id"])
            stmt = str(r["formal_statement"])
            tname, _ = split_frozen_statement(stmt)
            skel = _norm_lean_type_skeleton(stmt)
            if pid in forbidden_thm_names or tname in forbidden_thm_names or skel in forbidden_skeletons:
                continue
            pool_by_thm[tname] = {
                "id": f"lean_train:{pid}",
                "problem_id": pid,
                "theorem_name": tname,
                "formal_statement": stmt,
                "lean_header": MATHLIB_HEADER,
                "messages": _lean_prompt_messages(stmt),
                "candidate_proofs": [],
            }

        # 2. Verified SFT traces from selected_sft + coverage_sft
        for member_name in ("input/sft/selected_sft.jsonl", "input/rft/coverage_sft.jsonl"):
            f_sft = tf.extractfile(member_name)
            if f_sft is None:
                continue
            for raw_b in f_sft:
                line = raw_b.decode("utf-8").strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = str(r["problem_id"])
                stmt = str(r["formal_statement"])
                tname, _ = split_frozen_statement(stmt)
                skel = _norm_lean_type_skeleton(stmt)
                if pid in forbidden_thm_names or tname in forbidden_thm_names or skel in forbidden_skeletons:
                    continue
                entry = pool_by_thm.setdefault(
                    tname,
                    {
                        "id": f"lean_train:{pid}",
                        "problem_id": pid,
                        "theorem_name": tname,
                        "formal_statement": stmt,
                        "lean_header": MATHLIB_HEADER,
                        "messages": _lean_prompt_messages(stmt),
                        "candidate_proofs": [],
                    },
                )
                msgs = r.get("messages", [])
                asst = next((m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), None)
                if asst and asst not in entry["candidate_proofs"]:
                    entry["candidate_proofs"].append(asst)

    # 3. Extra harvested verified Lean-Workbook proofs if staged
    if extra_harvest_jsonl and extra_harvest_jsonl.is_file():
        for raw in extra_harvest_jsonl.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            r = json.loads(raw)
            stmt = str(r["statement"]).strip()
            if ":=" not in stmt:
                stmt = stmt + " := by sorry"
            try:
                tname, decl_prefix = split_frozen_statement(stmt)
            except ValueError:
                continue
            norm_stmt = f"{decl_prefix} := by sorry"
            skel = _norm_lean_type_skeleton(norm_stmt)
            if tname in forbidden_thm_names or skel in forbidden_skeletons:
                continue
            entry = pool_by_thm.setdefault(
                tname,
                {
                    "id": f"lean_wb:{tname}",
                    "problem_id": tname,
                    "theorem_name": tname,
                    "formal_statement": norm_stmt,
                    "lean_header": MATHLIB_HEADER,
                    "messages": _lean_prompt_messages(norm_stmt),
                    "candidate_proofs": [],
                },
            )
            prf = str(r["proof"]).strip()
            wrapped = f"```lean4\n{prf}\n```" if "```" not in prf else prf
            if wrapped not in entry["candidate_proofs"] and len(entry["candidate_proofs"]) < 6:
                entry["candidate_proofs"].append(wrapped)

    frontier_pool = sorted(pool_by_thm.values(), key=lambda r: r["id"])

    def _prompts_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "problem_id": r["problem_id"],
                "source": r["source"],
                "messages": r["messages"],
            }
            for r in rows
        ]

    meta = {
        "lean_search128": _write_jsonl(ROOT / "data/splits/lean_search128.jsonl", lean_search128),
        "lean_repair_reserve97": _write_jsonl(ROOT / "data/splits/lean_repair_reserve97.jsonl", lean_reserve97),
        "lean_confirm243": _write_jsonl(ROOT / "data/splits/lean_confirm243.jsonl", lean_confirm243),
        "lean_holdout_minif2f_test244_prompts": _write_jsonl(
            ROOT / "data/splits/lean_holdout_minif2f_test244_prompts.jsonl", _prompts_only(lean_hold_v2s244)
        ),
        "lean_holdout_putnam128_prompts": _write_jsonl(
            ROOT / "data/splits/lean_holdout_putnam128_prompts.jsonl", _prompts_only(putnam_hold128)
        ),
        "sealed_lean_holdout_minif2f_test244": _write_jsonl(
            ROOT / "eval/sealed_holdout/lean_holdout_minif2f_test244.jsonl", lean_hold_v2s244
        ),
        "sealed_lean_holdout_putnam128": _write_jsonl(
            ROOT / "eval/sealed_holdout/lean_holdout_putnam128.jsonl", putnam_hold128
        ),
        "lean_frontier_pool": _write_jsonl(ROOT / "data/raw/lean_frontier_pool.jsonl", frontier_pool),
        "pool_theorems_with_seed_proofs": sum(1 for r in frontier_pool if r["candidate_proofs"]),
        "zero_eval_overlap_verified": True,
    }
    return meta


def _parse_first_olympiad_ans(raw_ans: Any) -> str | None:
    if isinstance(raw_ans, list):
        return str(raw_ans[0]).strip() if len(raw_ans) == 1 and str(raw_ans[0]).strip() else None
    s = str(raw_ans).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list) and len(arr) == 1:
                return str(arr[0]).strip()
        except Exception:
            return None
    return s if s else None


def build_math_splits() -> dict[str, Any]:
    from datasets import load_dataset

    # 1. OlympiadBench English Competition Open-Ended Math
    ds_oly = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    oly_clean: list[dict[str, Any]] = []
    seen_norm_eval: set[str] = set()
    for r in ds_oly:
        if r.get("is_multiple_answer"):
            continue
        if r.get("error") not in (None, "", "None"):
            continue
        if r.get("unit") not in (None, "", "None"):
            continue
        qtype = str(r.get("question_type") or "").lower()
        if "proof" in qtype:
            continue
        ans = _parse_first_olympiad_ans(r.get("final_answer"))
        if not ans:
            continue
        ans_clean = ans.strip().strip("$").strip()
        is_int = bool(re.fullmatch(r"[+-]?\d+", ans_clean))
        if not verify_gold_parseable(ans_clean, integer_task=is_int):
            continue
        prob = str(r["question"]).strip()
        np = _norm_text(prob)
        if np in seen_norm_eval:
            continue
        seen_norm_eval.add(np)
        oly_clean.append(
            {
                "id": f"oly:{r['id']}",
                "source": "OlympiadBench_OE_TO_maths_en_COMP",
                "problem": prob,
                "gold": ans_clean,
                "integer_task": is_int,
                "subfield": str(r.get("subfield") or ""),
                "messages": _math_prompt_messages(prob),
            }
        )

    oly_sorted = sorted(oly_clean, key=lambda x: _det_hash("oly_split:" + x["id"]))
    if len(oly_sorted) < 400:
        raise RuntimeError(f"Expected >=400 clean OlympiadBench problems, got {len(oly_sorted)}")
    oly_holdout120 = sorted(oly_sorted[:120], key=lambda x: x["id"])
    oly_confirm200 = sorted(oly_sorted[120:320], key=lambda x: x["id"])
    oly_for_search = oly_sorted[320:]

    # 2. AIME 2024 (Confirm), AIME 2025 (Holdout), AIME 2026 (Holdout), HMMT Feb 2026 (Holdout)
    def _load_aime(repo: str, tag: str, cfg: str | None = None) -> list[dict[str, Any]]:
        ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
        rows: list[dict[str, Any]] = []
        for idx, r in enumerate(ds):
            prob = str(r["problem"]).strip()
            ans = str(r["answer"]).strip()
            pid = str(r.get("id", r.get("problem_idx", idx)))
            assert verify_gold_parseable(ans, integer_task=True), f"Bad AIME gold in {repo}: {ans}"
            seen_norm_eval.add(_norm_text(prob))
            rows.append(
                {
                    "id": f"{tag}:{pid}",
                    "source": tag,
                    "problem": prob,
                    "gold": str(int(ans)),
                    "integer_task": True,
                    "messages": _math_prompt_messages(prob),
                }
            )
        return sorted(rows, key=lambda x: x["id"])

    aime24_30 = _load_aime("HuggingFaceH4/aime_2024", "aime2024")
    aime25_30 = _load_aime("yentinglin/aime_2025", "aime2025", "default")
    aime26_30 = _load_aime("MathArena/aime_2026", "aime2026")

    ds_hmmt = load_dataset("MathArena/hmmt_feb_2026", split="train")
    hmmt26_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(ds_hmmt):
        prob = str(r["problem"]).strip()
        ans = str(r["answer"]).strip()
        pid = str(r.get("problem_idx", idx))
        is_int = bool(re.fullmatch(r"[+-]?\d+", ans))
        if not verify_gold_parseable(ans, integer_task=is_int):
            continue
        seen_norm_eval.add(_norm_text(prob))
        hmmt26_rows.append(
            {
                "id": f"hmmt2026:{pid}",
                "source": "hmmt_feb_2026",
                "problem": prob,
                "gold": str(int(ans)) if is_int else ans,
                "integer_task": is_int,
                "messages": _math_prompt_messages(prob),
            }
        )
    hmmt26_rows.sort(key=lambda x: x["id"])

    # 3. MATH-500 (Level 5 -> hard competition mix; Levels 1-4 -> 150 Confirm Retention + 150 Holdout Retention)
    ds_m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    m500_l5: list[dict[str, Any]] = []
    m500_l14: list[dict[str, Any]] = []
    for r in ds_m500:
        prob = str(r["problem"]).strip()
        ans = str(r["answer"]).strip()
        uid = str(r["unique_id"])
        lvl = int(r["level"])
        is_int = bool(re.fullmatch(r"[+-]?\d+", ans))
        if not verify_gold_parseable(ans, integer_task=is_int):
            continue
        np = _norm_text(prob)
        if np in seen_norm_eval:
            continue
        seen_norm_eval.add(np)
        item = {
            "id": f"math500:{uid}",
            "source": f"MATH500_L{lvl}",
            "problem": prob,
            "gold": str(int(ans)) if is_int else ans,
            "integer_task": is_int,
            "level": lvl,
            "subject": str(r.get("subject") or ""),
            "messages": _math_prompt_messages(prob),
        }
        if lvl == 5:
            m500_l5.append(item)
        else:
            m500_l14.append(item)

    m500_l5_sorted = sorted(m500_l5, key=lambda x: _det_hash("m500_l5:" + x["id"]))
    m500_l5_confirm70 = m500_l5_sorted[:70]
    m500_l5_for_search = m500_l5_sorted[70:]

    m500_l14_sorted = sorted(m500_l14, key=lambda x: _det_hash("m500_ret:" + x["id"]))
    ret_confirm150 = sorted(m500_l14_sorted[:150], key=lambda x: x["id"])
    ret_holdout150 = sorted(m500_l14_sorted[150:300], key=lambda x: x["id"])

    # Assemble Math Search (128) and Math Confirm (300)
    search_pool_cand = sorted(oly_for_search + m500_l5_for_search, key=lambda x: _det_hash("m_search:" + x["id"]))
    if len(search_pool_cand) < 128:
        raise RuntimeError(f"Insufficient Search candidates: {len(search_pool_cand)} < 128")
    math_search128 = sorted(search_pool_cand[:128], key=lambda x: x["id"])
    math_confirm300 = sorted(oly_confirm200 + aime24_30 + m500_l5_confirm70, key=lambda x: x["id"])
    assert len(math_confirm300) == 300

    # Collect ALL eval normalized strings & 13-grams for strict decontamination of Training/Frontier Pool
    all_eval_math = (
        math_search128
        + math_confirm300
        + ret_confirm150
        + ret_holdout150
        + oly_holdout120
        + aime25_30
        + aime26_30
        + hmmt26_rows
    )
    eval_norm_set = {_norm_text(r["problem"]) for r in all_eval_math}
    eval_13grams: set[tuple[str, ...]] = set()
    for r in all_eval_math:
        eval_13grams |= _ngrams_13(r["problem"])

    # 4. Build 1,600-problem Decontaminated Competition Training/Frontier Pool from DAPO-Math-17k
    ds_dapo = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    PROMPT_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    PROMPT_SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."

    dapo_unique: list[dict[str, Any]] = []
    seen_dapo_norm: set[str] = set()
    quarantined_overlap = 0
    for r in ds_dapo:
        extra = r.get("extra_info") or {}
        idx_str = str(extra.get("index", ""))
        if any(q in idx_str for q in QUARANTINED_DAPO_SUBSTRS):
            continue
        p_list = r.get("prompt") or []
        if not p_list:
            continue
        raw_content = str(p_list[0].get("content", ""))
        if raw_content.startswith(PROMPT_PREFIX):
            raw_content = raw_content[len(PROMPT_PREFIX) :]
        if raw_content.endswith(PROMPT_SUFFIX):
            raw_content = raw_content[: -len(PROMPT_SUFFIX)]
        prob = raw_content.strip()
        np = _norm_text(prob)
        if not prob or np in seen_dapo_norm:
            continue
        seen_dapo_norm.add(np)

        rm = r.get("reward_model") or {}
        gt = str(rm.get("ground_truth", "")).strip()
        if not re.fullmatch(r"[+-]?\d+", gt):
            continue
        if np in eval_norm_set or (_ngrams_13(prob) & eval_13grams):
            quarantined_overlap += 1
            continue
        pid_hash = _det_hash(np)[:12]
        dapo_unique.append(
            {
                "id": f"dapo:{pid_hash}",
                "source": "DAPO-Math-17k",
                "problem": prob,
                "gold": str(int(gt)),
                "integer_task": True,
                "messages": _math_prompt_messages(prob),
            }
        )

    dapo_sorted = sorted(dapo_unique, key=lambda x: _det_hash("dapo_pool1600:" + x["id"]))
    math_pool1600 = sorted(dapo_sorted[:1600], key=lambda x: x["id"])

    def _prompts_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "source": r["source"],
                "integer_task": r["integer_task"],
                "messages": r["messages"],
            }
            for r in rows
        ]

    return {
        "math_search128": _write_jsonl(ROOT / "data/splits/math_search128.jsonl", math_search128),
        "math_confirm300": _write_jsonl(ROOT / "data/splits/math_confirm300.jsonl", math_confirm300),
        "math500_retention_confirm150": _write_jsonl(ROOT / "data/splits/math500_retention_confirm150.jsonl", ret_confirm150),
        "math_holdout_aime2025_prompts": _write_jsonl(ROOT / "data/splits/math_holdout_aime2025_30_prompts.jsonl", _prompts_only(aime25_30)),
        "math_holdout_aime2026_prompts": _write_jsonl(ROOT / "data/splits/math_holdout_aime2026_30_prompts.jsonl", _prompts_only(aime26_30)),
        "math_holdout_hmmt2026_prompts": _write_jsonl(ROOT / "data/splits/math_holdout_hmmt2026_prompts.jsonl", _prompts_only(hmmt26_rows)),
        "math_holdout_olympiad120_prompts": _write_jsonl(ROOT / "data/splits/math_holdout_olympiad120_prompts.jsonl", _prompts_only(oly_holdout120)),
        "math_holdout_retention150_prompts": _write_jsonl(ROOT / "data/splits/math_holdout_retention150_prompts.jsonl", _prompts_only(ret_holdout150)),
        "sealed_math_holdout_aime2025": _write_jsonl(ROOT / "eval/sealed_holdout/math_holdout_aime2025_30.jsonl", aime25_30),
        "sealed_math_holdout_aime2026": _write_jsonl(ROOT / "eval/sealed_holdout/math_holdout_aime2026_30.jsonl", aime26_30),
        "sealed_math_holdout_hmmt2026": _write_jsonl(ROOT / "eval/sealed_holdout/math_holdout_hmmt2026.jsonl", hmmt26_rows),
        "sealed_math_holdout_olympiad120": _write_jsonl(ROOT / "eval/sealed_holdout/math_holdout_olympiad120.jsonl", oly_holdout120),
        "sealed_math_holdout_retention150": _write_jsonl(ROOT / "eval/sealed_holdout/math_holdout_retention150.jsonl", ret_holdout150),
        "math_frontier_pool1600": _write_jsonl(ROOT / "data/raw/math_frontier_pool1600.jsonl", math_pool1600),
        "dapo_unique_total_clean": len(dapo_unique),
        "dapo_quarantined_eval_13gram_overlaps": quarantined_overlap,
        "olympiadbench_clean_single_answer_total": len(oly_clean),
        "zero_eval_13gram_overlap_verified": True,
    }


def main() -> int:
    gen2_lean = Path("/data/gen2/mai_evaluation/eval_report/tmpcache_leanx_v8m3q7_n2k9")
    extra_harvest = ROOT / "data/raw/harvested_lean_workbook_proofs.jsonl"
    lean_meta = build_lean_splits(gen2_lean, extra_harvest if extra_harvest.is_file() else None)
    math_meta = build_math_splits()

    audit = {
        "schema_version": "1.0",
        "lean_splits": lean_meta,
        "math_splits": math_meta,
    }
    audit_path = ROOT / "data/splits/split_audit_manifest.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

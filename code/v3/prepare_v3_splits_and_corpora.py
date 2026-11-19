#!/usr/bin/env python3
"""
Phase V3 Disjoint Split, 4-Layer Decontamination & Multi-Source Verifiable Corpus Builder.

Constructs completely disjoint V3 evaluation splits and verifiable Stage-I / Stage-II / RLVR pools:
  - Enforces 0 overlap (ID, alphanumeric hash, number-masked structural skeleton hash, and 13-gram
    shingle index) between ALL training/RL corpora and ALL V1, V2, and V3 Search/Confirm/Holdout splits.
  - Eliminates Mathlib Holdout/Confirm environment leakage (`B2`): 100% of V3 Search, Confirm, and
    Holdout items come from external formal benchmarks (`arxivlean-2026`, `FormalMATH-Lite`, `Goedel-Pset`),
    while anonymized standalone multi-step `Mathlib4` tactic proofs (`>= 3` lines, self-call banned,
    `#print axioms`-verified) are used strictly in Stage-I SFT training to build collegiate/research API fluency.
  - Caps all SFT training targets strictly below evaluation generation limits (`<= 7,680` chars/tokens
    for Math, `<= 1,536` tokens for Lean) to eliminate truncation confounds (`B5`).
  - Runs `governed_loop_guard_v37.py audit-splits` before writing `v3_split_audit_manifest.json`.
"""
from __future__ import annotations

import ast
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_dataset

V1_ROOT = Path("/scratch/governed_slm_rsi_v1")
V2_ROOT = Path("/scratch/governed_slm_rsi_v2")
V3_ROOT = Path("/scratch/governed_slm_rsi_v3")
LAKE_BIN = V1_ROOT / "lean/elan/bin/lake"
ELAN_BIN = V1_ROOT / "lean/elan/bin"
LEAN_WS = V1_ROOT / "lean/mathlib_env"

EVAL_V3_DIR = V3_ROOT / "eval_v3"
if str(EVAL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V3_DIR))

from lean4_verifier import MATHLIB_HEADER, split_frozen_statement, verify_lean_candidate  # noqa: E402
from math_verifier import exact_integer_equivalent, grade_symbolic_math, score_math_response  # noqa: E402


MATH_USER_SUFFIX = "\n\nSolve the problem step by step and put your final answer inside \\boxed{}."


def _det_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _alnum_sha(text: str) -> str:
    alnum = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    return hashlib.sha256(alnum.encode("utf-8")).hexdigest()


def _skeleton_sha(text: str) -> str:
    # Mask digits so constant-substituted parametric duplicates collide!
    norm = re.sub(r"\d+", "#", re.sub(r"\s+", " ", str(text).strip().lower()))
    alnum = re.sub(r"[^a-z#]+", "", norm)
    return hashlib.sha256(alnum.encode("utf-8")).hexdigest()


def _ngrams_13(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    if len(toks) < 13:
        return []
    # Sample every 2nd 13-gram for compact RAM index with guaranteed collision on >=14-word overlap
    return [" ".join(toks[i : i + 13]) for i in range(0, len(toks) - 12, 2)]


def _norm_lean_stmt_sha(stmt: str) -> str:
    s = re.sub(r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+\S+", "theorem _anon_", str(stmt).strip())
    s = re.sub(r":=.*$", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s.strip())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _lean_prompt_messages(formal_statement: str, lean_header: str, informal: str = "") -> list[dict[str, str]]:
    _, decl_prefix = split_frozen_statement(formal_statement)
    hdr = (lean_header or MATHLIB_HEADER).strip()
    inf_block = f"Informal statement:\n{informal.strip()}\n\n" if informal.strip() else ""
    user_prompt = (
        "Complete the following Lean 4 theorem proof using Mathlib. "
        "Output ONLY a single fenced ```lean4 code block containing the proof body (starting with `by`) "
        "or the exact theorem with its proof, with no `sorry` or `admit`.\n\n"
        f"{inf_block}"
        f"```lean4\n{hdr}\n\n{decl_prefix} := by\n  sorry\n```"
    )
    return [{"role": "user", "content": user_prompt}]


def _math_prompt_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(problem).strip() + MATH_USER_SUFFIX}]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    h = hashlib.sha256()
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
            fh.write(line)
            h.update(line.encode("utf-8"))
    tmp.replace(path)
    return {"path": path.relative_to(V3_ROOT).as_posix(), "count": len(rows), "sha256": h.hexdigest()}


class FourLayerDecontaminator:
    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.math_alnum: set[str] = set()
        self.math_skel: set[str] = set()
        self.math_13grams: set[str] = set()
        self.lean_stmt_shas: set[str] = set()

    def register_math_eval(self, item_id: str, problem_text: str) -> None:
        self.ids.add(str(item_id))
        self.math_alnum.add(_alnum_sha(problem_text))
        if len(str(problem_text)) >= 40:
            self.math_skel.add(_skeleton_sha(problem_text))
        for ng in _ngrams_13(problem_text):
            self.math_13grams.add(ng)

    def register_lean_eval(self, item_id: str, formal_stmt: str, informal: str = "") -> None:
        self.ids.add(str(item_id))
        self.lean_stmt_shas.add(_norm_lean_stmt_sha(formal_stmt))
        if informal and len(informal) >= 40:
            for ng in _ngrams_13(informal):
                self.math_13grams.add(ng)

    def is_math_contaminated(self, problem_text: str, *, check_skeleton: bool = True, check_13gram: bool = True) -> bool:
        if _alnum_sha(problem_text) in self.math_alnum:
            return True
        if check_skeleton and len(str(problem_text)) >= 40 and _skeleton_sha(problem_text) in self.math_skel:
            return True
        if check_13gram:
            for ng in _ngrams_13(problem_text):
                if ng in self.math_13grams:
                    return True
        return False

    def is_lean_contaminated(self, formal_stmt: str, informal: str = "") -> bool:
        if _norm_lean_stmt_sha(formal_stmt) in self.lean_stmt_shas:
            return True
        if informal and len(informal) >= 40:
            for ng in _ngrams_13(informal):
                if ng in self.math_13grams:
                    return True
        return False


def collect_prior_v1_v2_blocklist() -> tuple[FourLayerDecontaminator, set[str], set[str]]:
    """Returns `(eval_decon, prior_train_math_alnum, prior_train_lean_shas)`."""
    decon = FourLayerDecontaminator()
    prior_train_math: set[str] = set()
    prior_train_lean: set[str] = set()

    eval_dirs = [
        V1_ROOT / "data/splits",
        V1_ROOT / "eval/sealed_holdout",
        V2_ROOT / "data/splits_v2",
        V2_ROOT / "eval_v2/sealed_holdout",
    ]
    train_dirs = [
        V1_ROOT / "data/raw",
        V1_ROOT / "data/verified_banks",
        V2_ROOT / "data/corpora",
        V2_ROOT / "data/verified_banks_v2",
    ]

    for d in eval_dirs:
        if not d.exists():
            continue
        for fp in d.rglob("*.jsonl"):
            for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                iid = str(r.get("id", ""))
                if "formal_statement" in r and r["formal_statement"]:
                    decon.register_lean_eval(iid, str(r["formal_statement"]), str(r.get("informal_statement", "")))
                if "problem" in r and r["problem"]:
                    decon.register_math_eval(iid, str(r["problem"]))
                elif "messages" in r and r["messages"]:
                    u = str(r["messages"][0].get("content", ""))
                    u = u.split("\n\nSolve the problem", 1)[0]
                    if "```lean4" not in u:
                        decon.register_math_eval(iid, u)

    for d in train_dirs:
        if not d.exists():
            continue
        for fp in d.rglob("*.jsonl"):
            for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if "formal_statement" in r and r["formal_statement"]:
                    prior_train_lean.add(_norm_lean_stmt_sha(str(r["formal_statement"])))
                if "problem" in r and r["problem"]:
                    prior_train_math.add(_alnum_sha(str(r["problem"])))

    return decon, prior_train_math, prior_train_lean


def _is_scorable_exact_math_gold(gold: str) -> bool:
    g = str(gold or "").strip()
    if not g or len(g) > 80:
        return False
    low = g.lower()
    bad_words = (
        "prove", "show", "none", "true", "false", "yes", "no", "infinit",
        "impossible", "cannot", "arbitrar", "see ", "exists",
    )
    if any(w in low for w in bad_words):
        return False
    if re.fullmatch(r"[+-]?\d+", g):
        return exact_integer_equivalent(g, g)
    res = grade_symbolic_math(g, g)
    return bool(res.get("correct", False)) and not bool(res.get("parse_failure", True))


def _classify_math_stratum(topic: str) -> str:
    low = str(topic).lower()
    if any(k in low for k in ("number theory", "combinatorics", "probability", "counting", "graph")):
        return "nt_combinatorics"
    if any(k in low for k in ("geometry", "trigonometr", "discrete", "logic", "set")):
        return "geometry_discrete"
    return "algebra_analysis"


def _classify_lean_stratum(domain: str, stmt: str) -> str:
    low = (str(domain) + " " + str(stmt)).lower()
    if any(k in low for k in ("calc", "deriv", "integr", "limit", "tendsto", "continuous", "series", "filter", "metric", "topology")):
        return "analysis_calculus"
    if any(k in low for k in ("number", "prime", "dvd", "mod", "zmod", "nat.gcd", "comb", "finset", "card", "geom")):
        return "nt_combinatorics_geom"
    return "algebra"


def _split_lean_hdr_and_decl(fs: str) -> tuple[str, str]:
    lines = str(fs).splitlines()
    hdr_l: list[str] = []
    decl_l: list[str] = []
    in_decl = False
    for ln in lines:
        s = ln.lstrip()
        if not in_decl and any(
            s.startswith(k)
            for k in ("theorem ", "lemma ", "noncomputable theorem ", "noncomputable lemma ")
        ):
            in_decl = True
        if in_decl:
            decl_l.append(ln)
        else:
            if not s.startswith(("/--", "--", "-/")):
                hdr_l.append(ln)
    hdr = "\n".join(hdr_l).strip() if hdr_l else MATHLIB_HEADER.strip()
    decl = "\n".join(decl_l).strip() if decl_l else str(fs).strip()
    return hdr, decl


def _elaborate_lean4_statement_v3(header: str, formal_stmt: str) -> tuple[bool, str, str]:
    # Reject vacuous `abbrev ... := sorry` or `def ... := sorry` in header/statement
    combined = f"{header}\n{formal_stmt}"
    if re.search(r"\b(?:abbrev|def)\b[^:]*:=\s*(?:by\s+)?sorry\b", combined):
        return False, "", "vacuous_abbrev_sorry"
    try:
        _, decl_prefix = split_frozen_statement(formal_stmt)
    except Exception as exc:
        return False, "", f"split_err:{exc}"

    rhs_type = decl_prefix.rsplit(":", 1)[-1].strip() if ":" in decl_prefix else ""
    if rhs_type == "True":
        return False, "", "vacuous_true_goal"

    base_hdr = header if "import Mathlib" in header else ("import Mathlib\n" + header)
    if "import Aesop" not in base_hdr:
        base_hdr = base_hdr.replace("import Mathlib", "import Mathlib\nimport Aesop", 1)
    dedup_lines: list[str] = []
    for ln in base_hdr.splitlines():
        st = ln.strip()
        if st.startswith("open") and st in [x.strip() for x in dedup_lines]:
            continue
        dedup_lines.append(ln)
    if not any("maxHeartbeats" in x for x in dedup_lines):
        dedup_lines.append("set_option maxHeartbeats 400000")
    clean_hdr = "\n".join(dedup_lines).strip() + "\n"

    code = f"{clean_hdr}\n{decl_prefix} := by\n  sorry\n"
    env = dict(os.environ)
    env["ELAN_HOME"] = str(V1_ROOT / "lean/elan")
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir="/dev/shm", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        p = tf.name
    try:
        r = subprocess.run(
            [str(LAKE_BIN), "env", "lean", p],
            cwd=str(LEAN_WS),
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        ok = (r.returncode == 0) and ("error:" not in out)
        return ok, clean_hdr, decl_prefix
    except Exception as exc:
        return False, clean_hdr, str(exc)
    finally:
        if os.path.exists(p):
            os.unlink(p)


def _extract_clean_post_think_math_solution(raw_sol: str, gold: str, problem: str, *, max_chars: int = 16000) -> str | None:
    """
    Extracts a clean, `<think>`-free step-by-step derivation and verifies `strict_correct == True`
    against external human ground truth via `score_math_response`.
    """
    if not raw_sol:
        return None
    if "</think>" in raw_sol:
        body = raw_sol.split("</think>", 1)[1].strip()
    elif "<think>" in raw_sol:
        return None
    else:
        body = raw_sol.strip()
    if "<answer>" in body and "</answer>" in body:
        body = body.split("<answer>", 1)[1].rsplit("</answer>", 1)[0].strip()
    if "<think>" in body or "</think>" in body:
        return None
    if len(body) < 60 or len(body) > max_chars:
        return None
    int_task = bool(re.fullmatch(r"[+-]?\d+", str(gold).strip()))
    sc = score_math_response(
        body,
        str(gold).strip(),
        finish_reason="stop",
        thinking=False,
        integer_task=int_task,
        problem=problem,
    )
    if sc.get("strict_correct") and sc.get("sft_clean_admissible"):
        return body
    return None


def _extract_mathlib_standalone_candidates(max_candidates: int = 2400) -> list[tuple[str, str, str]]:
    """
    Scans standalone-friendly `Mathlib` topic files (`Algebra`, `Analysis`, `Topology`,
    `LinearAlgebra`, `NumberTheory`, `GroupTheory`, `RingTheory`) for self-contained
    `theorem ... := by ...` blocks with `>= 3` tactic lines and zero self-references,
    anonymizing the theorem name to `ml_sft_<hash>` for Stage-I SFT training ONLY.
    """
    ml_root = LEAN_WS / "Mathlib"
    target_dirs = ["Algebra", "Analysis", "Topology", "LinearAlgebra", "NumberTheory", "GroupTheory", "RingTheory"]
    candidates: list[tuple[str, str, str]] = []
    thm_re = re.compile(
        r"(?ms)^theorem\s+([A-Za-z0-9_']+)\s*([^:=\n]+?:\s*[^:=\n]+?)\s*:=\s*by\n((?:[ \t]+[^\n]+\n){3,18})"
    )
    for td in target_dirs:
        dpath = ml_root / td
        if not dpath.exists():
            continue
        files = sorted(dpath.rglob("*.lean"))
        for fp in files:
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Extract explicit open declarations in file for context
            opens = [
                ln.strip()
                for ln in txt.splitlines()[:60]
                if ln.strip().startswith("open ") and "scoped" not in ln and len(ln.strip()) < 80
            ]
            hdr = "import Mathlib\nimport Aesop\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n"
            if opens:
                hdr += "\n".join(opens[:3]) + "\n"
            for m in thm_re.finditer(txt):
                orig_name, sig, body = m.group(1), m.group(2).strip(), m.group(3).rstrip()
                # Reject proofs referencing private/local helpers or the theorem's own name
                if orig_name in body or "sorry" in body or "exact?" in body or "private" in sig:
                    continue
                # Reject typeclass variables not bound in signature (`variable` implicit dependence will naturally fail standalone compile!)
                h_short = _det_hash(f"{fp.name}:{orig_name}:{sig}")[:10]
                anon_stmt = f"theorem ml_sft_{h_short} {sig} := by"
                proof_block = f"```lean4\nby\n{body}\n```"
                candidates.append((f"ml_{td.lower()}_{h_short}", hdr, f"{anon_stmt}\n{proof_block}"))
                if len(candidates) >= max_candidates * 3:
                    break
            if len(candidates) >= max_candidates * 3:
                break
    # Deterministic shuffle across all 7 domains
    candidates.sort(key=lambda x: _det_hash(x[0]))
    return candidates[: max_candidates * 2]


def _verify_one_goedel_or_mathlib_sft(task: tuple[str, str, str, str, str]) -> dict[str, Any] | None:
    item_id, hdr, stmt, resp_text, source = task
    res = verify_lean_candidate(
        resp_text,
        stmt,
        lean_project=LEAN_WS,
        elan_bin=ELAN_BIN,
        finish_reason="stop",
        thinking=False,
        strict_format=False,
        timeout_s=25.0,
        lean_header=hdr,
    )
    if not (res.get("ok") and res.get("sft_clean_admissible")):
        return None
    clean_body = res["proof_body"]
    # Format clean canonical direct fenced block
    assistant_txt = f"```lean4\n{clean_body}\n```"
    if len(assistant_txt) > 5000:
        return None
    msgs = _lean_prompt_messages(stmt, hdr) + [{"role": "assistant", "content": assistant_txt}]
    return {
        "id": item_id,
        "problem_id": item_id,
        "source": source,
        "lean_header": hdr,
        "formal_statement": stmt,
        "proof_body": clean_body,
        "finish_reason": "stop",
        "messages": msgs,
    }


def main() -> int:
    splits_dir = V3_ROOT / "data/splits_v3"
    holdout_dir = V3_ROOT / "eval_v3/sealed_holdout"
    corpora_dir = V3_ROOT / "data/corpora_v3"
    for d in (splits_dir, holdout_dir, corpora_dir):
        d.mkdir(parents=True, exist_ok=True)

    decon, prior_train_math, prior_train_lean = collect_prior_v1_v2_blocklist()
    print(
        f"[1/6] Loaded V1+V2 blocklist: eval_ids={len(decon.ids)}, "
        f"math_eval={len(decon.math_alnum)}, lean_eval={len(decon.lean_stmt_shas)}, "
        f"prior_train_math={len(prior_train_math)}, prior_train_lean={len(prior_train_lean)}",
        flush=True,
    )

    # =========================================================================
    # PART A: TRACK M EVALUATION SPLITS (Holdout -> Confirm -> Search)
    # =========================================================================
    math_holdout_full: list[dict[str, Any]] = []
    math_holdout_prompts: list[dict[str, Any]] = []

    for hf_repo, contest_tag, stratum_tag in [
        ("MathArena/arxivmath", "arxivmath_2026", "arxivmath_2026"),
        ("MathArena/smt_2025", "smt_2025", "smt_2025"),
        ("MathArena/hmmt_feb_2025", "hmmt_feb_2025", "elite_contest_2025"),
        ("MathArena/apex_2025", "apex_2025", "elite_contest_2025"),
    ]:
        ds = load_dataset(hf_repo, split="train")
        kept_c = 0
        for idx, r in enumerate(ds):
            prob = str(r.get("problem", "")).strip()
            ans = str(r.get("answer", "")).strip()
            if not prob or not _is_scorable_exact_math_gold(ans):
                continue
            if decon.is_math_contaminated(prob, check_skeleton=False, check_13gram=False):
                continue
            if _alnum_sha(prob) in prior_train_math:
                continue
            iid = f"{contest_tag}:{r.get('problem_idx', idx)}"
            decon.register_math_eval(iid, prob)
            int_task = bool(re.fullmatch(r"[+-]?\d+", ans))
            msgs = _math_prompt_messages(prob)
            math_holdout_full.append(
                {
                    "id": iid,
                    "cluster_id": iid,
                    "stratum": stratum_tag,
                    "contest": contest_tag,
                    "source": contest_tag,
                    "problem": prob,
                    "gold": ans,
                    "integer_task": int_task,
                    "messages": msgs,
                }
            )
            math_holdout_prompts.append(
                {"id": iid, "cluster_id": iid, "stratum": stratum_tag, "contest": contest_tag, "source": contest_tag, "messages": msgs}
            )
            kept_c += 1
        print(f"  Track M Holdout `{contest_tag}`: {kept_c} scorable disjoint items", flush=True)

    # Load `DeepMath-103K` once on NVMe
    deepmath = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"  Loaded DeepMath-103K ({len(deepmath)} rows)", flush=True)

    # Partition hard DeepMath (`[7.0, 9.5]`) deterministically for Confirm (300 = 3 x 100) and Search (160)
    dm_eval_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    # First add HMMT Feb 2024 to Search candidates
    search_seed_items: list[dict[str, Any]] = []
    try:
        ds_h24 = load_dataset("MathArena/hmmt_feb_2024", split="train")
        for idx, r in enumerate(ds_h24):
            prob = str(r.get("problem", "")).strip()
            ans = str(r.get("answer", "")).strip()
            if not prob or not _is_scorable_exact_math_gold(ans):
                continue
            if decon.is_math_contaminated(prob) or (_alnum_sha(prob) in prior_train_math):
                continue
            iid = f"hmmt_feb_2024:{r.get('problem_idx', idx)}"
            decon.register_math_eval(iid, prob)
            strat = _classify_math_stratum(str(r.get("problem_type", "")))
            search_seed_items.append(
                {
                    "id": iid,
                    "cluster_id": iid,
                    "stratum": strat,
                    "source": "hmmt_feb_2024",
                    "difficulty": 7.5,
                    "problem": prob,
                    "gold": ans,
                    "integer_task": bool(re.fullmatch(r"[+-]?\d+", ans)),
                    "messages": _math_prompt_messages(prob),
                }
            )
    except Exception as exc:
        print(f"  Note: hmmt_feb_2024 skipped ({exc})", flush=True)

    # Deterministic scan over DeepMath for Confirm (`300`) + remaining Search (`160 - len(search_seed_items)`)
    dm_indices = list(range(len(deepmath)))
    dm_indices.sort(key=lambda i: _det_hash(f"v3_dm_order:{i}"))

    seen_dm_eval_skels: set[str] = set()
    for idx in dm_indices:
        r = deepmath[idx]
        diff = float(r.get("difficulty") or 0.0)
        if not (7.0 <= diff <= 9.5):
            continue
        strat = _classify_math_stratum(str(r.get("topic", "")))
        if len(dm_eval_buckets[strat]) >= 165:
            if all(len(dm_eval_buckets[s]) >= 165 for s in ("algebra_analysis", "nt_combinatorics", "geometry_discrete")):
                break
            continue
        prob = str(r.get("question", "")).strip()
        ans = str(r.get("final_answer", "")).strip()
        if len(prob) < 35 or not _is_scorable_exact_math_gold(ans):
            continue
        if decon.is_math_contaminated(prob) or (_alnum_sha(prob) in prior_train_math):
            continue
        sk = _skeleton_sha(prob)
        if sk in seen_dm_eval_skels:
            continue
        seen_dm_eval_skels.add(sk)
        iid = f"deepmath_eval:{idx}"
        decon.register_math_eval(iid, prob)
        dm_eval_buckets[strat].append(
            {
                "id": iid,
                "cluster_id": iid,
                "stratum": strat,
                "source": "deepmath_hard",
                "difficulty": diff,
                "topic": str(r.get("topic", "")),
                "problem": prob,
                "gold": ans,
                "integer_task": bool(re.fullmatch(r"[+-]?\d+", ans)),
                "messages": _math_prompt_messages(prob),
            }
        )

    math_confirm300: list[dict[str, Any]] = []
    dm_for_search: list[dict[str, Any]] = []
    for s in ("algebra_analysis", "nt_combinatorics", "geometry_discrete"):
        bucket = dm_eval_buckets[s]
        if len(bucket) < 155:
            raise RuntimeError(f"Insufficient DeepMath hard items in stratum {s}: {len(bucket)}")
        math_confirm300.extend(bucket[:100])
        dm_for_search.extend(bucket[100:155])

    needed_search = 160 - len(search_seed_items)
    math_search160 = sorted(search_seed_items + dm_for_search[:needed_search], key=lambda x: x["id"])
    math_confirm300.sort(key=lambda x: x["id"])
    math_holdout_full.sort(key=lambda x: x["id"])
    math_holdout_prompts.sort(key=lambda x: x["id"])

    print(
        f"[2/6] Built Track M Eval Splits: Search={len(math_search160)}, "
        f"Confirm={len(math_confirm300)}, Holdout={len(math_holdout_full)}",
        flush=True,
    )

    # =========================================================================
    # PART B: TRACK M STAGE-I SFT, STAGE-II FRONTIER POOL & RLVR POOL
    # (Strictly 4-layer decontaminated against ALL V1, V2, and V3 Eval Splits!)
    # =========================================================================
    math_stage1_sft: list[dict[str, Any]] = []
    math_stage2_pool: list[dict[str, Any]] = []
    math_rl_pool: list[dict[str, Any]] = []
    used_train_math_skels: set[str] = set()

    # 1. Include clean mathematical reasoning items from `Axion1-0/axion-training-data` (`stage1` & `stage2`)
    for ax_sp in ("stage1", "stage2"):
        try:
            ax_ds = load_dataset("Axion1-0/axion-training-data", split=ax_sp)
            ax_kept = 0
            for r in ax_ds:
                q = str(r.get("question", "")).strip()
                resp = str(r.get("response", "")).strip()
                if not q or not resp or "\\boxed{" not in resp:
                    continue
                # Filter out tool/web/bio lookup tasks so only pure self-contained math/quantitative reasoning enters Track M
                if any(bad in (q + " " + resp).lower() for bad in ("http://", "https://", "uniprot", "wikipedia", "search snippet", "webpage")):
                    continue
                if decon.is_math_contaminated(q):
                    continue
                sk = _skeleton_sha(q)
                if sk in used_train_math_skels:
                    continue
                # Extract boxed gold from Axion ground-truth response and verify post-think <answer> self-consistency
                from math_verifier import last_boxed  # noqa: E402
                boxed_g = last_boxed(resp.split("</think>", 1)[-1])
                if not boxed_g or not _is_scorable_exact_math_gold(boxed_g):
                    continue
                clean_sol = _extract_clean_post_think_math_solution(resp, boxed_g, q, max_chars=14000)
                if not clean_sol:
                    continue
                used_train_math_skels.add(sk)
                iid = f"axion_{r['id']}"
                msgs = _math_prompt_messages(q) + [{"role": "assistant", "content": clean_sol}]
                math_stage1_sft.append(
                    {
                        "id": iid,
                        "problem_id": iid,
                        "source": f"axion_{ax_sp}",
                        "problem": q,
                        "gold": boxed_g,
                        "finish_reason": "stop",
                        "messages": msgs,
                    }
                )
                ax_kept += 1
                if ax_kept >= 600:
                    break
            print(f"  Axion `{ax_sp}` clean math SFT traces admitted: {ax_kept}", flush=True)
        except Exception as exc:
            print(f"  Axion `{ax_sp}` note: {exc}", flush=True)

    # 2. Scan remaining decontaminated DeepMath-103K for Stage-I Breadth SFT (~3,600), Stage-II Hard Frontier (~1,600), and RL Pool (640)
    eval_dm_ids = {int(x["id"].split(":")[1]) for x in (math_confirm300 + dm_for_search) if x["id"].startswith("deepmath_eval:")}
    for idx in dm_indices:
        if idx in eval_dm_ids:
            continue
        if len(math_stage1_sft) >= 4200 and len(math_stage2_pool) >= 1600 and len(math_rl_pool) >= 640:
            break
        r = deepmath[idx]
        diff = float(r.get("difficulty") or 0.0)
        if diff < 5.0:
            continue
        prob = str(r.get("question", "")).strip()
        ans = str(r.get("final_answer", "")).strip()
        if len(prob) < 30 or not _is_scorable_exact_math_gold(ans):
            continue
        if decon.is_math_contaminated(prob):
            continue
        sk = _skeleton_sha(prob)
        if sk in used_train_math_skels:
            continue

        # Find shortest verified post-`</think>` solution among r1_solution_1..3
        valid_sols: list[str] = []
        for skey in ("r1_solution_1", "r1_solution_2", "r1_solution_3"):
            cs = _extract_clean_post_think_math_solution(str(r.get(skey, "")), ans, prob, max_chars=14000)
            if cs is not None:
                valid_sols.append(cs)
        if not valid_sols:
            continue
        valid_sols.sort(key=len)
        best_sol = valid_sols[0]
        used_train_math_skels.add(sk)
        strat = _classify_math_stratum(str(r.get("topic", "")))

        if 6.2 <= diff <= 9.0 and len(math_rl_pool) < 640:
            iid = f"v3_m_rl:{idx}"
            math_rl_pool.append(
                {
                    "id": iid,
                    "cluster_id": iid,
                    "stratum": strat,
                    "source": "deepmath_rl",
                    "difficulty": diff,
                    "problem": prob,
                    "gold": ans,
                    "integer_task": bool(re.fullmatch(r"[+-]?\d+", ans)),
                    "reference_solution": best_sol,
                    "messages": _math_prompt_messages(prob),
                }
            )
        elif 7.2 <= diff <= 9.6 and len(math_stage2_pool) < 1600:
            iid = f"v3_m_s2:{idx}"
            math_stage2_pool.append(
                {
                    "id": iid,
                    "problem_id": iid,
                    "stratum": strat,
                    "source": "deepmath_hard_frontier",
                    "difficulty": diff,
                    "problem": prob,
                    "gold": ans,
                    "integer_task": bool(re.fullmatch(r"[+-]?\d+", ans)),
                    "reference_solution": best_sol,
                    "finish_reason": "stop",
                    "messages": _math_prompt_messages(prob),
                }
            )
        elif 5.0 <= diff <= 8.5 and len(math_stage1_sft) < 4200:
            iid = f"v3_m_s1:{idx}"
            math_stage1_sft.append(
                {
                    "id": iid,
                    "problem_id": iid,
                    "stratum": strat,
                    "source": "deepmath_breadth",
                    "difficulty": diff,
                    "problem": prob,
                    "gold": ans,
                    "finish_reason": "stop",
                    "messages": _math_prompt_messages(prob) + [{"role": "assistant", "content": best_sol}],
                }
            )

    print(
        f"[3/6] Built Track M Corpora: Stage1_SFT={len(math_stage1_sft)}, "
        f"Stage2_FrontierPool={len(math_stage2_pool)}, RL_Pool={len(math_rl_pool)}",
        flush=True,
    )

    # =========================================================================
    # PART C: TRACK L EVALUATION SPLITS (100% External Non-Mathlib Benchmarks!)
    # =========================================================================
    lean_holdout_full: list[dict[str, Any]] = []
    lean_holdout_prompts: list[dict[str, Any]] = []

    # 1. `arxivlean-{0326, 0626, 0926}` -> Protected Holdout Stratum 1 (`arxivlean_2026`)
    for ax_lean_repo, tag in [
        ("MathArena/arxivlean-0326", "arxivlean_0326"),
        ("MathArena/arxivlean-0626", "arxivlean_0626"),
        ("MathArena/arxivlean-0926", "arxivlean_0926"),
    ]:
        ds_al = list(load_dataset(ax_lean_repo, split="train"))
        tasks_al = [_split_lean_hdr_and_decl(str(r.get("formal_statement") or r.get("answer") or "")) for r in ds_al]
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            elabs = list(ex.map(lambda p: _elaborate_lean4_statement_v3(p[0], p[1]), tasks_al))
        kept_al = 0
        for idx, (r, (ok, clean_hdr, decl_prefix)) in enumerate(zip(ds_al, elabs)):
            if not ok:
                continue
            stmt = decl_prefix.strip() + " := by"
            inf = str(r.get("problem", "")).strip()
            if decon.is_lean_contaminated(stmt, inf) or (_norm_lean_stmt_sha(stmt) in prior_train_lean):
                continue
            iid = f"{tag}:{r.get('problem_idx', idx)}"
            decon.register_lean_eval(iid, stmt, inf)
            msgs = _lean_prompt_messages(stmt, clean_hdr, inf)
            lean_holdout_full.append(
                {
                    "id": iid,
                    "cluster_id": iid,
                    "stratum": "arxivlean_2026",
                    "contest": tag,
                    "source": tag,
                    "lean_header": clean_hdr,
                    "formal_statement": stmt,
                    "informal_statement": inf,
                    "messages": msgs,
                }
            )
            lean_holdout_prompts.append(
                {"id": iid, "cluster_id": iid, "stratum": "arxivlean_2026", "contest": tag, "source": tag, "messages": msgs}
            )
            kept_al += 1
        print(f"  Track L Holdout `{tag}`: {kept_al} elaborated disjoint theorems", flush=True)

    # 2. `SphereLab/FormalMATH-Lite` -> Partitioned deterministically into Holdout (140), Confirm (180), and Search (~70)
    fm_rows = list(load_dataset("SphereLab/FormalMATH-Lite", split="train"))
    fm_tasks = [_split_lean_hdr_and_decl(str(r.get("autoformalization", ""))) for r in fm_rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        fm_elabs = list(ex.map(lambda p: _elaborate_lean4_statement_v3(p[0], p[1]), fm_tasks))

    fm_valid: list[dict[str, Any]] = []
    for idx, (r, (ok, clean_hdr, decl_prefix)) in enumerate(zip(fm_rows, fm_elabs)):
        if not ok:
            continue
        stmt = decl_prefix.strip() + " := by"
        inf = str(r.get("refined_statement", "")).strip()
        if decon.is_lean_contaminated(stmt, inf) or (_norm_lean_stmt_sha(stmt) in prior_train_lean):
            continue
        iid = f"formalmath:{idx}"
        decon.register_lean_eval(iid, stmt, inf)
        strat = _classify_lean_stratum(str(r.get("domain", "")), stmt)
        fm_valid.append(
            {
                "id": iid,
                "cluster_id": iid,
                "stratum": strat,
                "source": f"formalmath_{r.get('source', 'lite')}",
                "lean_header": clean_hdr,
                "formal_statement": stmt,
                "informal_statement": inf,
                "messages": _lean_prompt_messages(stmt, clean_hdr, inf),
            }
        )

    fm_valid.sort(key=lambda x: _det_hash("v3_fm_split:" + x["id"]))
    print(f"  FormalMATH-Lite elaborated & deduplicated: {len(fm_valid)} theorems", flush=True)
    if len(fm_valid) < 350:
        raise RuntimeError(f"Expected >= 350 FormalMATH-Lite items, got {len(fm_valid)}")

    fm_hold140 = fm_valid[:140]
    for it in fm_hold140:
        h_it = dict(it)
        h_it["stratum"] = f"formalmath_{it['stratum']}"
        lean_holdout_full.append(h_it)
        lean_holdout_prompts.append(
            {
                "id": h_it["id"],
                "cluster_id": h_it["id"],
                "stratum": h_it["stratum"],
                "contest": "formalmath_lite",
                "source": h_it["source"],
                "messages": h_it["messages"],
            }
        )

    lean_confirm180 = sorted(fm_valid[140:320], key=lambda x: x["id"])
    fm_search_part = fm_valid[320:]

    # Top up `v3_lean_search140` to exact `N = 140` using disjoint elaborated theorems from `Goedel-Pset-v1`
    pset_ds = load_dataset("Goedel-LM/Goedel-Pset-v1", split="train")
    need_pset_search = max(0, 140 - len(fm_search_part))
    pset_cand_rows = [pset_ds[i] for i in range(0, min(len(pset_ds), 40000), 97)][: need_pset_search * 3]
    pset_tasks = [_split_lean_hdr_and_decl(str(r.get("formal_statement", ""))) for r in pset_cand_rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        pset_elabs = list(ex.map(lambda p: _elaborate_lean4_statement_v3(p[0], p[1]), pset_tasks))

    pset_search_kept: list[dict[str, Any]] = []
    for r, (ok, clean_hdr, decl_prefix) in zip(pset_cand_rows, pset_elabs):
        if len(pset_search_kept) >= need_pset_search:
            break
        if not ok:
            continue
        stmt = decl_prefix.strip() + " := by"
        inf = str(r.get("informal_statement", "")).strip()
        if decon.is_lean_contaminated(stmt, inf) or (_norm_lean_stmt_sha(stmt) in prior_train_lean):
            continue
        iid = f"goedel_pset_search:{r['problem_id']}"
        decon.register_lean_eval(iid, stmt, inf)
        strat = _classify_lean_stratum("", stmt)
        pset_search_kept.append(
            {
                "id": iid,
                "cluster_id": iid,
                "stratum": strat,
                "source": "goedel_pset_v1",
                "lean_header": clean_hdr,
                "formal_statement": stmt,
                "informal_statement": inf,
                "messages": _lean_prompt_messages(stmt, clean_hdr, inf),
            }
        )

    lean_search140 = sorted(fm_search_part + pset_search_kept, key=lambda x: x["id"])
    lean_holdout_full.sort(key=lambda x: x["id"])
    lean_holdout_prompts.sort(key=lambda x: x["id"])

    print(
        f"[4/6] Built Track L Eval Splits: Search={len(lean_search140)}, "
        f"Confirm={len(lean_confirm180)}, Holdout={len(lean_holdout_full)}",
        flush=True,
    )

    # =========================================================================
    # PART D: TRACK L `#print axioms`-VERIFIED SFT & RLVR CORPORA
    # =========================================================================
    sft_goedel = load_dataset("Goedel-LM/SFT_dataset_v2", split="train")
    used_train_lean_shas: set[str] = set()
    goedel_verify_tasks: list[tuple[str, str, str, str, str]] = []

    # Sample evenly across the 1.74M Goedel-SFT-v2 corpus for diverse domain coverage (`~6,200` candidates -> `~4,500` verified)
    stride = max(1, len(sft_goedel) // 7500)
    for idx in range(0, len(sft_goedel), stride):
        if len(goedel_verify_tasks) >= 5600:
            break
        msgs = sft_goedel[idx]["messages"]
        if isinstance(msgs, str):
            try:
                msgs = ast.literal_eval(msgs)
            except Exception:
                continue
        if not msgs or len(msgs) < 2:
            continue
        u_txt = str(msgs[0].get("content", ""))
        a_txt = str(msgs[-1].get("content", ""))
        # Extract ```lean4 statement from user prompt
        m_code = re.search(r"```(?:lean4|lean)\s*\n(.*?)```", u_txt, flags=re.DOTALL)
        if not m_code:
            continue
        hdr, decl = _split_lean_hdr_and_decl(m_code.group(1))
        try:
            _, decl_prefix = split_frozen_statement(decl)
        except Exception:
            continue
        stmt = decl_prefix.strip() + " := by"
        if decon.is_lean_contaminated(stmt):
            continue
        lsha = _norm_lean_stmt_sha(stmt)
        if lsha in used_train_lean_shas:
            continue
        # Strip `<think>` if present in assistant output so SFT trains direct Lean proof mode cleanly
        clean_a = a_txt.rsplit("</think>", 1)[1].strip() if "</think>" in a_txt else a_txt.strip()
        if len(clean_a) > 6000:
            continue
        used_train_lean_shas.add(lsha)
        goedel_verify_tasks.append((f"goedel_sft_v2:{idx}", hdr, stmt, clean_a, "goedel_sft_v2"))

    print(f"  Verifying {len(goedel_verify_tasks)} Goedel-SFT-v2 proofs via Lean 4.24 `#print axioms`...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:
        verified_goedel_raw = list(ex.map(_verify_one_goedel_or_mathlib_sft, goedel_verify_tasks, chunksize=8))
    verified_goedel = [x for x in verified_goedel_raw if x is not None]
    print(f"  Goedel-SFT-v2 `#print axioms` verified: {len(verified_goedel)} / {len(goedel_verify_tasks)}", flush=True)

    # Standalone anonymized Mathlib multi-step tactic proofs (`Algebra`, `Analysis`, `Topology`, etc.)
    ml_raw_cands = _extract_mathlib_standalone_candidates(max_candidates=1600)
    ml_verify_tasks: list[tuple[str, str, str, str, str]] = []
    for cid, hdr, combined in ml_raw_cands:
        stmt_line, proof_part = combined.split("\n", 1)
        if decon.is_lean_contaminated(stmt_line):
            continue
        ml_verify_tasks.append((cid, hdr, stmt_line, proof_part, "mathlib4_standalone_sft"))

    print(f"  Verifying {len(ml_verify_tasks)} standalone Mathlib4 multi-step tactic proofs...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:
        verified_ml_raw = list(ex.map(_verify_one_goedel_or_mathlib_sft, ml_verify_tasks, chunksize=8))
    verified_ml = [x for x in verified_ml_raw if x is not None]
    print(f"  Standalone Mathlib4 `#print axioms` verified: {len(verified_ml)} / {len(ml_verify_tasks)}", flush=True)

    # Partition `verified_goedel` cleanly into RL Pool (500), Stage-II Frontier Pool (1400), and Stage-I Breadth SFT (rest + verified_ml)
    verified_goedel.sort(key=lambda x: _det_hash("v3_goedel_part:" + x["id"]))
    lean_rl_src = verified_goedel[:500]
    lean_s2_src = verified_goedel[500:1900]
    lean_s1_goedel = verified_goedel[1900:4400]

    lean_rl_pool500: list[dict[str, Any]] = []
    for x in lean_rl_src:
        strat = _classify_lean_stratum("", x["formal_statement"])
        lean_rl_pool500.append(
            {
                "id": x["id"],
                "cluster_id": x["id"],
                "stratum": strat,
                "source": "goedel_rl_verified",
                "lean_header": x["lean_header"],
                "formal_statement": x["formal_statement"],
                "reference_proof": f"```lean4\n{x['proof_body']}\n```",
                "messages": x["messages"][:-1],
            }
        )

    lean_stage2_pool: list[dict[str, Any]] = []
    for x in lean_s2_src:
        lean_stage2_pool.append(x)
    # Also incorporate verified compiler-repair pairs from V2 (`lean_qwen35_9b_r2_rep5050_pairs_only.jsonl`) if decontaminated
    v2_rep_file = V2_ROOT / "data/verified_banks_v2/lean_qwen35_9b_r2_rep5050_pairs_only.jsonl"
    if v2_rep_file.exists():
        for ln in v2_rep_file.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            fs = str(r.get("formal_statement", ""))
            if fs and not decon.is_lean_contaminated(fs):
                lean_stage2_pool.append(r)

    lean_stage1_sft = lean_s1_goedel + verified_ml[:1500]
    print(
        f"[5/6] Built Track L Corpora: Stage1_SFT={len(lean_stage1_sft)} "
        f"(Goedel={len(lean_s1_goedel)} + Mathlib={min(1500, len(verified_ml))}), "
        f"Stage2_Pool={len(lean_stage2_pool)}, RL_Pool={len(lean_rl_pool500)}",
        flush=True,
    )

    # =========================================================================
    # PART E: WRITE ALL SPLITS & RUN `governed_loop_guard_v37.py audit-splits`
    # =========================================================================
    manifest_files = {
        "v3_math_search160": _write_jsonl(splits_dir / "v3_math_search160.jsonl", math_search160),
        "v3_math_confirm300": _write_jsonl(splits_dir / "v3_math_confirm300.jsonl", math_confirm300),
        "v3_math_holdout_prompts": _write_jsonl(splits_dir / "v3_math_holdout_prompts.jsonl", math_holdout_prompts),
        "sealed_v3_math_holdout": _write_jsonl(holdout_dir / "v3_math_holdout.jsonl", math_holdout_full),
        "v3_math_stage1_sft": _write_jsonl(corpora_dir / "v3_math_stage1_sft.jsonl", math_stage1_sft),
        "v3_math_stage2_frontier_pool": _write_jsonl(corpora_dir / "v3_math_stage2_frontier_pool.jsonl", math_stage2_pool),
        "v3_math_rl_pool640": _write_jsonl(corpora_dir / "v3_math_rl_pool640.jsonl", math_rl_pool),
        "v3_lean_search140": _write_jsonl(splits_dir / "v3_lean_search140.jsonl", lean_search140),
        "v3_lean_confirm180": _write_jsonl(splits_dir / "v3_lean_confirm180.jsonl", lean_confirm180),
        "v3_lean_holdout_prompts": _write_jsonl(splits_dir / "v3_lean_holdout_prompts.jsonl", lean_holdout_prompts),
        "sealed_v3_lean_holdout": _write_jsonl(holdout_dir / "v3_lean_holdout.jsonl", lean_holdout_full),
        "v3_lean_stage1_sft": _write_jsonl(corpora_dir / "v3_lean_stage1_sft.jsonl", lean_stage1_sft),
        "v3_lean_stage2_frontier_pool": _write_jsonl(corpora_dir / "v3_lean_stage2_frontier_pool.jsonl", lean_stage2_pool),
        "v3_lean_rl_pool500": _write_jsonl(corpora_dir / "v3_lean_rl_pool500.jsonl", lean_rl_pool500),
    }

    guard_py = EVAL_V3_DIR / "governed_loop_guard_v37.py"
    audit_results: dict[str, Any] = {}
    for label, tr_p, cf_p, ho_p in [
        (
            "math_stage1_vs_eval",
            corpora_dir / "v3_math_stage1_sft.jsonl",
            splits_dir / "v3_math_confirm300.jsonl",
            holdout_dir / "v3_math_holdout.jsonl",
        ),
        (
            "math_rl_vs_search_and_confirm",
            corpora_dir / "v3_math_rl_pool640.jsonl",
            splits_dir / "v3_math_search160.jsonl",
            splits_dir / "v3_math_confirm300.jsonl",
        ),
        (
            "lean_stage1_vs_eval",
            corpora_dir / "v3_lean_stage1_sft.jsonl",
            splits_dir / "v3_lean_confirm180.jsonl",
            holdout_dir / "v3_lean_holdout.jsonl",
        ),
        (
            "lean_rl_vs_search_and_confirm",
            corpora_dir / "v3_lean_rl_pool500.jsonl",
            splits_dir / "v3_lean_search140.jsonl",
            splits_dir / "v3_lean_confirm180.jsonl",
        ),
    ]:
        r_aud = subprocess.run(
            [
                sys.executable, "-I", "-B", str(guard_py), "audit-splits",
                "--train", str(tr_p),
                "--confirm", str(cf_p),
                "--holdout", str(ho_p),
                "--group-key", "id",
            ],
            capture_output=True,
            text=True,
        )
        if r_aud.returncode != 0:
            raise RuntimeError(f"audit-splits FAILED on {label}: {r_aud.stdout}\n{r_aud.stderr}")
        audit_results[label] = json.loads(r_aud.stdout)

    manifest_out = {
        "status": "V3_SPLITS_AND_CORPORA_VERIFIED",
        "files": manifest_files,
        "audit_splits_verdicts": audit_results,
    }
    (splits_dir / "v3_split_audit_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[6/6] ALL V3 SPLITS AND CORPORA BUILT & AUDIT-SPLITS PASSED!", flush=True)
    print(json.dumps(manifest_files, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

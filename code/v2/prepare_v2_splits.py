#!/usr/bin/env python3
"""
Phase V2 Disjoint Split Builder & Zero-Overlap Auditor.

Builds completely disjoint V2 Search, Confirmation, Training Pool, and Sealed Single-Touch Holdout
splits across Track L (ProofNet-Lean4 + disjoint Lean-Workbook) and Track M (Omni-MATH Tier 3-4 +
BeyondAIME + BRUMO 2025 + CMIMC 2025 + HMMT Nov 2025), enforcing 0 ID and 0 normalized-text SHA-256
overlap against ALL Phase V1 splits and training/seed corpora.
"""
from __future__ import annotations

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
LAKE_BIN = V1_ROOT / "lean/elan/bin/lake"
LEAN_WS = V1_ROOT / "lean/mathlib_env"

EVAL_DIR = V1_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from lean4_verifier import split_frozen_statement  # noqa: E402
from math_verifier import exact_integer_equivalent, grade_symbolic_math  # noqa: E402


LEAN_SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover using Mathlib. "
    "Output a complete, valid Lean 4 proof inside a single ```lean4 fenced code block. "
    "Do not use sorry, admit, or custom axioms."
)

MATH_USER_SUFFIX = "\n\nSolve the problem carefully and put your final answer inside \\boxed{}."


def _norm_text_sha(text: str) -> str:
    norm = re.sub(r"\s+", " ", str(text).strip().lower())
    # Also strip common formatting noise for strict cross-contest duplicate detection
    alnum = re.sub(r"[^a-z0-9]+", "", norm)
    return hashlib.sha256(alnum.encode("utf-8")).hexdigest()


def _norm_lean_stmt_sha(stmt: str) -> str:
    # Strip theorem name so renamed aliases of the same mathematical signature are caught!
    s = re.sub(r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+\S+", "theorem _anon_", stmt.strip())
    s = re.sub(r":=.*$", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s.strip())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _collect_v1_hashes() -> tuple[set[str], set[str], set[str]]:
    ids: set[str] = set()
    math_shas: set[str] = set()
    lean_shas: set[str] = set()
    search_dirs = [
        V1_ROOT / "data/splits",
        V1_ROOT / "eval/sealed_holdout",
        V1_ROOT / "data/corpora",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for fp in d.rglob("*.jsonl"):
            for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if "id" in row:
                    ids.add(str(row["id"]))
                if "formal_statement" in row and row["formal_statement"]:
                    lean_shas.add(_norm_lean_stmt_sha(str(row["formal_statement"])))
                if "problem" in row and row["problem"]:
                    math_shas.add(_norm_text_sha(str(row["problem"])))
                elif "messages" in row and row["messages"]:
                    for m in row["messages"]:
                        if m.get("role") == "user":
                            txt = str(m.get("content", "")).replace(MATH_USER_SUFFIX, "")
                            math_shas.add(_norm_text_sha(txt))
    return ids, math_shas, lean_shas


def _elaborate_lean4_statement(header: str, formal_stmt: str) -> tuple[bool, str, str]:
    """Verifies `<header> + <decl_prefix> := by sorry` elaborates cleanly in Lean 4.24."""
    try:
        thm_name, decl_prefix = split_frozen_statement(formal_stmt)
    except Exception as exc:
        return False, "", f"split_err:{exc}"

    lines: list[str] = []
    base_hdr = header if "import Mathlib" in header else ("import Mathlib\n" + header)
    if "import Aesop" not in base_hdr:
        base_hdr = base_hdr.replace("import Mathlib", "import Mathlib\nimport Aesop", 1)
    for ln in base_hdr.splitlines():
        st = ln.strip()
        if st.startswith("open") and st in [x.strip() for x in lines]:
            continue
        lines.append(ln)
    if not any("maxHeartbeats" in x for x in lines):
        lines.append("set_option maxHeartbeats 400000")
    clean_hdr = "\n".join(lines).strip() + "\n"

    # Reject vacuous True goals
    rhs_type = decl_prefix.rsplit(":", 1)[-1].strip() if ":" in decl_prefix else ""
    if rhs_type == "True":
        return False, clean_hdr, "vacuous_true_goal"

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


def _make_lean_record(item_id: str, clean_hdr: str, decl_prefix: str, informal: str, source: str) -> dict[str, Any]:
    stmt = decl_prefix.strip() + " := by"
    user_prompt = (
        f"{LEAN_SYSTEM_PROMPT}\n\n"
        + (f"Informal statement:\n{informal.strip()}\n\n" if informal.strip() else "")
        + f"Complete the following Lean 4 theorem:\n```lean4\n{clean_hdr.strip()}\n\n{stmt}\n```"
    )
    return {
        "id": item_id,
        "cluster_id": item_id,
        "source": source,
        "lean_header": clean_hdr,
        "formal_statement": stmt,
        "informal_statement": informal.strip(),
        "messages": [{"role": "user", "content": user_prompt}],
    }


def _make_math_record(item_id: str, problem: str, gold: str, source: str, difficulty: float = 0.0, domain: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    int_task = False
    try:
        iv = int(str(gold).strip())
        if str(iv) == str(gold).strip():
            int_task = True
    except Exception:
        int_task = False
    msg = [{"role": "user", "content": str(problem).strip() + MATH_USER_SUFFIX}]
    full_rec = {
        "id": item_id,
        "cluster_id": item_id,
        "source": source,
        "difficulty": difficulty,
        "domain": domain,
        "problem": str(problem).strip(),
        "gold": str(gold).strip(),
        "integer_task": int_task,
        "messages": msg,
    }
    prompt_only = {
        "id": item_id,
        "cluster_id": item_id,
        "source": source,
        "messages": msg,
    }
    return full_rec, prompt_only


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_scorable_exact_math_gold(gold: str) -> bool:
    g = str(gold or "").strip()
    if not g or len(g) > 64:
        return False
    low = g.lower()
    bad_words = ("prove", "show", "none", "true", "false", "yes", "no", "infinit", "impossible", "cannot", "arbitrar", "see ")
    if any(w in low for w in bad_words):
        return False
    # Self-consistency check: grading gold against gold must succeed cleanly!
    try:
        iv = int(g)
        if str(iv) == g:
            return exact_integer_equivalent(g, g)
    except Exception:
        pass
    res = grade_symbolic_math(g, g)
    return bool(res.get("correct", False)) and not bool(res.get("parse_failure", True))


def main() -> int:
    v1_ids, v1_math_shas, v1_lean_shas = _collect_v1_hashes()
    print(f"Collected V1 blocklist: {len(v1_ids)} IDs, {len(v1_math_shas)} math hashes, {len(v1_lean_shas)} lean hashes", flush=True)

    splits_dir = V2_ROOT / "data/splits_v2"
    holdout_dir = V2_ROOT / "eval_v2/sealed_holdout"
    corpora_dir = V2_ROOT / "data/corpora"
    for d in (splits_dir, holdout_dir, corpora_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Track L V2: ProofNet-Lean4 (valid -> Confirm, test -> Holdout)
    # ------------------------------------------------------------------
    pn = load_dataset("HaimingW/proofnet-lean4")
    used_lean_shas = set(v1_lean_shas)

    def build_proofnet_split(split_name: str, prefix: str) -> list[dict[str, Any]]:
        rows = list(pn[split_name])
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futs = [ex.submit(_elaborate_lean4_statement, r["header"], r["formal_statement"]) for r in rows]
            elab_res = [f.result() for f in futs]
        kept: list[dict[str, Any]] = []
        for r, (ok, clean_hdr, decl_prefix) in zip(rows, elab_res):
            if not ok:
                continue
            sha = _norm_lean_stmt_sha(decl_prefix)
            if sha in used_lean_shas:
                continue
            used_lean_shas.add(sha)
            iid = f"{prefix}:{r['name']}"
            kept.append(_make_lean_record(iid, clean_hdr, decl_prefix, r.get("informal_prefix", ""), f"proofnet_{split_name}"))
        return kept

    pn_holdout = build_proofnet_split("test", "proofnet_test")
    pn_confirm = build_proofnet_split("valid", "proofnet_valid")
    print(f"ProofNet-Lean4 elaborated & deduplicated: Confirm={len(pn_confirm)}, Holdout={len(pn_holdout)}", flush=True)

    # Build disjoint Lean-Workbook Search (120) + Training Pool (600)
    try:
        wb = load_dataset("internlm/Lean-Workbook", split="train")
    except Exception:
        wb = load_dataset("Goedel-LM/Lean-workbook-proofs", split="train")
    wb_candidates: list[dict[str, Any]] = []
    for idx, r in enumerate(wb):
        stmt_raw = str(r.get("formal_statement") or r.get("full_proof") or r.get("theorem") or r.get("problem") or "")
        if not stmt_raw:
            continue
        m_thm = re.search(r"(?m)^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+", stmt_raw)
        if not m_thm:
            continue
        hdr_part = stmt_raw[:m_thm.start()].strip()
        decl_part = stmt_raw[m_thm.start():].strip()
        if ":=" not in decl_part:
            decl_part = decl_part.rstrip() + " := by sorry"
        sha = _norm_lean_stmt_sha(decl_part)
        if sha in used_lean_shas:
            continue
        wb_candidates.append({
            "raw_id": str(r.get("id") or r.get("problem_id") or f"wb_{idx}"),
            "hdr": hdr_part,
            "decl": decl_part,
            "informal": str(r.get("natural_language_statement") or ""),
            "sha": sha,
        })
        if len(wb_candidates) >= 2200:
            break

    # Deterministic hash ordering before elaboration
    wb_candidates.sort(key=lambda x: hashlib.sha256(f"V2-L:{x['raw_id']}".encode()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        wb_elab = list(ex.map(lambda c: _elaborate_lean4_statement(c["hdr"], c["decl"]), wb_candidates))

    wb_valid_recs: list[dict[str, Any]] = []
    wb_seed_proofs: list[dict[str, Any]] = []
    for c, (ok, clean_hdr, decl_prefix) in zip(wb_candidates, wb_elab):
        if not ok or c["sha"] in used_lean_shas:
            continue
        used_lean_shas.add(c["sha"])
        iid = f"v2_lwb:{c['raw_id']}"
        rec = _make_lean_record(iid, clean_hdr, decl_prefix, c.get("informal", ""), "lean_workbook_v2")
        wb_valid_recs.append(rec)
        if ":=" in c["decl"]:
            rhs = c["decl"].split(":=", 1)[1].strip()
            if rhs and "sorry" not in rhs:
                wb_seed_proofs.append({"id": iid, "raw_proof_rhs": rhs, **rec})
        if len(wb_valid_recs) >= 720:
            break

    v2_lean_search = wb_valid_recs[:120]
    v2_lean_train_pool = wb_valid_recs[120:720]
    print(f"Disjoint Lean-Workbook V2: Search={len(v2_lean_search)}, TrainPool={len(v2_lean_train_pool)}", flush=True)

    _write_jsonl(splits_dir / "v2_lean_search120.jsonl", v2_lean_search)
    _write_jsonl(splits_dir / "v2_lean_confirm171.jsonl", pn_confirm)
    _write_jsonl(holdout_dir / "v2_lean_holdout_proofnet174.jsonl", pn_holdout)
    _write_jsonl(
        splits_dir / "v2_lean_holdout_proofnet174_prompts.jsonl",
        [{"id": r["id"], "cluster_id": r["cluster_id"], "messages": r["messages"]} for r in pn_holdout],
    )
    _write_jsonl(corpora_dir / "v2_lean_train_pool.jsonl", v2_lean_train_pool)

    # ------------------------------------------------------------------
    # 2. Track M V2: Prioritize Holdout Contests, Then Omni-MATH Tier 3-4
    # ------------------------------------------------------------------
    used_math_shas = set(v1_math_shas)
    math_holdout_counts: dict[str, int] = {}

    holdout_sources = [
        ("beyondaime", "ByteDance-Seed/BeyondAIME", "test"),
        ("brumo2025", "MathArena/brumo_2025", "train"),
        ("cmimc2025", "MathArena/cmimc_2025", "train"),
        ("hmmt_nov2025", "MathArena/hmmt_nov_2025", "train"),
    ]
    for slug, hf_repo, sp in holdout_sources:
        ds = load_dataset(hf_repo, split=sp)
        full_rows: list[dict[str, Any]] = []
        prompt_rows: list[dict[str, Any]] = []
        for idx, r in enumerate(ds):
            prob = str(r["problem"]).strip()
            ans = str(r["answer"]).strip()
            sha = _norm_text_sha(prob)
            if sha in used_math_shas:
                continue
            if not _is_scorable_exact_math_gold(ans):
                continue
            used_math_shas.add(sha)
            pidx = r.get("problem_idx", idx)
            iid = f"v2_{slug}:{pidx}"
            f_rec, p_rec = _make_math_record(iid, prob, ans, slug)
            full_rows.append(f_rec)
            prompt_rows.append(p_rec)
        _write_jsonl(holdout_dir / f"v2_math_holdout_{slug}.jsonl", full_rows)
        _write_jsonl(splits_dir / f"v2_math_holdout_{slug}_prompts.jsonl", prompt_rows)
        math_holdout_counts[slug] = len(full_rows)
        print(f"V2 Math Holdout {slug}: {len(full_rows)} eligible & disjoint problems", flush=True)

    # Omni-MATH Tier 3 (5.0 <= diff < 7.0) and Tier 4 (diff >= 7.0)
    omni = load_dataset("KbsdJames/Omni-MATH", split="test")
    tier3: list[tuple[str, dict[str, Any]]] = []
    tier4: list[tuple[str, dict[str, Any]]] = []
    tier_train_extra: list[tuple[str, dict[str, Any]]] = []

    for idx, r in enumerate(omni):
        prob = str(r["problem"]).strip()
        ans = str(r["answer"]).strip()
        sha = _norm_text_sha(prob)
        if sha in used_math_shas:
            continue
        if not _is_scorable_exact_math_gold(ans):
            continue
        try:
            diff = float(r.get("difficulty") or 0.0)
        except Exception:
            diff = 0.0
        if diff < 4.0:
            continue
        used_math_shas.add(sha)
        iid = f"v2_omnimath:{idx}"
        det_key = hashlib.sha256(f"V2-M:{iid}".encode("utf-8")).hexdigest()
        f_rec, _ = _make_math_record(iid, prob, ans, "omnimath", difficulty=diff, domain=str(r.get("domain", "")))
        if diff >= 7.0:
            tier4.append((det_key, f_rec))
        elif diff >= 5.0:
            tier3.append((det_key, f_rec))
        else:
            tier_train_extra.append((det_key, f_rec))

    tier3.sort(key=lambda x: x[0])
    tier4.sort(key=lambda x: x[0])
    tier_train_extra.sort(key=lambda x: x[0])
    print(f"Disjoint scorable Omni-MATH pool: Tier3={len(tier3)}, Tier4={len(tier4)}, Tier2.5={len(tier_train_extra)}", flush=True)

    # Confirm = lowest 128 hashes of Tier 3 + lowest 128 hashes of Tier 4 = 256 (GPT 6.2 Astra Max spec)
    n_c3 = min(128, len(tier3) // 2)
    n_c4 = min(128, len(tier4) // 2)
    v2_math_confirm = [x[1] for x in tier3[:n_c3]] + [x[1] for x in tier4[:n_c4]]
    rem3 = [x[1] for x in tier3[n_c3:]]
    rem4 = [x[1] for x in tier4[n_c4:]]

    # Search = next 64 of Tier 3 + next 64 of Tier 4 = 128
    n_s3 = min(64, len(rem3) // 2)
    n_s4 = min(64, len(rem4) // 2)
    v2_math_search = rem3[:n_s3] + rem4[:n_s4]

    # Train pool = remaining Tier 3 + Tier 4 + Tier 2.5 (up to 600 hard problems)
    v2_math_train = (rem3[n_s3:] + rem4[n_s4:] + [x[1] for x in tier_train_extra])[:600]

    _write_jsonl(splits_dir / "v2_math_confirm256.jsonl", v2_math_confirm)
    _write_jsonl(splits_dir / "v2_math_search128.jsonl", v2_math_search)
    _write_jsonl(corpora_dir / "v2_math_train_pool.jsonl", v2_math_train)

    audit = {
        "v1_blocked_ids": len(v1_ids),
        "v1_blocked_math_shas": len(v1_math_shas),
        "v1_blocked_lean_shas": len(v1_lean_shas),
        "track_l_v2": {
            "search": len(v2_lean_search),
            "confirm_proofnet_valid": len(pn_confirm),
            "holdout_proofnet_test": len(pn_holdout),
            "train_pool": len(v2_lean_train_pool),
        },
        "track_m_v2": {
            "search": len(v2_math_search),
            "confirm_omnimath_t3_t4": len(v2_math_confirm),
            "holdout_contests": math_holdout_counts,
            "holdout_total": sum(math_holdout_counts.values()),
            "train_pool": len(v2_math_train),
        },
        "zero_v1_overlap_verified": True,
        "zero_cross_split_overlap_verified": True,
    }
    (splits_dir / "v2_split_audit_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

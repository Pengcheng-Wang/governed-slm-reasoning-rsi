#!/usr/bin/env python3
"""
Wave-2 (`V4.1`) Unexposed Balanced-Stratum Split Builder (`B2`, `B8`, `B10`).

Constructs genuinely fresh, never-previously-evaluated `Confirm_V4_1` (`N=150`) and
single-touch `Holdout_V4_1` (`N=150`) splits for both Track L (Lean 4.24) and Track M (Math):
  1. Loads ALL 32 existing `.jsonl` split & training files across V1, V2, V3, and V4 into a
     4-layer decontaminator (`ID` + `alphanumeric SHA` + `digit-masked skeleton SHA` + `13-gram`
     + Lean type-skeleton SHA) so ZERO item, variant, or cluster ever seen in V1-V4 can enter.
  2. Samples fresh external theorems from `Goedel-LM/Goedel-Pset-v1` (elaboration-verified in
     Lean 4.24, non-vacuous) and fresh hard competition problems (`difficulty in [7.0, 9.5]`)
     from `zwhe99/DeepMath-103K` (scorable exact gold).
  3. Enforces exact `50 / 50 / 50` stratum balance (`N_eff = 150.0 = 100% N`, so `equal_item`
     and `equal_stratum` coincide identically, eliminating `113/18/9` stratum skew).
  4. Assigns whole skeleton-clusters within each stratum via pre-committed SHA-256 salt
     `SHA256("v4_1_salt_20261124:" || cluster_id)`: first 50 -> `Confirm_V4_1`, next 50 -> `Holdout_V4_1`.
  5. Runs `governed_loop_guard_v39.py audit-splits` and writes `v4_1_split_audit_manifest.json`.
"""
from __future__ import annotations

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

ROOT = Path("/scratch/governed_crossfamily_v4")
EVAL_V4_DIR = ROOT / "eval_v4"
if str(EVAL_V4_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V4_DIR))

from lean4_verifier_v4 import MATHLIB_HEADER, split_frozen_statement  # noqa: E402
from math_verifier_v4 import exact_integer_equivalent, grade_symbolic_math  # noqa: E402

LAKE_BIN = ROOT / "lean/elan/bin/lake"
ELAN_HOME = ROOT / "lean/elan"
LEAN_WS = ROOT / "lean/mathlib_env"
SALT = "v4_1_salt_20261124"
MATH_SUFFIX = "\n\nSolve the problem step by step and put your final answer inside \\boxed{}."


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _alnum_sha(text: str) -> str:
    return _sha256_str(re.sub(r"[^a-z0-9]+", "", str(text).lower()))


def _skeleton_sha(text: str) -> str:
    norm = re.sub(r"\d+", "#", re.sub(r"\s+", " ", str(text).strip().lower()))
    return _sha256_str(re.sub(r"[^a-z#]+", "", norm))


def _ngrams_13(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    if len(toks) < 13:
        return []
    return [" ".join(toks[i : i + 13]) for i in range(0, len(toks) - 12, 2)]


def _norm_lean_stmt_sha(stmt: str) -> str:
    s = re.sub(r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+\S+", "theorem _anon_", str(stmt).strip())
    s = re.sub(r":=.*$", "", s, flags=re.DOTALL)
    return _sha256_str(re.sub(r"\s+", " ", s.strip()))


def _classify_math_stratum(topic: str) -> str:
    low = str(topic).lower()
    if any(k in low for k in ("number theory", "combinatorics", "probability", "counting", "graph")):
        return "nt_combinatorics"
    if any(k in low for k in ("geometry", "trigonometr", "discrete", "logic", "set")):
        return "geometry_discrete"
    return "algebra_analysis"


def _classify_lean_stratum(stmt: str, inf: str = "") -> str:
    low = (str(stmt) + " " + str(inf)).lower()
    if any(k in low for k in ("calc", "deriv", "integr", "limit", "tendsto", "continuous", "series", "filter", "metric", "topology", "real.cos", "real.sin", "real.log", "real.exp")):
        return "analysis_calculus"
    if any(k in low for k in ("number", "prime", "dvd", "mod", "zmod", "nat.gcd", "comb", "finset", "card", "geom", "choose", "factorial")):
        return "nt_combinatorics_geom"
    return "algebra"


def _is_scorable_math_gold(gold: str) -> bool:
    g = str(gold or "").strip()
    if not g or len(g) > 80:
        return False
    low = g.lower()
    if any(w in low for w in ("prove", "show", "none", "true", "false", "yes", "no", "infinit", "impossible", "cannot", "arbitrar", "see ", "exists")):
        return False
    if re.fullmatch(r"[+-]?\d+", g):
        return exact_integer_equivalent(g, g)
    res = grade_symbolic_math(g, g)
    return bool(res.get("correct", False)) and not bool(res.get("parse_failure", True))


def _split_lean_hdr_and_decl(fs: str) -> tuple[str, str]:
    lines = str(fs).splitlines()
    hdr_l, decl_l = [], []
    in_decl = False
    for ln in lines:
        s = ln.lstrip()
        if not in_decl and any(s.startswith(k) for k in ("theorem ", "lemma ", "noncomputable theorem ", "noncomputable lemma ")):
            in_decl = True
        if in_decl:
            decl_l.append(ln)
        elif not s.startswith(("/--", "--", "-/")):
            hdr_l.append(ln)
    hdr = "\n".join(hdr_l).strip() if hdr_l else MATHLIB_HEADER.strip()
    decl = "\n".join(decl_l).strip() if decl_l else str(fs).strip()
    return hdr, decl


def _elaborate_lean4(task: tuple[str, str]) -> tuple[bool, str, str]:
    header, formal_stmt = task
    combined = f"{header}\n{formal_stmt}"
    if re.search(r"\b(?:abbrev|def)\b[^:]*:=\s*(?:by\s+)?sorry\b", combined):
        return False, "", "vacuous_abbrev"
    try:
        _, decl_prefix = split_frozen_statement(formal_stmt)
    except Exception:
        return False, "", "split_err"
    rhs_type = decl_prefix.rsplit(":", 1)[-1].strip() if ":" in decl_prefix else ""
    if rhs_type == "True":
        return False, "", "vacuous_true"

    base_hdr = header if "import Mathlib" in header else ("import Mathlib\n" + header)
    if "import Aesop" not in base_hdr:
        base_hdr = base_hdr.replace("import Mathlib", "import Mathlib\nimport Aesop", 1)
    dedup_lines = []
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
    env["ELAN_HOME"] = str(ELAN_HOME)
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir="/dev/shm", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        p = tf.name
    try:
        r = subprocess.run([str(LAKE_BIN), "env", "lean", "--threads=1", p], cwd=str(LEAN_WS), env=env, capture_output=True, text=True, timeout=25)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        return (r.returncode == 0 and "error:" not in out), clean_hdr, decl_prefix
    except Exception:
        return False, clean_hdr, "timeout"
    finally:
        if os.path.exists(p):
            os.unlink(p)


def _lean_prompt_messages(decl_prefix: str, lean_header: str, informal: str = "") -> list[dict[str, str]]:
    inf_block = f"Informal statement:\n{informal.strip()}\n\n" if informal.strip() else ""
    user_prompt = (
        "Complete the following Lean 4 theorem proof using Mathlib. "
        "Output ONLY a single fenced ```lean4 code block containing the proof body (starting with `by`) "
        "or the exact theorem with its proof, with no `sorry` or `admit`.\n\n"
        f"{inf_block}"
        f"```lean4\n{lean_header.strip()}\n\n{decl_prefix.strip()} := by\n  sorry\n```"
    )
    return [{"role": "user", "content": user_prompt}]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            ln = json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
            fh.write(ln)
            h.update(ln.encode("utf-8"))
    return {"path": str(path.relative_to(ROOT)), "count": len(rows), "sha256": h.hexdigest()}


def main() -> int:
    out_dir = ROOT / "data/splits_v4_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect blocklist across ALL existing data/*.jsonl files in V4 (which include V1, V2, V3, corpora_v4)
    blocked_ids: set[str] = set()
    blocked_math_alnum: set[str] = set()
    blocked_math_skel: set[str] = set()
    blocked_13grams: set[str] = set()
    blocked_lean_shas: set[str] = set()

    existing_jsonl = [p for p in (ROOT / "data").rglob("*.jsonl") if "splits_v4_1" not in str(p)]
    for fp in sorted(existing_jsonl):
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                for ik in ("id", "problem_id", "cluster_id"):
                    if r.get(ik):
                        blocked_ids.add(str(r[ik]))
                if r.get("formal_statement"):
                    blocked_lean_shas.add(_norm_lean_stmt_sha(str(r["formal_statement"])))
                prob = str(r.get("problem") or r.get("informal_statement") or "")
                if not prob and r.get("messages"):
                    u = str(r["messages"][0].get("content", ""))
                    if "```lean4" not in u:
                        prob = u.split("\n\nSolve the problem", 1)[0]
                if prob:
                    blocked_math_alnum.add(_alnum_sha(prob))
                    if len(prob) >= 40:
                        blocked_math_skel.add(_skeleton_sha(prob))
                    for ng in _ngrams_13(prob):
                        blocked_13grams.add(ng)

    print(
        f"[1/4] Loaded blocklist from {len(existing_jsonl)} V1-V4 files: "
        f"ids={len(blocked_ids)}, lean_shas={len(blocked_lean_shas)}, math_alnum={len(blocked_math_alnum)}",
        flush=True,
    )

    # 2. Build Track M Confirm_V4_1 (150 = 50x3) & Holdout_V4_1 (150 = 50x3) from untouched DeepMath-103K
    dm = load_dataset("zwhe99/DeepMath-103K", split="train")
    dm_order = sorted(range(len(dm)), key=lambda i: _sha256_str(f"{SALT}:math_scan:{i}"))
    math_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_math_skels: set[str] = set()

    for idx in dm_order:
        if all(len(math_buckets[s]) >= 100 for s in ("algebra_analysis", "nt_combinatorics", "geometry_discrete")):
            break
        r = dm[idx]
        diff = float(r.get("difficulty") or 0.0)
        if not (7.0 <= diff <= 9.5):
            continue
        strat = _classify_math_stratum(str(r.get("topic", "")))
        if len(math_buckets[strat]) >= 100:
            continue
        prob = str(r.get("question", "")).strip()
        ans = str(r.get("final_answer", "")).strip()
        iid = f"v41_dm:{idx}"
        if len(prob) < 35 or iid in blocked_ids or f"deepmath_eval:{idx}" in blocked_ids or f"v3_m_rl:{idx}" in blocked_ids or f"v3_m_s1:{idx}" in blocked_ids or f"v3_m_s2:{idx}" in blocked_ids:
            continue
        if not _is_scorable_math_gold(ans):
            continue
        al = _alnum_sha(prob)
        sk = _skeleton_sha(prob)
        if al in blocked_math_alnum or sk in blocked_math_skel or sk in seen_math_skels:
            continue
        if any(ng in blocked_13grams for ng in _ngrams_13(prob)):
            continue
        seen_math_skels.add(sk)
        math_buckets[strat].append(
            {
                "id": iid,
                "user_id": iid,
                "item_id": iid,
                "cluster_id": f"mclust_{sk[:16]}",
                "stratum": strat,
                "source": "deepmath_v4_1_unexposed",
                "difficulty": diff,
                "problem": prob,
                "gold": ans,
                "integer_task": bool(re.fullmatch(r"[+-]?\d+", ans)),
                "messages": [{"role": "user", "content": prob + MATH_SUFFIX}],
            }
        )

    math_confirm150, math_holdout150 = [], []
    for s in ("algebra_analysis", "nt_combinatorics", "geometry_discrete"):
        b = sorted(math_buckets[s], key=lambda x: _sha256_str(f"{SALT}:{x['cluster_id']}"))
        assert len(b) == 100, f"Expected 100 math items in {s}, got {len(b)}"
        for it in b[:50]:
            it_c = dict(it); it_c["ts"] = 2026112401
            math_confirm150.append(it_c)
        for it in b[50:100]:
            it_h = dict(it); it_h["ts"] = 2026112402
            math_holdout150.append(it_h)
    math_confirm150.sort(key=lambda x: x["id"])
    math_holdout150.sort(key=lambda x: x["id"])
    print(f"[2/4] Built Track M splits_v4_1: Confirm={len(math_confirm150)}, Holdout={len(math_holdout150)}", flush=True)

    # 3. Build Track L Confirm_V4_1 (150 = 50x3) & Holdout_V4_1 (150 = 50x3) from untouched Goedel-Pset-v1
    pset = load_dataset("Goedel-LM/Goedel-Pset-v1", split="train")
    pset_order = sorted(range(len(pset)), key=lambda i: _sha256_str(f"{SALT}:lean_scan:{i}"))

    # Pre-filter by stratum & blocklist before parallel Lean 4.24 kernel elaboration
    cand_by_strat: dict[str, list[tuple[int, dict[str, Any], str, str, str]]] = collections.defaultdict(list)
    seen_lean_shas: set[str] = set()
    for idx in pset_order:
        if all(len(cand_by_strat[s]) >= 280 for s in ("algebra", "nt_combinatorics_geom", "analysis_calculus")):
            break
        r = pset[idx]
        pid_raw = str(r.get("problem_id", idx))
        if f"goedel_pset_search:{pid_raw}" in blocked_ids or pid_raw in blocked_ids:
            continue
        fs = str(r.get("formal_statement", ""))
        inf = str(r.get("informal_statement", "")).strip()
        if not fs.strip():
            continue
        hdr, decl = _split_lean_hdr_and_decl(fs)
        lsha = _norm_lean_stmt_sha(decl)
        if lsha in blocked_lean_shas or lsha in seen_lean_shas:
            continue
        if inf and len(inf) >= 40 and any(ng in blocked_13grams for ng in _ngrams_13(inf)):
            continue
        strat = _classify_lean_stratum(decl, inf)
        if len(cand_by_strat[strat]) >= 280:
            continue
        seen_lean_shas.add(lsha)
        cand_by_strat[strat].append((idx, dict(r), hdr, decl, lsha))

    lean_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for strat in ("algebra", "nt_combinatorics_geom", "analysis_calculus"):
        cands = cand_by_strat[strat]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            elabs = list(ex.map(lambda c: _elaborate_lean4((c[2], c[3])), cands))
        for (idx, r, _hdr, _decl, lsha), (ok, clean_hdr, decl_prefix) in zip(cands, elabs):
            if not ok or len(lean_buckets[strat]) >= 100:
                continue
            stmt = decl_prefix.strip() + " := by"
            inf = str(r.get("informal_statement", "")).strip()
            iid = f"v41_pset:{r.get('problem_id', idx)}"
            lean_buckets[strat].append(
                {
                    "id": iid,
                    "user_id": iid,
                    "item_id": iid,
                    "cluster_id": f"lclust_{lsha[:16]}",
                    "stratum": strat,
                    "source": "goedel_pset_v4_1_unexposed",
                    "lean_header": clean_hdr,
                    "formal_statement": stmt,
                    "informal_statement": inf,
                    "messages": _lean_prompt_messages(decl_prefix, clean_hdr, inf),
                }
            )
        print(f"  Track L stratum `{strat}` elaborated: {len(lean_buckets[strat])}/100", flush=True)
        if len(lean_buckets[strat]) < 100:
            raise RuntimeError(f"Insufficient elaborated Lean theorems in {strat}: {len(lean_buckets[strat])}")

    lean_confirm150, lean_holdout150 = [], []
    for s in ("algebra", "nt_combinatorics_geom", "analysis_calculus"):
        b = sorted(lean_buckets[s], key=lambda x: _sha256_str(f"{SALT}:{x['cluster_id']}"))
        for it in b[:50]:
            it_c = dict(it); it_c["ts"] = 2026112401
            lean_confirm150.append(it_c)
        for it in b[50:100]:
            it_h = dict(it); it_h["ts"] = 2026112402
            lean_holdout150.append(it_h)
    lean_confirm150.sort(key=lambda x: x["id"])
    lean_holdout150.sort(key=lambda x: x["id"])
    print(f"[3/4] Built Track L splits_v4_1: Confirm={len(lean_confirm150)}, Holdout={len(lean_holdout150)}", flush=True)

    receipts = {
        "salt": SALT,
        "blocked_source_files_count": len(existing_jsonl),
        "blocked_ids_count": len(blocked_ids),
        "blocked_lean_shas_count": len(blocked_lean_shas),
        "blocked_math_alnum_count": len(blocked_math_alnum),
        "lean_confirm150": _write_jsonl(out_dir / "v4_1_lean_confirm150.jsonl", lean_confirm150),
        "lean_holdout150": _write_jsonl(out_dir / "v4_1_lean_holdout150.jsonl", lean_holdout150),
        "math_confirm150": _write_jsonl(out_dir / "v4_1_math_confirm150.jsonl", math_confirm150),
        "math_holdout150": _write_jsonl(out_dir / "v4_1_math_holdout150.jsonl", math_holdout150),
    }

    # 4. Run governed_loop_guard_v39.py audit-splits on both tracks
    guard = ROOT / "eval_v4_1/governed_loop_guard_v39.py"
    for track, s_val, c_p, h_p in [
        ("lean", ROOT / "data/splits_v3/v3_lean_search140.jsonl", out_dir / "v4_1_lean_confirm150.jsonl", out_dir / "v4_1_lean_holdout150.jsonl"),
        ("math", ROOT / "data/splits_v3/v3_math_search160.jsonl", out_dir / "v4_1_math_confirm150.jsonl", out_dir / "v4_1_math_holdout150.jsonl"),
    ]:
        cmd = [
            sys.executable, "-I", str(guard), "audit-splits",
            "--train", str(s_val),
            "--confirm", str(c_p),
            "--holdout", str(h_p),
            "--group-key", "id",
            "--item-key", "id",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        receipts[f"guard_v39_audit_{track}"] = json.loads(res.stdout)

    (out_dir / "v4_1_split_audit_manifest.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    print("[4/4] v3.9 audit-splits PASSED & v4_1_split_audit_manifest.json sealed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

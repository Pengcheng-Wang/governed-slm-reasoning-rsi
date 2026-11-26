#!/usr/bin/env python3
"""
Phase V4 Hardened Out-of-Process Lean 4.24 (`Mathlib f897ebcf`) Verifier (`B1` / `B3`).

Preserves 100% of Phase V1-V3 Lean 4.24 kernel & `#print axioms <thm>` soundness checks
(`{propext, Classical.choice, Quot.sound}`, zero `sorry`/`admit`/`native_decide`/block comments)
while normalizing cross-family reasoning channels (`inkling2`, `gemma4`, `muse_glimmer`, `qwen`)
via `channel_normalizer_v4.py` before proof-body extraction.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EVAL_V4_DIR = Path(__file__).resolve().parent
if str(EVAL_V4_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V4_DIR))

from channel_normalizer_v4 import normalize_reasoning_channel  # noqa: E402

MATHLIB_HEADER = """import Mathlib
import Aesop
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
"""

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

FENCED_LEAN_STRICT = re.compile(r"^\s*```(?:lean4|lean)\s*\n(?P<code>.*?)```\s*$", re.DOTALL)
FENCED_LEAN_ANY = re.compile(r"```(?:lean4|lean)?\s*\n(?P<code>.*?)```", re.DOTALL)

FORBIDDEN_TOKENS_RE = re.compile(
    r"(?:"
    r"\b(?:"
    r"sorry|admit|sorryAx|native_decide|ofReduceBool|trustCompiler|skipKernelTC|"
    r"axiom|constant|opaque|unsafe|partial|extern|implemented_by|run_tac|run_cmd|"
    r"elab|macro|macro_rules|syntax|notation|infix|infixl|infixr|prefix|postfix|"
    r"local|scoped|attribute|hint"
    r")\b"
    r"|(?:exact|apply|rw|simp|aesop|solve_by_elim|by_contra|polyrith|linarith)\?"
    r")"
)

FORBIDDEN_LINE_START_RE = re.compile(
    r"(?m)^\s*(?:"
    r"import|set_option|namespace|section|end|open|universe|variable|variables|"
    r"theorem|lemma|example|def|abbrev|instance|structure|class|inductive"
    r")\b"
)

SORRY_DIAG_RE = re.compile(r"declaration uses [`'\"]?sorry[`'\"]?|sorryAx")
HEADER_DECL_NAME_RE = re.compile(
    r"^\s*(?:noncomputable\s+)?(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'\u2080-\u2089\.\!\?]*)",
    re.MULTILINE,
)


class CandidateProofError(ValueError):
    pass


def _canonical_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _strip_line_comments(code: str) -> str:
    out_lines: list[str] = []
    for line in code.splitlines():
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def split_frozen_statement(formal_statement: str) -> tuple[str, str]:
    stmt = formal_statement.strip()
    lines = stmt.splitlines()
    decl_lines: list[str] = []
    started = False
    for ln in lines:
        s = ln.lstrip()
        if not started:
            if not s or s.startswith(("import ", "open ", "set_option ", "--")):
                continue
            started = True
        decl_lines.append(ln)
    clean_stmt = "\n".join(decl_lines).strip()
    m = HEADER_DECL_NAME_RE.search(clean_stmt)
    if not m:
        raise ValueError(f"Benchmark formal_statement lacks theorem/lemma name: {formal_statement[:120]!r}")
    thm_name = m.group(1)

    m_sorry = re.search(r":=\s*(?:by\s+)?sorry\s*$", clean_stmt)
    if m_sorry:
        decl_prefix = clean_stmt[: m_sorry.start()].rstrip()
    elif clean_stmt.endswith(":="):
        decl_prefix = clean_stmt[:-2].rstrip()
    elif ":=" in clean_stmt:
        decl_prefix = clean_stmt.rsplit(":=", 1)[0].rstrip()
    else:
        raise ValueError(f"Benchmark formal_statement lacks ':=' separator: {formal_statement[:120]!r}")
    return thm_name, decl_prefix


def extract_and_sanitize_proof_body(
    raw_text: str,
    formal_statement: str,
    *,
    thinking: bool = False,
    channel_family: str = "auto",
    strict: bool = False,
) -> str:
    norm = normalize_reasoning_channel(raw_text, channel_family=channel_family, expect_thinking=thinking)
    if norm.has_template_leak:
        raise CandidateProofError("raw_template_marker")
    if norm.has_unclosed_thinking:
        raise CandidateProofError("unclosed_thinking_channel")

    after = norm.final_text
    if not after:
        raise CandidateProofError("empty_final_channel")

    code: str | None = None
    if strict:
        m = FENCED_LEAN_STRICT.match(after)
        if m is None:
            raise CandidateProofError("strict_requires_single_fenced_lean_block")
        code = m.group("code").strip()
    else:
        matches = list(FENCED_LEAN_ANY.finditer(after))
        if matches:
            code = matches[-1].group("code").strip()
        else:
            stripped = after.strip()
            if stripped.startswith("by ") or stripped.startswith("by\n") or ":=" in stripped:
                code = stripped
            else:
                raise CandidateProofError("no_lean_block_or_tactic_found")

    if not code:
        raise CandidateProofError("empty_candidate")
    if len(code) > 24000:
        raise CandidateProofError("candidate_exceeds_char_cap")

    if "/-" in code or "-/" in code:
        raise CandidateProofError("block_comments_forbidden")

    uncommented = _strip_line_comments(code).strip()
    if not uncommented:
        raise CandidateProofError("comment_only_candidate")

    thm_name, decl_prefix = split_frozen_statement(formal_statement)
    body_lines: list[str] = []
    started_decl = False
    has_top_decl = bool(re.search(r"(?m)^\s*(?:noncomputable\s+)?(?:theorem|lemma)\b", uncommented))
    if has_top_decl:
        for ln in uncommented.splitlines():
            s = ln.lstrip()
            if not started_decl and (not s or s.startswith(("import Mathlib", "import Aesop", "open ", "set_option maxHeartbeats"))):
                continue
            started_decl = True
            body_lines.append(ln)
        candidate_decl = "\n".join(body_lines).strip()
        if ":=" not in candidate_decl:
            raise CandidateProofError("echoed_theorem_missing_colon_eq")
        cand_sig, cand_rhs = candidate_decl.split(":=", 1)
        if _canonical_ws(cand_sig) != _canonical_ws(decl_prefix):
            expected_canon = _canonical_ws(decl_prefix)
            full_canon = _canonical_ws(candidate_decl)
            if not full_canon.startswith(expected_canon + " :="):
                raise CandidateProofError("theorem_signature_tampered")
            tokens = re.split(r"(\s+)", decl_prefix.strip())
            pat = r"^\s*" + r"\s+".join(re.escape(t) for t in tokens if t.strip()) + r"\s*:="
            m_pre = re.match(pat, candidate_decl)
            if not m_pre:
                raise CandidateProofError("theorem_signature_parse_mismatch")
            cand_rhs = candidate_decl[m_pre.end() :]
        proof_body = cand_rhs.strip()
    else:
        proof_body = uncommented

    if not proof_body:
        raise CandidateProofError("empty_proof_body")

    if re.search(r"(?m)(?:^|\s)#[A-Za-z_]", proof_body):
        raise CandidateProofError("hash_command_forbidden")

    m_tok = FORBIDDEN_TOKENS_RE.search(proof_body)
    if m_tok:
        raise CandidateProofError(f"forbidden_token:{m_tok.group(0)}")

    m_line = FORBIDDEN_LINE_START_RE.search(proof_body)
    if m_line:
        raise CandidateProofError(f"forbidden_top_directive:{m_line.group(0).strip()}")

    if not (proof_body == "by" or proof_body.startswith(("by ", "by\n", "by\t", "fun ", "λ ", "⟨", "("))):
        tactic_starters = (
            "intro", "intros", "rintro", "exact", "apply", "refine", "have", "let", "suffices",
            "obtain", "rcases", "cases", "induction", "simp", "simpa", "norm_num", "ring", "ring_nf",
            "linarith", "nlinarith", "positivity", "omega", "decide", "aesop", "tauto", "abel",
            "field_simp", "push_neg", "contrapose", "by_contra", "by_cases", "constructor", "left",
            "right", "use", "exists", "rw", "erw", "calc", "congr", "ext", "funext", "gcongr", "polyrith",
            "norm_cast", "push_cast", "exact_mod_cast", "assumption", "trivial", "contradiction",
        )
        first_tok = proof_body.split()[0].rstrip(";")
        if "\n" in proof_body or first_tok in tactic_starters:
            proof_body = "by\n  " + "\n  ".join(proof_body.splitlines())

    return proof_body


def classify_lean_error(err_text: str) -> str:
    low = err_text.lower()
    if "unknown identifier" in low or "unknown constant" in low:
        return "UNKNOWN_IDENT"
    if "type mismatch" in low or "application type mismatch" in low:
        return "TYPE_MISMATCH"
    if "unsolved goals" in low:
        return "UNSOLVED_GOALS"
    if "tactic" in low and "failed" in low:
        return "TACTIC_FAILED"
    if "expected" in low or "unexpected token" in low or "parse" in low:
        return "SYNTAX_ERROR"
    if "maximum recursion" in low or "deterministic timeout" in low or "heartbeat" in low:
        return "HEARTBEAT_LIMIT"
    return "LEAN_COMPILE_ERROR"


def verify_lean_candidate(
    raw_text: str,
    formal_statement: str,
    *,
    lean_project: Path,
    elan_bin: Path,
    finish_reason: str = "stop",
    thinking: bool = False,
    channel_family: str = "auto",
    strict_format: bool = False,
    timeout_s: float = 60.0,
    lean_header: str | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    if finish_reason != "stop":
        return {
            "ok": False,
            "error_class": "TRUNCATED" if finish_reason == "length" else f"NON_STOP:{finish_reason}",
            "diagnostics": f"Generation did not terminate cleanly (finish_reason={finish_reason})",
            "proof_body": None,
            "axioms": [],
            "sft_clean_admissible": False,
            "elapsed_s": time.monotonic() - t0,
        }

    try:
        thm_name, decl_prefix = split_frozen_statement(formal_statement)
    except ValueError as exc:
        return {
            "ok": False,
            "error_class": "BENCHMARK_STATEMENT_INVALID",
            "diagnostics": str(exc),
            "proof_body": None,
            "axioms": [],
            "sft_clean_admissible": False,
            "elapsed_s": time.monotonic() - t0,
        }

    strict_body: str | None = None
    try:
        strict_body = extract_and_sanitize_proof_body(
            raw_text, formal_statement, thinking=thinking, channel_family=channel_family, strict=True
        )
    except CandidateProofError:
        strict_body = None

    try:
        proof_body = (
            strict_body
            if (strict_format or strict_body is not None)
            else extract_and_sanitize_proof_body(
                raw_text, formal_statement, thinking=thinking, channel_family=channel_family, strict=False
            )
        )
    except CandidateProofError as exc:
        return {
            "ok": False,
            "error_class": f"FORMAT:{exc}",
            "diagnostics": f"FORMAT:{exc}",
            "proof_body": None,
            "axioms": [],
            "sft_clean_admissible": False,
            "elapsed_s": time.monotonic() - t0,
        }

    if proof_body is None:
        return {
            "ok": False,
            "error_class": "FORMAT:strict_extraction_failed",
            "diagnostics": "FORMAT:strict_extraction_failed",
            "proof_body": None,
            "axioms": [],
            "sft_clean_admissible": False,
            "elapsed_s": time.monotonic() - t0,
        }

    header = (lean_header or MATHLIB_HEADER).strip()
    rendered_source = (
        f"{header}\n\n"
        f"{decl_prefix} :=\n"
        f"{proof_body}\n\n"
        f"#print axioms {thm_name}\n"
    )

    clean_env = {
        "PATH": f"{elan_bin}:/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "ELAN_HOME": str(Path(elan_bin).resolve().parent),
        "LANG": "C.UTF-8",
    }

    proc: subprocess.Popen[str] | None = None
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", encoding="utf-8", delete=False) as fh:
            fh.write(rendered_source)
            tmp_path = fh.name

        proc = subprocess.Popen(
            ["lake", "env", "lean", tmp_path],
            cwd=str(lean_project),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=clean_env,
            start_new_session=(sys.platform != "win32"),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return {
                "ok": False,
                "error_class": "TIMEOUT",
                "diagnostics": f"Lean compilation timed out after {timeout_s:.1f}s",
                "proof_body": proof_body,
                "axioms": [],
                "sft_clean_admissible": False,
                "elapsed_s": time.monotonic() - t0,
            }

        combined = ((stdout or "") + "\n" + (stderr or "")).replace(tmp_path, "<candidate>.lean").strip()
        has_error = bool(re.search(r"^(?:.*:\d+:\d+:\s*)?error:", combined, re.MULTILINE))
        has_sorry = bool(SORRY_DIAG_RE.search(combined))

        dep_pat = re.compile(
            rf"^'{re.escape(thm_name)}' depends on axioms: \[([^\]]*)\]\s*$",
            re.MULTILINE,
        )
        no_ax_pat = re.compile(
            rf"^'{re.escape(thm_name)}' does not depend on any axioms\s*$",
            re.MULTILINE,
        )
        dep_matches = dep_pat.findall(combined)
        no_ax_matches = no_ax_pat.findall(combined)
        audit_count = len(dep_matches) + len(no_ax_matches)

        used_axioms: list[str] = []
        for grp in dep_matches:
            for ax in grp.split(","):
                ax_s = ax.strip()
                if ax_s:
                    used_axioms.append(ax_s)

        bad_axioms = sorted({a for a in used_axioms if a not in ALLOWED_AXIOMS})
        ok = (
            proc.returncode == 0
            and not has_error
            and not has_sorry
            and audit_count == 1
            and len(bad_axioms) == 0
        )

        if ok:
            err_cls = "OK"
            diag_short = ""
        elif has_sorry:
            err_cls = "SORRY_DETECTED"
            diag_short = "Proof depends on sorry/sorryAx"
        elif bad_axioms:
            err_cls = "BAD_AXIOMS:" + ",".join(bad_axioms[:4])
            diag_short = f"Disallowed axioms: {bad_axioms}"
        elif proc.returncode == 0 and not has_error and audit_count != 1:
            err_cls = f"AXIOM_AUDIT_MISMATCH:count={audit_count}"
            diag_short = f"Expected exact 1 axiom audit for {thm_name}, got {audit_count}"
        else:
            err_cls = classify_lean_error(combined)
            err_idx = combined.find("error:")
            diag_short = combined[max(0, err_idx - 120) : err_idx + 1200] if err_idx >= 0 else combined[:1200]

        return {
            "ok": ok,
            "error_class": err_cls,
            "diagnostics": diag_short,
            "proof_body": proof_body,
            "axioms": sorted(set(used_axioms)),
            "sft_clean_admissible": bool(ok and strict_body is not None),
            "elapsed_s": time.monotonic() - t0,
        }
    finally:
        if proc is not None and proc.poll() is None:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

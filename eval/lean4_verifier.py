#!/usr/bin/env python3
"""
Hardened Out-of-Process Lean 4.24 Verifier (`B1` / `B3`).

Security & soundness invariants:
  1. Benchmark owns the exact frozen header and theorem declaration up to `:=`.
  2. Candidate output is stripped of `<think>...</think>` reasoning, parsed for a Lean code block
     (strict: single fenced ```lean4 / ```lean block; lenient: last fenced block or `:= by` tail),
     and reduced strictly to the proof body.
  3. Block comments (`/-`, `-/`) and `#` commands are forbidden in candidate proof bodies so a
     candidate can NEVER comment out or spoof the trailing `#print axioms <thm>` kernel audit.
  4. Scope, syntax, declaration, and soundness escapes (`sorry`, `admit`, `sorryAx`, `native_decide`,
     `axiom`, `constant`, `opaque`, `unsafe`, `partial`, `extern`, `implemented_by`, `run_tac`,
     `elab`, `macro`, `syntax`, `notation`, `infix`, `prefix`, `postfix`, `local`, `scoped`,
     `attribute`, `namespace`, `section`, `open`, `import`, `set_option`, nested top-level
     `theorem`/`lemma`/`example`/`def`/`instance`) are rejected pre-compile AND audited post-compile.
  5. Appends `#print axioms <EXACT_THM_NAME>` and requires `returncode == 0`, zero `"error:"`,
     zero `sorry`/`sorryAx` diagnostics, and an EXACT single axiom-audit line for `<EXACT_THM_NAME>`
     whose transitive axioms are a subset of `{propext, Classical.choice, Quot.sound}`.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

MATHLIB_HEADER = """import Mathlib
import Aesop
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
"""

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

RAW_TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<bos>", "</s>")

FENCED_LEAN_STRICT = re.compile(r"^\s*```(?:lean4|lean)\s*\n(?P<code>.*?)```\s*$", re.DOTALL)
FENCED_LEAN_ANY = re.compile(r"```(?:lean4|lean)?\s*\n(?P<code>.*?)```", re.DOTALL)

FORBIDDEN_TOKENS_RE = re.compile(
    r"\b(?:"
    r"sorry|admit|sorryAx|native_decide|ofReduceBool|trustCompiler|"
    r"axiom|constant|opaque|unsafe|partial|extern|implemented_by|run_tac|"
    r"elab|macro|macro_rules|syntax|notation|infix|infixl|infixr|prefix|postfix|"
    r"local|scoped|attribute"
    r")\b"
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
    """
    Extracts `(theorem_name, frozen_decl_prefix_up_to_colon_eq)` from the benchmark's
    `formal_statement` (which ends with `:= sorry`, `:= by sorry`, or `:=`).
    """
    stmt = formal_statement.strip()
    # Strip any leading import/open/set_option lines if present in raw benchmark statement
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

    # Split at the theorem declaration's outer `:=` (last `:=` before `sorry` or end)
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
    thinking: bool,
    strict: bool,
) -> str:
    """
    Extracts ONLY the proof body (typically `by ...` or a proof term), verifies zero forbidden
    constructs or signature tampering, and returns a clean body ready for splicing after `<decl_prefix> :=`.
    """
    if any(tok in raw_text for tok in RAW_TEMPLATE_MARKERS):
        raise CandidateProofError("raw_template_marker")

    if thinking:
        if raw_text.count("</think>") != 1:
            if strict or "</think>" not in raw_text:
                raise CandidateProofError("missing_think_close")
        after = raw_text.rsplit("</think>", 1)[1]
    else:
        # Qwen3.5 direct mode may emit an empty `<think>\n\n</think>` prefix or pure text
        if "</think>" in raw_text:
            after = raw_text.rsplit("</think>", 1)[1]
        elif "<think>" in raw_text:
            raise CandidateProofError("unclosed_think_tag")
        else:
            after = raw_text

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

    # Block `/-` and `-/` BEFORE stripping `--` comments so block comments cannot hide or swallow `#print axioms`
    if "/-" in code or "-/" in code:
        raise CandidateProofError("block_comments_forbidden")

    uncommented = _strip_line_comments(code).strip()
    if not uncommented:
        raise CandidateProofError("comment_only_candidate")

    # Strip leading benign `import Mathlib` / `open ...` ONLY if model echoed the full theorem with identical signature
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
            raise CandidateProofError(" echoed_theorem_missing_colon_eq")
        cand_sig, cand_rhs = candidate_decl.split(":=", 1)
        if _canonical_ws(cand_sig) != _canonical_ws(decl_prefix):
            # Check if `:=' split hit an inner `let x :=` inside type binders; match prefix canonically
            expected_canon = _canonical_ws(decl_prefix)
            full_canon = _canonical_ws(candidate_decl)
            if not full_canon.startswith(expected_canon + " :="):
                raise CandidateProofError("theorem_signature_tampered")
            # Locate the exact `:=` corresponding to `decl_prefix`
            # Match token-by-token through `candidate_decl`
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

    # Reject `#` commands anywhere (`#print`, `#check`, `#eval`, `#synth`, `#reduce`)
    if re.search(r"(?m)(?:^|\s)#[A-Za-z_]", proof_body):
        raise CandidateProofError("hash_command_forbidden")

    m_tok = FORBIDDEN_TOKENS_RE.search(proof_body)
    if m_tok:
        raise CandidateProofError(f"forbidden_token:{m_tok.group(0)}")

    m_line = FORBIDDEN_LINE_START_RE.search(proof_body)
    if m_line:
        raise CandidateProofError(f"forbidden_top_directive:{m_line.group(0).strip()}")

    # If the model wrote raw tactic lines (e.g., `norm_num` or `intro h\nexact h`) without leading `by`,
    # distinguish direct term proof vs tactic block: in miniF2F/Workbook, bare tactic lines start with a tactic keyword.
    if not (proof_body == "by" or proof_body.startswith(("by ", "by\n", "by\t", "fun ", "λ ", "⟨", "("))):
        # Check common tactic starters; wrap with `by\n` if multi-line or standard tactic
        tactic_starters = (
            "intro", "intros", "rintro", "exact", "apply", "refine", "have", "let", "suffices",
            " obtain", "rcases", "cases", "induction", "simp", "simpa", "norm_num", "ring", "ring_nf",
            "linarith", "nlinarith", "positivity", "omega", "decide", "aesop", "tauto", "abel",
            "field_simp", "push_neg", "contrapose", "by_contra", "by_cases", "constructor", "left",
            "right", "use", "exists", "rw", "erw", "calc", "congr", "ext", "funext", "gcongr", "polyrith",
            "norm_cast", "push_cast", "exact_mod_cast", "assumption", "trivial", " contradiction",
        )
        first_tok = proof_body.split()[0].rstrip(";")
        if "\n" in proof_body or first_tok in {t.strip() for t in tactic_starters}:
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
    strict_format: bool = False,
    timeout_s: float = 60.0,
    lean_header: str | None = None,
) -> dict[str, Any]:
    """
    Compiles `<FROZEN_HEADER> + <FROZEN_DECL_PREFIX> := <SANITIZED_PROOF_BODY> + #print axioms <THM_NAME>`
    out-of-process and verifies full kernel & transitive-axiom soundness.
    """
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

    # Check strict formatting independently for SFT/ReST^EM hygiene tracking
    strict_body: str | None = None
    try:
        strict_body = extract_and_sanitize_proof_body(raw_text, formal_statement, thinking=thinking, strict=True)
    except CandidateProofError:
        strict_body = None

    try:
        proof_body = (
            strict_body
            if (strict_format or strict_body is not None)
            else extract_and_sanitize_proof_body(raw_text, formal_statement, thinking=thinking, strict=False)
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", encoding="utf-8", delete=False, dir="/tmp") as fh:
            fh.write(rendered_source)
            tmp_path = fh.name

        proc = subprocess.Popen(
            ["lake", "env", "lean", tmp_path],
            cwd=str(lean_project),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=clean_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
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
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

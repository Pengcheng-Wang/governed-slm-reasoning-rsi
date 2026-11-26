#!/usr/bin/env python3
"""
Phase V4 Hardened Symbolic & Integer Competition Math Verifier (`B1` / `B3`).

Preserves 100% of Phase V1-V3 mathematical equivalence grading (`math_verify` + `SymPy`,
ordered-tuple vs finite-set guard, exact integer equivalence, percent/degree normalization)
while routing raw completions through `channel_normalizer_v4.normalize_reasoning_channel`
so `Inkling-2-Mini`, `Gemma 4`, `Muse Glimmer`, and `Qwen` are scored identically on their
extracted post-thinking final responses.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

EVAL_V4_DIR = Path(__file__).resolve().parent
if str(EVAL_V4_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V4_DIR))

from channel_normalizer_v4 import normalize_reasoning_channel  # noqa: E402

FORBIDDEN_SHORTCUT_RE = re.compile(r"\b(?:sorry|admit|native_decide)\b|os\._exit|sys\.exit")

WORD_NUMBERS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        start, depth = match.end(), 1
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{" and (idx == 0 or text[idx - 1] != "\\"):
                depth += 1
            elif ch == "}" and (idx == 0 or text[idx - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    ans = text[start:idx].strip()
                    if ans:
                        return ans
                    break
    unbraced = list(re.finditer(r"\\(?:boxed|fbox)\s+([+-]?\d+(?:\.\d+)?)\b", text))
    if unbraced:
        return unbraced[-1].group(1).strip()
    return None


def extract_answer_strict_v4(
    raw_text: str,
    *,
    channel_family: str = "auto",
    thinking: bool = False,
) -> tuple[str | None, Any]:
    norm = normalize_reasoning_channel(raw_text, channel_family=channel_family, expect_thinking=thinking)
    if norm.has_unclosed_thinking or not norm.final_text:
        return None, norm
    after = norm.final_text
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", after))
    for match in reversed(starts):
        start, depth = match.end(), 1
        for idx in range(start, len(after)):
            ch = after[idx]
            if ch == "{" and (idx == 0 or after[idx - 1] != "\\"):
                depth += 1
            elif ch == "}" and (idx == 0 or after[idx - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    ans = after[start:idx].strip()
                    if ans:
                        return ans, norm
                    break
    return None, norm


def extract_answer_lenient_v4(
    raw_text: str,
    *,
    channel_family: str = "auto",
    thinking: bool = False,
) -> str | None:
    norm = normalize_reasoning_channel(raw_text, channel_family=channel_family, expect_thinking=thinking)
    if norm.has_unclosed_thinking or not norm.final_text:
        return None
    after = norm.final_text

    boxed = last_boxed(after)
    if boxed is not None:
        cleaned = re.sub(r"^[a-zA-Z]\s*=\s*", "", boxed).strip()
        return cleaned if cleaned else boxed

    ans_lines = list(re.finditer(r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*([^\n]+)", after, re.IGNORECASE))
    if ans_lines:
        cand = ans_lines[-1].group(1).strip().rstrip(".").strip("$").strip()
        if cand:
            return cand
    return None


def exact_integer_equivalent(answer: str | None, gold: str) -> bool:
    gold_s = gold.strip()
    if not re.fullmatch(r"[+-]?\d+", gold_s):
        raise ValueError(f"Integer scorer requires exact integer gold, got {gold!r}")
    expected = int(gold_s)
    if answer is None:
        return False
    val = answer.strip().strip("$").strip()
    val = re.sub(r"^[a-zA-Z]\s*=\s*", "", val).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", val):
        return int(val.split(".", 1)[0]) == expected
    if re.fullmatch(r"[+-]?\d+\.\d+", val):
        return False
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", val):
        return int(val.replace(",", "")) == expected

    from math_verify import LatexExtractionConfig, parse
    from sympy import Expr, Float, Integer, simplify

    try:
        parsed = parse(
            rf"\boxed{{{val}}}",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            parsing_timeout=3,
        )
        for expr in parsed:
            if isinstance(expr, Expr) and not expr.has(Float) and expr.is_number:
                if simplify(expr - Integer(expected)) == 0:
                    return True
    except Exception:
        return False
    return False


def _normalize_pair(answer: str, gold: str, problem: str) -> tuple[str, str]:
    def _norm_one(v: str) -> str:
        v = v.strip().replace("\u00b0", r"^\circ")
        v = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", v)
        v = re.sub(r"\\left\s*", "", v)
        v = re.sub(r"\\right\s*", "", v)
        w = re.fullmatch(r"\\(?:mbox|text|mathrm)\{\s*([A-Za-z]+)\s*\}", v)
        key = w.group(1).casefold() if w else v.casefold()
        return WORD_NUMBERS.get(key, v)

    answer, gold = _norm_one(answer), _norm_one(gold)
    if re.search(r"\b(?:what|which)\s+(?:percentage|percent)\b", problem, re.IGNORECASE):
        answer = re.sub(r"\\?%\s*$", "", answer).strip()
        gold = re.sub(r"\\?%\s*$", "", gold).strip()
    else:
        answer = re.sub(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\\?%$", r"(\1)/100", answer)
        gold = re.sub(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\\?%$", r"(\1)/100", gold)

    if re.fullmatch(r"[+-]?\d+(?:,\s+[+-]?\d+)+", gold):
        if re.fullmatch(r"[+-]?\d+(?:,\s*[+-]?\d+)+", answer):
            answer = ", ".join(p.strip() for p in answer.split(","))
    return answer, gold


def _is_ordered_tuple_str(s: str) -> bool:
    s = s.strip()
    return bool(s.startswith("(") and s.endswith(")") and "," in s)


def grade_symbolic_math(answer: str | None, gold: str, *, problem: str = "") -> dict[str, Any]:
    if answer is None:
        return {"correct": False, "parse_failure": True, "verifier_error": None}

    from math_verify import LatexExtractionConfig, parse, verify
    from sympy import FiniteSet, Interval

    try:
        norm_ans, norm_gold = _normalize_pair(answer, gold, problem)

        if _is_ordered_tuple_str(norm_gold) and re.search(r"\bordered\s+(?:pair|triple|tuple|quadruple)\b", problem, re.IGNORECASE):
            if not _is_ordered_tuple_str(norm_ans):
                return {"correct": False, "parse_failure": False, "verifier_error": None}
            g_parts = [p.strip() for p in norm_gold[1:-1].split(",")]
            a_parts = [p.strip() for p in norm_ans[1:-1].split(",")]
            if len(g_parts) != len(a_parts):
                return {"correct": False, "parse_failure": False, "verifier_error": None}
            all_ok = True
            for gp, ap in zip(g_parts, a_parts):
                eg = parse(rf"\boxed{{{gp}}}", extraction_config=[LatexExtractionConfig()])
                ea = parse(rf"\boxed{{{ap}}}", extraction_config=[LatexExtractionConfig()])
                if not (eg and ea and verify(eg, ea)):
                    all_ok = False
                    break
            return {"correct": all_ok, "parse_failure": False, "verifier_error": None}

        expected = parse(rf"\boxed{{{norm_gold}}}", extraction_config=[LatexExtractionConfig()])
        actual = parse(rf"\boxed{{{norm_ans}}}", extraction_config=[LatexExtractionConfig()])
        if not expected:
            raise ValueError(f"Unparseable gold answer: {gold!r}")

        discrete_ref = isinstance(expected[0], FiniteSet)
        compat = not (discrete_ref and actual and isinstance(actual[0], Interval))
        ok = bool(
            actual
            and compat
            and verify(expected, actual, allow_set_relation_comp=not discrete_ref)
        )
        return {"correct": ok, "parse_failure": not bool(actual), "verifier_error": None}
    except Exception as exc:
        return {
            "correct": False,
            "parse_failure": True,
            "verifier_error": f"{type(exc).__name__}: {exc}",
        }


def score_math_response(
    text: str,
    gold: str,
    *,
    finish_reason: str,
    thinking: bool = False,
    channel_family: str = "auto",
    integer_task: bool = False,
    problem: str = "",
) -> dict[str, Any]:
    stopped = finish_reason == "stop"
    truncated = finish_reason == "length"
    if stopped:
        strict_ans, norm = extract_answer_strict_v4(text, channel_family=channel_family, thinking=thinking)
        lenient_ans = extract_answer_lenient_v4(text, channel_family=channel_family, thinking=thinking)
        if norm.has_unclosed_thinking or norm.has_template_leak:
            strict_ans = None
            lenient_ans = None
    else:
        norm = normalize_reasoning_channel(text, channel_family=channel_family, expect_thinking=thinking)
        strict_ans = None
        lenient_ans = None

    if integer_task:
        strict_ok = bool(stopped and strict_ans is not None and exact_integer_equivalent(strict_ans, gold))
        lenient_ok = bool(stopped and lenient_ans is not None and exact_integer_equivalent(lenient_ans, gold))
        strict_pf = bool(stopped and strict_ans is None)
        lenient_pf = bool(stopped and lenient_ans is None)
        verr = None
    else:
        gs = grade_symbolic_math(strict_ans, gold, problem=problem) if stopped else {"correct": False, "parse_failure": True, "verifier_error": None}
        gl = grade_symbolic_math(lenient_ans, gold, problem=problem) if stopped else {"correct": False, "parse_failure": True, "verifier_error": None}
        strict_ok = bool(stopped and gs["correct"])
        lenient_ok = bool(stopped and gl["correct"])
        strict_pf = bool(stopped and gs["parse_failure"])
        lenient_pf = bool(stopped and gl["parse_failure"])
        verr = gl["verifier_error"] or gs["verifier_error"]

    has_shortcut = bool(FORBIDDEN_SHORTCUT_RE.search(norm.final_text))
    sft_clean = bool(
        stopped
        and strict_ok
        and not norm.has_unclosed_thinking
        and not norm.has_template_leak
        and not has_shortcut
    )

    return {
        "finish_reason": finish_reason,
        "stopped": stopped,
        "truncated": truncated,
        "channel_family": norm.family,
        "strict_extracted": strict_ans,
        "lenient_extracted": lenient_ans,
        "strict_correct": strict_ok,
        "lenient_correct": lenient_ok,
        "strict_parse_failure": strict_pf,
        "lenient_parse_failure": lenient_pf,
        "verifier_error": verr,
        "sft_clean_admissible": sft_clean,
    }

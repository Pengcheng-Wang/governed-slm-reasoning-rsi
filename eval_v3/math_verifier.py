#!/usr/bin/env python3
"""
Hardened Symbolic & Integer Competition Math Verifier (`B1` / `B3`).

Decouples:
  - Strict SFT/ReST^EM admission (`finish_reason == 'stop'`, single `</think>` if thinking,
    balanced `\boxed{...}` after `</think>`, zero raw template/shortcut tokens)
  - Fault-tolerant lenient semantic scoring (`math_verify` + `SymPy`, ordered tuple vs finite-set
    disambiguation, exact integer equivalence for AIME/HMMT integer tasks, percent/degree handling).
"""

from __future__ import annotations

import re
from typing import Any

RAW_TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<bos>", "</s>")
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


def extract_answer_strict(text: str, *, thinking: bool) -> str | None:
    if thinking:
        if text.count("</think>") != 1:
            return None
        after = text.split("</think>", 1)[1]
    else:
        if "</think>" in text:
            after = text.rsplit("</think>", 1)[1]
        elif "<think>" in text:
            return None
        else:
            after = text
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
                        return ans
                    break
    return None


def extract_answer_lenient(text: str, *, thinking: bool) -> str | None:
    if thinking:
        if "</think>" not in text:
            return None
        after = text.rsplit("</think>", 1)[1]
    else:
        after = text.rsplit("</think>", 1)[1] if "</think>" in text else text

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

        # Ordered-tuple guard: if gold is `(a, b)` and problem asks for ordered pair/tuple,
        # reject unordered `{a, b}` or swapped `(b, a)` by checking elementwise when comma-split matches.
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


def verify_gold_parseable(gold: str, *, integer_task: bool) -> bool:
    try:
        if integer_task:
            return bool(re.fullmatch(r"[+-]?\d+", gold.strip()))
        from math_verify import LatexExtractionConfig, parse

        norm_g, _ = _normalize_pair(gold, gold, "")
        parsed = parse(rf"\boxed{{{norm_g}}}", extraction_config=[LatexExtractionConfig()])
        return bool(parsed)
    except Exception:
        return False


def score_math_response(
    text: str,
    gold: str,
    *,
    finish_reason: str,
    thinking: bool,
    integer_task: bool,
    problem: str = "",
) -> dict[str, Any]:
    stopped = finish_reason == "stop"
    truncated = finish_reason == "length"
    strict_ans = extract_answer_strict(text, thinking=thinking) if stopped else None
    lenient_ans = extract_answer_lenient(text, thinking=thinking) if stopped else None

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

    has_bad_markers = any(m in text for m in RAW_TEMPLATE_MARKERS) or bool(FORBIDDEN_SHORTCUT_RE.search(text))
    think_hygiene = (text.count("</think>") == 1 and "<think>" not in text.split("</think>", 1)[1]) if thinking else ("<think>" not in text or text.startswith("<think>\n\n</think>"))
    sft_clean = bool(stopped and strict_ok and not has_bad_markers and think_hygiene)

    return {
        "finish_reason": finish_reason,
        "stopped": stopped,
        "truncated": truncated,
        "strict_extracted": strict_ans,
        "lenient_extracted": lenient_ans,
        "strict_correct": strict_ok,
        "lenient_correct": lenient_ok,
        "strict_parse_failure": strict_pf,
        "lenient_parse_failure": lenient_pf,
        "verifier_error": verr,
        "sft_clean_admissible": sft_clean,
    }

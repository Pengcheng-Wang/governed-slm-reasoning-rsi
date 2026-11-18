#!/usr/bin/env python3
"""
12-case adversarial attack & positive-control unit suite for `eval/lean4_verifier.py`
plus 10-case symbolic/integer/tuple/set unit suite for `eval/math_verifier.py`.

Runs both offline parser/sanitizer checks AND live Lean 4.24.0 + Mathlib kernel tests when
`LEAN_PROJECT` & `ELAN_BIN` exist.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from lean4_verifier import (  # noqa: E402
    CandidateProofError,
    extract_and_sanitize_proof_body,
    split_frozen_statement,
    verify_lean_candidate,
)
from math_verifier import score_math_response, verify_gold_parseable  # noqa: E402


class TestMathVerifier(unittest.TestCase):
    def test_integer_equivalence_and_rejection(self) -> None:
        r_ok = score_math_response("<think>step</think>\n\\boxed{042}", "42", finish_reason="stop", thinking=True, integer_task=True)
        self.assertTrue(r_ok["strict_correct"])
        self.assertTrue(r_ok["lenient_correct"])
        self.assertTrue(r_ok["sft_clean_admissible"])

        r_dec = score_math_response("<think>step</think>\n\\boxed{42.5}", "42", finish_reason="stop", thinking=True, integer_task=True)
        self.assertFalse(r_dec["lenient_correct"])

        r_trunc = score_math_response("<think>rambling \\boxed{42}", "42", finish_reason="length", thinking=True, integer_task=True)
        self.assertFalse(r_trunc["lenient_correct"])
        self.assertFalse(r_trunc["sft_clean_admissible"])

    def test_box_only_inside_think_rejected(self) -> None:
        r = score_math_response("<think>guess \\boxed{7}</think>\nI am unsure.", "7", finish_reason="stop", thinking=True, integer_task=True)
        self.assertFalse(r["strict_correct"])
        self.assertFalse(r["lenient_correct"])
        self.assertFalse(r["sft_clean_admissible"])

    def test_finite_set_vs_interval_and_ordered_tuple(self) -> None:
        # Unordered finite set `{2, 5}` matches `{5, 2}`, rejects open interval `(2, 5)`
        r_set_ok = score_math_response("\\boxed{\\{5, 2\\}}", "\\{2, 5\\}", finish_reason="stop", thinking=False, integer_task=False)
        self.assertTrue(r_set_ok["lenient_correct"])

        r_int_bad = score_math_response("\\boxed{(2, 5)}", "\\{2, 5\\}", finish_reason="stop", thinking=False, integer_task=False)
        self.assertFalse(r_int_bad["lenient_correct"])

        # Ordered pair `(3, -1)` must NOT match `(-1, 3)` when problem specifies ordered pair
        prob = "Find the ordered pair (x, y) of real numbers satisfying..."
        r_ord_ok = score_math_response("\\boxed{(6/2, -1)}", "(3, -1)", finish_reason="stop", thinking=False, integer_task=False, problem=prob)
        self.assertTrue(r_ord_ok["lenient_correct"])

        r_ord_swap = score_math_response("\\boxed{(-1, 3)}", "(3, -1)", finish_reason="stop", thinking=False, integer_task=False, problem=prob)
        self.assertFalse(r_ord_swap["lenient_correct"])

    def test_percent_and_symbolic_fraction(self) -> None:
        prob_pct = "What percentage of the square is shaded?"
        r_pct = score_math_response("\\boxed{25\\%}", "25", finish_reason="stop", thinking=False, integer_task=False, problem=prob_pct)
        self.assertTrue(r_pct["lenient_correct"])

        r_frac = score_math_response("\\boxed{\\dfrac{\\sqrt{8}}{2}}", "\\sqrt{2}", finish_reason="stop", thinking=False, integer_task=False)
        self.assertTrue(r_frac["lenient_correct"])
        self.assertTrue(verify_gold_parseable("\\sqrt{2}", integer_task=False))


class TestLean4Verifier(unittest.TestCase):
    STMT = "theorem tst_sq_nonneg (x : ℝ) : x ^ 2 ≥ 0 := by sorry"

    def test_split_frozen_statement(self) -> None:
        name, prefix = split_frozen_statement(self.STMT)
        self.assertEqual(name, "tst_sq_nonneg")
        self.assertEqual(prefix, "theorem tst_sq_nonneg (x : ℝ) : x ^ 2 ≥ 0")

    def test_adversarial_precompile_blocks(self) -> None:
        attacks = [
            ("unclosed_block_comment", "```lean4\nby positivity /- swallow print axioms\n```"),
            ("sorry_tactic", "```lean4\nby sorry\n```"),
            ("admit_tactic", "```lean4\nby admit\n```"),
            ("native_decide_escape", "```lean4\nby native_decide\n```"),
            ("custom_axiom", "```lean4\naxiom cheat : False\nby exact False.elim cheat\n```"),
            ("signature_weakening", "```lean4\ntheorem tst_sq_nonneg (x : ℝ) : (0 : ℝ) ≥ 0 := by norm_num\n```"),
            ("local_notation_hijack", "```lean4\nlocal notation x \" ≥ \" y => True\nby trivial\n```"),
            ("hash_print_spoof", "```lean4\nby positivity\n#print axioms tst_sq_nonneg\n```"),
            ("run_tac_metaprogram", "```lean4\nby run_tac Lean.Elab.Tactic.evalTactic (<- `(tactic| trivial))\n```"),
            ("raw_template_marker", "```lean4\nby positivity\n```<|im_end|>"),
        ]
        for label, payload in attacks:
            with self.subTest(attack=label):
                with self.assertRaises(CandidateProofError):
                    extract_and_sanitize_proof_body(payload, self.STMT, thinking=False, strict=True)

    def test_valid_extractions(self) -> None:
        b1 = extract_and_sanitize_proof_body("```lean4\nby positivity\n```", self.STMT, thinking=False, strict=True)
        self.assertIn("positivity", b1)

        b2 = extract_and_sanitize_proof_body(
            "```lean4\ntheorem tst_sq_nonneg (x : ℝ) : x ^ 2 ≥ 0 := by\n  have h : 0 ≤ x ^ 2 := sq_nonneg x\n  linarith\n```",
            self.STMT,
            thinking=False,
            strict=True,
        )
        self.assertTrue(b2.startswith("by"))
        self.assertIn("sq_nonneg", b2)

    def test_live_lean_kernel_if_available(self) -> None:
        lean_proj = Path(os.environ.get("LEAN_PROJECT", "/scratch/governed_slm_rsi_v1/lean/mathlib_env"))
        elan_bin = Path(os.environ.get("ELAN_BIN", "/scratch/governed_slm_rsi_v1/lean/elan/bin"))
        if not (lean_proj.is_dir() and (elan_bin / "lake").is_file()):
            self.skipTest("Live Lean 4.24 environment not mounted locally")

        # 1. Valid positivity proof
        ok1 = verify_lean_candidate(
            "```lean4\nby positivity\n```",
            self.STMT,
            lean_project=lean_proj,
            elan_bin=elan_bin,
            finish_reason="stop",
            thinking=False,
        )
        self.assertTrue(ok1["ok"], msg=str(ok1))
        self.assertTrue(ok1["sft_clean_admissible"])

        # 2. Valid multi-line `have` subgoal proof
        ok2 = verify_lean_candidate(
            "<think>\n\n</think>\n```lean4\ntheorem tst_sq_nonneg (x : ℝ) : x ^ 2 ≥ 0 := by\n  have h := sq_nonneg x\n  linarith\n```",
            self.STMT,
            lean_project=lean_proj,
            elan_bin=elan_bin,
            finish_reason="stop",
            thinking=False,
        )
        self.assertTrue(ok2["ok"], msg=str(ok2))

        # 3. Wrong proof fails cleanly with compiler diagnostics
        bad1 = verify_lean_candidate(
            "```lean4\nby rfl\n```",
            "theorem tst_bad : (1 : ℕ) + 1 = 3 := by sorry",
            lean_project=lean_proj,
            elan_bin=elan_bin,
            finish_reason="stop",
            thinking=False,
        )
        self.assertFalse(bad1["ok"])
        self.assertTrue(len(bad1["diagnostics"]) > 0)

        # 4. Truncated finish_reason rejected even if code compiles
        bad_trunc = verify_lean_candidate(
            "```lean4\nby positivity\n```",
            self.STMT,
            lean_project=lean_proj,
            elan_bin=elan_bin,
            finish_reason="length",
            thinking=False,
        )
        self.assertFalse(bad_trunc["ok"])
        self.assertEqual(bad_trunc["error_class"], "TRUNCATED")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Unit tests for `eval_v3/isolated_scorer_v3.py`."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_V3_DIR = Path(__file__).resolve().parent
if str(EVAL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V3_DIR))

from isolated_scorer_v3 import (  # noqa: E402
    compute_paired_transitions,
    conservative_majority_vote,
    unbiased_pass_at_k,
)


class TestIsolatedScorerV3(unittest.TestCase):
    def test_unbiased_pass_at_k_exact_identities(self) -> None:
        self.assertAlmostEqual(unbiased_pass_at_k(8, 0, 1), 0.0)
        self.assertAlmostEqual(unbiased_pass_at_k(8, 0, 8), 0.0)
        self.assertAlmostEqual(unbiased_pass_at_k(8, 2, 1), 0.25)
        # 1 - C(6, 2)/C(8, 2) = 1 - 15/28 = 13/28
        self.assertAlmostEqual(unbiased_pass_at_k(8, 2, 2), 13.0 / 28.0)
        # 1 - C(6, 4)/C(8, 4) = 1 - 15/70 = 55/70
        self.assertAlmostEqual(unbiased_pass_at_k(8, 2, 4), 55.0 / 70.0)
        self.assertAlmostEqual(unbiased_pass_at_k(8, 2, 8), 1.0)
        self.assertAlmostEqual(unbiased_pass_at_k(8, 8, 4), 1.0)

    def test_conservative_majority_vote_order_invariant_and_symmetric(self) -> None:
        # Tie between correct "42" (2) and wrong "41" (2) MUST return False regardless of order
        self.assertFalse(
            conservative_majority_vote(
                ["42", "42", "41", "41"],
                [True, True, False, False],
            )
        )
        self.assertFalse(
            conservative_majority_vote(
                ["99", "99", "42", "42"],
                [False, False, True, True],
            )
        )
        # Equivalent correct forms ("1/2" and "0.5") merge to count 2 > max_wrong 1 -> True
        self.assertTrue(
            conservative_majority_vote(
                [r"\frac{1}{2}", "0.5", "3", "4"],
                [True, True, False, False],
            )
        )
        # Equivalent wrong forms ("1/2" and "0.5") ALSO merge to count 2 == c_true 2 -> False!
        self.assertFalse(
            conservative_majority_vote(
                ["42", "42", r"\frac{1}{2}", "0.5"],
                [True, True, False, False],
            )
        )

    def test_end_to_end_cli_and_paired_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            split_path = tdp / "split.jsonl"
            split_rows = [
                {"id": "p1", "stratum": "s_alg", "problem": "Find 6*7.", "gold": "42", "integer_task": True},
                {"id": "p2", "stratum": "s_alg", "problem": "Find 10+5.", "gold": "15", "integer_task": True},
                {"id": "p3", "stratum": "s_nt", "problem": "Find 2^5.", "gold": "32", "integer_task": True},
            ]
            split_path.write_text("\n".join(json.dumps(r) for r in split_rows) + "\n", encoding="utf-8")

            # Base: p1 has 2/8, p2 has 1/8, p3 has 0/8
            base_gen = tdp / "base_gen.jsonl"
            cand_gen = tdp / "cand_gen.jsonl"
            b_rows, c_rows = [], []
            for s in range(8):
                b_rows.append({
                    "id": "p1",
                    "sample_idx": s,
                    "finish_reason": "stop",
                    "completion_tokens": 200,
                    "text": "<think>ok</think>\\boxed{42}" if s < 2 else "<think>no</think>\\boxed{41}",
                })
                # Cand sharpens p1 to 6/8, forgets p2 (0/8), newly solves p3 (4/8), with 1 truncated unparsed rollout
                c_rows.append({
                    "id": "p1",
                    "sample_idx": s,
                    "finish_reason": "stop",
                    "completion_tokens": 180,
                    "text": "<think>ok</think>\\boxed{42}" if s < 6 else "<think>no</think>\\boxed{41}",
                })
                b_rows.append({
                    "id": "p2",
                    "sample_idx": s,
                    "finish_reason": "stop",
                    "completion_tokens": 300,
                    "text": "<think>ok</think>\\boxed{15}" if s == 0 else "<think>no</think>\\boxed{14}",
                })
                c_rows.append({
                    "id": "p2",
                    "sample_idx": s,
                    "finish_reason": "length" if s < 4 else "stop",
                    "completion_tokens": 500,
                    "text": "<think>unclosed" if s < 4 else "<think>no</think>\\boxed{14}",
                })
                b_rows.append({
                    "id": "p3",
                    "sample_idx": s,
                    "finish_reason": "stop",
                    "completion_tokens": 250,
                    "text": "<think>no</think>\\boxed{31}",
                })
                c_rows.append({
                    "id": "p3",
                    "sample_idx": s,
                    "finish_reason": "stop",
                    "completion_tokens": 220,
                    "text": "<think>ok</think>\\boxed{32}" if s < 4 else "<think>no</think>\\boxed{31}",
                })

            base_gen.write_text("\n".join(json.dumps(r) for r in b_rows) + "\n", encoding="utf-8")
            cand_gen.write_text("\n".join(json.dumps(r) for r in c_rows) + "\n", encoding="utf-8")

            base_out = tdp / "base_scored.json"
            cand_out = tdp / "cand_scored.json"
            for gen_p, out_p, cid, cmp_arg in [
                (base_gen, base_out, "base_m", []),
                (cand_gen, cand_out, "cand_m", ["--compare-baseline", str(base_out)]),
            ]:
                rc = subprocess.call([
                    sys.executable, "-I", "-B", str(EVAL_V3_DIR / "isolated_scorer_v3.py"),
                    "--track", "math",
                    "--split-file", str(split_path),
                    "--generations", str(gen_p),
                    "--expected-k", "8",
                    "--candidate-id", cid,
                    "--protocol-id", "test_v3",
                    "--thinking",
                    "--output", str(out_p),
                    *cmp_arg,
                ])
                self.assertEqual(rc, 0)

            b_pay = json.loads(base_out.read_text(encoding="utf-8"))
            c_pay = json.loads(cand_out.read_text(encoding="utf-8"))
            # Conditioned natural_stop_parse_failure_rate must be 0.0 on cand even though 4 rollouts truncated!
            self.assertAlmostEqual(c_pay["metrics"]["natural_stop_parse_failure_rate"], 0.0)
            self.assertAlmostEqual(c_pay["metrics"]["truncation_rate"], 4.0 / 24.0)

            tr = compute_paired_transitions(b_pay, c_pay, n_boot=500)
            self.assertEqual(tr["newly_solved_count"], 1)  # p3: 0 -> 4
            self.assertEqual(tr["forgotten_count"], 1)     # p2: 1 -> 0
            self.assertAlmostEqual(tr["net_frontier_expansion"], 0.0)
            # Interior items on base: p1 (2->6, +0.5) and p2 (1->0, -0.125) -> mean = +0.1875
            self.assertAlmostEqual(tr["interior_sharpening_delta"], 0.1875)


if __name__ == "__main__":
    unittest.main(verbosity=2)

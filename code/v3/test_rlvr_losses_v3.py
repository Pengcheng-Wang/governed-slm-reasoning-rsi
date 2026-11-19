#!/usr/bin/env python3
"""
Closed-form gradient (`d loss / d log_pi_t`) unit tests for `R1` (`dapo_asymclip`),
`R2` (`drgrpo_cispo`), and `R3` (`gspo_luffy`), verifying mathematical exactness and scale parity.
"""
from __future__ import annotations

import math
import unittest

import torch


def _import_loss():
    import importlib.util
    from pathlib import Path
    # Load ONLY pure PyTorch function `compute_sample_rlvr_loss` without requiring PEFT/Safetensors on host
    src = (Path(__file__).resolve().parent / "train_rlvr_v3.py").read_text(encoding="utf-8")
    ns: dict = {"torch": torch, "math": math}
    # Extract `compute_sample_rlvr_loss` block
    start = src.index("def compute_sample_rlvr_loss(")
    end = src.index("def _logp_chunk_step(")
    exec(src[start:end], ns)
    return ns["compute_sample_rlvr_loss"]


compute_sample_rlvr_loss = _import_loss()


class TestRLVRLossesV3(unittest.TestCase):
    def test_step1_onpolicy_gradient_scale_parity_across_all_three_arms(self) -> None:
        # At step 1 (`tok_logp == old_logp`), a sequence of length L == norm_tokens with A = +1.0
        # must produce identical sum_t |d loss / d logp_t| == 1.0 across R1 and R3, and 1.05 on R2 (with 0.05 anchor)!
        old_lp = torch.tensor([-0.5, -1.2, -0.8, -0.3], dtype=torch.float32)
        mask = torch.tensor([1, 1, 1, 1], dtype=torch.long)

        norms = {}
        for algo in ("dapo_asymclip", "drgrpo_cispo", "gspo_luffy"):
            lp = old_lp.clone().requires_grad_(True)
            loss, st = compute_sample_rlvr_loss(
                algo,
                lp,
                old_lp,
                mask,
                advantage=1.0,
                verified_positive=True,
                is_off_policy=False,
                norm_tokens=4.0,
                cispo_sft_alpha=0.05,
            )
            loss.backward()
            assert lp.grad is not None
            norms[algo] = float(lp.grad.abs().sum().item())
            self.assertEqual(st["clipped_frac"], 0.0)
            self.assertAlmostEqual(st["mean_ratio"], 1.0, places=6)

        self.assertAlmostEqual(norms["dapo_asymclip"], 1.0, places=6)
        self.assertAlmostEqual(norms["drgrpo_cispo"], 1.05, places=6)
        self.assertAlmostEqual(norms["gspo_luffy"], 1.0, places=6)

    def test_dapo_vs_cispo_clip_behavior_off_policy(self) -> None:
        # When r_t = exp(+0.40) ≈ 1.4918 > 1 + 0.28 and A > 0:
        # - DAPO zeroes gradient on the clipped token!
        # - CISPO preserves non-zero gradient weighted by detached `1.28`!
        old_lp = torch.tensor([-2.0, -2.0], dtype=torch.float32)
        new_base = torch.tensor([-1.6, -2.0], dtype=torch.float32)  # token 0 above 1.28, token 1 at 1.0
        mask = torch.tensor([1, 1], dtype=torch.long)

        lp_dapo = new_base.clone().requires_grad_(True)
        l_dapo, st_d = compute_sample_rlvr_loss(
            "dapo_asymclip", lp_dapo, old_lp, mask,
            advantage=1.0, verified_positive=False, is_off_policy=False, norm_tokens=2.0,
        )
        l_dapo.backward()
        self.assertAlmostEqual(float(lp_dapo.grad[0].item()), 0.0, places=6)
        self.assertAlmostEqual(float(lp_dapo.grad[1].item()), -0.5, places=6)
        self.assertAlmostEqual(st_d["clipped_frac"], 0.5, places=6)

        lp_cispo = new_base.clone().requires_grad_(True)
        l_cispo, st_c = compute_sample_rlvr_loss(
            "drgrpo_cispo", lp_cispo, old_lp, mask,
            advantage=1.0, verified_positive=False, is_off_policy=False, norm_tokens=2.0, cispo_sft_alpha=0.0,
        )
        l_cispo.backward()
        # Token 0 keeps capped detached weight `1.28 / 2.0 = -0.64`!
        self.assertAlmostEqual(float(lp_cispo.grad[0].item()), -0.64, places=5)
        self.assertAlmostEqual(float(lp_cispo.grad[1].item()), -0.50, places=5)
        self.assertAlmostEqual(st_c["clipped_frac"], 0.5, places=6)

    def test_gspo_sequence_clip_and_luffy_offpolicy_shaping(self) -> None:
        old_lp = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
        # Mean delta = +5e-4 > gspo_eps_high (4e-4) -> clipped for A > 0, NOT clipped for A < 0!
        shifted = old_lp + 5e-4
        mask = torch.tensor([1, 1, 1, 1], dtype=torch.long)

        lp_pos = shifted.clone().requires_grad_(True)
        l_pos, st_p = compute_sample_rlvr_loss(
            "gspo_luffy", lp_pos, old_lp, mask,
            advantage=1.0, verified_positive=True, is_off_policy=False, norm_tokens=4.0,
        )
        l_pos.backward()
        self.assertEqual(st_p["clipped_frac"], 1.0)
        self.assertTrue(torch.allclose(lp_pos.grad, torch.zeros_like(lp_pos)))

        lp_neg = shifted.clone().requires_grad_(True)
        l_neg, st_n = compute_sample_rlvr_loss(
            "gspo_luffy", lp_neg, old_lp, mask,
            advantage=-1.0, verified_positive=False, is_off_policy=False, norm_tokens=4.0,
        )
        l_neg.backward()
        self.assertEqual(st_n["clipped_frac"], 0.0)
        self.assertTrue(torch.all(lp_neg.grad > 0))

        # Off-policy LUFFY rescue: w_t = pi / (pi + 0.1), exact closed-form gradient
        lp_off = torch.tensor([math.log(0.3), math.log(0.1)], dtype=torch.float32, requires_grad=True)
        l_off, _ = compute_sample_rlvr_loss(
            "gspo_luffy", lp_off, old_lp[:2], torch.tensor([1, 1], dtype=torch.long),
            advantage=1.0, verified_positive=True, is_off_policy=True, norm_tokens=2.0, luffy_gamma=0.1,
        )
        l_off.backward()
        # Expected: -(0.3 / 0.4) / 2 = -0.375 and -(0.1 / 0.2) / 2 = -0.25
        self.assertAlmostEqual(float(lp_off.grad[0].item()), -0.375, places=5)
        self.assertAlmostEqual(float(lp_off.grad[1].item()), -0.250, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

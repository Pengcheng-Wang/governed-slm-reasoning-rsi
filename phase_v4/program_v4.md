# `program_v4.md` — Phase V4 Cross-Family Governed Post-Training (`Inkling-2-Mini`, `Gemma 4`, `Muse Glimmer`)

## 1. Objective & Scope

Extend our SHA-256-locked governed post-training and verifier-driven self-improvement benchmark from the Qwen family (`4B`, `9B`, `27B` Dense + `30B-A3B`, `35B-A3B` Sparse MoE) to **four non-Qwen open-weight model architectures** released in late 2026:

1. **`thinkingmachines/Inkling-2-Mini` (`26.4B` physical / `3.4B` physical active / `5.4B` 3-pass looped active Sparse MoE):** 24 physical layers (`6` prelude + `12` shared-weight loop core run `3` passes with rank-64 gated cross-pass message residual controllers + `6` coda = `48` layer visits/token; `128` routed experts `top-8`).
2. **`google/gemma-4-26B-A4B-it` (`25.2B` total / `3.8B` active Sparse MoE):** Size-matched non-Qwen Sparse MoE comparator (`30` hybrid sliding/global p-RoPE layers, `128` routed experts `top-8` + `1` shared dense MLP).
3. **`google/gemma-4-31B-it` (`30.7B` Dense):** Hybrid sliding/global p-RoPE + KV-shared `60`-layer Dense model.
4. **`meta-models/Muse-Glimmer-30B` (`29.6B` Dense):** Hybrid `[Local, Local, Local, Global]` gated-attention `52`-layer Dense model (`GQA 32:2`).
5. **`google/gemma-4-E4B-it` (`4.5B` effective / `8.0B` PLE Dense Reference):** Active-FLOPs-matched compact Gemma-4 reference evaluated zero-shot on `Holdout_V3` and Longitudinal suites.

All evaluations run on the **exact same SHA-256-locked Phase V3 splits** (without mutating `IMMUTABLE_MANIFEST.json`, `IMMUTABLE_MANIFEST_V2.json`, `IMMUTABLE_MANIFEST_V3.json`, or `IMMUTABLE_MANIFEST_V3_MOE.json`):
- **Track M (Competition & 2026 Research Mathematics):** `Search_V3` (`N=160`), `Confirm_V3` (`N=300`, `3 x 100` balanced strata), Single-Touch `Holdout_V3` (`N=194`: `SMT 2025`, `HMMT Feb 2025`, `APEX 2025`, `arXivMath 2026`), and Longitudinal `2025` 4-Contest Suite (`N=200`: `BeyondAIME`, `CMIMC 2025`, `BRUMO 2025`, `HMMT Nov 2025`).
- **Track L (Formal Lean `v4.24.0 + Mathlib f897ebcf` Theorem Proving):** `Search_V3` (`N=140`), `Confirm_V3` (`N=180`), Single-Touch `Holdout_V3` (`N=225`: `arXivLean 2026` + `FormalMATH-Lite`), and Longitudinal `MiniF2F-test` (`N=244`) + `ProofNet-test` (`N=163` unique IDs).

---

## 2. Immutable Evaluation Surface (`IMMUTABLE_MANIFEST_V4.json`)

Protected paths locked via `governed_loop_guard_v38.py freeze-manifest` prior to Plane-2 training:
- `configs_v4/`: All Search and Confirmation gate rules (`aggregation_weighting = "equal_stratum"` explicitly set, `claim_scope = "fixed_checkpoint"`, per-cell `alpha_one_sided = 0.025` with `max_confirm_looks = 2` -> $\alpha_{\text{eff}} = 0.0125$, `min_improvement = 0.015`, plus Holm-Bonferroni campaign-wide reporting across all `8` `(model, track)` cells).
- `eval_v4/`: `governed_loop_guard_v38.py`, `channel_normalizer_v4.py`, `math_verifier_v4.py`, `lean4_verifier_v4.py`, `isolated_scorer_v4.py`.
- `code/v4/`: `model_registry_v4.py`, `eval_engine_v4.py`, `train_sft_v4.py`, `train_rlvr_v4.py`, `run_v4_worker.py`.
- `data/splits_v3/` & `eval_v3/sealed_holdout/`: Exact byte-for-byte SHA-256 match to `IMMUTABLE_MANIFEST_V3.json`.

*(Audit note on `B10` cross-phase Holdout reuse: `Holdout_V3` is locked inside `eval_v3/sealed_holdout/` with zero item feedback or hyperparameter changes from Phases V1–V3, and each of the 8 new `(model, track)` cells in Phase V4 touches `Holdout_V3` at most once for its promoted candidate via `O_CREAT | O_EXCL` `HOLDOUT_SEAL_v4_<track>_<model>.json`.)*

---

## 3. Plane-1 Pre-Freeze Calibration & Parity Gates

Before freezing `IMMUTABLE_MANIFEST_V4.json`:
1. **Native Channel & Logit Softcap Parity:**
   - Verify `compute_family_logits_fp32` (`output_multiplier` + `final_logit_softcapping`) matches `model.forward()` logits (`max_abs_diff < 5e-3` in `BF16`->`FP32`), and verify `8-GPU FSDP FULL_SHARD` LoRA `FP32` merge parity (`top1_token_agreement >= 0.96` on multi-GPU reload).
   - Verify `channel_normalizer_v4.py` cleanly separates `<|content_thinking|>` (`inkling2`), `<|channel>thought` (`gemma4`), and `to=self<|message|>` (`muse_glimmer`) from final responses with `0%` template-leak false positives.
2. **Reasoning Effort & Horizon Calibration on `Search_V3`:**
   - Calibrate native reasoning effort / thinking toggles on `Search_V3` (`0` Confirm/Holdout reads) so each model operates in its highest-scoring non-truncating protocol (`stop_rate >= 60%` Math, `>= 55%` Lean).
3. **Empirical `A/A` Null Gate (`8/8 -> ROLLBACK`):**
   - Run an independent `A/A` evaluation on every `(model, track)` cell and require `governed_loop_guard_v38.py gate-decision` to return `ROLLBACK`.

---

## 4. Plane-2 Diagnostic Crossover (`Pure-B0 -> RLVR` vs. `Horizon-Matched 2-SFT -> RLVR`)

Inside `Search_V3` (`0` Confirmation looks spent), every `(model, track)` cell evaluates two orthogonal post-training families using 8-GPU `FSDP FULL_SHARD` (with frozen routers & frozen `128` routed experts on Sparse MoEs `inkling2_mini` and `gemma4_26b_a4b`):
- **Arm A — Zero-SFT On-Policy `RLVR` (`Pure-B0 -> DAPO` & `Pure-B0 -> Dr.GRPO+CISPO`):** Tests whether eliminating off-policy SFT shift and applying on-policy verifier RL + overlong shaping directly unlocks latent breadth (`pass@8`) and cures truncation (`stop_rate`).
- **Arm B — Horizon-Matched `2-SFT + 3-Proposal RLVR` (`Stage-I Breadth SFT -> Stage-II ZPD Frontier SFT -> {DAPO, Dr.GRPO+CISPO, GSPO+LUFFY}`):** Expands domain/formal-tactic breadth on 4-layer-decontaminated problems (`0` self-generated teacher labels on `30B/31B`), then sharpens reliability across `pct50` and `pct100` milestones.
- **Order-Invariant Selection:** `governed_loop_guard_v38.py select-search-winner` ranks all Search candidates order-invariantly against `B_0` under `equal_stratum`.

---

## 5. Statistical Confirmation (`Confirm_V4`) & Single-Touch Protected Holdout (`Holdout_V3`)

1. **Confirmation (`gate-confirm-iut`):** Evaluates the Search winner on `Confirm_V3` (`N=300` Math / `N=180` Lean) with `max_confirm_looks = 2` ($\alpha_{\text{eff}} = 0.0125$), `+1.50 pp` `equal_stratum` additive margin, `stop_rate` & `pass@8` non-degradation guards, and cryptographic `.safetensors` digest pinning (`candidate_pin_<id>.json`).
2. **Atomic Single-Touch Holdout (`gate-holdout`):** Unseals `Holdout_V3` (`reserve -> evaluate -> finalize`) **if and only if** `Confirm_V4` returns `PROMOTE`. Unpromoted cells record `CONFIRMED_ROLLBACK_NO_HOLDOUT_TOUCH` with `0` candidate Holdout evaluations.
3. **External Longitudinal Suite:** Post-lock transfer evaluation (`B_0` vs. Search/Confirm winner) on `MiniF2F-test` (`N=244`), `ProofNet-test` (`N=163`), and the `2025` 4-Contest Math Suite (`N=200`).

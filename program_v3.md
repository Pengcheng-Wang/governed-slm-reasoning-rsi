# Phase V3 Governed Post-Training Specification (`2-SFT + 3-Proposal RLVR` Across `4B`, `9B`, and `27B`)

## 1. Objective & Scope

Phase V3 studies verifiable post-training across three hybrid Gated-DeltaNet + full-attention model scales — **`Qwen3.5-4B`**, **`Qwen3.5-9B`**, and **`Qwen3.8-27B`** — on two deterministic-verifier domains:
- **Track M (Competition & Research Mathematics):** Direct concise derivations (`12,288`-token generation limit) verified out-of-process via exact symbolic/integer equivalence (`eval_v3/isolated_scorer_v3.py`).
- **Track L (Formal Lean 4.24 + Mathlib Theorem Proving):** Direct Lean 4 proof blocks (`2,048`-token generation limit) compiled out-of-process under `Lean v4.24.0 + Mathlib f897ebcf` with transitive `#print axioms` verification (`{propext, Classical.choice, Quot.sound}`) and strict proof-search/escape bans.

## 2. Three-Stage Post-Training Recipe (`2-SFT + RLVR`, No Mid-Training or DPO)

Adapted from `Axion 1.0` (`2 SFT + RL`), separating domain breadth expansion from reliability sharpening:
1. **Stage I — Multi-Source Breadth SFT (`constant_with_warmup`, 3 epochs):**
   - *Track M:* Decontaminated multi-domain verified concise derivations (`<= 7,680` tokens) from `DeepMath-103K` and quantitative `Axion 1.0` trajectories.
   - *Track L:* `#print axioms`-verified Lean 4.24 tactic proofs (`<= 1,536` tokens) combining decontaminated `Goedel-SFT-v2` theorems and standalone anonymized multi-step `Mathlib4` proofs (`Algebra`, `Analysis`, `Topology`, `LinearAlgebra`, `NumberTheory`, `GroupTheory`, `RingTheory`).
2. **Stage II — Model-Aware ZPD Hard-Frontier SFT (`cosine` with `8%` warmup after AdamW reset, 2 epochs + `15%` Stage-I replay):**
   - Consolidates Stage-I `FP32` low-rank adapters, resets optimizer moments, and specializes on problems where the target scale's base `B_0` (`k=4` probe) achieves `<= 2/4`, augmented for `4B`/`9B` with cross-family-audited strengthened problems (`27B` `>= 6/8` consensus with margin `>= 4` AND independent `Phi-4-mini-reasoning` `1/4..3/4` blind solve; `27B` itself trains strictly on external human-ground-truth items).
3. **Stage III — Scale-Matched 3-Proposal Multi-Epoch (`ppo_epochs = 2`) RLVR Tournament:**
   - Executed at identical rollout budgets against exact trainer-computed `FP32` `pi_old`:
     - **`R1` (`dapo_asymclip`):** Dynamic variance-filtered (`1 <= c <= 7`) token-level asymmetric clip `[0.20, 0.28]`, group-std advantage, soft-overlong termination shaping.
     - **`R2` (`drgrpo_cispo`):** Unbiased centered advantage `2(R_i - \mu_g)` + constant batch `L_norm` + detached `CISPO` clipped-IS weight + `0.05` verified-positive NLL anchor on `1 <= c <= 7` groups.
     - **`R3` (`gspo_luffy`):** Sequence-geometric importance ratio with `GSPO` clip `[3\times 10^{-4}, 4\times 10^{-4}]` on `1 <= c <= 7` groups + corpus-reference `LUFFY` `\pi/(\pi + 0.10)` rescue (`<= 20%` group cap) on `0/8` groups.
   - Rollout-matched checkpoints (`50%`, `75%`, `100%`) are ranked order-invariantly on disjoint `Search_V3` via `select-search-winner`.

## 3. Immutable Evaluation Surface, Multi-Metric Decomposition & Promotion Gates

- **Frozen Surface (`IMMUTABLE_MANIFEST_V3.json`):** Locks `program_v3.md`, `eval_v3/`, `configs_v3/`, and `data/splits_v3/`.
- **Zero-Leak Split Hierarchy (`audit-splits`):**
  - `Search_V3` (`N=160` Math, `N=140` Lean) is 100% disjoint from `RL_Pool_V3`, `Stage1/2_SFT`, `Confirm_V3`, `Holdout_V3`, and all prior V1/V2 splits.
  - `Confirm_V3` (`N=300` Math, `N=180` Lean) enforces Intersection-Union Testing (`IUT`, `max_confirm_looks = 2`, `effective_alpha = 0.0125`, `min_improvement = +1.5 pp` on `equal_stratum` `avg@8`) against `B_0` AND any prior promoted champion (`current_incumbent_id`).
  - `Holdout_V3` (`MathArena` `SMT 2025`, `HMMT Feb 2025`, `APEX 2025`, `arXivMath 2026`; `Lean` `arXivLean 2026` + `FormalMATH-Lite` holdout) is unsealed strictly once via `gate-holdout` (`O_CREAT | O_EXCL`).
- **Capability Decomposition:** Every stage reports unbiased `pass@1, 2, 4, 8`, conservative equivalence-clustered `maj@8`, `capability_slope_8_1 = pass@8 - pass@1`, paired `0 -> >0` newly-solved vs `>0 -> 0` forgotten rates, interior sharpening `\Delta \hat{p}`, `distinct_answers_at_8`, and `tokens_per_correct`.

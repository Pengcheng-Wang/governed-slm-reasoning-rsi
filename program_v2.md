# `program_v2.md` — Phase V2 Multi-Candidate, Multi-Benchmark & Causal-Isolation Governed SLM RSI Specification

## 1. Relationship to Sealed Phase V1
- Phase V1 (`program.md`, `IMMUTABLE_MANIFEST.json`, `manifest_root_sha256 = 82a61cdf6b91a58fb537dc83f8bbbb0ea2ad46928a5a2174683db543a4ccf468`) remains **permanently sealed and unmodified**. No Phase V2 experiment retroactively modifies any Phase V1 checkpoint, threshold, or verdict.
- Phase V2 is pre-registered under **`IMMUTABLE_MANIFEST_V2.json`** following independent frontier co-design reviews by **`GPT 6.2 Astra, Max`** and **`Opus 5.5, Max`** in M365 Copilot `Cowork` (`sources/cowork_v2_frontier_design_review.md`).

## 2. 4-Model Cross-Family Candidate Panel (`3.8B–9B`, 3 Architectures + 2 Reference Comparators)
1. **`qwen35_4b` (`Qwen/Qwen3.5-4B`, `4B` Hybrid Gated-DeltaNet + Softmax Anchor):** Track L V2 & Track M V2 + Causal Recursion & Causal Compiler-Repair Sub-Studies.
2. **`qwen3_4b` (`Qwen/Qwen3-4B`, `4B` Pure Dense Transformer Contrast):** Track L V2 & Track M V2 (calibrated native channel locked in Plane-1 V2 before weight updates).
3. **`phi4_mini` (`microsoft/Phi-4-mini-reasoning`, `3.8B` Non-Qwen Family Replication):** Track L V2 (gated on pre-registered baseline estimability floor `learnable_fraction(M_0) >= 0.10`) & Track M V2, evaluated and trained in its native `<think>...</think>` channel (`stop_token_ids = [199999, 200020]`, `<|end|>` supervised, post-`</think>` strict grading, parameter-matched fused LoRA `r=48, alpha=96` across all `128` linear modules).
4. **`qwen35_9b` (`Qwen/Qwen3.5-9B`, `9B` Hybrid Scale-Up Recovery):** Track L V2 & Track M V2, using `1,024`-token activation-checkpointed `lm_head` logprob chunks to train full `12,288`-token sequences with zero `2,688`-token prefix clipping and linear `10k -> 12k` soft-overlong reward shaping.
- **Zero-Shot Reference Comparators:** `goedel_8b` (`Goedel-LM/Goedel-Prover-V2-8B`, specialist Lean 4 prover on Track L V2) and `qwen38_27b` (`Qwen/Qwen3.8-27B`, `27B` general frontier reference on Track L V2 & Track M V2).

## 3. Disjoint Phase V2 Split Hierarchy (`0` Overlap with Phase V1)
- **Track L V2 (Formal Lean `v4.24.0 + Mathlib f897ebcf`, `#print axioms` Verified):**
  - *Search (`v2_lean_search120.jsonl`):* `120` disjoint `Lean-Workbook` theorems elaborated cleanly in Lean 4.24 (`TrainPool`: `600` disjoint theorems).
  - *Confirmation (`v2_lean_confirm171.jsonl`):* `164` V1-decontaminated `ProofNet-Lean4 valid` undergraduate real/complex analysis, abstract algebra, and topology theorems elaborated cleanly in Lean 4.24.
  - *Protected Single-Touch Holdout (`v2_lean_holdout_proofnet174.jsonl`):* `167` V1-decontaminated `ProofNet-Lean4 test` undergraduate theorems elaborated cleanly in Lean 4.24.
- **Track M V2 (Ultra-Hard Olympiad & 2025 Competition Mathematics):**
  - *Search (`v2_math_search128.jsonl`):* `128` disjoint `Omni-MATH` Tier 3–4 exact-answer olympiad problems.
  - *Confirmation (`v2_math_confirm256.jsonl`):* `256` disjoint `Omni-MATH` Tier 3–4 (`128` Tier-3 + `128` Tier-4) exact-answer olympiad problems + `math500_retention_confirm150` non-degradation check.
  - *Protected Single-Touch Holdout (`4` Fresh Benchmarks):* `BeyondAIME` (`100`), `BRUMO 2025` (`30`), `CMIMC 2025` (`40`), and `HMMT Nov 2025` (`30`), strictly deduplicated against all V1 and V2 Search/Confirm splits.

## 4. Causal-Isolation Sub-Studies & Statistical Gatekeeping
- **Fixed-Sequence Gatekeeping:** Primary Cross-Family Replication (`Holm-Bonferroni` across estimable Confirmation cells at one-sided `alpha = 0.025`, `min_improvement = +0.02`) -> Causal Recursion Isolation -> Causal Lean Compiler-Diagnostic Repair.
- **Dual Completion Guardrail (Plane-1 Calibrated on Search before Plane-2 Freeze):** Requires strict per-model baseline-relative non-inferiority (`max_degradation_vs_baseline.stop_rate <= 0.025` Search / `<= 0.020` Confirm, `natural_stop_parse_failure_rate <= 0.03`) plus Plane-1 split-calibrated completion floors (`stop_rate >= 0.44` on `Omni-MATH Tier 3-4` where `Phi-4-mini-reasoning`, `Qwen3-4B think16k`, and `Qwen3.5-9B` `B_0` stop rates are `46.4%`, `56.4%`, and `56.5%`, and `stop_rate >= 0.22` on multi-family Lean 4.24 where `Qwen3-4B think6k` `B_0` is `25.1%` while `Qwen3.5` `4B`/`9B` `B_0` `95.3%`/`96.8%` remain strictly bound to `>= 93.3%`/`94.8%` via the relative guardrail).
- **Causal Recursion Test (`qwen35_4b`):** Shares Round-1 `M_1` and frozen Round-2 prompt multiset `P_2`; compares `M_2^{frozen-gen}` (`M_0` generator) vs `M_2^{rec-yield}` (`M_1` generator subsampled to `M_0`'s deduplicated verified count) vs `M_2^{rec-full}` (all deduplicated `M_1` verified traces), all reset to `M_0` at matched optimizer steps.
- **Causal Compiler-Repair Test (`qwen35_4b`):** Mines natural `(failed_proof, lean4_diagnostic + unsolved_goal -> verified_fix)` pairs (`50/50` unmasked + masked diagnostic SFT); compares `compiler_repair_4+4` against `direct8`, `masked_retry_4+4`, and `shuffled_diag_4+4` reusing identical first `4` direct roots per theorem.

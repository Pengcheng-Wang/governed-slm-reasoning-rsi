# Phase V2 Pre-Registration Frontier Co-Design Review via M365 Copilot `Cowork` (`GPT 6.2 Astra, Max` & `Opus 5.5, Max`)

**Date (UTC):** 2026-11-18  
**Reviewers (M365 Copilot Desktop `Cowork` at `Max` Effort):**
1. **`GPT 6.2 Astra, Max`** (*"Latest model for tough problems"*) — Experimental Design, Statistical Gatekeeping, Causal Recursion Control & Compiler-Repair Protocol
2. **`Opus 5.5, Max`** (*"For complex, high stakes work"*) — Architecture/Tokenizer Pitfalls (`Phi-4-mini-reasoning` LongRoPE / `<|end|>` / Fused LoRA), Causal Confounds & `9B` `12k` Chunked-Logits Memory Engineering

---

## 1. Key Critiques & Binding Design Upgrades from `GPT 6.2 Astra, Max`

1. **Keep Phase V1 Sealed; Pre-Register Disjoint Phase V2 (`IMMUTABLE_MANIFEST_V2.json`):**
   - No Phase V2 experiment may alter Phase V1 verdicts or re-tune against Phase V1's single-touch Holdouts (`MiniF2F-test`, `AIME 2025/2026`, `HMMT Feb 2026`, `OlympiadBench`).
   - Construct completely disjoint V2 Confirm and Single-Touch Holdout suites verified by canonical statement/text SHA-256 deduplication against all V1 training, Search, Confirm, and Holdout corpora:
     - **Track L V2 (Formal Lean `v4.24.0 + Mathlib f897ebcf`):** `ProofNet-Lean4` upstream `valid` (`171/185` elaborating cleanly in Lean 4.24) -> **V2 Confirm**; upstream `test` (`174/186` elaborating cleanly in Lean 4.24) -> **V2 Single-Touch Holdout**; disjoint `Lean-Workbook` elaborated theorems -> **V2 Search & Training Pool**.
     - **Track M V2 (Hard Olympiad & 2025 Contest Math):** Disjoint `Omni-MATH` Tier 3–4 exact-answer problems (`N=256`, `128` Tier-3 + `128` Tier-4) -> **V2 Confirm**; **`BeyondAIME` (`100`) + `BRUMO 2025` (`30`) + `CMIMC 2025` (`40`) + `HMMT Nov 2025` (`30`)** -> **V2 Single-Touch Holdout** (after exact deduplication against V1 & V2 Confirm, prioritizing Holdout).

2. **4-Model Cross-Family SLM Candidate Matrix (`3.8B–9B`, 3 Distinct Architectures + Specialist/Frontier References):**
   - `Qwen/Qwen3.5-4B` (`4B` Hybrid Gated-DeltaNet + Softmax Anchor — Track L & Track M)
   - `Qwen/Qwen3-4B` (`4B` Size-Matched Pure Dense Transformer Contrast — Track L & Track M)
   - `microsoft/Phi-4-mini-reasoning` (`3.8B` Non-Qwen Family Replication — Track L [subject to pre-registered `M_0` estimability floor] & Track M)
   - `Qwen/Qwen3.5-9B` (`9B` Hybrid Scale-Up with Full-`12k` Chunked-Logits RLVR Recovery — Track L & Track M)
   - *Reference Comparators:* `Goedel-LM/Goedel-Prover-V2-8B` (specialist Lean 4 prover on Track L) and `Qwen/Qwen3.8-27B` (`27B` frontier reference on Track L & Track M).

3. **Causal Isolation #1 — Recursive Self-Improvement vs Non-Recursive Rollout Budget:**
   - Share Round-1 checkpoint `M_1` and freeze one Round-2 prompt set `P_2`.
   - Compare Round-2 **Frozen-Generator Control** (`M_0` generates rollouts on `P_2`) vs **Recursive Learner** (`M_1` generates rollouts on `P_2`), both reset fresh to `M_0` with matched optimizer budget.

4. **Causal Isolation #2 — True Lean 4.24 Compiler-Diagnostic Repair:**
   - Mine `>= 256` verified natural `(failed_proof, first_diagnostic + error_span + local_goal) -> verified_fix` pairs across `>= 128` training theorems (`1:1` mixed with direct proofs, supervising only the corrected proof, compared against an equal-step direct-only adapter).
   - Compare `compiler_repair_4+4` against `direct8` (reusing the exact same first `4` direct roots) and `masked_retry_4+4` under both call-matched (`<= 8` calls) and decode-token-matched budgets.

5. **Full-Length `9B` RLVR & Dual Completion Guardrail:**
   - Train across full `12,000`-token outputs without `2,688`-token prefix clipping; apply linear soft-overlong penalty rising `0 -> 0.20` from `10,000 -> 12,000` tokens (`0` reward on cutoffs/parse failures).
   - Separate statistical efficacy from operational promotion using a pre-registered dual completion guardrail (`delta_parseable_stop_vs_M0 >= -1.0 pp` plus split-calibrated completion floor).

---

## 2. Critical Technical Safeguards Added by `Opus 5.5, Max`

1. **`Phi-4-mini-reasoning` Native Channel, `<|end|>` Masking, LongRoPE & Fused LoRA:**
   - **Keep native `<think>` channel:** `Phi-4-mini-reasoning` is a pure long-CoT distillation; forcing empty `<think></think>` pushes `M_0` off-distribution. Pre-register `Phi-4`'s native `<think>...</think>` channel (`thinking=True` post-`</think>` strict grading so intermediate `\boxed{}` inside `<think>` are ignored) and pass explicit stop token IDs `[200020 (<|end|>), 199999 (<|endoftext|>)]`, keeping `<|end|>` inside the supervised loss mask.
   - **LongRoPE (`4,096` threshold) sampler/trainer parity:** Recompute `old_logprobs` inside the trainer under identical HF LongRoPE scaling with truncated importance ratio capping (`<= 2.0`) so vLLM-vs-HF RoPE factor switching past `4,096` tokens never inflates policy ratios.
   - **Fused `qkv_proj` / `gate_up_proj` parameter-matched LoRA:** `Phi3ForCausalLM` uses `['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']` (`128` linear modules vs `252` in `Qwen3-4B` and `248` in `Qwen3.5-4B`). Use `r=48, alpha=96` on `Phi-4-mini-reasoning` to match trainable parameter fraction to `r=32, alpha=64` on `Qwen`, and assert adapted module count at initialization.
   - **Pre-registered `M_0` Estimability Floor:** Pre-register `learnable_fraction(M_0) >= 0.10` (`>= 10%` of Search items with `0 < pass_count < 8`); below this floor, report `"below_estimability_floor"` rather than running zero-gradient noise updates.

2. **Refinements to Both Causal Controls:**
   - **Yield-Matched Recursion Sub-Arm:** Because `M_1` has higher accuracy and lower entropy than `M_0`, deduplicate exact solutions first and include **both** `M_2^{rec-yield-matched}` (subsampled down to `M_0`'s unique verified count, matching optimizer steps) and `M_2^{rec-full}` against `M_2^{frozen-gen}`.
   - **In-Distribution Masked & Shuffled-Diagnostic Placebo Controls for Repair:** Include masked-diagnostic variants (`50/50`) during repair SFT so `masked_retry_4+4` is in-distribution, use single-turn repair prompts, and add a **length-matched `shuffled_diag_4+4` placebo** (feeding a diagnostic from a different failed theorem) alongside `masked_retry_4+4` and `direct8` on the exact same `4` initial roots.

3. **Exact Memory-Safe Recipe for `Qwen3.5-9B` Full-`12k` RLVR (`28.9 GB` vs `39.7 GB` Logits OOM):**
   - Full `[1, 13300, 248320]` FP32 logits + backward consumes **`39.7 GB`/GPU** (the exact root cause of V1's OOM above `2,688` tokens).
   - **Fix:** Run the transformer backbone to produce BF16 `hidden_states` (`[1, T, d]`, ~`106 MB`), and apply `lm_head` + FP32 `log_softmax` + target-token `gather` inside **checkpointed `1,024`-token chunks on completion positions only** (`batch_size=1` long sequence per GPU, gradient checkpointing across layers, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Peak memory stays `< 30 GB` on `80 GB` A100s at full `12,288`-token completion length.

---

## 3. Mid-Campaign Optimization & `governed_loop_guard` v3.6 Amendment Review (`GPT 6.2 Astra, Max`)

**Date (UTC):** 2026-11-19  
**Context:** After reloading the updated `/governed-loop-optimization` (`Hardened Executable Guard v3 / v3.6`) skill and observing both `+13.57 pp` (`33.45% -> 47.02%`) on `Qwen3.5-9B` `Omni-MATH Tier 3-4` (`N=256`) Confirmation and `69.3%–87.5%` rollout truncation on `Phi-4-mini-reasoning` (`3.8B` LongRoPE `<think>`) under an `8,192`-token pool cap, we consulted **`GPT 6.2 Astra, Max`** in `Cowork` before touching Protected Holdout or spending `Phi-4-mini`'s single Confirmation look.

1. **`Phi-4-mini-reasoning` (`3.8B` LongRoPE `<think>`) Right-Censoring & Non-Progress Repetition Filter:**
   - **Diagnosis & Fix:** Raising pool `max_tokens` from `8,192 -> 12,288` (matching the `12,288`-token evaluation budget 1:1 rather than `14,336`, since answers arriving past `12,288` are evaluation-budget misses) removes artificial right-censoring on `<think>` derivations.
   - **Loop Hygiene:** Because SymPy can verify a final `\boxed{}` answer even after degenerate repetition blocks inside `<think>`, require natural stop + SymPy verification + **reject traces repeating `>= 24`-word blocks `>= 4` times without progress** (`--reject-repetitive-blocks`), then sample `<= 3` distinct traces uniformly with a frozen problem-specific seed, retaining `m1_phi4_mini_restem` (`8,192`-censored) in the Search ledger as an explicit right-censored control before running Checkpointed-Chunk `DAPO+LUFFY` RLVR and freezing one checkpoint prior to `Phi-4-mini`'s single Confirmation look.

2. **Conjunctive Dual-Ledger `governed_loop_guard` v3.6 Mid-Campaign Amendment (`Veto-Only, Never Rescue`):**
   - **Statistical Validity:** Keep `IMMUTABLE_MANIFEST_V2.json` (`20` files), `configs_v2/*`, and the original `v2` gate `100%` untouched as primary pre-registered law. Layer `governed_loop_guard` v3.6 (`audit-leaks`, percentile-bootstrap LCB, `--proposer-out` redaction, `equal_stratum` across the 4 Holdout 2025 contests, `per_stratum_hard_floors`, and `O_CREAT | O_EXCL` write-once `HOLDOUT_SEAL.json`) via a timestamped hash-linked mid-campaign amendment (`AMENDMENT_V3_6.json`) **before** any Protected Holdout access, under strict conjunction: **`v3.6` can veto (`"v2 passed; promotion withheld under v3.6"`), never rescue a `v2` `ROLLBACK`**.
   - **Item-Pooled vs. `equal_stratum` + `MATH-500` Retention Floor:** Report both pre-registered item-pooled (`N=200`) CIs (performance on a random pooled contest item) and v3.6 `equal_stratum` (`1/4` per contest across `BeyondAIME=100`, `CMIMC=40`, `BRUMO=30`, `HMMT=30`, resampling paired problems within each contest) CIs, while keeping `MATH-500` (`N=150`) outside the 4-contest average as an explicit retention floor.

3. **Strict Fail-Closed Governance on Severe OOD Formal Shift (`Lean-Workbook -> ProofNet-Lean4`):**
   - **Decision Rule (`Option a`):** On `ProofNet-Lean4 valid` (`N=164` Rudin/Munkres/Dummit-Foote textbook real analysis, topology, & Galois theory), where vocabulary shift drops even specialist `Goedel-Prover-V2-8B` `58.44% -> 8.38%` while our `astar_qwen35_9b` (`3.35% -> 3.89%`, beating `Qwen3.8-27B` `3.05%`) and `astar_qwen35_4b` (`2.21% -> 2.90%`) achieve positive `100%` axiom-clean point gains that do not clear Holm-adjusted `LCB > 0`, **strictly honor Confirm `ROLLBACK`**: retain `B_0` as the official promoted ProofNet policy, record `A*` on `ProofNet-test` as *"not evaluated — Confirmation gate failed"*, and evaluate only the pre-registered zero-shot baselines (`B_0`, `Goedel-Prover-V2-8B`, `Qwen3.8-27B`) on `ProofNet-test`.

---

## 4. Pre-Holdout Adversarial Code & Statistical Audit (`Opus 5.5, Max`)

**Date (UTC):** 2026-11-19  
**Verdict:** *"Bottom line: The design is sound. Before you unseal, fix three things: how candidate weights are bound, Holm's family and indexing, and the holdout decision test. In (C), the only warranted claim is 'no detectable effect.'"*

1. **Full-Checkpoint Shard & Metadata SHA-256 Binding + Local-NVMe Atomic Seal:**
   - Hashing `model.safetensors.index.json` or `config.json` alone would miss in-place BF16 shard edits. Fix: pass the full checkpoint directory (`--candidate-artifacts <ckpt_dir>`) and write a canonical manifest (`WEIGHTS_SHA256_MANIFEST.json`) hashing sorted relative paths, byte sizes, and SHA-256 digests of every file (`*.safetensors` shards, index, `config.json`, `generation_config.json`, tokenizer, `chat_template.jinja`), failing closed on any symlink or unlisted shard.
   - Reserve `HOLDOUT_SEAL_{track}_{model}.json` via `O_CREAT | O_EXCL` + `os.fsync` on local NVMe (`/scratch/governed_slm_rsi_v2/results/v3_governance/`, never on FUSE blob mounts), pinning the `AMENDMENT_V3_6.json` SHA-256, `governed_loop_guard_v3.py` SHA-256, Holdout split SHA-256s, and comparator (`b0`, `ref_qwen38_27b`) Holdout digests **before** evaluating any promoted `astar_*` candidate on Protected Holdout.
   - Ensure 0-based Holm-Bonferroni indexing (`alpha / (K - rank_0)`), stop at the first non-rejection across all `K` tested challengers in the track, and apply Holm-Bonferroni on Holdout as well whenever `>= 2` models in a track reach Holdout.

2. **Stratified Sign-Flip Permutation Gate for `equal_stratum` (`N = 100 / 40 / 30 / 30`):**
   - `Opus 5.5, Max` verified via a 2,000-run Monte Carlo null simulation (`k=1, 8`) that at `N=30` (`BRUMO 2025`, `HMMT Nov 2025`), equal-stratum percentile bootstrap understates variance by `(n-1)/n` (`0.9^30 = 4.2% < 5%`, empirical Type-I `5.3%–6.2%`), whereas a **stratified Rademacher sign-flip permutation test** on the exact `1/4`-weighted contest statistic maintains nominal size (`4.3%–5.2%`).
   - Enforce strict conjunction: require **both** item-pooled (`N=200`) **and** `equal_stratum` (`4 x 1/4`) to pass, assert exact stratum counts `100 / 40 / 30 / 30` with identical item sets across `astar`, `b0`, and `ref_qwen38_27b` (no inner-join drops), and evaluate `MATH-500` (`N=150`, `mean@8` per item) outside the 4-contest average as a retention check.

3. **Strict Anti-Overclaim Discipline on Causal Diagnostics:**
   - **3-Arm Recursion (`M_2^{rec-yield}` vs `M_2^{frozen-gen}`):** Do **not** quote the noisy ratio `"2.25x"` (`+0.88 pp / +0.39 pp`, unbounded ratio CI near zero). At matched trace count (`992`) and optimizer steps (`62`), the paired difference between recursive `M_1` and frozen `M_0` rollouts is small and statistically indistinguishable from zero (`+0.49 pp` Math, `-0.73 pp` Lean): recursive `M_1` expands pool frontier coverage (`+19` Math problems) and compresses Lean proof length (`2.04x`), while the decisive post-R1 capability jump comes from Round-3 Checkpointed-Chunk `DAPO+LUFFY` RLVR (`+4.20 pp` on `4B`, `+3.42 pp` on `9B`).
   - **4-Way Compiler Repair (`ProofNet-valid` vs `Lean-Workbook`):** Frame `ProofNet-valid` (`10/10/11/11` of `164`) strictly as *inconclusive / no detectable benefit over `direct8` at 8 attempts* (justifying withholding Holdout promotion), rather than claiming evidence of zero diagnostic effect, and treat the in-domain Search 4-way split (`compiler_repair_4+4` vs `shuffled_diag_4+4` primary content contrast) strictly as an exploratory mechanistic diagnostic.

---

## 5. Post-Holdout Adversarial Claim, Statistical & Provenance Audit

**Date (UTC):** 2026-11-19  
**Scope:** Independent verification of all `52` report metric dictionaries against raw `*_metrics.json` files (`52/52` exact matches), all `38` Holdout receipt SHA-256 digests (`38/38` exact matches), all `20` `IMMUTABLE_MANIFEST_V2.json` entries (`20/20` exact matches), `14/14` hash-chained ledgers (`VERIFIED`), and numerical replay of all `20,000`-draw stratified Rademacher sign-flip permutation tests.

1. **Transparent Protocol Reconciliation (`Qwen3-4B` Confirmation & Holm Family):**
   - `configs_v2/gate_rules_v2_math_confirm.json` and `program_v2.md` specified `min_improvement = 2.00 pp` (`alpha_family = 0.025`), whereas `run_v2_confirm_and_holdout.py` applied a stricter `3.00 pp` effect-size bar (`min_eff = 0.030`, per-track `alpha_family = 0.05`).
   - Recomputing under **both** rules: `Qwen3.5-9B` (`+13.57 pp`, `p = 6.46e-24`) and `Qwen3.5-4B` (`+9.28 pp`, `p = 1.36e-12`) pass every Holm-Bonferroni family (`alpha = 0.025` or `0.05`, per-track `K=3` or global `K=5`). `Qwen3-4B` (`+2.49 pp`, 95% CI `[+1.03, +4.05] pp`, `p = 0.00066`) passed both the `2.00 pp` config gate and `alpha = 0.025` Holm-Bonferroni, and was conservatively withheld from Protected Holdout solely by the orchestrator's `3.00 pp` filter (leaving its Holdout score strictly at `B_0`). We report this conservative filter explicitly rather than describing `3.00 pp` as pre-registered.

2. **Monte Carlo Permutation Resolution & Exploratory Cross-Scale Contrasts:**
   - Both `Qwen3.5-9B` (`+10.50 pp` pooled / `+10.80 pp` equal-stratum) and `Qwen3.5-4B` (`+7.19 pp` pooled / `+7.11 pp` equal-stratum) had `0` of `20,000` stratified Rademacher sign flips exceed the observed statistic, yielding the plus-one Monte Carlo resolution floor `p = 1/20,001 ≈ 5.0e-5` (reported as Monte Carlo `p = 5.0e-5` at simulation resolution, not an analytic exact `p`-value).
   - Cross-scale comparisons — specifically `Qwen3.5-4B (A*)` outperforming untuned `2.25x`-larger `Qwen3.5-9B (B_0)` (`+4.88 pp` pooled, `p = 0.00075`; `+4.66 pp` equal-stratum, `p = 0.00215`) and closing gap fractions toward `Qwen3.8-27B` — are labeled explicitly as **exploratory cross-model contrasts** outside the pre-registered `A* vs own B_0` promotion family.

3. **Retention Set & Single-Touch Provenance Framing:**
   - The `MATH-500` (`N=150`) evaluation (`math_holdout_retention150.jsonl`, `95.17% -> 96.00%` on `Qwen3.5-4B`, `96.08% -> 97.92%` on `Qwen3.5-9B`) is documented as a **reused V1 general-math retention check** outside the 4-contest Holdout average, not fresh Holdout evidence or formal TOST non-inferiority.
   - Governance is framed as **auditable two-phase Holdout reservation (`O_CREAT | O_EXCL` + `fsync`), comparator SHA-256 pinning, 14 hash-chained ledgers, and 38 verified receipts** (`run_v2_confirm_and_holdout.py` was also hardened to enforce `status == "RESERVED"` and `metrics_sha256` verification on reuse).




# Phase V3 Results — `Axion`-Style `2-SFT + RLVR` Scaling Across `4B`, `9B`, `27B` Dense, and `35B-A3B` Sparse MoE

This report documents **Phase V3** (`IMMUTABLE_MANIFEST_V3.json` root SHA-256 `3d86f390ae8acc2f7e1debd2930bb2be2cf8ba737f62d348632ad51cd37f8d0b` and `IMMUTABLE_MANIFEST_V3_MOE.json` root SHA-256 `d26ff13faf04fc7eb221bf81061480d345deadfdd003d3dc3a11d063f20a84dc`), expanding governed reasoning post-training in four directions:
1. **Verifiable Multi-Source Data Scaling (`2 SFT + RL`, No Mid-Training or DPO):** Stage-I multi-domain Breadth SFT followed by Stage-II model-aware Zone-of-Proximal-Development (`ZPD`) hard-frontier SFT (including cross-family-audited strengthened problems on `4B`/`9B`, strictly excluded on `27B` and `35B-A3B` to prevent self-distillation circularity) and Stage-III multi-proposal RLVR.
2. **Controlled 3-Proposal Multi-Epoch RLVR Tournament:** Rollout-matched (`ppo_epochs = 2`, exact trainer-computed `FP32` $\pi_{\text{old}}$) comparison of **`DAPO`** (`dapo_asymclip`), **`Dr.GRPO + CISPO`** (`drgrpo_cispo`), and **`GSPO + LUFFY`** (`gspo_luffy`) with intermediate checkpoint selection (`50%`, `100%`) and full `pass@1,2,4,8` capability-curve decomposition.
3. **Full `27B` Dense `FSDP` Post-Training:** End-to-end 8-GPU PyTorch `FSDP FULL_SHARD` training across all `496` linear projections (`linear_attn` + `self_attn` + `mlp`, exact `FP32` factor merge parity) on `Qwen3.8-27B`.
4. **Sparse MoE Extension (`Qwen3.8-35B-A3B`, `35B` Total / `3B` Active):** `FSDP FULL_SHARD` adaptation of all `310` unconditionally active attention and shared-expert linear projections (`r=64, alpha=128`, keeping `256` routed `top-8` experts and router gates frozen in `BF16`, `max_abs_logit_diff = 0.0` post-merge parity across multi-GPU `device_map="auto"`).

External reasoning models were used in an independent pre-registration and post-execution review loop to audit split disjointness, null-check calibration (`8/8` empirical `A/A` checks returned `ROLLBACK`), Intersection-Union Testing (`IUT`) gates against both `B_0` and prior V1/V2 champions, and causal scope boundaries.

---

## 1. Track M Discovery: Horizon-Alignment Failure Mode & Controlled Recovery (`0` Confirmation Looks Spent)

In **Round 1** (`r1_concise_postthink_sft`), training on concise post-`</think>` solution summaries (`median = 601` tokens) compressed test-time Olympiad reasoning chains **`10.8x`** (`5,577 -> 515` tokens on `9B`; `520` tokens on `4B`) and caused an immediate Search drop (`66.25% -> 50.00% avg@8` on `9B`; `64.92% -> 43.59% avg@8` on `4B`). Stage-II ZPD SFT (`50.39%`, `547` tokens) could not recover multi-step search depth once early EOS termination had been internalized.

Because this diagnostic failure occurred strictly on `Search_V3`, **zero Confirmation looks were spent** (`round1_concise_postthink_sft` remains recorded in the append-only Search ledgers). **Round 2** (`r2_s1_horizon_breadth_sft`) restored natural-stop full-horizon Olympiad derivations (`<= 10,000` tokens, strictly `< 12,288` evaluation cap) alongside clean unwrapped frontier rescues:

| Model | `B_0` (`avg@8` / `pass@8`) | Round-1 Concise SFT (`p50 = 601` tok) | Round-2 Horizon-Aligned Stage-I SFT (`<= 10K` tok) | Round-1 -> Round-2 Reversal | Confirmation Looks Spent |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B`** | `64.92%` / `86.25%` | `43.59%` (`520` test tok, `-21.33 pp`) | **`72.27%` / `89.38%` (`+7.35 pp` vs `B_0`)** | **`+28.68 pp`** | **`0`** |
| **`Qwen3.5-9B`** | `66.25%` / `87.50%` | `50.00%` (`515` test tok, `-16.25 pp`) | **`75.86%` / `90.00%` (`+9.61 pp` vs `B_0`)** | **`+25.86 pp`** | **`0`** |

*Scope note:* Because Round 2 simultaneously restored multi-thousand-token reasoning trajectories and curated natural-stop horizon-matched sources, this contrast establishes a severe **horizon-compression failure mode** and a **controlled recovery recipe** on `Search_V3`, rather than isolating token length independently of source filtering.

---

## 2. Stage-by-Stage Search Trajectories & 3-Proposal RLVR Tournament (`4B`, `9B`, `27B` Dense, `35B-A3B` MoE)

### 2.1 Track M — Competition & 2026 Research Mathematics (`v3_math_search160`, `N=160`, `k=8`, `12,288` tok)

| Model (`Scale / Active`) | `B_0` `avg@8` (`pass@8` / `stop`) | Stage-I Breadth SFT | Stage-II ZPD+Axion SFT | `DAPO` (`pct50` / `pct100`) | `Dr.GRPO+CISPO` (`pct50` / `pct100`) | `GSPO+LUFFY` (`pct50` / `pct100`) | Order-Invariant Search Winner (`Delta vs B_0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.5-4B` (`4B / 4B`)** | `64.92%` (`86.25%` / `86.17%`) | `72.27%` | **`73.67%`** *(anchor)* | `74.77%` / `75.62%` (`91.25% pass@8`) | `74.92%` / `74.69%` | `75.23%` / **`76.02%`** | **`GSPO+LUFFY pct100`: `76.02%` (`+11.10 pp`, `97.89% stop`)** |
| **`Qwen3.5-9B` (`9B / 9B`)** | `66.25%` (`87.50%` / `84.61%`) | **`75.86%`** *(anchor)* | `74.61%` | `78.52%` / `78.67%` | `76.09%` / `69.69%` (`77.11% stop`) | `78.05%` / **`80.39%`** | **`GSPO+LUFFY pct100`: `80.39%` (`+14.14 pp`, `84.38% maj@8`)** |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | `79.30%` (`91.25%` / `99.45%`) | **`79.84%`** *(anchor)* | `79.45%` | `81.72%` / `80.94%` | `81.64%` / `80.63%` (`93.13% pass@8`) | **`82.58%`** / `80.70%` | **`GSPO+LUFFY pct50`: `82.58%` (`+3.28 pp`, `86.88% maj@8`)** |
| **`Qwen3.8-35B-A3B` (`35B / 3B` MoE FSDP)** | `58.59%` (`85.00%` / `65.62%`) | `59.14%` (`66.17% stop`) | **`60.78%`** *(anchor, `67.97% stop`)* | `75.47%` / **`79.77%`** (`90.00% pass@8`, `93.36% stop`) | *Running on cluster* | *Running on cluster* | **`DAPO pct100`: `79.77%` (`+21.18 pp` so far; final tournament pending)** |

### 2.2 Track L — Formal Lean 4.24 + Mathlib Theorem Proving (`v3_lean_search140`, `N=140`, `k=8`, `2,048` tok, `100%` `#print axioms` Clean)

| Model (`Scale / Active`) | `B_0` `avg@8` (`eq_stratum` / `pass@8`) | Stage-I Breadth SFT | Stage-II ZPD SFT | `DAPO` (`pct50` / `pct100`) | `Dr.GRPO+CISPO` (`pct50` / `pct100`) | `GSPO+LUFFY` (`pct50` / `pct100`) | Order-Invariant Search Winner (`Delta vs B_0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.5-4B` (`4B / 4B`)** | `30.45%` (`13.43%` / `45.71%`) | `34.29%` (`16.71%`) | `34.91%` (`17.27%`) | `35.80%` / `35.45%` | **`36.96%`** / `35.18%` | `35.89%` / `36.16%` | **`Dr.GRPO+CISPO pct50`: `36.96%` (`+6.52 pp`, `18.83% eq_stratum`)** |
| **`Qwen3.5-9B` (`9B / 9B`)** | `33.57%` (`14.88%` / `46.43%`) | `35.00%` (`50.71% pass@8`) | `35.71%` (`55.00% pass@8`) | `37.14%` / `37.41%` | `38.12%` / **`39.55%`** | `36.43%` / `36.61%` | **`Dr.GRPO+CISPO pct100`: `39.55%` (`+5.98 pp`, `56.43% pass@8`)** |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | `33.48%` (`14.45%` / `51.43%`) | `41.70%` (`19.35%`) | `39.73%` (`19.78%`) | **`42.77%`** / `41.52%` | `41.16%` / `40.89%` | `41.61%` / `41.34%` | **`DAPO pct50`: `42.77%` (`+9.29 pp`, `21.27% eq_stratum`)** |
| **`Qwen3.8-35B-A3B` (`35B / 3B` MoE FSDP)** | `31.25%` (`15.46%` / `45.71%`) | `39.37%` (`18.39%`) | `42.14%` (`19.53%` / `57.86%`) | `41.34%` / `40.54%` (`20.58% eq`) | `41.25%` / `41.25%` (`98.30% stop`) | `42.23%` / **`42.95%`** | **`GSPO+LUFFY pct100`: `42.95%` (`+11.70 pp`, `20.91% eq_stratum`)** |

### 2.3 Methodological Findings from the 3-Proposal RLVR Tournament

1. **Sequence-Length Invariance Explains Domain-Specific RLVR Winners (`GSPO+LUFFY` on `10K` Math vs. `Dr.GRPO`/`DAPO` on Short-Horizon Lean):**
   - Among the tested recipes, **`GSPO + LUFFY` won `10K`-token competition math across all three dense scales (`4B`, `9B`, `27B`)** AND won `35B-A3B` Sparse MoE on Lean. Because `GSPO` computes the geometric sequence-level importance ratio $(\pi_\theta(y|x)/\pi_{\text{old}}(y|x))^{1/|y|}$, its effective trust region is invariant to whether a proof spans `300` or `10,000` tokens, while `LUFFY`'s `1-of-8` verified corpus rescue prevents dead zero-variance (`0/8`) groups on the hardest frontier items.
   - Conversely, **`Dr.GRPO + CISPO` won short-horizon Lean (`median ~ 200–350` tokens) on `4B` and `9B`**, where removing standard GRPO's $1/|o_i|$ normalization eliminates length bias without exploding gradient variance. On `10K`-token Math, however, summing unnormalized token gradients against a fixed $L_{\text{norm}}$ caused Epoch-2 step-size overshoot on `9B` (`76.09% -> 69.69%`, `stop_rate` dropping to `77.11%`).
2. **Intermediate Checkpoint Selection (`pct50` vs. `pct100`) Prevents Late-Epoch Over-Sharpening at Larger Scale:**
   - On `Qwen3.8-27B` Track M, **Epoch-1 (`pct50`) outperformed Epoch-2 (`pct100`) across all three RLVR objectives** (`DAPO`: `81.72% -> 80.94%`; `Dr.GRPO`: `81.64% -> 80.63%`; `GSPO+LUFFY`: `82.58% -> 80.70%`), and likewise won on `27B` Lean (`42.77% -> 41.52%`) and `4B` Lean (`36.96% -> 35.18%`, where Epoch 2 doubled forgotten theorems to `7.14%`). Evaluating intermediate checkpoints on disjoint `Search_V3` is essential for two-epoch RLVR.
3. **Capability-Slope Compression (`slope_8_1 = pass@8 - pass@1`):**
   - On `Qwen3.5-9B` Math, Stage-I Breadth SFT lifted `pass@8` (`87.50% -> 90.00%`) with only `1.87%` forgotten items, after which `GSPO + LUFFY` RLVR cut `capability_slope_8_1` more than in half (**`21.25% -> 10.23%`**), lifting `avg@8` (`pass@1`) to **`80.39%`** — surpassing zero-shot `Qwen3.8-27B` (`79.30%`) on `Search_V3`.

---

## 3. Intersection-Union Confirmation (`Confirm_V3`) & Single-Touch Protected Holdout (`Holdout_V3`)

Every Phase V3 candidate faced an **Intersection-Union Confirmation Gate (`IUT`)**: to unseal `Holdout_V3`, a candidate had to beat **both** its untuned base `B_0` **and** any previously promoted V1/V2 champion (`current_incumbent_id`) by at least `+1.50 pp` LCB at Holm/Bonferroni-adjusted $\alpha$. When a candidate improved over `B_0` (and even numerically edged the prior champion) without clearing the strict `+1.50 pp` incumbent LCB margin, `governed_loop_guard_v37.py` executed `CONFIRMED_ROLLBACK_NO_HOLDOUT_TOUCH` and preserved the incumbent honestly.

### 3.1 Track L — Formal Lean 4.24 Confirmation (`N=180`) & Sealed Single-Touch Holdout (`N=225`: `arXivLean 2026` + `FormalMATH-Lite`)

| Model | `Confirm_V3` (`N=180`) `B_0 -> Candidate` (`eq_stratum` / `avg@8`) | IUT Gate Verdict | Sealed `Holdout_V3` (`N=225`) `B_0 -> Promoted` `avg@8` | Sealed `Holdout_V3` `eq_stratum` `avg@8` | Sealed `Holdout_V3` `pass@8` | `stop_rate` (`B_0 -> Promoted`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B`** | `17.61% -> 23.84%` (`+6.23 pp` vs `B_0`; `23.84%` vs V1 `23.12%`) | **Hold V1 `A*`** *(IUT margin `< +1.5 pp` vs V1)* | **`15.28% -> 19.39%` (`+4.11 pp`, V1 `A*`)** | `15.86% -> 19.34%` (`+3.48 pp`) | `24.00% -> 27.56%` | `46.56% -> 99.17%` |
| **`Qwen3.5-9B`** | **`22.16% -> 29.71%` (`+7.55 pp`)** / **`28.06% -> 34.86%` (`+6.81 pp`)** | **`PROMOTE` (`HOLDOUT_SEAL` Finalized)** | **`16.22% -> 19.28%` (`+3.06 pp`)** | **`16.82% -> 18.75%` (`+1.93 pp`)** | **`24.44% -> 27.11%` (`+2.67 pp`)** | `48.72% -> 97.67%` |
| **`Qwen3.8-27B` (`FSDP`)** | **`22.49% -> 29.25%` (`+6.76 pp`)** / **`27.08% -> 34.58%` (`+7.50 pp`)** | **`PROMOTE` (`HOLDOUT_SEAL` Finalized)** | **`16.89% -> 22.06%` (`+5.17 pp`)** | **`17.48% -> 22.12%` (`+4.64 pp`)** | **`26.22% -> 31.56%` (`+5.33 pp`)** | **`43.39% -> 84.83%`** |

- **`Qwen3.8-27B` Formal Proving Breakthrough:** Full 8-GPU `FSDP` `2-SFT + DAPO (pct50)` lifted `27B` on `Holdout_V3` by **`+5.17 pp` `avg@8`** and **`+5.33 pp` `pass@8` (`59 -> 71 / 225` theorems proved, `100%` `#print axioms` clean)**, highlighted by **`formalmath_algebra` (`35.03% -> 45.89%`, `+10.86 pp`)** and unlocking non-zero research-level formalization solves on **`arxivlean_2026` (`0.00% -> 1.76%`)**.

### 3.2 Track M — Competition & 2026 Research Math Confirmation (`N=300`) & Sealed Single-Touch Holdout (`N=194`: `SMT 25`, `HMMT Feb 25`, `APEX 25`, `arXivMath 2026`)

| Model | `Confirm_V3` (`N=300`) `B_0 -> V2 A* -> V3 Cand` `avg@8` (`pass@8`) | IUT Gate Verdict | Sealed `Holdout_V3` (`N=194`) `B_0 -> Promoted` `avg@8` | Sealed `Holdout_V3` `eq_stratum` `avg@8` | Sealed `Holdout_V3` `maj@8` / `pass@8` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B`** | `68.79% -> 78.42% -> 80.67%` (`95.33% pass@8`) | **Hold V2 `A*`** *(`+11.88 pp` vs `B_0`; `+2.25 pp` vs V2 `INCONCLUSIVE`)* | **`24.16% -> 29.38%` (`+5.22 pp`, V2 `A*`)** | **`28.22% -> 33.80%` (`+5.58 pp`)** | `36.60% -> 39.69%` / `48.97% -> 52.06%` |
| **`Qwen3.5-9B`** | **`69.75% -> 80.42% -> 84.29%` (`96.33% pass@8`)** | **`PROMOTE` vs BOTH `B_0` & V2 `A*` (`HOLDOUT_SEAL` Finalized)** | **`23.90% -> 36.08%` (`+12.18 pp` vs `9B B_0`; `+1.67 pp` vs V2 `34.41%`)** | **`27.57% -> 40.80%` (`+13.23 pp` vs `9B B_0`)** | **`38.14% -> 45.36%` (`+7.22 pp`)** / **`48.97% -> 56.70%` (`+7.73 pp`)** |
| **`Qwen3.8-27B` (`FSDP`)** | `83.67% -> — -> 83.79%` (`B_0 pass@8 = 95.00%`) | **Hold `B_0`** *(`CONFIRMED_ROLLBACK_NO_HOLDOUT_TOUCH`)* | `40.40%` *(zero-shot `B_0` retained)* | `46.45%` *(zero-shot `B_0`)* | `45.88%` / `59.79%` *(zero-shot `B_0`)* |

- **Cross-Scale Anchoring (`Qwen3.5-9B` vs. `Qwen3.8-27B`):** On `Search_V3`, Phase-V3 `Qwen3.5-9B` (`80.39% avg@8`, `84.38% maj@8`) surpassed zero-shot `Qwen3.8-27B` (`79.30% avg@8`, `83.75% maj@8`). On the substantially harder `Holdout_V3` (`2025` USAMO/IMO-tier + `2026` `arXivMath`), `9B`'s sealed `+12.18 pp` `avg@8` gain (`23.90% -> 36.08%`) is paired against its own `9B B_0` baseline, closing **`73.8%` of the `9B -> 27B` `avg@8` gap** (`40.40%` on `27B B_0`) and **`93.3%` of the `maj@8` gap** (`45.36%` vs `45.88%`).
- **Gate Discipline on `Qwen3.8-27B` Track M:** Although `GSPO + LUFFY (pct50)` gained `+3.28 pp` on `Search_V3` (`79.30% -> 82.58%`), `Confirm_V3` already had near-ceiling `B_0` coverage (`pass@8 = 95.00%`, leaving only `15/300` unsolved problems), and `avg@8` was essentially tied (`83.79%` vs `83.67%` for `B_0`). Holding `B_0` properly reflects "no Confirmation win established" (reinforced longitudinally on `BeyondAIME-100`: `50.00%` for `B_0` vs `48.13%` for `r2_s3`, while `CMIMC-40` rose `49.69% -> 53.75%` and `HMMT` `pass@8` rose `90.00% -> 93.33%`), without claiming significant regression.

---

## 4. Six-Benchmark Longitudinal Generalization (`MiniF2F`, `ProofNet`, and `2025` 4-Contest Math)

To verify that Phase V3 gains generalize cleanly back onto external benchmarks from Phases V1 and V2 without catastrophic forgetting, we evaluated `B_0` vs. Phase V3 winners across **6 external benchmarks**:

| Model | `MiniF2F-test` (`N=244`) `avg@8` (`pass@8`) | `ProofNet-test` (`N=167`, OOD Undergrad) `avg@8` (`pass@8`) | `2025` 4-Contest Math (`N=200` Pooled) `avg@8` | `BeyondAIME` (`100`) | `CMIMC 25` (`40`) | `BRUMO 25` (`30`) | `HMMT 25` (`30`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B` (`B_0 -> V3`)** | `28.33% -> 29.30%` | `1.84% -> 2.38%` | **`36.25% -> 43.69%` (`+7.44 pp`)** | `29.88% -> 36.88%` | `34.06% -> 37.81%` | `56.67% -> 67.08%` (`93.3% pass@8`) | `40.00% -> 50.83%` |
| **`Qwen3.5-9B` (`B_0 -> V3`)** | **`30.28% -> 33.61%` (`+3.33 pp`)** | **`2.15% -> 5.29%` (`2.46x`! `4.29% -> 9.82% pass@8`)** | **`38.00% -> 49.50%` (`+11.50 pp`)** | **`31.75% -> 44.50%` (`+12.75 pp`)** | **`33.12% -> 43.13%` (`+10.01 pp`)** | **`60.00% -> 66.25%`** | **`43.33% -> 57.92%` (`+14.59 pp`)** |
| **`Qwen3.8-27B` (`B_0 -> V3`)** | **`33.66% -> 35.96%` (`+2.30 pp`, `48.36% -> 51.64% pass@8`)** | **`2.68% -> 3.45%` (`+0.77 pp`, `7.36% -> 9.82% pass@8`, `stop 29.8% -> 88.8%`)** | `57.25%` (`B_0` held; `r2_s3 = 56.69%`) | `50.00%` (`48.13%`) | `49.69%` (`53.75%`) | `78.75%` (`75.00%`) | `66.25%` (`66.25%`, `93.3% pass@8`) |

- **Breaking the Phase V2 `ProofNet` OOD Wall:** In Phase V2, single-source `MiniF2F`-heavy training failed to transfer to undergraduate `ProofNet-test` (`ROLLBACK` on `ProofNet-valid`). Phase V3's multi-domain `Mathlib4` (`Algebra`, `Analysis`, `Topology`, `LinearAlgebra`, `GroupTheory`, `RingTheory`) + `Goedel` `2-SFT + RLVR` corpus **more than doubled `Qwen3.5-9B` `ProofNet-test` `avg@8` (`2.15% -> 5.29%`, `pass@8` `4.29% -> 9.82%`)** and lifted **`Qwen3.8-27B` `MiniF2F-test` `pass@8` past `51.64%` (`126/244`)** while raising `27B` `ProofNet` completion rate from `29.83%` to `88.80%`.

---

## 5. Sparse MoE (`Qwen3.8-35B-A3B`) vs. Dense (`Qwen3.8-27B`) Post-Training Dynamics & Active-Capacity Hypothesis

Comparing **`Qwen3.8-35B-A3B`** (`35B` total parameters, `3B` active per token across `256` routed experts `top-8`, `FSDP` LoRA `r=64` on `310` unconditionally active attention + shared-expert projections) against **`Qwen3.8-27B` Dense** (`27B` total / `27B` active) reveals a sharp domain asymmetry between short-horizon formal proving and long-horizon mathematical derivations:

| Horizon Regime | Empirical `35B-A3B` Sparse MoE Trajectory (`Search_V3`) | Comparison vs. `27B` Dense (`Search_V3`) |
| :--- | :--- | :--- |
| **Short-Horizon Formal Proving (`Track L`, `<= 2,048` tok, `median ~ 280` tok)** | Zero-shot `B_0 = 31.25% avg@8` (`15.46% eq_stratum`, already edging `27B` `14.45%` on rare Mathlib strata). **`2-SFT` alone surged `+10.89 pp` to `42.14% avg@8` (`57.86% pass@8`!)**, and **`GSPO + LUFFY (pct100)` reached `42.95% avg@8` (`+11.70 pp`, `20.91% eq_stratum`)**. | **Matches/edges post-trained `27B` Dense (`42.95%` vs `42.77% avg@8`) at `9x` lower active FLOPs/token.** |
| **Long-Horizon Olympiad Math (`Track M`, `<= 12,288` tok)** | Zero-shot `B_0` exhibited high latent knowledge reach (**`85.00% pass@8`, `79.37% maj@8`**) but severe multi-thousand-token looping (`stop_rate = 65.62%`, `avg@8 = 58.59%`, `slope_8_1 = 26.41%`). **`2-SFT` barely shifted termination (`60.78% avg@8`, `67.97% stop_rate`)**, whereas **`DAPO` online RLVR (`pct100`) drove `stop_rate` `+27.74 pp` (`65.62% -> 93.36%`) and `avg@8` `+21.18 pp` (`58.59% -> 79.77%`, `90.00% pass@8`, `slope_8_1 = 10.23%`)**! | **Surpasses zero-shot `27B` Dense (`79.77%` vs `79.30% avg@8`) once RLVR suppresses long-horizon non-termination.** |

- **Mechanistic Active-vs-Total Capacity Hypothesis:** Because `35B` total and `3B` active parameters are coupled within one frozen-router MoE architecture (rather than varied in an independent factorial grid), we frame this divergence as a falsifiable hypothesis: **total expert capacity (`35B`) governs premise/knowledge breadth (`pass@8`: `57.86%` on Lean, `85.00% -> 90.00%` on Math), whereas small active per-token capacity (`3B`) bottlenecks autoregressive state tracking over `5K–12K`-token trajectories (`65.62%` `stop_rate`) in a way that off-policy SFT (`+2.19 pp`) fails to fix, making online verifier RL (`DAPO` `+18.99 pp` over `2-SFT`) indispensable for long-horizon MoE reasoning.**

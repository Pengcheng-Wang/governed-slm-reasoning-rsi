# Detailed Experimental Results & Causal Analyses — Phases V1 & V2 (`3.8B–9B` SLMs)

This report contains the complete experimental tables, causal mechanism isolations, cross-architecture boundaries, and statistical audit reconciliations for **Phase V1** (`IMMUTABLE_MANIFEST.json`, `33` files, `6` ledgers, `28` Holdout receipts) and **Phase V2** (`IMMUTABLE_MANIFEST_V2.json`, `20` files, Conjunctive `v2 + v3.6` governance, `14` ledgers, `38` Holdout receipts).

---

## 1. Phase V1 — Initial Two-Plane SLM Study (`Qwen3.5-4B` & `Qwen3.5-9B`)

### 1.1 Plane-1 Decoding & Channel Calibration (`N=128`, `k=8`)

| Track & Split | Model | Uncalibrated `thinking` (`avg@8` / Truncation) | Calibrated `direct` Protocol | Calibrated `B_0` `avg@8` | Calibrated `B_0` `pass@8` | Calibrated Truncation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track L (`lean_search128`)** | `Qwen3.5-4B` | `3.81%` (`91.60%` @ `8K`) | `direct` (`2,048` tok, `T=0.7`) | **`33.79%`** | **`60.16%`** (`77/128`) | **`0.78%`** |
| **Track L (`lean_search128`)** | `Qwen3.5-9B` | `23.05%` (`69.43%` @ `8K`) | `direct` (`2,048` tok, `T=0.7`) | **`41.99%`** | **`72.66%`** (`93/128`) | **`0.20%`** |
| **Track L (`lean_search128`)** | `Qwen3.8-27B` (Ref) | — | `direct` (`2,048` tok, `T=0.7`) | **`40.53%`** | **`76.56%`** (`98/128`) | **`7.81%`** |
| **Track M (`math_search128`)** | `Qwen3.5-4B` | `43.55%` (`56.25%` @ `16K`) | `direct12k` (`12,288` tok, `T=0.7`) | **`76.27%`** | **`91.41%`** (`117/128`) | **`13.87%`** |
| **Track M (`math_search128`)** | `Qwen3.5-9B` | `54.69%` (`44.34%` @ `16K`) | `direct12k` (`12,288` tok, `T=0.7`) | **`76.76%`** | **`90.62%`** (`116/128`) | **`14.16%`** |
| **Track M (`math_search128`)** | `Qwen3.8-27B` (Ref) | — | `direct12k` (`12,288` tok, `T=0.7`) | **`87.01%`** | **`93.75%`** (`120/128`) | **`0.20%`** |

### 1.2 Phase V1 Confirmation (`Holm-Bonferroni K=2`) & Single-Touch Holdout

- **Hybrid Gated-DeltaNet LoRA vs. Full-FT (`L1a` vs `L1b` on `Qwen3.5-4B` Lean):** Full-projection LoRA (`r=32`, all `248` linear projections across `self_attn`, `linear_attn`, and `mlp`, `FP32` exact merge) beat Full-FT (`49.12%` vs `46.78%` `lean_avg_at_8`, `78.91%` vs `74.22%` `pass@8`).
- **Track L Protected Holdout (`MiniF2F-v2s test`, `N=244`, Lean `v4.24.0 + Mathlib f897ebcf`):**
  - Promoted `Qwen3.5-4B` (`l2_q4b_restem_repair`): **`27.66% -> 33.15%` `lean_avg_at_8` (`+5.48 pp`, 95% CI `[+2.92, +8.09] pp`)**, **`43.44% -> 45.90%` `pass@8` (`106 -> 112/244`, `100%` `#print axioms` clean)** — outperforming untuned `Qwen3.5-9B` (`30.02%`) and numerically matching untuned `Qwen3.8-27B` (`32.27%` `avg@8`, `48.77%` `pass@8`). All `2,048`-token single-pass whole-proof models scored `0/128` on `PutnamBench`.
- **Track M Protected Holdout (`N=213` Contest Suite: `AIME 2025`, `AIME 2026`, `HMMT Feb 2026`, `OlympiadBench` + `N=150` `MATH-500` Retention):**
  - Promoted `Qwen3.5-4B` (`m3_q4b_dapo_luffy`): **`55.52% -> 59.45%` pooled `math_avg_at_8` (`+3.93 pp`, 95% CI `[+1.47, +6.40] pp`)**, highlighted by **`AIME 2025`** (`37.08% -> 46.25% avg@8`, **`70.00% -> 86.67% pass@8` [`21 -> 26/30`]**) and **`OlympiadBench`** (`67.71% -> 71.77% avg@8`), with `MATH-500` retention `94.83% -> 95.83%`.
  - In V1, `Qwen3.5-9B` (`m3_q9b_dapo_luffy`) improved Confirmation `70.25% -> 73.75%` (`+3.50 pp`, `p = 6.11e-5`) and `stop_rate` `78.00% -> 83.92%`, but was held back by a `2,688`-token training memory cap and a static `85%` `stop_rate` floor — motivating Phase V2's Checkpointed-Chunk `12k` RLVR kernel.

---

## 2. Phase V2 — Multi-Architecture & Multi-Benchmark Expansion (`0` External Teacher Traces)

### 2.1 Plane-2 Search Trajectories Across `8` `(Model x Track)` Pipelines

| Track & Split | Model (`Architecture / Channel`) | `B_0` `avg@8` (`pass@8`) | Round-1 `ReST^EM` | Round-2 Recursive `ReST^EM` | Round-3 Chunked-`12k` `DAPO + LUFFY` | Search Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track L (`v2_lean_search120`)** | **`Qwen3.5-9B` (`Hybrid GDN`, `direct`)** | `48.02%` (`63.33%`) | `50.73%` | `53.13%` | **`55.52%` (`+7.50 pp`)** | **`3/3` `PROVISIONAL_KEEP`** |
| **Track L (`v2_lean_search120`)** | **`Qwen3.5-4B` (`Hybrid GDN`, `direct`)** | `40.73%` (`55.00%`) | `46.77%` | `48.54%` | **`49.58%` (`+8.85 pp`)** | **`3/3` `PROVISIONAL_KEEP`** |
| **Track L (`v2_lean_search120`)** | `Qwen3-4B` (`Pure Dense`, `think6k`) | `5.42%` (`13.33%`) | `5.83%` | `5.63%` | `5.10%` | `3/3` `ROLLBACK` (`B_0` kept) |
| **Track L (`v2_lean_search120`)** | `Phi-4-mini-reasoning` (`3.8B`, `think6k`) | `0.31%` (`2.50%`) | — | — | — | Pre-registered skip (`learnable < 10%`) |
| **Track L References** | `Goedel-Prover-V2-8B` / `Qwen3.8-27B` | `58.44%` / `41.98%` | — | — | — | Zero-shot references |
| **Track M (`v2_math_search128`)** | **`Qwen3.5-9B` (`Hybrid GDN`, `direct12k`)** | `37.60%` (`54.69%`) | `42.29%` | `45.90%` | **`49.32%` (`+11.72 pp`)** | **`3/3` `PROVISIONAL_KEEP`** |
| **Track M (`v2_math_search128`)** | **`Qwen3.5-4B` (`Hybrid GDN`, `direct12k`)** | `36.72%` (`56.25%`) | `41.50%` | `42.29%` | **`45.70%` (`+8.98 pp`)** | **`m1` & `m3` `PROVISIONAL_KEEP`** |
| **Track M (`v2_math_search128`)** | **`Qwen3-4B` (`Pure Dense`, `think16k`)** | `35.94%` (`44.53%`) | `36.62%` | **`37.01%` (`+1.07 pp`)** | `36.43%` | **`m2` `PROVISIONAL_KEEP`** |
| **Track M (`v2_math_search128`)** | `Phi-4-mini-reasoning` (`3.8B`, `think12k`) | `25.88%` (`39.84%`) | `25.68%` | `25.29%` (`12k`) | `24.41%` | `3/3` `ROLLBACK` (`B_0` kept) |
| **Track M Reference** | `Qwen3.8-27B` (`direct12k`) | `51.95%` (`65.62%`) | — | — | — | Zero-shot reference |

### 2.2 Controlled Causal Isolations (`Qwen3.5-4B`)

1. **3-Arm Recursive vs. Frozen-Generator Control (Round 2 from `M_0`, Matched `992` Traces & `62` Steps):**
   - *Track M (`v2_math_search128`):* `M_2^{frozen-gen} = 41.89%`, `M_2^{rec-full} = 42.29%` (`+0.39 pp`, CI `[-1.95, +2.73]`), `M_2^{rec-yield} = 42.38%` (`+0.49 pp`, CI `[-2.05, +2.93]`).
   - *Track L (`v2_lean_search120`):* `M_2^{frozen-gen} = 49.79%`, `M_2^{rec-yield} = 49.06%` (`-0.73 pp`, CI `[-5.21, +3.65]`), `M_2^{rec-full} = 48.54%` (`-1.25 pp`, CI `[-5.42, +2.71]`).
   - *Conclusion:* At matched trace counts and steps, `M_2^{rec-yield}` and `M_2^{frozen-gen}` are statistically indistinguishable: recursive `M_1` expands pool frontier coverage (`+19` Math problems) and compresses Lean proof length (`median_tokens`: `232.5 -> 122.5`), while Round-3 Checkpointed-Chunk `DAPO + LUFFY` online RLVR drives the decisive post-Round-1 jump (`+4.20 pp` on `4B`, `+3.42 pp` on `9B` Math).
2. **4-Way Call-Matched (`<= 8` Calls) Lean Compiler-Diagnostic Repair Isolation:**
   - *OOD `ProofNet-valid` (`N=164`):* `direct8 = 6.10%` (`10/164`), `compiler_repair_4+4 = 6.10%` (`10/164`), `masked_retry_4+4 = 6.71%` (`11/164`), `shuffled_diag_4+4 = 6.71%` (`11/164`) -> no detectable gain over `direct8` at 8 attempts (repair promotion withheld).
   - *Exploratory In-Domain `v2_lean_search120` (`N=120`):* `direct8 = 60.83%` (`73/120`, `8.00` calls) vs `57.50%` (`69/120`, `5.84` calls) across all three `4+4` repair variants, showing failed-draft anchoring on `4B` single-turn self-repair relative to fresh independent sampling.

### 2.3 Conjunctive `v2 + v3.6` Confirmation (`Omni-MATH Tier 3–4` `N=256` & `ProofNet-valid` `N=164`)

| Track & Split | Model | `B_0 -> A*` `avg@8` (`pass@8`) | Paired Delta (95% CI) | One-Sided `p` | `stop_rate` (`B_0 -> A*`) | Gate Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track M (`Omni-MATH`, `N=256`)** | **`Qwen3.5-9B`** | **`33.45% -> 47.02%`** (`58.98% -> 65.23%`) | **`+13.57 pp` (`[+10.99, +16.26]`)** | **`6.46e-24`** | **`51.42% -> 79.25%`** | **`PROMOTE`** |
| **Track M (`Omni-MATH`, `N=256`)** | **`Qwen3.5-4B`** | **`34.13% -> 43.41%`** (`58.20% -> 64.84%`) | **`+9.28 pp` (`[+6.74, +11.96]`)** | **`1.36e-12`** | **`62.35% -> 86.57%`** | **`PROMOTE`** |
| **Track M (`Omni-MATH`, `N=256`)** | `Qwen3-4B` (`think16k`) | `29.79% -> 32.28%` (`41.41% -> 42.97%`) | `+2.49 pp` (`[+1.03, +4.05]`) | `6.58e-4` | `46.53% -> 53.71%` | **Withheld (`< 3.00 pp` filter)** |
| **Track L (`ProofNet-valid`, `N=164`)** | `Qwen3.5-9B` | `3.35% -> 3.89%` (`7.32% -> 9.76%`) | `+0.53 pp` (`[-0.53, +1.68]`) | `p_exact = 0.500` | `86.28% -> 99.16%` | **`ROLLBACK` (`B_0` kept)** |
| **Track L (`ProofNet-valid`, `N=164`)** | `Qwen3.5-4B` | `2.21% -> 2.90%` (`6.71% -> 7.93%`) | `+0.69 pp` (`[-0.53, +2.06]`) | `p_exact = 0.363` | `86.74% -> 99.54%` | **`ROLLBACK` (`B_0` kept)** |

### 2.4 Phase V2 Single-Touch Protected Holdout (`38` Receipts, `N=200` `2025` Contests & `N=167` `ProofNet-test`)

| Contest / Metric | `Phi-4-mini B_0` | `Qwen3-4B B_0` | `Qwen3.5-4B B_0` | **`Qwen3.5-4B A*`** (`Delta vs 4B B_0`) | `Qwen3.5-9B B_0` | **`Qwen3.5-9B A*`** (`Delta vs 9B B_0`) | `Qwen3.8-27B` (Untuned Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BeyondAIME` (`N=100`)** | `14.38%` | `27.63%` | `29.50%` | **`37.75%` (`+8.25 pp`)** | `32.25%` | **`42.13%` (`+9.88 pp`)** | `50.75%` |
| **`CMIMC 2025` (`N=40`)** | `20.00%` | `30.00%` | `37.50%` | **`38.13%` (`+0.63 pp`)** | `35.00%` | **`46.25%` (`+11.25 pp`)** | `53.44%` |
| **`BRUMO 2025` (`N=30`)** | `33.75%` | `51.67%` | `53.75%` | **`61.67%` (`+7.92 pp`)** | `57.92%` | **`71.67%` (`+13.75 pp`)** | `77.50%` |
| **`HMMT Nov 2025` (`N=30`)** | `16.67%` | `37.50%` | `38.75%` | **`50.42%` (`+11.67 pp`)** | `44.17%` | **`52.50%` (`+8.33 pp`)** | `67.08%` |
| **Item-Pooled `avg@8` (`N=200`)** | `18.75%` | `33.19%` | `36.13%` | **`43.31%` (`+7.19 pp`, CI `[+4.81, +9.63]`)** | `38.44%` | **`48.94%` (`+10.50 pp`, CI `[+7.88, +13.19]`)** | `57.75%` |
| **Equal-Stratum `avg@8` (`4 x 1/4`)** | `21.20%` | `36.70%` | `39.88%` | **`46.99%` (`+7.11 pp`, CI `[+4.29, +9.85]`)** | `42.33%` | **`53.14%` (`+10.80 pp`, CI `[+7.83, +13.86]`)** | `62.19%` |
| **Item-Pooled `pass@8` (`N=200`)** | `33.00%` | `52.50%` | `66.00%` | **`70.50%` (`+4.50 pp`)** | `70.00%` | **`74.00%` (`+4.00 pp`)** | `82.00%` |
| **Item-Pooled `stop_rate` (`N=200`)** | `33.50%` | `44.88%` | `66.50%` | **`88.00%` (`+21.50 pp`)** | `55.38%` | **`80.88%` (`+25.50 pp`)** | `98.13%` |
| **`ProofNet-test` `avg@8` / `pass@8` (`N=167`)** | `0.00% / 0.00%` | `0.07% / 0.60%` | `1.72% / 4.19%` | *(`B_0` retained)* | `2.69% / 7.78%` | *(`B_0` retained)* | `2.62% / 7.19%` (`Goedel-8B: 7.86% / 12.57%`) |

### 2.5 Audit & Protocol Transparency Notes
- **Confirmation Threshold & Multiplicity:** `configs_v2/gate_rules_v2_math_confirm.json` specified `min_improvement = 2.00 pp` (`alpha = 0.025`), whereas `run_v2_confirm_and_holdout.py` applied a stricter `3.00 pp` filter (`alpha = 0.05` per track). `Qwen3.5-9B` (`p = 6.46e-24`) and `Qwen3.5-4B` (`p = 1.36e-12`) pass under both family definitions; `Qwen3-4B` (`+2.49 pp`, `p = 6.58e-4`) passed the `2.00 pp` rule and was conservatively withheld from Holdout by the `3.00 pp` orchestrator bar.
- **Monte Carlo Sign-Flip Resolution:** `p = 5.0e-5` on Holdout reflects `0` of `20,000` stratified Rademacher sign flips exceeding the observed statistic (`1 / 20,001` simulation resolution).
- **Exploratory Cross-Scale Contrast & Retention Scope:** `Qwen3.5-4B A*` beating untuned `Qwen3.5-9B B_0` (`+4.88 pp` pooled, `p = 0.00075`; `+4.66 pp` equal-stratum, `p = 0.00215`) is an exploratory cross-model contrast outside the `A* vs own B_0` gate family. `MATH-500` (`N=150`: `95.17% -> 96.00%` on `4B`, `96.08% -> 97.92%` on `9B`) reuses the V1 general-math retention set outside the 4-contest average.

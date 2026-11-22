# Phase V3 Results — `2-SFT + RLVR` & Zero-SFT Crossovers Across `4B`, `9B`, `27B` Dense and `30B-A3B`, `35B-A3B` Sparse MoE

This report documents **Phase V3** (`IMMUTABLE_MANIFEST_V3.json` root SHA-256 `7940ee090af186cace0dd32518866315716c7ff0d6859e3603ff2f3e52dfaa94` and `IMMUTABLE_MANIFEST_V3_MOE.json` root SHA-256 `da03dc170a7062e7b50e8359e9158b61f31664cbd23a7244b27a3da730ebad81`), spanning **all 10 `(model, track)` cells** across Dense models (`Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B` `FSDP`) and two `3B`-active Sparse Mixture-of-Experts architectures (`Qwen3-30B-A3B` and `Qwen3.5-35B-A3B`, internal config slug `qwen38_35b_a3b`):

1. **Verifiable Multi-Source Data Scaling (`2 SFT + RL` & Zero-SFT `RLVR` Crossovers, No Mid-Training or DPO):** Stage-I multi-domain Breadth SFT, Stage-II model-aware Zone-of-Proximal-Development (`ZPD`) hard-frontier SFT (excluding same-family teacher traces on `27B` and `35B-A3B` to prevent self-distillation circularity), and Stage-III multi-proposal RLVR — paired with controlled **`Pure B_0 -> RLVR`** vs. **`Breadth SFT + RLVR`** crossovers on both tracks.
2. **Controlled 3-Proposal Multi-Epoch RLVR Tournament:** Rollout-matched (`ppo_epochs = 2` alongside 1-epoch trust-region ablations, exact trainer-computed `FP32` $\pi_{\text{old}}$) comparison of **`DAPO`** (`dapo_asymclip`), **`Dr.GRPO + CISPO`** (`drgrpo_cispo`), and **`GSPO + LUFFY`** (`gspo_luffy`) with intermediate checkpoint selection (`50%`, `100%`) and full `pass@1,2,4,8` capability-curve decomposition.
3. **Full `27B` Dense & Frozen-Router `3B`-Active Sparse MoE `FSDP` Post-Training:**
   - **`Qwen3.8-27B` Dense (`27B` active):** 8-GPU PyTorch `FSDP FULL_SHARD` across all `496` linear projections (`linear_attn` + `self_attn` + `mlp`).
   - **`Qwen3-30B-A3B` Sparse MoE (`30B` total / `3B` active, pure attention, `128` routed experts `top-8`, concise non-thinking prior):** 8-GPU `FSDP FULL_SHARD` LoRA (`r=64, alpha=128`) on `192` unconditionally active `self_attn` projections (`0.17%` of total weights; `0` routed experts modified).
   - **`Qwen3.5-35B-A3B` Sparse MoE (`35B` total / `3B` active, Hybrid Gated-DeltaNet + shared expert, `256` routed experts `top-8`, native long-CoT prior):** 8-GPU `FSDP FULL_SHARD` LoRA (`r=64, alpha=128`) on `310` unconditionally active attention, GDN, and shared-expert projections (`0.38%` of total weights; `0` routed experts modified).

Multiple independent external reasoning models were used in pre-registration and post-execution review loops to audit split disjointness, empirical `A/A` null checks (`10/10` across all 5 models x 2 tracks returned `ROLLBACK`), Intersection-Union Testing (`IUT`), Bonferroni multiplicity (`max_confirm_looks = 2`), weighting sensitivity, and causal scope boundaries.

---

## 1. Track M Horizon-Alignment Failure Mode & Guardrail Recovery

### 1.1 Concise Post-Think SFT Compression on Long-CoT Dense SLMs (`0` Confirmation Looks Spent)

In **Round 1** (`r1_concise_postthink_sft`), supervising long-CoT models on concise post-`</think>` solution summaries (`median = 601` tokens) compressed test-time Olympiad reasoning chains **`10.8x`** (`5,577 -> 515` tokens on `9B`; `520` tokens on `4B`) and caused an immediate `Search_V3` drop (`66.25% -> 50.00% avg@8` on `9B`; `64.92% -> 43.59% avg@8` on `4B`). **Round 2** (`r2_s1_horizon_breadth_sft`) restored natural-stop full-horizon Olympiad derivations (`<= 10,000` tokens, `< 12,288` evaluation cap), spending **zero Confirmation looks**:

| Model | `B_0` (`avg@8` / `pass@8`) | Round-1 Concise SFT (`p50 = 601` tok) | Round-2 Horizon-Aligned Stage-I SFT (`<= 10K` tok) | Round-1 -> Round-2 Reversal | Confirmation Looks Spent |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B`** | `64.92%` / `86.25%` | `43.59%` (`520` test tok, `-21.33 pp`) | **`72.27%` / `89.38%` (`+7.35 pp` vs `B_0`)** | **`+28.68 pp`** | **`0`** |
| **`Qwen3.5-9B`** | `66.25%` / `87.50%` | `50.00%` (`515` test tok, `-16.25 pp`) | **`75.86%` / `90.00%` (`+9.61 pp` vs `B_0`)** | **`+25.86 pp`** | **`0`** |

*Scope note:* Because Round 2 simultaneously restored multi-thousand-token trajectories and filtered natural-stop horizon-matched sources, this contrast establishes an empirical **horizon-compression failure mode** and **controlled recovery recipe** rather than isolating length independently of source composition.

### 1.2 Inverse Horizon Mismatch on Concise Non-Thinking `Qwen3-30B-A3B` (`stop_rate` Guardrail Catch)

The symmetric inverse failure mode emerged on **`Qwen3-30B-A3B` Track M**: its untuned `B_0` solves concisely (`median = 1,320` tokens on `Confirm_V3`, `stop_rate = 99.83%`). Supervising `30B-A3B` on `10,000`-token long-CoT `2-SFT + GSPO+LUFFY` (**Round 2**) improved `Search_V3` (`57.42% -> 62.11% avg@8`, `61.25% -> 70.63% maj@8`) and raised `Confirm_V3` **Look 1** by **`+3.38 pp` (`60.54% -> 63.92% avg@8`, `83.33% -> 89.00% pass@8`)**, *but* lowered `stop_rate` from `99.83%` to `96.71%` (`-3.13 pp > 2.50 pp` safety cap).
- **Guardrail Validation:** `governed_loop_guard_v37.py` automatically executed a fail-closed **`ROLLBACK`** (`0` candidate Holdout evaluations). External longitudinal scoring confirmed the guardrail was essential: on harder 2025 contests, that long-CoT checkpoint suffered severe truncation (`CMIMC 2025` `stop_rate`: `97.50% -> 89.38%`, `avg@8`: `18.44% -> 15.94%`; `HMMT Nov 2025` `stop_rate`: `97.50% -> 82.50%`).
- **Round-3 Horizon-Disciplined Recovery:** Testing both `2-SFT + Overlong-Penalized Compact RLVR` (`--max-gen-tokens 6144 --overlong-window 2560`, restoring `Search_V3` `stop_rate` to `98.12%`) and native-horizon **`Pure B_0 -> Dr.GRPO+CISPO`** (`99.84%` Search `stop_rate`) eliminated all tripwires on `Confirm_V3` **Look 2** (`stop_rate`: `99.83% -> 99.92%`, `avg@8`: `60.54% -> 63.33%` [**`+2.79 pp`**, `95% CI [+0.92, +4.75] pp`], `maj@8`: `66.67% -> 70.00%`), returning **`INCONCLUSIVE_RETEST`** against the strict `+1.50 pp` Bonferroni (`alpha/2 = 0.0125`) additive margin (`LCB_98.75% = +0.58 pp`) and leaving `Holdout_V3` untouched.

---

## 2. Stage-by-Stage Search Trajectories & Multi-Proposal RLVR Tournaments (`10` Cells)

### 2.1 Track M — Competition & 2026 Research Mathematics (`v3_math_search160`, `N=160`, `k=8`, `12,288` tok)

| Model (`Scale / Active`) | `B_0` `avg@8` (`pass@8` / `stop`) | Stage-I Breadth SFT | Stage-II ZPD+Axion SFT | `DAPO` (`pct50` / `pct100`) | `Dr.GRPO+CISPO` (`pct50` / `pct100`) | `GSPO+LUFFY` (`pct50` / `pct100`) | Order-Invariant Search Winner (`Delta vs B_0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.5-4B` (`4B / 4B`)** | `64.92%` (`86.25%` / `86.17%`) | `72.27%` | **`73.67%`** *(anchor)* | `74.77%` / `75.62%` | `74.92%` / `74.69%` | `75.23%` / **`76.02%`** | **`GSPO+LUFFY pct100`: `76.02%` (`+11.10 pp`, `97.89% stop`)** |
| **`Qwen3.5-9B` (`9B / 9B`)** | `66.25%` (`87.50%` / `84.61%`) | **`75.86%`** *(anchor)* | `74.61%` | `78.52%` / `78.67%` | `76.09%` / `69.69%` | `78.05%` / **`80.39%`** | **`GSPO+LUFFY pct100`: `80.39%` (`+14.14 pp`, `84.38% maj@8`)** |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | `79.30%` (`91.25%` / `99.45%`) | **`79.84%`** *(anchor)* | `79.45%` | `81.72%` / `80.94%` | `81.64%` / `80.63%` | **`82.58%`** / `80.70%` | **`GSPO+LUFFY pct50`: `82.58%` (`+3.28 pp`, `86.88% maj@8`)** |
| **`Qwen3-30B-A3B` (`30B / 3B` MoE)** | `57.42%` (`76.88%` / `99.69%`) | `56.64%` | **`59.38%`** *(R2 anchor)* | `59.06%` / `59.14%` *(R3 compact: `59.38%`)* | `58.98%` / `58.67%` *(Pure-B0: **`59.14%`**, `99.84% stop`)* | `58.98%` / **`62.11%`** *(`95.31% stop`)* | **R2 `GSPO+LUFFY`: `62.11%` (`+4.69 pp`)**; **R3 Stop-Clean `Pure-B0 CISPO`: `59.14%` (`99.84% stop`)** |
| **`Qwen3.5-35B-A3B` (`35B / 3B` MoE)** | **`56.87%`** *(anchor, `82.50%` / `62.89%`)* | `57.58%` (`64.06% stop`) | `56.87%` (`62.66% stop`) | `73.20%` / **`78.98%`** (`90.62% pass@8`, `91.33% stop`) | `77.27%` / `77.66%` (`90.08% stop`) | `69.30%` / `76.48%` (`90.86% stop`) | **Pure-`B_0` `DAPO pct100`: `78.98%` (`+22.11 pp`, `+28.44 pp stop_rate`)** |

### 2.2 Track L — Formal Lean 4.24 + Mathlib Theorem Proving (`v3_lean_search140`, `N=140`, `k=8`, `2,048` tok, `100%` `#print axioms` Clean)

| Model (`Scale / Active`) | `B_0` `avg@8` (`eq_stratum` / `pass@8`) | Stage-I Breadth SFT | Stage-II ZPD SFT | `DAPO` (`pct50` / `pct100`) | `Dr.GRPO+CISPO` (`pct50` / `pct100`) | `GSPO+LUFFY` (`pct50` / `pct100`) | Order-Invariant Search Winner (`Delta vs B_0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.5-4B` (`4B / 4B`)** | `30.45%` (`13.43%` / `45.71%`) | `34.29%` (`16.71%`) | `34.91%` (`17.27%`) | `35.80%` / `35.45%` | **`36.96%`** / `35.18%` | `35.89%` / `36.16%` | **`Dr.GRPO+CISPO pct50`: `36.96%` (`+6.52 pp`, `18.83% eq_stratum`)** |
| **`Qwen3.5-9B` (`9B / 9B`)** | `33.57%` (`14.88%` / `46.43%`) | `35.00%` (`50.71% pass@8`) | `35.71%` (`55.00% pass@8`) | `37.14%` / `37.41%` | `38.12%` / **`39.55%`** | `36.43%` / `36.61%` | **`Dr.GRPO+CISPO pct100`: `39.55%` (`+5.98 pp`, `56.43% pass@8`)** |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | `33.48%` (`14.45%` / `51.43%`) | `41.70%` (`19.35%`) | `39.73%` (`19.78%`) | **`42.77%`** / `41.52%` | `41.16%` / `40.89%` | `41.61%` / `41.34%` | **`DAPO pct50`: `42.77%` (`+9.29 pp`, `21.27% eq_stratum`)** |
| **`Qwen3-30B-A3B` (`30B / 3B` MoE)** | `4.64%` (`1.92%` / `7.86%`) | **`22.05%`** *(anchor, `40.71% pass@8`)* | `18.84%` (`37.86% pass@8`) | `23.39%` / `23.13%` | `23.57%` / **`24.91%`** (`94.91% stop`) | `22.95%` / `24.55%` | **`Dr.GRPO+CISPO pct100`: `24.91%` (`+20.27 pp` [`5.37x`], `42.86% pass@8`)** |
| **`Qwen3.5-35B-A3B` (`35B / 3B` MoE)** | `30.54%` (`14.08%` / `47.14%`) | **`39.82%`** *(anchor, `55.71% pass@8`)* | `38.21%` *(ReST-S2: `39.91%`)* | R1: **`40.54%`** / `39.02%` *(R2 400p: `38.12%` / `39.91%`)* | R1: `39.64%` / `39.02%` *(R2 400p: `40.00%` / **`40.71%`**, `99.64% stop`)* | `39.11%` / `39.11%` | **R2 `S1 + CISPO400 pct100`: `40.71%` (`+10.18 pp`, `99.64% stop`)** *(R1 `DAPO pct50`: `40.54%`)* |

### 2.3 Track L Round-2 Controlled Crossover on `Qwen3.5-35B-A3B` (`16` Search Candidates)

After Round-1 `s3_lean_qwen38_35b_a3b_dapo_asymclip_pct50` (`40.54%` Search) yielded `+4.03 pp` on `Confirm_V3` Look 1 (`28.06% -> 32.08%`, `stop_rate = 78.47%`, `INCONCLUSIVE_RETEST`), we ran a controlled `400`-prompt Round-2 crossover (`Search_V3` only, blind to Confirm/Holdout items) testing whether `35B-A3B` on Lean preferred Pure-`B_0` RLVR (as on Math) or Breadth SFT + length-unbiased RLVR:
- **`Pure B_0 -> DAPO400` (`pct50` / `pct100`):** `35.36%` / `34.64% avg@8` (`49.29% pass@8`, `78.57%–80.63% stop_rate`).
- **`Gentle Stratum-Balanced SFT` (`600` rows, `lr=4e-5`) + `DAPO400` vs. `CISPO400`:** `Gentle SFT` reached `36.07%`; `+ DAPO400` hit `37.59% / 35.00%`, whereas **`+ Dr.GRPO+CISPO400` hit `39.64% / 39.29%` (`99.55% stop_rate`)**.
- **`Full Stage-I Breadth SFT` (`39.82%`) + `DAPO400` vs. `CISPO400` vs. `ReST^EM Trust-CISPO400`:** `Full S1 + DAPO400` reached `39.91%`, `ReST^EM + Trust-CISPO400 pct50` reached `40.54%` (`1.43%` Search forgetting), and **`Full S1 + Dr.GRPO+CISPO400 pct100` won the unified 16-candidate `select-search-winner` tournament at `40.71% avg@8` (`+10.18 pp` over `B_0`, `99.64% stop_rate`)**.
- **Controlled Interaction:** Unlike Track M where `Pure B_0 -> DAPO` gained `+22.11 pp` and `2-SFT` gained `+0.00 pp`, on Lean 4.24 `Breadth SFT` outgained `Pure B_0 -> DAPO400` by **`+4.46 pp` before RL (`39.82%` vs `35.36%`)** and **`+5.35 pp` after RL (`40.71%` vs `35.36%`)**, while removing GRPO's $1/|o_i|$ length normalization (`Dr.GRPO + CISPO`) eliminated verbose proof drift (`stop_rate`: `87.14% -> 99.64%` on Search; `78.47% -> 99.38%` on Confirm Look 2).

---

## 3. Intersection-Union Confirmation (`Confirm_V3`), Weighting Sensitivity, & Single-Touch Holdout (`Holdout_V3`)

Every Phase V3 candidate faced a predeclared Confirmation gate (`max_confirm_looks = 2`, Bonferroni one-sided $\alpha_{\text{eff}} = 0.025 / 2 = 0.0125$, additive superiority margin `+1.50 pp`, and `IUT` against both `B_0` and any promoted V1/V2 incumbent).

### 3.1 Track L — Formal Lean 4.24 Confirmation (`N=180`) & Sealed Single-Touch Holdout (`N=225`: `arXivLean 2026` + `FormalMATH-Lite`)

| Model | `Confirm_V3` (`N=180`) `B_0 -> Cand` `avg@8` (`eq_stratum`) | Gate Verdict (`K <= 2`) | Sealed `Holdout_V3` (`N=225`) `avg@8` | Sealed `Holdout_V3` `eq_stratum` `avg@8` | Sealed `Holdout_V3` `pass@8` | `stop_rate` (`B_0 -> Final`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B`** | `20.97% -> 27.64%` (`17.61% -> 23.84%`; V1 `23.12%`) | **Hold V1 `A*`** *(IUT `< +1.5 pp` vs V1)* | **`15.28% -> 19.39%` (`+4.11 pp`, V1 `A*`)** | `15.86% -> 19.34%` (`+3.48 pp`) | `24.00% -> 27.56%` | `46.56% -> 99.17%` |
| **`Qwen3.5-9B`** | **`28.06% -> 34.86%` (`22.16% -> 29.71%`, `+7.55 pp`)** | **`PROMOTE` (`HOLDOUT_SEAL` Finalized)** | **`16.22% -> 19.28%` (`+3.06 pp`)** | **`16.82% -> 18.75%` (`+1.93 pp`)** | **`24.44% -> 27.11%` (`+2.67 pp`)** | `48.72% -> 97.67%` |
| **`Qwen3.8-27B` (`FSDP`)** | **`27.08% -> 34.58%` (`22.49% -> 29.25%`, `+6.76 pp`)** | **`PROMOTE` (`HOLDOUT_SEAL` Finalized)** | **`16.89% -> 22.06%` (`+5.17 pp`)** | **`17.48% -> 22.12%` (`+4.64 pp`)** | **`26.22% -> 31.56%` (`+5.33 pp`)** | **`43.39% -> 84.83%`** |
| **`Qwen3-30B-A3B` (`MoE`)** | **`4.03% -> 20.14%` (`+16.11 pp`; `3.58% -> 15.46% eq`)** | **`PROMOTE` (`HOLDOUT_SEAL` Finalized)** | **`1.00% -> 11.33%` (`+10.33 pp`, `11.33x`)** | **`1.00% -> 10.70%` (`+9.70 pp`)** | **`3.56% -> 19.11%` (`+15.56 pp`)** | **`67.06% -> 88.61%`** |
| **`Qwen3.5-35B-A3B` (`MoE`)** | L1: `28.06% -> 32.08%`; **L2: `28.06% -> 33.33%` (`+5.28 pp`; `22.99% -> 28.39% eq`)** | **Hold `B_0`** *(`INCONCLUSIVE_RETEST`: `LCB_98.75% = +1.11 pp <= +1.50 pp`)* | `17.89%` *(zero-shot `B_0`; `0` cand touches)* | `18.73%` *(zero-shot `B_0`)* | `26.67%` *(zero-shot `B_0`)* | `57.56%` *(Confirm L2: `99.38%`)* |
| *`Goedel-Prover-V2-8B` (`Ref`)* | — | *Shared `2,048`-tok budget ref* | *`17.61%` (`mean_tok = 1,791`)* | *`18.04%`* | *`21.78%`* | *`20.22%` (`79.78%` truncated at `2,048` tok)* |

- **`Qwen3-30B-A3B` `11.33x` Formal Proving Transfer Without Expert Weight Updates:** Updating only `0.17%` unconditionally active `self_attn` parameters (`0` of `128` routed experts modified) lifted `30B-A3B` on Single-Touch `Holdout_V3` from **`1.00% -> 11.33% avg@8` (`95% CI [+7.17, +13.78] pp`, `0/225` observed forgotten theorems)** and `3.56% -> 19.11% pass@8` (`formalmath_algebra`: `1.81% -> 23.52%`; `formalmath_nt_combinatorics_geom`: `2.19% -> 18.75%`; `formalmath_aime`: `0.00% -> 7.29%`).
- **`Goedel-Prover-V2-8B` Budget-Matched Context:** Under the shared `max_tokens = 2,048` direct-proving budget, domain-specialized `Goedel-Prover-V2-8B` scored `17.61% avg@8` (`21.78% pass@8`), trailing promoted general-purpose `Qwen3.8-27B` (`22.06%`), `Qwen3.5-4B` (`19.39%`), and `Qwen3.5-9B` (`19.28%`); however, `Goedel-Prover-V2-8B` emits long chain-of-thought preambles and hit `79.78%` truncation (`stop_rate = 20.22%`) at `2,048` tokens, so this comparison reflects efficiency under a strict `2,048`-token cap rather than unconstrained long-context proving ceiling.

### 3.2 Track M — Competition & 2026 Research Math Confirmation (`N=300`) & Sealed Single-Touch Holdout (`N=194`: `SMT 25`, `HMMT Feb 25`, `APEX 25`, `arXivMath 2026`)

| Model | `Confirm_V3` (`N=300`) `B_0 -> Cand` `avg@8` (`pass@8`) | Gate Verdict (`K <= 2`) | Sealed `Holdout_V3` (`N=194`) `avg@8` | Sealed `Holdout_V3` `eq_stratum` `avg@8` | Sealed `Holdout_V3` `maj@8` / `pass@8` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B`** | `68.79% -> 80.67%` (`95.33%`; V2 `A*` `78.42%`) | **Hold V2 `A*`** *(`+2.25 pp` vs V2 `INCONCLUSIVE`)* | **`24.16% -> 29.38%` (`+5.22 pp`, V2 `A*`)** | **`28.22% -> 33.80%` (`+5.58 pp`)** | `36.60% -> 39.69%` / `48.97% -> 52.06%` |
| **`Qwen3.5-9B`** | **`69.75% -> 84.29%` (`96.33%`; V2 `A*` `80.42%`)** | **`PROMOTE` vs `B_0` & V2 (`HOLDOUT_SEAL` Finalized)** | **`23.90% -> 36.08%` (`+12.18 pp` vs `9B B_0`)** | **`27.57% -> 40.80%` (`+13.23 pp`)** | **`38.14% -> 45.36%` / `48.97% -> 56.70%`** |
| **`Qwen3.8-27B` (`FSDP`)** | `83.67% -> 83.79%` (`B_0 pass@8 = 95.00%`) | **Hold `B_0`** *(`CONFIRMED_ROLLBACK_NO_HOLDOUT_TOUCH`)* | `40.40%` *(zero-shot `B_0`; `0` cand touches)* | `46.45%` *(zero-shot `B_0`)* | `45.88%` / `59.79%` *(zero-shot `B_0`)* |
| **`Qwen3-30B-A3B` (`MoE`)** | L1: `60.54% -> 63.92%` *(stop guard)*; **L2: `60.54% -> 63.33%` (`84.33%`, `99.92% stop`)** | **Hold `B_0`** *(L1 `ROLLBACK`; L2 `INCONCLUSIVE_RETEST`, `LCB_98.75% = +0.58 pp`)* | `16.37%` *(zero-shot `B_0`; `0` cand touches)* | `17.54%` *(zero-shot `B_0`)* | `19.59%` / `32.99%` *(zero-shot `B_0`)* |
| **`Qwen3.5-35B-A3B` (`MoE`)** | **`61.46% -> 82.96%` (`+21.50 pp`, `88.00% -> 93.67% pass@8`)** | **`PROMOTE` (`HOLDOUT_SEAL` Finalized)** | **`20.43% -> 40.34%` (`+19.91 pp`, `[+16.43, +23.65]`)** | **`24.22% -> 46.21%` (`+21.99 pp`)** | **`40.21% -> 48.45%` / `42.27% -> 59.28%`** |

- **`Qwen3.5-35B-A3B` Single-Touch Holdout Breakthrough (`+19.91 pp` Pooled / `+21.99 pp` Equal-Stratum):** Pure Zero-SFT On-Policy `DAPO + Overlong` lifted `stop_rate` on `Holdout_V3` from **`34.47% -> 76.80%` (`+42.33 pp`)**, cut `tokens_per_correct` more than in half (`49,367 -> 21,692`), and expanded accuracy across every stratum with only `1/194` (`0.52%`) observed forgotten problem:
  - **`arXivMath 2026` (`N=102` research problems):** `7.60% -> 22.67% avg@8` (**`2.98x`**).
  - **`HMMT Feb 2025`:** `24.17% -> 58.75% avg@8` (**`+34.58 pp`**).
  - **`SMT 2025`:** `46.46% -> 70.75% avg@8` (**`+24.29 pp`**).
  - Descriptively, post-trained `Qwen3.5-35B-A3B` (`40.34% avg@8`, `46.21% equal_stratum`, `48.45% maj@8`, `59.28% pass@8`) matches zero-shot `Qwen3.8-27B` Dense (`40.40% avg@8`, `46.45% equal_stratum`, `45.88% maj@8`, `59.79% pass@8`) while activating `9x` fewer parameters per token (`3B` vs `27B`, a parameter-count ratio rather than a wall-clock latency equivalence claim) and exceeds V3 `Qwen3.5-9B` (`36.08%`) by `+4.26 pp`.

### 3.3 Weighting Estimand Disclosure & `equal_stratum` Sensitivity Audit (`8/8` MoE Decisions Invariant)

In `configs_v3*/gate_rules_*.json`, stratum weighting was written under key `"weighting": "equal_stratum"` whereas `governed_loop_guard_v37.py` reads `"aggregation_weighting"` (defaulting to `"equal_item"`). Rather than rewriting immutable hash-chained ledgers, we preserve the historical `equal_item` ledger entries intact and report exact `10,000`-resample stratified paired bootstrap sensitivity under `equal_stratum` across all MoE Confirm and Holdout evaluations (`Confirm_V3` Math has `100` items in each of its `3` strata, so `equal_item` and `equal_stratum` point estimates are identical):

| Cell & Stage | Recorded `equal_item` Delta (`LCB_98.75%`) | Exact `equal_stratum` `B_0 -> Cand` (`Delta`, `95% CI`) | Exact `equal_stratum` `LCB_98.75%` vs `+1.50 pp` | Decision Under Both Estimands |
| :--- | :---: | :---: | :---: | :---: |
| **`35B-A3B` Math `Confirm_V3` (L1)** | `+21.50 pp` (`+18.00 pp`) | `61.46% -> 82.96%` (`+21.50 pp`, `[+18.46, +24.67]`) | **`+18.04 pp` (`> +1.50 pp`)** | **`PROMOTE` (Invariant)** |
| **`35B-A3B` Math `Holdout_V3`** | `+19.91 pp` (`95% CI [+16.43, +23.65]`) | **`24.22% -> 46.21%` (`+21.99 pp`, `[+17.84, +26.15]`)** | **`+17.31 pp` (`> +1.50 pp`)** | **`REPLICATED` (Invariant)** |
| **`30B-A3B` Lean `Confirm_V3` (L1)** | `+16.11 pp` (`+11.32 pp`) | `3.58% -> 15.46%` (`+11.89 pp`, `[+8.56, +15.59]`) | **`+8.17 pp` (`> +1.50 pp`)** | **`PROMOTE` (Invariant)** |
| **`30B-A3B` Lean `Holdout_V3`** | `+10.33 pp` (`95% CI [+7.17, +13.78]`) | **`1.00% -> 10.70%` (`+9.70 pp`, `[+6.77, +12.83]`)** | **`+6.44 pp` (`> +1.50 pp`)** | **`REPLICATED` (Invariant)** |
| **`35B-A3B` Lean `Confirm_V3` (L1 / L2)** | `+4.03 pp` (`-0.28 pp`) / `+5.28 pp` (`+1.11 pp`) | `22.99% -> 28.39%` (`+5.40 pp` L2, `[+1.38, +9.81]`) | `+0.72 pp` (L1) / `+0.80 pp` (L2 `<= +1.50 pp`) | **`INCONCLUSIVE_RETEST` (Invariant)** |
| **`30B-A3B` Math `Confirm_V3` (L1 / L2)** | `+3.38 pp` *(stop trip)* / `+2.79 pp` (`+0.58 pp`) | `60.54% -> 63.33%` (`+2.79 pp` L2, `[+0.92, +4.75]`) | `+0.67 pp` (L1, stop trip) / `+0.71 pp` (L2 `<= +1.50 pp`) | **`ROLLBACK` / `INCONCLUSIVE_RETEST` (Invariant)** |

---

## 4. Six-Benchmark Longitudinal Generalization (`MiniF2F`, `ProofNet`, and `2025` 4-Contest Math)

External longitudinal evaluations across `MiniF2F-test` (`N=244`), `ProofNet-test` (`N=163` unique deduplicated IDs on V3; `N=167` raw), and the `2025` 4-Contest Math suite (`N=200`: `BeyondAIME-100`, `CMIMC-40`, `BRUMO-30`, `HMMT Nov-30`):

| Model | `MiniF2F-test` (`N=244`) `avg@8` (`pass@8`) | `ProofNet-test` (`N=163`) `avg@8` (`pass@8` / `stop`) | `2025` 4-Contest Math (`N=200` Pooled) `avg@8` | `BeyondAIME` (`100`) | `CMIMC 25` (`40`) | `BRUMO 25` (`30`) | `HMMT 25` (`30`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B` (`B_0 -> V3`)** | `28.33% -> 29.30%` | `1.84% -> 2.38%` | **`36.25% -> 43.69%` (`+7.44 pp`)** | `29.88% -> 36.88%` | `34.06% -> 37.81%` | `56.67% -> 67.08%` | `40.00% -> 50.83%` |
| **`Qwen3.5-9B` (`B_0 -> V3`)** | **`30.28% -> 33.61%` (`+3.33 pp`)** | **`2.15% -> 5.29%` (`2.46x`, `9.82% pass@8`)** | **`38.00% -> 49.50%` (`+11.50 pp`)** | **`31.75% -> 44.50%`** | **`33.12% -> 43.13%`** | **`60.00% -> 66.25%`** | **`43.33% -> 57.92%`** |
| **`Qwen3.8-27B` (`B_0 -> V3`)** | **`33.66% -> 35.96%` (`51.64% pass@8`)** | **`2.68% -> 3.45%` (`9.82% pass@8`, `88.8% stop`)** | `57.25%` (`B_0` held; `r2_s3 = 56.69%`) | `50.00%` (`48.13%`) | `49.69%` (`53.75%`) | `78.75%` (`75.00%`) | `66.25%` (`66.25%`) |
| **`Qwen3-30B-A3B` (`B_0 -> V3`)** | **`1.02% -> 15.47%` (`15.17x`, `4.10% -> 29.51% pass@8`)** | **`0.00% -> 0.92%` (`0.00% -> 2.45% pass@8`, `87.6% stop`)** | `17.63%` (`B_0` held; R2 long-CoT `17.50%` rolled back) | `15.25%` (`14.63%`) | `18.44%` (`15.94%`) | `36.25%` (`36.25%`) | `10.00%` (`10.42%`) |
| **`Qwen3.5-35B-A3B` (`B_0 -> V3`)** | **`35.25% -> 37.65%` (`48.77% -> 52.46% pass@8`, `99.5% stop`)** | **`3.60% -> 4.68%` (`10.43% pass@8`, `99.9% stop`)** | **`27.63% -> 57.00%` (`+29.37 pp`, `0/200` obs. forgotten)** | **`22.50% -> 49.25%` (`+26.75 pp`)** | **`25.31% -> 55.94%` (`+30.63 pp`)** | **`49.17% -> 78.33%` (`+29.17 pp`)** | **`26.25% -> 62.92%` (`+36.67 pp`)** |

---

## 5. Cross-Architecture Synthesis: Knowledge-Breadth vs. Termination-Depth Diagnostic & Audit Notes

### 5.1 Refining the Sparse MoE Hypothesis via Two Contrasting `3B`-Active Architectures

In our initial single-MoE observation on `Qwen3.5-35B-A3B` (`stop_rate = 62.89%` on `12K` Math), a plausible hypothesis was that `3B` active parameters inherently drift under a `12,288`-token generation cap. Adding **`Qwen3-30B-A3B`** (`3B` active parameters, `stop_rate = 99.69%` on Search and `99.36%` on Holdout at `median ~ 1,320–1,640` tokens) **refutes the claim that a `12K` budget alone forces `3B`-active MoE non-termination**, showing instead that post-training behavior aligns with two observable `B_0` diagnostics (`pass@8` breadth vs. `stop_rate` termination discipline):

| Diagnostic `B_0` Regime | Representative Cells (`B_0` Profile) | Within-Model Crossover Evidence | Winning Post-Training Mechanism |
| :--- | :--- | :--- | :--- |
| **High Latent Breadth + Long-CoT Non-Termination** | `Qwen3.5-35B-A3B` Math (`Search pass@8 = 82.50%`, `stop_rate = 62.89%`; `Holdout stop_rate = 34.47%`) | Off-policy `2-SFT` stagnated (`56.87% -> 57.58% -> 56.87%`); purely on-policy `DAPO` (`78.98%`) beat off-policy-mixed `GSPO+LUFFY` (`76.48%`). | **Zero-SFT On-Policy `DAPO + Overlong` (`B_0 -> RLVR`)** (`+19.91 pp` Holdout, `+29.37 pp` `2025` 4-Contest, `+42.33 pp` Holdout `stop_rate`). |
| **Breadth- or Syntax-Bottlenecked** | All 5 models on Lean 4.24 (`30B-A3B` `B_0 = 4.64%`; `35B-A3B` `B_0 = 30.54%`) + `4B`/`9B`/`30B-A3B` Math | On `35B-A3B` Lean, `Pure-B0 DAPO400` (`35.36%`) trailed `Breadth SFT + CISPO400` (`40.71%`) by `-5.35 pp`; on `30B-A3B` Math, `Pure-B0 CISPO` (`59.14%`) trailed `2-SFT + GSPO+LUFFY` (`62.11%`) on raw Search accuracy while preserving `99.84% stop_rate`. | **Horizon-Matched Verified Breadth SFT + `Dr.GRPO+CISPO` (short-horizon Lean) or `GSPO+LUFFY` (horizon-matched `10K` Math)**. |

*Causal & Architectural Caveats:* Because `Qwen3-30B-A3B` and `Qwen3.5-35B-A3B` differ simultaneously in pretraining generation, Gated-DeltaNet + shared-expert blocks vs. pure attention, native thinking vs. concise priors, and adapted parameter fraction (`0.17%` `self_attn` vs. `0.38%` `self_attn + linear_attn + shared`), cross-model differences should be interpreted as **architectural/prior contrasts**, whereas recipe claims rely on the **within-model crossovers** above.

### 5.2 Governance, Provenance, and Statistical Scope Disclosures

1. **Artifact & Bundle Provenance:** All 4 Sparse MoE cells (`Qwen3-30B-A3B` & `Qwen3.5-35B-A3B` x `2` tracks, plus `Goedel-Prover-V2-8B` reference) have complete item-level Confirm/Holdout/Search/Longitudinal JSONs, `HOLDOUT_SEAL` files, and **`12 / 12` SHA-256 hash-chained ledgers** committed directly in `results_v3/` and verified by `python -I -B code/v3/build_v3_report_bundle.py .` (`2 / 2` promoted MoE cells replicated on Holdout; `2 / 2` `INCONCLUSIVE_RETEST` cells made `0` candidate Holdout touches). Dense V3 (`4B`, `9B`, `27B`) ran on an earlier identically configured `24x A100-80GB` cluster against the same pinned split IDs; its stage metrics and `3 / 3` Holdout replications were recorded at execution time in commit `62a94b9` before that cluster's ephemeral scratch was reclaimed.
2. **Fixed-Checkpoint Statistical Scope (`claim_scope = fixed_checkpoint`):** Reported bootstrap CIs and Rademacher/Monte-Carlo sign-flip $p$-values measure paired problem-cluster uncertainty conditional on the trained checkpoints (`1` distributed training run per recipe), not variance across independent retraining seeds. Candidate pin files record adapter metadata and manifest digests rather than full multi-gigabyte `.safetensors` shard hashes.
3. **Quarantine & Future Reuse:** For `INCONCLUSIVE_RETEST` cells (`35B-A3B` Lean and `30B-A3B` Math), `0` candidate checkpoints were evaluated on `Holdout_V3` (`B_0` baselines were scored once as fixed reference anchors), and their `max_confirm_looks = 2` Confirmation budget is exhausted — any future iteration must mint a fresh Confirmation split.


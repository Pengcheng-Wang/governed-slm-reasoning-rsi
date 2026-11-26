# Cross-Family Post-Training Dynamics for Verifiable Mathematical and Formal Reasoning Under Sealed Governance

## Abstract
Empirical laws for reinforcement learning with verifiable rewards (RLVR) and reasoning distillation in open-weight language models have been derived predominantly within the Qwen family. We extend a cryptographically sealed (`Search -> Confirm -> Single-Touch Holdout`) evaluation harness with out-of-process SymPy and Lean 4.24 (`#print axioms`) kernel verifiers to five non-Qwen open-weight models—`Gemma-4-31B-it` (`30.7B` Dense), `Gemma-4-26B-A4B-it` (`25.2B` total / `3.8B` active MoE), `Inkling-2-Mini` (`26.4B` total / `3.4B` physical active, `48`-visit 3-pass looped MoE), `Muse-Glimmer-30B` (`29.6B` Dense), and `Gemma-4-E4B-it` (`8.0B / 4.5B` Dense)—evaluated alongside five Phase-V3 `Qwen3/3.5/3.8` models (`4B` to `35B-A3B`). Across two governed experimental waves (`V4.0` and `V4.1` under `governed_loop_guard` `v3.9.0`), we establish four findings: (1) **domain-orthogonal base capabilities** (`Gemma-4-31B` leads zero-shot competition mathematics [`42.91%` `Holdout_V3`, `61.19%` `2025` 4-Contest], whereas `Inkling-2-Mini` and thinking-enabled `Qwen3.5/3.8` outperform `Gemma-4` by `3.4x–7.4x` on Lean 4.24 `Search_V3` and `2.4x–3.0x` on `Holdout_V3` across `{thinking off, on} x {2048, 3072}` protocol sweeps, at `4.57x` higher recurrent training wall-time per token for `Inkling`); (2) **partial causal recovery via reasoning-channel alignment** (training thoughtless Lean proofs under an active thinking prompt suppresses stop/channel calibration, whereas a `0.000%`-target-token-delta effort-matched wrapper on `Inkling-2-Mini` recovers **`+5.75 pp avg@8` [`95%` CI `[+3.33 pp, +8.50 pp]`] and `+16.67 pp stop_rate`** over stripped SFT on unexposed `Confirm_V4_1`, though `stop_rate` remains `-13.67 pp` below `B_0` and `Muse-30B` Lean still collapses `has_thinking_rate` to `0%`); (3) **precision-safe multi-step RLVR** (combining `40`-step temperature-matched `T=0.7` `GSPO` updates with unbiased 16-bit stochastic `FP32 -> BF16` LoRA consolidation [`100.00%` inner-product retention] delivers **`+24.08 pp`** pure on-policy `avg@8` gain on `Holdout_V4_1`, vs. a null `+0.84 pp` under `V4.0`'s `11`-step `T_train=1.0` round-to-nearest `Dr.GRPO+CISPO` setup); and (4) **a controlled single-seed Breadth–Depth trade-off in off-policy teacher shaping** (on a common `GSPO` backbone with `1` channel-aligned teacher + `7` student negatives on zero-pass prompts, rare-token-preserving log-shaping $w(p)=\gamma/(p+\gamma)$ reaches the highest `Search_V3` `pass@8` [`87.50%`, `3.34` distinct answers/problem], whereas differentiated `LUFFY` $f(p)=p/(p+\gamma)$ maximizes `avg@8` depth and promotes to single-touch `Holdout_V4_1` at **`73.33% avg@8` (`+30.58 pp` over `B_0`, guard `97.5%` LCB `+24.58 pp`, Monte Carlo `8,000`-sign-flip `p <= 1.25e-4`; `+6.50 pp` over `Pure GSPO`, `95%` CI `[+4.00 pp, +9.25 pp]`)** and **`90.00% pass@8`** with `98.25% stop_rate`).

---

## 1. Experimental Setup & Sealed Governance (`V4.0` & `V4.1`)

- **Evaluation & Exposure Protocol:**
  - **Wave-1 Cross-Family Benchmark (`Holdout_V3`, `IMMUTABLE_MANIFEST_V4.json` root `8f69efbd508ae4291fde22faa6edcc6a6cae759228d45dbcb147ea5f85e47c50`):** `Track M` (`N=194`: `SMT 2025`, `HMMT Feb 2025`, `APEX 2025`, `arXivMath 2026`) and `Track L` (`N=225`: `arXivLean 2026` + `FormalMATH-Lite`), plus longitudinal `2025` 4-Contest Math (`N=200`: `BeyondAIME`, `CMIMC 2025`, `BrUMO 2025`, `HMMT Nov 2025`) and `MiniF2F-test` (`N=244`) / `ProofNet-test` (`N=163`), evaluated at `k=8` rollouts (`T=0.7`) with out-of-process SymPy and Lean 4.24 (`#print axioms`) verifiers.
  - **Fresh Sealed Wave-2 Splits (`Confirm_V4_1` & `Holdout_V4_1`):** Because `Holdout_V3` was evaluated for `B_0` baselines in Wave 1, Wave 2 (`V4.1`) quarantined `Holdout_V3` as historical transfer and constructed genuinely unexposed `50/50/50` balanced-stratum splits (`N=150` `Confirm_V4_1`, `N=150` `Holdout_V4_1` per track; `N_eff = 150.0`) drawn from unused `DeepMath-103K` (`difficulty >= 6.0`) and `Goedel-Pset-v1` problems. Zero overlap (`0` exact ID, normalized-alphanumeric, numeric-skeleton, and stride-1 13-gram collisions) was verified against all `42` Phase V1–V3 split/corpus JSONLs (`4,777` unique prior items) and `32` V4 split/corpus files via `governed_loop_guard` `v3.9.0` `audit-splits`.
  - **Pre-Confirm Schema Amendment Disclosure (`IMMUTABLE_MANIFEST_V4_1.json` -> `IMMUTABLE_MANIFEST_V4_1_R2.json`):** `IMMUTABLE_MANIFEST_V4_1.json` (`b9c58dcb...`) sealed all splits, prompts, and candidate budgets prior to Wave-2 training. After `Search_V3` scoring completed (`2026-11-25T08:28Z`) and before **any** `Confirm_V4_1` or `Holdout_V4_1` item was evaluated (`0` Confirm/Holdout looks consumed), `governed_loop_guard` `v3.9.0` rejected `"bootstrap_mode": "cluster"` (renamed to `"cluster_1d"` in `v3.9.0`). `IMMUTABLE_MANIFEST_V4_1_R2.json` (`f30b2020...`, `09:06Z`, prior to the first `Confirm_V4_1` run at `09:19Z`) updated `configs_v4_1_r1/` to `"cluster_1d"`, harmonized `stop_rate` non-inferiority to `0.05` across Math and Lean (promoted `Muse` Math `Confirm_V4_1` `stop_rate` LCB is `-1.75 pp` [`95%`] / `-2.10 pp` [`97.5%`], passing both the original `0.025` and harmonized `0.05` thresholds), and evaluated `Pure GSPO` alongside `Faithful LUFFY` as a non-gating diagnostic contrast. On `lean_inkling2_mini`, where all Search SFT arms remained below `B_0` (`eligible_count = 0`), `Confirm_V4_1` was executed solely as a pre-declared out-of-sample causal contrast between `effort_matched` and `stripped_mismatch` (`ROLLBACK` vs. `B_0`; `Holdout_V4_1` untouched).
  - **Statistical Estimators & Fixed-Checkpoint Scope:** At `26B–31B` scale (`8x A100-80GB FSDP FULL_SHARD`), trained arms are **single-seed (`seed = 20261124`) fixed-checkpoint** interventions. Gate promotion decisions (`Confirm_V4_1` / `Holdout_V4_1` at one-sided $\alpha = 0.025$) use `governed_loop_guard` `v3.9.0` (`4,000` problem-cluster bootstrap draws for the `97.5%` one-sided LCB + `8,000`-draw Monte Carlo cluster sign-flip permutation test, resolution floor $p \ge 1/8001 \approx 1.25 \times 10^{-4}$); paired mechanism contrasts additionally report `10,000`-draw problem-cluster bootstrap `95%` two-sided CIs and `95%` one-sided LCBs. All accuracy deltas are reported in **percentage points (`pp`)**.

---

## 2. Cross-Family Base Capabilities vs. Governed Post-Training (`Holdout_V3`, `k=8`)

Table 1 separates **Pure Untuned (`B_0`)** performance from **Governed Post-Trained (`A*`)** checkpoints on identical `Holdout_V3` and `2025` longitudinal benchmarks (`pooled avg@8`):

| Model (`Total / Active`) | Math `Holdout_V3` (`N=194`) **Raw `B_0`** | Math **Post-Trained `A*`** | Lean `Holdout_V3` (`N=225`) **Raw `B_0`** | Lean **Post-Trained `A*`** | `2025` 4-Contest Math (`N=200`) | `MiniF2F-test` (`N=244`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Gemma-4-31B-it` (`30.7B / 30.7B` Dense) | 42.91% | *Held `B_0`* | 7.00% | *Held `B_0`* | `61.19%` *(raw `B_0`)* | `13.47%` (`20.49% pass@8`) |
| `Gemma-4-26B-A4B-it` (`25.2B / 3.8B` MoE) | 38.66% | *Held `B_0`* | 5.94% | *Held `B_0`* | `53.94%` *(raw `B_0`)* | `7.38%` (`13.52% pass@8`) |
| `Inkling-2-Mini` (`26.4B / 3.4B` phys., `48`-visit Looped MoE) | 30.86% | *Held `B_0`* | 17.06% | *Held `B_0`* | `44.38%` *(raw `B_0`)* | `33.81%` (`48.77% pass@8`) |
| `Muse-Glimmer-30B` (`29.6B / 29.6B` Dense) | 37.63% | `38.47%` (`+0.84 pp`, V4.0 *n.s.*)‡ | 10.00% | *Held `B_0`* | `59.88% -> 59.25%` | `18.55%` (`32.38% pass@8`) |
| `Gemma-4-E4B-it` (`8.0B / 4.5B` Dense)† | 5.54% | *Held `B_0`* | 0.67% | *Held `B_0`* | `3.38%` *(raw `B_0`)* | `0.31%` (`2.05% pass@8`) |
| `Qwen3.8-27B` (`27B / 27B` Dense)* | 40.40% | *Held `B_0`* | 16.89% | **22.06%** (`+5.17 pp`) | `57.25%` *(raw `B_0`)* | `32.27% -> 34.43%` (`51.64% pass@8`) |
| `Qwen3.5-35B-A3B` (`35B / 3B` MoE)* | 20.43% | **40.34%** (`+19.91 pp`) | **17.89%** | *Held `B_0`* | `27.62% ->` **57.00%** (`+29.38 pp`) | `35.25% -> 37.65%` *(Search winner)* |
| `Qwen3.5-9B` (`9B / 9B` Dense)* | 23.90% | **36.08%** (`+12.18 pp`) | 16.22% | **19.28%** (`+3.06 pp`) | `38.00% -> 49.50%` (`+11.50 pp`) | `30.28% -> 33.61%` |
| `Qwen3.5-4B` (`4B / 4B` Dense)* | 24.16% | **29.38%** (`+5.22 pp`) | 15.28% | **19.39%** (`+4.11 pp`) | `36.25% -> 43.69%` (`+7.44 pp`) | `28.33% -> 29.30%` |
| `Qwen3-30B-A3B` (`30B / 3B` MoE, non-think)* | 16.37% | *Held `B_0`* | 1.00% | **11.33%** (`+10.33 pp`) | `18.25%` *(raw `B_0`)* | `1.02% -> 15.47%` (`15.10x`) |

*\*Phase-V3 Qwen rows are imported from `reports/03_phase_v3_data_scaling_rl_and_27b_results.md` (`Holdout_V3` and `2025` longitudinal splits identical by SHA-256).*
*†`Gemma-4-E4B-it` (`think=off`) concludes `67.50%` of natural-stop Math solutions with prose/unboxed `$...$` answers rather than `\boxed{}` (`Search_V3` `13.36%` vs. `Holdout_V3` `5.54%`); we exclude `E4B` from capability-floor claims (`eval_v4_1` rescoring imported the pre-patch extractor reference and left `13.36%` unchanged).*
*‡In Wave-1 (`V4.0`), `Muse-Glimmer-30B` Math `Arm A` (`11`-step `T_train=1.0` round-to-nearest `BF16` `Dr.GRPO+CISPO`) passed `Confirm_V4` narrowly (`+2.42 pp`) and produced a statistically null `+0.84 pp` pooled (`+0.22 pp` equal-stratum, `p = 1.0`) delta on `Holdout_V3`; Wave-2 (`V4.1`) diagnoses and resolves this stall on unexposed `Holdout_V4_1` (`+24.08 pp` `Pure GSPO`, `+30.58 pp` `Faithful LUFFY`; §5–§6).*
*Note on Wave-1 `Inkling-2-Mini` Math:* In `V4.0`, `Inkling-2-Mini` `Arm B` (`SFT + DAPO`) gained `+11.88 pp avg@8` (`51.08% -> 62.96%`) and `+19.33 pp pass@8` (`68.67% -> 88.00%`) on `Confirm_V4` (`N=300`), but was held (`ROLLBACK`, `0` candidate `Holdout_V3` touches) because V4.0's stripped-thought SFT lowered `stop_rate` by `-4.25 pp` (`79.83% -> 75.58%` vs. `-2.50 pp` cap).

---

## 3. Protocol Equalization: Separating Thinking Mode from Token Cap (`Gate G2`)

In Wave 1 (`V4.0`), `Gemma-4` Lean used `enable_thinking=False` at `max_tokens=2048` (`58.84%` `stop_rate` on `26B-A4B`), whereas `Inkling-2-Mini` and `Muse-Glimmer-30B` used active thinking at `max_tokens=3072`. Table 2 evaluates the orthogonal `{thinking off, on} x {2048, 3072}` grid on `Search_V3` (`T=0.7`, `k=8`):

| Candidate / Protocol | Track | Thinking Mode | `max_tokens` | `avg@8` | `pass@8` | `stop_rate` | `has_thinking_rate` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lean_b0_gemma4_26b_a4b_think_off_cap_2048` | `lean` | `enable_thinking=False` | `2048` | **4.91%** | 8.57% | 58.84% | 0.00% |
| `lean_b0_gemma4_26b_a4b_think_off_cap_3072` | `lean` | `enable_thinking=False` | `3072` | **4.73%** | 8.57% | 65.09% | 0.00% |
| `lean_b0_gemma4_26b_a4b_think_on_cap_2048` | `lean` | `enable_thinking=True` | `2048` | **3.30%** | 5.71% | 3.75% | 100.00% |
| `lean_b0_gemma4_26b_a4b_think_on_cap_3072` | `lean` | `enable_thinking=True` | `3072` | **4.82%** | 7.86% | 5.98% | 100.00% |
| `lean_b0_gemma4_31b_think_off_cap_2048` | `lean` | `enable_thinking=False` | `2048` | **9.55%** | 13.57% | 71.96% | 0.00% |
| `lean_b0_gemma4_31b_think_off_cap_3072` | `lean` | `enable_thinking=False` | `3072` | **10.54%** | 15.00% | 78.30% | 0.00% |
| `lean_b0_gemma4_31b_think_on_cap_2048` | `lean` | `enable_thinking=True` | `2048` | **8.04%** | 14.29% | 10.00% | 100.00% |
| `lean_b0_gemma4_31b_think_on_cap_3072` | `lean` | `enable_thinking=True` | `3072` | **10.54%** | 19.29% | 15.54% | 100.00% |
| `lean_b0_inkling2_mini_effort_low_3072` | `lean` | `reasoning_effort=low` | `3072` | **35.80%** | 49.29% | 63.39% | 100.00% |
| `lean_b0_inkling2_mini_effort_none_3072` | `lean` | `reasoning_effort=none` | `3072` | **33.39%** | 43.57% | 45.71% | 100.00% |
| `math_b0_gemma4_e4b_think_off_12288†` | `math` | `enable_thinking=False` | `12288` | **13.36%** | 45.62% | 100.00% | 0.00% |

**Protocol & Compute Caveats:** Raising `Gemma-4` Lean from `2,048` to `3,072` tokens under `think=off` confirms the Lean gap (`4.73%` on `26B-A4B`, `10.54%` on `31B` at `65.09%–78.30%` `stop_rate` vs. `33.39%` on `Inkling` `effort=none`). Under `think=on` at `3,072` tokens, however, `Gemma-4` generates verbose English proof sketches before emitting Lean code (`stop_rate = 5.98%–15.54%`), so `think=on` Lean parity at `>=12,288` tokens remains open. Furthermore, although `Inkling-2-Mini` activates `3.4B` physical parameters per pass, its `3`-pass (`48`-layer-visit) looped architecture requires **`4.57x` more training wall-clock time per token** (`1.54 ms/tok` vs. `0.34 ms/tok` on `8x A100-80GB`) and `3.33x` higher peak FSDP HBM (`75.6 GiB` vs. `22.7 GiB`) than single-pass `3.8B`-active `Gemma-4-26B-A4B`.

---

## 4. Causal Test of Reasoning-Channel Alignment in Thoughtless Lean SFT (`Gate G3`)

In `V4.0`, supervising `Goedel-Prover-V2` thoughtless tactic proofs under an active thinking prompt (`Thinking effort level: 0.2` on `Inkling-2-Mini`; `Reasoning strength: low` on `Muse-Glimmer-30B`) trained models to bypass their native thinking headers. In `V4.1`, `effort_matched_wrapper` binds the **exact same `736,175` supervised proof tokens (`74` steps)** to `Thinking effort level: 0` / `Reasoning strength: none` (`0.000%` target-token delta), testing whether prompt-header conditioning alone protects active thinking at evaluation (`effort=0.2` / `strength=low`):

| Model & Arm (`Search_V3`, `max_tokens=3072`, `T=0.7`) | Supervised Budget | Merge Retention (`IP`, `Energy`) | Lean `avg@8` | Lean `pass@8` | `stop_rate` | `has_thinking_rate` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `lean_b0_inkling2_mini_effort_low_3072` | — | — | **35.80%** | 49.29% | 63.39% | 100.00% |
| `inkling_lean_sft_stripped` | `736,175` toks (`74` st) | `100.00%` (`E=1.59x`) | **28.57%** | 41.43% | 37.41% | 100.00% |
| `inkling_lean_sft_effort_matched` | `736,175` toks (`74` st) | `100.00%` (`E=1.57x`) | **32.77%** | 45.71% | 50.54% | 96.61% |
| `inkling_lean_sft_matched_plus_gspo` | `24` RL steps | `99.99%` (`E=1.56x`) | **33.48%** | 49.29% | 52.77% | 94.73% |
| `lean_b0_muse_glimmer_30b` | — | — | **17.59%** | 32.86% | 93.48% | 100.00% |
| `muse_lean_sft_effort_matched` | `710,865` toks (`78` st) | `100.00%` (`E=1.66x`) | **22.59%** | 40.71% | 78.66% | 0.00% |
| `muse_lean_sft_matched_plus_gspo` | `28` RL steps | `100.00%` (`E=1.63x`) | **25.00%** | 42.14% | 80.54% | 0.00% |

**Partial Mitigation vs. Cross-Architecture Channel Leakage:**
1. **`Inkling-2-Mini` (`g3a`):** Changing only two masked prompt tokens (`0.2 -> 0`) recovers **`+4.20 pp avg@8` (`95%` CI `[+1.88 pp, +6.70 pp]`) and `+13.13 pp stop_rate` on `Search_V3`**, and replicates out-of-sample on **`Confirm_V4_1` (`N=150`): `29.17%` vs. `23.42%` (`+5.75 pp avg@8`, `95%` CI `[+3.33 pp, +8.50 pp]`; `+16.67 pp stop_rate`, `52.08%` vs. `35.42%`)**. However, mitigation is **partial**: while `effort_matched`'s `Confirm_V4_1` `avg@8` (`29.17%`) is within `-0.75 pp` of `B_0` (`29.92%`, `95%` CI `[-2.83 pp, +1.42 pp]`), `stop_rate` remains `-13.67 pp` below `B_0` (`52.08%` vs. `65.75%`), triggering `Confirm_V4_1` `ROLLBACK`.
2. **`Muse-Glimmer-30B` (`g3b` — Informative Negative Result on Channel Preservation):** On `Muse-30B`, `effort_matched` SFT + `GSPO` lifts Lean `avg@8` from `17.59% -> 25.00%` (`Search_V3`) and `16.17% -> 20.50%` (`+4.33 pp`, `Confirm_V4_1`), **but `has_thinking_rate` still collapses from `100.00% -> 0.00%`** and `stop_rate` drops `-4.83 pp` (`guard 97.5% LCB = -10.08 pp <= -5.00 pp`, `ROLLBACK`). Unlike `Inkling` (which uses distinct `<|content_thinking|>` / `<|content_text|>` tokens), `Muse` shares `<|start|>assistant` across `to=self` and `to=user`; fine-tuning LoRA weights exclusively on `to=user` Lean proofs under `strength=none` shifts the `to=self` vs. `to=user` logit margin globally even at `strength=low`. Preserving `to=self` on shared-prefix templates requires explicit `to=self` reasoning traces rather than prompt-metadata conditioning alone.

---

## 5. Off-Policy `LUFFY` vs. Log-Shaping vs. `Channel-Aligned SFT -> On-Policy GSPO` (`Gate G4`)

### 5.1 Rare-Token Gradient Weighting & Stochastic `FP32 -> BF16` Consolidation
1. **Rare-Token Gradient Weighting:** Under differentiated `LUFFY` shaping $f(p) = \frac{p}{p+\gamma}$ ($\gamma = 0.1$), the gradient multiplier with respect to $\log p_\theta$ is $\frac{d f(p)}{d \log p} = \frac{\gamma p}{(p+\gamma)^2}$. At a low-probability teacher pivot token $p_\theta = 10^{-5}$, $\frac{d f}{d \log p} = 9.998 \times 10^{-5}$ (**`10,002x` smaller** than SFT's unit weight `1.0`), peaking near $p_\theta = \gamma = 0.1$ (`0.25`) and decaying to `0.0826` at $p_\theta = 1$. By contrast, detached log-shaping $w_{\text{log}}(p) = \frac{\gamma}{p+\gamma}$ assigns `0.9999` at $p_\theta = 10^{-5}$ and `0.0909` at $p_\theta = 1$. Note that $w_{\text{log}}(p) / \frac{d f}{d \log p} = \frac{p+\gamma}{p} \ge 1.10$ pointwise, so `log_shaped` applies both higher rare-token weight and a larger total positive teacher step norm than `faithful_luffy` at identical learning rate.
2. **Unbiased Stochastic `FP32 -> BF16` LoRA Consolidation:** In a synthetic Gaussian perturbation test ($W_0 \sim \mathcal{N}(0, 0.02^2)$ in `BF16`), deterministic round-to-nearest `FP32 -> BF16` merging erases `96.5%` of entries (`11.4%` energy retention) at micro-step RL scale ($\text{sd} = 3 \times 10^{-6}$) and `88.3%` at $\text{sd} = 10^{-5}$, vs. `36.3%` at multi-step $\text{sd} = 10^{-4}$. In `V4.1`, accumulating `FP32` master LoRA updates across `40` `T=0.7` steps and applying unbiased 16-bit stochastic `FP32 -> BF16` rounding achieves `inner_product_retention_ratio = 0.9999–1.0000` across every arm (`delta_energy_retention_ratio = ||\widehat{\Delta W}||_F^2 / ||\Delta W||_F^2 = 2.90–3.71` on RL arms and `1.56–1.66` on SFT arms, where the excess `R - 1 > 0` reflects zero-mean rounding dither variance $\sum_i p_i(1-p_i)\text{ulp}_i^2$). Because `V4.1` simultaneously upgrades step budget (`11 -> 40`), rollout/train temperature (`1.0 -> 0.7`), backbone (`Dr.GRPO+CISPO -> GSPO`), and stochastic `BF16` consolidation, the `+24.08 pp` `Holdout_V4_1` gain of `Pure GSPO` reflects their combined recipe effect rather than rounding alone; the `Gate G4` contrasts below hold all four fixed on a common `40`-step `T=0.7` `GSPO` backbone.

### 5.2 Common-Backbone `GSPO` Ablation (`Muse-Glimmer-30B` Math, `40` Steps, `T=0.7`, `Search_V3`)

All RL arms below share the **exact same `240`-prompt `B_0` rollout cache** (`43` mixed on-policy student groups, `37` saturated `8/8` groups skipped, `160` zero-pass `0/8` student groups; `40` optimizer steps = `~2.8` passes over `57` active groups). On `14` zero-pass prompts where a verified `OpenMathReasoning` solution formatted in native `to=self` + `to=user` channels fits within `8,192` tokens, `faithful_luffy` and `log_shaped` form `1` teacher + `7` student-negative groups (`14 / 57 = 24.6%` active group share; `14 / 456 = 3.07%` off-policy sample share, keeping `96.93%` of training rollouts on concise student trajectories):

| Arm (`Common GSPO Backbone`, `T_train=T_eval=0.7`) | Off-Policy Share (`Group`, `Sample`) | Teacher Token-$p$ (`32` Step Records) | Math `avg@8` | Math `pass@8` | `stop_rate` | Mean Toks | `distinct@8` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `b0_base` | `0.0%` (`0.0%`) | — | **51.09%** | 68.75% | 94.45% | `3453` | `3.42` |
| `muse_math_gspo_pure_onpolicy` | `0.0%` (`0.00%`) | — | **67.03%** | 82.50% | 88.59% | `4194` | `2.14` |
| `muse_math_gspo_faithful_luffy` | `24.6%` (`3.07%`) | med `0.9989` (mean `0.9961`), `<1e-3`: med `1.66%` (mean `1.75%`) | **72.73%** | 83.75% | 92.19% | `3822` | `2.02` |
| `muse_math_gspo_log_shaped` | `24.6%` (`3.07%`) | med `0.9972` (mean `0.9936`), `<1e-3`: med `0.71%` (mean `0.98%`) | **59.92%** | 87.50% | 95.08% | `2515` | `3.34` |
| `muse_math_sft_channel_aligned` | `0.0%` (`0.0%`) | — | **63.91%** | 83.75% | 74.69% | `5657` | `1.43` |
| `muse_math_sft_aligned_then_gspo` | `0.0%` (`0.00%`) | — | **64.30%** | 83.75% | 77.19% | `5359` | `1.52` |

---

## 6. Sealed Unexposed `Confirm_V4_1` (`N=150`) & Single-Touch `Holdout_V4_1` (`N=150`) Verdicts

### 6.1 Gate Decisions & Paired Effect Sizes

| Cell (`50/50/50` Balanced Strata) | Evaluated Candidate | `Confirm_V4_1` (`alpha=0.025`, `K=1`) | `Confirm_V4_1` `Delta avg@8` vs. `B_0` | Single-Touch `Holdout_V4_1` `Delta avg@8` vs. `B_0` |
| :--- | :--- | :---: | :---: | :---: |
| `lean_inkling2_mini` | `inkling_lean_sft_effort_matched` | `ROLLBACK` *(diag. contrast)* | `-0.75 pp` (guard `97.5%` LCB `-2.83 pp`, `95%` CI `[-2.83 pp, +1.42 pp]`) | *Untouched (`paired_aux_noninferiority_failed:stop_rate:lcb=-0.176667<=-0.050000, paired_aux_noninferiority_failed:lean_pass_at_8:lcb=-0.086667<=-0.020000`)* |
| `lean_muse_glimmer_30b` | `muse_lean_sft_matched_plus_gspo` | `ROLLBACK` | `+4.33 pp` (guard `97.5%` LCB `+1.08 pp`, `95%` CI `[+1.00 pp, +7.83 pp]`) | *Untouched (`paired_aux_noninferiority_failed:stop_rate:lcb=-0.100833<=-0.050000`)* |
| `math_muse_glimmer_30b` | `muse_math_gspo_faithful_luffy` | `PROMOTE` | `+25.08 pp` (guard `97.5%` LCB `+19.25 pp`, `95%` CI `[+19.50 pp, +30.83 pp]`) | **`+30.58 pp`** (guard `97.5%` LCB `+24.58 pp`, `95%` CI `[+24.50 pp, +36.67 pp]`, MC8000 `p <= 0.000125`) |

### 6.2 Full Unexposed `Confirm_V4_1` & `Holdout_V4_1` Score Breakdown

| Cell | Unexposed Split | Checkpoint | `avg@8` | `pass@8` | `stop_rate` | `has_thinking_rate` | Mean Toks |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `lean_inkling2_mini` | `Confirm_V4_1 (N=150)` | `b0_base` | **29.92%** | 45.33% | 65.75% | 100.00% | `1950` |
| `lean_inkling2_mini` | `Confirm_V4_1 (N=150)` | `inkling_lean_sft_effort_matched` | **29.17%** | 42.67% | 52.08% | 97.17% | `2234` |
| `lean_inkling2_mini` | `Confirm_V4_1 (N=150)` | `inkling_lean_sft_stripped` | **23.42%** | 34.00% | 35.42% | 99.83% | `2488` |
| `lean_muse_glimmer_30b` | `Confirm_V4_1 (N=150)` | `b0_base` | **16.17%** | 27.33% | 87.42% | 100.00% | `1330` |
| `lean_muse_glimmer_30b` | `Confirm_V4_1 (N=150)` | `muse_lean_sft_matched_plus_gspo` | **20.50%** | 34.67% | 82.58% | 0.00% | `913` |
| `math_muse_glimmer_30b` | `Confirm_V4_1 (N=150)` | `b0_base` | **42.75%** | 66.00% | 97.42% | 100.00% | `2511` |
| `math_muse_glimmer_30b` | `Holdout_V4_1 (N=150)` | `b0_base` | **42.75%** | 64.67% | 98.17% | 100.00% | `2408` |
| `math_muse_glimmer_30b` | `Confirm_V4_1 (N=150)` | `muse_math_gspo_faithful_luffy` | **67.83%** | 82.67% | 96.67% | 100.00% | `2800` |
| `math_muse_glimmer_30b` | `Holdout_V4_1 (N=150)` | `muse_math_gspo_faithful_luffy` | **73.33%** | 90.00% | 98.25% | 100.00% | `2665` |
| `math_muse_glimmer_30b` | `Confirm_V4_1 (N=150)` | `muse_math_gspo_pure_onpolicy` | **63.75%** | 74.67% | 95.67% | 100.00% | `3182` |
| `math_muse_glimmer_30b` | `Holdout_V4_1 (N=150)` | `muse_math_gspo_pure_onpolicy` | **66.83%** | 82.67% | 96.25% | 100.00% | `3007` |

---

## 7. Summary of Findings & Limitations

1. **Pretraining Priors Are Domain-Orthogonal Across Families:** Under `think=off` at equalized `3,072` tokens, `Inkling-2-Mini` (`33.39%` `effort=none`, `35.80%` `effort=low` on `Search_V3`; `17.06%` `Holdout_V3`) and `Qwen3.5/3.8` (`15.28%–17.89%` `Holdout_V3`) outperform `Gemma-4` (`4.73%–10.54%` `Search_V3`; `5.94%–7.00%` `Holdout_V3`) by `2.4x–7.1x` on Lean 4.24 kernel verification, whereas `Gemma-4-31B` (`42.91%` `Holdout_V3`, `61.19%` `2025` 4-Contest) leads raw competition mathematics ahead of `Qwen3.8-27B` (`40.40%` `Holdout_V3`, `57.25%` Contest) and `Gemma-4-26B-A4B` (`38.66%` `Holdout_V3`, `53.94%` Contest at `3.8B` single-pass active parameters) and `Muse-Glimmer-30B` (`37.63%` `Holdout_V3`, `59.88%` Contest).
2. **Reasoning-Channel Alignment Partially Recovers Thoughtless Lean SFT—Only When Channel Tokens Are Disjoint:** Holding all `736,175` supervised proof tokens and `74` steps fixed on `Inkling-2-Mini`, binding thoughtless proofs to `Thinking effort level: 0` recovers `+4.20 pp avg@8` (`Search_V3`) and **`+5.75 pp avg@8` (`95%` CI `[+3.33 pp, +8.50 pp]`) plus `+16.67 pp stop_rate` on unexposed `Confirm_V4_1`**, though residual `stop_rate` degradation (`-13.67 pp` vs. `B_0`) prevents beating raw `B_0`. On `Muse-Glimmer-30B`, where `to=self` and `to=user` share `<|start|>assistant`, thoughtless SFT under `strength=none` lifts `Confirm_V4_1` `avg@8` by `+4.33 pp` (`16.17% -> 20.50%`) but still suppresses `has_thinking_rate` to `0.00%`.
3. **Why Prior Off-Policy `LUFFY` Stalled—and How Mixed-Policy `LUFFY` Outperforms Pure `GSPO` and Offline `SFT -> GSPO`:** Fixing five implementation bottlenecks on `Muse-Glimmer-30B` Math (`1` `to=self`-aligned teacher + `7` on-policy student negatives on `14` zero-pass prompts [`24.6%` group share, `3.07%` sample share], `T_train = T_eval = 0.7`, `40` `GSPO` steps, and stochastic `FP32 -> BF16` merging) reveals that teacher token probabilities are bimodal (median `p50` `0.9989` [mean `0.9961`], with median `1.66%` [mean `1.75%`] rare pivot tokens below $10^{-3}$). In this controlled single-seed comparison:
   - **`Log-Shaped` ($w(p)=\gamma/(p+\gamma)$)** retains higher answer diversity (`3.34` `distinct@8` vs. `2.02`, `2,515` mean tokens) and reaches the highest `Search_V3` `pass@8` (`87.50%`), at lower `avg@8` (`59.92%`).
   - **`Faithful LUFFY` ($\frac{df}{d\log p} = \frac{\gamma p}{(p+\gamma)^2}$)** sharpens consensus (`2.02` `distinct@8`, `3,822` mean tokens on Search; `2,665` on Holdout) without inheriting `OpenMathReasoning`'s long-CoT length drift (`Channel-Aligned SFT -> GSPO`: `5,359` mean tokens, `77.19%` `stop_rate`, `64.30%` `avg@8`), replicating cleanly from **`Search_V3` (`72.73%`) -> `Confirm_V4_1` (`67.83%`, `+25.08 pp` vs. `B_0`, `+4.08 pp` vs. `Pure GSPO` [`95%` CI `[+1.58 pp, +6.67 pp]`]) -> single-touch `Holdout_V4_1` (`73.33%`, `+30.58 pp` vs. `B_0` [guard `97.5%` LCB `+24.58 pp`, MC8000 `p <= 1.25e-4`], `+6.50 pp` vs. `Pure GSPO` [`95%` CI `[+4.00 pp, +9.25 pp]`], `90.00% pass@8`, `98.25% stop_rate`)**.
4. **Limitations:** (i) All `26B–31B` post-training runs use a single training seed (`20261124`) over a static `240`-prompt rollout cache (`~2.8` passes across `40` steps), so CIs quantify problem-sampling uncertainty conditional on the trained checkpoint rather than multi-seed training variance or online multi-epoch curriculum dynamics; (ii) `w_log(p)` and `df/d(log p)` differ in overall teacher gradient norm alongside rare-token shape; (iii) `Gemma-4` `think=on` Lean requires `>=12,288` generation tokens to avoid CoT truncation; and (iv) Lean RLVR rewards rely on whole-proof kernel verification rather than intermediate tactic-state feedback.

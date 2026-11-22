# Governed Reasoning Post-Training & Self-Improvement (`3.8B` Dense to `35B-A3B` Sparse MoE)

Systematic post-training and verifier-gated self-improvement across **Dense models (`3.8B`–`27B`)** and **`3B`-Active Sparse Mixture-of-Experts (`30B-A3B`, `35B-A3B`)** on **Competition / 2026 Research Mathematics** and **Formal Lean 4.24 Theorem Proving**.

## Overview

We combine **model-aware verifiable data construction** (`2 SFT + RL` and zero-SFT `RLVR` crossovers, without mid-training or DPO) with **executable two-plane governance** and **independent multi-agent external model review**:
1. **Plane 1 (Pre-Freeze Calibration):** Channel and token-budget alignment, deterministic out-of-process verifiers (`Lean v4.24.0 + Mathlib` `#print axioms` kernel check and timeout-isolated symbolic math grading), empirical `A/A` null checks (`10/10 -> ROLLBACK`), and SHA-256-locked `Search -> Confirmation -> Single-Touch Holdout` manifests.
2. **Plane 2 (Diagnostic Post-Training & Multi-Proposal RLVR):**
   - **Breadth (`2-SFT`):** Horizon-matched multi-domain verified SFT + Zone-of-Proximal-Development (`ZPD`) frontier strengthening (`DDP` and 8-GPU `FSDP FULL_SHARD`; on Sparse MoEs, adapting only `0.17%–0.38%` unconditionally active shared projections with all `128`/`256` routed experts frozen).
   - **Depth & Termination (`RLVR`):** Checkpointed-chunk `12K` verifier RL tournaments (`DAPO`, `Dr.GRPO + CISPO`, `GSPO + LUFFY`) to sharpen `pass@1`, `pass@2`, and `maj@8` while eliminating runaway truncation (`stop_rate`).
3. **Statistical Promotion & Single-Touch Holdout:** Hash-chained trial ledgers, Intersection-Union (`IUT`) / Bonferroni (`K=2`) Confirmation gates (`+1.50 pp` LCB margin + `stop_rate` safety ceiling), and write-once `HOLDOUT_SEAL` evaluation (`O_CREAT | O_EXCL`). Every candidate promoted by `Confirm_V3` replicated positively on Single-Touch `Holdout_V3`, while all unpromoted candidates made `0` candidate Holdout evaluations.

---

## Headline Results

### 1. Phase V3 — Dense (`4B`, `9B`, `27B`) & `3B`-Active Sparse MoE (`30B-A3B`, `35B-A3B`) Across All 10 Cells (`k=8`)

Holdout suites: **Track L** (`N=225`: `arXivLean 2026` + `FormalMATH-Lite`) and **Track M** (`N=194`: `SMT 2025`, `HMMT Feb 2025`, `APEX 2025`, `arXivMath 2026`), plus external longitudinal transfer (`MiniF2F-test` `N=244`, `ProofNet-test` `N=163`, `2025` 4-Contest Math `N=200`). Fixed-checkpoint evaluations (`claim_scope = fixed_checkpoint`).

| Model (`Total / Active`) | Track | Search `B_0 -> Winner` `avg@8` (`Recipe`) | `Confirm_V3` Verdict (`K <= 2`) | Sealed `Holdout_V3` `avg@8` (`pass@8`) | External Longitudinal (`B_0 -> Winner avg@8`) |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **`Qwen3.5-4B` (`4B / 4B`)** | **Lean** | `30.45% -> 36.96%` (`2-SFT + CISPO`) | Hold V1 `A*` *(IUT `< +1.5 pp`)* | **`15.28% -> 19.39%` (`+4.11 pp`)** (`27.56%`) | `MiniF2F`: `28.33% -> 29.30%`; `ProofNet`: `1.84% -> 2.38%` |
| **`Qwen3.5-9B` (`9B / 9B`)** | **Lean** | `33.57% -> 39.55%` (`2-SFT + CISPO`) | **`PROMOTE`** (`+6.81 pp`) | **`16.22% -> 19.28%` (`+3.06 pp`)** (`27.11%`) | **`ProofNet`: `2.15% -> 5.29%` (`2.46x`)**; `MiniF2F`: `30.28% -> 33.61%` |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | **Lean** | `33.48% -> 42.77%` (`2-SFT + DAPO`) | **`PROMOTE`** (`+7.50 pp`) | **`16.89% -> 22.06%` (`+5.17 pp`)** (**`31.56%`**) | **`MiniF2F pass@8`: `48.36% -> 51.64%`**; `ProofNet`: `2.68% -> 3.45%` |
| **`Qwen3-30B-A3B` (`30B / 3B` MoE)** | **Lean** | **`4.64% -> 24.91%` (`5.37x`, `S1 + CISPO`)** | **`PROMOTE`** (`+16.11 pp`, `5.00x`) | **`1.00% -> 11.33%` (`+10.33 pp`, `11.33x`)** (`19.11%`) | **`MiniF2F`: `1.02% -> 15.47%` (`15.17x`, `29.51% pass@8`)** |
| **`Qwen3.5-35B-A3B` (`35B / 3B` MoE)** | **Lean** | **`30.54% -> 40.71%` (`S1 + CISPO400`)** | Hold `B_0` *(`+5.28 pp` L2, LCB `+1.11 pp`)* | `17.89%` *(zero-shot `B_0`, `26.67% pass@8`; `0` cand touches)* | **`MiniF2F`: `35.25% -> 37.65%` (`52.46% pass@8`)**; `ProofNet`: `3.60% -> 4.68%` |
| *`Goedel-Prover-V2-8B` (`8B` Ref)* | *Lean* | *Zero-shot `2,048`-tok budget ref* | — | *`17.61%` (`21.78% pass@8`, `20.22% stop_rate`)* | *Truncates `79.78%` under shared `2,048`-token cap* |
| **`Qwen3.5-4B` (`4B / 4B`)** | **Math** | `64.92% -> 76.02%` (`2-SFT + GSPO+LUFFY`) | Hold V2 `A*` *(IUT `< +1.5 pp`)* | **`24.16% -> 29.38%` (`+5.22 pp`)** (`52.06%`) | `2025` 4-Contest (`N=200`): **`36.25% -> 43.69%` (`+7.44 pp`)** |
| **`Qwen3.5-9B` (`9B / 9B`)** | **Math** | **`66.25% -> 80.39%` (`2-SFT + GSPO+LUFFY`)** | **`PROMOTE` (vs `B_0` & V2)** | **`23.90% -> 36.08%` (`+12.18 pp` vs `9B B_0`)** (`56.70%`) | `2025` 4-Contest (`N=200`): **`38.00% -> 49.50%` (`+11.50 pp`)** |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | **Math** | `79.30% -> 82.58%` (`2-SFT + GSPO+LUFFY`) | Hold `B_0` *(`95% pass@8` ceiling)* | `40.40%` *(zero-shot `B_0`, `59.79% pass@8`; `0` cand touches)* | `2025` 4-Contest (`N=200`): `57.25%` (`B_0` retained) |
| **`Qwen3-30B-A3B` (`30B / 3B` MoE)** | **Math** | `57.42% -> 62.11%` (R2) / `59.14%` (`Pure-B0 CISPO`) | Hold `B_0` *(L1 stop guard; L2 `+2.79 pp`, LCB `+0.58 pp`)* | `16.37%` *(zero-shot `B_0`, `32.99% pass@8`; `0` cand touches)* | `stop_rate` guard (`99.8% -> 96.7%`) prevented contest truncation |
| **`Qwen3.5-35B-A3B` (`35B / 3B` MoE)** | **Math** | **`56.87% -> 78.98%` (`+22.11 pp`, `Pure-B0 DAPO`)** | **`PROMOTE`** (`+21.50 pp`: `61.46% -> 82.96%`) | **`20.43% -> 40.34%` (`+19.91 pp`)** (**`59.28%`**) | **`2025` 4-Contest: `27.63% -> 57.00%` (`+29.37 pp`, `0/200` forgotten)** |

**Key Cross-Architecture Takeaways (Detailed in [`reports/03_phase_v3_data_scaling_rl_and_27b_results.md`](reports/03_phase_v3_data_scaling_rl_and_27b_results.md)):**
- **Breadth vs. Termination-Depth Diagnostic:** When `B_0` already possesses high latent breadth (`pass@8 = 82.50%`) but suffers long-CoT non-termination (`stop_rate = 62.89%` on `Qwen3.5-35B-A3B` Math), `2-SFT` stagnates (`56.87%`) whereas **Zero-SFT On-Policy `DAPO + Overlong`** unlocks **`+19.91 pp` on Single-Touch `Holdout_V3` (`40.34%`)** — descriptively matching zero-shot `Qwen3.8-27B` Dense (`40.40%`) at `9x` fewer active parameters (`3B` vs `27B`). Conversely, when `B_0` is breadth-bottlenecked (`Qwen3-30B-A3B` Lean `1.00%` Holdout, or `Qwen3.5-35B-A3B` Lean where `Pure-B0 RL` hits `35.36%` vs `40.71%` with Breadth SFT), **Verified Breadth SFT + `Dr.GRPO + CISPO`** is essential (`11.33x` Holdout & `15.17x` `MiniF2F` on `30B-A3B` updating only `0.17%` `self_attn` weights).
- **Fail-Closed Horizon & Multiplicity Discipline:** Supervising concise non-thinking `Qwen3-30B-A3B` (`99.83% stop_rate`) on `10K` long-CoTs gained `+3.38 pp` on `Confirm_V3` Look 1 (`60.54% -> 63.92%`) but tripped the `2.50 pp` `stop_rate` degradation guard (`-3.13 pp`), automatically blocking a checkpoint that truncated on external contests; native-horizon `Pure-B0 + CISPO` restored `99.92% stop_rate` (`+2.79 pp`, `INCONCLUSIVE_RETEST` at Bonferroni `alpha/2 = 0.0125`), keeping `Holdout_V3` untouched.

### 2. Olympiad & 2025 Contest Mathematics — Phase V2 (`N=200`, `k=8`, Zero Teacher Traces)

Benchmarks: `BeyondAIME` (`100`), `CMIMC 2025` (`40`), `BRUMO 2025` (`30`), `HMMT Nov 2025` (`30`).

| Model (`Architecture`) | Stage | `BeyondAIME` | `CMIMC 25` | `BRUMO 25` | `HMMT Nov 25` | Pooled `avg@8` (`Delta`, 95% CI) | Equal-Stratum `avg@8` | `pass@8` | `stop_rate` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Phi-4-mini` (`3.8B` LongRoPE) | Base `B_0` | `14.38%` | `20.00%` | `33.75%` | `16.67%` | `18.75%` *(3/3 Search rollbacks)* | `21.20%` | `33.00%` | `33.50%` |
| `Qwen3-4B` (`4B` Dense) | Base `B_0` | `27.63%` | `30.00%` | `51.67%` | `37.50%` | `33.19%` *(Confirm `+2.49 pp`, held)* | `36.70%` | `52.50%` | `44.88%` |
| `Qwen3.5-4B` (`4B` Hybrid GDN) | Base `B_0` | `29.50%` | `37.50%` | `53.75%` | `38.75%` | `36.13%` | `39.88%` | `66.00%` | `66.50%` |
| **`Qwen3.5-4B` (`4B` Hybrid GDN)** | **Tuned `A*`** | **`37.75%`** | **`38.13%`** | **`61.67%`** | **`50.42%`** | **`43.31%` (`+7.19 pp`, `[+4.81, +9.63]`)** | **`46.99%` (`+7.11 pp`)** | **`70.50%`** | **`88.00%`** |
| `Qwen3.5-9B` (`9B` Hybrid GDN) | Base `B_0` | `32.25%` | `35.00%` | `57.92%` | `44.17%` | `38.44%` | `42.33%` | `70.00%` | `55.38%` |
| **`Qwen3.5-9B` (`9B` Hybrid GDN)** | **Tuned `A*`** | **`42.13%`** | **`46.25%`** | **`71.67%`** | **`52.50%`** | **`48.94%` (`+10.50 pp`, `[+7.88, +13.19]`)** | **`53.14%` (`+10.80 pp`)** | **`74.00%`** | **`80.88%`** |
| `Qwen3.8-27B` (`27B` Hybrid GDN) | Untuned Ref | `50.75%` | `53.44%` | `77.50%` | `67.08%` | `57.75%` | `62.19%` | `82.00%` | `98.13%` |

### 3. Phase V1 Competition Math & Formal Lean 4.24 Holdout (`Qwen3.5-4B`, `k=8`)

| Track & Benchmark (`N`) | Base `B_0` (`avg@8` / `pass@8`) | **Tuned `A*` (`avg@8` / `pass@8`)** | Paired `avg@8` Gain (`95% CI`) | Untuned `9B` Ref | Untuned `27B` Ref |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lean 4.24 — `MiniF2F-v2s test` (`N=244`)** | `27.66%` / `43.44%` | **`33.15%` / `45.90%`** (`100%` axiom-clean) | **`+5.48 pp` (`[+2.92, +8.09]`)** | `30.02%` / `45.90%` | `32.27%` / `48.77%` |
| **Math — `AIME 25 + AIME 26 + HMMT Feb 26 + OlympiadBench` (`N=213`)** | `55.52%` / `78.87%` | **`59.45%` / `84.04%`** (`AIME 25 pass@8: 86.67%`) | **`+3.93 pp` (`[+1.47, +6.40]`)** | `55.40%` / `82.63%` | `74.65%` / `91.08%` |
| **Lean OOD — `ProofNet-test` (`N=167`, Phase V2)** | `1.72%` / `4.19%` | *Held at `B_0` (Confirm OOD rollback)* | — | `2.69%` / `7.78%` | `2.62%` / `7.19%` (`Goedel-8B: 7.86%`) |

---

## Dedicated Technical Reports

- **[`reports/01_post_training_methodology_and_data_scaling.md`](reports/01_post_training_methodology_and_data_scaling.md)** — Data-hygiene preconditions, Breadth (`SFT`) vs. Depth (`RLVR`) capability-curve dynamics (`Δpass@8 - Δpass@1`), RLVR algorithm comparisons (`DAPO`, `Dr.GRPO + CISPO`, `GSPO + LUFFY`), and `Axion 1.0`-style `2 SFT + RL` data scaling.
- **[`reports/02_phase_v1_v2_slm_results.md`](reports/02_phase_v1_v2_slm_results.md)** — Full Phase V1 & V2 Search/Confirm/Holdout tables, 3-arm recursion and 4-way compiler-repair causal isolations, and statistical audit notes.
- **[`reports/03_phase_v3_data_scaling_rl_and_27b_results.md`](reports/03_phase_v3_data_scaling_rl_and_27b_results.md)** — Complete 10-cell Phase V3 + Sparse MoE study (`4B`, `9B`, `27B` Dense + `30B-A3B`, `35B-A3B` MoE), Round-2/Round-3 crossovers, `equal_stratum` sensitivity analysis, Single-Touch Holdout seals, and 6-benchmark longitudinal transfer.

---

## Repository Layout & Verification

- `IMMUTABLE_MANIFEST.json`, `IMMUTABLE_MANIFEST_V2.json`, `IMMUTABLE_MANIFEST_V3.json`, `IMMUTABLE_MANIFEST_V3_MOE.json`: Cryptographic split, scorer, and rule manifests.
- `eval/`, `eval_v2/`, `eval_v3/`: Out-of-process Lean 4.24 (`#print axioms`) and timeout-isolated symbolic math verifiers + `governed_loop_guard` (`v3.7`).
- `code/`, `code/v2/`, `code/v3/`: Training (`ReST^EM`, `DDP` & 8-GPU `FSDP FULL_SHARD` `2-SFT`, Checkpointed-Chunk `12K` RLVR across `DAPO`, `Dr.GRPO+CISPO`, `GSPO+LUFFY`, and frozen-router Sparse MoE shared-projection LoRA), multi-GPU `vLLM` evaluation, and report compilers.
- `results/`, `results_v2/`, `results_v3/`: Verified hash-chained ledgers, `HOLDOUT_SEAL` records, item-level MoE Confirm/Holdout scores, and consolidated report bundles.

```bash
python -I -B eval/governed_loop_guard.py verify-manifest --manifest IMMUTABLE_MANIFEST.json --root .
python -I -B code/v2/governed_loop_guard_v3.py verify-manifest --manifest IMMUTABLE_MANIFEST_V2.json --root . --expected-root-sha256 f79f2fcf32262a0455ba85256ea2c4a8ab72b3eef94c76f0399190932e74ccf2
python -I -B eval_v3/governed_loop_guard_v37.py verify-manifest --manifest IMMUTABLE_MANIFEST_V3.json --root . --expected-root-sha256 7940ee090af186cace0dd32518866315716c7ff0d6859e3603ff2f3e52dfaa94
python -I -B eval_v3/governed_loop_guard_v37.py verify-manifest --manifest IMMUTABLE_MANIFEST_V3_MOE.json --root . --expected-root-sha256 da03dc170a7062e7b50e8359e9158b61f31664cbd23a7244b27a3da730ebad81
python -I -B code/v3/build_v3_report_bundle.py .
```
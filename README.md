# Governed Reasoning Post-Training & Self-Improvement (`3.8B` to `35B-A3B` MoE)

Systematic post-training and verifier-gated self-improvement for language models (`3.8B`–`27B` Dense and `35B-A3B` Sparse MoE) on **Competition / Research Mathematics** and **Formal Lean 4.24 Theorem Proving**.

## Overview

We combine **model-aware verifiable data construction** (`2 SFT + RL`, without mid-training or DPO) with **executable two-plane governance** and **independent external model review**:
1. **Plane 1 (Pre-Freeze Calibration):** Token-budget and channel alignment, deterministic out-of-process verifiers (`Lean v4.24.0 + Mathlib` `#print axioms` kernel check and timeout-isolated symbolic math grading), and SHA-256-locked `Search -> Confirmation -> Single-Touch Holdout` split manifests.
2. **Plane 2 (Post-Training & RLVR Optimization):**
   - **Breadth (`2-SFT`):** Horizon-aligned multi-domain verified reasoning SFT + model-aware Zone-of-Proximal-Development (`ZPD`) hard-frontier strengthening (`DDP` and 8-GPU `FSDP FULL_SHARD`).
   - **Depth (`RLVR`):** Activation-checkpointed `12K` chunked-logits verifier RL tournament (`DAPO`, `Dr.GRPO + CISPO`, `GSPO + LUFFY`) on nonzero-variance prompts to sharpen `pass@1`, `pass@2`, and `maj@8` while raising natural completion rates (`stop_rate`).
3. **Statistical Promotion & Single-Touch Holdout:** Hash-chained trial ledgers, Intersection-Union (`IUT`) / Holm-Bonferroni Confirmation gates against `B_0` and prior champions, and write-once `HOLDOUT_SEAL` evaluation (`O_CREAT | O_EXCL` + SHA-256 receipts).

---

## Headline Results

### 1. Phase V3 — `2-SFT + 3-Proposal RLVR` Across `4B`, `9B`, `27B` Dense (`FSDP`), and `35B-A3B` Sparse MoE (`k=8`)

Holdout suites: **Track L** (`N=225`: `arXivLean 2026` + `FormalMATH-Lite` holdout) and **Track M** (`N=194`: `SMT 2025`, `HMMT Feb 2025`, `APEX 2025`, `arXivMath 2026`), plus external longitudinal transfer (`MiniF2F-test`, `ProofNet-test`, `2025` 4-Contest Math).

| Model (`Total / Active`) | Track | Search Winner (`B_0 -> Winner avg@8`) | `Confirm_V3` Verdict | Sealed `Holdout_V3` (`B_0 -> Promoted avg@8` / `pass@8`) | External Longitudinal Highlight (`B_0 -> V3`) |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **`Qwen3.5-4B` (`4B / 4B`)** | **Lean** | `Dr.GRPO+CISPO`: `30.45% -> 36.96%` (`+6.52 pp`) | Hold V1 `A*` *(IUT)* | **`15.28% -> 19.39%` (`+4.11 pp`)** / `24.00% -> 27.56%` | `MiniF2F`: `28.33% -> 29.30%`; `ProofNet`: `1.84% -> 2.38%` |
| **`Qwen3.5-9B` (`9B / 9B`)** | **Lean** | `Dr.GRPO+CISPO`: `33.57% -> 39.55%` (`+5.98 pp`) | **`PROMOTE`** | **`16.22% -> 19.28%` (`+3.06 pp`)** / `24.44% -> 27.11%` | **`ProofNet`: `2.15% -> 5.29%` (`2.46x`!)**; `MiniF2F`: `30.28% -> 33.61%` |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | **Lean** | `DAPO (pct50)`: `33.48% -> 42.77%` (`+9.29 pp`) | **`PROMOTE`** | **`16.89% -> 22.06%` (`+5.17 pp`)** / **`26.22% -> 31.56%`** | **`MiniF2F pass@8`: `48.36% -> 51.64%`**; `ProofNet stop`: `29.8% -> 88.8%` |
| **`Qwen3.8-35B-A3B` (`35B / 3B` MoE)** | **Lean** | **`GSPO+LUFFY`: `31.25% -> 42.95%` (`+11.70 pp`)** | *In progress* | *In progress* | Matches tuned `27B` Search (`42.95%` vs `42.77%`) at `9x` lower active FLOPs |
| **`Qwen3.5-4B` (`4B / 4B`)** | **Math** | `GSPO+LUFFY`: `64.92% -> 76.02%` (`+11.10 pp`) | Hold V2 `A*` *(IUT)* | **`24.16% -> 29.38%` (`+5.22 pp`)** / `48.97% -> 52.06%` | `2025` 4-Contest (`N=200`): **`36.25% -> 43.69%` (`+7.44 pp`)** |
| **`Qwen3.5-9B` (`9B / 9B`)** | **Math** | **`GSPO+LUFFY`: `66.25% -> 80.39%` (`+14.14 pp`)** | **`PROMOTE` (vs `B_0` & V2)** | **`23.90% -> 36.08%` (`+12.18 pp` vs `9B B_0`)** / `48.97% -> 56.70%` | `2025` 4-Contest (`N=200`): **`38.00% -> 49.50%` (`+11.50 pp`)** |
| **`Qwen3.8-27B` (`27B / 27B` FSDP)** | **Math** | `GSPO+LUFFY (pct50)`: `79.30% -> 82.58%` (`+3.28 pp`) | Hold `B_0` *(`95% pass@8` ceiling)* | `40.40% avg@8` (`45.88% maj@8`, `59.79% pass@8` zero-shot `B_0`) | `2025` 4-Contest: `57.25%` (`CMIMC`: `49.69% -> 53.75%`) |
| **`Qwen3.8-35B-A3B` (`35B / 3B` MoE)** | **Math** | **`DAPO`: `58.59% -> 79.77%` (`+21.18 pp` so far)** | *In progress* | *In progress* | RLVR lifts `stop_rate` **`65.62% -> 93.36%` (`+27.74 pp`)** vs `2-SFT` `67.97%` |

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
- **[`reports/03_phase_v3_data_scaling_rl_and_27b_results.md`](reports/03_phase_v3_data_scaling_rl_and_27b_results.md)** — Phase V3 horizon-alignment study, 3-proposal RLVR tournament across `4B`/`9B`/`27B` Dense (`FSDP`) and `35B-A3B` Sparse MoE (`FSDP`), IUT Confirmation & Single-Touch Holdout seals, and 6-benchmark longitudinal transfer.

---

## Repository Layout & Verification

- `IMMUTABLE_MANIFEST.json`, `IMMUTABLE_MANIFEST_V2.json`, `IMMUTABLE_MANIFEST_V3.json`: Cryptographic split, scorer, and rule manifests.
- `eval/`, `eval_v2/`, `eval_v3/`: Out-of-process Lean 4.24 (`#print axioms`) and symbolic math verifiers + `governed_loop_guard` (`v3.7`).
- `code/`, `code/v2/`, `code/v3/`: Training (`ReST^EM`, `DDP` & 8-GPU `FSDP FULL_SHARD` `2-SFT`, Checkpointed-Chunk `12K` RLVR across `DAPO`, `Dr.GRPO+CISPO`, `GSPO+LUFFY`), multi-GPU `vLLM` evaluation, and report compilers.
- `results/`, `results_v2/`, `results_v3/`: Verified hash-chained ledgers, `HOLDOUT_SEAL` records, and Holdout SHA-256 receipts.

```bash
python -I -B eval/governed_loop_guard.py verify-manifest --manifest IMMUTABLE_MANIFEST.json --root .
python -I -B code/v2/governed_loop_guard_v3.py verify-manifest --manifest IMMUTABLE_MANIFEST_V2.json --root . --expected-root-sha256 f79f2fcf32262a0455ba85256ea2c4a8ab72b3eef94c76f0399190932e74ccf2
python -I -B code/v2/build_v2_report_bundle.py .
```
# Governed Reasoning Post-Training & Self-Improvement (`3.8B` to `27B`)

Systematic post-training and verifier-gated self-improvement for language models (`3.8B`–`27B`) on **Competition / Olympiad Mathematics** and **Formal Lean 4.24 Theorem Proving**.

## Overview

We combine **model-aware verifiable data construction** (`2 SFT + RL`, without mid-training or DPO) with **executable two-plane governance** and **multi-agent external frontier review**:
1. **Plane 1 (Pre-Freeze Calibration):** Token-budget and channel alignment, deterministic out-of-process verifiers (`Lean v4.24.0 + Mathlib` `#print axioms` kernel check and timeout-isolated symbolic math grading), and SHA-256-locked `Search -> Confirmation -> Single-Touch Holdout` split manifests.
2. **Plane 2 (Post-Training & RLVR Optimization):**
   - **Breadth (`SFT`):** Multi-domain verified reasoning SFT + model-aware hard-frontier strengthening to expand reachable problem coverage (`pass@k`).
   - **Depth (`RLVR`):** Activation-checkpointed `12K` chunked-logits verifier RL (`DAPO`, `Dr.GRPO + CISPO`, `GSPO + LUFFY`) on nonzero-variance prompts to sharpen `pass@1`, `pass@2`, and `maj@8` while raising natural completion rates (`stop_rate`).
3. **Statistical Promotion & Single-Touch Holdout:** Hash-chained trial ledgers, Holm-Bonferroni Confirmation gates, and write-once `HOLDOUT_SEAL` evaluation (`O_CREAT | O_EXCL` + SHA-256 receipts).

---

## Headline Single-Touch Holdout Results

### 1. Olympiad & 2025 Contest Mathematics — Phase V2 (`N=200`, `k=8`, Zero Teacher Traces)

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

### 2. Phase V1 Competition Math & Formal Lean 4.24 Holdout (`Qwen3.5-4B`, `k=8`)

| Track & Benchmark (`N`) | Base `B_0` (`avg@8` / `pass@8`) | **Tuned `A*` (`avg@8` / `pass@8`)** | Paired `avg@8` Gain (`95% CI`) | Untuned `9B` Ref | Untuned `27B` Ref |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lean 4.24 — `MiniF2F-v2s test` (`N=244`)** | `27.66%` / `43.44%` | **`33.15%` / `45.90%`** (`100%` axiom-clean) | **`+5.48 pp` (`[+2.92, +8.09]`)** | `30.02%` / `45.90%` | `32.27%` / `48.77%` |
| **Math — `AIME 25 + AIME 26 + HMMT Feb 26 + OlympiadBench` (`N=213`)** | `55.52%` / `78.87%` | **`59.45%` / `84.04%`** (`AIME 25 pass@8: 86.67%`) | **`+3.93 pp` (`[+1.47, +6.40]`)** | `55.40%` / `82.63%` | `74.65%` / `91.08%` |
| **Lean OOD — `ProofNet-test` (`N=167`, Phase V2)** | `1.72%` / `4.19%` | *Held at `B_0` (Confirm OOD rollback)* | — | `2.69%` / `7.78%` | `2.62%` / `7.19%` (`Goedel-8B: 7.86%`) |

---

## Dedicated Technical Reports

- **[`reports/01_post_training_methodology_and_data_scaling.md`](reports/01_post_training_methodology_and_data_scaling.md)** — Data-hygiene preconditions, Breadth (`SFT`) vs. Depth (`RLVR`) capability-curve dynamics (`Δpass@8 - Δpass@1`), RLVR algorithm comparisons (`DAPO`, `Dr.GRPO + CISPO`, `GSPO + LUFFY`), and `Axion 1.0`-style `2 SFT + RL` data scaling.
- **[`reports/02_phase_v1_v2_slm_results.md`](reports/02_phase_v1_v2_slm_results.md)** — Full Phase V1 & V2 Search/Confirm/Holdout tables, 3-arm recursion and 4-way compiler-repair causal isolations, and statistical audit notes.

---

## Repository Layout & Verification

- `IMMUTABLE_MANIFEST.json`, `IMMUTABLE_MANIFEST_V2.json`: Cryptographic split, scorer, and rule manifests.
- `eval/`, `eval_v2/`: Out-of-process Lean 4.24 (`#print axioms`) and symbolic math verifiers + governance guards.
- `code/`, `code/v2/`: Training (`ReST^EM`, Checkpointed-Chunk `12K` RLVR), multi-GPU `vLLM` evaluation, and report compilers.
- `results/`, `results_v2/`: Verified hash-chained ledgers (`20` ledgers across V1+V2), `HOLDOUT_SEAL` records, and `66` Holdout SHA-256 receipts.

```bash
python -I -B eval/governed_loop_guard.py verify-manifest --manifest IMMUTABLE_MANIFEST.json --root .
python -I -B code/v2/governed_loop_guard_v3.py verify-manifest --manifest IMMUTABLE_MANIFEST_V2.json --root . --expected-root-sha256 f79f2fcf32262a0455ba85256ea2c4a8ab72b3eef94c76f0399190932e74ccf2
python -I -B code/v2/build_v2_report_bundle.py .
```
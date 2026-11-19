# `AILab_Governed_SLM_RSI` — Two-Plane Governed Recursive Self-Improvement for Small Language Models (`3.8B–9B`) on Formal Lean 4.24 & Competition Mathematics (Phases V1 & V2)

## 1. Executive Summary

Across **Phase V1** (`IMMUTABLE_MANIFEST.json`, `33` files, `6` ledgers, `28` Holdout receipts) and **Phase V2** (`IMMUTABLE_MANIFEST_V2.json`, `20` files, Conjunctive `v2 + v3.6` dual-ledger governance, `14` ledgers, `38` Holdout receipts), we studied whether **Small Language Models (`3.8B–9B`) can self-improve on challenging formal Lean 4.24 theorem proving and competition/olympiad mathematics with zero external teacher traces** under executable two-plane governance (`/governed-loop-optimization`) and frontier adversarial review (`/copilot-app-frontier-cowork` at `Max` effort: `GPT-6.2 Sol Max`, `GPT 6.2 Astra, Max`, and `Opus 5.5, Max`).

### Headline Verified Outcomes Across Both Phases (Single-Touch Protected Holdout)

1. **Phase V2 Multi-Architecture Olympiad & 2025 Contest Mathematics (`N=200` Decontaminated 4-Contest Holdout: `BeyondAIME` `100`, `CMIMC 2025` `40`, `BRUMO 2025` `30`, `HMMT Nov 2025` `30`, zero external teacher traces):**
   - **`Qwen3.5-9B` (`m3_qwen35_9b_dapo_chunked12k` — Checkpointed-Chunk `12k` `DAPO + LUFFY` RLVR):**
     - **Item-pooled (`N=200`):** **`38.44% -> 48.94%` `math_avg_at_8` (`+10.50 pp`, 95% CI `[+7.88, +13.19] pp`, 20k stratified sign-flip Monte Carlo `p = 5.0e-5` [`0/20,000` flips exceeded observed gain])**, `pass@8`: **`70.00% -> 74.00%`**, `stop_rate`: **`55.38% -> 80.88%` (`+25.50 pp`)**.
     - **Equal-stratum (`4 x 1/4`):** **`42.33% -> 53.14%` `math_avg_at_8` (`+10.80 pp`, 95% CI `[+7.83, +13.86] pp`, Monte Carlo `p = 5.0e-5`)**, improving every contest (`BeyondAIME +9.88 pp`, `BRUMO 2025 +13.75 pp`, `CMIMC 2025 +11.25 pp`, `HMMT Nov 2025 +8.33 pp`).
   - **`Qwen3.5-4B` (`m3_qwen35_4b_dapo_chunked12k`):**
     - **Item-pooled (`N=200`):** **`36.13% -> 43.31%` `math_avg_at_8` (`+7.19 pp`, 95% CI `[+4.81, +9.63] pp`, Monte Carlo `p = 5.0e-5`)**, `pass@8`: **`66.00% -> 70.50%`**, `stop_rate`: **`66.50% -> 88.00%` (`+21.50 pp`)**.
     - **Equal-stratum (`4 x 1/4`):** **`39.88% -> 46.99%` `math_avg_at_8` (`+7.11 pp`, 95% CI `[+4.29, +9.85] pp`, Monte Carlo `p = 5.0e-5`)**.
     - *Exploratory cross-scale contrast:* Self-improved `Qwen3.5-4B` (`A*`) outperforms untuned `2.25x`-larger `Qwen3.5-9B` (`B_0`) across all 4 contests (**`+4.88 pp` item-pooled, `p = 0.00075`; `+4.66 pp` equal-stratum, `p = 0.00215`**), while zero-shot `Qwen3.8-27B` (`57.75%` pooled / `62.19%` equal-stratum) remains higher.
   - **Architectural & Cold-Start Boundaries (`Qwen3-4B` Pure Dense & `Phi-4-mini-reasoning` `3.8B` LongRoPE):**
     - `Qwen3-4B` (`think16k`) improved `+1.07 pp` on Search and `+2.49 pp` on `Omni-MATH` Confirmation (`29.79% -> 32.28%`, `p = 0.00066`, passing the `2.00 pp` config gate) and was conservatively withheld from Holdout by the orchestrator's `3.00 pp` filter (`B_0 = 33.19%` pooled retained).
     - `Phi-4-mini-reasoning` (`3.8B` `<think>`) rolled back on all `3/3` Search attempts (`25.88%` `B_0` retained; Holdout `B_0 = 18.75%` pooled) despite fixing a LongRoPE `96`-dim `inv_freq` DDP broadcast deadlock and uncensoring pool generation (`8k -> 12k`), demonstrating that extreme `<think>` verbosity (`54–69%` truncation) resists weight-only self-play without length-curriculum distillation.

2. **Phase V1 & V2 Formal Lean `v4.24.0 + Mathlib` Theorem Proving (In-Domain Gains vs. Cross-Distribution Vocabulary Wall):**
   - **In-Domain Competition Formalization:** In Phase V1 Holdout (`MiniF2F-v2s test`, `N=244`), `Qwen3.5-4B` rose **`27.66% -> 33.15%` `lean_avg_at_8` (`+5.48 pp`, 95% CI `[+2.92, +8.09] pp`, `100%` `#print axioms` clean)**, surpassing untuned `Qwen3.5-9B` (`30.02%`). In Phase V2 `Lean-Workbook` Search (`N=120`), `Qwen3.5-9B` rose **`48.02% -> 55.52%` (`+7.50 pp`)** (surpassing `Qwen3.8-27B` `41.98%` and closing `72%` of the Search gap to specialist `Goedel-Prover-V2-8B` `58.44%`) and `Qwen3.5-4B` rose **`40.73% -> 49.58%` (`+8.85 pp`)**.
   - **Cross-Distribution Formal Shift (`Lean-Workbook -> ProofNet-Lean4`):** On undergraduate textbook `ProofNet-valid` (`N=164`, Rudin/Munkres/Dummit-Foote), Mathlib definition shift dropped even specialist `Goedel-Prover-V2-8B` to `8.38%`; `Qwen3.5-9B` (`3.35% -> 3.89%`, exact paired sign-test `p = 0.500`) and `Qwen3.5-4B` (`2.21% -> 2.90%`, `p = 0.363`) showed no statistically detectable gain, so Conjunctive governance rolled both back and left `ProofNet-test` (`N=167`) strictly untouched by `A*` (`B_0` retained).

3. **Controlled Causal Isolations (Phase V2):**
   - **3-Arm Recursion Isolation (`M_2^{frozen-gen}` vs `M_2^{rec-yield}` vs `M_2^{rec-full}`):** At matched trace count (`992`) and optimizer steps (`62`), `M_2^{rec-yield}` vs `M_2^{frozen-gen}` showed no statistically detectable difference (`+0.49 pp` Math, 95% CI `[-2.05, +2.93] pp`; `-0.73 pp` Lean, 95% CI `[-5.21, +3.65] pp`), showing Round-2 `ReST^EM` saturates quickly while Round-3 Checkpointed-Chunk `DAPO + LUFFY` online RLVR drives the decisive post-Round-1 leap (`+4.20 pp` on `4B`, `+3.42 pp` on `9B` Math).
   - **4-Way Lean Compiler-Repair Isolation (`direct8` vs `masked_retry_4+4` vs `shuffled_diag_4+4` vs `compiler_repair_4+4`):** `10/10/11/11` of `164` on `ProofNet-valid` (no detectable benefit over `direct8`) and exploratory `60.83%` (`direct8`) vs `57.50%` across all three draft-conditioned `4+4` repair arms on `v2_lean_search120`, diagnosing failed-draft anchoring on `4B` single-turn repair and cleanly withholding repair promotion to Holdout.

---

### Phase V1 Background & Motivation

Prior `AILab*` studies exposed two severe walls when post-training hybrid Gated-DeltaNet Small Language Models (`Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`) on hard formal theorem proving and competition mathematics:
- In **`AILab_SLM_Qwen35_v1`** (Competition Math), every `Qwen3.5-4B` post-training attempt regressed (`-6 pp` LoRA, `-56` to `-63 pp` Full SFT due to chat-template/EOS/merge mismatches), and hard contest benchmarks (`AIME 2026`, `HMMT Feb 2026`) suffered runaway `<think>` truncation (`24–33%+` unfinished traces).
- In **`AILab_SLM_Qwen35_v2`** (Formal Lean 4), untuned `Qwen3.5-9B` scored `1/100` (`direct`) and `7/100 pass@8` (`thinking`, with `42.5–79.4%` truncation censoring), untuned compiler-repair degraded solve rate (`6/96` vs `4/96`), and zero tuned `4B`/`9B` checkpoints reached evaluation.
- In **`AILab_Qwen27B_Governed_v2`**, `27B` math tuning hit benchmark ceiling saturation (`95.3%`) and underpowered Confirmation sample sizes (`N=200`).

In **`AILab_Governed_SLM_RSI_v1`**, we synthesized those failure lessons under **`/governed-loop-optimization` Archetype D (Two-Plane Hybrid Governed RSI)** coupled with **three rounds of independent Frontier Adversarial Review (`GPT-6.2 Sol Max`)** (`sources/frontier_design_review_r1.md`, `sources/frontier_pre_confirm_review_r2.md`, `sources/frontier_post_holdout_audit_r3.md`):

1. **Plane 1 — Pre-Freeze Decoding, Verifier & Split Calibration (`IMMUTABLE_MANIFEST.json`):**
   - Replaced runaway `<think>` decoding with calibrated channel-aligned `direct` protocols (`enable_thinking=False`, `2,048` tokens on Lean 4.24; `direct12k`, `12,288` tokens on Competition Math), cutting Search truncation from **`91.60% -> 0.78%`** (`4B` Lean `avg@8`: **`3.81% -> 33.79%`**) and **`56.25% -> 13.87%`** (`4B` Math `avg@8`: **`43.55% -> 76.27%`**).
   - Caught and eliminated an upstream `MiniF2F` IMO `abbrev ... := True` vacuity hazard before freeze, locking pure non-vacuous `MiniF2F-v2s` (`N=243` Confirm / `N=244` Holdout) under Lean `v4.24.0 + Mathlib f897ebcf` with mandatory `#print axioms` verification (`100%` subset of `{propext, Classical.choice, Quot.sound}`).
   - Sealed **`IMMUTABLE_MANIFEST.json`** (`33` protected entries across `program.md`, `eval/`, `configs/`, `data/splits/`; `manifest_root_sha256 = 82a61cdf6b91a58fb537dc83f8bbbb0ea2ad46928a5a2174683db543a4ccf468`).

2. **Plane 2 — Multi-Round Verifier-Gated Weight-Space RSI (`15` Hash-Chained Search Trials):**
   - **Hybrid Gated-DeltaNet LoRA > Full-FT (`L1a` vs `L1b` controlled ablation):** Full-architecture LoRA (`r=32`, targeting all `248` linear projections across `self_attn`, `linear_attn`, and `mlp`, merged in `FP32` with `100.0%` top-1 parity and reset fresh to `M_0` each `ReST^EM` round) outperformed Full-FT on `Qwen3.5-4B` (`49.12%` vs `46.78%` `lean_avg_at_8`, `78.91%` vs `74.22%` `pass@8`).
   - **Lean Compiler-Diagnostic Repair Synthesis (`L2`) & `DAPO + LUFFY` Verifier-RLVR (`L3`/`M3`):** On `Qwen3.5-4B` Lean, Round-2 frontier self-play + compiler-repair synthesis (`1,703` verified traces across `1,457` unique theorems) pushed Search `lean_avg_at_8` from **`33.79% -> 49.12% -> 53.91%`** (`+20.12 pp`). On `Qwen3.5-9B` Lean, `DAPO + LUFFY` online RLVR broke past `ReST^EM` saturation (**`41.99% -> 53.22% -> 56.74%`** `lean_avg_at_8`, **`85.16% pass@8`**). On Competition Math, `DAPO + LUFFY` with soft-overlong length penalties pushed Search `math_avg_at_8` from **`76.27% -> 80.08%`** (`4B`) and **`76.76% -> 80.86%`** (`9B`) while automatically rolling back an off-distribution `27B`-mixed `M2` collapse (`stop_rate = 81.84% < 85%`).

3. **Confirmation Firewall (`Holm-Bonferroni K=2`) & Single-Touch Protected Holdout (`28` Receipts):**
   - Both **`Qwen3.5-4B`** champions cleared Holm-Bonferroni (`K=2`) Confirmation (`+3.09 pp`, `p = 9.16e-4` on `lean_confirm243`; `+3.92 pp`, `p = 2.72e-5` on `math_confirm300` with `95.50% -> 95.50%` `MATH-500` retention) and generalized cleanly to untouched **Protected Holdout**:
     - **Track L Holdout (`MiniF2F-v2s test`, `N=244`):** `Qwen3.5-4B` (`l2_q4b_restem_repair`) improved `lean_avg_at_8` from **`27.66% -> 33.15%` (`+5.48 pp`, paired 95% CI `[+2.92, +8.09] pp`)** and `pass@8` from **`43.44% -> 45.90%` (`106 -> 112/244` solved, `100%` `#print axioms` clean)** — outperforming untuned `Qwen3.5-9B` on `avg@8` (`30.02%`, `+3.13 pp`, nominal 95% CI `[+0.31, +5.99] pp`; tied at `112/244` `pass@8`) and numerically edging untuned `Qwen3.8-27B` on `avg@8` (`32.27%`, `+0.87 pp`, 95% CI `[-2.41, +4.25] pp`), while `27B` retained the `pass@8` lead (`48.77%`, `119/244`). All four arms (`4B B_0`, `4B A*`, `9B B_0`, `27B`) scored **`0/128`** across `k=8` whole-proof attempts on `PutnamBench`, marking a clear architectural boundary for single-turn `2,048`-token whole-proof generation without tactic tree search.
     - **Track M Holdout (`N=213` Hard Contest Suite + `N=150` `MATH-500` Retention):** `Qwen3.5-4B` (`m3_q4b_dapo_luffy`) improved pooled `math_avg_at_8` across `AIME 2025`, `AIME 2026`, `HMMT Feb 2026`, and `OlympiadBench` from **`55.52% -> 59.45%` (`+3.93 pp`, paired 95% CI `[+1.47, +6.40] pp`)** and 4-benchmark macro `avg@8` from **`46.95% -> 50.82%` (`+3.88 pp`)**, highlighted by **`AIME 2025`** (`37.08% -> 46.25% avg@8`, **`70.00% -> 86.67% pass@8` [`21 -> 26/30` solved]**) and **`OlympiadBench`** (`67.71% -> 71.77% avg@8`, `+4.06 pp`), with **no `MATH-500` loss** (`94.83% -> 95.83% avg@4`, 95% CI `[-0.67, +2.83] pp`).
   - Both **`Qwen3.5-9B`** challengers improved on Confirmation (`+1.59 pp`, `p = 0.063` on Lean; `+3.50 pp`, `z = 3.84`, `p = 6.11e-5` on Math with `stop_rate` improving `78.00% -> 83.92%`), but were disciplinedly held back by the frozen Confirmation gate (Lean missed `alpha = 0.025` / `+3.0 pp`; Math missed the absolute `85%` `stop_rate` floor by `1.08 pp`) and were **never evaluated on Protected Holdout**.

---

## 2. Two-Plane Governed RSI Architecture

```mermaid
flowchart TD
    P1["Plane 1: Pre-Freeze Calibration<br/>Channel/Budget Audit + Out-of-Process Verifiers + Non-Vacuous Splits"] --> FRZ["Freeze Trust Root<br/>IMMUTABLE_MANIFEST.json (33 files, SHA-256 pinned)"]
    FRZ --> S1["Plane 2 Round 1: Verified ReST^EM from M_0<br/>Hybrid Gated-DeltaNet LoRA (248 linear modules, FP32 merge)"]
    S1 --> S2["Plane 2 Round 2: Frontier Self-Play + Repair / 27B 0-Trace Assist<br/>Reset to M_0 + Governed Ledger Gate (Auto-Rollback on Guardrail Breach)"]
    S2 --> S3["Plane 2 Round 3: DAPO + LUFFY Online Verifier-RLVR<br/>Asymmetric Clip [0.20, 0.28] + Soft-Overlong Penalty + 1-of-8 Off-Policy Rescue"]
    S3 --> CONF["Phase-1 Confirmation Firewall (Holm-Bonferroni K=2)<br/>lean_confirm243 (N=243) & math_confirm300 (N=300) + math500_confirm150"]
    CONF -->|CONFIRMED_PROMOTE (4B Lean & 4B Math)| HOLD["Phase-2 Single-Touch Protected Holdout (28 SHA-256 Receipts)<br/>MiniF2F-test (244), PutnamBench (128), AIME25 (30), AIME26 (30), HMMT26 (33), OlympiadBench (120), MATH-500 (150)"]
    CONF -->|HELD BACK (9B Lean & 9B Math)| STOP["Excluded from Holdout<br/>Zero Post-Hoc Threshold Relaxation"]
```

---

## 3. Plane-1 Decoding & Verifier Calibration (`lean_search128` & `math_search128`)

Before touching any model weights, Plane-1 calibration (`results/plane1/plane1_calibration_report.json`) tested decoding mode (`thinking` vs `direct`), token caps, and presence penalties across `N=128`, `k=8`:

| Track & Model | Uncalibrated `thinking` (`avg@8` / Truncation) | Calibrated `direct` Protocol | Calibrated `B_0` `avg@8` | Calibrated `B_0` `pass@8` | Calibrated Truncation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Track L (`lean_search128`) — `Qwen3.5-4B`** | `3.81%` (`91.60%` trunc @ `8K`) | `direct` (`2,048` tok, `T=0.7`) | **`33.79%`** | **`60.16%`** (`77/128`) | **`0.78%`** |
| **Track L (`lean_search128`) — `Qwen3.5-9B`** | `23.05%` (`69.43%` trunc @ `8K`) | `direct` (`2,048` tok, `T=0.7`) | **`41.99%`** | **`72.66%`** (`93/128`) | **`0.20%`** |
| **Track L (`lean_search128`) — `Qwen3.8-27B` (Ref)** | — | `direct` (`2,048` tok, `T=0.7`) | **`40.53%`** | **`76.56%`** (`98/128`) | **`7.81%`** |
| **Track M (`math_search128`) — `Qwen3.5-4B`** | `43.55%` (`56.25%` trunc @ `16K`, `pp=0`) | `direct12k` (`12,288` tok, `T=0.7`) | **`76.27%`** | **`91.41%`** (`117/128`) | **`13.87%`** |
| **Track M (`math_search128`) — `Qwen3.5-9B`** | `54.69%` (`44.34%` trunc @ `16K`, `pp=1`) | `direct12k` (`12,288` tok, `T=0.7`) | **`76.76%`** | **`90.62%`** (`116/128`) | **`14.16%`** |
| **Track M (`math_search128`) — `Qwen3.8-27B` (Ref)** | — | `direct12k` (`12,288` tok, `T=0.7`) | **`87.01%`** | **`93.75%`** (`120/128`) | **`0.20%`** |

*Note:* Comparing `think` vs `direct` above reflects full operational decoding configurations (`enable_thinking`, stop tokens `[248044, 248046]`, and token cap) rather than a budget-matched `<think>` toggle alone. Both `eval/lean4_verifier.py` and `eval/math_verifier.py` passed `8/8` adversarial unit tests and an `80`-sample manual audit with `0` false positives (`results/plane1/verifier_40sample_audit_summary.json`).

---

## 4. Plane-2 Hash-Chained Weight-Space RSI Trajectory (`Search`)

Every Plane-2 candidate was gated against the active incumbent by `eval/governed_loop_guard.py` (`+1.0 pp` minimum improvement + non-regression guardrails) and recorded in cryptographically hash-chained JSONL ledgers (`results/plane2/`):

### 4.1 Track L — Formal Lean 4.24 (`lean_search128`, `9` Verified Ledger Entries)

| Trial ID | Model | Method (Reset / Init Policy) | `lean_avg_at_8` | `pass@8` (`Solved/128`) | Stop Rate | Axiom Clean | Ledger Decision |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `B_0 (4B)` | `4B` | Frozen calibrated `direct` baseline | `33.79%` | `60.16%` (`77`) | `99.22%` | `1.000` | Baseline |
| `l1a_q4b_hybrid_lora` | `4B` | Round-1 `ReST^EM` Hybrid LoRA (`r=32`, `248` linear proj, from `M_0`) | **`49.12%`** (`+15.33 pp`) | `78.91%` (`101`) | `99.02%` | `1.000` | `PROVISIONAL_KEEP` |
| `l1b_q4b_full_ft_lm` | `4B` | Controlled Full-FT ablation on identical `L1` bank (from `M_0`) | `46.78%` (`-2.34 pp` vs `l1a`) | `74.22%` (`95`) | `98.83%` | `1.000` | `ROLLBACK` |
| **`l2_q4b_restem_repair`** | **`4B`** | **Round-2 `ReST^EM` (`1,703` direct + compiler-repair traces, fresh from `M_0`)** | **`53.91%`** (`+20.12 pp` vs `B_0`) | **`79.69%` (`102`)** | **`99.02%`** | **`1.000`** | **`PROVISIONAL_KEEP` (`4B` Winner)** |
| `l3_q4b_dapo_luffy` | `4B` | Round-3 `DAPO + LUFFY` online Lean-verifier RLVR (from `l2_4b`) | `51.95%` (`-1.95 pp` vs `l2`) | `78.12%` (`100`) | `100.00%` | `1.000` | `ROLLBACK` |
| `B_0 (9B)` | `9B` | Frozen calibrated `direct` baseline | `41.99%` | `72.66%` (`93`) | `99.80%` | `1.000` | Baseline |
| `l1_q9b_restem` | `9B` | Round-1 `ReST^EM` Hybrid LoRA (`r=32`, from `M_0`) | **`53.22%`** (`+11.23 pp`) | `81.25%` (`104`) | `99.02%` | `1.000` | `PROVISIONAL_KEEP` |
| `l2_q9b_restem_repair` | `9B` | Round-2 `ReST^EM` (`1,703` cumulative traces, fresh from `M_0`) | `53.71%` (`+0.49 pp < +1.0 pp`) | `81.25%` (`104`) | `99.32%` | `1.000` | `ROLLBACK` |
| **`l3_q9b_dapo_luffy`** | **`9B`** | **Round-3 `DAPO + LUFFY` online Lean-verifier RLVR (from `l1_9b`)** | **`56.74%`** (`+14.75 pp` vs `B_0`) | **`85.16%` (`109`)** | **`99.61%`** | **`1.000`** | **`PROVISIONAL_KEEP` (`9B` Winner)** |

- **Plane-2B Call-Matched Outer Compiler-Repair (`lean_repair_reserve97`, `N=97`):** Under a strict `<= 8` verifier/LLM call cap (`4 direct + up to 4 compiler-diagnostic repairs` vs `8 direct`), `l3_q9b_dapo_luffy` matched `direct8` at **`83.51%` (`81/97`)** while using **`40.7%` fewer LLM calls (`4.74` vs `8.00` calls/theorem)** (`l4_q9b`: `ROLLBACK` under strict superiority gate; `direct8` retained for fixed `k=8` Confirmation/Holdout).

### 4.2 Track M — Hard Competition Math (`math_search128`, `6` Verified Ledger Entries)

| Trial ID | Model | Method (Reset / Init Policy) | `math_avg_at_8` | `maj@8` | `pass@8` (`Solved/128`) | Stop Rate | Parse Fail | Ledger Decision |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `B_0 (4B)` | `4B` | Frozen calibrated `direct12k` baseline | `76.27%` | `89.06%` | `91.41%` (`117`) | `86.72%` | `0.00%` | Baseline |
| `m1_q4b_restem` | `4B` | Round-1 Self-Verified `ReST^EM` Hybrid LoRA (from `M_0`) | `77.34%` (`+1.07 pp`) | `89.06%` | `90.62%` (`116`) | `90.92%` | `0.00%` | `PROVISIONAL_KEEP` |
| `m2_q4b_frontier_restem` | `4B` | Round-2 Self + `27B` 0-trace `ReST^EM` (`+94` `27B` traces, from `M_0`) | `77.15%` (`-0.20 pp` vs `m1`) | `88.28%` | `92.19%` (`118`) | `88.67%` | `0.00%` | `ROLLBACK` |
| **`m3_q4b_dapo_luffy`** | **`4B`** | **Round-3 `DAPO + LUFFY` Soft-Overlong RLVR (from `m1_4b`)** | **`80.08%`** (`+3.81 pp` vs `B_0`) | **`89.84%`** | **`92.97%` (`119`)** | **`93.95%`** | **`0.00%`** | **`PROVISIONAL_KEEP` (`4B` Winner)** |
| `B_0 (9B)` | `9B` | Frozen calibrated `direct12k` baseline | `76.76%` | `89.06%` | `90.62%` (`116`) | `85.84%` | `0.11%` | Baseline |
| `m1_q9b_restem` | `9B` | Round-1 Self-Verified `ReST^EM` Hybrid LoRA (from `M_0`) | `79.39%` (`+2.64 pp`) | `90.62%` | `91.41%` (`117`) | `86.72%` | `0.00%` | `PROVISIONAL_KEEP` |
| `m2_q9b_frontier_restem` | `9B` | Round-2 Self + `27B` 0-trace `ReST^EM` (`2,688` tok cap, from `M_0`) | `76.27%` (`-3.12 pp` vs `m1`) | `88.28%` | `91.41%` (`117`) | **`81.84%`** (`<85%`) | `0.00%` | **`ROLLBACK` (Guardrail Caught!)** |
| **`m3_q9b_dapo_luffy`** | **`9B`** | **Round-3 `DAPO + LUFFY` Soft-Overlong RLVR (from `m1_9b`)** | **`80.86%`** (`+4.10 pp` vs `B_0`) | **`89.06%`** | **`92.19%` (`118`)** | **`88.18%`** | **`0.00%`** | **`PROVISIONAL_KEEP` (`9B` Winner)** |

---

## 5. Phase-1 Confirmation Firewall (`Holm-Bonferroni K=2` Per Track)

All training banks were verified to have **zero ID and zero normalized-text SHA-256 overlap** with all `1,528` Confirmation and Protected Holdout items (`results/confirm/confirm_summary.json`). Ground-truth fields were stripped before worker evaluation (`code/vllm_eval_engine.py`).

| Track & Split | Challenger vs `B_0` | Primary `avg@8` (`B_0 -> A*`) | Paired Delta (`SE`) | Raw One-Sided `p` | Guardrails (`stop_rate` / `MATH-500`) | Confirmation Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track L (`lean_confirm243`, `N=243`)** | **`Qwen3.5-4B` (`l2_q4b_restem_repair`)** | **`21.71% -> 24.79%`** (`pass@8`: `30.45% -> 35.80%`) | **`+3.09 pp`** (`0.99 pp`) | **`9.16e-4`** (`< 0.0125`) | `stop_rate = 94.44%`, `100%` axiom-clean | **`CONFIRMED_PROMOTE`** |
| **Track L (`lean_confirm243`, `N=243`)** | `Qwen3.5-9B` (`l3_q9b_dapo_luffy`) | `25.46% -> 27.06%` (`pass@8`: `36.21% -> 37.86%`) | `+1.59 pp` (`1.04 pp`) | `0.0631` (`>= 0.025`) | `stop_rate = 97.63%`, `100%` axiom-clean | **`ROLLBACK` (Missed `alpha` & `+3 pp`)** |
| **Track M (`math_confirm300`, `N=300`)** | **`Qwen3.5-4B` (`m3_q4b_dapo_luffy`)** | **`69.63% -> 73.54%`** (`maj@8`: `83.67% -> 85.67%`) | **`+3.92 pp`** (`0.96 pp`) | **`2.72e-5`** (`< 0.0125`) | `stop_rate = 91.00%`, `MATH-500`: `573 -> 573/600` (`95.50%`) | **`CONFIRMED_PROMOTE`** |
| **Track M (`math_confirm300`, `N=300`)** | `Qwen3.5-9B` (`m3_q9b_dapo_luffy`) | `70.25% -> 73.75%` (`maj@8`: `83.33% -> 86.67%`) | `+3.50 pp` (`0.91 pp`) | **`6.11e-5`** (`z = +3.84`) | **`stop_rate = 83.92% < 85%`** (`B_0 = 78.00%`), `MATH-500`: `585 -> 590/600` | **`ROLLBACK` (Guardrail Floor Only)** |

*Comparator reference on Confirmation:* Untuned `Qwen3.8-27B` scored `25.36% avg@8` (`44.03% pass@8`) on `lean_confirm243` and `82.71% avg@8` (`88.67% maj@8`, `93.67% pass@8`) on `math_confirm300`.  
*Governance note on `Qwen3.5-9B` Math:* `m3_q9b_dapo_luffy` cleared statistical significance (`p = 6.11e-5`, paired-SE; `p = 7.5e-5`, ledger effective-SE) and raised `stop_rate` by `+5.92 pp` over `9B B_0` (`78.00% -> 83.92%`), but fell short of the frozen absolute `stop_rate >= 85%` gate (`88%` in prose `program.md`). Under strict governance, the threshold was **not** relaxed post-hoc, and neither `9B` challenger was authorized for Protected Holdout.

---

## 6. Phase-2 Single-Touch Protected Holdout Results (`results/holdout/holdout_final_summary.json`)

Protected Holdout was touched **exactly once** (`28` cryptographic `*_receipt.json` records verified, `manifest_root_sha256` verified before and after execution). Reported 95% confidence intervals below are pointwise problem-paired cluster-bootstrap intervals (`B = 10,000`).

### 6.1 Track L — Formal Lean 4.24 Protected Holdout (`k = 8` Whole-Proof Attempts/Problem)

| Benchmark (`N`) | `Qwen3.5-4B B_0` | **`Qwen3.5-4B A*` (`l2_q4b_restem_repair`)** | Paired Delta vs `4B B_0` (95% CI) | Untuned `Qwen3.5-9B B_0` | Untuned `Qwen3.8-27B` (Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`MiniF2F-v2s test` (`N=244`) — `lean_avg_at_8`** | `27.66%` | **`33.15%`** | **`+5.48 pp` (`[+2.92, +8.09]`)** | `30.02%` (`4B A*` `+3.13 pp`, CI `[+0.31, +5.99]`) | `32.27%` (`4B A*` `+0.87 pp`, CI `[-2.41, +4.25]`) |
| **`MiniF2F-v2s test` (`N=244`) — `lean_pass_at_8`** | `43.44%` (`106/244`) | **`45.90%` (`112/244`)** | **`+2.46 pp` (`+6` theorems)** | `45.90%` (`112/244`) | **`48.77%` (`119/244`)** |
| **`MiniF2F-v2s test` — `#print axioms` Clean Rate** | `100.0%` | **`100.0%`** | `0` unsound / `0` `sorry` | `100.0%` | `100.0%` |
| **`PutnamBench` (`N=128`) — `lean_avg_at_8` / `pass@8`** | `0.00%` (`0/128`) | `0.00%` (`0/128`) | `0.00 pp` (`[0.00, 0.00]`) | `0.00%` (`0/128`) | `0.00%` (`0/128`) |

- **Interpretation & Scope:** On `MiniF2F-v2s test`, `l2_q4b_restem_repair` (`4B`) substantially outperforms its own calibrated baseline (`+5.48 pp`) and surpasses untuned `Qwen3.5-9B` (`2.25x` larger) on `avg@8` (`+3.13 pp`, tying at `112/244` `pass@8`). Against `Qwen3.8-27B` (`6.75x` larger), `4B A*`'s `avg@8` point estimate (`33.15%` vs `32.27%`, CI `[-2.41, +4.25] pp`) shows neither statistical superiority nor formal TOST equivalence, and `27B` retains higher `pass@8` (`119/244` vs `112/244`). On `PutnamBench`, `0/128` across all four arms (including `27B`) with `k=8` whole-proof attempts (`2,048` tok) demonstrates that college Putnam formalization requires multi-turn interactive tactic tree search / lemma decomposition beyond single-turn whole-proof generation.

### 6.2 Track M — Hard Competition & Olympiad Mathematics Protected Holdout (`direct12k`, `k = 8`)

| Benchmark (`N`) | Metric | `Qwen3.5-4B B_0` | **`Qwen3.5-4B A*` (`m3_q4b_dapo_luffy`)** | Paired `avg` Delta vs `4B B_0` (95% CI) | Untuned `Qwen3.5-9B B_0` | Untuned `Qwen3.8-27B` (Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled Hard Contest Suite (`N=213`)** | **`math_avg_at_8`** | `55.52%` | **`59.45%`** | **`+3.93 pp` (`[+1.47, +6.40]`)** | `55.40%` (`4B A*` `+4.05 pp`, CI `[+1.12, +7.04]`) | **`74.65%`** |
| **4-Benchmark Macro Average** | **`math_avg_at_8`** | `46.95%` | **`50.82%`** | **`+3.88 pp`** | `48.77%` (`4B A*` `+2.05 pp`) | **`70.27%`** |
| **`AIME 2025` (`N=30`)** | `avg@8` / `maj@8` / `pass@8` | `37.08%` / `56.67%` / `70.00%` (`21/30`) | **`46.25%` / `63.33%` / `86.67%` (`26/30`)** | **`+9.17 pp` (`[+3.75, +14.58]`)** | `46.25%` / `70.00%` / `76.67%` (`23/30`) | `65.00%` / `86.67%` / `90.00%` (`27/30`) |
| **`AIME 2026` (`N=30`)** | `avg@8` / `maj@8` / `pass@8` | `50.42%` / `73.33%` / `76.67%` (`23/30`) | **`50.42%` / `73.33%` / `83.33%` (`25/30`)** | `0.00 pp` (`[-8.75, +8.33]`, `+6.67 pp pass@8`) | `54.17%` / `80.00%` / `86.67%` (`26/30`) | `80.00%` / `93.33%` / `93.33%` (`28/30`) |
| **`HMMT Feb 2026` (`N=33`)** | `avg@8` / `maj@8` / `pass@8` | `32.58%` / `48.48%` / `54.55%` (`18/33`) | **`34.85%` / `45.45%` / `57.58%` (`19/33`)** | `+2.27 pp` (`[-4.55, +9.47]`, `-3.03 pp maj@8`) | `29.55%` / `60.61%` / `63.64%` (`21/33`) | `54.92%` / `60.61%` / `78.79%` (`26/33`) |
| **`OlympiadBench` (`N=120`)** | `avg@8` / `maj@8` / `pass@8` | `67.71%` / `79.17%` / `85.00%` (`102/120`) | **`71.77%` / `80.00%` / `85.83%` (`103/120`)** | **`+4.06 pp` (`[+1.25, +7.08]`)** | `65.10%` / `84.17%` / `88.33%` (`106/120`) | `81.15%` / `85.83%` / `94.17%` (`113/120`) |
| **`MATH-500` Retention (`N=150`, `k=4`)** | `avg@4` / `maj@4` / `pass@4` | `94.83%` / `96.00%` / `98.00%` (`147/150`) | **`95.83%` / `97.33%` / `98.00%` (`147/150`)** | `+1.00 pp` (`[-0.67, +2.83]`, no loss) | `97.17%` / `98.67%` / `99.33%` (`149/150`) | `98.17%` / `100.0%` / `100.0%` (`150/150`) |

- **Completion / Truncation Reduction Across All 4 Contest Benchmarks (`4B B_0 -> 4B A*`):**
  - `AIME 2025` `stop_rate`: **`62.50% -> 80.83%` (`+18.33 pp`)** (residual truncation `19.17%`)
  - `AIME 2026` `stop_rate`: **`67.92% -> 85.42%` (`+17.50 pp`)** (residual truncation `14.58%`)
  - `HMMT Feb 2026` `stop_rate`: **`60.98% -> 76.14%` (`+15.15 pp`)** (residual truncation `23.86%`)
  - `OlympiadBench` `stop_rate`: **`80.10% -> 89.38%` (`+9.27 pp`)** (residual truncation `10.62%`)
  - `natural_stop_parse_failure_rate`: **`0.00%`** across every `4B A*` benchmark.
- **Slice Nuance:** `OlympiadBench` contributes `120 / 213` problems to the pooled hard-contest holdout (`+6.67 pp` over `9B B_0`); on combined `AIME 2025 + 2026` (`N=60`), `4B A*` improved `+4.58 pp` `avg@8` (`43.75% -> 48.33%`, one-sided 95% LCB `+0.21 pp`) and `+11.67 pp` `pass@8` (`73.33% -> 85.00%`, `44 -> 51/60`) over `4B B_0`, trailing `9B B_0` slightly on `avg@8` (`48.33%` vs `50.21%`) while solving two more AIME problems under `pass@8` (`51/60` vs `49/60`). `Qwen3.8-27B` (`74.65%` pooled) remains substantially stronger on competition math.

---

## 7. Key Scientific & Methodological Takeaways

1. **Decoding Channel & Token-Budget Alignment Must Precede Weight Updates (Plane 1 before Plane 2):** On `Qwen3.5-4B`/`9B`, `<think>` verbosity consumed `4K–16K` budgets before reaching `\boxed{}` or valid Lean tactics. Calibrating `direct` / `direct12k` with merged stop IDs `[248044, 248046]` unlocked `+29.98 pp` (Lean) and `+32.72 pp` (Math) *before* any gradient step, and aligning SFT/RLVR targets to that exact `direct` channel eliminated the `-56 pp` post-training collapse seen in `AILab_SLM_Qwen35_v1`.
2. **Full-Projection Hybrid Gated-DeltaNet LoRA Outperforms Full-FT on Small Verified Proof Banks:** Adapting all `248` linear projections (`self_attn`, `linear_attn` Gated-DeltaNet `in_proj_qkv/z/a/b`/`out_proj`, and `mlp`) with `r=32, alpha=64` and `FP32` exact merge (`100%` top-1 token agreement) beat Full-FT by `+2.34 pp` `avg@8` and `+4.69 pp` `pass@8` on Lean 4.24 while resetting cleanly to `M_0` each `ReST^EM` round.
3. **Why Off-Policy Teacher Mixing (`M2`) Regressed While `DAPO + LUFFY` (`M3`/`L3`) Succeeded:** Mixing verbose `27B` derivations directly into `ReST^EM` SFT (`M2`) shifted generation length upward and tripped the `85%` stop-rate guardrail on `9B` (`81.84%` -> automatic `ROLLBACK`). By contrast, `DAPO + LUFFY` (`M3` and `L3`) used off-policy traces *only* as `1-of-8` group rescue when all `8` on-policy rollouts failed (`0/8`), combined with soft-overlong length penalties — simultaneously raising `stop_rate` (`+5.9` to `+18.3 pp`) and accuracy (`+3.8` to `+4.1 pp`).
4. **Dual-Guardrail Preregistration Lesson for Governed Loops:** On `math_confirm300`, `m3_q9b_dapo_luffy` achieved `+3.50 pp` (`p = 6.11e-5`) and improved `stop_rate` by `+5.92 pp` (`78.00% -> 83.92%`), yet was blocked by a static `85%` floor calibrated on an easier Search split (`85.84%` baseline). Future governed RSI manifests should preregister a **two-part completion guardrail**: (a) a baseline-relative non-inferiority margin (`stop_rate >= stop_rate(B_0) - delta`) plus (b) a split-calibrated absolute operational floor.

---

## 8. Repository Structure & Verified Artifacts

- `program.md` & `IMMUTABLE_MANIFEST.json`: Frozen Archetype D specification and `33`-file cryptographic manifest (`manifest_root_sha256 = 82a61cdf6b91a58fb537dc83f8bbbb0ea2ad46928a5a2174683db543a4ccf468`).
- `eval/`: Frozen out-of-process `lean4_verifier.py` (`Lean v4.24.0 + Mathlib f897ebcf` + `#print axioms`), `math_verifier.py`, and `governed_loop_guard.py`.
- `configs/` & `data/splits/`: Frozen gate rules (`Search` & `Confirmation`) and decontaminated split manifests.
- `code/`: Mutable training, `DAPO + LUFFY` RLVR, `vLLM` evaluation (`tp>1` `disable_custom_all_reduce` + prompt-only label stripping), Confirmation/Holdout runner, and final report bundle compiler.
- `results/`:
  - `plane1/`: `plane1_calibration_report.json`, `preconditions_lean.json`, `preconditions_math.json`, `verifier_40sample_audit_summary.json`.
  - `plane2/`: Verified hash-chained ledgers (`track_l_ledger.jsonl`, `track_m_4b_ledger.jsonl`, `track_m_9b_ledger.jsonl`).
  - `confirm/`: `confirm_summary.json` and per-track Confirmation ledgers (`track_l_confirm_ledger.jsonl`, `track_m_4b_confirm_ledger.jsonl`, `track_m_9b_confirm_ledger.jsonl`).
  - `holdout/`: `holdout_final_summary.json` (all `28` single-touch receipts, metrics, and `10,000`-resample paired bootstrap CIs).
- `sources/`: Three rounds of independent `GPT-6.2 Sol Max` frontier adversarial design, pre-Confirmation, and post-Holdout claim audits (Phase V1) plus five-section M365 Copilot `Cowork` (`GPT 6.2 Astra, Max` & `Opus 5.5, Max`) + post-Holdout adversarial review dossier (`sources/cowork_v2_frontier_design_review.md`).

---

## 9. Phase V2 — Multi-Architecture & Multi-Benchmark Expansion (`IMMUTABLE_MANIFEST_V2.json` + `AMENDMENT_V3_6.json`)

Phase V2 kept Phase V1 `100%` sealed and expanded governed zero-teacher self-improvement across **4 Small Language Models (`3.8B–9B`, 3 distinct Transformer/SSM families) + 2 specialist/frontier references** on **completely disjoint** benchmarks (`0` statement/text SHA-256 overlap with Phase V1):
- **SLM Candidates (`0` external teacher traces):**
  1. `Qwen/Qwen3.5-4B` (`4B` Hybrid Gated-DeltaNet + Softmax, `248` linear projections, `r=32`)
  2. `Qwen/Qwen3-4B` (`4B` Size-Matched Pure Dense Transformer, `252` linear projections, `r=32`)
  3. `microsoft/Phi-4-mini-reasoning` (`3.8B` LongRoPE `<think>` family, `128` fused projections, parameter-matched `r=48`)
  4. `Qwen/Qwen3.5-9B` (`9B` Hybrid Gated-DeltaNet + Softmax, recovering V1's `2,688`-token OOM via Checkpointed-Chunk `12,288`-token RLVR)
- **Zero-Shot Reference Comparators:** `Goedel-LM/Goedel-Prover-V2-8B` (specialist Lean 4 prover) and `Qwen/Qwen3.8-27B`.
- **Disjoint Splits (`IMMUTABLE_MANIFEST_V2.json`, `20/20` entries `VERIFIED`, root SHA-256 `f79f2fcf32262a0455ba85256ea2c4a8ab72b3eef94c76f0399190932e74ccf2`):**
  - *Track L (Lean `v4.24.0 + Mathlib`):* `Lean-Workbook` Search (`N=120`) + `1,000`-theorem training pool -> **`ProofNet-Lean4 valid` (`N=164` unique elaborated theorems) Confirmation** -> **`ProofNet-Lean4 test` (`N=167` unique elaborated theorems) Protected Holdout**.
  - *Track M (Olympiad & 2025 Contest Math):* Disjoint Contest Search (`N=128`) + `360`-problem training pool -> **`Omni-MATH Tier 3–4` (`N=256`) Confirmation** -> **Decontaminated `2025` 4-Contest Protected Holdout (`N=200`: `BeyondAIME` `100`, `CMIMC 2025` `40`, `BRUMO 2025` `30`, `HMMT Nov 2025` `30`)**, plus reused V1 `MATH-500` (`N=150`) general-math retention check outside the 4-contest average.

---

## 10. Phase V2 Plane-1 Calibration & Plane-2 Weight-Space RSI (`8` Model x Track Pipelines)

| Track & Split | Model (`Architecture / Channel`) | Calibrated `B_0` `avg@8` (`pass@8`) | Round-1 `ReST^EM` | Round-2 Recursive `ReST^EM` | Round-3 Checkpointed-Chunk `12k` `DAPO + LUFFY` | Search Outcome (`Incumbent`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track L (`v2_lean_search120`)** | **`Qwen3.5-9B` (`Hybrid GDN`, `direct`)** | `48.02%` (`63.33%`) | `50.73%` | `53.13%` | **`55.52%` (`+7.50 pp`)** | **`3/3` `PROVISIONAL_KEEP`** |
| **Track L (`v2_lean_search120`)** | **`Qwen3.5-4B` (`Hybrid GDN`, `direct`)** | `40.73%` (`55.00%`) | `46.77%` | `48.54%` | **`49.58%` (`+8.85 pp`)** | **`3/3` `PROVISIONAL_KEEP`** |
| **Track L (`v2_lean_search120`)** | `Qwen3-4B` (`Pure Dense`, `think6k`) | `5.42%` (`13.33%`) | `5.83%` | `5.63%` | `5.10%` | `3/3` `ROLLBACK` (`B_0` kept) |
| **Track L (`v2_lean_search120`)** | `Phi-4-mini-reasoning` (`3.8B`, `think6k`) | `0.31%` (`2.50%`) | — | — | — | Pre-registered skip (`learnable = 2.5% < 10%`) |
| **Track L Comparators** | `Goedel-Prover-V2-8B` / `Qwen3.8-27B` | `58.44%` / `41.98%` | — | — | — | Zero-shot references |
| **Track M (`v2_math_search128`)** | **`Qwen3.5-9B` (`Hybrid GDN`, `direct12k`)** | `37.60%` (`54.69%`, `stop 56.5%`) | `42.29%` | `45.90%` | **`49.32%` (`+11.72 pp`, `stop 81.5%`)** | **`3/3` `PROVISIONAL_KEEP`** |
| **Track M (`v2_math_search128`)** | **`Qwen3.5-4B` (`Hybrid GDN`, `direct12k`)** | `36.72%` (`56.25%`, `stop 64.8%`) | `41.50%` | `42.29%` (`< +1 pp`) | **`45.70%` (`+8.98 pp`, `stop 88.5%`)** | **`m1` & `m3` `PROVISIONAL_KEEP`** |
| **Track M (`v2_math_search128`)** | **`Qwen3-4B` (`Pure Dense`, `think16k`)** | `35.94%` (`44.53%`, `stop 56.5%`) | `36.62%` | **`37.01%` (`+1.07 pp`)** | `36.43%` | **`m2` `PROVISIONAL_KEEP`** |
| **Track M (`v2_math_search128`)** | `Phi-4-mini-reasoning` (`3.8B`, `think12k`) | `25.88%` (`39.84%`, `stop 46.4%`) | `25.68%` | `25.29%` (`12k` uncensored) | `24.41%` (`stop` breach) | `3/3` `ROLLBACK` (`B_0` kept) |
| **Track M Comparator** | `Qwen3.8-27B` (`direct12k`) | `51.95%` (`65.62%`, `stop 99.2%`) | — | — | — | Zero-shot reference |

- **Systems & Memory Engineering Highlights:**
  1. **Checkpointed-Chunk `12,288`-Token Logits RLVR (`code/v2/train_dapo_chunked_v2.py`):** Applying `lm_head + log_softmax + gather` in checkpointed `1,024`-token completion chunks reduced `Qwen3.5-9B` `12k` RLVR peak VRAM from `39.7 GB` (OOM in V1) to `< 30 GB` on `80 GB` A100s, unlocking `+11.72 pp` on Search (`stop_rate`: `56.54% -> 81.45%`) and `+13.57 pp` on Confirmation!
  2. **`Phi-4-mini` LongRoPE `inv_freq` DDP Deadlock Fix:** `Phi3LongRoPEScaledRotaryEmbedding` dynamically swaps its `96`-dim `inv_freq` buffer across the `4,096`-token threshold (`short_factor` vs `long_factor`), causing `WorkNCCL(OpType=BROADCAST, NumelIn=96)` rank hangs when ranks hold mixed `< 4,096` and `> 4,096` microbatches. Setting `DDP(..., broadcast_buffers=False)` + pre-merge `dist.destroy_process_group()` eliminated the deadlock cleanly.

---

## 11. Phase V2 Controlled Causal Isolations (`Qwen3.5-4B`)

1. **3-Arm Recursive vs. Frozen-Generator Control (Round 2 from `M_0`, Matched Prompts & Optimizer Budget):**
   - **Track M (`v2_math_search128`):** `B_0 = 36.72%`, `M_1 = 41.50%` -> `M_2^{frozen-gen} = 41.89%`, `M_2^{rec-full} = 42.29%` (`+0.39 pp` vs `frozen-gen`, 95% CI `[-1.95, +2.73] pp`), `M_2^{rec-yield} = 42.38%` (`+0.49 pp` vs `frozen-gen`, 95% CI `[-2.05, +2.93] pp`, matched `992` traces & `62` steps).
   - **Track L (`v2_lean_search120`):** `B_0 = 40.73%`, `M_1 = 46.77%` -> `M_2^{frozen-gen} = 49.79%`, `M_2^{rec-yield} = 49.06%` (`-0.73 pp` vs `frozen-gen`, 95% CI `[-5.21, +3.65] pp`), `M_2^{rec-full} = 48.54%` (`-1.25 pp`, 95% CI `[-5.42, +2.71] pp`).
   - **Finding:** At matched trace counts and gradient steps, `M_2^{rec-yield}` and `M_2^{frozen-gen}` show **no statistically detectable difference**: recursive `M_1` expands pool frontier coverage (`+19` solved Math pool problems) and halves Lean proof length (`median_tokens`: `232.5 -> 122.5`), while off-policy Round-2 `ReST^EM` saturates quickly and **Round-3 Checkpointed-Chunk `DAPO + LUFFY` online RLVR drives the decisive post-R1 gain (`+4.20 pp` on `4B`, `+3.42 pp` on `9B` Math)**.

2. **4-Way Call-Matched (`<= 8` Calls) Lean Compiler-Diagnostic Repair Isolation:**
   - **OOD `ProofNet-valid` (`N=164`):** `direct8 = 6.10%` (`10/164`), `compiler_repair_4+4 = 6.10%` (`10/164`), `masked_retry_4+4 = 6.71%` (`11/164`), `shuffled_diag_4+4 = 6.71%` (`11/164`) -> **inconclusive / no detectable benefit over `direct8` at 8 attempts**, cleanly withholding 2-pass repair promotion to Holdout.
   - **Exploratory In-Domain `v2_lean_search120` (`N=120`):** `direct8 = 60.83%` (`73/120`, `8.00` calls) vs `57.50%` (`69/120`, `5.84` calls) across all three `4+4` repair variants (`compiler_repair_4+4`, `masked_retry_4+4`, `shuffled_diag_4+4`), showing that conditioning a `4B` model on its own failed proof draft induces **failed-draft anchoring** relative to `4` fresh independent samples.

---

## 12. Phase V2 Conjunctive `v2 + v3.6` Confirmation Firewall (`results_v2/confirm/v2_confirm_summary.json`)

Before any Protected Holdout access, `AMENDMENT_V3_6.json` (`85cd4407bc7a6c4e72ccf120edc40d17c9f5f1c7774eb5f1961e8f26ad1ee45b`) layered `governed_loop_guard_v3.py` (`v3.6`) as a strict **veto-only conjunction** over `IMMUTABLE_MANIFEST_V2.json`:

| Track & Split | Model | `B_0 -> A*` `avg@8` (`pass@8`) | Paired Delta (95% CI) | One-Sided `p` | `stop_rate` (`B_0 -> A*`) | Conjunctive Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track M (`Omni-MATH Tier 3–4`, `N=256`)** | **`Qwen3.5-9B`** | **`33.45% -> 47.02%`** (`58.98% -> 65.23%`) | **`+13.57 pp` (`[+10.99, +16.26]`)** | **`6.46e-24`** | **`51.42% -> 79.25%`** | **`PROMOTE` (`v2` & `v3.6`)** |
| **Track M (`Omni-MATH Tier 3–4`, `N=256`)** | **`Qwen3.5-4B`** | **`34.13% -> 43.41%`** (`58.20% -> 64.84%`) | **`+9.28 pp` (`[+6.74, +11.96]`)** | **`1.36e-12`** | **`62.35% -> 86.57%`** | **`PROMOTE` (`v2` & `v3.6`)** |
| **Track M (`Omni-MATH Tier 3–4`, `N=256`)** | `Qwen3-4B` (`think16k`) | `29.79% -> 32.28%` (`41.41% -> 42.97%`) | `+2.49 pp` (`[+1.03, +4.05]`) | `6.58e-4` | `46.53% -> 53.71%` | **Withheld (`< 3.00 pp` orchestrator bar; see §14)** |
| **Track M (`Omni-MATH Tier 3–4`, `N=256`)** | `Phi-4-mini` / `Qwen3.8-27B` | `20.80%` (`B_0`) / `51.66%` (`Ref`) | — | — | `40.97%` / `98.83%` | Untested (`0` Search promotions) / Ref |
| **Track L (`ProofNet-valid`, `N=164`)** | `Qwen3.5-9B` | `3.35% -> 3.89%` (`7.32% -> 9.76%`) | `+0.53 pp` (`[-0.53, +1.68]`) | `p_exact = 0.500` | `86.28% -> 99.16%` | **`ROLLBACK` (`B_0` retained)** |
| **Track L (`ProofNet-valid`, `N=164`)** | `Qwen3.5-4B` | `2.21% -> 2.90%` (`6.71% -> 7.93%`) | `+0.69 pp` (`[-0.53, +2.06]`) | `p_exact = 0.363` | `86.74% -> 99.54%` | **`ROLLBACK` (`B_0` retained)** |
| **Track L (`ProofNet-valid`, `N=164`)** | `Goedel-8B` / `Qwen3.8-27B` | `8.38%` / `3.05%` | — | — | `67.15%` / `32.70%` | Zero-shot references |

---

## 13. Phase V2 Single-Touch Protected Holdout (`38` Verified Receipts, `HOLDOUT_SEAL_*` `FINALIZED`)

Zero-shot comparators were evaluated and SHA-256 pinned (`COMPARATOR_PIN_track_m_{qwen35_4b,qwen35_9b}.json`) and `HOLDOUT_SEAL_track_m_{qwen35_4b,qwen35_9b}.json` were atomically reserved (`O_CREAT | O_EXCL` + `fsync`) **before** evaluating the two promoted `A*` models strictly once on Protected Holdout. Unpromoted challengers were **never evaluated on Protected Holdout**.

### 13.1 Track M Protected Holdout — Decontaminated `2025` 4-Contest Suite (`N=200`, `k=8`)

| Contest / Aggregation (`N`) | `Phi-4-mini B_0` | `Qwen3-4B B_0` | `Qwen3.5-4B B_0` | **`Qwen3.5-4B A*`** (`Delta vs 4B B_0`) | `Qwen3.5-9B B_0` | **`Qwen3.5-9B A*`** (`Delta vs 9B B_0`) | `Qwen3.8-27B` (Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BeyondAIME` (`N=100`)** | `14.38%` | `27.63%` | `29.50%` | **`37.75%` (`+8.25 pp`)** | `32.25%` | **`42.13%` (`+9.88 pp`)** | `50.75%` |
| **`CMIMC 2025` (`N=40`)** | `20.00%` | `30.00%` | `37.50%` | **`38.13%` (`+0.63 pp`)** | `35.00%` | **`46.25%` (`+11.25 pp`)** | `53.44%` |
| **`BRUMO 2025` (`N=30`)** | `33.75%` | `51.67%` | `53.75%` | **`61.67%` (`+7.92 pp`)** | `57.92%` | **`71.67%` (`+13.75 pp`)** | `77.50%` |
| **`HMMT Nov 2025` (`N=30`)** | `16.67%` | `37.50%` | `38.75%` | **`50.42%` (`+11.67 pp`)** | `44.17%` | **`52.50%` (`+8.33 pp`)** | `67.08%` |
| **Item-Pooled `avg@8` (`N=200`)** | `18.75%` | `33.19%` | `36.13%` | **`43.31%` (`+7.19 pp`, CI `[+4.81, +9.63]`)** | `38.44%` | **`48.94%` (`+10.50 pp`, CI `[+7.88, +13.19]`)** | `57.75%` |
| **Equal-Stratum `avg@8` (`4 x 1/4`)** | `21.20%` | `36.70%` | `39.88%` | **`46.99%` (`+7.11 pp`, CI `[+4.29, +9.85]`)** | `42.33%` | **`53.14%` (`+10.80 pp`, CI `[+7.83, +13.86]`)** | `62.19%` |
| **Item-Pooled `pass@8` (`N=200`)** | `33.00%` | `52.50%` | `66.00%` | **`70.50%` (`+4.50 pp`)** | `70.00%` | **`74.00%` (`+4.00 pp`)** | `82.00%` |
| **Item-Pooled `stop_rate` (`N=200`)** | `33.50%` | `44.88%` | `66.50%` | **`88.00%` (`+21.50 pp`)** | `55.38%` | **`80.88%` (`+25.50 pp`)** | `98.13%` |
| **20k Stratified Sign-Flip `p` (vs own `B_0`)** | — | — | — | **Monte Carlo `p = 5.0e-5` (`0/20,000`)** | — | **Monte Carlo `p = 5.0e-5` (`0/20,000`)** | — |
| **Reused V1 `MATH-500` Check (`N=150`, `mean@8`)** | — | — | `95.17%` | **`96.00%` (`+0.83 pp`, `100% pass@8`)** | `96.08%` | **`97.92%` (`+1.83 pp`, CI `[+0.42, +3.50]`)** | — |

- **Exploratory Cross-Scale Contrasts (Outside Pre-Registered `A* vs own B_0` Family):**
  - `Qwen3.5-4B A*` vs untuned `Qwen3.5-9B B_0`: **`+4.88 pp` item-pooled (`95%` CI `[+2.00, +7.75] pp`, stratified sign-flip `p = 0.00075`)** and **`+4.66 pp` equal-stratum (`95%` CI `[+1.50, +7.79] pp`, `p = 0.00215`)**, winning on all 4 contests (`+5.50`, `+3.13`, `+3.75`, `+6.25 pp`).
  - Against zero-shot `Qwen3.8-27B`, `Qwen3.5-9B A*` (`-8.81 pp` pooled, `-9.06 pp` equal-stratum) and `Qwen3.5-4B A*` (`-14.44 pp` pooled, `-15.20 pp` equal-stratum) close `54.4%` and `33.2%` of their respective `B_0 -> 27B` gaps while remaining below `27B`.

### 13.2 Track L Protected Holdout — `ProofNet-Lean4 test` (`N=167`, Zero-Shot Baselines Only; `A*` Untouched)

| Metric (`N=167`, `k=8`) | `Phi-4-mini B_0` | `Qwen3-4B B_0` | `Qwen3.5-4B B_0` | `Qwen3.5-9B B_0` | `Qwen3.8-27B` (Ref) | `Goedel-Prover-V2-8B` (Specialist Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lean_avg_at_8`** | `0.00%` | `0.07%` | `1.72%` | **`2.69%`** | `2.62%` | **`7.86%`** |
| **`lean_pass_at_8`** | `0.00%` | `0.60%` | `4.19%` | **`7.78%`** | `7.19%` | **`12.57%`** |

---

## 14. Protocol Transparency & Audit Reconciliations (`sources/cowork_v2_frontier_design_review.md` §5)

1. **`Qwen3-4B` Confirmation Threshold & Holm Family:** `configs_v2/gate_rules_v2_math_confirm.json` and `program_v2.md` specified `min_improvement = 2.00 pp` and one-sided family `alpha = 0.025`, whereas `code/v2/run_v2_confirm_and_holdout.py` enforced a stricter `3.00 pp` orchestrator filter (`min_eff = 0.030`, per-track `alpha = 0.05`). Under **both** family definitions (`alpha = 0.025` or `0.05`, per-track `K=3` or global `K=5`), `Qwen3.5-9B` (`p = 6.46e-24`) and `Qwen3.5-4B` (`p = 1.36e-12`) unequivocally pass. `Qwen3-4B` (`+2.49 pp`, `p = 6.58e-4`) passed both the `2.00 pp` config gate (`v2` & `v3.6` gate JSONs returned `PROMOTE`) and `alpha = 0.025` Holm-Bonferroni, and was conservatively withheld from Holdout solely by the orchestrator's `3.00 pp` filter (`B_0` retained).
2. **Monte Carlo Permutation Resolution:** `p = 5.0e-5` on Holdout reflects `0` of `20,000` stratified Rademacher sign flips exceeding the observed statistic (`1 / 20,001` Monte Carlo resolution floor under paired-difference symmetry), not an exact analytic `p`-value.
3. **Retention Set & Provenance Scope:** `math_holdout_retention150.jsonl` (`N=150`) is a reused V1 descriptive retention check outside the 4-contest Holdout average rather than fresh Holdout evidence or formal TOST non-inferiority. Single-touch governance relies on auditable two-phase local-NVMe seal files (`HOLDOUT_SEAL_*.json`), `COMPARATOR_PIN_*.json`, `14/14` verified hash-chained ledgers, and `38/38` verified SHA-256 receipts (`52/52` report metric dicts verified bit-for-bit against raw `*_metrics.json` files).

---

## 15. Reproducible Verification Commands

```bash
# 1. Verify Phase V1 & Phase V2 Immutable Split/Verifier Manifests
python -I -B eval/governed_loop_guard.py verify-manifest --manifest IMMUTABLE_MANIFEST.json --root .
python -I -B code/v2/governed_loop_guard_v3.py verify-manifest --manifest IMMUTABLE_MANIFEST_V2.json --root . --expected-root-sha256 f79f2fcf32262a0455ba85256ea2c4a8ab72b3eef94c76f0399190932e74ccf2

# 2. Verify Zero Holdout/Test Path Leaks Across All Phase V2 Candidate/Training/Eval Scripts
python -I -B code/v2/governed_loop_guard_v3.py audit-leaks --paths code/v2/model_registry_v2.py code/v2/vllm_eval_v2.py code/v2/build_v2_banks_and_controls.py code/v2/train_slm_v2.py code/v2/train_dapo_chunked_v2.py code/v2/repair_policy_v2.py code/v2/run_v2_plane2_track_l.py code/v2/run_v2_plane2_track_m.py --fail-on-warn

# 3. Re-Verify All 14 Phase V2 Hash-Chained Ledgers, 38 Holdout Receipts & Rebuild Statistical Bundle
python -I -B code/v2/build_v2_report_bundle.py .
```


# `AILab_Governed_SLM_RSI_v1` — Two-Plane Governed Recursive Self-Improvement for Small Language Models on Formal Lean 4.24 & Competition Mathematics

## 1. Executive Summary

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
- `sources/`: Three rounds of independent `GPT-6.2 Sol Max` frontier adversarial design, pre-Confirmation, and post-Holdout claim audits.

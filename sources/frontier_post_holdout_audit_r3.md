# Round-3 Post-Holdout Frontier Scientific Claim & Governance Audit (`GPT-6.2 Sol Max`)

**Date (UTC):** 2026-11-18  
**Reviewer Model:** `gpt-6.2-sol` (`reasoning_effort = max`, independent `rubber-duck` subagent)  
**Audited Artifacts:**
- `IMMUTABLE_MANIFEST.json` (`33` entries, `manifest_root_sha256 = 82a61cdf6b91a58fb537dc83f8bbbb0ea2ad46928a5a2174683db543a4ccf468`)
- `results/plane1/plane1_calibration_report.json`
- `results/plane2/{track_l_ledger,track_m_4b_ledger,track_m_9b_ledger}.jsonl`
- `results/confirm/{confirm_summary.json,track_l_confirm_ledger.jsonl,track_m_4b_confirm_ledger.jsonl,track_m_9b_confirm_ledger.jsonl}`
- `results/holdout/holdout_final_summary.json`

---

## 1. Independent Audit Verdict

1. **Cryptographic Integrity Confirmed:** All `33` frozen file hashes in `IMMUTABLE_MANIFEST.json` and all `6` append-only hash-chained ledgers (`19` total recorded Plane-2 + Confirmation gate decisions) verified cleanly (`status = "VERIFIED"`), alongside `28` recorded Single-Touch Holdout receipts.
2. **Core Empirical Gains Confirmed:** Both Confirmation-promoted `Qwen3.5-4B` checkpoints (`l2_q4b_restem_repair` on Track L Formal Lean 4.24; `m3_q4b_dapo_luffy` on Track M Hard Competition Math) cleared Holm-Bonferroni (`K=2`) Confirmation and achieved statistically significant `avg@8` gains over their calibrated `4B` baselines (`B_0`) on Protected Holdout without capability loss on `MATH-500`.

---

## 2. Binding Reporting Refinements Adopted in `README.md`

To prevent any overclaiming, every caveat identified by the Round-3 Frontier Audit is enforced verbatim in the final documentation:

1. **Plane-1 Calibration Wording:** Describe Plane-1 channel/budget calibration (`direct` `2,048` tok on Lean; `direct12k` `12,288` tok on Math) as **dramatically reducing truncation across calibrated decoding configurations** (`91.60% -> 0.78%` on Lean Search; `56.25% -> 13.87%` on Math Search), not "eliminating" truncation or isolating `<think>` at matched token length. Explicitly note residual truncation on the hardest contest holdouts (`19.17%` on `AIME 2025`, `23.86%` on `HMMT Feb 2026`).
2. **Track L (`MiniF2F-v2s test`, `N=244`) vs `27B` & `9B`:** Do **not** claim formal equivalence ("statistical parity") with `Qwen3.8-27B` from a two-sided CI crossing zero (`33.15%` vs `32.27%`, `+0.87 pp`, 95% CI `[-2.41, +4.25] pp`), and explicitly state that `27B` retains higher `pass@8` (`48.77%` [`119/244`] vs `45.90%` [`112/244`]). Against untuned `Qwen3.5-9B` (`30.02%`), report the `+3.13 pp` `avg@8` lead (nominal paired 95% CI `[+0.31, +5.99] pp`) alongside their exact tie on `pass@8` (`112/244 = 45.90%`). On `PutnamBench` (`N=128`), state accurately that all four arms scored `0/128` across **`k=8` whole-proof attempts per problem** (not single-shot `k=1`).
3. **Track M (`N=213` Pooled Hard Contest Holdout) Scope & Slice Heterogeneity:** Report both the paired-SE Confirmation p-value (`p = 2.72e-5`) and ledger effective-SE p-value (`p = 2.4e-5`). Scope the pooled Holdout lead over untuned `Qwen3.5-9B` (`59.45%` vs `55.40%`, `+4.05 pp`, nominal 95% CI `[+1.12, +7.04] pp`) by noting that `OlympiadBench` supplies `120 / 213` problems (`71.77%` vs `65.10%`, `+6.67 pp`), whereas on combined `AIME 2025+2026` (`N=60`) `4B A*` slightly trails `9B B_0` on `avg@8` (`48.33%` vs `50.21%`) while leading on `pass@8` (`85.00%` [`51/60`] vs `81.67%` [`49/60`]). Report `MATH-500` holdout retention (`94.83% -> 95.83%`, 95% CI `[-0.67, +2.83] pp`) as **no observed degradation** rather than a statistically proven gain, and emphasize that `Qwen3.8-27B` (`74.65%` pooled) remains substantially ahead on competition math.
4. **`Qwen3.5-9B` Confirmation Discipline:** Separate statistical evidence from operational guardrail eligibility. On Track M Confirmation, `m3_q9b_dapo_luffy` was highly statistically significant (`70.25% -> 73.75%`, `+3.50 pp`, `z = 3.84`, `p = 6.11e-5`) and improved `stop_rate` by `+5.92 pp` (`78.00% -> 83.92%`), but missed the frozen `85%` absolute `stop_rate` floor (`88%` in prose `program.md`), so neither `9B` tuned arm was authorized or evaluated on Protected Holdout. Recommend preregistering a **dual guardrail** (baseline-relative non-inferiority check + justified operational floor) in future governed cycles.

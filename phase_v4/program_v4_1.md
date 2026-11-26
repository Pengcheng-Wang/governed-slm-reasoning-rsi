# Program V4.1 — Peer-Review-Hardened Cross-Family Causal & Off-Policy Post-Training (`governed_loop_guard` v3.9)

## 1. Objective & Scope (`B9`, `B10`)
Wave 2 (`V4.1`) resolves the exact methodological confounds identified during the 7-model pre-submission adversarial review of `V4.0` without mutating any frozen `V1–V4.0` manifest or artifact (`IMMUTABLE_MANIFEST_V4.json` root `8f69efbd508ae4291fde22faa6edcc6a6cae759228d45dbcb147ea5f85e47c50` remains untouched):

1. **`G1` — Sealed Unexposed Balanced Splits (`B2`, `B8`, `B10`):**
   - `data/splits_v4_1/v4_1_lean_confirm150.jsonl` (`N=150`, `50/50/50` balanced strata, `N_eff=150.0`) & `v4_1_lean_holdout150.jsonl` (`N=150`, `50/50/50`)
   - `data/splits_v4_1/v4_1_math_confirm150.jsonl` (`N=150`, `50/50/50` balanced strata, `N_eff=150.0`) & `data/splits_v4_1/v4_1_math_holdout150.jsonl` (`N=150`, `50/50/50`)
   - 4-layer decontaminated against all 32 existing V1–V4 files (`0` ID, alphanumeric, skeleton, or 13-gram overlap), partitioned via pre-committed `SHA-256("v4_1_salt_20261124:" || cluster_id)`, and locked in `IMMUTABLE_MANIFEST_V4_1.json` prior to any `V4.1` evaluation.
   - `Holdout_V3` (`N=194` Math, `N=225` Lean) is strictly quarantined as zero-retuning historical transfer.
2. **`G2` — Cross-Family Protocol Equalization (`C1`):**
   - `2x2` evaluation of `Gemma-4-31B` and `Gemma-4-26B-A4B` on Lean `Search_V3` across `{enable_thinking=False, enable_thinking=True} x {max_tokens=2048, max_tokens=3072}` at `T=0.7`, `k=8`.
   - Inference-wrapper control on `Inkling-2-Mini` Lean `B_0` (`reasoning_effort="none"` vs. `"low"` at `max_tokens=3072`, `T=0.7`).
3. **`G3` — Controlled Wrapper-Only & Channel-Aligned SFT Intervention (`C3`):**
   - **`Inkling-2-Mini` Track L:** Exact matched-proof, matched-token (`0.0%` target-token delta), matched-80-step intervention comparing `stripped_mismatch` (`Thinking effort level: 0.2` + thoughtless `<|content_text|>`) vs. `effort_matched_wrapper` (`Thinking effort level: 0` + identical `<|content_text|>`, preserving `0.2` thinking trigger) vs. `native_thought_aligned` (uniformly sampled verified `<|content_thinking|>...<|content_text|>` rollouts), evaluated identically at `reasoning_effort="low"`, `max_tokens=3072`, `T=0.7`, `k=8`.
   - **`Muse-Glimmer-30B` Track L:** Channel-aligned `effort_matched_wrapper` / `native_thought_aligned` (`to=self` preserved) + precision-safe `GSPO`, testing whether fixing channel alignment resolves V4.0's `-5.28 pp` `stop_rate` rollback while retaining its `+7.04 pp` primary gain.
4. **`G4` — Precision-Safe Multi-Step RLVR & Common-Backbone `LUFFY` Ablation (`C2`, `C4`):**
   - On `Muse-Glimmer-30B` (`T_train == T_rollout == T_eval = 0.7`, `cispo_sft_alpha = 0.0`, unbiased stochastic `FP32 -> BF16` merge with logged `delta_energy_retention`, `40` optimizer steps), compare on one shared `GSPO` backbone:
     1. `Pure On-Policy GSPO` (`0%` off-policy)
     2. `Faithful Differentiated LUFFY` (`1` channel-aligned teacher + `7` student negatives on zero-pass prompts, joint group-normalized `A_i`, differentiated `f(p) = p/(p+0.1)`, logging token-`p` quantiles)
     3. `Log-Shaped Rare-Token Off-Policy` (`w(p) = 0.1/(p+0.1)`, `1+7` mixed groups)
     4. `Channel-Aligned SFT -> Pure On-Policy GSPO`
   - Governed promotion strictly via `governed_loop_guard_v39.py`: `select-search-winner -> gate-confirm-iut (1 look, alpha=0.025) -> gate-holdout (reserve -> score once -> finalize, requiring holdout_accepted == true)`.

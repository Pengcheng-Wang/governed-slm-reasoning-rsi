# Frontier Adversarial Review — Round 2 (Pre-Confirmation & Pre-Holdout Audit)

- **Reviewer:** `GPT-6.2 Sol` (`reasoning_effort = max`, independent `rubber-duck` subagent)
- **Stage:** Post-Plane-2 Search Freeze / Pre-Confirmation & Pre-Holdout Unlock
- **Trust Root Verified:** `IMMUTABLE_MANIFEST.json` (`33` entries, `manifest_root_sha256 = 82a61cdf6b91a58fb537dc83f8bbbb0ea2ad46928a5a2174683db543a4ccf468`) + all 3 Plane-2 hash-chained ledgers `VERIFIED`.

## Reviewer Findings & Pre-Unlock Binding Resolutions

1. **Finding 1 — Strict Separation of Confirmation and Protected Holdout (`confirm` vs `holdout`):**
   - *Critique:* `run_confirm_and_holdout.py` previously ran Confirmation and Protected Holdout back-to-back in one call regardless of Confirmation gate outcome.
   - *Binding Resolution:* Refactored `code/run_confirm_and_holdout.py` into strictly separate `confirm`, `finalize-confirm`, and `holdout` commands. Confirmation runs first across `node-0..2` without touching `eval/sealed_holdout/*.jsonl`. `finalize-confirm` computes exact Holm step-down (`K=2` per track, `alpha_family = 0.025`: smallest p-value tested at `0.0125`, second at `0.025` iff first rejects) plus all hard guardrails and seals `results/confirm/confirm_summary.json`. Only candidates passing Holm-adjusted statistical significance (`Holm LCB > 0`) AND all hard guardrails advance to single-touch Protected Holdout (`CONFIRMED_PROMOTE` if `raw_delta >= +0.03`, `CONFIRMED_POSITIVE_SUB3PP` if `0 < raw_delta < +0.03` on the already-76.8% calibrated Math baseline; any candidate failing Holm `LCB > 0` or any guardrail is `ROLLBACK` and excluded from Holdout). Holdout roster is strictly limited to `B_0`, Confirmation-cleared `A*`, and `Qwen3.8-27B` comparator (`think16k` excluded from Holdout).

2. **Finding 2 — Explicit Enforcement of `math500_retention_confirm150` & Guardrail Reconciliation:**
   - *Critique:* `math500_retention_confirm150` (`150 x 4 = 600` rollouts) was scored but its delta was not enforced in code before Holdout; `program.md` noted `88%/12%` stop/truncation vs `configs/gate_rules_math_confirm.json` `85%/15%`.
   - *Binding Resolution:* `finalize-confirm` explicitly enforces integer-exact retention `correct_count(A*) - correct_count(B_0) >= -12` out of `600` (`delta >= -0.0200`), checks `SE <= 0.0195`, records both `85%/15%` (frozen JSON) and `88%/12%` (`program.md`) stop/truncation checks transparently in `confirm_summary.json`, and fails closed (`ROLLBACK`) on any retention breach.

3. **Finding 3 — Exact Holm Step-Down (`K=2` per track) vs Conservative Bonferroni:**
   - *Critique:* `governed_loop_guard.py` applies per-call Bonferroni (`0.025 / 2 = 0.0125` to both `4B` and `9B` independently).
   - *Binding Resolution:* `finalize-confirm` combines the paired cluster-level difference vectors `(d_4b, d_9b)` per track (`Track L` `N=243`, `Track M` `N=300`) to compute exact paired bootstrap / t-distribution one-sided p-values and exact Holm step-down (`0.0125` on rank 1, `0.025` on rank 2 iff rank 1 rejects), recording both conservative Bonferroni and exact Holm verdicts in `confirm_summary.json`.

4. **Finding 4 — Fail-Closed Return-Code Checking & Artifact SHA-256 Binding:**
   - *Critique:* `_gate_confirm` allowed `rc=1` alongside `(0, 3)`.
   - *Binding Resolution:* Fixed `_gate_confirm` to allow ONLY `(0, 3)` (`rc=1` raises `SystemExit` immediately) and bound every Confirmation and Holdout output to SHA-256 digests of the split file and model checkpoint metadata.

5. **Finding 5 — Label-Blind Prompt Projection in `vllm_eval_engine.py` & Blind Final-Bank Zero-Overlap Audit:**
   - *Critique:* `vllm_eval_engine.py` serialized full split dicts (including `gold`) into `worker_cfg_*.json`.
   - *Binding Resolution:* Updated `vllm_eval_engine.py` to project every input row strictly to `{"id": str(it["id"]), "messages": it["messages"]}` before spawning worker subprocesses, guaranteeing zero ground-truth label exposure during generation. Executed blind normalized-statement / normalized-problem SHA-256 overlap audit between all Plane-2 cumulative verified banks / RL rollout files and all Confirmation + Protected Holdout splits (`0` overlaps). Also report exact split sizes (`N=33` on `hmmt_feb2026` per `split_audit_manifest.json`) in all Holdout tables.

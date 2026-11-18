# Frontier Design & Governance Review (Round 1 — Pre-Freeze)

**Reviewer:** Frontier Reasoning Agent (`GPT-6.2 Sol`, `reasoning_effort = max`, independent read-only critique)
**Scope:** Two-Track Governed RSI (`governed-loop-optimization`, Archetype D Two-Plane Hybrid) on `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` across Track L (Formal Lean 4 Theorem Proving) and Track M (Hard Olympiad & Competition Mathematics).

---

## 1. Key Findings & Adopted Protocol Hardening

1. **Full-Gate Statistical Power (`gain >= eps AND LCB > 0`):**
   - Sizing must satisfy $S \le S_* = \min\{(\Delta_* - \epsilon)/0.8416, \Delta_*/(z_{1-\alpha/K} + 0.8416)\}$.
   - **Adopted Rule:** Set `multiplicity_k = 2` (`alpha_one_sided = 0.025 -> eff_alpha = 0.0125`, $z = 2.2414$) for Confirmation per track, with `min_improvement = 0.03` (`+3.0 pp` minimum worthwhile observed gain) and design gain $\Delta_* \ge 0.06$ (`+6.0 pp`), giving $S_* = 1.95\text{ pp}$ (`>= 80%` full-gate power at $N_d = 243$ Lean theorem clusters and $N_d = 300$ Math problem clusters). All exploratory candidate screening occurs strictly on the disjoint Search split (`decision_stage = "search"`).
   - **Retention Guardrail:** On the disjoint `MATH-500` retention panel (`n = 150`), enforce both point non-degradation and a paired cluster-bootstrap lower confidence bound (`retention_lcb95_vs_base >= -0.04`).

2. **Out-of-Process Verifier Hardening (`B1`, `B3`):**
   - **Lean 4.24 (`eval/lean4_verifier.py`):**
     - Block comment tokens (`/-`, `-/`) inside candidate proof bodies so candidates cannot swallow trailing harness commands.
     - Block all scope/declaration/metaprogramming directives (`namespace`, `section`, `end`, `open`, `local`, `scoped`, `attribute`, `notation`, `infix`, `prefix`, `postfix`, `macro`, `syntax`, `elab`, `axiom`, `constant`, `opaque`, `unsafe`, `partial`, `extern`, `implemented_by`, `run_tac`, `sorry`, `admit`, `sorryAx`, `native_decide`, `import`, `set_option`, and nested top-level `theorem`/`lemma`/`example`/`def`/`instance`).
     - Mechanically splice only the sanitized tactic/term body onto the benchmark's exact frozen header + theorem declaration, append `#print axioms <EXACT_THM_NAME>`, and verify exact single-match axiom output for `<EXACT_THM_NAME>` with transitive axioms $\subseteq \{\texttt{propext}, \texttt{Classical.choice}, \texttt{Quot.sound}\}$.
     - Qualify against a 12-case adversarial attack & positive-control unit suite (`12/12` required before freeze).
   - **Competition Math (`eval/math_verifier.py`):**
     - Exclude proof-only / non-verifiable open-ended items at corpus construction; freeze per-item answer type (`integer` vs `symbolic`), handle tuple order vs finite-set membership and percent/unit scaling deterministically, and run blind gold-parseability + consensus checks before split freeze.
     - Explicitly label `AIME 2026` and `HMMT Feb 2026` as held-out external benchmark panels (untouched during all Search and Confirmation steps in this study, with historical `SLM_Qwen35_v1` `PRIMARY_UNEVALUABLE` baseline provenance disclosed).

3. **Two-Plane Separation for Proof+Repair Policy (`L4`):**
   - Keep `L1–L3` strictly on the frozen 1-turn `k = 8` direct evaluation protocol `h_0` on `minif2f_v2c_valid`.
   - Evaluate `L4` (`4 direct + up to 4 compiler-diagnostic repair calls` vs `8 direct calls` on the identical frozen checkpoint) on the **disjoint 97-theorem validation reserve** of `powered_validation.jsonl` (fresh theorem clusters untouched by Search `128` and untouched by weight Confirmation), reporting call-matched `any_verified_solve_at_8`, total tokens, and wall/GPU-seconds separately from weight-only gains.

4. **Hybrid Gated-DeltaNet + Multimodal `Qwen3_5ForConditionalGeneration` Training Hygiene:**
   - Enumerate exact `language_model` linear projections across both Full-Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`), Gated-DeltaNet (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`), and MLP (`gate_proj`, `up_proj`, `down_proj`), strictly excluding `visual` tower, multimodal merger, `embed_tokens`, and `lm_head`.
   - Disable sequence packing (`packing = False`) so Gated-DeltaNet `conv1d` and recurrent states never bleed across sample boundaries; supervise exact `<|im_end|>` (`248046`).
   - Merge LoRA deltas once in `FP32` before casting to `BF16`, verify attached-vs-merged logit parity in the identical HF reference runtime, and run a `vLLM --language-model-only` serving parity smoke test before Round 1.
   - Log truncation and `finish_reason` across **all** raw rollouts (not only admitted traces); non-`stop` rollouts earn `0` correctness reward.

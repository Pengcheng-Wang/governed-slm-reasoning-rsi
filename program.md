# Governed Self-Improving SLM Program (`program.md`)
**Experiment ID:** `AILab_Governed_SLM_RSI_v1`  
**Archetype:** **Archetype D — Two-Plane Hybrid RSI** (`/governed-loop-optimization`) + Frontier Adversarial Review (`GPT-6.2 Sol Max`)  
**Compute Envelope:** `3 nodes × 8 NVIDIA A100-SXM4-80GB` (`24 GPUs`, BF16, `vLLM 0.29.0+cu129`, `PyTorch 2.13.0+cu129`, `Lean 4.24.0 + Mathlib f897ebcf`)

---

## 1. Scientific & Engineering Objective
Prior `AILab*` experiments exposed two severe Small Language Model (`4B–9B`) failure walls on `Qwen3.5-4B` (`851bf6e8...`) and `Qwen3.5-9B` (`c2022362...`), where `75%` of text layers are Gated-DeltaNet (`linear_attn`) and `25%` are full attention (`self_attn`):
1. **Track L — Formal Lean 4 Theorem Proving (`MiniF2F-v2s` + `PutnamBench`):**
   - In `AILab_SLM_Qwen35_v2`, untuned `Qwen3.5-9B` scored `1/100 (1%)` pass@8 in direct mode and `7/100` in thinking mode (`42.5%–79.4%` truncated), and **zero tuned `4B`/`9B` models were ever trained or evaluated**.
   - In `AILab_BOss_Dense_Lean_v1` (`NM-011`), untuned compiler-error repair degraded `pass@8` (`4/96` vs `6/96`) because base models were never trained on `(failed_proof, lean_diagnostics) -> verified_proof` pairs.
2. **Track M — Hard Competition & Olympiad Math (`AIME 2024/2025/2026`, `HMMT Feb 2025/2026`, `BeyondAIME`, `OlympiadBench`):**
   - In `AILab_SLM_Qwen35_v1`, all four `Qwen3.5-4B` post-training runs failed (`LoRA -6.67 pp NO-GO`, `Full SFT -56 to -63 pp NO-GO`) due to chat-channel mismatch (`enable_thinking=False` stripping native `<think>` priors), `14%–33%` right-truncated supervision, and unchunked `248,320`-vocab CE OOM/fallback.

Our goal is to break both walls under strict Two-Plane Governed RSI discipline and surpass all prior `AILab` SLM results — while benchmarking directly against the `3x–7x` larger `Qwen3.8-27B` frontier comparator under identical verifiers and protocols.

---

## 2. Immutable Trust Root (Frozen After Plane-1 Calibration)
Once `IMMUTABLE_MANIFEST.json` is sealed via `eval/governed_loop_guard.py freeze-manifest`, the root SHA-256 is recorded outside the mutable tree and **zero** files listed below may be edited:
- `eval/governed_loop_guard.py` (stdlib precondition, manifest, paired-bootstrap, Holm-Bonferroni, and ledger verifier)
- `eval/lean4_verifier.py` (benchmark-owned frozen Lean 4.24.0 theorem header + `#print axioms` audit $\subseteq \{\texttt{propext}, \texttt{Classical.choice}, \texttt{Quot.sound}\}$)
- `eval/math_verifier.py` (`math_verify 0.9.0` + `SymPy 1.14.0`, ordered-tuple vs set vs interval disambiguation, decoupled strict SFT admission vs lenient evaluation)
- `eval/isolated_scorer.py` (out-of-process `python -I -B` scorer enforcing complete `(id, sample_idx)` grids)
- `data/splits/*.jsonl` and `eval/sealed_holdout/*.jsonl` (exact theorem-stem / 13-gram normalized problem decontamination verified in `split_audit_manifest.json`)
- `configs/gate_rules_*.json` and this `program.md`

Every iteration begins with `python -I -B eval/governed_loop_guard.py verify-manifest`; any hash mismatch immediately aborts (`STOP_INFRA_DRIFT`).

---

## 3. Two-Plane Mutability Surface
### Plane 2A — Inner Weight-Optimization Plane (Single-Call Frozen Harness)
Allowed mutations (`code/train_slm_rsi.py`, `code/train_dapo_luffy.py`, `code/build_frontier_bank.py`, `configs/exp_*.json`):
- **Hybrid Adapter Coverage (`HYBRID_LM_LINEAR_REGEX`) vs `full_ft_lm`:** Explicit LoRA targeting of `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`, and `mlp.{gate,up,down}_proj` (`model.language_model` only; vision tower & MTP frozen), plus controlled comparison against full `model.language_model` fine-tuning (`lr=5e-6`).
- **Cumulative `ReST^EM` Always Resetting to `M_0`:** Each round expands the kernel/symbolically verified trajectory bank from frontier rollouts (`k=8` on `0`-trace, `k=4` on `1`-trace, `k=0` on `>=2`-trace graduated items), uniformly retains $\le 3$ deduplicated traces per problem (never shortest-trace selection) with problem-balanced weights $w_i = 1/m(p_i)$, and **retrains fresh from base checkpoint `M_0`** (`0%` truncated traces, `packing=False`, 1,024-token chunked `248,320`-vocab CE, `FP32` LoRA merge + logit parity gate `< 2e-2`).
- **Frontier-Curated `DAPO + LUFFY` RLVR (`L3` / `M3`):** Clipped policy-gradient updates (`eps_low=0.20, eps_high=0.28`, zero KL penalty, soft-overlong penalty, unmasked truncated completions, `f(\pi)=\pi/(\pi+0.1)` off-policy shaping on `1-of-8` rescued `0/8` groups) restricted strictly to active frontier problems ($1 \le \text{pass}/8 \le 6$ plus `LUFFY`-rescued `0/8`).

### Plane 2B — Outer Inference/Repair Policy Plane (Frozen Weights Only)
- Tested **strictly on frozen weights** (`L4`) via `code/repair_policy_eval.py` on the disjoint `lean_repair_reserve97.jsonl` (`n=97` clusters, untouched by `lean_search128`):
- Compares call-matched `4 direct + up to 4 compiler-diagnostic repairs` ($\le 8$ LLM calls, $\le 8$ Lean kernel calls) against `8 independent direct calls` on the identical frozen checkpoint.

---

## 4. Statistical Gates, Confirmation Power Contract, and Guardrails
- **Multiplicity Control (`Holm-Bonferroni`):** Confirmation tests `multiplicity_k = 2` primary hypotheses per track (`4B` winner vs `4B B_0`, and `9B` winner vs `9B B_0`) at one-sided familywise $\alpha = 0.025$ (`eff_alpha = 0.0125` for the lead test, `0.025` for the second).
- **Minimum Material Improvement ($\delta_{\min}$):** `+0.03` (`+3.0 pp`) on paired Confirmation (`lean_confirm243`, $N_d=243$ clusters; `math_confirm300`, $N_d=300$ clusters), requiring paired-bootstrap $SE(\Delta) \le 1.95\text{ pp}$ (`>= 80%` full-gate power at design gain $\Delta_* = +6.0\text{ pp}$).
- **Primary Endpoints:**
  - **Track L (Inner Plane 2A):** `avg@8` (`mean_verified_pass_rate`, dense per-theorem $c_i/8 \in [0, 1]$) on `lean_search128` (`n=128`) and `lean_confirm243` (`n=243`), reporting `pass@8` (`any_verified_solve_at_8`) alongside as co-reported theorem-discovery rate.
  - **Track L (Outer Plane 2B `L4`):** `any_verified_solve_at_8` (`<= 8` calls total) on `lean_repair_reserve97` (`n=97`).
  - **Track M (Inner Plane 2A):** `avg@8` (`mean_accuracy`) on `math_search128` (`n=128`) and `math_confirm300` (`n=300`), reporting `pass@8` and `maj@8` alongside.
- **Hard Guardrails (Zero Silent Bypass):**
  - `axiom_audit_clean_rate == 1.0` (Track L)
  - `truncation_rate <= 0.15` (Track L) / `<= 0.12` (Track M Confirmation)
  - `stop_rate >= 0.85` (Track L) / `>= 0.88` (Track M Confirmation)
  - `math500_retention_confirm150` `avg@4` drop `<= 2.0 pp` vs `B_0` (anti-forgetting guardrail on Track M Confirmation)
  - Compute cap per iteration: `<= 24 GPU-hours` per training round; `<= 4` inner rounds per model/track (`STOP_DIMINISHING_RETURNS` if 2 consecutive rounds improve Search primary metric by `< 1.5 * se_AA`).

---

## 5. Protected Single-Touch Holdout Firewall
`eval/sealed_holdout/*` (`minif2f_v2s_test` `n=244`, `putnambench_holdout128` `n=128`, `aime2025_holdout30`, `aime2026_holdout30`, `hmmt_feb2026_holdout30`, `olympiadbench_holdout120`, `math500_retention_holdout150`) may be evaluated **once** at campaign conclusion on:
1. Untuned `B_0` (`Qwen3.5-4B` and `Qwen3.5-9B`),
2. Confirmation-promoted `A*` (`4B` and `9B`),
3. Untuned `Qwen3.8-27B` frontier reference comparator.
Holdout results are reported verbatim and **never** feed back into hyperparameter, prompt, checkpoint, or policy selection.

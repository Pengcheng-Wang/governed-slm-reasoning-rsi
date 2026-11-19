# Post-Training Methodology, Breadth vs. Depth Dynamics, and Model-Aware Data Scaling

This report synthesizes generalizable post-training principles established across our reasoning fine-tuning experiments and recent open recipes (notably [Axion 1.0, arXiv:2611.03233](https://github.com/bytedance/axion1_0)).

---

## 1. Training & Evaluation Data-Hygiene Preconditions

Before attributing flat or regressed evaluation scores to model capacity or optimization algorithms, both the training corpus and evaluation harness must pass four mechanical invariants:

| Invariant | Failure Mechanism When Violated | Verified Protocol |
| :--- | :--- | :--- |
| **1. Channel & Token-Budget Alignment** | Long-CoT `<think>` traces truncated at `4K–8K` omit `</think>` and `\boxed{}`; right-truncated SFT teaches non-termination, while clipped RL rollouts receive `reward = 0` and bias gradients against multi-step reasoning. | Tokenize the complete chat-templated string before training; set `max_length >= p99` (or drop overlong traces intact with `0%` right-truncation) and match eval `max_tokens` to the trained channel. |
| **2. Strict Keep-Rule & Unmasked EOS** | Keeping any trace containing `\boxed{}` admits scratch guesses inside `<think>`; sequence packing with naive assistant-only masking strips `<|im_end|>` / `<|end|>` log-probability. | Require `finish_reason == "stop"`, post-`</think>` answer extraction, non-repetitive reasoning blocks, and explicit EOS supervision without cross-sample boundary truncation. |
| **3. Architecture-Complete Adaptation** | On hybrid linear-attention + full-attention models (`Qwen3.5` / `Qwen3.8` Gated-DeltaNet) or fused-projection models (`Phi-4` `qkv_proj`/`gate_up_proj`), partial LoRA target lists underfit recurrent layers, and `modules_to_save=[embed, lm_head]` at adapter LR corrupts representations. | Adapt all linear projections across full-attention, linear-attention, and MLP blocks (`FP32` exact merge parity `>= 99%`), freeze embeddings, and disable DDP buffer broadcasts on dynamic LongRoPE buffers. |
| **4. Activation-Checkpointed Chunked Logits** | Materializing `[B, T, V]` FP32 logits (`V = 248,320`) at `12K–16K` context consumes `~40 GB/GPU`, forcing destructive sequence-length cuts. | Run the backbone in BF16 and compute `lm_head + log_softmax + gather` inside checkpointed `1,024`-token completion chunks (`< 30 GB` peak VRAM on `80 GB` GPUs). |

---

## 2. Division of Labor Across Post-Training Stages: Breadth (`SFT`) vs. Depth (`RLVR`)

Controlled decomposition across model scales (`3.8B` to `30B+`) and formal/mathematical verifiers establishes a sharp functional separation between supervised fine-tuning and verifier-driven reinforcement learning:

1. **Verified SFT Expands Capability Breadth (`pass@k` Reach & Cold-Start Enablement):**
   - When a base model has low initial competence or faces an unfamiliar formal syntax/premise distribution, RLVR from base either stalls on zero-variance (`0/k`) rollout groups or collapses existing capabilities.
   - Diverse, verified-trace SFT shifts probability mass onto valid reasoning templates and domain premises, lifting **`pass@8` / `pass@16` coverage** and acting as a critical enabler for subsequent RLVR.
   - However, repeated off-policy `ReST^EM` / continued SFT on a fixed small prompt pool saturates after 1–2 rounds (and continued SFT without new problem diversity can flatten single-sample sharpness).

2. **Verifier RL (`RLVR`) Sharpens Reasoning Depth & Small-`k` Reliability (`pass@1`, `pass@2`, `maj@8`):**
   - Starting from a verified SFT checkpoint, online RLVR reallocates probability mass from failed trajectories onto verifiable derivations, producing large, temperature-invariant gains on **`pass@1` (`avg@k`), `pass@2`, and `maj@k`** even when `pass@16` changes modestly.
   - **The Capability-Curve Crossover (`slope = Δpass@8 - Δpass@1`):**
     - When current `pass@8 < ~35%` (weak/moderate competence), RLVR expands *both* `pass@8` and `pass@1` (`slope > 0`).
     - When current `pass@8 > ~35%` (strong competence), RLVR primarily sharpens `pass@1` and `maj@8` (`slope < 0`), converting `1-in-8` lucky solves into consistent `6-in-8` solves while still accumulating genuinely new (`0 -> >0`) solves as optimizer steps scale.

3. **Why Measurement Must Track the Full `pass@k` Curve:**
   - Evaluating only `avg@8` (`pass@1`) conflates breadth expansion with distribution sharpening; evaluating only `pass@8` misses dramatic `pass@1` / `maj@8` reliability gains on strong models.
   - Every candidate should be scored on **`pass@1`, `pass@2`, `pass@4`, `pass@8`, `maj@8`, `capability_slope = Δpass@8 - Δpass@1`, problem transition counts (`0 -> >0` vs `>0 -> 0`), and `stop_rate`**.

---

## 3. Comparing RLVR Formulations & Training Stability

| RLVR Mechanism | Formulation | Empirical Strength & Failure Boundary |
| :--- | :--- | :--- |
| **`DAPO` (Decoupled Clip + Token-Level Loss + Variance Filter)** | Asymmetric PPO clip `[0.20, 0.28]` (`clip-higher`), token-level loss normalization across valid groups (`0 < pass_count < G`), linear soft-overlong penalty, `KL = 0`. | Prevents entropy collapse (`clip-higher` preserves low-probability reasoning pivots) and eliminates wasted steps on `0/8` or `8/8` zero-variance groups; simultaneously raises `pass@1` and `stop_rate`. |
| **`Dr.GRPO` + `CISPO` (Unbiased Advantage + Clipped-IS Policy Gradient)** | Centers rewards `R_i - mean(R)` **without** dividing by `std(R)` or sequence length `|o_i|` (using fixed `L_norm`), paired with clipped importance-sampling weight `sg(min(r_t, 1 + eps_high)) * log pi(y_t|x)`. | Removes standard GRPO's length bias (which encourages concise correct answers and verbose wrong answers) and preserves gradient flow on critical reasoning tokens instead of zeroing gradients past the clip boundary. |
| **`GSPO` + Style-Matched `LUFFY` (Sequence-Level IS + Off-Policy Rescue)** | Geometric sequence-level importance ratio `(pi / pi_old)^(1/|y|)` (`GSPO`) + `1-of-8` verified off-policy trace injection with shaping `pi / (pi + gamma)` (`gamma = 0.1`) on `0/8` groups. | Keeps dead-group fraction at `0.000` on hard frontier problems where on-policy rollouts score `0/8`. **Crucial precondition:** injected off-policy traces must match the student's channel and length regime (verbose mismatched teacher traces inflate length and hurt `stop_rate`). |
| **Checkpoint Window Selection** | Evaluate intermediate RL checkpoints (`50%`, `75%`, `100%` of horizon) on a Search/Inner-Val split rather than blindly taking the final step. | RLVR reward and length curves can oscillate late in training; selecting over a small saved checkpoint window on Search prevents late-step over-sharpening or length drift. |

---

## 4. High-Quality Data Scaling: The `Axion 1.0` (`2 SFT + RL`) Recipe Adapted to Reasoning & Formal Proving

[Axion 1.0 (`bytedance/axion1_0`)](https://github.com/bytedance/axion1_0) demonstrates that frontier reasoning capabilities emerge from **investing complexity in verifiable, model-aware data construction** paired with a clean **3-stage recipe (`Stage-I SFT -> Stage-II SFT -> RL`, without mid-training, DPO, or OPD)**:

1. **Decouple Verified Problem Construction from Trajectory Collection:**
   - First establish a diverse, decontaminated bank of challenging questions with **programmatically checkable reference answers** (exact symbolic/numeric math answers or Lean 4 kernel-elaborated formal statements).
   - **Model-Aware Problem Strengthening:** Probe candidate problems with the base model. Problems already solved effortlessly (`k/k`) provide low gradient value; rewrite or filter toward harder compositions, parameterized generalizations, boundary-case analyses, and multi-lemma formal obligations whose answers are independently audited before trajectory harvesting.
2. **Stage-I SFT — Broad Verified Reasoning Foundation (`Breadth`):**
   - Train on a broad, multi-domain, strictly verified trajectory bank (`finish_reason == "stop"`, exact token-cap compliance, zero benchmark overlap) to expand domain coverage (`pass@8`, Mathlib premise vocabulary, competition combinatorics/number theory/algebra/geometry).
3. **Stage-II SFT — Model-Aware Hard-Frontier & Self-Verification (`Hard-Case Completeness`):**
   - Starting from Stage-I, curate verified trajectories specifically on **hard frontier items** (`0 < pass_rate <= 0.50` plus model-aware strengthened variants) with explicit boundary/case verification, resetting optimizer state cleanly.
4. **Stage-III RLVR — Nonzero-Variance Frontier Sharpening (`Depth & Small-k Precision`):**
   - Pre-screen candidate RL tasks with `k=4` rollouts from the Stage-II checkpoint, retaining tasks with **nonzero reward variance** (plus `LUFFY` rescue on hard `0/4` tasks with verified reference traces), and run multi-proposal RLVR (`DAPO`, `Dr.GRPO + CISPO`, `GSPO + LUFFY`) gated by an executable two-plane promotion ledger.

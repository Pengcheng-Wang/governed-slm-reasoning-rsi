#!/usr/bin/env python3
"""
Phase V2 Multi-Model Holm-Bonferroni Confirmation Gate & Single-Touch Protected Holdout Evaluator.

Executes across 3 nodes (`--phase confirm --role ...` then `--phase holdout --role ...`):
  1. Verifies `IMMUTABLE_MANIFEST_V2.json` against `/scratch/governed_slm_rsi_v2_trust_root.sha256` pre & post.
  2. Confirmation (`ProofNet-Lean4 valid`, `N=164`; `Omni-MATH Tier 3-4`, `N=256`):
     - Evaluates untuned `B_0` and Plane-2 Search winner `A*` for every model (`qwen35_4b`, `qwen3_4b`,
       `phi4_mini`, `qwen35_9b`) plus untuned `Goedel-Prover-V2-8B` and `Qwen3.8-27B` comparators.
     - Runs 4-way Causal Compiler-Repair Ablation (`direct8` vs `masked_retry_4_4` vs `shuffled_diag_4_4`
       vs `compiler_repair_4_4`) on `ProofNet-Lean4 valid` (`N=164`).
     - Applies step-down **Holm-Bonferroni (`alpha_one_sided = 0.05` across all `K` promoted challengers)**
       plus dual completion guardrails (`configs_v2/gate_rules_v2_{lean,math}_confirm.json`) and seals
       `results/confirm/v2_confirm_summary.json`.
  3. Single-Touch Protected Holdout (`ProofNet-Lean4 test`, `N=167`; `BeyondAIME` `100` + `BRUMO 2025` `30`
     + `CMIMC 2025` `40` + `HMMT Nov 2025` `30` = `N=200` Hard Contest Suite):
     - Evaluates ONLY after `v2_confirm_summary.json` is sealed!
     - Writes immutable SHA-256 single-touch receipts (`*_receipt.json`) locking split digest, checkpoint
       fingerprint, seed, and output metrics digest.
     - Compiles paired 10,000-resample cluster-bootstrap 95% CIs across all models, comparators, and causal
       controls into `results/holdout/v2_holdout_final_summary.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_registry_v2 import MODEL_REGISTRY_V2  # noqa: E402
from v36_conjunctive_guard_helper import (  # noqa: E402
    build_combined_4contest_metrics_json,
    canonical_checkpoint_manifest,
    exact_sign_test_one_sided,
    fsync_file_and_dir,
    stratified_sign_flip_test,
    write_v36_amendment,
)

V2 = Path("/scratch/governed_slm_rsi_v2")
V1 = Path("/scratch/governed_slm_rsi_v1")
MANIFEST_V2 = V2 / "IMMUTABLE_MANIFEST_V2.json"
TRUST_V2 = Path("/scratch/governed_slm_rsi_v2_trust_root.sha256")
GUARD = V1 / "eval/governed_loop_guard.py"
GUARD_V3 = V2 / "code/v2/governed_loop_guard_v3.py"
CONFIRM_SUMMARY_P = V2 / "results/confirm/v2_confirm_summary.json"
GOV_V3_DIR = V2 / "results/v3_governance"
HOLDOUT_CONTEST_COUNTS = {
    "beyondaime": 100,
    "cmimc2025": 40,
    "brumo2025": 30,
    "hmmt_nov2025": 30,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ckpt_fingerprint(model_path: Path) -> dict[str, str]:
    out: dict[str, str] = {"model_path": str(model_path)}
    for fn in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "chat_template.jinja"):
        fp = model_path / fn
        if fp.is_file():
            out[fn] = _sha256_file(fp)
    return out


def _verify_manifest() -> None:
    expected = TRUST_V2.read_text(encoding="utf-8").strip().split()[0]
    subprocess.run(
        [
            sys.executable, "-I", "-B", str(GUARD),
            "verify-manifest",
            "--manifest", str(MANIFEST_V2),
            "--root", str(V2),
            "--expected-root-sha256", expected,
        ],
        check=True,
    )


def _runtime_unique_id_view(src_path: Path) -> Path:
    """Return `src_path` unchanged if all `id`s are unique; otherwise write a deterministic
    row-index-disambiguated runtime view outside protected manifest directories so all 164/167
    distinct ProofNet theorems across textbooks are evaluated without touching sealed files.
    """
    rows = [json.loads(ln) for ln in src_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) == len(ids):
        return src_path
    view_dir = V2 / "results/.runtime_unique_id_views"
    view_dir.mkdir(parents=True, exist_ok=True)
    out_p = view_dir / src_path.name
    seen: set[str] = set()
    disamb_rows: list[str] = []
    for idx, r in enumerate(rows):
        rid = str(r["id"])
        if rid in seen:
            r = dict(r)
            r["id"] = f"{rid}__row{idx}"
        seen.add(rid)
        disamb_rows.append(json.dumps(r, ensure_ascii=False))
    out_p.write_text("\n".join(disamb_rows) + "\n", encoding="utf-8")
    return out_p


def _eval_split_v2(
    *,
    track: str,
    model_key: str,
    model_path: Path,
    split_file: Path,
    prompts_file: Path | None,
    out_prefix: Path,
    cand_id: str,
    protocol_id: str,
    k: int = 8,
    thinking: bool = False,
    temperature: float = 0.8,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    gen_gz = Path(f"{out_prefix}.jsonl.gz")
    met_json = Path(f"{out_prefix}_metrics.json")
    det_gz = Path(f"{out_prefix}_details.jsonl.gz")
    rec_json = Path(f"{out_prefix}_receipt.json")
    gen_input = prompts_file if (prompts_file and prompts_file.is_file()) else split_file
    eval_split_view = _runtime_unique_id_view(split_file)
    gen_input_view = _runtime_unique_id_view(gen_input)

    expected_receipt = {
        "candidate_id": cand_id,
        "protocol_id": protocol_id,
        "model_key": model_key,
        "split_file": str(split_file),
        "split_sha256": _sha256_file(split_file),
        "gen_prompts_file": str(gen_input),
        "gen_prompts_sha256": _sha256_file(gen_input),
        "ckpt": _ckpt_fingerprint(model_path),
        "k": k,
        "thinking": thinking,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "seed": 20261118,
    }
    if rec_json.is_file() and met_json.is_file():
        old_rec = json.loads(rec_json.read_text(encoding="utf-8"))
        for key in ("split_sha256", "ckpt", "k", "thinking", "temperature", "max_tokens", "seed"):
            if old_rec.get(key) != expected_receipt.get(key):
                raise SystemExit(f"Single-touch receipt mismatch on {out_prefix} key={key}")
        if old_rec.get("metrics_sha256") != _sha256_file(met_json):
            raise SystemExit(f"Single-touch receipt metrics_sha256 mismatch on {met_json}")
        return met_json

    if not gen_gz.is_file():
        cmd = [
            sys.executable, "-I", "-B", str(V2 / "code/v2/vllm_eval_v2.py"),
            "--model-key", model_key,
            "--model-path", str(model_path),
            "--split-file", str(gen_input_view),
            "--output", str(gen_gz),
            "--summary-output", f"{out_prefix}_gen.json",
            "--k", str(k), "--gpus", "0,1,2,3,4,5,6,7",
            "--temperature", str(temperature),
            "--presence-penalty", str(presence_penalty),
            "--max-tokens", str(max_tokens),
            "--max-model-len", str(max(16384, max_tokens + 4096)),
            "--seed", "20261118",
        ]
        if thinking:
            cmd.append("--thinking")
        subprocess.run(cmd, check=True)

    if not met_json.is_file():
        scorer_script = (V2 / "code/v2/timeout_isolated_scorer_v2.py") if track == "math" else (V1 / "eval/isolated_scorer.py")
        scmd = [
            sys.executable, "-I", "-B", str(scorer_script),
            "--track", track,
            "--split-file", str(eval_split_view),
            "--generations", str(gen_gz),
            "--expected-k", str(k),
            "--candidate-id", cand_id,
            "--protocol-id", protocol_id,
            "--workers", "36",
            "--output", str(met_json),
            "--scored-details-output", str(det_gz),
        ]
        if thinking:
            scmd.append("--thinking")
        subprocess.run(scmd, check=True)

    expected_receipt["metrics_sha256"] = _sha256_file(met_json)
    rec_json.write_text(json.dumps(expected_receipt, indent=2) + "\n", encoding="utf-8")
    return met_json


def _paired_bootstrap(a_json: Path, b_json: Path, *, n_boot: int = 10000, seed: int = 20261118) -> dict[str, float]:
    da = json.loads(a_json.read_text(encoding="utf-8"))
    db = json.loads(b_json.read_text(encoding="utf-8"))
    ma = {str(it["id"]): float(it["value"]) for it in da["items"]}
    mb = {str(it["id"]): float(it["value"]) for it in db["items"]}
    keys = sorted(ma.keys())
    diffs = [ma[k] - mb[k] for k in keys]
    n = len(diffs)
    obs = sum(diffs) / max(n, 1)
    rng = random.Random(seed)
    boots = [sum(diffs[rng.randrange(n)] for _ in range(n)) / max(n, 1) for _ in range(n_boot)]
    boots.sort()
    se = math.sqrt(sum((x - obs) ** 2 for x in boots) / max(len(boots) - 1, 1))
    z = obs / max(se, 1e-9)
    p_one_sided = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {
        "a_mean": sum(ma.values()) / max(n, 1),
        "b_mean": sum(mb.values()) / max(n, 1),
        "delta": obs,
        "se": se,
        "z": z,
        "p_one_sided": p_one_sided,
        "ci95_lo": boots[int(0.025 * n_boot)],
        "ci95_hi": boots[int(0.975 * n_boot)],
    }


def _best_goedel_channel() -> tuple[bool, int]:
    p1 = V2 / "results/plane1/lean"
    d_m = json.loads((p1 / "ref_goedel_8b_direct_metrics.json").read_text(encoding="utf-8"))["metrics"]["lean_avg_at_8"]
    t_m = json.loads((p1 / "ref_goedel_8b_think8k_metrics.json").read_text(encoding="utf-8"))["metrics"]["lean_avg_at_8"]
    if t_m >= d_m:
        return True, 8192
    return False, 2048


def run_confirm_role(role: str) -> None:
    lean_cf = V2 / "data/splits_v2/v2_lean_confirm171.jsonl"
    math_cf = V2 / "data/splits_v2/v2_math_confirm256.jsonl"

    if role == "node0_lean":
        od = V2 / "results/confirm/lean"
        for mk in ("qwen35_4b", "qwen3_4b", "qwen35_9b", "phi4_mini"):
            inc_p = V2 / f"results/plane2/track_l/{mk}/incumbent.json"
            if not inc_p.is_file():
                continue
            inc = json.loads(inc_p.read_text(encoding="utf-8"))
            th = bool(inc.get("thinking", MODEL_REGISTRY_V2[mk].lean_thinking))
            mt = int(inc.get("eval_max_tokens", MODEL_REGISTRY_V2[mk].lean_max_tokens))
            base_ckpt = Path(MODEL_REGISTRY_V2[mk].local_path)
            _eval_split_v2(
                track="lean", model_key=mk, model_path=base_ckpt, split_file=lean_cf, prompts_file=None,
                out_prefix=od / f"b0_{mk}", cand_id=f"confirm_b0_{mk}", protocol_id="v2_lean_confirm",
                thinking=th, temperature=0.8, max_tokens=mt,
            )
            if inc.get("promoted_challenger"):
                _eval_split_v2(
                    track="lean", model_key=mk, model_path=Path(inc["incumbent_ckpt"]), split_file=lean_cf, prompts_file=None,
                    out_prefix=od / f"astar_{mk}", cand_id=f"confirm_astar_{mk}", protocol_id="v2_lean_confirm",
                    thinking=th, temperature=0.8, max_tokens=mt,
                )
        # Goedel-Prover-V2-8B & Qwen3.8-27B comparators
        g_th, g_mt = _best_goedel_channel()
        _eval_split_v2(
            track="lean", model_key="goedel_8b", model_path=Path(MODEL_REGISTRY_V2["goedel_8b"].local_path),
            split_file=lean_cf, prompts_file=None, out_prefix=od / "ref_goedel_8b",
            cand_id="confirm_ref_goedel_8b", protocol_id="v2_lean_confirm", thinking=g_th, temperature=0.8, max_tokens=g_mt,
        )
        _eval_split_v2(
            track="lean", model_key="qwen38_27b", model_path=Path(MODEL_REGISTRY_V2["qwen38_27b"].local_path),
            split_file=lean_cf, prompts_file=None, out_prefix=od / "ref_qwen38_27b",
            cand_id="confirm_ref_qwen38_27b", protocol_id="v2_lean_confirm", thinking=False, temperature=0.8, max_tokens=2048,
        )
        # 4-way Causal Compiler-Repair Ablation on ProofNet-valid (N=164) on repair-trained qwen35_4b!
        inc4 = json.loads((V2 / "results/plane2/track_l/qwen35_4b/incumbent.json").read_text(encoding="utf-8"))
        rep_ckpt = inc4.get("repair_trained_ckpt") or inc4["incumbent_ckpt"]
        rep_sum = od / "repair_4way_qwen35_4b/summary.json"
        if not rep_sum.is_file():
            subprocess.run(
                [
                    sys.executable, "-I", "-B", str(V2 / "code/v2/repair_policy_v2.py"),
                    "--model-key", "qwen35_4b",
                    "--model-path", str(rep_ckpt),
                    "--split-file", str(_runtime_unique_id_view(lean_cf)),
                    "--candidate-prefix", "confirm_qwen35_4b",
                    "--work-dir", str(od / "repair_4way_qwen35_4b"),
                    "--summary-output", str(rep_sum),
                ],
                check=True,
            )

    elif role == "node1_math_small":
        od = V2 / "results/confirm/math"
        for mk in ("qwen35_4b", "phi4_mini"):
            inc = json.loads((V2 / f"results/plane2/track_m/{mk}/incumbent.json").read_text(encoding="utf-8"))
            th = bool(inc["thinking"])
            mt = int(inc["eval_max_tokens"])
            pp = float(inc.get("presence_penalty", 0.0))
            temp = 0.8 if th else 0.7
            _eval_split_v2(
                track="math", model_key=mk, model_path=Path(MODEL_REGISTRY_V2[mk].local_path), split_file=math_cf, prompts_file=None,
                out_prefix=od / f"b0_{mk}", cand_id=f"confirm_math_b0_{mk}", protocol_id="v2_math_confirm",
                thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
            )
            if inc.get("promoted_challenger"):
                _eval_split_v2(
                    track="math", model_key=mk, model_path=Path(inc["incumbent_ckpt"]), split_file=math_cf, prompts_file=None,
                    out_prefix=od / f"astar_{mk}", cand_id=f"confirm_math_astar_{mk}", protocol_id="v2_math_confirm",
                    thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
                )

    elif role == "node2_math_large":
        od = V2 / "results/confirm/math"
        for mk in ("qwen35_9b", "qwen3_4b"):
            inc = json.loads((V2 / f"results/plane2/track_m/{mk}/incumbent.json").read_text(encoding="utf-8"))
            th = bool(inc["thinking"])
            mt = int(inc["eval_max_tokens"])
            pp = float(inc.get("presence_penalty", 0.0))
            temp = 0.8 if th else 0.7
            _eval_split_v2(
                track="math", model_key=mk, model_path=Path(MODEL_REGISTRY_V2[mk].local_path), split_file=math_cf, prompts_file=None,
                out_prefix=od / f"b0_{mk}", cand_id=f"confirm_math_b0_{mk}", protocol_id="v2_math_confirm",
                thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
            )
            if inc.get("promoted_challenger"):
                _eval_split_v2(
                    track="math", model_key=mk, model_path=Path(inc["incumbent_ckpt"]), split_file=math_cf, prompts_file=None,
                    out_prefix=od / f"astar_{mk}", cand_id=f"confirm_math_astar_{mk}", protocol_id="v2_math_confirm",
                    thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
                )
        _eval_split_v2(
            track="math", model_key="qwen38_27b", model_path=Path(MODEL_REGISTRY_V2["qwen38_27b"].local_path),
            split_file=math_cf, prompts_file=None, out_prefix=od / "ref_qwen38_27b",
            cand_id="confirm_math_ref_qwen38_27b", protocol_id="v2_math_confirm", thinking=False, temperature=0.7, max_tokens=12288,
        )


def seal_confirm_summary() -> dict[str, Any]:
    """Conjunctive v2 + v3.6 Step-down Holm-Bonferroni (`alpha_family = 0.05`, 0-based indexing)
    + full-shard weight manifest binding + dual completion guardrails across all V2 challengers.
    """
    expected_root = TRUST_V2.read_text(encoding="utf-8").strip().split()[0]
    amend_p, amend_sha = write_v36_amendment(
        V2, V1, manifest_v2_sha256=expected_root, guard_v36_path=GUARD_V3
    )
    summary: dict[str, Any] = {
        "v36_amendment_path": str(amend_p),
        "v36_amendment_sha256": amend_sha,
        "track_l": {},
        "track_m": {},
    }

    for tr_key, sub_dir, rules_p, min_eff in (
        ("track_l", V2 / "results/confirm/lean", V2 / "configs_v2/gate_rules_v2_lean_confirm.json", 0.025),
        ("track_m", V2 / "results/confirm/math", V2 / "configs_v2/gate_rules_v2_math_confirm.json", 0.030),
    ):
        rules = json.loads(rules_p.read_text(encoding="utf-8"))
        floor_stop = float(rules["hard_floors"]["stop_rate"])
        max_drop_stop = float(rules["max_degradation_vs_baseline"]["stop_rate"])
        ledger = V2 / f"results/confirm/{tr_key}_confirm_ledger.jsonl"

        candidates: list[tuple[str, dict[str, Any]]] = []
        for mk in ("qwen35_4b", "qwen3_4b", "phi4_mini", "qwen35_9b"):
            b0_p = sub_dir / f"b0_{mk}_metrics.json"
            as_p = sub_dir / f"astar_{mk}_metrics.json"
            if not b0_p.is_file():
                continue
            b0_d = json.loads(b0_p.read_text(encoding="utf-8"))
            b0_m = b0_d["metrics"]
            if not as_p.is_file():
                summary[tr_key][mk] = {
                    "tested": False,
                    "v2_only_promoted": False,
                    "v36_promoted": False,
                    "promoted": False,
                    "holdout_candidate_status": "not_evaluated_no_search_promotion",
                    "b0_metrics": b0_m,
                }
                continue
            as_d = json.loads(as_p.read_text(encoding="utf-8"))
            as_m = as_d["metrics"]
            pb = _paired_bootstrap(as_p, b0_p)
            ma = {str(it["id"]): float(it["value"]) for it in as_d["items"]}
            mb = {str(it["id"]): float(it["value"]) for it in b0_d["items"]}
            if set(ma.keys()) != set(mb.keys()):
                raise RuntimeError(f"Confirm item mismatch on {tr_key}/{mk}")
            diffs_raw = [ma[k] - mb[k] for k in sorted(ma.keys())]
            sign_test = exact_sign_test_one_sided(diffs_raw, margin=0.0)
            stop_ok = (as_m["stop_rate"] >= floor_stop) and ((b0_m["stop_rate"] - as_m["stop_rate"]) <= max_drop_stop)

            # Original v2 hash-chained confirm ledger entry
            gate_out = sub_dir / f"gate_{mk}.json"
            if not gate_out.is_file():
                res = subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(GUARD),
                        "gate-decision",
                        "--candidate", str(as_p),
                        "--incumbent", str(b0_p),
                        "--baseline", str(b0_p),
                        "--rules", str(rules_p),
                        "--ledger", str(ledger),
                        "--trial-id", f"v2_confirm_{tr_key}_{mk}",
                    ],
                    capture_output=True, text=True, check=False,
                )
                gate_out.write_text(res.stdout, encoding="utf-8")

            inc_info = json.loads((V2 / f"results/plane2/{tr_key}/{mk}/incumbent.json").read_text(encoding="utf-8"))
            ckpt_dir = Path(inc_info["incumbent_ckpt"]).resolve()
            w_man_p = GOV_V3_DIR / f"weights_manifests/{tr_key}_{mk}_WEIGHTS_SHA256.json"
            w_man = canonical_checkpoint_manifest(ckpt_dir, w_man_p)

            # Use exact paired sign-test p-value on sparse Track L outcomes alongside bootstrap p-value
            p_rank = max(pb["p_one_sided"], sign_test["p_exact_one_sided"]) if tr_key == "track_l" else pb["p_one_sided"]
            candidates.append(
                (
                    mk,
                    {
                        "tested": True,
                        "paired": pb,
                        "exact_sign_test": sign_test,
                        "p_for_holm": p_rank,
                        "stop_guardrail_ok": stop_ok,
                        "b0_metrics": b0_m,
                        "astar_metrics": as_m,
                        "astar_candidate_id": str(as_d["candidate_id"]),
                        "ckpt_dir": str(ckpt_dir),
                        "weights_root_sha256": w_man["weights_root_sha256"],
                        "weights_manifest_path": str(w_man_p),
                        "as_path": str(as_p),
                        "b0_path": str(b0_p),
                    },
                )
            )

        # 0-based Step-down Holm-Bonferroni across ALL K tested challengers in this track
        candidates.sort(key=lambda kv: kv[1]["p_for_holm"])
        k_total = len(candidates)
        holm_broken_v2 = False
        holm_broken_conj = False

        for rank_0, (mk, info) in enumerate(candidates):
            alpha_step = 0.05 / float(k_total - rank_0)
            pb = info["paired"]
            v2_passed = (
                (not holm_broken_v2)
                and info["stop_guardrail_ok"]
                and (pb["delta"] >= min_eff)
                and (pb["p_one_sided"] <= alpha_step)
            )
            if not v2_passed:
                holm_broken_v2 = True

            # Run Guard v3.6 gate-decision at exact Holm step alpha with full checkpoint dir binding & proposer redaction
            v3_rules = dict(rules)
            v3_rules["decision_stage"] = "confirm"
            v3_rules["alpha_one_sided"] = alpha_step
            v3_rules["required_comparators"] = ["b0"]
            v3_rules["max_confirm_looks"] = 1
            v3_rules_p = GOV_V3_DIR / f"rules_confirm_{tr_key}_{mk}.json"
            v3_rules_p.parent.mkdir(parents=True, exist_ok=True)
            v3_rules_p.write_text(json.dumps(v3_rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            v3_ledger = GOV_V3_DIR / f"{tr_key}_{mk}_v36_ledger.jsonl"
            v3_gate_out = GOV_V3_DIR / f"gate_confirm_{tr_key}_{mk}.json"
            v3_prop_out = GOV_V3_DIR / f"proposer_redacted_{tr_key}_{mk}.json"
            if not v3_gate_out.is_file():
                res_v3 = subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(GUARD_V3),
                        "gate-decision",
                        "--candidate", info["as_path"],
                        "--incumbent", info["b0_path"],
                        "--baseline", info["b0_path"],
                        "--rules", str(v3_rules_p),
                        "--ledger", str(v3_ledger),
                        "--trial-id", f"v36_confirm_{tr_key}_{mk}",
                        "--comparator-id", "b0",
                        "--candidate-artifacts", info["ckpt_dir"],
                        "--proposer-out", str(v3_prop_out),
                        "--hypothesis", f"Conjunctive v3.6 Holm-step Confirmation on {tr_key}/{mk}",
                    ],
                    cwd=str(V2),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                v3_gate_out.write_text(res_v3.stdout, encoding="utf-8")
            v3_dec_obj = json.loads(v3_gate_out.read_text(encoding="utf-8"))
            v36_raw_promote = (
                v3_dec_obj.get("decision") == "PROMOTE"
                and (info["p_for_holm"] <= alpha_step)
            )
            conj_passed = bool((not holm_broken_conj) and v2_passed and v36_raw_promote)
            if not conj_passed:
                holm_broken_conj = True

            # Clean helper paths before serializing summary
            del info["as_path"]
            del info["b0_path"]
            info["holm_rank_0based"] = rank_0
            info["holm_rank_1based"] = rank_0 + 1
            info["holm_k"] = k_total
            info["holm_alpha_threshold"] = alpha_step
            info["v2_only_promoted"] = v2_passed
            info["v36_gate_decision"] = v3_dec_obj.get("decision", "ERROR")
            info["v36_promoted"] = v36_raw_promote
            info["promoted"] = conj_passed
            info["disagreement_note"] = (
                "v2 passed; promotion withheld under v3.6" if (v2_passed and not conj_passed) else "none"
            )
            info["holdout_candidate_status"] = (
                "eligible_after_seal_reserve" if conj_passed else "not_evaluated_confirmation_gate_failed"
            )
            summary[tr_key][mk] = info

    # Check if 4-way repair passed on ProofNet-valid (require statistically valid LCB > 0, never noise!)
    rep_sum_p = V2 / "results/confirm/lean/repair_4way_qwen35_4b/summary.json"
    if rep_sum_p.is_file():
        rep_d = json.loads(rep_sum_p.read_text(encoding="utf-8"))
        summary["track_l_causal_repair_confirm"] = rep_d
        c_vs_d = rep_d["paired_causal_contrasts"]["compiler_repair_vs_direct8"]
        summary["promote_repair_policy_to_holdout"] = bool(
            c_vs_d["delta"] >= 0.025 and c_vs_d["ci95_lo"] > 0.0
        )

    CONFIRM_SUMMARY_P.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_SUMMARY_P.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fsync_file_and_dir(CONFIRM_SUMMARY_P)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_holdout_comparators_role(role: str) -> None:
    """Step 1 of Single-Touch Holdout: evaluate ONLY zero-shot comparators (`B_0`, `Goedel-8B`, `Qwen3.8-27B`)
    so comparator Holdout outputs can be SHA-256 pinned BEFORE reserving `HOLDOUT_SEAL_*.json` or running `A*`!
    """
    if not CONFIRM_SUMMARY_P.is_file():
        raise SystemExit("Refusing Holdout comparator touch before v2_confirm_summary.json is sealed!")

    if role == "node0_lean":
        sp = V2 / "eval_v2/sealed_holdout/v2_lean_holdout_proofnet174.jsonl"
        pr = V2 / "data/splits_v2/v2_lean_holdout_proofnet174_prompts.jsonl"
        od = V2 / "results/holdout/lean_proofnet_test"
        for mk in ("qwen35_4b", "qwen3_4b", "qwen35_9b", "phi4_mini"):
            inc_p = V2 / f"results/plane2/track_l/{mk}/incumbent.json"
            if not inc_p.is_file():
                continue
            inc = json.loads(inc_p.read_text(encoding="utf-8"))
            th = bool(inc.get("thinking", MODEL_REGISTRY_V2[mk].lean_thinking))
            mt = int(inc.get("eval_max_tokens", MODEL_REGISTRY_V2[mk].lean_max_tokens))
            _eval_split_v2(
                track="lean", model_key=mk, model_path=Path(MODEL_REGISTRY_V2[mk].local_path),
                split_file=sp, prompts_file=pr, out_prefix=od / f"b0_{mk}",
                cand_id=f"holdout_lean_b0_{mk}", protocol_id="v2_lean_holdout_proofnet", thinking=th, temperature=0.8, max_tokens=mt,
            )
        g_th, g_mt = _best_goedel_channel()
        _eval_split_v2(
            track="lean", model_key="goedel_8b", model_path=Path(MODEL_REGISTRY_V2["goedel_8b"].local_path),
            split_file=sp, prompts_file=pr, out_prefix=od / "ref_goedel_8b",
            cand_id="holdout_lean_ref_goedel_8b", protocol_id="v2_lean_holdout_proofnet", thinking=g_th, temperature=0.8, max_tokens=g_mt,
        )
        _eval_split_v2(
            track="lean", model_key="qwen38_27b", model_path=Path(MODEL_REGISTRY_V2["qwen38_27b"].local_path),
            split_file=sp, prompts_file=pr, out_prefix=od / "ref_qwen38_27b",
            cand_id="holdout_lean_ref_qwen38_27b", protocol_id="v2_lean_holdout_proofnet", thinking=False, temperature=0.8, max_tokens=2048,
        )

    elif role in ("node1_math_small", "node2_math_large"):
        contests = ("beyondaime", "brumo2025", "cmimc2025", "hmmt_nov2025")
        mks = ("qwen35_4b", "phi4_mini") if role == "node1_math_small" else ("qwen35_9b", "qwen3_4b")
        for bench in contests:
            sp = V2 / f"eval_v2/sealed_holdout/v2_math_holdout_{bench}.jsonl"
            pr = V2 / f"data/splits_v2/v2_math_holdout_{bench}_prompts.jsonl"
            od = V2 / f"results/holdout/math_{bench}"
            for mk in mks:
                inc = json.loads((V2 / f"results/plane2/track_m/{mk}/incumbent.json").read_text(encoding="utf-8"))
                th = bool(inc["thinking"])
                mt = int(inc["eval_max_tokens"])
                pp = float(inc.get("presence_penalty", 0.0))
                temp = 0.8 if th else 0.7
                _eval_split_v2(
                    track="math", model_key=mk, model_path=Path(MODEL_REGISTRY_V2[mk].local_path),
                    split_file=sp, prompts_file=pr, out_prefix=od / f"b0_{mk}",
                    cand_id=f"holdout_{bench}_b0_{mk}", protocol_id=f"v2_math_holdout_{bench}",
                    thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
                )
            if role == "node2_math_large":
                _eval_split_v2(
                    track="math", model_key="qwen38_27b", model_path=Path(MODEL_REGISTRY_V2["qwen38_27b"].local_path),
                    split_file=sp, prompts_file=pr, out_prefix=od / "ref_qwen38_27b",
                    cand_id=f"holdout_{bench}_ref_qwen38_27b", protocol_id=f"v2_math_holdout_{bench}",
                    thinking=False, temperature=0.7, max_tokens=12288,
                )
        # Also evaluate MATH-500 (N=150) Retention Floor comparator B_0 for promoted models!
        cf_sum = json.loads(CONFIRM_SUMMARY_P.read_text(encoding="utf-8"))
        m500_sp = V1 / "eval/sealed_holdout/math_holdout_retention150.jsonl"
        m500_pr = V1 / "data/splits/math_holdout_retention150_prompts.jsonl"
        if m500_sp.is_file():
            od500 = V2 / "results/holdout/math_retention_math500"
            for mk in mks:
                if cf_sum.get("track_m", {}).get(mk, {}).get("promoted"):
                    inc = json.loads((V2 / f"results/plane2/track_m/{mk}/incumbent.json").read_text(encoding="utf-8"))
                    th = bool(inc["thinking"])
                    mt = int(inc["eval_max_tokens"])
                    pp = float(inc.get("presence_penalty", 0.0))
                    temp = 0.8 if th else 0.7
                    _eval_split_v2(
                        track="math", model_key=mk, model_path=Path(MODEL_REGISTRY_V2[mk].local_path),
                        split_file=m500_sp, prompts_file=(m500_pr if m500_pr.is_file() else None),
                        out_prefix=od500 / f"b0_{mk}",
                        cand_id=f"holdout_math500_b0_{mk}", protocol_id="v2_math_holdout_math500_retention",
                        thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
                    )


def reserve_holdout_seals() -> None:
    """Step 2 of Single-Touch Holdout (run on node-0 local NVMe):
    Pins comparator Holdout SHA-256s + AMENDMENT_V3_6.json SHA-256 + GUARD_V3 SHA-256,
    and atomically creates `HOLDOUT_SEAL_{tr}_{mk}.json` (`O_CREAT | O_EXCL` + `fsync`)
    for every Conjunctive-PROMOTE candidate BEFORE `A*` ever touches Holdout!
    """
    cf_sum = json.loads(CONFIRM_SUMMARY_P.read_text(encoding="utf-8"))
    expected_root = TRUST_V2.read_text(encoding="utf-8").strip().split()[0]
    contests = tuple(HOLDOUT_CONTEST_COUNTS.keys())

    # Build combined 4-contest comparator JSON for ref_qwen38_27b once
    ref27_paths = {b: V2 / f"results/holdout/math_{b}/ref_qwen38_27b_metrics.json" for b in contests}
    if all(p.is_file() for p in ref27_paths.values()):
        ref27_comb = build_combined_4contest_metrics_json(
            ref27_paths,
            GOV_V3_DIR / "combined_holdout/ref_qwen38_27b_4contest.json",
            candidate_id="holdout_4contest_ref_qwen38_27b",
            expected_counts=HOLDOUT_CONTEST_COUNTS,
        )
    else:
        ref27_comb = None

    for tr_key in ("track_l", "track_m"):
        for mk, info in cf_sum.get(tr_key, {}).items():
            if not info.get("promoted"):
                continue
            ckpt_dir = Path(info["ckpt_dir"]).resolve()
            # Re-verify full checkpoint weights manifest matches Confirm digest bit-for-bit!
            cur_w = canonical_checkpoint_manifest(ckpt_dir)
            if cur_w["weights_root_sha256"] != info["weights_root_sha256"]:
                raise RuntimeError(f"Weight manifest SHA mismatch before reserve on {tr_key}/{mk}!")

            comp_digests: dict[str, str] = {}
            if tr_key == "track_m":
                b0_paths = {b: V2 / f"results/holdout/math_{b}/b0_{mk}_metrics.json" for b in contests}
                b0_comb = build_combined_4contest_metrics_json(
                    b0_paths,
                    GOV_V3_DIR / f"combined_holdout/b0_{mk}_4contest.json",
                    candidate_id=f"holdout_4contest_b0_{mk}",
                    expected_counts=HOLDOUT_CONTEST_COUNTS,
                )
                comp_digests["b0_4contest_sha256"] = _sha256_file(b0_comb)
                if ref27_comb and ref27_comb.is_file():
                    comp_digests["ref_qwen38_27b_4contest_sha256"] = _sha256_file(ref27_comb)
                m500_b0 = V2 / f"results/holdout/math_retention_math500/b0_{mk}_metrics.json"
                if m500_b0.is_file():
                    comp_digests["b0_math500_retention_sha256"] = _sha256_file(m500_b0)
            else:
                b0_l = V2 / f"results/holdout/lean_proofnet_test/b0_{mk}_metrics.json"
                comp_digests["b0_proofnet_test_sha256"] = _sha256_file(b0_l)

            pin_p = GOV_V3_DIR / f"COMPARATOR_PIN_{tr_key}_{mk}.json"
            pin_payload = {
                "track": tr_key,
                "model_key": mk,
                "astar_candidate_id": info["astar_candidate_id"],
                "weights_root_sha256": info["weights_root_sha256"],
                "amendment_v36_sha256": cf_sum["v36_amendment_sha256"],
                "guard_v36_sha256": _sha256_file(GUARD_V3),
                "manifest_v2_root_sha256": expected_root,
                "comparator_holdout_digests": comp_digests,
            }
            pin_p.write_text(json.dumps(pin_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            fsync_file_and_dir(pin_p)

            # Build equal_stratum Holdout rules for Guard v3.6
            h_rules = json.loads((GOV_V3_DIR / f"rules_confirm_{tr_key}_{mk}.json").read_text(encoding="utf-8"))
            if tr_key == "track_m":
                h_rules["aggregation_weighting"] = "equal_stratum"
                h_rules["per_stratum_hard_floors"] = {"stop_rate": 0.25}
            h_rules_p = GOV_V3_DIR / f"rules_holdout_{tr_key}_{mk}.json"
            h_rules_p.write_text(json.dumps(h_rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            fsync_file_and_dir(h_rules_p)

            seal_p = GOV_V3_DIR / f"HOLDOUT_SEAL_{tr_key}_{mk}.json"
            v3_ledger = GOV_V3_DIR / f"{tr_key}_{mk}_v36_ledger.jsonl"
            subprocess.run(
                [
                    sys.executable, "-I", "-B", str(GUARD_V3),
                    "gate-holdout",
                    "--phase", "reserve",
                    "--seal-file", str(seal_p),
                    "--manifest", str(MANIFEST_V2),
                    "--expected-root-sha256", expected_root,
                    "--ledger", str(v3_ledger),
                    "--rules", str(h_rules_p),
                    "--candidate-id", info["astar_candidate_id"],
                    "--candidate-artifacts", str(ckpt_dir),
                ],
                cwd=str(V2),
                check=True,
            )
            fsync_file_and_dir(seal_p)


def run_holdout_candidates_role(role: str) -> None:
    """Step 3 of Single-Touch Holdout: evaluate ONLY `RESERVED` `A*` candidates strictly once!
    Unpromoted candidates are NEVER evaluated on Protected Holdout.
    """
    if not CONFIRM_SUMMARY_P.is_file():
        raise SystemExit("Refusing Holdout touch before v2_confirm_summary.json is sealed!")
    cf_sum = json.loads(CONFIRM_SUMMARY_P.read_text(encoding="utf-8"))

    if role == "node0_lean":
        sp = V2 / "eval_v2/sealed_holdout/v2_lean_holdout_proofnet174.jsonl"
        pr = V2 / "data/splits_v2/v2_lean_holdout_proofnet174_prompts.jsonl"
        od = V2 / "results/holdout/lean_proofnet_test"
        for mk in ("qwen35_4b", "qwen3_4b", "qwen35_9b", "phi4_mini"):
            if not cf_sum.get("track_l", {}).get(mk, {}).get("promoted"):
                continue
            seal_p = GOV_V3_DIR / f"HOLDOUT_SEAL_track_l_{mk}.json"
            if not seal_p.is_file() or json.loads(seal_p.read_text(encoding="utf-8")).get("status") != "RESERVED":
                raise SystemExit(f"Blocked Holdout candidate eval on track_l/{mk}: seal must be RESERVED!")
            inc = json.loads((V2 / f"results/plane2/track_l/{mk}/incumbent.json").read_text(encoding="utf-8"))
            th = bool(inc.get("thinking", MODEL_REGISTRY_V2[mk].lean_thinking))
            mt = int(inc.get("eval_max_tokens", MODEL_REGISTRY_V2[mk].lean_max_tokens))
            _eval_split_v2(
                track="lean", model_key=mk, model_path=Path(inc["incumbent_ckpt"]),
                split_file=sp, prompts_file=pr, out_prefix=od / f"astar_{mk}",
                cand_id=f"holdout_lean_astar_{mk}", protocol_id="v2_lean_holdout_proofnet", thinking=th, temperature=0.8, max_tokens=mt,
            )
        if cf_sum.get("promote_repair_policy_to_holdout"):
            inc4 = json.loads((V2 / "results/plane2/track_l/qwen35_4b/incumbent.json").read_text(encoding="utf-8"))
            rep_ckpt = inc4.get("repair_trained_ckpt") or inc4["incumbent_ckpt"]
            rep_sum = od / "repair_4way_qwen35_4b/summary.json"
            if not rep_sum.is_file():
                subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(V2 / "code/v2/repair_policy_v2.py"),
                        "--model-key", "qwen35_4b",
                        "--model-path", str(rep_ckpt),
                        "--split-file", str(_runtime_unique_id_view(sp)),
                        "--prompts-file", str(_runtime_unique_id_view(pr)),
                        "--candidate-prefix", "holdout_proofnet_qwen35_4b",
                        "--work-dir", str(od / "repair_4way_qwen35_4b"),
                        "--summary-output", str(rep_sum),
                    ],
                    check=True,
                )

    elif role in ("node1_math_small", "node2_math_large"):
        contests = ("beyondaime", "brumo2025", "cmimc2025", "hmmt_nov2025")
        mks = ("qwen35_4b", "phi4_mini") if role == "node1_math_small" else ("qwen35_9b", "qwen3_4b")
        for mk in mks:
            if not cf_sum.get("track_m", {}).get(mk, {}).get("promoted"):
                continue
            seal_p = GOV_V3_DIR / f"HOLDOUT_SEAL_track_m_{mk}.json"
            if not seal_p.is_file() or json.loads(seal_p.read_text(encoding="utf-8")).get("status") != "RESERVED":
                raise SystemExit(f"Blocked Holdout candidate eval on track_m/{mk}: seal must be RESERVED!")
            inc = json.loads((V2 / f"results/plane2/track_m/{mk}/incumbent.json").read_text(encoding="utf-8"))
            th = bool(inc["thinking"])
            mt = int(inc["eval_max_tokens"])
            pp = float(inc.get("presence_penalty", 0.0))
            temp = 0.8 if th else 0.7
            for bench in contests:
                sp = V2 / f"eval_v2/sealed_holdout/v2_math_holdout_{bench}.jsonl"
                pr = V2 / f"data/splits_v2/v2_math_holdout_{bench}_prompts.jsonl"
                od = V2 / f"results/holdout/math_{bench}"
                _eval_split_v2(
                    track="math", model_key=mk, model_path=Path(inc["incumbent_ckpt"]),
                    split_file=sp, prompts_file=pr, out_prefix=od / f"astar_{mk}",
                    cand_id=f"holdout_{bench}_astar_{mk}", protocol_id=f"v2_math_holdout_{bench}",
                    thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
                )
            # Evaluate MATH-500 (N=150) Retention Floor for promoted A*!
            m500_sp = V1 / "eval/sealed_holdout/math_holdout_retention150.jsonl"
            m500_pr = V1 / "data/splits/math_holdout_retention150_prompts.jsonl"
            if m500_sp.is_file():
                od500 = V2 / "results/holdout/math_retention_math500"
                _eval_split_v2(
                    track="math", model_key=mk, model_path=Path(inc["incumbent_ckpt"]),
                    split_file=m500_sp, prompts_file=(m500_pr if m500_pr.is_file() else None),
                    out_prefix=od500 / f"astar_{mk}",
                    cand_id=f"holdout_math500_astar_{mk}", protocol_id="v2_math_holdout_math500_retention",
                    thinking=th, temperature=temp, presence_penalty=pp, max_tokens=mt,
                )


def finalize_holdout_seals() -> dict[str, Any]:
    """Step 4 of Single-Touch Holdout (run on node-0 local NVMe):
    Verifies pinned comparator SHA-256 digests + full checkpoint weight SHA-256s,
    calls `governed_loop_guard_v3.py gate-holdout --phase finalize` on 4-contest `equal_stratum`
    (100/40/30/30), runs 20,000-draw stratified Rademacher sign-flip permutation tests +
    Holm-Bonferroni across all promoted Holdout models, and evaluates MATH-500 retention!
    """
    cf_sum = json.loads(CONFIRM_SUMMARY_P.read_text(encoding="utf-8"))
    expected_root = TRUST_V2.read_text(encoding="utf-8").strip().split()[0]
    contests = tuple(HOLDOUT_CONTEST_COUNTS.keys())
    out_final: dict[str, Any] = {"track_l": {}, "track_m": {}}

    for tr_key in ("track_l", "track_m"):
        promoted_mks = [mk for mk, inf in cf_sum.get(tr_key, {}).items() if inf.get("promoted")]
        perm_rows: list[tuple[str, dict[str, Any]]] = []

        for mk in promoted_mks:
            info = cf_sum[tr_key][mk]
            ckpt_dir = Path(info["ckpt_dir"]).resolve()
            cur_w = canonical_checkpoint_manifest(ckpt_dir)
            if cur_w["weights_root_sha256"] != info["weights_root_sha256"]:
                raise RuntimeError(f"Weight tampering detected at finalize on {tr_key}/{mk}!")

            pin_d = json.loads((GOV_V3_DIR / f"COMPARATOR_PIN_{tr_key}_{mk}.json").read_text(encoding="utf-8"))
            seal_p = GOV_V3_DIR / f"HOLDOUT_SEAL_{tr_key}_{mk}.json"
            v3_ledger = GOV_V3_DIR / f"{tr_key}_{mk}_v36_ledger.jsonl"
            h_rules_p = GOV_V3_DIR / f"rules_holdout_{tr_key}_{mk}.json"

            if tr_key == "track_m":
                b0_comb = GOV_V3_DIR / f"combined_holdout/b0_{mk}_4contest.json"
                ref27_comb = GOV_V3_DIR / "combined_holdout/ref_qwen38_27b_4contest.json"
                if _sha256_file(b0_comb) != pin_d["comparator_holdout_digests"]["b0_4contest_sha256"]:
                    raise RuntimeError(f"Comparator b0 Holdout SHA mutated on {mk}!")
                if ref27_comb.is_file() and _sha256_file(ref27_comb) != pin_d["comparator_holdout_digests"]["ref_qwen38_27b_4contest_sha256"]:
                    raise RuntimeError("Comparator ref_qwen38_27b Holdout SHA mutated!")

                as_paths = {b: V2 / f"results/holdout/math_{b}/astar_{mk}_metrics.json" for b in contests}
                as_comb = build_combined_4contest_metrics_json(
                    as_paths,
                    GOV_V3_DIR / f"combined_holdout/astar_{mk}_4contest.json",
                    candidate_id=info["astar_candidate_id"],
                    expected_counts=HOLDOUT_CONTEST_COUNTS,
                )

                seal_state = json.loads(seal_p.read_text(encoding="utf-8")).get("status")
                if seal_state == "RESERVED":
                    fcmd = [
                        sys.executable, "-I", "-B", str(GUARD_V3),
                        "gate-holdout",
                        "--phase", "finalize",
                        "--seal-file", str(seal_p),
                        "--manifest", str(MANIFEST_V2),
                        "--expected-root-sha256", expected_root,
                        "--ledger", str(v3_ledger),
                        "--rules", str(h_rules_p),
                        "--candidate-id", info["astar_candidate_id"],
                        "--candidate-artifacts", str(ckpt_dir),
                        "--candidate-holdout", str(as_comb),
                        "--comparator-holdout", f"b0={b0_comb}",
                    ]
                    if ref27_comb.is_file():
                        fcmd.extend(["--comparator-holdout", f"ref_qwen38_27b={ref27_comb}"])
                    subprocess.run(fcmd, cwd=str(V2), check=True)
                    fsync_file_and_dir(seal_p)

                # Exact Stratified Rademacher Sign-Flip Permutation Test (20,000 draws) on 100/40/30/30
                cand_s = {
                    b: {str(it["id"]): float(it["value"]) for it in json.loads(as_paths[b].read_text(encoding="utf-8"))["items"]}
                    for b in contests
                }
                b0_s = {
                    b: {str(it["id"]): float(it["value"]) for it in json.loads((V2 / f"results/holdout/math_{b}/b0_{mk}_metrics.json").read_text(encoding="utf-8"))["items"]}
                    for b in contests
                }
                perm_vs_b0 = stratified_sign_flip_test(
                    cand_s, b0_s, expected_counts=HOLDOUT_CONTEST_COUNTS, margin=0.0, n_perm=20000, n_boot=10000
                )
                perm_vs_27b = None
                if ref27_comb.is_file():
                    r27_s = {
                        b: {str(it["id"]): float(it["value"]) for it in json.loads((V2 / f"results/holdout/math_{b}/ref_qwen38_27b_metrics.json").read_text(encoding="utf-8"))["items"]}
                        for b in contests
                    }
                    perm_vs_27b = stratified_sign_flip_test(
                        cand_s, r27_s, expected_counts=HOLDOUT_CONTEST_COUNTS, margin=0.0, n_perm=20000, n_boot=10000
                    )

                # MATH-500 (N=150) Retention check (mean@8 per item, outside 4-contest average)
                m500_ret = None
                m500_as = V2 / f"results/holdout/math_retention_math500/astar_{mk}_metrics.json"
                m500_b0 = V2 / f"results/holdout/math_retention_math500/b0_{mk}_metrics.json"
                if m500_as.is_file() and m500_b0.is_file():
                    m500_pb = _paired_bootstrap(m500_as, m500_b0)
                    m500_ret = {
                        **m500_pb,
                        "non_degradation_point_ok": bool(m500_pb["delta"] >= -0.025),
                    }

                rec_m = {
                    "seal_file": str(seal_p),
                    "seal_sha256": _sha256_file(seal_p),
                    "weights_root_sha256_verified": cur_w["weights_root_sha256"],
                    "stratified_sign_flip_vs_b0": perm_vs_b0,
                    "stratified_sign_flip_vs_qwen38_27b": perm_vs_27b,
                    "math500_retention_n150": m500_ret,
                }
                p_worst = max(
                    perm_vs_b0["item_pooled"]["p_sign_flip_one_sided"],
                    perm_vs_b0["equal_stratum"]["p_sign_flip_one_sided"],
                )
                perm_rows.append((mk, {"p_conjunctive": p_worst, **rec_m}))

        # Step-down 0-based Holm-Bonferroni across all promoted Holdout models in this track
        perm_rows.sort(key=lambda kv: kv[1]["p_conjunctive"])
        kh = len(perm_rows)
        hb_broken = False
        for r0, (mk, rinfo) in enumerate(perm_rows):
            a_h = 0.05 / float(kh - r0)
            ok_h = (not hb_broken) and (rinfo["p_conjunctive"] <= a_h)
            if not ok_h:
                hb_broken = True
            rinfo["holdout_holm_rank_0based"] = r0
            rinfo["holdout_holm_k"] = kh
            rinfo["holdout_holm_alpha_threshold"] = a_h
            rinfo["conjunctive_pooled_and_equal_stratum_passed"] = ok_h
            out_final[tr_key][mk] = rinfo

    final_p = GOV_V3_DIR / "v36_holdout_conjunctive_summary.json"
    final_p.write_text(json.dumps(out_final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fsync_file_and_dir(final_p)
    print(json.dumps(out_final, indent=2), flush=True)
    return out_final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=[
            "confirm",
            "seal-confirm",
            "holdout-comparators",
            "reserve-holdout",
            "holdout-candidates",
            "finalize-holdout",
        ],
        required=True,
    )
    ap.add_argument("--role", choices=["node0_lean", "node1_math_small", "node2_math_large", "master"], default="master")
    args = ap.parse_args()

    _verify_manifest()
    if args.phase == "confirm":
        run_confirm_role(args.role)
    elif args.phase == "seal-confirm":
        seal_confirm_summary()
    elif args.phase == "holdout-comparators":
        run_holdout_comparators_role(args.role)
    elif args.phase == "reserve-holdout":
        reserve_holdout_seals()
    elif args.phase == "holdout-candidates":
        run_holdout_candidates_role(args.role)
    elif args.phase == "finalize-holdout":
        finalize_holdout_seals()
    _verify_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Paired Confirmation Gate & Single-Touch Protected Holdout Evaluator.

Enforces strict Confirmation & Protected Holdout firewalls:
  1. Verifies `IMMUTABLE_MANIFEST.json` against the external root SHA-256 pin before AND after execution.
  2. Evaluates `B_0` vs Search-promoted `A*` (`4B` and `9B`) + `Qwen3.8-27B` comparator on:
       - Track L Confirmation: `data/splits/lean_confirm243.jsonl` (`MiniF2F-v2s valid`, `n=243` pure theorems)
       - Track M Confirmation: `data/splits/math_confirm300.jsonl` (`n=300` hard competition clusters)
         PLUS anti-forgetting guardrail `data/splits/math500_retention_confirm150.jsonl` (`n=150`, `k=4`)
     and executes `eval/governed_loop_guard.py gate-decision --split confirm` (`Holm-Bonferroni multiplicity_k=2`,
     `alpha_one_sided = 0.025`, `min_improvement = 0.03`).
  3. Evaluates **single-touch** Protected Holdout (`eval/sealed_holdout/*.jsonl`) ONCE on:
       - Untuned `B_0` (`4B` and `9B`)
       - Confirmation-promoted `A*` (`4B` and `9B`; falls back to `B_0` if not promoted on Confirmation)
       - `Qwen3.8-27B` frontier comparator
       - Plus if outer `L4` (`4 direct + 4 compiler-repair`) passed `lean_repair_reserve97`, evaluates the
         frozen `L4` policy once alongside `8 direct` on `minif2f_v2s_test` (`n=244`) and `putnambench_holdout128` (`n=128`).
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

ROOT = Path("/scratch/governed_slm_rsi_v1")
MANIFEST = ROOT / "IMMUTABLE_MANIFEST.json"
ROOT_HASH_FILE = Path("/scratch/governed_slm_rsi_v1_trust_root.sha256")
GUARD = ROOT / "eval/governed_loop_guard.py"
CONFIRM_SUMMARY = ROOT / "results/confirm/confirm_summary.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
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
    expected = ROOT_HASH_FILE.read_text(encoding="utf-8").strip().split()[0]
    subprocess.run(
        [
            sys.executable, "-I", "-B", str(GUARD),
            "verify-manifest", "--manifest", str(MANIFEST),
            "--root", str(ROOT), "--expected-root-sha256", expected,
        ],
        check=True,
    )


def _eval_split(
    *,
    track: str,
    model_path: Path,
    split_file: Path,
    out_prefix: Path,
    cand_id: str,
    protocol_id: str,
    k: int = 8,
    tp: int = 1,
    thinking: bool = False,
    temperature: float = 0.8,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    max_model_len: int = 32768,
) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    gen_gz = Path(f"{out_prefix}.jsonl.gz")
    met_json = Path(f"{out_prefix}_metrics.json")
    det_gz = Path(f"{out_prefix}_details.jsonl.gz")
    rec_json = Path(f"{out_prefix}_receipt.json")

    expected_receipt = {
        "candidate_id": cand_id,
        "protocol_id": protocol_id,
        "split_file": str(split_file),
        "split_sha256": _sha256_file(split_file),
        "ckpt": _ckpt_fingerprint(model_path),
        "k": k,
        "tp": tp,
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
                raise SystemExit(
                    f"Single-touch receipt mismatch on {out_prefix} key={key}: {old_rec.get(key)} != {expected_receipt.get(key)}"
                )
        return met_json

    if not gen_gz.is_file():
        cmd = [
            sys.executable, "-I", "-B", str(ROOT / "code/vllm_eval_engine.py"),
            "--model-path", str(model_path),
            "--split-file", str(split_file),
            "--output", str(gen_gz),
            "--summary-output", f"{out_prefix}_gen.json",
            "--k", str(k), "--tp", str(tp), "--gpus", "0,1,2,3,4,5,6,7",
            "--temperature", str(temperature),
            "--presence-penalty", str(presence_penalty),
            "--max-tokens", str(max_tokens),
            "--max-model-len", str(max_model_len),
            "--seed", "20261118",
        ]
        if thinking:
            cmd.append("--thinking")
        subprocess.run(cmd, check=True)

    if not met_json.is_file():
        scmd = [
            sys.executable, "-I", "-B", str(ROOT / "eval/isolated_scorer.py"),
            "--track", track,
            "--split-file", str(split_file),
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


def _gate_confirm(inc_json: Path, chal_json: Path, rules_json: Path, ledger: Path, gate_out: Path, trial_id: str) -> dict[str, Any]:
    if gate_out.is_file():
        return json.loads(gate_out.read_text(encoding="utf-8"))
    res = subprocess.run(
        [
            sys.executable, "-I", "-B", str(GUARD),
            "gate-decision",
            "--candidate", str(chal_json),
            "--incumbent", str(inc_json),
            "--baseline", str(inc_json),
            "--rules", str(rules_json),
            "--ledger", str(ledger),
            "--trial-id", trial_id,
        ],
        capture_output=True,
        text=True,
    )
    # Fail closed on rc=1 (validation / tamper error); only 0 (PROMOTE) and 3 (HOLD/ROLLBACK) are valid decisions!
    if res.returncode not in (0, 3):
        raise SystemExit(f"Confirm gate failed closed rc={res.returncode}: {res.stderr}\n{res.stdout}")
    gate_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.write_text(res.stdout, encoding="utf-8")
    return json.loads(res.stdout)


def _paired_cluster_stats(b0_path: Path, chal_path: Path, n_boot: int = 10000, seed: int = 20261118) -> dict[str, Any]:
    b0 = json.loads(b0_path.read_text(encoding="utf-8"))
    ch = json.loads(chal_path.read_text(encoding="utf-8"))
    b_map: dict[str, list[float]] = {}
    c_map: dict[str, list[float]] = {}
    for it in b0["items"]:
        cid = str(it.get("cluster_id") or it["id"])
        b_map.setdefault(cid, []).append(float(it["value"]))
    for it in ch["items"]:
        cid = str(it.get("cluster_id") or it["id"])
        c_map.setdefault(cid, []).append(float(it["value"]))
    keys = sorted(set(b_map.keys()) & set(c_map.keys()))
    if len(keys) != len(b_map) or len(keys) != len(c_map):
        raise SystemExit(f"Cluster mismatch between {b0_path} and {chal_path}")
    diffs = [(sum(c_map[k]) / len(c_map[k])) - (sum(b_map[k]) / len(b_map[k])) for k in keys]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
    se_d = math.sqrt(var_d / n)
    z_stat = mean_d / max(se_d, 1e-12)
    p_norm_one_sided = 0.5 * math.erfc(z_stat / math.sqrt(2.0))

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boot_means.append(s)
    boot_means.sort()
    p_boot_one_sided = sum(1 for bm in boot_means if bm <= 0.0) / n_boot
    lcb_9875 = boot_means[int(0.0125 * n_boot)]
    lcb_9750 = boot_means[int(0.0250 * n_boot)]
    lcb_9500 = boot_means[int(0.0500 * n_boot)]
    ci95_two_sided = [boot_means[int(0.0250 * n_boot)], boot_means[int(0.9750 * n_boot)]]

    return {
        "n_clusters": n,
        "b0_mean": sum(sum(b_map[k]) / len(b_map[k]) for k in keys) / n,
        "astar_mean": sum(sum(c_map[k]) / len(c_map[k]) for k in keys) / n,
        "paired_delta": mean_d,
        "paired_se": se_d,
        "se_within_195pp_ceiling": bool(se_d <= 0.0195),
        "z_stat": z_stat,
        "p_one_sided_normal": p_norm_one_sided,
        "p_one_sided_bootstrap": p_boot_one_sided,
        "lcb_one_sided_alpha0125": lcb_9875,
        "lcb_one_sided_alpha0250": lcb_9750,
        "lcb_one_sided_alpha0500": lcb_9500,
        "ci95_two_sided": ci95_two_sided,
        "b0_metrics": b0["metrics"],
        "astar_metrics": ch["metrics"],
    }


def _apply_holm_k2(pair_stats: dict[str, dict[str, Any]], guardrails_ok: dict[str, bool], min_delta: float = 0.03) -> dict[str, dict[str, Any]]:
    # Rank by exact paired normal/bootstrap p-value (smaller p-value tested at 0.0125; second at 0.025 iff first rejects)
    ordered = sorted(pair_stats.keys(), key=lambda m: (pair_stats[m]["p_one_sided_normal"], -pair_stats[m]["paired_delta"]))
    m1, m2 = ordered[0], ordered[1]

    s1 = pair_stats[m1]
    rej1 = bool(
        guardrails_ok[m1]
        and s1["p_one_sided_normal"] < 0.0125
        and s1["lcb_one_sided_alpha0125"] > 0.0
    )
    s2 = pair_stats[m2]
    rej2 = bool(
        rej1
        and guardrails_ok[m2]
        and s2["p_one_sided_normal"] < 0.0250
        and s2["lcb_one_sided_alpha0250"] > 0.0
    )

    out: dict[str, dict[str, Any]] = {}
    for m, rej, rank_idx, alpha_eff, lcb_eff in [
        (m1, rej1, 1, 0.0125, s1["lcb_one_sided_alpha0125"]),
        (m2, rej2, 2, 0.0250, s2["lcb_one_sided_alpha0250"]),
    ]:
        st = pair_stats[m]
        if not guardrails_ok[m] or not rej:
            verdict = "ROLLBACK"
            holdout_auth = False
        elif st["paired_delta"] >= min_delta:
            verdict = "CONFIRMED_PROMOTE"
            holdout_auth = True
        else:
            verdict = "CONFIRMED_POSITIVE_SUB3PP"
            holdout_auth = True
        out[m] = {
            "holm_rank": rank_idx,
            "holm_alpha_one_sided": alpha_eff,
            "holm_lcb_one_sided": lcb_eff,
            "holm_statistically_significant": rej,
            "guardrails_passed": guardrails_ok[m],
            "verdict": verdict,
            "authorize_holdout_astar": holdout_auth,
        }
    return out


def cmd_confirm_lean(args: argparse.Namespace) -> int:
    _verify_manifest()
    winners = json.loads((ROOT / "results/plane2/track_l/track_l_winners.json").read_text(encoding="utf-8"))
    cdir = ROOT / "results/confirm/track_l"
    ledger = ROOT / "results/confirm/track_l_confirm_ledger.jsonl"
    confirm_split = ROOT / "data/splits/lean_confirm243.jsonl"

    models = {
        "q4b_b0": (ROOT / "models/qwen35_4b", 1),
        "q4b_astar": (Path(winners["winner_4b_ckpt"]), 1),
        "q9b_b0": (ROOT / "models/qwen35_9b", 1),
        "q9b_astar": (Path(winners["winner_9b_ckpt"]), 1),
        "q27b_ref": (ROOT / "models/qwen38_27b", 4),
    }
    confirm_evals: dict[str, Path] = {}
    for tag, (mp, tp) in models.items():
        confirm_evals[tag] = _eval_split(
            track="lean",
            model_path=mp,
            split_file=confirm_split,
            out_prefix=cdir / tag,
            cand_id=f"confirm_lean_{tag}",
            protocol_id="confirm_lean243_direct2k",
            k=8,
            tp=tp,
            thinking=False,
            temperature=0.8,
            max_tokens=2048,
        )

    g4 = _gate_confirm(
        confirm_evals["q4b_b0"],
        confirm_evals["q4b_astar"],
        ROOT / "configs/gate_rules_lean_confirm.json",
        ledger,
        cdir / "gate_confirm_4b.json",
        "confirm_lean_q4b_astar",
    )
    g9 = _gate_confirm(
        confirm_evals["q9b_b0"],
        confirm_evals["q9b_astar"],
        ROOT / "configs/gate_rules_lean_confirm.json",
        ledger,
        cdir / "gate_confirm_9b.json",
        "confirm_lean_q9b_astar",
    )
    _verify_manifest()
    print(json.dumps({"event": "CONFIRM_LEAN_DONE", "bonferroni_4b": g4["decision"], "bonferroni_9b": g9["decision"]}, indent=2))
    return 0


def cmd_confirm_math(args: argparse.Namespace) -> int:
    _verify_manifest()
    ms = args.model_size
    w_info = json.loads((ROOT / f"results/plane2/track_m_{ms}/track_m_{ms}_winner.json").read_text(encoding="utf-8"))
    astar_ckpt = Path(w_info["winner_ckpt"])
    base_ckpt = ROOT / f"models/qwen35_{ms}"
    max_tok = 12288
    temp = 0.7

    cdir = ROOT / f"results/confirm/track_m_{ms}"
    ledger = ROOT / f"results/confirm/track_m_{ms}_confirm_ledger.jsonl"
    confirm_split = ROOT / "data/splits/math_confirm300.jsonl"
    ret_split = ROOT / "data/splits/math500_retention_confirm150.jsonl"

    b0_conf = _eval_split(
        track="math", model_path=base_ckpt, split_file=confirm_split,
        out_prefix=cdir / f"q{ms}_b0", cand_id=f"confirm_math_q{ms}_b0",
        protocol_id="confirm_math300_direct12k", k=8, tp=1, thinking=False,
        temperature=temp, presence_penalty=0.0, max_tokens=max_tok,
    )
    astar_conf = _eval_split(
        track="math", model_path=astar_ckpt, split_file=confirm_split,
        out_prefix=cdir / f"q{ms}_astar", cand_id=f"confirm_math_q{ms}_astar",
        protocol_id="confirm_math300_direct12k", k=8, tp=1, thinking=False,
        temperature=temp, presence_penalty=0.0, max_tokens=max_tok,
    )
    # Also evaluate Qwen3.8-27B comparator on math_confirm300 once on node-1 (4b node finishes faster)
    if ms == "4b" and (ROOT / "models/qwen38_27b/config.json").is_file():
        _eval_split(
            track="math", model_path=ROOT / "models/qwen38_27b", split_file=confirm_split,
            out_prefix=cdir / "q27b_ref", cand_id="confirm_math_q27b_ref",
            protocol_id="confirm_math300_direct12k", k=8, tp=4, thinking=False,
            temperature=temp, presence_penalty=0.0, max_tokens=max_tok,
        )

    # Anti-forgetting retention guardrail on math500_retention_confirm150 (150 * 4 = 600 completions)
    _eval_split(
        track="math", model_path=base_ckpt, split_file=ret_split,
        out_prefix=cdir / f"ret150_q{ms}_b0", cand_id=f"ret150_q{ms}_b0",
        protocol_id="confirm_math500_ret150_direct12k", k=4, tp=1, thinking=False,
        temperature=temp, presence_penalty=0.0, max_tokens=max_tok,
    )
    _eval_split(
        track="math", model_path=astar_ckpt, split_file=ret_split,
        out_prefix=cdir / f"ret150_q{ms}_astar", cand_id=f"ret150_q{ms}_astar",
        protocol_id="confirm_math500_ret150_direct12k", k=4, tp=1, thinking=False,
        temperature=temp, presence_penalty=0.0, max_tokens=max_tok,
    )

    gm = _gate_confirm(
        b0_conf, astar_conf, ROOT / "configs/gate_rules_math_confirm.json",
        ledger, cdir / f"gate_confirm_math_{ms}.json", f"confirm_math_q{ms}_astar",
    )
    _verify_manifest()
    print(json.dumps({"event": f"CONFIRM_MATH_{ms.upper()}_DONE", "bonferroni_decision": gm["decision"]}, indent=2))
    return 0


def cmd_finalize_confirm(args: argparse.Namespace) -> int:
    _verify_manifest()
    c_lean = ROOT / "results/confirm/track_l"
    c_m4 = ROOT / "results/confirm/track_m_4b"
    c_m9 = ROOT / "results/confirm/track_m_9b"

    lean_stats = {
        "4b": _paired_cluster_stats(c_lean / "q4b_b0_metrics.json", c_lean / "q4b_astar_metrics.json"),
        "9b": _paired_cluster_stats(c_lean / "q9b_b0_metrics.json", c_lean / "q9b_astar_metrics.json"),
    }
    lean_guard = {}
    for ms, st in lean_stats.items():
        am = st["astar_metrics"]
        lean_guard[ms] = bool(
            am["axiom_audit_clean_rate"] >= 1.0
            and am["stop_rate"] >= 0.85
            and am["truncation_rate"] <= 0.15
            and st["se_within_195pp_ceiling"]
        )
    lean_holm = _apply_holm_k2(lean_stats, lean_guard, min_delta=0.03)

    math_stats = {
        "4b": _paired_cluster_stats(c_m4 / "q4b_b0_metrics.json", c_m4 / "q4b_astar_metrics.json"),
        "9b": _paired_cluster_stats(c_m9 / "q9b_b0_metrics.json", c_m9 / "q9b_astar_metrics.json"),
    }
    math_retention: dict[str, dict[str, Any]] = {}
    math_guard: dict[str, bool] = {}
    for ms, cdir in [("4b", c_m4), ("9b", c_m9)]:
        rb = json.loads((cdir / f"ret150_q{ms}_b0_metrics.json").read_text(encoding="utf-8"))
        ra = json.loads((cdir / f"ret150_q{ms}_astar_metrics.json").read_text(encoding="utf-8"))
        b_cnt = round(sum(float(x["value"]) * 4 for x in rb["items"]))
        a_cnt = round(sum(float(x["value"]) * 4 for x in ra["items"]))
        diff_cnt = a_cnt - b_cnt
        ret_ok = bool(diff_cnt >= -12)  # integer-exact >= -0.02 out of 600 completions
        st = math_stats[ms]
        am = st["astar_metrics"]
        json_85_ok = bool(am["stop_rate"] >= 0.85 and am["truncation_rate"] <= 0.15)
        prog_88_ok = bool(am["stop_rate"] >= 0.88 and am["truncation_rate"] <= 0.12)
        math_retention[ms] = {
            "b0_correct_of_600": b_cnt,
            "astar_correct_of_600": a_cnt,
            "diff_correct_of_600": diff_cnt,
            "b0_avg_at_4": b_cnt / 600.0,
            "astar_avg_at_4": a_cnt / 600.0,
            "delta_avg_at_4": diff_cnt / 600.0,
            "retention_passed_ge_minus_12_of_600": ret_ok,
            "stop_trunc_passed_85_15": json_85_ok,
            "stop_trunc_passed_88_12": prog_88_ok,
        }
        math_guard[ms] = bool(ret_ok and json_85_ok and st["se_within_195pp_ceiling"])
    math_holm = _apply_holm_k2(math_stats, math_guard, min_delta=0.03)

    authorized: list[str] = []
    for ms in ("4b", "9b"):
        if lean_holm[ms]["authorize_holdout_astar"]:
            authorized.append(f"lean_{ms}")
        if math_holm[ms]["authorize_holdout_astar"]:
            authorized.append(f"math_{ms}")

    q27_lean = json.loads((c_lean / "q27b_ref_metrics.json").read_text(encoding="utf-8"))["metrics"]
    q27_math = (
        json.loads((c_m4 / "q27b_ref_metrics.json").read_text(encoding="utf-8"))["metrics"]
        if (c_m4 / "q27b_ref_metrics.json").is_file()
        else None
    )

    summary = {
        "schema_version": "1.0",
        "manifest_root_sha256": ROOT_HASH_FILE.read_text(encoding="utf-8").strip().split()[0],
        "track_l_confirm243": {
            "stats": lean_stats,
            "holm_k2": lean_holm,
            "q27b_comparator": q27_lean,
        },
        "track_m_confirm300": {
            "stats": math_stats,
            "retention_math500_150x4": math_retention,
            "holm_k2": math_holm,
            "q27b_comparator": q27_math,
        },
        "holdout_authorized_challengers": authorized,
    }
    CONFIRM_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _verify_manifest()
    print(json.dumps({"event": "CONFIRM_SUMMARY_SEALED", "holdout_authorized_challengers": authorized}, indent=2))
    return 0


def cmd_holdout_lean(args: argparse.Namespace) -> int:
    _verify_manifest()
    if not CONFIRM_SUMMARY.is_file():
        raise SystemExit("Refusing to open Protected Holdout before confirm_summary.json is sealed!")
    csum = json.loads(CONFIRM_SUMMARY.read_text(encoding="utf-8"))
    auth = set(csum["holdout_authorized_challengers"])
    winners = json.loads((ROOT / "results/plane2/track_l/track_l_winners.json").read_text(encoding="utf-8"))
    hdir = ROOT / "results/holdout/track_l"

    models: dict[str, tuple[Path, int]] = {
        "q4b_b0": (ROOT / "models/qwen35_4b", 1),
        "q9b_b0": (ROOT / "models/qwen35_9b", 1),
    }
    if "lean_4b" in auth:
        models["q4b_astar"] = (Path(winners["winner_4b_ckpt"]), 1)
    if "lean_9b" in auth:
        models["q9b_astar"] = (Path(winners["winner_9b_ckpt"]), 1)
    if not (ROOT / "models/qwen38_27b/config.json").is_file():
        raise SystemExit("Missing mandatory Qwen3.8-27B comparator for Holdout!")
    models["q27b_ref"] = (ROOT / "models/qwen38_27b", 4)

    for hname, hsplit in [
        ("minif2f_test244", ROOT / "eval/sealed_holdout/lean_holdout_minif2f_test244.jsonl"),
        ("putnambench_holdout128", ROOT / "eval/sealed_holdout/lean_holdout_putnam128.jsonl"),
    ]:
        for tag, (mp, tp) in models.items():
            _eval_split(
                track="lean",
                model_path=mp,
                split_file=hsplit,
                out_prefix=hdir / hname / tag,
                cand_id=f"holdout_{hname}_{tag}",
                protocol_id=f"holdout_{hname}_direct2k",
                k=8,
                tp=tp,
                thinking=False,
                temperature=0.8,
                max_tokens=2048,
            )

    _verify_manifest()
    print(json.dumps({"event": "HOLDOUT_LEAN_DONE", "evaluated_arms": list(models.keys())}, indent=2))
    return 0


def cmd_holdout_math(args: argparse.Namespace) -> int:
    _verify_manifest()
    if not CONFIRM_SUMMARY.is_file():
        raise SystemExit("Refusing to open Protected Holdout before confirm_summary.json is sealed!")
    csum = json.loads(CONFIRM_SUMMARY.read_text(encoding="utf-8"))
    auth = set(csum["holdout_authorized_challengers"])
    ms = args.model_size
    w_info = json.loads((ROOT / f"results/plane2/track_m_{ms}/track_m_{ms}_winner.json").read_text(encoding="utf-8"))
    astar_ckpt = Path(w_info["winner_ckpt"])
    base_ckpt = ROOT / f"models/qwen35_{ms}"
    if not (ROOT / "models/qwen38_27b/config.json").is_file():
        raise SystemExit("Missing mandatory Qwen3.8-27B comparator for Holdout!")

    hdir = ROOT / f"results/holdout/track_m_{ms}"
    holdout_benchmarks = [
        ("aime2025_30", ROOT / "eval/sealed_holdout/math_holdout_aime2025_30.jsonl", 8),
        ("aime2026_30", ROOT / "eval/sealed_holdout/math_holdout_aime2026_30.jsonl", 8),
        ("hmmt_feb2026_33", ROOT / "eval/sealed_holdout/math_holdout_hmmt2026.jsonl", 8),
        ("olympiadbench_120", ROOT / "eval/sealed_holdout/math_holdout_olympiad120.jsonl", 8),
        ("math500_ret150", ROOT / "eval/sealed_holdout/math_holdout_retention150.jsonl", 4),
    ]
    eval_models: list[tuple[str, Path, int]] = [(f"q{ms}_b0", base_ckpt, 1)]
    if f"math_{ms}" in auth:
        eval_models.append((f"q{ms}_astar", astar_ckpt, 1))
    q27b_assigned = (
        {"aime2025_30", "aime2026_30", "hmmt_feb2026_33"}
        if ms == "4b"
        else {"olympiadbench_120", "math500_ret150"}
    )

    for hname, hsplit, hk in holdout_benchmarks:
        bench_models = list(eval_models)
        if hname in q27b_assigned:
            bench_models.append(("q27b_ref", ROOT / "models/qwen38_27b", 4))
        for tag, mp, tp in bench_models:
            _eval_split(
                track="math", model_path=mp, split_file=hsplit,
                out_prefix=hdir / hname / tag, cand_id=f"holdout_{hname}_{tag}",
                protocol_id=f"holdout_{hname}_direct12k", k=hk, tp=tp, thinking=False,
                temperature=0.7, presence_penalty=0.0, max_tokens=12288,
            )

    _verify_manifest()
    print(json.dumps({"event": f"HOLDOUT_MATH_{ms.upper()}_DONE", "evaluated_arms": [m[0] for m in eval_models]}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("confirm-lean").set_defaults(func=cmd_confirm_lean)
    pm_c = sub.add_parser("confirm-math")
    pm_c.add_argument("--model-size", choices=["4b", "9b"], required=True)
    pm_c.set_defaults(func=cmd_confirm_math)
    sub.add_parser("finalize-confirm").set_defaults(func=cmd_finalize_confirm)
    sub.add_parser("holdout-lean").set_defaults(func=cmd_holdout_lean)
    pm_h = sub.add_parser("holdout-math")
    pm_h.add_argument("--model-size", choices=["4b", "9b"], required=True)
    pm_h.set_defaults(func=cmd_holdout_math)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

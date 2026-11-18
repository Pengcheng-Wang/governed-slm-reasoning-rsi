#!/usr/bin/env python3
"""
governed_loop_guard.py — Hardened Executable Governance Guard for Autonomous Optimization Loops (v2).

Peer-reviewed by Anthropic Opus 5.5 (Max) & OpenAI GPT-6 Astra and hardened against:
  - NaN / Infinity / missing-metric silent pass-throughs (strict finite JSON, fail-closed checks)
  - Direction-blind degradation & ratio bugs on minimize metrics (latency, memory, error rate)
  - Survivorship bias on crashed/omitted items (exact item-ID set verification)
  - Overconfident i.i.d. mean bootstrap on systems P95/P99 & clustered queries (cluster/run-level
    bootstrap recomputing the exact target statistic + max(se_boot, se_aa) noise floor)
  - Multi-step cumulative drift & Dev p-hacking (Two-Tier PROVISIONAL_KEEP vs PROMOTE vs
    INCONCLUSIVE_RETEST vs ROLLBACK, cumulative baseline confidence bound, registered Dev looks)
  - Symlink / shadow-file / ledger tampering (lstat manifest with mode+symlink checks, external
    manifest root digest pinning, and SHA-256 hash-chained trial_ledger.jsonl).

Zero external dependencies (Python 3.9+ standard library only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import stat
import statistics
import sys
from pathlib import Path
from typing import Any


def _load_strict_json(path: Path) -> Any:
    def _reject_nonfinite(tok: str) -> Any:
        raise ValueError(f"Non-finite JSON constant {tok!r} is forbidden in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _require_finite(val: Any, label: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Metric {label!r} must be numeric, got {val!r}")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"Metric {label!r} is non-finite: {val!r}")
    return f


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_or_abs(target: Path, root: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.as_posix()


def _collect_lstat_entries(paths: list[str], root: Path) -> dict[str, dict[str, Any]]:
    """Walks protected paths via os.lstat without following symlinks outside root."""
    root_real = root.resolve()
    entries: dict[str, dict[str, Any]] = {}

    def _visit(p: Path) -> None:
        if "__pycache__" in p.parts:
            return
        st = os.lstat(p)
        rel = _rel_or_abs(p, root_real)
        if stat.S_ISLNK(st.st_mode):
            link_target = os.readlink(p)
            resolved = (p.parent / link_target).resolve()
            try:
                resolved.relative_to(root_real)
                escapes_root = False
            except ValueError:
                escapes_root = True
            if escapes_root:
                raise ValueError(f"Symlink {rel!r} escapes workspace root -> {link_target!r}")
            entries[rel] = {
                "type": "symlink",
                "target": Path(link_target).as_posix(),
                "sha256": hashlib.sha256(os.fsencode(link_target)).hexdigest(),
                "exec": False,
            }
        elif stat.S_ISREG(st.st_mode):
            entries[rel] = {
                "type": "file",
                "sha256": _sha256_file(p),
                "size": int(st.st_size),
                "exec": bool(st.st_mode & stat.S_IXUSR),
            }
        elif stat.S_ISDIR(st.st_mode):
            for child_name in sorted(os.listdir(p)):
                if child_name == "__pycache__":
                    continue
                _visit(p / child_name)

    for raw in paths:
        cand = Path(raw)
        target = cand if cand.is_absolute() else (root_real / cand)
        if not os.path.lexists(target):
            raise FileNotFoundError(f"Protected path does not exist: {raw} -> {target}")
        _visit(target)

    return dict(sorted(entries.items()))


def _compute_manifest_digest(entries: dict[str, dict[str, Any]], protected_roots: list[str]) -> str:
    canonical = json.dumps(
        {"protected_roots": sorted(protected_roots), "entries": entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cmd_freeze_manifest(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    try:
        entries = _collect_lstat_entries(args.paths, root)
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2))
        return 2
    if not entries:
        print(json.dumps({"status": "ERROR", "error": "No files found under protected paths"}, indent=2))
        return 2

    root_digest = _compute_manifest_digest(entries, args.paths)
    payload = {
        "schema_version": "2.0",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": root.as_posix(),
        "protected_roots": args.paths,
        "entry_count": len(entries),
        "manifest_root_sha256": root_digest,
        "entries": entries,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "manifest": manifest_path.as_posix(),
                "entry_count": len(entries),
                "manifest_root_sha256": root_digest,
            },
            indent=2,
        )
    )
    return 0


def cmd_verify_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(json.dumps({"status": "TAMPERED", "error": f"Manifest missing: {manifest_path}"}), file=sys.stderr)
        return 2
    manifest = _load_strict_json(manifest_path)
    root = Path(args.root).resolve() if args.root else Path(manifest["root"]).resolve()
    expected: dict[str, dict[str, Any]] = manifest["entries"]
    protected_roots: list[str] = manifest["protected_roots"]

    recomputed_digest = _compute_manifest_digest(expected, protected_roots)
    if recomputed_digest != manifest.get("manifest_root_sha256"):
        print(json.dumps({"status": "TAMPERED", "error": "Manifest internal digest mismatch"}, indent=2))
        return 1
    if args.expected_root_sha256 and recomputed_digest != args.expected_root_sha256:
        print(
            json.dumps(
                {
                    "status": "TAMPERED",
                    "error": "Manifest root SHA-256 does not match externally pinned digest",
                    "expected": args.expected_root_sha256,
                    "actual": recomputed_digest,
                },
                indent=2,
            )
        )
        return 1

    try:
        current = _collect_lstat_entries(protected_roots, root)
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "TAMPERED", "error": str(exc)}, indent=2))
        return 1

    missing = [k for k in expected if k not in current]
    added = [k for k in current if k not in expected]
    modified = [k for k in expected if k in current and current[k] != expected[k]]

    ok = not (missing or added or modified)
    report = {
        "status": "VERIFIED" if ok else "TAMPERED",
        "checked_entries": len(expected),
        "manifest_root_sha256": recomputed_digest,
        "missing": missing,
        "modified": modified,
        "added_inside_protected_roots": added,
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def _approx_inv_norm(p: float) -> float:
    """Rational approximation (Acklam) to standard normal inverse CDF for p in (0, 1)."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def cmd_check_preconditions(args: argparse.Namespace) -> int:
    """
    Separates HARD admission blockers (NO_GO) from sizing & calibration diagnostics (WARN),
    following GPT-6 Astra & Opus 5.5 consensus.
    """
    cfg = _load_strict_json(Path(args.config))
    blockers: list[str] = []
    warnings: list[str] = []

    archetype = str(cfg.get("archetype", "pipeline")).lower()
    vtype = str(cfg.get("verifier_type", "")).lower()
    if vtype not in ("deterministic", "reference_grounded"):
        blockers.append(
            f"B1_UNVERIFIABLE_OBJECTIVE: verifier_type={vtype!r}. Autonomous loops require 'deterministic' "
            "or 'reference_grounded' verification; self-judging LLMs or manual-only rubrics cause rubber-stamp collapse."
        )

    if not bool(cfg.get("has_disjoint_dev_and_holdout", False)):
        blockers.append(
            "B2_MISSING_SPLIT_HIERARCHY: Separate Search/Train, Confirmation/Dev Gate, and single-touch Holdout Test are mandatory."
        )
    if archetype == "systems" and not bool(cfg.get("multi_shape_e2e_workload", True)):
        blockers.append(
            "B2_MICROBENCHMARK_ONLY: Systems loops require a multi-shape end-to-end workload, never a single microbenchmark shape."
        )
    if not bool(cfg.get("evaluator_locked_and_isolated", cfg.get("evaluator_locked", False))):
        blockers.append(
            "B3_MUTABLE_OR_INPROCESS_SCORER: Lock evaluators/splits via freeze-manifest and run scoring out-of-process."
        )
    if not bool(cfg.get("hermetic_reset_available", cfg.get("clean_reset_available", False))):
        blockers.append(
            "B4_NO_HERMETIC_RESET: Fresh git worktree / git clean -fdx (+ DB/cache reset) or reproducible M_0 recipe reset required."
        )
    if bool(cfg.get("offline_online_sign_conflict", False)) and not bool(cfg.get("has_trust_region_guard", False)):
        blockers.append(
            "B5_UNTRUSTED_OFFLINE_PROXY: Offline metric disagrees with online outcomes and lacks a trust-region constraint (e.g., top-k overlap / PSI)."
        )

    # Sizing & planning diagnostics (route to 1-shot fix or warn rather than rigid false-blocking)
    trial_min = _require_finite(cfg.get("single_trial_minutes", 0.0), "single_trial_minutes")
    budget_min = _require_finite(cfg.get("total_budget_minutes", 0.0), "total_budget_minutes")
    if trial_min > 0 and budget_min > 0:
        affordable = budget_min / trial_min
        min_loop_trials = 3.0 if archetype in ("weight_rsi", "hybrid") else 8.0
        if affordable < 2.0 and not bool(cfg.get("has_fast_proxy_stage", False)):
            blockers.append(
                f"B6_BUDGET_BELOW_TWO_TRIALS: budget affords only {affordable:.1f} trial(s). Do 1-shot engineering instead of a loop."
            )
        elif affordable < min_loop_trials and not bool(cfg.get("has_fast_proxy_stage", False)):
            warnings.append(
                f"S1_LOW_TRIAL_BUDGET: budget affords {affordable:.1f} trials (< {min_loop_trials:.0f} typical for {archetype!r}). "
                "Prefer 1-shot root-cause fixes + paired A/B confirmation or add a cheap proxy filter."
            )

    if archetype in ("weight_rsi", "hybrid"):
        if "learnable_fraction" in cfg:
            lf = _require_finite(cfg["learnable_fraction"], "learnable_fraction")
            if lf < 0.05:
                blockers.append(
                    f"B7_ZERO_LEARNABLE_FRONTIER: learnable_fraction={lf:.3f} (tasks with 0 < pass < k). "
                    "Pool is either impossible (0%) or already trivial (100%)."
                )
            elif lf < 0.15:
                warnings.append(f"S2_NARROW_FRONTIER: learnable_fraction={lf:.3f} (< 0.15); expect rapid saturation.")
        elif "baseline_score" in cfg:
            b = _require_finite(cfg["baseline_score"], "baseline_score")
            if b < 0.08:
                blockers.append(
                    f"B7_FLOOR_STARVATION: baseline_score={b:.3f} (< 0.08). Fix harness/prompts or distill from teacher first."
                )
            elif b > 0.95:
                blockers.append(
                    f"B7_CEILING_SATURATED: baseline_score={b:.3f} (> 0.95). Switch to a harder training/search pool."
                )

    if archetype == "systems" and "target_component_e2e_share" in cfg:
        share = _require_finite(cfg["target_component_e2e_share"], "target_component_e2e_share")
        if share < 0.10:
            warnings.append(
                f"S2_AMDAHL_CAP: target component is only {share*100:.1f}% of end-to-end runtime (max theoretical speedup < {share/(1-share+1e-9)*100:.1f}%)."
            )

    if "paired_noise_se" in cfg or "baseline_noise_se" in cfg:
        se = _require_finite(cfg.get("paired_noise_se", cfg.get("baseline_noise_se", 0.0)), "noise_se")
        se_aa = _require_finite(cfg.get("aa_noise_se", 0.0), "aa_noise_se")
        eff_se = max(se, se_aa)
        k_confirm = max(1, int(cfg.get("planned_confirmations", 2)))
        alpha = _require_finite(cfg.get("confirm_alpha", 0.025), "confirm_alpha")
        z_req = _approx_inv_norm(1.0 - alpha / k_confirm)
        mde_80 = (z_req + 0.8416) * eff_se
        if "min_worthwhile_effect" in cfg or "expected_total_gain" in cfg:
            target = abs(_require_finite(cfg.get("min_worthwhile_effect", cfg.get("expected_total_gain")), "target_effect"))
            if eff_se > 0 and target < mde_80:
                warnings.append(
                    f"S3_UNDERPOWERED_CONFIRMATION: 80%-power MDE ({mde_80:.4f} at K={k_confirm}, alpha={alpha}) "
                    f"> target effect ({target:.4f}). Increase paired blocks/runs or reduce variance."
                )

    if not bool(cfg.get("one_shot_bug_ruled_out", True)):
        warnings.append(
            "S4_PREFLIGHT_AUDIT_PENDING: Inspect 20-50 baseline failures/passes first to fix deterministic parser/config bugs in 1 shot."
        )

    n_planned = int(cfg.get("planned_search_trials", 20))
    recommended_tier = "tier1_standard_10_30" if n_planned <= 35 else "tier2_high_multiplicity_50_plus"
    verdict = "NO_GO" if blockers else ("GO_WITH_CALIBRATION_WARNINGS" if warnings else "GO")
    out = {
        "verdict": verdict,
        "recommended_tier": recommended_tier,
        "blockers": blockers,
        "warnings": warnings,
    }
    print(json.dumps(out, indent=2))
    return 0 if verdict != "NO_GO" else 1


def _compute_stat(values: list[float], stat_name: str) -> float:
    if not values:
        raise ValueError("Cannot compute statistic on empty list")
    s = stat_name.lower()
    if s == "mean":
        return statistics.fmean(values)
    ordered = sorted(values)
    if s in ("median", "p50"):
        return statistics.median(ordered)
    q_map = {"p90": 0.90, "p95": 0.95, "p99": 0.99}
    if s in q_map:
        q = q_map[s]
        pos = q * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    raise ValueError(f"Unsupported statistic={stat_name!r}; use mean, median, p90, p95, or p99")


def _extract_keyed_items(
    payload: dict[str, Any],
    label: str,
) -> tuple[dict[str, float], dict[str, str]] | None:
    """
    Accepts either:
      "items": {"q1": 1.0, "q2": 0.0}
    or:
      "items": [{"id": "q1", "cluster_id": "run_1", "value": 12.4}, ...]
    Returns (id_to_val, id_to_cluster).
    """
    raw = payload.get("items")
    if raw is None:
        return None
    vals: dict[str, float] = {}
    clusters: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            sid = str(k)
            vals[sid] = _require_finite(v, f"{label}.items[{sid}]")
            clusters[sid] = sid
        return vals, clusters
    if isinstance(raw, list):
        for idx, row in enumerate(raw):
            if not isinstance(row, dict) or "id" not in row or "value" not in row:
                raise ValueError(f"{label}.items[{idx}] must be an object with 'id' and 'value'")
            sid = str(row["id"])
            if sid in vals:
                raise ValueError(f"Duplicate item id {sid!r} in {label}.items")
            vals[sid] = _require_finite(row["value"], f"{label}.items[{sid}]")
            clusters[sid] = str(row.get("cluster_id", sid))
        return vals, clusters
    raise ValueError(f"{label}.items must be a dict or list of {{id, value, cluster_id?}} objects")


def _paired_cluster_bootstrap(
    cand_vals: dict[str, float],
    ref_vals: dict[str, float],
    clusters: dict[str, str],
    *,
    stat_name: str = "mean",
    direction_sign: float = 1.0,
    alpha_one_sided: float = 0.025,
    n_boot: int = 4000,
    seed: int = 20261116,
) -> dict[str, Any]:
    if set(cand_vals.keys()) != set(ref_vals.keys()):
        missing_c = sorted(set(ref_vals.keys()) - set(cand_vals.keys()))[:5]
        extra_c = sorted(set(cand_vals.keys()) - set(ref_vals.keys()))[:5]
        raise ValueError(f"Item ID mismatch in paired evaluation: missing={missing_c}, extra={extra_c}")
    if not cand_vals:
        raise ValueError("Item dictionary cannot be empty")

    cluster_to_ids: dict[str, list[str]] = {}
    for iid in sorted(cand_vals.keys()):
        cid = clusters.get(iid, iid)
        cluster_to_ids.setdefault(cid, []).append(iid)

    cluster_keys = sorted(cluster_to_ids.keys())
    n_clusters = len(cluster_keys)
    all_ids = sorted(cand_vals.keys())
    obs_cand = _compute_stat([cand_vals[i] for i in all_ids], stat_name)
    obs_ref = _compute_stat([ref_vals[i] for i in all_ids], stat_name)
    obs_signed_gain = direction_sign * (obs_cand - obs_ref)

    if n_clusters < 2:
        return {
            "n_items": len(all_ids),
            "n_clusters": n_clusters,
            "statistic": stat_name,
            "obs_candidate": obs_cand,
            "obs_reference": obs_ref,
            "signed_gain": obs_signed_gain,
            "se": 0.0,
            "lcb_one_sided": obs_signed_gain,
            "ci95_low": obs_signed_gain,
            "ci95_high": obs_signed_gain,
            "low_cluster_warning": True,
        }

    rng = random.Random(seed)
    boot_gains: list[float] = []
    for _ in range(n_boot):
        sampled_c: list[float] = []
        sampled_r: list[float] = []
        for _ in range(n_clusters):
            ck = cluster_keys[rng.randrange(n_clusters)]
            for iid in cluster_to_ids[ck]:
                sampled_c.append(cand_vals[iid])
                sampled_r.append(ref_vals[iid])
        sc = _compute_stat(sampled_c, stat_name)
        sr = _compute_stat(sampled_r, stat_name)
        boot_gains.append(direction_sign * (sc - sr))

    boot_gains.sort()
    se = statistics.stdev(boot_gains)
    lcb_idx = max(0, min(n_boot - 1, int(math.floor(alpha_one_sided * n_boot))))
    lo95 = boot_gains[max(0, int(math.floor(0.025 * n_boot)))]
    hi95 = boot_gains[min(n_boot - 1, int(math.ceil(0.975 * n_boot)))]
    tail_stat = stat_name.lower() in ("p90", "p95", "p99")
    return {
        "n_items": len(all_ids),
        "n_clusters": n_clusters,
        "statistic": stat_name,
        "obs_candidate": obs_cand,
        "obs_reference": obs_ref,
        "signed_gain": obs_signed_gain,
        "se": se,
        "lcb_one_sided": boot_gains[lcb_idx],
        "ci95_low": lo95,
        "ci95_high": hi95,
        "low_cluster_warning": bool(n_clusters < 10 or (tail_stat and n_clusters < 20)),
    }


def _read_and_verify_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    prev_hash = "GENESIS"
    for line_no, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        recorded_prev = row.get("prev_entry_sha256")
        recorded_hash = row.get("entry_sha256")
        if recorded_prev != prev_hash:
            raise ValueError(f"Ledger hash-chain broken at line {line_no}: expected prev={prev_hash}, got {recorded_prev}")
        body = {k: v for k, v in row.items() if k != "entry_sha256"}
        expect_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if recorded_hash != expect_hash:
            raise ValueError(f"Ledger entry tampered at line {line_no}: hash mismatch")
        prev_hash = recorded_hash
        rows.append(row)
    return rows


def cmd_verify_ledger(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger).resolve()
    try:
        rows = _read_and_verify_ledger(ledger_path)
    except Exception as exc:
        print(json.dumps({"status": "TAMPERED", "error": str(exc)}, indent=2))
        return 1
    head = rows[-1]["entry_sha256"] if rows else "GENESIS"
    if args.expected_head_sha256 and head != args.expected_head_sha256:
        print(json.dumps({"status": "TAMPERED", "error": "Ledger head SHA mismatch", "actual_head": head}, indent=2))
        return 1
    print(json.dumps({"status": "VERIFIED", "entries": len(rows), "head_sha256": head}, indent=2))
    return 0


def cmd_gate_decision(args: argparse.Namespace) -> int:
    cand = _load_strict_json(Path(args.candidate))
    inc = _load_strict_json(Path(args.incumbent))
    base = _load_strict_json(Path(args.baseline))
    rules = _load_strict_json(Path(args.rules))

    cm_raw = cand.get("metrics", {})
    im_raw = inc.get("metrics", {})
    bm_raw = base.get("metrics", {})

    pm = str(rules["primary_metric"])
    direction = str(rules.get("direction", "maximize")).lower()
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"Unsupported direction={direction!r}")
    sign = 1.0 if direction == "maximize" else -1.0
    stat_name = str(rules.get("primary_statistic", "mean"))
    stage = str(rules.get("decision_stage", "confirm")).lower()  # "search" | "confirm" | "frontier_explore"
    alpha = _require_finite(rules.get("alpha_one_sided", 0.025), "alpha_one_sided")
    k_confirm = max(1, int(rules.get("multiplicity_k", 1)))
    eff_alpha = alpha / k_confirm

    tripwires: list[str] = []
    warnings: list[str] = []

    # Optional environment fingerprint parity (prevents comparing across thermal/hardware/config epochs)
    if rules.get("require_env_fingerprint_match", False):
        if cand.get("env_fingerprint") != inc.get("env_fingerprint"):
            tripwires.append(
                f"env_fingerprint_mismatch:cand={cand.get('env_fingerprint')!r}!=inc={inc.get('env_fingerprint')!r}"
            )

    for label, mdict in (("candidate", cm_raw), ("incumbent", im_raw), ("baseline", bm_raw)):
        if pm not in mdict:
            tripwires.append(f"missing_primary_metric:{label}.{pm}")

    if tripwires:
        print(json.dumps({"decision": "ROLLBACK", "reason": "invalid_evidence", "tripwires": tripwires}, indent=2))
        return 1

    cand_pm = _require_finite(cm_raw[pm], f"candidate.{pm}")
    inc_pm = _require_finite(im_raw[pm], f"incumbent.{pm}")
    base_pm = _require_finite(bm_raw[pm], f"baseline.{pm}")

    # Item-keyed pairing & anti-survivorship verification
    cand_items = _extract_keyed_items(cand, "candidate")
    inc_items = _extract_keyed_items(inc, "incumbent")
    base_items = _extract_keyed_items(base, "baseline")

    if rules.get("expected_item_ids"):
        exp_ids = {str(x) for x in rules["expected_item_ids"]}
        if not cand_items or set(cand_items[0].keys()) != exp_ids:
            tripwires.append("item_id_coverage_mismatch_vs_expected_dev_ids")

    paired_vs_inc: dict[str, Any] | None = None
    paired_vs_base: dict[str, Any] | None = None
    try:
        if cand_items and inc_items:
            seed_str = f"{args.trial_id}:{cand.get('candidate_id', '')}"
            det_seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            paired_vs_inc = _paired_cluster_bootstrap(
                cand_items[0],
                inc_items[0],
                cand_items[1],
                stat_name=stat_name,
                direction_sign=sign,
                alpha_one_sided=eff_alpha,
                seed=det_seed,
            )
            # Recompute aggregate from items to guard against misreported summary headers
            cand_pm = paired_vs_inc["obs_candidate"]
            inc_pm = paired_vs_inc["obs_reference"]
            if paired_vs_inc["low_cluster_warning"]:
                warnings.append(
                    f"low_cluster_count_for_{stat_name}:n_clusters={paired_vs_inc['n_clusters']}"
                )
        if cand_items and base_items:
            paired_vs_base = _paired_cluster_bootstrap(
                cand_items[0],
                base_items[0],
                cand_items[1],
                stat_name=stat_name,
                direction_sign=sign,
                alpha_one_sided=alpha,
                seed=20261116,
            )
            base_pm = paired_vs_base["obs_reference"]
    except ValueError as exc:
        tripwires.append(f"pairing_integrity_failed:{exc}")

    se_aa = _require_finite(rules.get("aa_noise_se", 0.0), "aa_noise_se")
    se_reported = _require_finite(
        cm_raw.get("primary_se", im_raw.get("primary_se", rules.get("fallback_se", 0.0))),
        "reported_se",
    )
    se_boot = paired_vs_inc["se"] if paired_vs_inc is not None else 0.0
    se = max(se_boot, se_reported, se_aa)
    z_req = _approx_inv_norm(1.0 - eff_alpha)

    raw_delta = cand_pm - inc_pm
    signed_gain = sign * raw_delta
    lcb_vs_inc = (
        min(paired_vs_inc["lcb_one_sided"], signed_gain - z_req * se)
        if paired_vs_inc is not None
        else (signed_gain - z_req * se)
    )

    signed_vs_base = sign * (cand_pm - base_pm)
    z_base = _approx_inv_norm(1.0 - alpha)
    se_base = max(paired_vs_base["se"] if paired_vs_base else 0.0, se)
    lcb_vs_base = (
        min(paired_vs_base["lcb_one_sided"], signed_vs_base - z_base * se_base)
        if paired_vs_base is not None
        else (signed_vs_base - z_base * se_base)
    )

    # Fail-closed hard floors & ceilings
    for m_name, floor_val in rules.get("hard_floors", {}).items():
        if m_name not in cm_raw:
            tripwires.append(f"missing_guardrail_metric:candidate.{m_name}")
            continue
        v = _require_finite(cm_raw[m_name], f"candidate.{m_name}")
        f = _require_finite(floor_val, f"hard_floors.{m_name}")
        if v < f:
            tripwires.append(f"hard_floor_failed:{m_name}={v}<{f}")

    for m_name, ceil_val in rules.get("hard_ceilings", {}).items():
        if m_name not in cm_raw:
            tripwires.append(f"missing_guardrail_metric:candidate.{m_name}")
            continue
        v = _require_finite(cm_raw[m_name], f"candidate.{m_name}")
        c = _require_finite(ceil_val, f"hard_ceilings.{m_name}")
        if v > c:
            tripwires.append(f"hard_ceiling_failed:{m_name}={v}>{c}")

    # Direction-aware baseline ratio & degradation checks
    metric_dirs: dict[str, str] = {k: str(v).lower() for k, v in rules.get("metric_directions", {}).items()}

    for m_name, min_ratio in rules.get("min_ratio_vs_baseline", {}).items():
        if m_name not in cm_raw or m_name not in bm_raw:
            tripwires.append(f"missing_guardrail_metric:{m_name}")
            continue
        cv = _require_finite(cm_raw[m_name], f"candidate.{m_name}")
        bv = _require_finite(bm_raw[m_name], f"baseline.{m_name}")
        mr = _require_finite(min_ratio, f"min_ratio_vs_baseline.{m_name}")
        if bv <= 1e-12:
            tripwires.append(f"invalid_baseline_denominator:{m_name}={bv}<=0")
        elif (cv / bv) < mr:
            tripwires.append(f"baseline_ratio_too_low:{m_name}_ratio={cv/bv:.4f}<{mr}")

    for m_name, max_ratio in rules.get("max_ratio_vs_baseline", {}).items():
        if m_name not in cm_raw or m_name not in bm_raw:
            tripwires.append(f"missing_guardrail_metric:{m_name}")
            continue
        cv = _require_finite(cm_raw[m_name], f"candidate.{m_name}")
        bv = _require_finite(bm_raw[m_name], f"baseline.{m_name}")
        xr = _require_finite(max_ratio, f"max_ratio_vs_baseline.{m_name}")
        if bv <= 1e-12:
            tripwires.append(f"invalid_baseline_denominator:{m_name}={bv}<=0")
        elif (cv / bv) > xr:
            tripwires.append(f"baseline_ratio_too_high:{m_name}_ratio={cv/bv:.4f}>{xr}")

    degrad_cfg = rules.get("max_degradation_vs_baseline", rules.get("max_drop_vs_baseline", {}))
    for m_name, max_deg in degrad_cfg.items():
        if m_name not in cm_raw or m_name not in bm_raw:
            tripwires.append(f"missing_guardrail_metric:{m_name}")
            continue
        cv = _require_finite(cm_raw[m_name], f"candidate.{m_name}")
        bv = _require_finite(bm_raw[m_name], f"baseline.{m_name}")
        md = _require_finite(max_deg, f"max_degradation_vs_baseline.{m_name}")
        m_dir = metric_dirs.get(m_name, direction if m_name == pm else "maximize")
        d_sign = 1.0 if m_dir == "maximize" else -1.0
        # Positive degradation means candidate got WORSE than baseline along m_dir
        degradation = d_sign * (bv - cv)
        if degradation > md:
            tripwires.append(f"baseline_degradation_exceeded:{m_name}_degradation={degradation:.6f}>{md}")

    # Cumulative non-regression anchor vs B_0 (prevents multi-step drift)
    if "max_cumulative_primary_regression" in rules:
        max_cum_reg = _require_finite(rules["max_cumulative_primary_regression"], "max_cumulative_primary_regression")
        if lcb_vs_base < -max_cum_reg:
            tripwires.append(
                f"cumulative_baseline_regression:lcb_vs_base={lcb_vs_base:.6f}<-{max_cum_reg:.6f}"
            )

    # Ledger verification, anti-reroll check & Dev look budget
    ledger_path = Path(args.ledger).resolve() if args.ledger else None
    prior_rows: list[dict[str, Any]] = []
    if ledger_path and ledger_path.exists():
        try:
            prior_rows = _read_and_verify_ledger(ledger_path)
        except Exception as exc:
            print(json.dumps({"decision": "ROLLBACK", "reason": "ledger_tampered", "error": str(exc)}, indent=2))
            return 1

    cand_id = str(cand.get("candidate_id", args.trial_id))
    protocol_id = str(cand.get("protocol_id", stage))
    for r in prior_rows:
        if r.get("candidate_id") == cand_id and r.get("protocol_id") == protocol_id:
            if not bool(rules.get("allow_registered_retest", False)):
                tripwires.append(f"duplicate_dev_eval_for_candidate:{cand_id}:{protocol_id}")
                break

    dev_looks_used = len(prior_rows) + 1
    max_dev_looks = int(rules.get("max_dev_looks", 0))
    if max_dev_looks > 0 and dev_looks_used > max_dev_looks:
        tripwires.append(f"dev_look_budget_exhausted:{dev_looks_used}>{max_dev_looks}")

    # Stage-specific promotion logic (separating Search vs Confirm vs Frontier Explore)
    min_effect = _require_finite(rules.get("min_improvement", 0.0), "min_improvement")
    decision = "ROLLBACK"
    decision_reason = "guardrail_or_regression"

    if not tripwires:
        if stage == "search":
            # Lightweight Tier-1 sandbox search step: keep strictly better point estimates that pass all safety gates
            if signed_gain > max(0.0, min_effect):
                decision = "PROVISIONAL_KEEP"
                decision_reason = "search_point_improvement_passed_safety"
            else:
                decision = "ROLLBACK"
                decision_reason = f"search_no_gain:signed_gain={signed_gain:.6f}<=min={max(0.0, min_effect):.6f}"
        elif stage == "confirm":
            # Rigorous confirmation gate: requires both practical effect (>= min_effect) and statistical LCB > 0
            if signed_gain >= min_effect and signed_gain > 0.0 and lcb_vs_inc > 0.0:
                decision = "PROMOTE"
                decision_reason = "confirmed_statistically_and_practically"
            elif signed_gain >= min_effect and signed_gain > 0.0 and lcb_vs_inc <= 0.0 and se > 0.0:
                decision = "INCONCLUSIVE_RETEST"
                decision_reason = (
                    f"positive_point_gain_but_underpowered:signed_gain={signed_gain:.6f},lcb={lcb_vs_inc:.6f}<=0"
                )
            else:
                decision = "ROLLBACK"
                decision_reason = f"confirm_insufficient_gain:signed_gain={signed_gain:.6f},lcb={lcb_vs_inc:.6f}"
        elif stage == "frontier_explore":
            # ReST^EM / curriculum expansion: non-regression vs incumbent + mandatory frontier expansion + B_0 anchor
            max_step_reg = _require_finite(rules.get("max_regression_se_mult", 1.5), "max_regression_se_mult") * se
            fc = rules.get("frontier_coverage_metric")
            min_fc_gain = _require_finite(rules.get("min_frontier_gain", 1.0), "min_frontier_gain")
            fc_ok = True
            if fc:
                if fc not in cm_raw or fc not in im_raw:
                    tripwires.append(f"missing_frontier_metric:{fc}")
                    fc_ok = False
                else:
                    fc_gain = _require_finite(cm_raw[fc], f"cand.{fc}") - _require_finite(im_raw[fc], f"inc.{fc}")
                    fc_ok = fc_gain >= min_fc_gain
            if not tripwires and signed_gain >= -max_step_reg and (fc_ok or lcb_vs_inc > 0.0):
                decision = "PROMOTE" if lcb_vs_inc > 0.0 or fc_ok else "PROVISIONAL_KEEP"
                decision_reason = "frontier_expanded_within_non_regression_bound"
            else:
                decision = "ROLLBACK"
                decision_reason = "frontier_stagnant_or_step_regressed"
        else:
            raise ValueError(f"Unsupported decision_stage={stage!r}; use search, confirm, or frontier_explore")

    non_keep = decision in ("ROLLBACK",)
    consecutive_rollbacks = 1 if non_keep else 0
    for row in reversed(prior_rows):
        if row.get("decision") == "ROLLBACK":
            consecutive_rollbacks += 1
        elif row.get("decision") in ("PROMOTE", "PROVISIONAL_KEEP"):
            break

    max_rb = int(rules.get("max_consecutive_rollbacks", 0))
    should_halt = bool(
        (max_rb > 0 and consecutive_rollbacks >= max_rb)
        or (max_dev_looks > 0 and dev_looks_used >= max_dev_looks)
    )

    prev_hash = prior_rows[-1]["entry_sha256"] if prior_rows else "GENESIS"
    record_body: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trial_id": args.trial_id,
        "candidate_id": cand_id,
        "protocol_id": protocol_id,
        "hypothesis": args.hypothesis or "",
        "decision_stage": stage,
        "primary_metric": pm,
        "primary_statistic": stat_name,
        "direction": direction,
        "candidate_primary": cand_pm,
        "incumbent_primary": inc_pm,
        "baseline_primary": base_pm,
        "raw_delta_vs_incumbent": raw_delta,
        "signed_improvement": signed_gain,
        "lcb_one_sided_vs_incumbent": lcb_vs_inc,
        "lcb_one_sided_vs_baseline": lcb_vs_base,
        "effective_se": se,
        "effective_alpha_one_sided": eff_alpha,
        "tripwires": tripwires,
        "warnings": warnings,
        "decision": decision,
        "decision_reason": decision_reason,
        "dev_looks_used": dev_looks_used,
        "consecutive_rollbacks": consecutive_rollbacks,
        "halt_recommended": should_halt,
        "prev_entry_sha256": prev_hash,
    }
    entry_hash = hashlib.sha256(
        json.dumps(record_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    record = {**record_body, "entry_sha256": entry_hash}

    if ledger_path:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(json.dumps(record, indent=2))
    if decision in ("PROMOTE", "PROVISIONAL_KEEP"):
        return 0
    if decision == "INCONCLUSIVE_RETEST":
        return 3
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hardened Executable Governance Guard for Autonomous Optimization Loops")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze-manifest", help="lstat SHA-256 lock of evaluators, tests & split files")
    p_freeze.add_argument("--manifest", required=True, help="Output JSON path for IMMUTABLE_MANIFEST.json")
    p_freeze.add_argument("--root", default=".", help="Workspace root directory")
    p_freeze.add_argument("--paths", nargs="+", required=True, help="Protected files or directories")
    p_freeze.set_defaults(func=cmd_freeze_manifest)

    p_verify = sub.add_parser("verify-manifest", help="Verify zero modifications, symlinks, or injections")
    p_verify.add_argument("--manifest", required=True, help="Path to IMMUTABLE_MANIFEST.json")
    p_verify.add_argument("--root", default=None, help="Optional workspace root override")
    p_verify.add_argument("--expected-root-sha256", default=None, help="Optional externally pinned manifest SHA-256")
    p_verify.set_defaults(func=cmd_verify_manifest)

    p_pre = sub.add_parser("check-preconditions", help="Evaluate hard Go/No-Go blockers & sizing diagnostics")
    p_pre.add_argument("--config", required=True, help="JSON file describing task preconditions")
    p_pre.set_defaults(func=cmd_check_preconditions)

    p_gate = sub.add_parser("gate-decision", help="Run search/confirm/frontier gate with cluster bootstrap & ledger")
    p_gate.add_argument("--candidate", required=True, help="Candidate metrics JSON (with optional keyed items)")
    p_gate.add_argument("--incumbent", required=True, help="Incumbent metrics JSON (re-measured or paired)")
    p_gate.add_argument("--baseline", required=True, help="Initial B_0 / M_0 metrics JSON")
    p_gate.add_argument("--rules", required=True, help="Gate, direction & tripwire rules JSON")
    p_gate.add_argument("--ledger", default=None, help="Optional hash-chained trial_ledger.jsonl path")
    p_gate.add_argument("--trial-id", default="trial_01", help="Trial or round identifier")
    p_gate.add_argument("--hypothesis", default="", help="Falsifiable hypothesis tested by this trial")
    p_gate.set_defaults(func=cmd_gate_decision)

    p_vled = sub.add_parser("verify-ledger", help="Verify SHA-256 hash chain of trial_ledger.jsonl")
    p_vled.add_argument("--ledger", required=True, help="Path to trial_ledger.jsonl")
    p_vled.add_argument("--expected-head-sha256", default=None, help="Optional externally pinned ledger head SHA-256")
    p_vled.set_defaults(func=cmd_verify_ledger)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

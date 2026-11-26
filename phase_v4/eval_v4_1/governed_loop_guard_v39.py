#!/usr/bin/env python3
"""
governed_loop_guard.py — Hardened Executable Governance Guard for Autonomous Optimization Loops (v3.9).

Peer-reviewed and stress-tested with GPT 6.2 Astra (Max) & Anthropic Claude Opus 5.5 (Max).
Enforces mechanical, fail-closed governance across iterative propose-evaluate-keep (RSI) loops
spanning scientific ML, multi-GPU model training, and multi-language coding repositories:
  1. Immutable Surface Lock (`freeze-manifest` / `verify-manifest`):
     lstat SHA-256 verification with symlink-escape blocking and externally pinned root digest.
  2. Pre-Flight Admission & Sizing (`check-preconditions`):
     Hard B1-B11 Go/No-Go blockers (including exact/Monte Carlo sign-flip resolution,
     multi-seed pilot checks and B10 unsealed-holdout quarantine) + S1-S6 power/MDE
     (`delta + (z + 0.84)*SE`), boundary-margin, incumbent-inclusion, and warmup/tail-batch diagnostics.
  3. Static Protocol-Hygiene, Mutable-Allowlist & Split-Leak Scanners (`audit-leaks`, `audit-splits`):
     Multi-language diff + directory allowlist enforcement, binary/startup-hook (.pyc/.so/.pth)
     blocking, build-hook & test-skip marker detection, AST dynamic-import/GC-heap inspection checks,
     Lean 4 (`sorry`/`admit`/`axiom`/`native_decide`) proof checks, and train/confirm/holdout
     row-deduplication, entity-disjointness & temporal-cutoff verification.
  4. Out-of-Process Isolated Runners (`run-isolated`):
     - `predict`: single-script (`python -I -B`) or `.git`-free multi-file repo snapshot (`--workspace-root`)
     - `causal_online`: step-by-step causal stdio (`bin_packing` and generic hidden-reward `generic_step`)
     - `build_test_bench`: `.git`-free offline repo snapshot -> build & pin binary SHA-256 -> trusted tests
       with executed test-count parity -> paired benchmark of that exact artifact
     - `train_lock_score`: `.git`-free offline training (`torchrun`/trainer, zero Confirm/Holdout access) ->
       `.safetensors`/non-pickle checkpoint lock & inner-val selection check -> isolated immutable scorer
  5. Order-Invariant Batch Search, Parallel IUT Confirmation & Sufficient-Statistic Resampler
     (`select-search-winner`, `gate-decision`, `gate-confirm-iut`):
     - O(S * G * |S|) sufficient-statistic cell pre-reduction for `mean`, `ratio_from_sums`,
       `micro_f1_from_counts`, and `macro_f1_from_counts`, plus SHA-256-pinned `.json.gz`/`.jsonl.gz`/`.npz`
       `items_sidecar` loading.
     - Batch M-comparator Intersection-Union Confirmation (`gate-confirm-iut`) & cross-platform
       exclusive file-locked ledger appends (`<ledger>.lock`).
     - Paired auxiliary slice non-inferiority guardrails (`paired_aux_noninferiority`).
     - Independent 2-/3-way ANOVA mean-square Satterthwaite bounds and a paired user/item pigeonhole
       bootstrap; finite Monte Carlo permutation floor; semantic identity and reserved alpha spending.
  6. Atomic Single-Touch Holdout Seal (`gate-holdout`):
     Auto-requires `current_incumbent_id` alongside `required_comparators`, verifies artifact/manifest/ledger
     SHA-256 digests, atomically locks `HOLDOUT_SEAL.json` (`O_CREAT | O_EXCL`), and re-verifies global,
     per-stratum & paired-aux guardrails on Holdout.

Zero external dependencies required (Python 3.9+ standard library; optional `.npz` sidecar when NumPy is present).
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import contextlib
import csv
import datetime as dt
import fnmatch
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

GUARD_VERSION = "3.9.0"
DEFAULT_N_BOOTSTRAP = 4000
DEFAULT_N_PERM = 8000


# ---------------------------------------------------------------------------
# Strict JSON, Numeric & Hashing Primitives
# ---------------------------------------------------------------------------


def _decode_json(raw: bytes, path: Path) -> Any:
    def _reject_nonfinite(tok: str) -> Any:
        raise ValueError(f"Non-finite JSON constant {tok!r} is forbidden in {path}")

    def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            obj[key] = value
        return obj

    raw = gzip.decompress(raw) if path.name.lower().endswith(".gz") else raw
    return json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite, object_pairs_hook=_unique_keys)


def _load_strict_json(path: Path) -> Any:
    return _decode_json(path.read_bytes(), path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_finite(val: Any, label: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Metric {label!r} must be numeric, got {val!r}")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"Metric {label!r} is non-finite: {val!r}")
    return f


def _require_nonneg_finite(val: Any, label: str) -> float:
    f = _require_finite(val, label)
    if f < 0.0:
        raise ValueError(f"Count/weight {label!r} must be >= 0, got {f}")
    return f


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _confirm_alpha_plan(rules: dict[str, Any], override: list[str] | None = None) -> tuple[float, tuple[float, ...]]:
    alpha = _require_finite(rules.get("alpha_one_sided", 0.025), "alpha_one_sided")
    if not 0.0 < alpha < 0.5 or 1.0 - alpha == 1.0:
        raise ValueError("alpha_one_sided must be a resolvable probability in (0, 0.5)")
    cap = rules.get("max_confirm_looks", 0)
    if type(cap) is not int or cap < 0:
        raise ValueError("max_confirm_looks must be a nonnegative integer")
    legacy_k = _require_positive_int(rules.get("multiplicity_k", 1), "multiplicity_k")
    explicit = rules.get("alpha_schedule")
    if override is not None:
        values = [float(token) for group in override for token in group.split(",")]
        if explicit is not None and explicit != values:
            raise ValueError("--alpha-schedule conflicts with the schedule in the rules")
        explicit = values
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit or (cap and cap != len(explicit)):
            raise ValueError("alpha_schedule must list one positive allocation per Confirm look")
        schedule = tuple(_require_finite(x, "alpha_schedule allocation") for x in explicit)
        if math.fsum(schedule) > alpha:
            raise ValueError("alpha_schedule exceeds the total Confirm alpha")
    else:
        count = cap or legacy_k
        denominator = max(count, legacy_k)
        if count > 100_000:
            raise ValueError("Confirm look budget exceeds 100,000")
        allocation = alpha / denominator
        if allocation * count > alpha:
            allocation = math.nextafter(allocation, 0.0)
        schedule = (allocation,) * count
    if any(x <= 0.0 or 1.0 - x == 1.0 for x in schedule):
        raise ValueError("every Confirm alpha allocation must be positive and resolvable")
    return alpha, schedule


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_artifacts(paths: list[str], root: Path | None = None) -> dict[str, str]:
    base = root.resolve() if root else Path.cwd().resolve()
    entries = _collect_lstat_entries(paths, base)
    digest = _compute_manifest_digest(entries, paths)
    return {
        "artifact_bundle_sha256": digest,
        "entry_count": str(len(entries)),
    }


def _rel_or_abs(target: Path, root: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.as_posix()


# ---------------------------------------------------------------------------
# 1. Immutable Manifest Freeze & Verification
# ---------------------------------------------------------------------------


def _collect_lstat_entries(paths: list[str], root: Path, *, regular_only: bool = False) -> dict[str, dict[str, Any]]:
    """Walks protected paths via os.lstat without following symlinks outside root."""
    root_real = root.resolve()
    entries: dict[str, dict[str, Any]] = {}

    def _visit(p: Path) -> None:
        if "__pycache__" in p.parts:
            return
        st = os.lstat(p)
        rel = _rel_or_abs(p, root_real)
        if regular_only and (stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"Candidate artifact is a symbolic link or reparse point: {rel}")
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
        elif regular_only:
            raise ValueError(f"Candidate artifact is not an ordinary file or directory: {rel}")

    for raw in paths:
        cand = Path(raw)
        target = cand if cand.is_absolute() else (root_real / cand)
        if regular_only and target.resolve() != Path(os.path.abspath(target)):
            raise ValueError(f"Candidate artifact path traverses an alias: {raw}")
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


def _artifact_fingerprints(paths: list[str]) -> dict[str, str]:
    root = Path.cwd().resolve()
    entries = _collect_lstat_entries(paths, root, regular_only=True)
    if not entries:
        raise ValueError("Candidate artifact bundle contains no files")

    class StripDocstrings(ast.NodeTransformer):
        def strip(self, node: Any) -> Any:
            if ast.get_docstring(node, clean=False) is not None:
                node.body.pop(0)
                if not node.body and not isinstance(node, ast.Module):
                    node.body.append(ast.Pass())
            return self.generic_visit(node)

        visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = strip

    components: list[tuple[str, str]] = []
    for rel, entry in entries.items():
        path = root / rel
        suffix = path.suffix.lower()
        digest = entry["sha256"]
        if suffix in {".py", ".pyw", ".json"} or path.name.lower().endswith(".json.gz"):
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"Candidate artifact changed during fingerprinting: {rel}")
            if suffix in {".py", ".pyw"}:
                try:
                    tree = StripDocstrings().visit(ast.parse(raw, filename=rel))
                except SyntaxError as exc:
                    raise ValueError(f"Cannot fingerprint Python artifact {rel}: {exc}") from exc
                digest = _canonical_sha256(ast.dump(tree, include_attributes=False))
                suffix = ".py"
            else:
                digest = _canonical_sha256(_decode_json(raw, path))
                suffix = ".json"
        components.append((suffix, digest))
    # Names do not buy a new look; the separate byte/manifest identity still binds them.
    return {
        "candidate_artifact_sha256": _compute_manifest_digest(entries, paths),
        "artifact_semantic_sha256": _canonical_sha256({"schema": 1, "components": sorted(components)}),
    }


def _verify_manifest_internal(
    manifest_path: Path,
    root_override: Path | None = None,
    expected_root_sha256: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    if not manifest_path.exists():
        return False, {"status": "TAMPERED", "error": f"Manifest missing: {manifest_path}"}
    manifest = _load_strict_json(manifest_path)
    root = root_override.resolve() if root_override else Path(manifest["root"]).resolve()
    expected: dict[str, dict[str, Any]] = manifest["entries"]
    protected_roots: list[str] = manifest["protected_roots"]

    recomputed_digest = _compute_manifest_digest(expected, protected_roots)
    if recomputed_digest != manifest.get("manifest_root_sha256"):
        return False, {"status": "TAMPERED", "error": "Manifest internal digest mismatch"}
    if expected_root_sha256 and recomputed_digest != expected_root_sha256:
        return False, {
            "status": "TAMPERED",
            "error": "Manifest root SHA-256 does not match externally pinned digest",
            "expected": expected_root_sha256,
            "actual": recomputed_digest,
        }

    try:
        current = _collect_lstat_entries(protected_roots, root)
    except (FileNotFoundError, ValueError) as exc:
        return False, {"status": "TAMPERED", "error": str(exc)}

    missing = [k for k in expected if k not in current]
    added = [k for k in current if k not in expected]
    modified = [k for k in expected if k in current and current[k] != expected[k]]

    ok = not (missing or added or modified)
    return ok, {
        "status": "VERIFIED" if ok else "TAMPERED",
        "checked_entries": len(expected),
        "manifest_root_sha256": recomputed_digest,
        "missing": missing,
        "modified": modified,
        "added_inside_protected_roots": added,
    }


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
        "schema_version": "3.0",
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
    root_ov = Path(args.root) if args.root else None
    ok, report = _verify_manifest_internal(manifest_path, root_ov, args.expected_root_sha256)
    if not ok and "Manifest missing" in report.get("error", ""):
        print(json.dumps(report), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 2. Pre-Flight Admission & Sizing Audit (`check-preconditions`)
# ---------------------------------------------------------------------------


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


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for regularized incomplete beta I_x(a, b)."""
    max_it = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _inv_student_t(p: float, df: float) -> float:
    """Exact inverse CDF of Student's t(df) for p in (0, 1) via incomplete-beta bisection."""
    if not 0.0 < p < 1.0 or not 0.0 < df < math.inf:
        raise ValueError("Student's t inversion requires 0 < p < 1 and finite positive df")
    if abs(p - 0.5) < 1e-15:
        return 0.0
    if df >= 1.0e5:
        return _approx_inv_norm(p)
    upper_tail = (1.0 - p) if p > 0.5 else p

    def _one_sided_tail(t_pos: float) -> float:
        if t_pos <= 0.0:
            return 0.5
        x = df / (df + t_pos * t_pos)
        return 0.5 * _reg_inc_beta(x, 0.5 * df, 0.5)

    lo = 0.0
    z0 = abs(_approx_inv_norm(1.0 - upper_tail))
    hi = max(4.0, z0 * 4.0)
    while _one_sided_tail(hi) > upper_tail and hi < 1.0e12:
        hi *= 2.0
    if _one_sided_tail(hi) > upper_tail:
        raise ValueError("Crossed t bound is unresolvable at the effective degrees of freedom")
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _one_sided_tail(mid) > upper_tail:
            lo = mid
        else:
            hi = mid
    q_pos = 0.5 * (lo + hi)
    return q_pos if p > 0.5 else -q_pos


def cmd_check_preconditions(args: argparse.Namespace) -> int:
    """Evaluates HARD admission blockers (NO_GO) and sizing/power diagnostics (WARN)."""
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
    if cfg.get("candidate_has_label_file_access", False):
        blockers.append(
            "B3_CANDIDATE_LABEL_EXPOSURE: Candidate subprocess can read gold Dev/Holdout labels; enforce data-only/causal-action boundary."
        )
    if not bool(cfg.get("hermetic_reset_available", cfg.get("clean_reset_available", False))):
        blockers.append(
            "B4_NO_HERMETIC_RESET: Fresh git worktree / git clean -fdx (+ DB/cache reset) or reproducible M_0 recipe reset required."
        )
    if bool(cfg.get("offline_online_sign_conflict", False)) and not bool(cfg.get("has_trust_region_guard", False)):
        blockers.append(
            "B5_UNTRUSTED_OFFLINE_PROXY: Offline metric disagrees with online outcomes and lacks a trust-region constraint (e.g., top-k overlap / PSI)."
        )

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
                f"S2_AMDAHL_CAP: target component is only {share*100:.1f}% of end-to-end runtime "
                f"(max theoretical speedup < {share/(1-share+1e-9)*100:.1f}%)."
            )

    alpha_pre, alpha_plan = _confirm_alpha_plan({
        **cfg,
        "alpha_one_sided": cfg.get("confirm_alpha", cfg.get("alpha_one_sided", 0.025)),
        "max_confirm_looks": cfg.get("max_confirm_looks", cfg.get("planned_confirmations", 0 if "alpha_schedule" in cfg else 2)),
    })
    k_confirm_pre = len(alpha_plan)
    eff_alpha_pre = min(alpha_plan)
    n_perm = _require_positive_int(cfg.get("n_perm", DEFAULT_N_PERM), "n_perm")
    p_floor = 1.0 / (1 + n_perm)

    n_cl_plan = None
    if "planned_confirm_clusters" in cfg or "n_confirm_clusters" in cfg:
        n_cl_plan = cfg.get("planned_confirm_clusters", cfg.get("n_confirm_clusters"))
        if type(n_cl_plan) is not int or n_cl_plan < 0:
            raise ValueError("planned_confirm_clusters must be a nonnegative integer")
        signflip_floor = (0.5 ** n_cl_plan) if n_cl_plan <= 16 else p_floor
        if n_cl_plan < 2:
            blockers.append(
                f"B8_INSUFFICIENT_CONFIRM_CLUSTERS: n_confirm_clusters={n_cl_plan} (< 2)."
            )
        elif (
            n_cl_plan < 20 and signflip_floor > eff_alpha_pre
            and not cfg.get("item_cluster_key") and bool(cfg.get("enforce_small_cluster_signflip", True))
        ):
            blockers.append(
                f"B8_UNDERPOWERED_CLUSTER_SIGNFLIP_FLOOR: "
                f"{'exact' if n_cl_plan <= 16 else 'Monte Carlo'} floor={signflip_floor:.6g} > "
                f"smallest allocated alpha={eff_alpha_pre:.6g}; increase replication or Monte Carlo draws, or revisit the look plan."
            )

    claim_scope_pre = str(cfg.get("claim_scope", "")).lower()
    pilot_source = str(cfg.get("pilot_source", "")).strip().lower()
    n_seeds_plan = int(cfg.get("planned_training_seeds", cfg.get("num_seeds", 0)))
    multi_seed = n_seeds_plan > 1 or claim_scope_pre == "training_recipe"
    synthetic_pilot = pilot_source == "synthetic_jitter" or (
        multi_seed and (
            pilot_source in {"synthetic_noise", "synthetic_noise_injection"}
            or (cfg.get("synthetic_noise_injection") is True and pilot_source != "paired_seed_retrain")
        )
    )
    if synthetic_pilot:
        blockers.append(
            "B11_SYNTHETIC_JITTER_PILOT: Synthetic perturbations do not estimate seed-retrain noise. "
            "Use actual paired refits on independent pilot blocks before making a multi-seed power claim."
        )
    if claim_scope_pre == "training_recipe" and ("planned_training_seeds" in cfg or "num_seeds" in cfg):
        if n_seeds_plan < 2:
            blockers.append(
                f"B9_SINGLE_SEED_TRAINING_CLAIM: planned_training_seeds={n_seeds_plan} (< 2). "
                "Training-recipe claims require >= 2 seeds (3-5+ recommended with crossed Satterthwaite t-bound)."
            )
        elif n_seeds_plan < 5:
            warnings.append(
                f"S3_SMALL_TRAINING_SEED_COUNT: planned_training_seeds={n_seeds_plan} (< 5); "
                "use a crossed design and inspect the seed contribution and effective degrees of freedom."
            )

    if cfg.get("current_incumbent_id") and isinstance(cfg.get("required_comparators"), list):
        inc_str = str(cfg["current_incumbent_id"])
        req_list = [str(x) for x in cfg["required_comparators"]]
        if inc_str not in req_list:
            warnings.append(
                f"S5_MISSING_INCUMBENT_IN_COMPARATORS: current_incumbent_id={inc_str!r} is not listed in "
                "required_comparators; include it so candidates cannot promote while regressing vs the incumbent."
            )

    if str(cfg.get("margin_mode", "additive")).lower() == "additive" and "comparator_search_metric" in cfg:
        comp_m = _require_finite(cfg["comparator_search_metric"], "comparator_search_metric")
        dir_c = str(cfg.get("direction", "minimize")).lower()
        bound_c = _require_finite(cfg.get("metric_bound", 0.0 if dir_c == "minimize" else 1.0), "metric_bound")
        headroom = abs(comp_m - bound_c)
        if "min_improvement" in cfg or "min_worthwhile_effect" in cfg:
            m_val = abs(_require_finite(cfg.get("min_improvement", cfg.get("min_worthwhile_effect", 0.0)), "margin"))
            if headroom > 0.0 and m_val >= 0.5 * headroom:
                warnings.append(
                    f"S3_ADDITIVE_MARGIN_NEAR_BOUNDARY: additive margin ({m_val:.6g}) is >= 50% of comparator "
                    f"headroom ({headroom:.6g}) toward bound {bound_c}; prefer margin_mode='relative_headroom' or 'log_ratio'."
                )

    mde_80_out: float | None = None
    resolution: dict[str, Any] = {}
    se_input = cfg.get("seed_retrain_noise_se", cfg.get("paired_noise_se", cfg.get("baseline_noise_se")))
    if se_input is not None and not synthetic_pilot:
        se = _require_nonneg_finite(se_input, "paired_noise_se")
        se_aa = _require_nonneg_finite(cfg.get("aa_noise_se", 0.0), "aa_noise_se")
        eff_se = max(se, se_aa)
        pilot_clusters = (
            _require_positive_int(cfg.get("pilot_n_clusters", n_cl_plan), "pilot_n_clusters")
            if n_cl_plan is not None and n_cl_plan >= 2 else None
        )
        if pilot_clusters is not None and pilot_clusters < 2:
            raise ValueError("pilot_n_clusters must be at least two")
        z_alpha = _approx_inv_norm(1.0 - eff_alpha_pre)
        null_margin_pre = _require_nonneg_finite(cfg.get("min_improvement", cfg.get("null_margin", 0.0)), "min_improvement")
        mde_80 = null_margin_pre + (z_alpha + 0.8416) * eff_se
        mde_80_out = mde_80
        if "min_worthwhile_effect" in cfg or "expected_total_gain" in cfg:
            target = abs(_require_finite(cfg.get("min_worthwhile_effect", cfg.get("expected_total_gain")), "target_effect"))
            if pilot_clusters is not None:
                target_power = _require_finite(cfg.get("target_power", 0.8), "target_power")
                if not 0.5 < target_power < 1.0:
                    raise ValueError("target_power must be in (0.5, 1)")
                noise_multiplier = z_alpha + _approx_inv_norm(target_power)
                design_n_star = cfg.get("required_paired_clusters")
                if target <= null_margin_pre:
                    n_star, method = None, "unresolved"
                elif design_n_star is not None:
                    n_star = max(2, _require_positive_int(design_n_star, "required_paired_clusters"))
                    method = "registered_design"
                elif eff_se <= 0.0:
                    n_star, method = None, "unresolved"
                else:
                    factor = (noise_multiplier * eff_se) / (target - null_margin_pre)
                    estimate = pilot_clusters * factor * factor
                    n_star = max(2, math.ceil(estimate)) if math.isfinite(estimate) else None
                    method = "inverse_sqrt_cluster_scaling_approximation"
                resolution = {
                    "n_clusters": n_cl_plan, "n_clusters_star": n_star,
                    "paired_resolution_ratio_q": n_cl_plan / n_star if n_star is not None else None,
                    "paired_resolution_method": method, "paired_resolution_target_power": target_power,
                }
            if eff_se > 0 and target < mde_80:
                warnings.append(
                    f"S3_UNDERPOWERED_CONFIRMATION: 80%-power MDE ({mde_80:.4f} = margin {null_margin_pre:.4f} + "
                    f"(z+0.84)*SE at K={k_confirm_pre}, alpha={alpha_pre}) > target effect ({target:.4f}). "
                    "Increase paired blocks/runs or reduce variance."
                )

    if bool(cfg.get("holdout_previously_unsealed", False)):
        blockers.append(
            "B10_REUSED_UNSEALED_HOLDOUT_FORBIDDEN: Protected Holdout was already unsealed in an earlier wave/run; "
            "single-touch governance strictly forbids reusing an unsealed holdout for subsequent optimization waves. "
            "Provision fresh unsealed holdout benchmarks and quarantine prior holdouts to zero-retuning transfer checks."
        )

    if bool(cfg.get("has_latency_ratio_guardrail", False)) and not bool(
        cfg.get("benchmark_warmup_and_tail_batch_verified", False)
    ):
        warnings.append(
            "S6_BENCHMARK_WARMUP_OR_TAIL_BATCH_UNVERIFIED: Latency/wall-clock ratio guardrails are active without "
            "explicit warmup and uneven tail-batch verification; run untimed warmup passes and test N % batch_size != 0 "
            "before freezing latency ratios."
        )

    if not bool(cfg.get("one_shot_bug_ruled_out", True)):
        warnings.append(
            "S4_PREFLIGHT_AUDIT_PENDING: Inspect 20-50 baseline failures/passes first to fix deterministic parser/config bugs in 1 shot."
        )

    n_planned = int(cfg.get("planned_search_trials", 20))
    recommended_tier = "tier1_standard_10_30" if n_planned <= 35 else "tier2_high_multiplicity_50_plus"
    verdict = "NO_GO" if blockers else ("GO_WITH_CALIBRATION_WARNINGS" if warnings else "GO")
    out: dict[str, Any] = {
        "verdict": verdict,
        "recommended_tier": recommended_tier,
        "blockers": blockers,
        "warnings": warnings,
        "n_perm": n_perm,
        "p_floor": p_floor,
        "alpha_schedule": alpha_plan,
        **resolution,
    }
    if mde_80_out is not None:
        out["mde_80_power"] = mde_80_out
        out["mde_pilot_source"] = pilot_source or "unspecified"
        out["mde_pilot_n_clusters"] = pilot_clusters
    print(json.dumps(out, indent=2))
    return 0 if verdict != "NO_GO" else 1


# ---------------------------------------------------------------------------
# 3. Static Protocol-Hygiene & Anti-Cheating Scanner (`audit-leaks`)
# ---------------------------------------------------------------------------

_LEAK_REGEX_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("CRITICAL", "SYS_MODULES_MONKEYPATCH", re.compile(r"\bsys\.modules\s*\[")),
    ("CRITICAL", "OS_EXIT_BYPASS", re.compile(r"\bos\._exit\s*\(")),
    ("CRITICAL", "MOCK_FRAMEWORK_IN_CANDIDATE", re.compile(r"\bunittest\.mock\b|\bmock\.patch\b")),
    ("CRITICAL", "CLOCK_SPOOFING", re.compile(r"\btime\.(?:time|perf_counter|monotonic)\s*=")),
    (
        "CRITICAL",
        "EVALUATOR_ATTR_MONKEYPATCH",
        re.compile(r"\bsetattr\s*\([^,]+,\s*['\"](?:validate|evaluate|test|score|forward_eval)['\"]"),
    ),
    (
        "CRITICAL",
        "DP_NOISE_SUBTRACTION_OR_CLEAN_CACHE",
        re.compile(r"\b_base_coef\b|\bself\.coef_\s*-\s*noise\b"),
    ),
    (
        "HIGH",
        "HOLD_OR_TEST_PATH_LITERAL",
        re.compile(r"['\"][^'\"]*(?:holdout|test_labels|gold_labels|IMMUTABLE_MANIFEST|HOLDOUT_SEAL)[^'\"]*['\"]", re.IGNORECASE),
    ),
    (
        "WARN",
        "TEST_LOADER_IN_TRAINING_CODE",
        re.compile(r"\b(?:test_loader|test_data|test_dataset|val_test)\b"),
    ),
]


_UNTRUSTED_BINARY_OR_HOOK_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".pth")
_SENSITIVE_BUILD_OR_HOOK_NAMES = {
    "build.rs",
    "cmakelists.txt",
    "setup.py",
    "conftest.py",
    "sitecustomize.py",
    "usercustomize.py",
    "directory.build.props",
    "directory.build.targets",
}
_TEST_SKIP_DIFF_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"@(?:pytest\.mark\.skip|pytest\.mark\.xfail|unittest\.skip)"),
    re.compile(r"\bpytest\.skip\s*\("),
    re.compile(r"#\s*\[\s*ignore\b"),
    re.compile(r"\bt\.Skip(?:Now|f)?\s*\("),
    re.compile(r"\b(?:it|describe|test)\.skip\s*\(|\b(?:xit|xdescribe)\s*\("),
    re.compile(r"\[\s*(?:Fact|Theory)\s*\([^\]]*\bSkip\s*="),
]


def _matches_allowlist(rel_posix: str, allowlist: list[str]) -> bool:
    norm = rel_posix.strip("/").replace("\\", "/")
    for raw_pat in allowlist:
        pat = raw_pat.strip().strip("/").replace("\\", "/")
        if not pat:
            continue
        if norm == pat or norm.startswith(pat + "/") or fnmatch.fnmatch(norm, pat):
            return True
    return False


def _strip_lean_comments(src: str) -> str:
    """Strips nested `/- ... -/` block comments and `-- ...` line comments from Lean 4 source."""
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    while i < n:
        if i + 1 < n and src[i] == "/" and src[i + 1] == "-":
            depth += 1
            i += 2
            continue
        if depth > 0:
            if i + 1 < n and src[i] == "-" and src[i + 1] == "/":
                depth -= 1
                i += 2
            else:
                if src[i] == "\n":
                    out.append("\n")
                i += 1
            continue
        if i + 1 < n and src[i] == "-" and src[i + 1] == "-":
            while i < n and src[i] != "\n":
                i += 1
            continue
        out.append(src[i])
        i += 1
    return "".join(out)


def _scan_lean_file_leaks(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [{"file": path.as_posix(), "line": 0, "severity": "CRITICAL", "code": "UNREADABLE_FILE", "detail": str(exc)}]
    cleaned = _strip_lean_comments(raw)
    lean_rules = [
        ("CRITICAL", "LEAN_SORRY_OR_ADMIT", re.compile(r"\b(?:sorry|admit)\b")),
        ("CRITICAL", "LEAN_CUSTOM_AXIOM", re.compile(r"^\s*axiom\b")),
        ("CRITICAL", "LEAN_NATIVE_DECIDE_OR_EXTERN", re.compile(r"\bnative_decide\b|@\[\s*(?:implemented_by|extern)\b")),
    ]
    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        for sev, code, pat in lean_rules:
            if pat.search(line):
                findings.append(
                    {"file": path.as_posix(), "line": lineno, "severity": sev, "code": code, "detail": line.strip()[:200]}
                )
    return findings


def _scan_unified_diff_leaks(
    diff_text: str,
    source_label: str,
    allowlist: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Audits a unified diff (.diff / .patch) for benchmark protocol mutations, allowlist escapes & test skips."""
    findings: list[dict[str, Any]] = []
    cur_file = source_label
    checked_diff_files: set[str] = set()
    for lineno, line in enumerate(diff_text.splitlines(), start=1):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            rel_f = line[6:].strip()
            if line.startswith("+++ b/"):
                cur_file = rel_f
            if rel_f and rel_f != "/dev/null" and rel_f not in checked_diff_files:
                checked_diff_files.add(rel_f)
                if allowlist and not _matches_allowlist(rel_f, allowlist):
                    findings.append(
                        {
                            "file": rel_f,
                            "line": lineno,
                            "severity": "CRITICAL",
                            "code": "UNAUTHORIZED_FILE_OUTSIDE_ALLOWLIST",
                            "detail": f"Modified path {rel_f!r} not in allowlist {allowlist}",
                        }
                    )
                base_lower = Path(rel_f).name.lower()
                if base_lower in _SENSITIVE_BUILD_OR_HOOK_NAMES or "/.git/hooks/" in ("/" + rel_f.replace("\\", "/")):
                    if not (allowlist and _matches_allowlist(rel_f, allowlist)):
                        findings.append(
                            {
                                "file": rel_f,
                                "line": lineno,
                                "severity": "HIGH",
                                "code": "BUILD_OR_STARTUP_HOOK_MODIFIED",
                                "detail": f"Build/startup hook {rel_f!r} modified in diff",
                            }
                        )
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:].strip()
        if re.search(r"\btrain_test_split\s*\(.*\b(?:stratify|test_size|random_state)\s*=", added):
            findings.append(
                {
                    "file": cur_file,
                    "line": lineno,
                    "severity": "CRITICAL",
                    "code": "DIFF_SPLIT_PROTOCOL_MUTATION",
                    "detail": added[:200],
                }
            )
        if re.search(r"(?:--|\b)(?:seq_len|pred_len|lookback|eps|epsilon|delta|significance)\s*(?:=|\s+)\S+", added):
            findings.append(
                {
                    "file": cur_file,
                    "line": lineno,
                    "severity": "HIGH",
                    "code": "DIFF_PROTOCOL_HYPERPARAM_OVERRIDE",
                    "detail": added[:200],
                }
            )
        if re.search(r"\bdef\s+(?:validate|evaluate|test|compute_coverage)\s*\(", added):
            findings.append(
                {
                    "file": cur_file,
                    "line": lineno,
                    "severity": "HIGH",
                    "code": "DIFF_EVALUATOR_FUNCTION_MODIFIED",
                    "detail": added[:200],
                }
            )
        for skip_pat in _TEST_SKIP_DIFF_PATTERNS:
            if skip_pat.search(added):
                findings.append(
                    {
                        "file": cur_file,
                        "line": lineno,
                        "severity": "HIGH",
                        "code": "TEST_SKIP_MARKER_ADDED",
                        "detail": added[:200],
                    }
                )
                break
        if Path(cur_file).name.lower() == "package.json" and re.search(r'"(?:preinstall|postinstall|prepare)"\s*:', added):
            findings.append(
                {
                    "file": cur_file,
                    "line": lineno,
                    "severity": "HIGH",
                    "code": "NPM_LIFECYCLE_HOOK_ADDED",
                    "detail": added[:200],
                }
            )
        for sev, code, pat in _LEAK_REGEX_RULES:
            if pat.search(added):
                findings.append(
                    {
                        "file": cur_file,
                        "line": lineno,
                        "severity": sev,
                        "code": code,
                        "detail": added[:200],
                    }
                )
    return findings


def _scan_python_file_leaks(
    path: Path,
    locked_constants: dict[str, str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [{"file": path.as_posix(), "line": 0, "severity": "CRITICAL", "code": "UNREADABLE_FILE", "detail": str(exc)}]

    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for sev, code, pat in _LEAK_REGEX_RULES:
            if pat.search(line):
                findings.append(
                    {
                        "file": path.as_posix(),
                        "line": lineno,
                        "severity": sev,
                        "code": code,
                        "detail": stripped[:200],
                    }
                )

    try:
        tree = ast.parse(src, filename=path.as_posix())
    except SyntaxError as exc:
        findings.append(
            {
                "file": path.as_posix(),
                "line": exc.lineno or 0,
                "severity": "CRITICAL",
                "code": "SYNTAX_ERROR",
                "detail": str(exc),
            }
        )
        return findings

    for node in ast.walk(tree):
        # Detect dynamic import / GC heap-inspection / builtins tampering
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "__import__" and node.args and isinstance(node.args[0], ast.Constant):
                mod_n = str(node.args[0].value)
                if mod_n in ("unittest.mock", "mock", "ctypes", "gc", "builtins"):
                    findings.append(
                        {
                            "file": path.as_posix(),
                            "line": getattr(node, "lineno", 0),
                            "severity": "CRITICAL",
                            "code": "DYNAMIC_IMPORT_SANDBOX_ESCAPE",
                            "detail": f"__import__({mod_n!r})",
                        }
                    )
            elif isinstance(fn, ast.Attribute) and fn.attr in ("get_objects", "get_referrers"):
                findings.append(
                    {
                        "file": path.as_posix(),
                        "line": getattr(node, "lineno", 0),
                        "severity": "CRITICAL",
                        "code": "GC_HEAP_INSPECTION_EXPLOIT",
                        "detail": f"Call to .{fn.attr}()",
                    }
                )
            elif isinstance(fn, ast.Attribute) and fn.attr == "import_module" and node.args and isinstance(node.args[0], ast.Constant):
                mod_n = str(node.args[0].value)
                if mod_n in ("unittest.mock", "mock", "ctypes", "gc", "builtins"):
                    findings.append(
                        {
                            "file": path.as_posix(),
                            "line": getattr(node, "lineno", 0),
                            "severity": "CRITICAL",
                            "code": "DYNAMIC_IMPORT_SANDBOX_ESCAPE",
                            "detail": f"import_module({mod_n!r})",
                        }
                    )

        # Detect attribute assignment onto eval/validator methods (`model.validate = ...` or `builtins.* = ...`)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in ("validate", "evaluate", "score_holdout"):
                    findings.append(
                        {
                            "file": path.as_posix(),
                            "line": getattr(node, "lineno", 0),
                            "severity": "CRITICAL",
                            "code": "METHOD_MONKEYPATCH_ASSIGN",
                            "detail": f"Assignment to .{target.attr}",
                        }
                    )
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in ("builtins", "__builtins__"):
                    findings.append(
                        {
                            "file": path.as_posix(),
                            "line": getattr(node, "lineno", 0),
                            "severity": "CRITICAL",
                            "code": "BUILTINS_MONKEYPATCH",
                            "detail": f"Assignment to {target.value.id}.{target.attr}",
                        }
                    )
                if locked_constants:
                    t_name: str | None = None
                    if isinstance(target, ast.Name):
                        t_name = target.id
                    elif isinstance(target, ast.Attribute):
                        t_name = target.attr
                    if t_name and t_name in locked_constants and isinstance(node.value, ast.Constant):
                        actual_val = str(node.value.value)
                        if actual_val != locked_constants[t_name]:
                            findings.append(
                                {
                                    "file": path.as_posix(),
                                    "line": getattr(node, "lineno", 0),
                                    "severity": "CRITICAL",
                                    "code": "LOCKED_PROTOCOL_OVERRIDE",
                                    "detail": f"{t_name} assigned {actual_val!r} != locked {locked_constants[t_name]!r}",
                                }
                            )
    return findings


def _check_directory_allowlist(cand_dir: Path, base_dir: Path, allowlist: list[str]) -> list[dict[str, Any]]:
    """Compares candidate directory against clean base directory and flags any modified/added/deleted file outside allowlist."""
    findings: list[dict[str, Any]] = []

    def _tree_hashes(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for sub in sorted(root.rglob("*")):
            if "__pycache__" in sub.parts or ".git" in sub.parts:
                continue
            if sub.is_file():
                rel = sub.relative_to(root).as_posix()
                out[rel] = _sha256_file(sub)
        return out

    base_h = _tree_hashes(base_dir.resolve())
    cand_h = _tree_hashes(cand_dir.resolve())
    all_rels = sorted(set(base_h.keys()) | set(cand_h.keys()))
    for rel in all_rels:
        if base_h.get(rel) != cand_h.get(rel):
            if not _matches_allowlist(rel, allowlist):
                change_kind = "added" if rel not in base_h else ("deleted" if rel not in cand_h else "modified")
                findings.append(
                    {
                        "file": rel,
                        "line": 0,
                        "severity": "CRITICAL",
                        "code": "UNAUTHORIZED_FILE_OUTSIDE_ALLOWLIST",
                        "detail": f"File {rel!r} was {change_kind} outside allowlist {allowlist}",
                    }
                )
    return findings


def _audit_paths_for_leaks(
    paths: list[str],
    locked_constants: dict[str, str],
    allowlist: list[str] | None = None,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen_files: set[Path] = set()
    for raw in paths:
        p = Path(raw).resolve()
        if not p.exists():
            findings.append({"file": raw, "line": 0, "severity": "CRITICAL", "code": "MISSING_PATH", "detail": "Path not found"})
            continue
        if p.is_file():
            if p.suffix.lower() in _UNTRUSTED_BINARY_OR_HOOK_SUFFIXES:
                findings.append(
                    {
                        "file": p.as_posix(),
                        "line": 0,
                        "severity": "CRITICAL",
                        "code": "UNTRUSTED_BINARY_OR_STARTUP_HOOK",
                        "detail": f"Binary/startup hook artifact {p.name!r} is forbidden in mutable candidate paths",
                    }
                )
            elif p.suffix.lower() in (".py", ".sh", ".yaml", ".yml", ".json", ".diff", ".patch", ".lean"):
                seen_files.add(p)
        elif p.is_dir():
            if allowlist and base_dir is not None:
                findings.extend(_check_directory_allowlist(p, base_dir, allowlist))
            for sub in sorted(p.rglob("*")):
                if "__pycache__" in sub.parts or ".git" in sub.parts:
                    continue
                if sub.is_file():
                    if sub.suffix.lower() in _UNTRUSTED_BINARY_OR_HOOK_SUFFIXES:
                        findings.append(
                            {
                                "file": sub.as_posix(),
                                "line": 0,
                                "severity": "CRITICAL",
                                "code": "UNTRUSTED_BINARY_OR_STARTUP_HOOK",
                                "detail": f"Binary/startup hook artifact {sub.name!r} is forbidden in mutable candidate paths",
                            }
                        )
                    elif sub.suffix.lower() in (".py", ".sh", ".diff", ".patch", ".lean"):
                        seen_files.add(sub)

    for fpath in sorted(seen_files):
        suf = fpath.suffix.lower()
        if suf in (".diff", ".patch"):
            txt = fpath.read_text(encoding="utf-8", errors="replace")
            if txt.strip():
                findings.extend(_scan_unified_diff_leaks(txt, fpath.as_posix(), allowlist=allowlist))
        elif suf == ".py":
            findings.extend(_scan_python_file_leaks(fpath, locked_constants))
        elif suf == ".lean":
            findings.extend(_scan_lean_file_leaks(fpath))
        elif suf == ".sh":
            txt = fpath.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(txt.splitlines(), start=1):
                for k, v in locked_constants.items():
                    m = re.search(rf"--{re.escape(k)}(?:\s+|=)(\S+)", line)
                    if m and m.group(1).strip("'\"") != v:
                        findings.append(
                            {
                                "file": fpath.as_posix(),
                                "line": lineno,
                                "severity": "CRITICAL",
                                "code": "SHELL_LOCKED_ARG_OVERRIDE",
                                "detail": f"--{k}={m.group(1)} != locked {v}",
                            }
                        )
    return findings


def cmd_audit_leaks(args: argparse.Namespace) -> int:
    locked: dict[str, str] = {}
    for kv in args.locked_constant or []:
        if "=" not in kv:
            print(json.dumps({"status": "ERROR", "error": f"Expected KEY=VALUE for --locked-constant, got {kv!r}"}))
            return 2
        k, v = kv.split("=", 1)
        locked[k.strip()] = v.strip()

    allowlist = [str(x) for x in (args.allowlist or []) if str(x).strip()] or None
    base_dir = Path(args.base_dir).resolve() if getattr(args, "base_dir", None) else None
    findings = _audit_paths_for_leaks(args.paths, locked, allowlist=allowlist, base_dir=base_dir)
    blocking_sevs = {"CRITICAL", "HIGH"} if not args.fail_on_warn else {"CRITICAL", "HIGH", "WARN"}
    blockers = [f for f in findings if f["severity"] in blocking_sevs]
    status = "CLEAN" if not blockers else "LEAK_DETECTED"
    report = {
        "status": status,
        "total_findings": len(findings),
        "blocking_findings": len(blockers),
        "findings": findings,
    }
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 1


def _load_tabular_rows(path: Path) -> list[dict[str, Any]]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return [_decode_json(line, path) for line in path.read_bytes().splitlines() if line.strip()]
    if suf == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    data = _load_strict_json(path)
    if isinstance(data, list):
        return [dict(r) for r in data]
    if isinstance(data, dict):
        for k in ("rows", "items", "records", "data"):
            if isinstance(data.get(k), list):
                return [dict(r) for r in data[k]]
    raise ValueError(f"Unsupported split file schema in {path}; use JSON list, JSONL, or CSV")


def cmd_audit_splits(args: argparse.Namespace) -> int:
    """Audit as-built split files for duplicate features, entity overlap and strict chronology."""
    splits: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for name, raw_p in (("train", args.train), ("val", args.val), ("confirm", args.confirm), ("holdout", args.holdout)):
        if not raw_p:
            continue
        p = Path(raw_p).resolve()
        if not p.exists():
            print(json.dumps({"status": "ERROR", "error": f"Split {name!r} missing: {p}"}))
            return 2
        splits.append((name, p, _load_tabular_rows(p)))

    names = [name for name, _, _ in splits]
    raw_counts = _load_strict_json(Path(args.raw_counts)) if args.raw_counts else {}
    if not isinstance(raw_counts, dict) or any(name not in names for name in raw_counts):
        raise ValueError("--raw-counts must be a JSON object keyed by supplied split names")
    for name, counts in raw_counts.items():
        if not isinstance(counts, dict):
            raise ValueError(f"Raw counts for {name} must contain n_users and n_items")
        for key in ("n_users", "n_items"):
            _require_positive_int(counts.get(key), f"raw_counts.{name}.{key}")

    label_keys = {str(k) for k in (args.exclude_keys or ["label", "target", "y"])}
    feat_keys = [str(k) for k in (args.feature_keys or []) if str(k).strip()]
    group_key = str(args.group_key) if args.group_key else None
    ts_key = str(args.timestamp_key) if args.timestamp_key else None
    user_key = args.user_key
    item_key = args.item_key
    if not all(x.strip() for x in (user_key, item_key, args.user_partition)):
        raise ValueError("User and item keys and the partition mode must be nonempty")
    require_disjoint_users = args.require_disjoint_users or args.user_partition.strip().upper() != "NONE"
    embargo = _require_nonneg_finite(args.embargo_gap, "embargo_gap")
    excluded = label_keys | {group_key, ts_key} | ({user_key} if require_disjoint_users else set())

    findings: list[dict[str, Any]] = []
    split_hashes: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    split_ts_bounds: dict[str, tuple[int | float, int | float]] = {}
    split_users: dict[str, set[str]] = {}
    split_sizes: dict[str, dict[str, Any]] = {}

    def entity_id(value: Any) -> str | None:
        if type(value) in (str, int) and str(value).strip():
            return str(value).strip()
        return None

    for s_name, _, rows in splits:
        h_set: set[str] = set()
        g_set: set[str] = set()
        u_set: set[str] = set()
        i_set: set[str] = set()
        ts_vals: list[int | float] = []
        internal_dups = missing_users = missing_items = missing_groups = missing_timestamps = 0
        for row in rows:
            if feat_keys:
                feat_obj = {k: row.get(k) for k in feat_keys}
            else:
                feat_obj = {k: v for k, v in row.items() if k not in excluded}
            if feat_obj:
                rh = hashlib.sha256(json.dumps(feat_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                if rh in h_set:
                    internal_dups += 1
                h_set.add(rh)
            if group_key:
                group = entity_id(row.get(group_key))
                if group is None:
                    missing_groups += 1
                else:
                    g_set.add(group)
            user = entity_id(row.get(user_key))
            if user is None:
                missing_users += 1
            else:
                u_set.add(user)
            item = entity_id(row.get(item_key))
            if item is None:
                missing_items += 1
            else:
                i_set.add(item)
            if ts_key:
                raw_stamp = row.get(ts_key)
                try:
                    if type(raw_stamp) is int:
                        stamp = raw_stamp
                    elif isinstance(raw_stamp, str) and re.fullmatch(r"-?\d+", raw_stamp.strip()):
                        stamp = int(raw_stamp)
                    else:
                        stamp = float(raw_stamp) if not isinstance(raw_stamp, bool) else math.nan
                except (ValueError, TypeError):
                    stamp = math.nan
                if type(stamp) is int or math.isfinite(stamp):
                    ts_vals.append(stamp)
                else:
                    missing_timestamps += 1
        split_hashes[s_name] = h_set
        split_groups[s_name] = g_set
        split_users[s_name] = u_set
        split_sizes[s_name] = {
            "n_rows": len(rows), "n_users": len(u_set), "n_items": len(i_set),
            "rows_missing_user_id": missing_users, "rows_missing_item_id": missing_items,
            "item_to_user_ratio": len(i_set) / len(u_set) if u_set and i_set and not (missing_users or missing_items) else None,
        }
        if s_name in raw_counts:
            raw = raw_counts[s_name]
            raw_ratio = raw["n_items"] / raw["n_users"]
            ratio = split_sizes[s_name]["item_to_user_ratio"]
            fold = ratio / raw_ratio if ratio is not None else None
            split_sizes[s_name].update(
                raw_n_users=raw["n_users"], raw_n_items=raw["n_items"],
                raw_item_to_user_ratio=raw_ratio, item_to_user_ratio_vs_raw=fold,
            )
            if fold is not None and fold > 2.0:
                findings.append({
                    "severity": "WARN", "code": "ITEM_USER_RATIO_DISTORTION", "split": s_name,
                    "raw_ratio": raw_ratio, "as_built_ratio": ratio, "fold": fold,
                })
        if ts_vals:
            split_ts_bounds[s_name] = (min(ts_vals), max(ts_vals))
            split_sizes[s_name].update(min_timestamp=min(ts_vals), max_timestamp=max(ts_vals))
        for code, count, required in (
            ("MISSING_USER_IDS", missing_users, require_disjoint_users),
            ("MISSING_GROUP_IDS", missing_groups, group_key is not None),
            ("MISSING_OR_INVALID_TIMESTAMPS", missing_timestamps, ts_key is not None),
            ("EMPTY_SPLIT", int(not rows), True),
        ):
            if required and count:
                findings.append({"severity": "CRITICAL", "code": code, "split": s_name, "count": count})
        if internal_dups > 0:
            findings.append(
                {
                    "severity": "WARN",
                    "code": "INTERNAL_DUPLICATE_FEATURE_ROWS",
                    "split": s_name,
                    "duplicate_count": internal_dups,
                }
            )

    pair_counts: list[dict[str, Any]] = []
    for a, b in itertools.combinations(names, 2):
        users_shared = len(split_users[a] & split_users[b])
        na, nb = len(split_users[a]), len(split_users[b])
        pair_report = {
            "splits": f"{a}_vs_{b}", "shared_users": users_shared,
            "earlier_n_users": na, "later_n_users": nb,
            "fraction_of_earlier_users": users_shared / na if na else None,
            "fraction_of_later_users": users_shared / nb if nb else None,
            "complete_user_coverage": not (split_sizes[a]["rows_missing_user_id"] or split_sizes[b]["rows_missing_user_id"]),
        }
        pair_counts.append(pair_report)
        if require_disjoint_users and users_shared:
            findings.append({"severity": "CRITICAL", "code": "USER_ID_INTERSECTION", **pair_report})
        ratio_a, ratio_b = split_sizes[a]["item_to_user_ratio"], split_sizes[b]["item_to_user_ratio"]
        if ratio_a is not None and ratio_b is not None:
            fold = max(ratio_a, ratio_b) / min(ratio_a, ratio_b)
            if fold > 2.0:
                findings.append({
                    "severity": "WARN", "code": "ITEM_USER_RATIO_DISTORTION",
                    "splits": f"{a}_vs_{b}", "earlier_ratio": ratio_a, "later_ratio": ratio_b, "fold": fold,
                })
        dup_cross = len(split_hashes[a] & split_hashes[b])
        if dup_cross:
            findings.append({"severity": "CRITICAL", "code": "EXACT_FEATURE_ROW_LEAKAGE",
                             "splits": f"{a}_vs_{b}", "shared_unique_feature_rows": dup_cross})
        if group_key:
            grp_cross = len(split_groups[a] & split_groups[b])
            if grp_cross:
                findings.append({"severity": "CRITICAL", "code": "GROUP_ENTITY_LEAKAGE",
                                 "splits": f"{a}_vs_{b}", "shared_groups": grp_cross})
        if ts_key and a in split_ts_bounds and b in split_ts_bounds:
            _, max_a = split_ts_bounds[a]
            min_b, _ = split_ts_bounds[b]
            if max_a >= min_b or min_b - max_a < embargo:
                findings.append({
                    "severity": "CRITICAL", "code": "TEMPORAL_LOOKAHEAD_LEAKAGE",
                    "splits": f"{a}_to_{b}", "earlier_max_ts": max_a,
                    "later_min_ts": min_b, "required_embargo": embargo,
                })

    blockers = [f for f in findings if f["severity"] in ("CRITICAL", "HIGH")]
    report = {
        "status": "CLEAN" if not blockers else "SPLIT_LEAK_DETECTED",
        "splits_checked": names,
        "disjoint_users_required": require_disjoint_users,
        "split_counts": split_sizes,
        "user_intersections": pair_counts,
        "blocking_findings": len(blockers),
        "findings": findings,
    }
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 1


# ---------------------------------------------------------------------------
# 4. Out-of-Process Isolated Runner (`run-isolated`)
# ---------------------------------------------------------------------------

_SAFE_ENV_PASSTHROUGH = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _scrubbed_isolated_env(work_tmp: Path, extra_pythonpath: Path | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    for k in _SAFE_ENV_PASSTHROUGH:
        if k in os.environ:
            env[k] = os.environ[k]
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["DATASETS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"
    env["CARGO_NET_OFFLINE"] = "true"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["GOPROXY"] = "off"
    env["NO_PROXY"] = "*"
    env["TMPDIR"] = work_tmp.as_posix()
    env["TEMP"] = str(work_tmp)
    env["TMP"] = str(work_tmp)
    if extra_pythonpath is not None:
        env["PYTHONPATH"] = str(extra_pythonpath)
    return env


def _is_inside_any(target: Path, roots: list[Path]) -> bool:
    for r in roots:
        try:
            target.relative_to(r)
            return True
        except ValueError:
            pass
    return False


def _materialize_clean_snapshot(
    src_root: Path,
    dst_root: Path,
    forbid_resolved: list[Path],
) -> None:
    """
    Copies workspace files into `dst_root` while stripping `.git`/VCS history (blocking `git log --all`
    benchmark leaks), `__pycache__`, `HOLDOUT_SEAL*`, `IMMUTABLE_MANIFEST*`, and any `--forbid-paths`.
    """
    src_real = src_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    skip_dirs = {".git", ".hg", ".svn", "__pycache__"}

    def _copy_tree(cur_src: Path, cur_dst: Path) -> None:
        if _is_inside_any(cur_src, forbid_resolved):
            return
        cur_dst.mkdir(parents=True, exist_ok=True)
        for name in sorted(os.listdir(cur_src)):
            if name in skip_dirs:
                continue
            if name.startswith("HOLDOUT_SEAL") or name.startswith("IMMUTABLE_MANIFEST"):
                continue
            child_s = cur_src / name
            if _is_inside_any(child_s.resolve() if not os.path.islink(child_s) else child_s, forbid_resolved):
                continue
            st = os.lstat(child_s)
            if stat.S_ISLNK(st.st_mode):
                link_t = (child_s.parent / os.readlink(child_s)).resolve()
                try:
                    link_t.relative_to(src_real)
                except ValueError:
                    raise ValueError(f"Symlink {child_s} escapes workspace root")
                if _is_inside_any(link_t, forbid_resolved):
                    continue
                if link_t.is_file():
                    shutil.copy2(link_t, cur_dst / name)
            elif stat.S_ISDIR(st.st_mode):
                _copy_tree(child_s, cur_dst / name)
            elif stat.S_ISREG(st.st_mode):
                shutil.copy2(child_s, cur_dst / name)

    _copy_tree(src_real, dst_root)


def _parse_executed_test_count(output_text: str) -> int | None:
    m_unit = re.search(r"\bRan\s+(\d+)\s+tests?\b", output_text)
    if m_unit:
        return int(m_unit.group(1))
    m_py = re.search(r"(\d+)\s+passed\b", output_text)
    if m_py:
        return int(m_py.group(1))
    return None


def _format_cmd_tokens(raw_cmd: str, subs: dict[str, str]) -> list[str]:
    raw_tokens = shlex.split(raw_cmd, posix=(os.name != "nt"))
    out_tokens: list[str] = []
    for tok in raw_tokens:
        if os.name == "nt" and len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
            tok = tok[1:-1]
        for k, v in subs.items():
            tok = tok.replace("{" + k + "}", v)
        out_tokens.append(tok)
    return out_tokens


def cmd_run_isolated(args: argparse.Namespace) -> int:
    """
    Executes untrusted candidates in scrubbed out-of-process sandboxes across four modes:
      - `predict`: single-script (`python -I -B`) or `.git`-free repo snapshot (`--workspace-root`)
      - `causal_online`: step-by-step causal stdio (`bin_packing` or hidden-reward `generic_step`)
      - `build_test_bench`: `.git`-free snapshot -> build & hash artifact -> trusted tests -> benchmark
      - `train_lock_score`: `.git`-free training -> non-pickle checkpoint lock -> isolated immutable scorer
    """
    mode = args.mode.lower()
    timeout_sec = _require_finite(args.timeout_sec, "timeout_sec")
    python_exe = args.python_exe or sys.executable
    forbid_resolved = [Path(p).resolve() for p in (args.forbid_paths or [])]
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_inside_any(output_path, forbid_resolved):
        print(json.dumps({"status": "ISOLATION_FAILED", "error": f"Output {output_path} lies inside forbidden path"}))
        return 1

    allowlist = [str(x) for x in (getattr(args, "allowlist", None) or []) if str(x).strip()] or None
    base_dir = Path(args.base_dir).resolve() if getattr(args, "base_dir", None) else None

    if mode in ("predict", "causal_online"):
        if not args.candidate_script or not args.input_json:
            print(json.dumps({"status": "ERROR", "error": f"--mode {mode} requires --candidate-script and --input-json"}))
            return 2
        script_path = Path(args.candidate_script).resolve()
        if not script_path.exists():
            print(json.dumps({"status": "ISOLATION_FAILED", "error": f"Candidate script missing: {script_path}"}))
            return 2
        input_path = Path(args.input_json).resolve()
        for checked in (script_path, input_path):
            if _is_inside_any(checked, forbid_resolved):
                print(
                    json.dumps(
                        {"status": "ISOLATION_FAILED", "error": f"Argument {checked} lies inside forbidden path"},
                        indent=2,
                    )
                )
                return 1

        audit_targets = [str(Path(args.workspace_root).resolve())] if getattr(args, "workspace_root", None) else [str(script_path)]
        if not args.skip_static_audit:
            leaks = _audit_paths_for_leaks(audit_targets, {}, allowlist=allowlist, base_dir=base_dir)
            crit = [x for x in leaks if x["severity"] in ("CRITICAL", "HIGH")]
            if crit:
                print(json.dumps({"status": "BLOCKED_BY_STATIC_AUDIT", "blocking_findings": crit}, indent=2))
                return 1

        with tempfile.TemporaryDirectory(prefix="gov_iso_") as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            snap_dir: Path | None = None
            exec_script = script_path
            if getattr(args, "workspace_root", None):
                ws_root = Path(args.workspace_root).resolve()
                snap_dir = tmp_path / "workspace_snap"
                _materialize_clean_snapshot(ws_root, snap_dir, forbid_resolved)
                try:
                    rel_script = script_path.relative_to(ws_root)
                    exec_script = snap_dir / rel_script
                except ValueError:
                    exec_script = script_path
            env = _scrubbed_isolated_env(tmp_path, extra_pythonpath=snap_dir)

            if mode == "predict":
                cand_out = tmp_path / "predictions_out.json"
                py_flags = ["-B", "-s"] if snap_dir is not None else ["-I", "-B"]
                cmd = [
                    python_exe,
                    *py_flags,
                    str(exec_script),
                    "--input-json",
                    str(input_path),
                    "--output-json",
                    str(cand_out),
                    *(args.extra_args or []),
                ]
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(snap_dir or tmp_path),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    print(json.dumps({"status": "TIMEOUT", "timeout_sec": timeout_sec}, indent=2))
                    return 1
                wall_sec = time.perf_counter() - t0

                if proc.returncode != 0 or not cand_out.exists():
                    print(
                        json.dumps(
                            {
                                "status": "CANDIDATE_CRASHED",
                                "returncode": proc.returncode,
                                "wall_clock_sec": wall_sec,
                                "stderr_tail": (proc.stderr or "")[-1000:],
                            },
                            indent=2,
                        )
                    )
                    return 1

                try:
                    pred_payload = _load_strict_json(cand_out)
                except Exception as exc:
                    print(json.dumps({"status": "INVALID_OUTPUT_JSON", "error": str(exc)}, indent=2))
                    return 1

                if not isinstance(pred_payload, dict) or "predictions" not in pred_payload:
                    print(json.dumps({"status": "INVALID_OUTPUT_SCHEMA", "error": "Output must contain 'predictions' dict or list"}))
                    return 1

                preds = pred_payload["predictions"]
                if args.expected_ids_json:
                    exp_ids = {str(x) for x in _load_strict_json(Path(args.expected_ids_json))}
                    got_ids = set(str(k) for k in preds.keys()) if isinstance(preds, dict) else {str(r.get("id")) for r in preds if isinstance(r, dict)}
                    if got_ids != exp_ids:
                        print(
                            json.dumps(
                                {
                                    "status": "PREDICTION_ID_MISMATCH",
                                    "missing_count": len(exp_ids - got_ids),
                                    "extra_count": len(got_ids - exp_ids),
                                },
                                indent=2,
                            )
                        )
                        return 1

                envelope = {
                    "status": "OK",
                    "mode": "predict",
                    "candidate_sha256": _sha256_file(script_path),
                    "guard_wall_clock_sec": wall_sec,
                    "predictions": preds,
                }
                output_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "status": "OK",
                            "mode": "predict",
                            "guard_wall_clock_sec": wall_sec,
                            "output_json": output_path.as_posix(),
                        },
                        indent=2,
                    )
                )
                return 0

            # mode == "causal_online"
            streams_payload = _load_strict_json(input_path)
            protocol = str(streams_payload.get("protocol", "bin_packing")).lower()
            streams = streams_payload.get("streams", [])
            if not isinstance(streams, list) or not streams:
                print(json.dumps({"status": "ERROR", "error": "causal_online input_json requires non-empty 'streams' list"}))
                return 2

            py_flags = ["-B", "-s"] if snap_dir is not None else ["-I", "-B"]
            cmd = [python_exe, *py_flags, str(exec_script), "--causal-stdio", *(args.extra_args or [])]
            t0 = time.perf_counter()
            proc_p = subprocess.Popen(
                cmd,
                cwd=str(snap_dir or tmp_path),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            decisions_all: list[dict[str, Any]] = []
            try:
                assert proc_p.stdin is not None and proc_p.stdout is not None
                if protocol == "generic_step":
                    primary_name = str(streams_payload.get("primary_metric", "mean_step_value"))
                    for st_obj in streams:
                        sid = str(st_obj["id"])
                        steps = st_obj.get("steps", [])
                        if not isinstance(steps, list) or not steps:
                            raise ValueError(f"stream[{sid}] requires non-empty 'steps'")
                        step_vals: list[float] = []
                        for step_idx, step_spec in enumerate(steps):
                            if (time.perf_counter() - t0) > timeout_sec:
                                print(json.dumps({"status": "TIMEOUT", "timeout_sec": timeout_sec}, indent=2))
                                return 1
                            action_values = step_spec.get("action_values")
                            if not isinstance(action_values, dict) or not action_values:
                                raise ValueError(f"stream[{sid}].steps[{step_idx}] missing 'action_values' dict")
                            valid_actions = step_spec.get("valid_actions", sorted(action_values.keys()))
                            # Expose ONLY causal observation & valid action names — NEVER action_values or future steps!
                            req_line = json.dumps(
                                {
                                    "stream_id": sid,
                                    "step": step_idx,
                                    "obs": step_spec.get("obs"),
                                    "valid_actions": valid_actions,
                                },
                                separators=(",", ":"),
                            )
                            proc_p.stdin.write(req_line + "\n")
                            proc_p.stdin.flush()
                            resp_raw = proc_p.stdout.readline()
                            if not resp_raw:
                                err_tail = (proc_p.stderr.read() if proc_p.stderr else "")[-800:]
                                print(json.dumps({"status": "CANDIDATE_EOF", "stderr_tail": err_tail}, indent=2))
                                return 1
                            resp = json.loads(resp_raw)
                            act_key = str(resp.get("action", ""))
                            if act_key not in action_values or act_key not in {str(a) for a in valid_actions}:
                                print(
                                    json.dumps(
                                        {
                                            "status": "ILLEGAL_CAUSAL_ACTION",
                                            "stream_id": sid,
                                            "step": step_idx,
                                            "action": act_key,
                                        },
                                        indent=2,
                                    )
                                )
                                return 1
                            step_vals.append(_require_finite(action_values[act_key], f"action_values[{act_key}]"))
                        decisions_all.append(
                            {
                                "id": sid,
                                "cluster_id": str(st_obj.get("cluster_id", sid)),
                                "stratum_id": str(st_obj.get("stratum_id", "default")),
                                "value": statistics.fmean(step_vals),
                            }
                        )
                    proc_p.stdin.close()
                    proc_p.wait(timeout=5.0)
                    wall_sec = time.perf_counter() - t0
                    mean_v = statistics.fmean(d["value"] for d in decisions_all)
                    envelope = {
                        "status": "OK",
                        "mode": "causal_online",
                        "protocol": "generic_step",
                        "candidate_sha256": _sha256_file(script_path),
                        "guard_wall_clock_sec": wall_sec,
                        "metrics": {primary_name: mean_v, "guard_wall_clock_sec": wall_sec},
                        "items": decisions_all,
                    }
                    output_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
                    print(
                        json.dumps(
                            {
                                "status": "OK",
                                "mode": "causal_online",
                                "protocol": "generic_step",
                                "streams_evaluated": len(decisions_all),
                                primary_name: mean_v,
                                "guard_wall_clock_sec": wall_sec,
                                "output_json": output_path.as_posix(),
                            },
                            indent=2,
                        )
                    )
                    return 0

                for st_obj in streams:
                    sid = str(st_obj["id"])
                    cap = _require_finite(st_obj["capacity"], f"stream[{sid}].capacity")
                    seq = [_require_finite(x, f"stream[{sid}].item") for x in st_obj["items"]]
                    bins: list[float] = []
                    chosen_bins: list[int] = []
                    for step_idx, item_size in enumerate(seq):
                        if (time.perf_counter() - t0) > timeout_sec:
                            print(json.dumps({"status": "TIMEOUT", "timeout_sec": timeout_sec}, indent=2))
                            return 1
                        req_line = json.dumps(
                            {"stream_id": sid, "step": step_idx, "item": item_size, "open_bins": bins, "capacity": cap},
                            separators=(",", ":"),
                        )
                        proc_p.stdin.write(req_line + "\n")
                        proc_p.stdin.flush()
                        resp_raw = proc_p.stdout.readline()
                        if not resp_raw:
                            err_tail = (proc_p.stderr.read() if proc_p.stderr else "")[-800:]
                            print(json.dumps({"status": "CANDIDATE_EOF", "stderr_tail": err_tail}, indent=2))
                            return 1
                        resp = json.loads(resp_raw)
                        b_idx = int(resp["bin_index"])
                        if b_idx == len(bins):
                            if item_size > cap + 1e-9:
                                raise ValueError(f"Item {item_size} exceeds capacity {cap}")
                            bins.append(cap - item_size)
                        elif 0 <= b_idx < len(bins) and bins[b_idx] + 1e-9 >= item_size:
                            bins[b_idx] -= item_size
                        else:
                            print(
                                json.dumps(
                                    {
                                        "status": "ILLEGAL_CAUSAL_ACTION",
                                        "stream_id": sid,
                                        "step": step_idx,
                                        "bin_index": b_idx,
                                        "num_open_bins": len(bins),
                                    },
                                    indent=2,
                                )
                            )
                            return 1
                        chosen_bins.append(b_idx)
                    l1_bound = math.ceil(sum(seq) / cap)
                    excess_ratio = (len(bins) - l1_bound) / max(1, l1_bound)
                    decisions_all.append(
                        {
                            "id": sid,
                            "cluster_id": str(st_obj.get("cluster_id", sid)),
                            "stratum_id": str(st_obj.get("stratum_id", "default")),
                            "bins_used": len(bins),
                            "l1_bound": l1_bound,
                            "value": excess_ratio,
                        }
                    )
                proc_p.stdin.close()
                proc_p.wait(timeout=5.0)
            except Exception as exc:
                print(json.dumps({"status": "CAUSAL_PROTOCOL_ERROR", "error": str(exc)}, indent=2))
                return 1
            finally:
                if proc_p.stdin is not None and not proc_p.stdin.closed:
                    with contextlib.suppress(BrokenPipeError):
                        proc_p.stdin.close()
                try:
                    proc_p.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc_p.kill()
                    proc_p.wait()
                for pipe in (proc_p.stdout, proc_p.stderr):
                    if pipe is not None:
                        pipe.close()
            wall_sec = time.perf_counter() - t0
            envelope = {
                "status": "OK",
                "mode": "causal_online",
                "candidate_sha256": _sha256_file(script_path),
                "guard_wall_clock_sec": wall_sec,
                "metrics": {
                    "excess_bin_ratio": statistics.fmean(d["value"] for d in decisions_all),
                    "guard_wall_clock_sec": wall_sec,
                },
                "items": decisions_all,
            }
            output_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": "OK",
                        "mode": "causal_online",
                        "streams_evaluated": len(decisions_all),
                        "excess_bin_ratio": envelope["metrics"]["excess_bin_ratio"],
                        "guard_wall_clock_sec": wall_sec,
                        "output_json": output_path.as_posix(),
                    },
                    indent=2,
                )
            )
            return 0

    elif mode == "build_test_bench":
        if not args.workspace_root or not args.test_cmd or not args.bench_cmd:
            print(
                json.dumps(
                    {
                        "status": "ERROR",
                        "error": "--mode build_test_bench requires --workspace-root, --test-cmd, and --bench-cmd",
                    }
                )
            )
            return 2
        ws_root = Path(args.workspace_root).resolve()
        if not ws_root.is_dir():
            print(json.dumps({"status": "ISOLATION_FAILED", "error": f"Workspace missing: {ws_root}"}))
            return 2
        if not args.skip_static_audit:
            leaks = _audit_paths_for_leaks([str(ws_root)], {}, allowlist=allowlist, base_dir=base_dir)
            crit = [x for x in leaks if x["severity"] in ("CRITICAL", "HIGH")]
            if crit:
                print(json.dumps({"status": "BLOCKED_BY_STATIC_AUDIT", "blocking_findings": crit}, indent=2))
                return 1

        with tempfile.TemporaryDirectory(prefix="gov_btb_") as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            repo_snap = tmp_path / "repo_snap"
            _materialize_clean_snapshot(ws_root, repo_snap, forbid_resolved)
            if args.trusted_tests_dir:
                tt_src = Path(args.trusted_tests_dir).resolve()
                tt_rel = args.trusted_tests_relpath or tt_src.name
                tt_dst = repo_snap / tt_rel
                if tt_dst.exists():
                    shutil.rmtree(tt_dst)
                _materialize_clean_snapshot(tt_src, tt_dst, [])

            env = _scrubbed_isolated_env(tmp_path, extra_pythonpath=repo_snap)
            subs = {"workspace": str(repo_snap), "python": python_exe}

            built_sha: str | None = None
            build_sec = 0.0
            if args.build_cmd:
                t_b0 = time.perf_counter()
                proc_b = subprocess.run(
                    _format_cmd_tokens(args.build_cmd, subs),
                    cwd=str(repo_snap),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                build_sec = time.perf_counter() - t_b0
                if proc_b.returncode != 0:
                    print(
                        json.dumps(
                            {
                                "status": "BUILD_FAILED",
                                "returncode": proc_b.returncode,
                                "stderr_tail": (proc_b.stderr or "")[-1000:],
                            },
                            indent=2,
                        )
                    )
                    return 1
                if args.artifact_glob:
                    matched_arts = sorted(p for p in repo_snap.glob(args.artifact_glob) if p.is_file())
                    if not matched_arts:
                        print(json.dumps({"status": "ARTIFACT_MISSING", "glob": args.artifact_glob}, indent=2))
                        return 1
                    built_sha = _sha256_artifacts([str(p) for p in matched_arts], repo_snap)["artifact_bundle_sha256"]

            t_t0 = time.perf_counter()
            proc_t = subprocess.run(
                _format_cmd_tokens(args.test_cmd, subs),
                cwd=str(repo_snap),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            test_sec = time.perf_counter() - t_t0
            combined_t_out = (proc_t.stdout or "") + "\n" + (proc_t.stderr or "")
            if proc_t.returncode != 0:
                print(
                    json.dumps(
                        {
                            "status": "TESTS_FAILED",
                            "returncode": proc_t.returncode,
                            "output_tail": combined_t_out[-1000:],
                        },
                        indent=2,
                    )
                )
                return 1
            tests_exec = _parse_executed_test_count(combined_t_out)
            if args.min_tests_executed > 0:
                if tests_exec is None or tests_exec < args.min_tests_executed:
                    print(
                        json.dumps(
                            {
                                "status": "TEST_COUNT_REGRESSION",
                                "tests_executed": tests_exec,
                                "min_tests_executed": args.min_tests_executed,
                            },
                            indent=2,
                        )
                    )
                    return 1

            bench_out = tmp_path / "bench_out.json"
            subs_bench = {**subs, "output_json": str(bench_out)}
            bench_tokens = _format_cmd_tokens(args.bench_cmd, subs_bench)
            if "{output_json}" not in args.bench_cmd:
                bench_tokens.extend(["--output-json", str(bench_out)])

            t_m0 = time.perf_counter()
            proc_m = subprocess.run(
                bench_tokens,
                cwd=str(repo_snap),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            bench_sec = time.perf_counter() - t_m0
            if proc_m.returncode != 0 or not bench_out.exists():
                print(
                    json.dumps(
                        {
                            "status": "BENCH_FAILED",
                            "returncode": proc_m.returncode,
                            "stderr_tail": (proc_m.stderr or "")[-1000:],
                        },
                        indent=2,
                    )
                )
                return 1

            if built_sha is not None and args.artifact_glob:
                post_arts = sorted(p for p in repo_snap.glob(args.artifact_glob) if p.is_file())
                post_sha = _sha256_artifacts([str(p) for p in post_arts], repo_snap)["artifact_bundle_sha256"]
                if post_sha != built_sha:
                    print(json.dumps({"status": "ARTIFACT_MUTATED_DURING_BENCH", "expected": built_sha, "actual": post_sha}))
                    return 1

            bench_payload = _load_strict_json(bench_out)
            metrics = dict(bench_payload.get("metrics", {}))
            metrics.setdefault("unit_pass_rate", 1.0)
            metrics["guard_build_sec"] = build_sec
            metrics["guard_test_sec"] = test_sec
            metrics["guard_wall_clock_sec"] = bench_sec
            if tests_exec is not None:
                metrics["tests_executed"] = float(tests_exec)

            envelope = {
                **bench_payload,
                "status": "OK",
                "mode": "build_test_bench",
                "built_artifact_sha256": built_sha,
                "guard_wall_clock_sec": bench_sec,
                "metrics": metrics,
            }
            output_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": "OK",
                        "mode": "build_test_bench",
                        "tests_executed": tests_exec,
                        "built_artifact_sha256": built_sha,
                        "guard_wall_clock_sec": bench_sec,
                        "output_json": output_path.as_posix(),
                    },
                    indent=2,
                )
            )
            return 0

    elif mode == "train_lock_score":
        if not args.workspace_root or not args.train_cmd or not args.checkpoint_relpath or not args.score_cmd:
            print(
                json.dumps(
                    {
                        "status": "ERROR",
                        "error": "--mode train_lock_score requires --workspace-root, --train-cmd, --checkpoint-relpath, and --score-cmd",
                    }
                )
            )
            return 2
        ws_root = Path(args.workspace_root).resolve()
        if not ws_root.is_dir():
            print(json.dumps({"status": "ISOLATION_FAILED", "error": f"Workspace missing: {ws_root}"}))
            return 2
        if not args.skip_static_audit:
            leaks = _audit_paths_for_leaks([str(ws_root)], {}, allowlist=allowlist, base_dir=base_dir)
            crit = [x for x in leaks if x["severity"] in ("CRITICAL", "HIGH")]
            if crit:
                print(json.dumps({"status": "BLOCKED_BY_STATIC_AUDIT", "blocking_findings": crit}, indent=2))
                return 1

        unsafe_ckpt_suffixes = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".joblib", ".dill"}
        forbidden_ckpt_splits = {"confirm", "dev_confirm", "holdout", "test", "sealed_holdout"}

        with tempfile.TemporaryDirectory(prefix="gov_tls_") as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            train_snap = tmp_path / "train_snap"
            # Phase 1: Train inside `.git`-free snapshot with Confirm/Holdout & immutable_eval stripped!
            _materialize_clean_snapshot(ws_root, train_snap, forbid_resolved)
            env_train = _scrubbed_isolated_env(tmp_path, extra_pythonpath=train_snap)
            subs_train = {"workspace": str(train_snap), "python": python_exe}
            t_tr0 = time.perf_counter()
            proc_tr = subprocess.run(
                _format_cmd_tokens(args.train_cmd, subs_train),
                cwd=str(train_snap),
                env=env_train,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            train_sec = time.perf_counter() - t_tr0
            if proc_tr.returncode != 0:
                print(
                    json.dumps(
                        {
                            "status": "TRAIN_FAILED",
                            "returncode": proc_tr.returncode,
                            "stderr_tail": (proc_tr.stderr or "")[-1000:],
                        },
                        indent=2,
                    )
                )
                return 1

            # Phase 2: Validate safe non-pickle checkpoint format, selection hygiene & lock SHA-256
            ckpt_src = (train_snap / args.checkpoint_relpath).resolve()
            try:
                ckpt_src.relative_to(train_snap)
            except ValueError:
                print(json.dumps({"status": "ISOLATION_FAILED", "error": "checkpoint_relpath escapes train snapshot"}))
                return 1
            if not ckpt_src.exists():
                print(json.dumps({"status": "CHECKPOINT_MISSING", "path": args.checkpoint_relpath}, indent=2))
                return 1

            ckpt_files = [ckpt_src] if ckpt_src.is_file() else sorted(p for p in ckpt_src.rglob("*") if p.is_file())
            if not ckpt_files:
                print(json.dumps({"status": "CHECKPOINT_EMPTY", "path": args.checkpoint_relpath}, indent=2))
                return 1

            if not args.allow_unsafe_pickle_checkpoints:
                bad_ckpts = [p.name for p in ckpt_files if p.suffix.lower() in unsafe_ckpt_suffixes]
                if bad_ckpts:
                    print(
                        json.dumps(
                            {
                                "status": "UNSAFE_PICKLE_CHECKPOINT_FORBIDDEN",
                                "files": bad_ckpts,
                                "detail": "Use safe data-only checkpoint formats (.safetensors, .npz, .json, .onnx) so scoring cannot execute pickled code.",
                            },
                            indent=2,
                        )
                    )
                    return 1

            meta_candidates = [p for p in ckpt_files if p.name == "checkpoint_meta.json"]
            for mp in meta_candidates:
                meta_obj = _load_strict_json(mp)
                sel_split = str(meta_obj.get("selected_on_split", "")).lower().strip()
                if sel_split in forbidden_ckpt_splits:
                    print(
                        json.dumps(
                            {
                                "status": "CHECKPOINT_PEEKING_FORBIDDEN",
                                "selected_on_split": sel_split,
                                "detail": "Checkpoints must be selected on a train-carved inner validation split, never Confirm/Holdout/Test.",
                            },
                            indent=2,
                        )
                    )
                    return 1

            locked_ckpt_dir = tmp_path / "locked_ckpt"
            locked_ckpt_dir.mkdir(parents=True, exist_ok=True)
            if ckpt_src.is_file():
                locked_ckpt_target = locked_ckpt_dir / ckpt_src.name
                shutil.copy2(ckpt_src, locked_ckpt_target)
            else:
                locked_ckpt_target = locked_ckpt_dir / "checkpoint"
                shutil.copytree(ckpt_src, locked_ckpt_target)
            ckpt_sha = _sha256_artifacts([str(locked_ckpt_target)], locked_ckpt_dir)["artifact_bundle_sha256"]

            # Physically remove mutable candidate code BEFORE launching the immutable scorer!
            shutil.rmtree(train_snap)

            # Phase 3: Execute immutable out-of-process scorer on locked checkpoint
            score_cwd = tmp_path / "score_cwd"
            score_cwd.mkdir(parents=True, exist_ok=True)
            score_out = tmp_path / "score_out.json"
            env_score = _scrubbed_isolated_env(score_cwd)
            subs_score = {
                "checkpoint_path": str(locked_ckpt_target),
                "output_json": str(score_out),
                "python": python_exe,
            }
            score_tokens = _format_cmd_tokens(args.score_cmd, subs_score)
            if "{output_json}" not in args.score_cmd:
                score_tokens.extend(["--output-json", str(score_out)])

            t_sc0 = time.perf_counter()
            proc_sc = subprocess.run(
                score_tokens,
                cwd=str(score_cwd),
                env=env_score,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            score_sec = time.perf_counter() - t_sc0
            if proc_sc.returncode != 0 or not score_out.exists():
                print(
                    json.dumps(
                        {
                            "status": "SCORE_FAILED",
                            "returncode": proc_sc.returncode,
                            "stderr_tail": (proc_sc.stderr or "")[-1000:],
                        },
                        indent=2,
                    )
                )
                return 1

            post_ckpt_sha = _sha256_artifacts([str(locked_ckpt_target)], locked_ckpt_dir)["artifact_bundle_sha256"]
            if post_ckpt_sha != ckpt_sha:
                print(json.dumps({"status": "CHECKPOINT_MUTATED_DURING_SCORING", "expected": ckpt_sha, "actual": post_ckpt_sha}))
                return 1

            score_payload = _load_strict_json(score_out)
            metrics = dict(score_payload.get("metrics", {}))
            metrics["train_wall_clock_sec"] = train_sec
            metrics["score_wall_clock_sec"] = score_sec
            metrics["guard_wall_clock_sec"] = train_sec + score_sec

            envelope = {
                **score_payload,
                "status": "OK",
                "mode": "train_lock_score",
                "checkpoint_sha256": ckpt_sha,
                "train_wall_clock_sec": train_sec,
                "score_wall_clock_sec": score_sec,
                "guard_wall_clock_sec": train_sec + score_sec,
                "metrics": metrics,
            }
            output_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": "OK",
                        "mode": "train_lock_score",
                        "checkpoint_sha256": ckpt_sha,
                        "train_wall_clock_sec": train_sec,
                        "score_wall_clock_sec": score_sec,
                        "output_json": output_path.as_posix(),
                    },
                    indent=2,
                )
            )
            return 0

    else:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": f"Unsupported mode={mode!r}; use predict, causal_online, build_test_bench, or train_lock_score",
                }
            )
        )
        return 2


# ---------------------------------------------------------------------------
# 5. Unified Paired Resampling Engine (1D Cluster, 2-Way Crossed, Strata & Counts)
# ---------------------------------------------------------------------------


def _compute_scalar_stat(values: list[float], stat_name: str) -> float:
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
    raise ValueError(
        f"Unsupported statistic={stat_name!r}; use mean, median, p90, p95, p99, "
        "macro_f1_from_counts, micro_f1_from_counts, or ratio_from_sums"
    )


def _load_items_sidecar(sidecar_spec: dict[str, Any], label: str, base_dir: Path | None = None) -> list[dict[str, Any]]:
    raw_p = sidecar_spec.get("path")
    exp_sha = str(sidecar_spec.get("sha256", "")).strip().lower()
    if not raw_p or len(exp_sha) != 64:
        raise ValueError(f"{label}.items_sidecar requires 'path' and 64-hex 'sha256'")
    cand_p = Path(str(raw_p))
    sc_path = cand_p.resolve() if cand_p.is_absolute() else ((base_dir or Path.cwd()) / cand_p).resolve()
    if not sc_path.exists():
        raise ValueError(f"{label}.items_sidecar missing: {sc_path}")
    actual_sha = _sha256_file(sc_path).lower()
    if actual_sha != exp_sha:
        raise ValueError(
            f"{label}.items_sidecar SHA-256 mismatch on {sc_path.name}: expected={exp_sha[:16]} != actual={actual_sha[:16]}"
        )
    low_name = sc_path.name.lower()
    if low_name.endswith(".npz"):
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(f"{label}.items_sidecar uses .npz but numpy is unavailable; use .jsonl.gz or install numpy") from exc
        with np.load(sc_path, allow_pickle=False) as z:
            ids_arr = z["id"]
            n_rows = int(ids_arr.shape[0])
            cl_arr = z["cluster_id"] if "cluster_id" in z.files else ids_arr
            sd_arr = z["seed_id"] if "seed_id" in z.files else None
            st_arr = z["stratum_id"] if "stratum_id" in z.files else None
            cls_arr = z["class_id"] if "class_id" in z.files else None
            val_arr = z["value"] if "value" in z.files else None
            tp_arr = z["tp"] if "tp" in z.files else None
            fp_arr = z["fp"] if "fp" in z.files else None
            fn_arr = z["fn"] if "fn" in z.files else None
            num_arr = z["num"] if "num" in z.files else None
            den_arr = z["den"] if "den" in z.files else None
            aux_keys = [k[4:] for k in z.files if k.startswith("aux_")]
            aux_cols = {ak: z[f"aux_{ak}"] for ak in aux_keys}
            out_rows: list[dict[str, Any]] = []
            for idx in range(n_rows):
                row_d: dict[str, Any] = {
                    "id": str(ids_arr[idx]),
                    "cluster_id": str(cl_arr[idx]),
                }
                if sd_arr is not None:
                    sv = str(sd_arr[idx])
                    if sv != "":
                        row_d["seed_id"] = sv
                if st_arr is not None:
                    stv = str(st_arr[idx])
                    if stv != "":
                        row_d["stratum_id"] = stv
                if cls_arr is not None:
                    row_d["class_id"] = str(cls_arr[idx])
                if val_arr is not None:
                    row_d["value"] = float(val_arr[idx])
                if tp_arr is not None and fp_arr is not None and fn_arr is not None:
                    row_d["tp"] = float(tp_arr[idx])
                    row_d["fp"] = float(fp_arr[idx])
                    row_d["fn"] = float(fn_arr[idx])
                if num_arr is not None and den_arr is not None:
                    row_d["num"] = float(num_arr[idx])
                    row_d["den"] = float(den_arr[idx])
                if aux_cols:
                    row_d["aux"] = {ak: float(col[idx]) for ak, col in aux_cols.items()}
                out_rows.append(row_d)
            return out_rows
    if low_name.endswith(".jsonl") or low_name.endswith(".jsonl.gz"):
        opener = gzip.open if low_name.endswith(".gz") else open
        with opener(sc_path, "rt", encoding="utf-8") as f_in:
            return [json.loads(line) for line in f_in if line.strip()]
    if low_name.endswith(".csv") or low_name.endswith(".csv.gz"):
        opener = gzip.open if low_name.endswith(".gz") else open
        with opener(sc_path, "rt", encoding="utf-8", newline="") as f_csv:
            parsed_csv: list[dict[str, Any]] = []
            for raw_r in csv.DictReader(f_csv):
                d_r: dict[str, Any] = {"id": raw_r["id"], "cluster_id": raw_r.get("cluster_id") or raw_r["id"]}
                for opt_s in ("seed_id", "stratum_id", "class_id"):
                    if raw_r.get(opt_s):
                        d_r[opt_s] = raw_r[opt_s]
                for opt_f in ("value", "tp", "fp", "fn", "num", "den"):
                    if raw_r.get(opt_f) not in (None, ""):
                        d_r[opt_f] = float(raw_r[opt_f])
                aux_d = {k[4:]: float(v) for k, v in raw_r.items() if k.startswith("aux_") and v not in (None, "")}
                if aux_d:
                    d_r["aux"] = aux_d
                parsed_csv.append(d_r)
            return parsed_csv
    loaded = _load_strict_json(sc_path)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
        return loaded["items"]
    raise ValueError(f"Unsupported items_sidecar format in {sc_path}")


def _extract_structured_items(
    payload: dict[str, Any],
    label: str,
    stat_name: str,
    base_dir: Path | None = None,
    item_cluster_key: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    """
    Normalizes keyed items (inline `items` or SHA-256-verified `items_sidecar`) into:
      id -> {
        "id": str,
        "cluster_id": str,
        "seed_id": str | None,
        "stratum_id": str | None,
        "class_id": str,
        "value": float,        # for scalar stats
        "tp": float, "fp": float, "fn": float,  # for *_f1_from_counts
        "num": float, "den": float,             # for ratio_from_sums
      }
    """
    raw = payload.get("items")
    if raw is None and isinstance(payload.get("items_sidecar"), dict):
        raw = _load_items_sidecar(payload["items_sidecar"], label, base_dir=base_dir)
    if raw is None:
        return None
    s_lower = stat_name.lower()
    needs_counts = s_lower in ("macro_f1_from_counts", "micro_f1_from_counts")
    needs_ratio = s_lower == "ratio_from_sums"

    records: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        if needs_counts or needs_ratio or item_cluster_key:
            raise ValueError(f"{label}.items must be a list of objects when primary_statistic={stat_name!r}")
        for k, v in raw.items():
            sid = str(k)
            records[sid] = {
                "id": sid,
                "cluster_id": sid,
                "seed_id": None,
                "stratum_id": None,
                "class_id": "0",
                "value": _require_finite(v, f"{label}.items[{sid}]"),
            }
        return records

    if isinstance(raw, list):
        for idx, row in enumerate(raw):
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"{label}.items[{idx}] must be an object with 'id'")
            sid = str(row["id"])
            if sid in records:
                raise ValueError(f"Duplicate item id {sid!r} in {label}.items")
            rec: dict[str, Any] = {
                "id": sid,
                "cluster_id": str(row.get("cluster_id", sid)),
                "seed_id": str(row["seed_id"]) if row.get("seed_id") is not None else None,
                "stratum_id": str(row["stratum_id"]) if row.get("stratum_id") is not None else (
                    str(row["cell_id"]) if row.get("cell_id") is not None else None
                ),
                "class_id": str(row.get("class_id", "0")),
            }
            if item_cluster_key or "item_cluster_id" in row:
                key = item_cluster_key or "item_cluster_id"
                item = row.get(key)
                user = row.get("cluster_id")
                if type(item) not in (str, int) or not str(item).strip():
                    raise ValueError(f"{label}.items[{sid}] requires a nonempty {key!r}")
                if type(user) not in (str, int) or not str(user).strip():
                    raise ValueError(f"{label}.items[{sid}] requires an explicit user cluster_id")
                rec["item_cluster_id"] = str(item)
            if needs_counts:
                rec["tp"] = _require_nonneg_finite(row.get("tp"), f"{label}.items[{sid}].tp")
                rec["fp"] = _require_nonneg_finite(row.get("fp"), f"{label}.items[{sid}].fp")
                rec["fn"] = _require_nonneg_finite(row.get("fn"), f"{label}.items[{sid}].fn")
                denom = 2.0 * rec["tp"] + rec["fp"] + rec["fn"]
                rec["value"] = (2.0 * rec["tp"] / denom) if denom > 0.0 else 0.0
            elif needs_ratio:
                rec["num"] = _require_finite(row.get("num"), f"{label}.items[{sid}].num")
                rec["den"] = _require_nonneg_finite(row.get("den"), f"{label}.items[{sid}].den")
                rec["value"] = (rec["num"] / rec["den"]) if rec["den"] > 0.0 else 0.0
            else:
                if "value" not in row:
                    raise ValueError(f"{label}.items[{idx}] missing required 'value'")
                rec["value"] = _require_finite(row["value"], f"{label}.items[{sid}].value")
            if isinstance(row.get("aux"), dict):
                rec["aux"] = {str(ak): _require_finite(av, f"{label}.items[{sid}].aux[{ak}]") for ak, av in row["aux"].items()}
            else:
                rec["aux"] = {}
            records[sid] = rec
        return records

    raise ValueError(f"{label}.items must be a dict or list of item objects")


def _verify_item_structure_parity(
    cand_recs: dict[str, dict[str, Any]],
    ref_recs: dict[str, dict[str, Any]],
    ref_label: str,
) -> None:
    if set(cand_recs.keys()) != set(ref_recs.keys()):
        missing_c = sorted(set(ref_recs.keys()) - set(cand_recs.keys()))[:5]
        extra_c = sorted(set(cand_recs.keys()) - set(ref_recs.keys()))[:5]
        raise ValueError(f"Item ID mismatch vs {ref_label}: missing={missing_c}, extra={extra_c}")
    if not cand_recs:
        raise ValueError("Item list cannot be empty")
    for iid, cr in cand_recs.items():
        rr = ref_recs[iid]
        for meta_key in ("cluster_id", "seed_id", "stratum_id", "class_id", "item_cluster_id"):
            if cr.get(meta_key) != rr.get(meta_key):
                raise ValueError(
                    f"Item metadata tamper/mismatch on id={iid!r} field {meta_key!r}: "
                    f"candidate={cr.get(meta_key)!r} != {ref_label}={rr.get(meta_key)!r}"
                )


def _aggregate_records(
    recs: list[dict[str, Any]],
    stat_name: str,
    weighting: str,
    expected_strata: list[str] | None = None,
    stratum_multiplicity: dict[str, float] | None = None,
) -> float:
    if not recs:
        raise ValueError("Cannot aggregate empty record sample")
    s = stat_name.lower()
    equal_strata = weighting == "equal_stratum"

    def _weighted_stratum_mean(st_to_val: dict[str, float], keys: list[str]) -> float:
        present = [k for k in keys if k in st_to_val]
        if not present:
            raise ValueError("No non-empty strata in equal_stratum aggregation")
        if stratum_multiplicity:
            ws = [float(stratum_multiplicity.get(k, 1.0)) for k in present]
            w_sum = sum(ws)
            if w_sum > 0.0:
                return sum(st_to_val[k] * w for k, w in zip(present, ws)) / w_sum
        return statistics.fmean(st_to_val[k] for k in present)

    if s == "macro_f1_from_counts":
        # Group by stratum_id (or single global stratum) -> class_id -> sum(tp, fp, fn)
        by_stratum: dict[str, dict[str, list[float]]] = {}
        for r in recs:
            st = r["stratum_id"] if (equal_strata and r["stratum_id"] is not None) else "__all__"
            cl = r["class_id"]
            acc = by_stratum.setdefault(st, {}).setdefault(cl, [0.0, 0.0, 0.0])
            acc[0] += r["tp"]
            acc[1] += r["fp"]
            acc[2] += r["fn"]
        strata_keys = expected_strata if (equal_strata and expected_strata) else sorted(by_stratum.keys())
        st_f1_map: dict[str, float] = {}
        for st in strata_keys:
            class_map = by_stratum.get(st)
            if not class_map:
                continue
            cf1s: list[float] = []
            for cl in sorted(class_map.keys()):
                tp, fp, fn = class_map[cl]
                denom = 2.0 * tp + fp + fn
                cf1s.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
            st_f1_map[st] = statistics.fmean(cf1s)
        return _weighted_stratum_mean(st_f1_map, strata_keys)

    if s == "micro_f1_from_counts":
        tp = sum(r["tp"] for r in recs)
        fp = sum(r["fp"] for r in recs)
        fn = sum(r["fn"] for r in recs)
        denom = 2.0 * tp + fp + fn
        return (2.0 * tp / denom) if denom > 0.0 else 0.0

    if s == "ratio_from_sums":
        if equal_strata:
            by_st: dict[str, list[float]] = {}
            for r in recs:
                st = r["stratum_id"] if r["stratum_id"] is not None else "__all__"
                acc = by_st.setdefault(st, [0.0, 0.0])
                acc[0] += r["num"]
                acc[1] += r["den"]
            st_ratios: dict[str, float] = {}
            for st, (num, den) in sorted(by_st.items()):
                if den <= 0.0:
                    raise ValueError(f"Zero denominator in ratio_from_sums stratum {st!r}")
                st_ratios[st] = num / den
            s_keys = expected_strata if expected_strata else sorted(st_ratios.keys())
            return _weighted_stratum_mean(st_ratios, s_keys)
        num = sum(r["num"] for r in recs)
        den = sum(r["den"] for r in recs)
        if den <= 0.0:
            raise ValueError("Zero total denominator in ratio_from_sums")
        return num / den

    if equal_strata:
        by_st_vals: dict[str, list[float]] = {}
        for r in recs:
            st = r["stratum_id"] if r["stratum_id"] is not None else "__all__"
            by_st_vals.setdefault(st, []).append(r["value"])
        strata_keys = expected_strata if expected_strata else sorted(by_st_vals.keys())
        st_val_map = {st: _compute_scalar_stat(by_st_vals[st], s) for st in strata_keys if st in by_st_vals}
        return _weighted_stratum_mean(st_val_map, strata_keys)

    return _compute_scalar_stat([r["value"] for r in recs], s)


_SUFFICIENT_STAT_NAMES = {"mean", "ratio_from_sums", "micro_f1_from_counts", "macro_f1_from_counts"}


def _record_cell(row: dict[str, Any]) -> tuple[str, ...]:
    cell = (row["seed_id"] if row["seed_id"] is not None else "__noseed__", row["cluster_id"])
    item = row.get("item_cluster_id")
    return cell if item is None else (*cell, item)


def _build_cell_sufficient_stats(
    all_ids: list[str],
    cand_recs: dict[str, dict[str, Any]],
    ref_recs: dict[str, dict[str, Any]],
    stat_name: str,
    weighting: str,
) -> dict[str, Any]:
    """
    Pre-reduces N item records in a single O(N) pass into additive sufficient statistics per
    `(seed_id, cluster_id)` cell (and stratum / class when applicable), enabling O(S*G*|S|)
    leave-one-cell-out jackknifing and 4,000-draw crossed bootstrap with exact IEEE-754 parity.
    """
    s = stat_name.lower()
    equal_strata = weighting == "equal_stratum"
    cell_counts: dict[tuple[str, ...], int] = {}

    if s == "mean":
        # cell -> st -> (sum_cand, sum_ref, count)
        mean_cells: dict[tuple[str, ...], dict[str, tuple[float, float, float]]] = {}
        tmp_c: dict[tuple[str, ...], dict[str, list[float]]] = {}
        tmp_r: dict[tuple[str, ...], dict[str, list[float]]] = {}
        for iid in all_ids:
            cr = cand_recs[iid]
            rr = ref_recs[iid]
            cell = _record_cell(cr)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            st = (cr["stratum_id"] if cr["stratum_id"] is not None else "__all__") if equal_strata else "__all__"
            tmp_c.setdefault(cell, {}).setdefault(st, []).append(cr["value"])
            tmp_r.setdefault(cell, {}).setdefault(st, []).append(rr["value"])
        tot_mean: dict[str, list[float]] = {}
        for cell, st_map in tmp_c.items():
            mean_cells[cell] = {}
            for st, c_list in st_map.items():
                r_list = tmp_r[cell][st]
                sc_v = math.fsum(c_list)
                sr_v = math.fsum(r_list)
                cnt = float(len(c_list))
                mean_cells[cell][st] = (sc_v, sr_v, cnt)
                acc = tot_mean.setdefault(st, [0.0, 0.0, 0.0])
                acc[0] += sc_v
                acc[1] += sr_v
                acc[2] += cnt
        return {"kind": "mean", "equal_strata": equal_strata, "cells": mean_cells, "totals": tot_mean, "cell_counts": cell_counts}

    if s == "ratio_from_sums":
        # cell -> st -> (num_c, den_c, num_r, den_r)
        ratio_cells: dict[tuple[str, ...], dict[str, tuple[float, float, float, float]]] = {}
        tot_ratio: dict[str, list[float]] = {}
        acc_map: dict[tuple[str, ...], dict[str, list[float]]] = {}
        for iid in all_ids:
            cr = cand_recs[iid]
            rr = ref_recs[iid]
            cell = _record_cell(cr)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            st = (cr["stratum_id"] if cr["stratum_id"] is not None else "__all__") if equal_strata else "__all__"
            a = acc_map.setdefault(cell, {}).setdefault(st, [0.0, 0.0, 0.0, 0.0])
            a[0] += cr["num"]
            a[1] += cr["den"]
            a[2] += rr["num"]
            a[3] += rr["den"]
        for cell, st_m in acc_map.items():
            ratio_cells[cell] = {}
            for st, vals in st_m.items():
                ratio_cells[cell][st] = (vals[0], vals[1], vals[2], vals[3])
                t = tot_ratio.setdefault(st, [0.0, 0.0, 0.0, 0.0])
                for j in range(4):
                    t[j] += vals[j]
        return {"kind": "ratio_from_sums", "equal_strata": equal_strata, "cells": ratio_cells, "totals": tot_ratio, "cell_counts": cell_counts}

    if s == "micro_f1_from_counts":
        # cell -> (tp_c, fp_c, fn_c, tp_r, fp_r, fn_r)
        micro_cells: dict[tuple[str, ...], tuple[float, float, float, float, float, float]] = {}
        tot_micro = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        acc_m: dict[tuple[str, ...], list[float]] = {}
        for iid in all_ids:
            cr = cand_recs[iid]
            rr = ref_recs[iid]
            cell = _record_cell(cr)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            a = acc_m.setdefault(cell, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            a[0] += cr["tp"]
            a[1] += cr["fp"]
            a[2] += cr["fn"]
            a[3] += rr["tp"]
            a[4] += rr["fp"]
            a[5] += rr["fn"]
        for cell, vals in acc_m.items():
            micro_cells[cell] = (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
            for j in range(6):
                tot_micro[j] += vals[j]
        return {"kind": "micro_f1_from_counts", "equal_strata": False, "cells": micro_cells, "totals": tot_micro, "cell_counts": cell_counts}

    # macro_f1_from_counts: cell -> st -> cl -> (tp_c, fp_c, fn_c, tp_r, fp_r, fn_r)
    macro_cells: dict[tuple[str, ...], dict[str, dict[str, tuple[float, float, float, float, float, float]]]] = {}
    tot_macro: dict[str, dict[str, list[float]]] = {}
    acc_mc: dict[tuple[str, ...], dict[str, dict[str, list[float]]]] = {}
    for iid in all_ids:
        cr = cand_recs[iid]
        rr = ref_recs[iid]
        cell = _record_cell(cr)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        st = (cr["stratum_id"] if (equal_strata and cr["stratum_id"] is not None) else "__all__")
        cl = cr["class_id"]
        a = acc_mc.setdefault(cell, {}).setdefault(st, {}).setdefault(cl, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a[0] += cr["tp"]
        a[1] += cr["fp"]
        a[2] += cr["fn"]
        a[3] += rr["tp"]
        a[4] += rr["fp"]
        a[5] += rr["fn"]
    for cell, st_m in acc_mc.items():
        macro_cells[cell] = {}
        for st, cl_m in st_m.items():
            macro_cells[cell][st] = {}
            for cl, vals in cl_m.items():
                macro_cells[cell][st][cl] = (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
                t = tot_macro.setdefault(st, {}).setdefault(cl, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for j in range(6):
                    t[j] += vals[j]
    return {"kind": "macro_f1_from_counts", "equal_strata": equal_strata, "cells": macro_cells, "totals": tot_macro, "cell_counts": cell_counts}


def _eval_weighted_sufficient_stats(
    suff: dict[str, Any],
    cell_weights: list[tuple[tuple[str, ...], float]],
    expected_strata: list[str] | None = None,
    stratum_multiplicity: dict[str, float] | None = None,
) -> tuple[float, float] | None:
    """Evaluates (cand_stat, ref_stat) from cell multiplicities in O(#occupied_cells * |S|)."""
    kind = suff["kind"]
    equal_strata = suff["equal_strata"]
    cells = suff["cells"]
    cell_counts = suff["cell_counts"]

    total_items = 0
    for cell, w in cell_weights:
        if w > 0.0 and cell in cell_counts:
            total_items += cell_counts[cell]
    if total_items == 0:
        return None

    def _w_stratum_pair(c_map: dict[str, float], r_map: dict[str, float], keys: list[str]) -> tuple[float, float]:
        present = [k for k in keys if k in c_map]
        if not present:
            raise ValueError("No non-empty strata in equal_stratum aggregation")
        if stratum_multiplicity:
            ws = [float(stratum_multiplicity.get(k, 1.0)) for k in present]
            w_sum = sum(ws)
            if w_sum > 0.0:
                return (
                    sum(c_map[k] * w for k, w in zip(present, ws)) / w_sum,
                    sum(r_map[k] * w for k, w in zip(present, ws)) / w_sum,
                )
        return statistics.fmean(c_map[k] for k in present), statistics.fmean(r_map[k] for k in present)

    if kind == "mean":
        st_acc: dict[str, list[float]] = {}
        for cell, w in cell_weights:
            st_map = cells.get(cell)
            if not st_map:
                continue
            for st, (sc_v, sr_v, cnt) in st_map.items():
                a = st_acc.get(st)
                if a is None:
                    st_acc[st] = [w * sc_v, w * sr_v, w * cnt]
                else:
                    a[0] += w * sc_v
                    a[1] += w * sr_v
                    a[2] += w * cnt
        if not equal_strata:
            only = st_acc["__all__"]
            return only[0] / only[2], only[1] / only[2]
        c_st = {st: vals[0] / vals[2] for st, vals in st_acc.items() if vals[2] > 0.0}
        r_st = {st: vals[1] / vals[2] for st, vals in st_acc.items() if vals[2] > 0.0}
        s_keys = expected_strata if expected_strata else sorted(c_st.keys())
        return _w_stratum_pair(c_st, r_st, s_keys)

    if kind == "ratio_from_sums":
        st_r_acc: dict[str, list[float]] = {}
        for cell, w in cell_weights:
            st_map = cells.get(cell)
            if not st_map:
                continue
            for st, (nc, dc, nr, dr) in st_map.items():
                a = st_r_acc.get(st)
                if a is None:
                    st_r_acc[st] = [w * nc, w * dc, w * nr, w * dr]
                else:
                    a[0] += w * nc
                    a[1] += w * dc
                    a[2] += w * nr
                    a[3] += w * dr
        if not equal_strata:
            only = st_r_acc["__all__"]
            if only[1] <= 0.0 or only[3] <= 0.0:
                raise ValueError("Zero total denominator in ratio_from_sums")
            return only[0] / only[1], only[2] / only[3]
        c_st = {}
        r_st = {}
        for st, vals in sorted(st_r_acc.items()):
            if vals[1] <= 0.0 or vals[3] <= 0.0:
                raise ValueError(f"Zero denominator in ratio_from_sums stratum {st!r}")
            c_st[st] = vals[0] / vals[1]
            r_st[st] = vals[2] / vals[3]
        s_keys = expected_strata if expected_strata else sorted(c_st.keys())
        return _w_stratum_pair(c_st, r_st, s_keys)

    if kind == "micro_f1_from_counts":
        tpc = fpc = fnc = tpr = fpr = fnr = 0.0
        for cell, w in cell_weights:
            tup = cells.get(cell)
            if tup is None:
                continue
            tpc += w * tup[0]
            fpc += w * tup[1]
            fnc += w * tup[2]
            tpr += w * tup[3]
            fpr += w * tup[4]
            fnr += w * tup[5]
        dc = 2.0 * tpc + fpc + fnc
        dr = 2.0 * tpr + fpr + fnr
        return ((2.0 * tpc / dc) if dc > 0.0 else 0.0), ((2.0 * tpr / dr) if dr > 0.0 else 0.0)

    # macro_f1_from_counts
    by_st_cl: dict[str, dict[str, list[float]]] = {}
    for cell, w in cell_weights:
        st_m = cells.get(cell)
        if not st_m:
            continue
        for st, cl_m in st_m.items():
            dst_cl = by_st_cl.setdefault(st, {})
            for cl, tup in cl_m.items():
                a = dst_cl.get(cl)
                if a is None:
                    dst_cl[cl] = [w * tup[j] for j in range(6)]
                else:
                    for j in range(6):
                        a[j] += w * tup[j]
    strata_keys = expected_strata if (equal_strata and expected_strata) else sorted(by_st_cl.keys())
    c_f1_st: dict[str, float] = {}
    r_f1_st: dict[str, float] = {}
    for st in strata_keys:
        cl_map = by_st_cl.get(st)
        if not cl_map:
            continue
        cf_c: list[float] = []
        cf_r: list[float] = []
        for cl in sorted(cl_map.keys()):
            tpc, fpc, fnc, tpr, fpr, fnr = cl_map[cl]
            dc = 2.0 * tpc + fpc + fnc
            dr = 2.0 * tpr + fpr + fnr
            cf_c.append((2.0 * tpc / dc) if dc > 0.0 else 0.0)
            cf_r.append((2.0 * tpr / dr) if dr > 0.0 else 0.0)
        c_f1_st[st] = statistics.fmean(cf_c)
        r_f1_st[st] = statistics.fmean(cf_r)
    return _w_stratum_pair(c_f1_st, r_f1_st, strata_keys)


def _eval_loo_sufficient_stats(
    suff: dict[str, Any],
    drop_cell: tuple[str, ...],
    all_occupied_cells: list[tuple[str, ...]],
    expected_strata: list[str] | None = None,
) -> tuple[float, float] | None:
    """Computes exact leave-one-cell-out (cand_stat, ref_stat) in O(|S|) via total subtraction."""
    cell_counts = suff["cell_counts"]
    tot_n = sum(cell_counts.values())
    rem_n = tot_n - cell_counts.get(drop_cell, 0)
    if rem_n <= 0:
        return None
    if suff["kind"] == "mean":
        totals = suff["totals"]
        drop_map = suff["cells"].get(drop_cell, {})
        if not suff["equal_strata"]:
            tc, tr, tn = totals["__all__"]
            dc, dr, dn = drop_map.get("__all__", (0.0, 0.0, 0.0))
            return (tc - dc) / (tn - dn), (tr - dr) / (tn - dn)
        c_st: dict[str, float] = {}
        r_st: dict[str, float] = {}
        for st, (tc, tr, tn) in totals.items():
            dc, dr, dn = drop_map.get(st, (0.0, 0.0, 0.0))
            rn = tn - dn
            if rn > 0.0:
                c_st[st] = (tc - dc) / rn
                r_st[st] = (tr - dr) / rn
        s_keys = [k for k in (expected_strata if expected_strata else sorted(c_st.keys())) if k in c_st]
        if not s_keys:
            return None
        return statistics.fmean(c_st[k] for k in s_keys), statistics.fmean(r_st[k] for k in s_keys)
    weights = [(c, 1.0) for c in all_occupied_cells if c != drop_cell]
    return _eval_weighted_sufficient_stats(suff, weights, expected_strata)


def _independent_ms_variance(
    values: dict[tuple[str, ...], float], sizes: tuple[int, ...],
) -> tuple[float, float, bool]:
    """Orthogonal ANOVA on a complete two- or three-factor cell grid."""
    n = math.prod(sizes)
    if len(sizes) not in (2, 3) or min(sizes) < 2 or len(values) != n:
        raise ValueError("Independent mean squares require a complete crossed grid")

    # Each interaction is the cell mean minus all lower-order projections.
    grand = statistics.fmean(values.values())
    projected: dict[tuple[int, ...], dict[tuple[str, ...], float]] = {(): {(): grand}}
    terms: list[tuple[int, float, int]] = []
    main_ms: list[float] = []
    for order in range(1, len(sizes) + 1):
        for axes in itertools.combinations(range(len(sizes)), order):
            repeats = n // math.prod(sizes[i] for i in axes)
            sums: dict[tuple[str, ...], list[float]] = {}
            for cell, value in values.items():
                key = tuple(cell[i] for i in axes)
                sums.setdefault(key, []).append(value)
            lower = [
                (sub, tuple(axes.index(i) for i in sub))
                for degree in range(order)
                for sub in itertools.combinations(axes, degree)
            ]
            projection = {
                key: math.fsum([
                    math.fsum(xs) / repeats,
                    *(-projected[sub][tuple(key[j] for j in indexes)] for sub, indexes in lower),
                ]) for key, xs in sums.items()
            }
            projected[axes] = projection
            df = math.prod(sizes[i] - 1 for i in axes)
            ms = math.fsum(x * x for x in projection.values()) * (repeats / df)
            terms.append((1 if order % 2 else -1, ms, df))
            if order == 1:
                main_ms.append(ms)

    total_ms = math.fsum(sign * ms for sign, ms, _ in terms)
    if total_ms <= 0.0:
        return max(main_ms) / n, float(min(sizes) - 1), True
    scale = max(ms for _, ms, _ in terms)
    denom = math.fsum((ms / scale) ** 2 / df for _, ms, df in terms)
    return total_ms / n, (total_ms / scale) ** 2 / denom, False


def _compute_crossed_random_effects_bound(
    *,
    obs_signed_gain: float,
    obs_null_margin: float,
    seed_keys: list[str],
    cluster_keys: list[str],
    cell_to_ids: dict[tuple[str, ...], list[str]],
    cand_recs: dict[str, dict[str, Any]],
    ref_recs: dict[str, dict[str, Any]],
    stat_name: str,
    weighting: str,
    expected_strata: list[str] | None,
    direction_sign: float,
    margin_mode: str,
    null_margin: float,
    metric_bound: float | None,
    alpha_one_sided: float,
    external_noise_se: float,
    suff_stats: dict[str, Any] | None = None,
    item_keys: list[str] | None = None,
) -> dict[str, Any] | None:
    axes = (seed_keys, cluster_keys) if item_keys is None else (seed_keys, cluster_keys, item_keys)
    sizes = tuple(map(len, axes))
    axis_sets = tuple(map(set, axes))
    occupied_cells = sorted(cell_to_ids)
    n_cells = len(occupied_cells)
    obs_excess = obs_signed_gain - obs_null_margin
    if min(sizes) < 2 or n_cells != math.prod(sizes) or any(
        len(cell) != len(sizes) or any(level not in levels for level, levels in zip(cell, axis_sets))
        for cell in occupied_cells
    ):
        return None

    pseudo_gain: dict[tuple[str, ...], float] = {}
    pseudo_excess: dict[tuple[str, ...], float] = {}

    if stat_name.lower() == "mean" and weighting == "equal_item":
        cell_sizes = {len(cell_to_ids[cell]) for cell in occupied_cells}
        if len(cell_sizes) == 1:
            if suff_stats is not None and suff_stats["kind"] == "mean":
                for cell in occupied_cells:
                    sc_v, sr_v, cnt = suff_stats["cells"][cell]["__all__"]
                    mc = sc_v / cnt
                    mr = sr_v / cnt
                    g_cell = direction_sign * (mc - mr)
                    m_cell = _compute_margin_requirement(mr, null_margin, margin_mode, direction_sign, metric_bound)
                    pseudo_gain[cell] = g_cell
                    pseudo_excess[cell] = g_cell - m_cell
            else:
                for cell in occupied_cells:
                    ids_c = cell_to_ids[cell]
                    mc = statistics.fmean(cand_recs[i]["value"] for i in ids_c)
                    mr = statistics.fmean(ref_recs[i]["value"] for i in ids_c)
                    g_cell = direction_sign * (mc - mr)
                    m_cell = _compute_margin_requirement(mr, null_margin, margin_mode, direction_sign, metric_bound)
                    pseudo_gain[cell] = g_cell
                    pseudo_excess[cell] = g_cell - m_cell

    if not pseudo_gain and suff_stats is not None:
        for cell in occupied_cells:
            loo_pair = _eval_loo_sufficient_stats(suff_stats, cell, occupied_cells, expected_strata)
            if loo_pair is None:
                pseudo_gain[cell] = obs_signed_gain
                pseudo_excess[cell] = obs_excess
                continue
            sc_loo, sr_loo = loo_pair
            gain_loo = direction_sign * (sc_loo - sr_loo)
            m_loo = _compute_margin_requirement(sr_loo, null_margin, margin_mode, direction_sign, metric_bound)
            pseudo_gain[cell] = n_cells * obs_signed_gain - (n_cells - 1) * gain_loo
            pseudo_excess[cell] = n_cells * obs_excess - (n_cells - 1) * (gain_loo - m_loo)

    if not pseudo_gain:
        all_ids = sorted(cand_recs.keys())
        c_all = [cand_recs[i] for i in all_ids]
        r_all = [ref_recs[i] for i in all_ids]
        for cell in occupied_cells:
            drop_set = set(cell_to_ids[cell])
            c_loo = [x for x in c_all if x["id"] not in drop_set]
            r_loo = [x for x in r_all if x["id"] not in drop_set]
            if not c_loo:
                pseudo_gain[cell] = obs_signed_gain
                pseudo_excess[cell] = obs_excess
                continue
            sc_loo = _aggregate_records(c_loo, stat_name, weighting, expected_strata)
            sr_loo = _aggregate_records(r_loo, stat_name, weighting, expected_strata)
            gain_loo = direction_sign * (sc_loo - sr_loo)
            m_loo = _compute_margin_requirement(sr_loo, null_margin, margin_mode, direction_sign, metric_bound)
            pseudo_gain[cell] = n_cells * obs_signed_gain - (n_cells - 1) * gain_loo
            pseudo_excess[cell] = n_cells * obs_excess - (n_cells - 1) * (gain_loo - m_loo)

    def _bound(p_map: dict[tuple[str, ...], float], center: float) -> tuple[float, float, float, bool]:
        variance, df, fallback = _independent_ms_variance(p_map, sizes)
        se = math.hypot(math.sqrt(variance), external_noise_se)
        bound = center - _inv_student_t(1.0 - alpha_one_sided, df) * se if se else center
        return se, df, bound, fallback

    se_cr, df_satt, lcb_cr, fallback_gain = _bound(pseudo_gain, obs_signed_gain)
    _, df_excess, ex_lcb_cr, fallback_excess = _bound(pseudo_excess, obs_excess)
    return {
        "se_crossed": se_cr,
        "df_satterthwaite": df_satt,
        "margin_excess_df_satterthwaite": df_excess,
        "lcb_crossed_t": lcb_cr,
        "margin_excess_lcb_crossed_t": ex_lcb_cr,
        "nonpositive_crossed_variance_fallback": fallback_gain or fallback_excess,
        "crossed_equal_cell_counts": len({len(cell_to_ids[cell]) for cell in occupied_cells}) == 1,
    }


def _transform_records_log(recs: dict[str, dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for iid, r in recs.items():
        v = r.get("value", 0.0)
        if v <= 0.0:
            raise ValueError(f"margin_mode='log_ratio' requires strictly positive values; {label}[{iid}]={v}")
        out[iid] = {**r, "value": math.log(v)}
    return out


def _compute_margin_requirement(
    ref_val: float,
    min_improvement: float,
    margin_mode: str,
    direction_sign: float,
    metric_bound: float | None,
) -> float:
    mm = margin_mode.lower()
    if mm == "additive":
        return min_improvement
    if mm == "relative_headroom":
        bound = metric_bound if metric_bound is not None else (1.0 if direction_sign > 0 else 0.0)
        headroom = max(0.0, (bound - ref_val) if direction_sign > 0 else (ref_val - bound))
        return min_improvement * headroom
    if mm == "relative_reference":
        return min_improvement * abs(ref_val)
    if mm == "log_ratio":
        if min_improvement <= -1.0:
            raise ValueError(f"log_ratio min_improvement must be > -1.0, got {min_improvement}")
        return math.log(1.0 + min_improvement)
    raise ValueError(f"Unsupported margin_mode={margin_mode!r}; use additive, relative_headroom, or log_ratio")


def _paired_bootstrap(
    cand_recs: dict[str, dict[str, Any]],
    ref_recs: dict[str, dict[str, Any]],
    *,
    ref_label: str = "incumbent",
    stat_name: str = "mean",
    weighting: str = "equal_item",
    bootstrap_mode: str = "auto",
    direction_sign: float = 1.0,
    alpha_one_sided: float = 0.025,
    null_margin: float = 0.0,
    margin_mode: str = "additive",
    metric_bound: float | None = None,
    claim_scope: str = "training_recipe",
    external_noise_se: float = 0.0,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 20261117,
    force_raw_resampling: bool = False,
) -> dict[str, Any]:
    _verify_item_structure_parity(cand_recs, ref_recs, ref_label)
    _require_positive_int(n_perm, "n_perm")
    p_floor = 1.0 / (1 + n_perm)

    mm_eff = margin_mode.lower()
    if mm_eff == "log_ratio":
        if stat_name.lower() in ("macro_f1_from_counts", "micro_f1_from_counts", "ratio_from_sums"):
            raise ValueError(f"margin_mode='log_ratio' is incompatible with count/ratio stat {stat_name!r}")
        cand_recs = _transform_records_log(cand_recs, "candidate")
        ref_recs = _transform_records_log(ref_recs, ref_label)

    all_ids = sorted(cand_recs.keys())
    c_all = [cand_recs[i] for i in all_ids]
    r_all = [ref_recs[i] for i in all_ids]

    strata_set = sorted({r["stratum_id"] for r in c_all if r["stratum_id"] is not None})
    expected_strata = strata_set if (weighting == "equal_stratum" and strata_set) else None

    use_suff = (not force_raw_resampling) and (stat_name.lower() in _SUFFICIENT_STAT_NAMES)
    suff_stats: dict[str, Any] | None = (
        _build_cell_sufficient_stats(all_ids, cand_recs, ref_recs, stat_name, weighting) if use_suff else None
    )

    if suff_stats is not None:
        all_cell_weights = [(cell, 1.0) for cell in suff_stats["cell_counts"].keys()]
        obs_pair = _eval_weighted_sufficient_stats(suff_stats, all_cell_weights, expected_strata)
        assert obs_pair is not None
        obs_cand, obs_ref = obs_pair
    else:
        obs_cand = _aggregate_records(c_all, stat_name, weighting, expected_strata)
        obs_ref = _aggregate_records(r_all, stat_name, weighting, expected_strata)
    obs_signed_gain = direction_sign * (obs_cand - obs_ref)
    eff_null_margin = _compute_margin_requirement(obs_ref, null_margin, mm_eff, direction_sign, metric_bound)

    seed_keys = sorted({r["seed_id"] for r in c_all if r["seed_id"] is not None})
    cluster_keys = sorted({r["cluster_id"] for r in c_all})
    item_keys = sorted({r["item_cluster_id"] for r in c_all if r.get("item_cluster_id") is not None})
    if item_keys and any(r.get("item_cluster_id") is None for r in c_all):
        raise ValueError("Item cluster metadata must cover all paired items")
    n_clusters = len(cluster_keys)
    n_seeds = len(seed_keys)
    n_item_clusters = len(item_keys)

    mode_eff = bootstrap_mode.lower()
    if mode_eff == "auto":
        mode_eff = "crossed_seed_cluster" if (n_seeds >= 2 and n_clusters >= 2) else "cluster_1d"
        if item_keys:
            mode_eff = "crossed_seed_user_item" if n_seeds >= 2 and claim_scope.lower() != "fixed_checkpoint" else "crossed_user_item"
    elif mode_eff == "crossed_seed_cluster" and item_keys:
        mode_eff = "crossed_seed_user_item"
    crossed_items = mode_eff in {"crossed_user_item", "crossed_seed_user_item"}
    crossed_seeds = mode_eff in {"crossed_seed_cluster", "crossed_seed_user_item"}
    if mode_eff not in {"cluster_1d", "crossed_seed_cluster", "crossed_user_item", "crossed_seed_user_item"}:
        raise ValueError(f"Unsupported bootstrap_mode {bootstrap_mode!r}")
    if bool(item_keys) != crossed_items:
        raise ValueError("Use a crossed user/item bootstrap mode when item-cluster metadata is supplied")
    if mode_eff == "crossed_user_item" and n_seeds > 1 and claim_scope.lower() != "fixed_checkpoint":
        raise ValueError("A multi-seed user/item inference requires crossed_seed_user_item")

    # Compute per-cluster observed signed gains & cluster-specific margins
    cluster_to_ids: dict[str, list[str]] = {}
    for iid in all_ids:
        cluster_to_ids.setdefault(cand_recs[iid]["cluster_id"], []).append(iid)

    cluster_signed_deltas: list[float] = []
    shifted_cluster_deltas: list[float] = []
    if not crossed_items and suff_stats is not None:
        cells_by_cluster: dict[str, list[tuple[tuple[str, str], float]]] = {}
        for cell in suff_stats["cell_counts"].keys():
            cells_by_cluster.setdefault(cell[1], []).append((cell, 1.0))
        for ck in cluster_keys:
            try:
                cl_pair = _eval_weighted_sufficient_stats(suff_stats, cells_by_cluster.get(ck, []), None)
                if cl_pair is None:
                    continue
                dc, dr = cl_pair
                d_val = direction_sign * (dc - dr)
                cl_margin = _compute_margin_requirement(dr, null_margin, mm_eff, direction_sign, metric_bound)
                cluster_signed_deltas.append(d_val)
                shifted_cluster_deltas.append(d_val - cl_margin)
            except ValueError:
                pass
    elif not crossed_items:
        for ck in cluster_keys:
            c_sub = [cand_recs[i] for i in cluster_to_ids[ck]]
            r_sub = [ref_recs[i] for i in cluster_to_ids[ck]]
            try:
                dc = _aggregate_records(c_sub, stat_name, weighting, None)
                dr = _aggregate_records(r_sub, stat_name, weighting, None)
                d_val = direction_sign * (dc - dr)
                cl_margin = _compute_margin_requirement(dr, null_margin, mm_eff, direction_sign, metric_bound)
                cluster_signed_deltas.append(d_val)
                shifted_cluster_deltas.append(d_val - cl_margin)
            except ValueError:
                pass

    distinct_cluster_deltas = len({round(d, 12) for d in cluster_signed_deltas})
    tol_pos = 1e-12
    n_pos_clusters = sum(1 for d in shifted_cluster_deltas if d > tol_pos)

    # Paired cluster-level Rademacher sign-flip (arm-swap) one-sided p-value aligned to margin:
    # Exhaustive for G <= 16; Phipson-Smyth Monte Carlo for G > 16.
    signflip_p = 1.0
    signflip_method = "not_applicable_crossed_user_item" if crossed_items else "none"
    signflip_resolution = None
    if not crossed_items and n_clusters >= 2 and len(shifted_cluster_deltas) == n_clusters:
        signflip_resolution = 0.5 ** n_clusters if n_clusters <= 16 else p_floor
        obs_mean_cl = statistics.fmean(shifted_cluster_deltas)
        if obs_mean_cl > tol_pos:
            if n_clusters <= 16:
                signflip_method = "exact"
                total_flips = 1 << n_clusters
                ge_count = 0
                for mask in range(total_flips):
                    s_sum = 0.0
                    for bit_i, d_val in enumerate(shifted_cluster_deltas):
                        s_sum += d_val if not ((mask >> bit_i) & 1) else -d_val
                    if (s_sum / n_clusters) >= obs_mean_cl - 1e-15:
                        ge_count += 1
                signflip_p = ge_count / float(total_flips)
            else:
                signflip_method = f"monte_carlo_{n_perm}"
                rng_sf = random.Random(seed ^ 0x5A17F119)
                b_hits = 0
                for _ in range(n_perm):
                    s_sum = sum(d_val if rng_sf.getrandbits(1) else -d_val for d_val in shifted_cluster_deltas)
                    if (s_sum / n_clusters) >= obs_mean_cl - 1e-15:
                        b_hits += 1
                signflip_p = (1.0 + b_hits) / float(1 + n_perm)

    signflip_fields = {
        "n_perm": n_perm, "p_floor": p_floor,
        "signflip_p_one_sided": signflip_p, "signflip_method": signflip_method,
        "signflip_resolution": signflip_resolution,
    }
    if n_clusters < 2 or (crossed_seeds and n_seeds < 2) or (crossed_items and n_item_clusters < 2):
        return {
            "n_items": len(all_ids),
            "n_clusters": n_clusters,
            "n_seeds": n_seeds,
            "n_item_clusters": n_item_clusters,
            "n_strata": len(strata_set),
            "bootstrap_mode": mode_eff,
            "weighting": weighting,
            "statistic": stat_name,
            "margin_mode": mm_eff,
            "effective_null_margin": eff_null_margin,
            "obs_candidate": obs_cand,
            "obs_reference": obs_ref,
            "signed_gain": obs_signed_gain,
            "se": external_noise_se,
            "lcb_one_sided": obs_signed_gain,
            "margin_excess_lcb_one_sided": obs_signed_gain - eff_null_margin,
            "ci95_low": obs_signed_gain,
            "ci95_high": obs_signed_gain,
            "distinct_cluster_deltas": distinct_cluster_deltas,
            "n_positive_clusters": n_pos_clusters,
            **signflip_fields,
            "insufficient_replication": True,
            "low_cluster_warning": True,
        }

    rng = random.Random(seed)
    rng_ext = random.Random(seed ^ 0x3C6EF372) if external_noise_se > 0.0 else None
    boot_gains: list[float] = []
    boot_excess: list[float] = []

    cluster_to_strata: dict[str, set[str]] = {}
    for r in c_all:
        if r["stratum_id"] is not None:
            cluster_to_strata.setdefault(r["cluster_id"], set()).add(r["stratum_id"])
    clusters_nested_in_strata = bool(
        expected_strata
        and cluster_to_strata
        and all(len(v) == 1 for v in cluster_to_strata.values())
        and set(cluster_keys) == set(strata_set)
    )

    crossed_re_info: dict[str, Any] | None = None
    if crossed_seeds or crossed_items:
        cell_to_ids: dict[tuple[str, ...], list[str]] = {}
        for iid in all_ids:
            cell_to_ids.setdefault(_record_cell(cand_recs[iid]), []).append(iid)
    if crossed_seeds:
        crossed_re_info = _compute_crossed_random_effects_bound(
            obs_signed_gain=obs_signed_gain,
            obs_null_margin=eff_null_margin,
            seed_keys=seed_keys,
            cluster_keys=cluster_keys,
            cell_to_ids=cell_to_ids,
            cand_recs=cand_recs,
            ref_recs=ref_recs,
            stat_name=stat_name,
            weighting=weighting,
            expected_strata=expected_strata,
            direction_sign=direction_sign,
            margin_mode=mm_eff,
            null_margin=null_margin,
            metric_bound=metric_bound,
            alpha_one_sided=alpha_one_sided,
            external_noise_se=external_noise_se,
            suff_stats=suff_stats,
            item_keys=item_keys if crossed_items else None,
        )

    if crossed_items:
        for _ in range(n_boot):
            s_counts = Counter(rng.choices(seed_keys, k=n_seeds)) if crossed_seeds else None
            u_counts = Counter(rng.choices(cluster_keys, k=n_clusters))
            i_counts = Counter(rng.choices(item_keys, k=n_item_clusters))
            st_mult = (
                {next(iter(cluster_to_strata[c])): float(n) for c, n in u_counts.items()}
                if clusters_nested_in_strata else None
            )
            draws = []
            for cell in cell_to_ids:
                sk, ck, ik = cell
                multiplicity = u_counts[ck] * i_counts[ik] * (s_counts[sk] if s_counts is not None else 1)
                if multiplicity:
                    draws.append((cell, multiplicity))
            if not draws:
                continue
            if suff_stats is not None:
                pair_draw = _eval_weighted_sufficient_stats(suff_stats, draws, expected_strata, st_mult)
                assert pair_draw is not None
                sc, sr = pair_draw
            else:
                sampled_ids = [iid for cell, count in draws for _ in range(count) for iid in cell_to_ids[cell]]
                sc = _aggregate_records([cand_recs[i] for i in sampled_ids], stat_name, weighting, expected_strata, st_mult)
                sr = _aggregate_records([ref_recs[i] for i in sampled_ids], stat_name, weighting, expected_strata, st_mult)
            ext_draw = rng_ext.gauss(0.0, external_noise_se) if rng_ext is not None else 0.0
            g_draw = direction_sign * (sc - sr) + ext_draw
            m_draw = _compute_margin_requirement(sr, null_margin, mm_eff, direction_sign, metric_bound)
            boot_gains.append(g_draw)
            boot_excess.append(g_draw - m_draw)
    elif crossed_seeds:
        if suff_stats is not None:
            suff_cells = suff_stats["cell_counts"]
            for _ in range(n_boot):
                s_draws = [seed_keys[rng.randrange(n_seeds)] for _ in range(n_seeds)]
                c_draws = [cluster_keys[rng.randrange(n_clusters)] for _ in range(n_clusters)]
                st_mult: dict[str, float] | None = None
                if clusters_nested_in_strata:
                    st_mult = {}
                    for ck in c_draws:
                        st_only = next(iter(cluster_to_strata[ck]))
                        st_mult[st_only] = st_mult.get(st_only, 0.0) + 1.0
                s_counts: dict[str, float] = {}
                for sk in s_draws:
                    s_counts[sk] = s_counts.get(sk, 0.0) + 1.0
                c_counts: dict[str, float] = {}
                for ck in c_draws:
                    c_counts[ck] = c_counts.get(ck, 0.0) + 1.0
                w_cells = [
                    ((sk, ck), sw * cw)
                    for sk, sw in s_counts.items()
                    for ck, cw in c_counts.items()
                    if (sk, ck) in suff_cells
                ]
                pair_draw = _eval_weighted_sufficient_stats(suff_stats, w_cells, expected_strata, st_mult)
                if pair_draw is None:
                    continue
                sc, sr = pair_draw
                ext_draw = rng_ext.gauss(0.0, external_noise_se) if rng_ext is not None else 0.0
                g_draw = direction_sign * (sc - sr) + ext_draw
                m_draw = _compute_margin_requirement(sr, null_margin, mm_eff, direction_sign, metric_bound)
                boot_gains.append(g_draw)
                boot_excess.append(g_draw - m_draw)
        else:
            for _ in range(n_boot):
                s_draws = [seed_keys[rng.randrange(n_seeds)] for _ in range(n_seeds)]
                c_draws = [cluster_keys[rng.randrange(n_clusters)] for _ in range(n_clusters)]
                st_mult = None
                if clusters_nested_in_strata:
                    st_mult = {}
                    for ck in c_draws:
                        st_only = next(iter(cluster_to_strata[ck]))
                        st_mult[st_only] = st_mult.get(st_only, 0.0) + 1.0
                sampled_c: list[dict[str, Any]] = []
                sampled_r: list[dict[str, Any]] = []
                for sk in s_draws:
                    for ck in c_draws:
                        for iid in cell_to_ids.get((sk, ck), ()):
                            sampled_c.append(cand_recs[iid])
                            sampled_r.append(ref_recs[iid])
                if not sampled_c:
                    continue
                sc = _aggregate_records(sampled_c, stat_name, weighting, expected_strata, st_mult)
                sr = _aggregate_records(sampled_r, stat_name, weighting, expected_strata, st_mult)
                ext_draw = rng_ext.gauss(0.0, external_noise_se) if rng_ext is not None else 0.0
                g_draw = direction_sign * (sc - sr) + ext_draw
                m_draw = _compute_margin_requirement(sr, null_margin, mm_eff, direction_sign, metric_bound)
                boot_gains.append(g_draw)
                boot_excess.append(g_draw - m_draw)
    else:
        if suff_stats is not None:
            cells_for_ck: dict[str, list[tuple[str, str]]] = {}
            for cell in suff_stats["cell_counts"].keys():
                cells_for_ck.setdefault(cell[1], []).append(cell)
            for _ in range(n_boot):
                c_draws = [cluster_keys[rng.randrange(n_clusters)] for _ in range(n_clusters)]
                st_mult = None
                if clusters_nested_in_strata:
                    st_mult = {}
                    for ck in c_draws:
                        st_only = next(iter(cluster_to_strata[ck]))
                        st_mult[st_only] = st_mult.get(st_only, 0.0) + 1.0
                c_counts_1d: dict[str, float] = {}
                for ck in c_draws:
                    c_counts_1d[ck] = c_counts_1d.get(ck, 0.0) + 1.0
                w_cells_1d = [
                    (cell, cw)
                    for ck, cw in c_counts_1d.items()
                    for cell in cells_for_ck.get(ck, ())
                ]
                pair_1d = _eval_weighted_sufficient_stats(suff_stats, w_cells_1d, expected_strata, st_mult)
                if pair_1d is None:
                    continue
                sc, sr = pair_1d
                ext_draw = rng_ext.gauss(0.0, external_noise_se) if rng_ext is not None else 0.0
                g_draw = direction_sign * (sc - sr) + ext_draw
                m_draw = _compute_margin_requirement(sr, null_margin, mm_eff, direction_sign, metric_bound)
                boot_gains.append(g_draw)
                boot_excess.append(g_draw - m_draw)
        else:
            for _ in range(n_boot):
                c_draws = [cluster_keys[rng.randrange(n_clusters)] for _ in range(n_clusters)]
                st_mult = None
                if clusters_nested_in_strata:
                    st_mult = {}
                    for ck in c_draws:
                        st_only = next(iter(cluster_to_strata[ck]))
                        st_mult[st_only] = st_mult.get(st_only, 0.0) + 1.0
                sampled_c = []
                sampled_r = []
                for ck in c_draws:
                    for iid in cluster_to_ids[ck]:
                        sampled_c.append(cand_recs[iid])
                        sampled_r.append(ref_recs[iid])
                sc = _aggregate_records(sampled_c, stat_name, weighting, expected_strata, st_mult)
                sr = _aggregate_records(sampled_r, stat_name, weighting, expected_strata, st_mult)
                ext_draw = rng_ext.gauss(0.0, external_noise_se) if rng_ext is not None else 0.0
                g_draw = direction_sign * (sc - sr) + ext_draw
                m_draw = _compute_margin_requirement(sr, null_margin, mm_eff, direction_sign, metric_bound)
                boot_gains.append(g_draw)
                boot_excess.append(g_draw - m_draw)

    boot_gains.sort()
    boot_excess.sort()
    n_eff = len(boot_gains)
    if n_eff < 2:
        raise ValueError("Too few nonempty paired bootstrap draws")
    se_boot = statistics.stdev(boot_gains) if n_eff > 1 else 0.0
    lcb_idx = max(0, min(n_eff - 1, int(math.floor(alpha_one_sided * n_eff))))
    boot_lcb = boot_gains[lcb_idx]
    excess_lcb = boot_excess[lcb_idx]
    se_final = se_boot
    lcb_final = boot_lcb

    if crossed_re_info is not None and claim_scope.lower() != "fixed_checkpoint":
        se_final = max(se_boot, crossed_re_info["se_crossed"])
        lcb_final = min(boot_lcb, crossed_re_info["lcb_crossed_t"])
        excess_lcb = min(excess_lcb, crossed_re_info["margin_excess_lcb_crossed_t"])

    lo95 = boot_gains[max(0, int(math.floor(0.025 * n_eff)))]
    hi95 = boot_gains[min(n_eff - 1, int(math.ceil(0.975 * n_eff)))]
    tail_stat = stat_name.lower() in ("p90", "p95", "p99")
    res_out: dict[str, Any] = {
        "n_items": len(all_ids),
        "n_clusters": n_clusters,
        "n_seeds": n_seeds,
        "n_item_clusters": n_item_clusters,
        "n_strata": len(strata_set),
        "bootstrap_mode": mode_eff,
        "weighting": weighting,
        "statistic": stat_name,
        "margin_mode": mm_eff,
        "effective_null_margin": eff_null_margin,
        "obs_candidate": obs_cand,
        "obs_reference": obs_ref,
        "signed_gain": obs_signed_gain,
        "se": se_final,
        "lcb_one_sided": lcb_final,
        "margin_excess_lcb_one_sided": excess_lcb,
        "ci95_low": lo95,
        "ci95_high": hi95,
        "distinct_cluster_deltas": distinct_cluster_deltas,
        "n_positive_clusters": n_pos_clusters,
        **signflip_fields,
        "insufficient_replication": (
            crossed_seeds and crossed_re_info is None and claim_scope.lower() != "fixed_checkpoint"
            or crossed_items and n_eff < math.ceil(0.8 * n_boot)
        ),
        "low_cluster_warning": bool(n_clusters < 10 or (tail_stat and n_clusters < 20)),
        "crossed_design_complete": crossed_re_info is not None if crossed_seeds else None,
        "n_boot_usable": n_eff,
    }
    if crossed_re_info is not None:
        res_out.update(crossed_re_info)
    return res_out


# ---------------------------------------------------------------------------
# 6. Hash-Chained Ledger & Rollback-Streak State Machine
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _ledger_exclusive_lock(ledger_path: Path | None, timeout_sec: float = 60.0) -> Iterator[None]:
    """Cross-platform atomic lockfile (`<ledger>.lock`) guarding read-verify-append cycles."""
    if ledger_path is None:
        yield
        return
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    t_start = time.monotonic()
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()}:{time.time():.3f}\n".encode("utf-8"))
            break
        except FileExistsError:
            if (time.monotonic() - t_start) > timeout_sec:
                raise TimeoutError(f"Timed out acquiring ledger lock {lock_path} after {timeout_sec}s")
            time.sleep(0.02)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def _read_and_verify_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    prev_hash = "GENESIS"
    for line_no, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = _decode_json(raw.encode("utf-8"), ledger_path)
        if not isinstance(row, dict):
            raise ValueError(f"Ledger line {line_no} must contain an object")
        recorded_prev = row.get("prev_entry_sha256")
        recorded_hash = row.get("entry_sha256")
        if recorded_prev != prev_hash:
            raise ValueError(f"Ledger hash-chain broken at line {line_no}: expected prev={prev_hash}, got {recorded_prev}")
        body = {k: v for k, v in row.items() if k != "entry_sha256"}
        expect_hash = _canonical_sha256(body)
        if recorded_hash != expect_hash:
            raise ValueError(f"Ledger entry tampered at line {line_no}: hash mismatch")
        prev_hash = recorded_hash
        rows.append(row)
    return rows


def _append_ledger_entry(ledger_path: Path, prior_rows: list[dict[str, Any]], record_body: dict[str, Any]) -> dict[str, Any]:
    prev_hash = prior_rows[-1]["entry_sha256"] if prior_rows else "GENESIS"
    full_body = {**record_body, "prev_entry_sha256": prev_hash}
    entry_hash = _canonical_sha256(full_body)
    record = {**full_body, "entry_sha256": entry_hash}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return record


def _compute_rollback_streak(
    prior_rows: list[dict[str, Any]],
    current_decision: str,
    reset_policy: str = "confirm_only",
) -> tuple[int, int]:
    """
    Returns (active_consecutive_rollbacks, unresolved_rollbacks_since_reset).
    Fixes the v2.0 bug where `prior_rows` was summed even on PROVISIONAL_KEEP/PROMOTE,
    while preventing unconfirmed PROVISIONAL_KEEP from laundering rollback streaks:
      - `PROMOTE` resets both counters to 0.
      - `PROVISIONAL_KEEP` / `INCONCLUSIVE_RETEST`:
          * Under default `confirm_only`: active streak is 0 on THIS turn (candidate is
            pending confirmation/retest, so do NOT prematurely halt before Confirm runs!),
            while `unresolved_rollbacks_since_reset` preserves prior rollbacks. If a
            subsequent trial rolls back without an intervening `PROMOTE`, all prior
            unresolved `ROLLBACK`s accumulate!
          * Under `provisional_or_confirm`: `PROVISIONAL_KEEP` also resets to 0.
      - `ROLLBACK`: increments unresolved rollbacks since last reset and returns that count
        as `active_consecutive_rollbacks`.
    """
    reset_set = {"PROMOTE"} if reset_policy == "confirm_only" else {"PROMOTE", "PROVISIONAL_KEEP"}
    prior_unresolved = 0
    for row in reversed(prior_rows):
        if row.get("decision_stage") in {"holdout", "confirm_reservation"}:
            continue
        dec = row.get("confirm_look_outcome") if row.get("confirm_look_id") else row.get("decision")
        if dec in reset_set:
            break
        if dec == "ROLLBACK":
            prior_unresolved += 1
        # Pending states (PROVISIONAL_KEEP under confirm_only, INCONCLUSIVE_RETEST) neither reset nor increment

    if current_decision in reset_set:
        return 0, 0
    if current_decision in ("PROVISIONAL_KEEP", "INCONCLUSIVE_RETEST"):
        return 0, prior_unresolved
    if current_decision == "ROLLBACK":
        total = prior_unresolved + 1
        return total, total
    return 0, prior_unresolved


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


# ---------------------------------------------------------------------------
# 7. Two-Stage Promotion Gate (`gate-decision`)
# ---------------------------------------------------------------------------


def _required_comparators(rules: dict[str, Any], extras: list[str] | None = None, *, default: tuple[str, ...] = ()) -> list[str]:
    declared = rules.get("required_comparators", [])
    if not isinstance(declared, list) or any(not isinstance(x, str) or not x.strip() for x in declared):
        raise ValueError("required_comparators must be a list of nonempty names")
    names = [x.strip() for x in declared]
    incumbent = rules.get("current_incumbent_id")
    if incumbent and rules.get("require_incumbent_comparator", True):
        names.append(str(incumbent).strip())
    names.extend(x.strip() for x in extras or [] if x.strip())
    return list(dict.fromkeys(names or default))


def _parse_comparators(entries: list[str], flag: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or not path or name in result:
            raise ValueError(f"{flag} requires unique nonempty NAME=PATH entries: {entry!r}")
        result[name] = path
    return result


def _confirm_evidence_blocks(rules: dict[str, Any], count: int) -> list[list[str]] | None:
    blocks = rules.get("confirm_evidence_blocks")
    if blocks is None:
        return None
    if not isinstance(blocks, list) or len(blocks) != count:
        raise ValueError("confirm_evidence_blocks must supply one cluster-ID list per scheduled look")
    seen: set[str] = set()
    for block in blocks:
        if not isinstance(block, list) or len(block) < 2 or any(not isinstance(v, str) or not v.strip() for v in block):
            raise ValueError("Each Confirm evidence block must name at least two distinct cluster IDs")
        if seen.intersection(block) or len(block) != len(set(block)):
            raise ValueError("Confirm evidence blocks must be disjoint and contain no duplicates")
        seen.update(block)
    return blocks


def _confirm_look_debits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in rows if r.get("decision_stage") == "confirm_reservation"
        or (r.get("decision_stage") == "confirm" and not r.get("confirm_look_id"))
    ]


def _emit_gate_result(args: argparse.Namespace, record: dict[str, Any], stage: str, *, batch: bool = False) -> int:
    decision = record["iut_decision" if batch else "decision"]
    if args.proposer_out:
        keys = (
            ("trial_prefix", "decision_stage", "iut_decision", "comparators_tested", "comparators_promoted", "halt_recommended")
            if batch else ("trial_id", "candidate_id", "comparator_id", "decision_stage", "decision", "consecutive_rollbacks", "halt_recommended")
        )
        redacted = {k: record[k] for k in keys if k in record}
        if not batch:
            trips = record.get("tripwires", [])
            redacted["tripwires"] = ["confirm_gate_or_guardrail_failed"] if stage == "confirm" and trips else trips
        p_out = Path(args.proposer_out).resolve()
        p_out.parent.mkdir(parents=True, exist_ok=True)
        p_out.write_text(json.dumps(redacted, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if decision in {"PROMOTE", "PROVISIONAL_KEEP"} else 3 if decision == "INCONCLUSIVE_RETEST" else 1


def _run_gate_look(
    args: argparse.Namespace, rules: dict[str, Any], rules_sha: str, comp_paths: dict[str, str] | None = None,
) -> int:
    stage = str(rules.get("decision_stage", "confirm")).lower()
    batch = comp_paths is not None
    trial = args.trial_prefix if batch else args.trial_id
    alpha, schedule = _confirm_alpha_plan(rules, args.alpha_schedule if stage == "confirm" else None)
    if stage != "confirm" and (args.alpha_schedule or args.allow_registered_retest):
        raise ValueError("Alpha spending and registered retests apply only to Confirm")
    blocks = _confirm_evidence_blocks(rules, len(schedule)) if stage == "confirm" else None
    item_key = _item_cluster_key(args, rules)
    n_perm = _require_positive_int(rules.get("n_perm", DEFAULT_N_PERM), "n_perm")
    aux_margins = _resolve_aux_margins(rules.get("paired_aux_noninferiority", {}), args.aux_noninferiority_margins)
    rules = {**rules, "paired_aux_noninferiority": aux_margins}
    policy_sha = _canonical_sha256({
        "rules_sha256": rules_sha, "alpha_schedule": schedule, "item_cluster_key": item_key,
        "paired_aux_noninferiority": aux_margins,
    })
    ledger = Path(args.ledger).resolve() if args.ledger else None
    with _ledger_exclusive_lock(ledger):
        rows = _read_and_verify_ledger(ledger) if ledger else []
        looks = _confirm_look_debits(rows)
        used = len(looks)
        eff_alpha = schedule[min(used, len(schedule) - 1)] if stage == "confirm" else alpha / int(rules.get("multiplicity_k", 1))
        charges = [row.get("alpha_debit", row.get("effective_alpha_one_sided", 0.0)) for row in looks]
        spent = math.fsum(charges)
        spent_after = math.fsum([*charges, eff_alpha])
        prior_ids = {r.get("confirm_look_id") for r in rows if r.get("confirm_look_outcome")}
        pending = any(r.get("decision_stage") == "confirm_reservation" and r["entry_sha256"] not in prior_ids for r in looks)
        terminal = pending or any(r.get("halt_recommended") is True or r.get("decision_stage") == "holdout" for r in rows)
        dev_before = used + sum(r.get("decision_stage") not in {"confirm", "confirm_reservation", "holdout"} for r in rows)
        max_dev = int(rules.get("max_dev_looks", 0))
        tripwires: list[str] = []
        context: dict[str, Any] = {
            "guard_version": GUARD_VERSION,
            "trial_prefix" if batch else "trial_id": trial,
            "decision_stage": stage, "effective_alpha_one_sided": eff_alpha,
            "n_perm": n_perm, "p_floor": 1.0 / (1 + n_perm),
            "confirm_looks_used": used, "alpha_spent": spent, "alpha_budget": alpha,
            "look_accounting": "ledger" if ledger else "unlogged_diagnostic",
        }

        def denied() -> int:
            return _emit_gate_result(args, {
                **context, "status": "LOOK_BLOCKED", "iut_decision" if batch else "decision": "ROLLBACK",
                "tripwires": tripwires, "halt_recommended": terminal or used >= len(schedule),
            }, stage, batch=batch)

        if any(
            row.get("rules_sha256") not in (None, rules_sha)
            or row.get("confirm_policy_sha256") not in (None, policy_sha)
            for row in looks
        ) and stage == "confirm":
            tripwires.append("confirm_rules_mutated_mid_campaign")
        if terminal:
            tripwires.append("campaign_halted" if not pending else "interrupted_confirm_look_consumed")
        if stage == "confirm" and (
            used >= len(schedule) or spent_after > alpha
        ):
            tripwires.append("confirm_look_budget_exhausted")
        if max_dev > 0 and dev_before >= max_dev:
            tripwires.append("dev_look_budget_exhausted")
        if tripwires:
            return denied()
        if args.allow_registered_retest and (ledger is None or blocks is None):
            tripwires.append("registered_retest_requires_ledger_and_fresh_evidence_blocks")
            return denied()

        cand = _load_strict_json(Path(args.candidate))
        cand_id = str(cand.get("candidate_id", trial))
        protocol = str(cand.get("protocol_id", stage))
        identities = _artifact_fingerprints(args.candidate_artifacts) if args.candidate_artifacts else {
            "candidate_artifact_sha256": None, "artifact_semantic_sha256": None,
        }
        art_sha = identities["candidate_artifact_sha256"]
        semantic = identities["artifact_semantic_sha256"]
        context.update(candidate_id=cand_id, protocol_id=protocol, **identities)
        if stage == "confirm" and ledger and (len(schedule) > 1 or rules.get("require_candidate_artifacts")) and not art_sha:
            tripwires.append("candidate_artifacts_required_for_registered_confirm")
        if (claimed := cand.get("candidate_artifact_sha256")) and art_sha and claimed != art_sha:
            tripwires.append("candidate_envelope_artifact_mismatch")
        if any(
            r.get("candidate_id") == cand_id and r.get("candidate_artifact_sha256") is not None
            and r["candidate_artifact_sha256"] != art_sha for r in rows
        ):
            tripwires.append("candidate_id_rebinding")

        same_candidate = [r for r in looks if r.get("candidate_id") == cand_id]
        retest_requested = args.allow_registered_retest or bool(same_candidate and rules.get("allow_registered_retest"))
        previous_refs: dict[str, tuple[Any, Any]] | None = None
        if stage == "confirm":
            if semantic and any(r.get("artifact_semantic_sha256") == semantic and r.get("candidate_id") != cand_id for r in looks):
                tripwires.append("semantic_duplicate_confirm_artifact")
            if retest_requested:
                prior = same_candidate[-1] if same_candidate else None
                outcomes = [
                    r for r in rows
                    if prior and (
                        r.get("confirm_look_id") == prior.get("entry_sha256") and r.get("confirm_look_outcome")
                        or r is prior and r.get("decision_stage") == "confirm"
                    )
                ]
                previous_outcome = outcomes[-1] if outcomes else {}
                previous_decision = previous_outcome.get("confirm_look_outcome") or previous_outcome.get("decision")
                if previous_decision != "INCONCLUSIVE_RETEST":
                    tripwires.append("no_registered_inconclusive_retest")
                elif (
                    blocks is None or not semantic or not prior or prior.get("artifact_semantic_sha256") != semantic
                    or prior.get("protocol_id") != protocol
                    or prior.get("comparator_ids") != list(comp_paths or [args.comparator_id])
                ):
                    tripwires.append("retest_binding_or_evidence_mismatch")
                else:
                    previous_refs = {
                        r["comparator_id"]: (r.get("reference_id"), r.get("baseline_id"))
                        for r in rows if r.get("confirm_look_id") == prior["entry_sha256"]
                    }
            elif same_candidate:
                tripwires.append(f"duplicate_dev_eval_for_candidate:{cand_id}:{protocol}")
        elif any(r.get("candidate_id") == cand_id and r.get("protocol_id") == protocol and r.get("decision_stage") == stage for r in rows):
            tripwires.append(f"duplicate_dev_eval_for_candidate:{cand_id}:{protocol}")
        if tripwires:
            return denied()

        common = {
            **context, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "rules_sha256": rules_sha, "item_cluster_key": item_key, "hypothesis": args.hypothesis or "",
        }
        reservation = None
        fresh = blocks[used] if blocks is not None else None
        comp_ids = list(comp_paths) if comp_paths else [args.comparator_id]
        if stage == "confirm":
            common.update(
                confirm_policy_sha256=policy_sha, confirm_looks_used=used + 1,
                alpha_spent=spent_after, registered_retest=retest_requested,
                evidence_block_sha256=_canonical_sha256(sorted(fresh)) if fresh is not None else None,
                comparator_ids=comp_ids, paired_aux_noninferiority=aux_margins,
            )
            if ledger:
                reservation = _append_ledger_entry(ledger, rows, {
                    **common, "decision_stage": "confirm_reservation",
                    "decision": "CONFIRM_LOOK_RESERVED", "alpha_debit": eff_alpha,
                })
                rows.append(reservation)
        scores: list[dict[str, Any]] = []
        inputs = list(comp_paths.items()) if batch else [(args.comparator_id, args.incumbent)]
        seed_scope = f"{policy_sha}:{used + 1}:{semantic or 'unbound'}"
        try:
            base = _load_strict_json(Path(args.baseline))
            stat_name = str(rules.get("primary_statistic", "mean"))
            cached = (
                _extract_structured_items(cand, "candidate", stat_name, Path(args.candidate).resolve().parent, item_key),
                _extract_structured_items(base, "baseline", stat_name, Path(args.baseline).resolve().parent, item_key),
            )
            for name, path in inputs:
                comp_args = argparse.Namespace(
                    candidate=args.candidate, baseline=args.baseline, incumbent=path, comparator_id=name,
                )
                scores.append(_score_gate(
                    comp_args, rules, cand, base, eff_alpha, seed_scope, item_key,
                    preloaded_items=cached, enforce_fresh_block=fresh, aux_alpha=eff_alpha / len(inputs),
                ))
            if stage == "confirm" and not batch:
                score = scores[0]
                missing = set(_required_comparators(rules)) - {score["comparator_id"]}
                if missing:
                    score["tripwires"].extend(f"missing_required_comparator:{name}" for name in sorted(missing))
                    score.update(decision="ROLLBACK", decision_reason="missing_required_comparator")
            if batch and len(scores) > 1:
                observed_refs = [score.get("reference_id") for score in scores]
                if any(name is None for name in observed_refs) or len(set(observed_refs)) != len(observed_refs):
                    raise ValueError("IUT requires distinct registered reference identities")
            if previous_refs is not None:
                actual_refs = {score["comparator_id"]: (score.get("reference_id"), score.get("baseline_id")) for score in scores}
                if actual_refs != previous_refs or any(None in pair for pair in actual_refs.values()):
                    raise ValueError("Retest baseline or comparator identities differ from the prior inconclusive look")
            if _sha256_file(Path(args.rules)) != rules_sha:
                raise ValueError("Confirm rules changed during scoring")
            if art_sha and _sha256_artifacts(args.candidate_artifacts)["artifact_bundle_sha256"] != art_sha:
                raise ValueError("Candidate artifacts changed during scoring")
        except (OSError, ValueError, KeyError) as exc:
            scores = [
                {"comparator_id": name, "decision": "ROLLBACK", "decision_reason": "invalid_evidence",
                 "tripwires": [f"invalid_item_records:{exc}"]}
                for name, _ in inputs
            ]
        decisions = [s["decision"] for s in scores]
        outcome = decisions[0] if not batch else (
            "PROMOTE" if all(d == "PROMOTE" for d in decisions)
            else "INCONCLUSIVE_RETEST" if all(d in {"PROMOTE", "INCONCLUSIVE_RETEST"} for d in decisions)
            else "ROLLBACK"
        )
        c_rb, u_rb = _compute_rollback_streak(rows, outcome, str(rules.get("streak_reset_policy", "confirm_only")).lower())
        max_rb = int(rules.get("max_consecutive_rollbacks", 0))
        halt = (
            (stage == "confirm" and common["confirm_looks_used"] >= len(schedule))
            or (max_dev > 0 and dev_before + 1 >= max_dev)
            or (max_rb > 0 and c_rb >= max_rb)
        )
        results = {}
        for idx, score in enumerate(scores):
            label = score.get("comparator_id") or "incumbent"
            result = {
                **common, **score, "trial_id": f"{trial}_{label}" if batch else trial,
                "dev_looks_used": dev_before + 1, "consecutive_rollbacks": c_rb,
                "unresolved_rollbacks_since_promote": u_rb, "halt_recommended": halt,
            }
            if reservation:
                result["confirm_look_id"] = reservation["entry_sha256"]
                if idx == len(scores) - 1:
                    result["confirm_look_outcome"] = outcome
            if ledger:
                result = _append_ledger_entry(ledger, rows, result)
                rows.append(result)
            else:
                result.update(prev_entry_sha256="UNLOGGED", entry_sha256="UNLOGGED")
            results[label] = result
        if not batch:
            output = next(iter(results.values()))
        else:
            output = {
                **common, "status": "OK", "iut_decision": outcome, "halt_recommended": halt,
                "comparators_tested": len(scores), "comparators_promoted": decisions.count("PROMOTE"),
                "failed_comparators": [k for k, r in results.items() if r["decision"] != "PROMOTE"],
                "results_by_comparator": {
                    k: {key: r.get(key) for key in (
                        "decision", "signed_improvement", "lcb_one_sided_vs_incumbent",
                        "margin_excess_lcb_one_sided", "tripwires", "df_satterthwaite",
                        "nonpositive_crossed_variance_fallback",
                    )} for k, r in results.items()
                },
            }
    return _emit_gate_result(args, output, stage, batch=batch)


def cmd_gate_decision(args: argparse.Namespace) -> int:
    rules_path = Path(args.rules)
    raw = rules_path.read_bytes()
    rules = _decode_json(raw, rules_path)
    if args.cmd == "gate-confirm" and str(rules.get("decision_stage", "confirm")).lower() != "confirm":
        raise ValueError("gate-confirm requires rules with decision_stage='confirm'")
    return _run_gate_look(args, rules, hashlib.sha256(raw).hexdigest())


def _item_cluster_key(args: argparse.Namespace, rules: dict[str, Any]) -> str | None:
    from_rules = rules.get("item_cluster_key")
    override = getattr(args, "item_cluster_key", None)
    if override is not None and from_rules is not None and override != from_rules:
        raise ValueError("--item-cluster-key conflicts with the rules")
    key = override if override is not None else from_rules
    if key is not None and (not isinstance(key, str) or not key.strip()):
        raise ValueError("item_cluster_key must be a nonempty column name")
    return key


def _score_gate(
    args: argparse.Namespace, rules: dict[str, Any], cand: dict[str, Any], base: dict[str, Any],
    eff_alpha: float, seed_scope: str, item_key: str | None, *,
    aux_alpha: float | None = None, holdout: bool = False,
    preloaded_items: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None,
    preloaded_reference: dict[str, Any] | None = None,
    enforce_fresh_block: list[str] | None = None,
) -> dict[str, Any]:
    inc = preloaded_reference if preloaded_reference is not None else _load_strict_json(Path(args.incumbent))

    cm_raw = cand.get("metrics", {})
    im_raw = inc.get("metrics", {})
    bm_raw = base.get("metrics", {})

    pm = str(rules["primary_metric"])
    direction = str(rules.get("direction", "maximize")).lower()
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"Unsupported direction={direction!r}")
    sign = 1.0 if direction == "maximize" else -1.0
    stat_name = str(rules.get("primary_statistic", "mean"))
    weighting = str(rules.get("aggregation_weighting", "equal_item")).lower()
    bootstrap_mode = str(rules.get("bootstrap_mode", "auto")).lower()
    stage = str(rules.get("decision_stage", "confirm")).lower()
    decision_mode = str(rules.get("decision_mode", "superiority")).lower()
    if decision_mode not in ("superiority", "noninferiority_efficiency"):
        raise ValueError(f"Unsupported decision_mode={decision_mode!r}; use superiority or noninferiority_efficiency")
    margin_mode = str(rules.get("margin_mode", "additive")).lower()
    metric_bound = _require_finite(rules["metric_bound"], "metric_bound") if "metric_bound" in rules else None
    claim_scope = str(rules.get("claim_scope", "training_recipe")).lower()
    alpha = _require_finite(rules.get("alpha_one_sided", 0.025), "alpha_one_sided")
    n_perm = _require_positive_int(rules.get("n_perm", DEFAULT_N_PERM), "n_perm")
    comparator_id = str(args.comparator_id or rules.get("comparator_id", inc.get("candidate_id", "incumbent")))
    common = {
        "comparator_id": comparator_id,
        "reference_id": inc.get("candidate_id"),
        "baseline_id": base.get("candidate_id"),
        "n_perm": n_perm,
        "p_floor": 1.0 / (1 + n_perm),
        "effective_alpha_one_sided": eff_alpha,
    }

    tripwires: list[str] = []
    warnings: list[str] = []

    if rules.get("require_env_fingerprint_match", False):
        if not cand.get("env_fingerprint") or cand.get("env_fingerprint") != inc.get("env_fingerprint"):
            tripwires.append(
                f"env_fingerprint_mismatch:cand={cand.get('env_fingerprint')!r}!=inc={inc.get('env_fingerprint')!r}"
            )

    for label, mdict in (("candidate", cm_raw), ("incumbent", im_raw), ("baseline", bm_raw)):
        if pm not in mdict:
            tripwires.append(f"missing_primary_metric:{label}.{pm}")

    if tripwires:
        return {**common, "decision": "ROLLBACK", "reason": "invalid_evidence", "tripwires": tripwires}

    cand_pm = _require_finite(cm_raw[pm], f"candidate.{pm}")
    inc_pm = _require_finite(im_raw[pm], f"incumbent.{pm}")
    base_pm = _require_finite(bm_raw[pm], f"baseline.{pm}")

    cand_path_res = Path(args.candidate).resolve()
    inc_path_res = Path(args.incumbent).resolve()
    base_path_res = Path(args.baseline).resolve()
    cand_items: dict[str, dict[str, Any]] | None = None
    inc_items: dict[str, dict[str, Any]] | None = None
    base_items: dict[str, dict[str, Any]] | None = None
    try:
        if preloaded_items is None:
            cand_items = _extract_structured_items(cand, "candidate", stat_name, cand_path_res.parent, item_key)
            base_items = _extract_structured_items(base, "baseline", stat_name, base_path_res.parent, item_key)
        else:
            cand_items, base_items = preloaded_items
        inc_items = _extract_structured_items(inc, "incumbent", stat_name, inc_path_res.parent, item_key)
    except ValueError as exc:
        tripwires.append(f"invalid_item_records:{exc}")

    ids_key = "holdout_expected_item_ids" if holdout else "expected_item_ids"
    strata_key = "holdout_expected_strata" if holdout else "expected_strata"
    if rules.get(ids_key):
        exp_ids = {str(x) for x in rules[ids_key]}
        if not cand_items or set(cand_items.keys()) != exp_ids:
            tripwires.append("item_id_coverage_mismatch_vs_expected_dev_ids")

    if rules.get(strata_key):
        exp_strata = {str(x) for x in rules[strata_key]}
        got_strata = {r["stratum_id"] for r in (cand_items or {}).values() if r["stratum_id"] is not None}
        if got_strata != exp_strata:
            tripwires.append(f"stratum_coverage_mismatch:got={sorted(got_strata)}!=expected={sorted(exp_strata)}")

    if stage == "confirm" and not bool(rules.get("allow_aggregate_only_confirm", False)):
        if cand_items is None or inc_items is None:
            tripwires.append("missing_paired_items_for_confirm")
        if decision_mode == "noninferiority_efficiency" and base_items is None:
            tripwires.append("missing_paired_baseline_items_for_m0_noninferiority")
        if "max_cumulative_primary_regression" in rules and base_items is None:
            tripwires.append("missing_paired_baseline_items_for_cumulative_guardrail")
    if enforce_fresh_block is not None:
        expected = set(enforce_fresh_block)
        for label, items in (("candidate", cand_items), ("incumbent", inc_items), ("baseline", base_items)):
            if not items or {r["cluster_id"] for r in items.values()} != expected:
                tripwires.append(f"confirm_evidence_block_mismatch:{label}")

    min_effect = _require_finite(rules.get("min_improvement", 0.0), "min_improvement")
    se_aa = _require_finite(rules.get("aa_noise_se", 0.0), "aa_noise_se")
    se_reported = (
        _require_finite(
            cm_raw.get("primary_se", im_raw.get("primary_se", rules.get("fallback_se", 0.0))),
            "reported_se",
        )
        if bool(rules.get("allow_aggregate_only_confirm", False)) or stage != "confirm"
        else 0.0
    )
    se_extra = max(se_aa, se_reported)

    paired_vs_inc: dict[str, Any] | None = None
    paired_vs_base: dict[str, Any] | None = None
    eff_paired: dict[str, Any] | None = None
    obs_log_speedup: float | None = None
    eff_lcb_log_speedup: float | None = None
    eff_req_log_speedup: float = 0.0

    q_null_margin = 0.0 if decision_mode == "noninferiority_efficiency" else min_effect
    q_margin_mode = "additive" if decision_mode == "noninferiority_efficiency" else margin_mode

    if not tripwires:
        try:
            if cand_items is not None and inc_items is not None:
                seed_str = f"{seed_scope}:{comparator_id}:primary"
                det_seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
                paired_vs_inc = _paired_bootstrap(
                    cand_items,
                    inc_items,
                    ref_label="incumbent",
                    stat_name=stat_name,
                    weighting=weighting,
                    bootstrap_mode=bootstrap_mode,
                    direction_sign=sign,
                    alpha_one_sided=eff_alpha,
                    null_margin=q_null_margin,
                    margin_mode=q_margin_mode,
                    metric_bound=metric_bound,
                    claim_scope=claim_scope,
                    external_noise_se=se_extra,
                    seed=det_seed,
                    n_perm=n_perm,
                )
                cand_pm = paired_vs_inc["obs_candidate"]
                inc_pm = paired_vs_inc["obs_reference"]
                if paired_vs_inc["low_cluster_warning"]:
                    warnings.append(
                        f"low_cluster_count_for_{stat_name}:n_clusters={paired_vs_inc['n_clusters']}"
                    )
            if cand_items is not None and base_items is not None:
                base_alpha = eff_alpha if stage == "confirm" else alpha
                paired_vs_base = _paired_bootstrap(
                    cand_items,
                    base_items,
                    ref_label="baseline",
                    stat_name=stat_name,
                    weighting=weighting,
                    bootstrap_mode=bootstrap_mode,
                    direction_sign=sign,
                    alpha_one_sided=base_alpha,
                    null_margin=0.0,
                    margin_mode=q_margin_mode,
                    metric_bound=metric_bound,
                    claim_scope=claim_scope,
                    external_noise_se=se_extra,
                    seed=20261117,
                    n_perm=n_perm,
                )
                base_pm = paired_vs_base["obs_reference"]

            if decision_mode == "noninferiority_efficiency":
                if "efficiency_metric" not in rules:
                    tripwires.append("missing_efficiency_metric_in_rules")
                else:
                    eff_m = str(rules["efficiency_metric"])
                    eff_dir = str(rules.get("efficiency_direction", "minimize")).lower()
                    eff_sign = -1.0 if eff_dir == "minimize" else 1.0
                    min_speedup = _require_finite(rules.get("min_speedup_ratio", 1.10), "min_speedup_ratio")
                    if min_speedup <= 1.0:
                        raise ValueError(f"min_speedup_ratio must be > 1.0, got {min_speedup}")
                    eff_req_log_speedup = math.log(min_speedup)
                    if (
                        cand_items is not None
                        and inc_items is not None
                        and all(eff_m in r.get("aux", {}) for r in cand_items.values())
                        and all(eff_m in r.get("aux", {}) for r in inc_items.values())
                    ):
                        c_eff_recs = {iid: {**r, "value": r["aux"][eff_m]} for iid, r in cand_items.items()}
                        i_eff_recs = {iid: {**r, "value": r["aux"][eff_m]} for iid, r in inc_items.items()}
                        det_eff_seed = int(hashlib.sha256(f"eff:{seed_scope}:{comparator_id}".encode("utf-8")).hexdigest()[:8], 16)
                        eff_paired = _paired_bootstrap(
                            c_eff_recs,
                            i_eff_recs,
                            ref_label="incumbent_efficiency",
                            stat_name="mean",
                            weighting=weighting,
                            bootstrap_mode=bootstrap_mode,
                            direction_sign=eff_sign,
                            alpha_one_sided=eff_alpha,
                            null_margin=min_speedup - 1.0,
                            margin_mode="log_ratio",
                            claim_scope=claim_scope,
                            external_noise_se=_require_nonneg_finite(rules.get("aa_efficiency_log_se", 0.0), "aa_efficiency_log_se"),
                            seed=det_eff_seed,
                            n_perm=n_perm,
                        )
                        obs_log_speedup = eff_paired["signed_gain"]
                        eff_lcb_log_speedup = eff_paired["lcb_one_sided"]
                    elif eff_m in cm_raw and eff_m in im_raw and (stage != "confirm" or bool(rules.get("allow_aggregate_only_confirm", False))):
                        ce_v = _require_finite(cm_raw[eff_m], f"candidate.{eff_m}")
                        ie_v = _require_finite(im_raw[eff_m], f"incumbent.{eff_m}")
                        if ce_v <= 0.0 or ie_v <= 0.0:
                            raise ValueError(f"Efficiency metric {eff_m!r} must be > 0")
                        obs_log_speedup = eff_sign * (math.log(ce_v) - math.log(ie_v))
                        eff_se_log = _require_nonneg_finite(rules.get("aa_efficiency_log_se", 0.0), "aa_efficiency_log_se")
                        eff_lcb_log_speedup = obs_log_speedup - _approx_inv_norm(1.0 - eff_alpha) * eff_se_log
                    else:
                        tripwires.append(f"missing_paired_efficiency_aux:{eff_m}")
        except ValueError as exc:
            tripwires.append(f"pairing_integrity_failed:{exc}")

    min_confirm_clusters = max(2, int(rules.get("min_confirm_clusters", 2)))
    min_confirm_seeds = max(2, int(rules.get("min_confirm_seeds", 2)))
    if stage == "confirm" and paired_vs_inc is not None:
        if paired_vs_inc["insufficient_replication"] or paired_vs_inc["n_clusters"] < min_confirm_clusters:
            tripwires.append(
                f"insufficient_clusters_for_confirm:n_clusters={paired_vs_inc['n_clusters']}<{min_confirm_clusters}"
            )
        if paired_vs_inc.get("crossed_design_complete") is False and claim_scope != "fixed_checkpoint":
            tripwires.append("incomplete_crossed_anova_grid")
        if paired_vs_inc.get("crossed_equal_cell_counts") is False and claim_scope != "fixed_checkpoint":
            tripwires.append("unequal_crossed_anova_cell_counts")
        if paired_vs_inc["bootstrap_mode"] in {"crossed_seed_cluster", "crossed_seed_user_item"} and paired_vs_inc["n_seeds"] < min_confirm_seeds:
            tripwires.append(
                f"insufficient_seeds_for_crossed_confirm:n_seeds={paired_vs_inc['n_seeds']}<{min_confirm_seeds}"
            )
        min_items = max(2, int(rules.get("min_confirm_item_clusters", 2)))
        if paired_vs_inc["bootstrap_mode"] in {"crossed_user_item", "crossed_seed_user_item"} and paired_vs_inc["n_item_clusters"] < min_items:
            tripwires.append(f"insufficient_item_clusters_for_confirm:{paired_vs_inc['n_item_clusters']}<{min_items}")

    se_boot = paired_vs_inc["se"] if paired_vs_inc is not None else 0.0
    se = se_boot if paired_vs_inc is not None else se_extra
    z_req = _approx_inv_norm(1.0 - eff_alpha)

    if margin_mode == "log_ratio" and paired_vs_inc is None and not tripwires:
        if cand_pm <= 0.0 or inc_pm <= 0.0 or base_pm <= 0.0:
            tripwires.append("nonpositive_scalar_metric_for_log_ratio")
        else:
            cand_pm = math.log(cand_pm)
            inc_pm = math.log(inc_pm)
            base_pm = math.log(base_pm)

    raw_delta = cand_pm - inc_pm
    signed_gain = sign * raw_delta
    eff_min_effect = (
        paired_vs_inc["effective_null_margin"]
        if paired_vs_inc is not None
        else _compute_margin_requirement(inc_pm, q_null_margin, q_margin_mode, sign, metric_bound)
    )
    if paired_vs_inc is not None:
        lcb_vs_inc = paired_vs_inc["lcb_one_sided"]
        margin_excess_lcb = paired_vs_inc["margin_excess_lcb_one_sided"]
    else:
        lcb_vs_inc = signed_gain - z_req * se
        margin_excess_lcb = lcb_vs_inc - max(0.0, eff_min_effect)

    signed_vs_base = sign * (cand_pm - base_pm)
    z_base = _approx_inv_norm(1.0 - (eff_alpha if stage == "confirm" else alpha))
    if paired_vs_base is not None:
        lcb_vs_base = paired_vs_base["lcb_one_sided"]
    else:
        lcb_vs_base = signed_vs_base - z_base * se

    # Guard against SE == 0 / degenerate constant-shift & small-cluster under-coverage on Confirm
    zero_variance_blocked = False
    small_cluster_signflip_blocked = False
    if stage == "confirm" and decision_mode == "superiority" and not tripwires:
        if paired_vs_inc is not None:
            n_cl = paired_vs_inc["n_clusters"]
            sf_p = paired_vs_inc["signflip_p_one_sided"]
            if paired_vs_inc["bootstrap_mode"] in {"crossed_user_item", "crossed_seed_user_item"}:
                zero_variance_blocked = se_boot <= 1e-15
            elif se_boot <= 1e-15:
                zero_variance_blocked = sf_p > eff_alpha
            elif paired_vs_inc["bootstrap_mode"] == "cluster_1d" and paired_vs_inc["distinct_cluster_deltas"] < 2:
                zero_variance_blocked = sf_p > eff_alpha
            elif n_cl < 20 and bool(rules.get("enforce_small_cluster_signflip", True)):
                if sf_p > eff_alpha:
                    small_cluster_signflip_blocked = True
        elif se <= 1e-15:
            zero_variance_blocked = True

    def classify() -> tuple[str, str]:
        if tripwires:
            return "ROLLBACK", "invalid_evidence"
        if decision_mode == "noninferiority_efficiency":
            max_q_drop = _require_nonneg_finite(rules.get("max_quality_drop_vs_baseline", 0.0), "max_quality_drop_vs_baseline")
            max_q_inc = _require_nonneg_finite(rules.get("max_quality_drop_vs_incumbent", max_q_drop), "max_quality_drop_vs_incumbent")
            if stage == "search":
                if signed_vs_base >= -max_q_drop and signed_gain >= -max_q_inc and obs_log_speedup is not None and obs_log_speedup > eff_req_log_speedup:
                    return "PROVISIONAL_KEEP", "search_ni_quality_and_speedup_passed"
                return "ROLLBACK", f"search_ni_efficiency_failed:q_vs_base={signed_vs_base:.6f},log_speedup={obs_log_speedup}"
            if stage != "confirm":
                raise ValueError("noninferiority_efficiency requires search or confirm")
            if paired_vs_base is not None and paired_vs_base["insufficient_replication"]:
                return "ROLLBACK", "insufficient_baseline_replication"
            if lcb_vs_base <= -max_q_drop:
                return "ROLLBACK", f"m0_quality_noninferiority_failed:lcb_vs_base={lcb_vs_base:.6f}<=-{max_q_drop:.6f}"
            if lcb_vs_inc <= -max_q_inc:
                return "ROLLBACK", f"incumbent_quality_noninferiority_failed:lcb_vs_inc={lcb_vs_inc:.6f}<=-{max_q_inc:.6f}"
            eff_excess_ok = (
                not eff_paired["insufficient_replication"] and eff_paired["margin_excess_lcb_one_sided"] > 0.0
                if eff_paired is not None
                else eff_lcb_log_speedup is not None and eff_lcb_log_speedup > eff_req_log_speedup
            )
            if obs_log_speedup is None or obs_log_speedup < eff_req_log_speedup or not eff_excess_ok:
                return "ROLLBACK", f"efficiency_gain_insufficient:obs_log_speedup={obs_log_speedup},lcb={eff_lcb_log_speedup},req={eff_req_log_speedup:.6f}"
            if (
                eff_paired is not None and eff_paired["n_clusters"] < 20
                and eff_paired["signflip_resolution"] is not None
                and bool(rules.get("enforce_small_cluster_signflip", True))
                and eff_paired["signflip_p_one_sided"] > eff_alpha
            ):
                return "ROLLBACK", "efficiency_signflip_failed"
            return "PROMOTE", "confirmed_m0_noninferiority_and_efficiency_superiority"
        if stage == "search":
            if signed_gain > max(0.0, eff_min_effect):
                return "PROVISIONAL_KEEP", "search_point_improvement_passed_safety"
            return "ROLLBACK", f"search_no_gain:signed_gain={signed_gain:.6f}<=min={max(0.0, eff_min_effect):.6f}"
        if stage == "confirm":
            if zero_variance_blocked:
                return "ROLLBACK", f"degenerate_zero_variance_insufficient_clusters:signed_gain={signed_gain:.6f},se_boot={se_boot:.6e}"
            if small_cluster_signflip_blocked:
                return "ROLLBACK", f"small_cluster_signflip_failed:signflip_p={paired_vs_inc['signflip_p_one_sided']:.5f}>eff_alpha={eff_alpha:.5f}"
            if signed_gain >= eff_min_effect and signed_gain > 0.0:
                if margin_excess_lcb > 0.0:
                    return "PROMOTE", "confirmed_statistically_and_practically"
                if se > 0.0:
                    return "INCONCLUSIVE_RETEST", f"positive_point_gain_but_underpowered:lcb={lcb_vs_inc:.6f},margin={eff_min_effect:.6f}"
            return "ROLLBACK", f"confirm_insufficient_gain:signed_gain={signed_gain:.6f},lcb={lcb_vs_inc:.6f},margin={eff_min_effect:.6f}"
        if stage == "frontier_explore":
            max_step_reg = _require_finite(rules.get("max_regression_se_mult", 1.5), "max_regression_se_mult") * se
            fc = rules.get("frontier_coverage_metric")
            min_fc_gain = _require_finite(rules.get("min_frontier_gain", 1.0), "min_frontier_gain")
            fc_ok = True
            if fc:
                if fc not in cm_raw or fc not in im_raw:
                    tripwires.append(f"missing_frontier_metric:{fc}")
                    return "ROLLBACK", "frontier_stagnant_or_step_regressed"
                fc_gain = _require_finite(cm_raw[fc], f"cand.{fc}") - _require_finite(im_raw[fc], f"inc.{fc}")
                fc_ok = fc_gain >= min_fc_gain
            if signed_gain >= -max_step_reg and (fc_ok or lcb_vs_inc > 0.0):
                outcome = "PROMOTE" if (lcb_vs_inc > 0.0 and se > 0.0) or fc_ok else "PROVISIONAL_KEEP"
                return outcome, "frontier_expanded_within_non_regression_bound"
            return "ROLLBACK", "frontier_stagnant_or_step_regressed"
        raise ValueError(f"Unsupported decision_stage={stage!r}")

    decision, decision_reason = classify()
    primary_replicated = decision == "PROMOTE"
    guardrail_rules = {**rules, "expected_strata": rules.get("holdout_expected_strata", [])} if holdout else rules
    tripwires.extend(_evaluate_candidate_guardrails(cand, cand_items, guardrail_rules))
    tripwires.extend(_evaluate_baseline_guardrails(cm_raw, bm_raw, rules, pm, direction, lcb_vs_base))
    if rules.get("paired_aux_noninferiority"):
        if not cand_items or not inc_items:
            tripwires.append("missing_paired_items_for_aux_noninferiority")
        else:
            try:
                tripwires.extend(_evaluate_paired_aux_noninferiority(
                    cand_items=cand_items, ref_items=inc_items, rules=rules,
                    weighting=weighting, bootstrap_mode=bootstrap_mode, claim_scope=claim_scope,
                    eff_alpha=(aux_alpha if aux_alpha is not None else eff_alpha) / len(rules["paired_aux_noninferiority"]),
                    seed_prefix=f"aux:{seed_scope}:{comparator_id}",
                ))
            except ValueError as exc:
                tripwires.append(f"invalid_paired_aux_evidence:{exc}")
    if tripwires:
        decision, decision_reason = "ROLLBACK", "guardrail_or_regression"

    record: dict[str, Any] = {
        **common,
        "decision_mode": decision_mode, "margin_mode": margin_mode,
        "effective_null_margin": eff_min_effect, "primary_metric": pm, "primary_statistic": stat_name,
        "aggregation_weighting": weighting, "direction": direction,
        "bootstrap_mode": paired_vs_inc["bootstrap_mode"] if paired_vs_inc else "none",
        "candidate_primary": cand_pm, "incumbent_primary": inc_pm, "baseline_primary": base_pm,
        "raw_delta_vs_incumbent": raw_delta, "signed_improvement": signed_gain,
        "lcb_one_sided_vs_incumbent": lcb_vs_inc, "margin_excess_lcb_one_sided": margin_excess_lcb,
        "ci95_vs_incumbent": [paired_vs_inc["ci95_low"], paired_vs_inc["ci95_high"]] if paired_vs_inc else None,
        "signflip_p_one_sided_vs_incumbent": paired_vs_inc["signflip_p_one_sided"] if paired_vs_inc else None,
        "signflip_method": paired_vs_inc["signflip_method"] if paired_vs_inc else "none",
        "signflip_resolution": paired_vs_inc["signflip_resolution"] if paired_vs_inc else None,
        "lcb_one_sided_vs_baseline": lcb_vs_base, "effective_se": se,
        "tripwires": tripwires, "warnings": warnings,
        "decision": decision, "decision_reason": decision_reason,
    }
    if paired_vs_inc:
        for name in ("df_satterthwaite", "margin_excess_df_satterthwaite", "se_crossed", "nonpositive_crossed_variance_fallback",
                     "n_item_clusters", "crossed_equal_cell_counts"):
            if name in paired_vs_inc:
                record[name] = paired_vs_inc[name]
    if decision_mode == "noninferiority_efficiency":
        record.update(obs_log_speedup=obs_log_speedup, lcb_log_speedup=eff_lcb_log_speedup, required_log_speedup=eff_req_log_speedup)
    if holdout:
        record["primary_replicated"] = primary_replicated
        record["paired_summary"] = paired_vs_inc
    return record


def _evaluate_baseline_guardrails(
    candidate: dict[str, Any], baseline: dict[str, Any], rules: dict[str, Any],
    primary_metric: str, direction: str, lcb_vs_base: float | None = None,
) -> list[str]:
    trips: list[str] = []
    for op in ("min", "max"):
        for name, raw_limit in rules.get(f"{op}_ratio_vs_baseline", {}).items():
            if name not in candidate or name not in baseline:
                trips.append(f"missing_guardrail_metric:{name}")
                continue
            cv = _require_nonneg_finite(candidate[name], f"candidate.{name}")
            bv = _require_nonneg_finite(baseline[name], f"baseline.{name}")
            limit = _require_nonneg_finite(raw_limit, f"{op}_ratio_vs_baseline.{name}")
            if bv <= 1e-12:
                trips.append(f"invalid_baseline_denominator:{name}={bv}<=0")
            elif op == "min" and cv / bv < limit:
                trips.append(f"baseline_ratio_too_low:{name}_ratio={cv/bv:.4f}<{limit}")
            elif op == "max" and cv / bv > limit:
                trips.append(f"baseline_ratio_too_high:{name}_ratio={cv/bv:.4f}>{limit}")
    for name, raw in rules.get("max_degradation_vs_baseline", rules.get("max_drop_vs_baseline", {})).items():
        if name not in candidate or name not in baseline:
            trips.append(f"missing_guardrail_metric:{name}")
            continue
        cv = _require_finite(candidate[name], f"candidate.{name}")
        bv = _require_finite(baseline[name], f"baseline.{name}")
        limit = _require_nonneg_finite(raw, f"max_degradation_vs_baseline.{name}")
        metric_dir = str(rules.get("metric_directions", {}).get(name, direction if name == primary_metric else "maximize")).lower()
        if metric_dir not in {"maximize", "minimize"}:
            raise ValueError(f"Invalid guardrail direction for {name}")
        degraded = bv - cv if metric_dir == "maximize" else cv - bv
        if degraded > limit:
            trips.append(f"baseline_degradation_exceeded:{name}_degradation={degraded:.6f}>{limit}")
    if lcb_vs_base is not None and "max_cumulative_primary_regression" in rules:
        limit = _require_nonneg_finite(rules["max_cumulative_primary_regression"], "max_cumulative_primary_regression")
        if lcb_vs_base < -limit:
            trips.append(f"cumulative_baseline_regression:lcb_vs_base={lcb_vs_base:.6f}<-{limit:.6f}")
    return trips


def _evaluate_paired_aux_noninferiority(
    *,
    cand_items: dict[str, dict[str, Any]],
    ref_items: dict[str, dict[str, Any]],
    rules: dict[str, Any],
    weighting: str,
    bootstrap_mode: str,
    claim_scope: str,
    eff_alpha: float,
    seed_prefix: str,
) -> list[str]:
    """Enforces paired bootstrap non-inferiority on auxiliary item-level slice/diagnostic metrics."""
    trips: list[str] = []
    aux_ni_cfg = rules.get("paired_aux_noninferiority") or {}
    aux_dirs = {k: str(v).lower() for k, v in (rules.get("paired_aux_directions") or {}).items()}
    n_perm = _require_positive_int(rules.get("n_perm", DEFAULT_N_PERM), "n_perm")
    if not 0.0 < eff_alpha < 0.5 or 1.0 - eff_alpha == 1.0:
        raise ValueError("Auxiliary multiplicity-adjusted alpha is unresolvable")
    for aux_name, margin in aux_ni_cfg.items():
        relative = isinstance(margin, dict)
        max_drop = margin["relative"] if relative else margin
        aux_dir = aux_dirs.get(aux_name, "maximize")
        if aux_dir not in {"maximize", "minimize"}:
            raise ValueError(f"Invalid paired auxiliary direction for {aux_name}")
        aux_sign = -1.0 if aux_dir == "minimize" else 1.0
        if not all(aux_name in r.get("aux", {}) for r in cand_items.values()) or not all(
            aux_name in r.get("aux", {}) for r in ref_items.values()
        ):
            trips.append(f"missing_paired_aux_metric:{aux_name}")
            continue
        c_aux = {iid: {**r, "value": r["aux"][aux_name]} for iid, r in cand_items.items()}
        r_aux = {iid: {**r, "value": r["aux"][aux_name]} for iid, r in ref_items.items()}
        det_seed = int(hashlib.sha256(f"{seed_prefix}:{aux_name}".encode("utf-8")).hexdigest()[:8], 16)
        aux_res = _paired_bootstrap(
            c_aux,
            r_aux,
            ref_label=f"ref_aux[{aux_name}]",
            stat_name="mean",
            weighting=weighting,
            bootstrap_mode=bootstrap_mode,
            direction_sign=aux_sign,
            alpha_one_sided=eff_alpha,
            null_margin=-max_drop if relative else 0.0,
            margin_mode="relative_reference" if relative else "additive",
            claim_scope=claim_scope,
            seed=det_seed,
            n_perm=n_perm,
        )
        if aux_res["insufficient_replication"] or (
            aux_res.get("crossed_equal_cell_counts") is False and claim_scope != "fixed_checkpoint"
        ):
            trips.append(f"insufficient_paired_aux_replication:{aux_name}")
        elif relative and aux_res["margin_excess_lcb_one_sided"] <= 0.0:
            trips.append(
                f"paired_aux_noninferiority_failed:{aux_name}:relative_margin_excess_lcb="
                f"{aux_res['margin_excess_lcb_one_sided']:.6f}<=0"
            )
        elif not relative and aux_res["lcb_one_sided"] <= -max_drop:
            trips.append(
                f"paired_aux_noninferiority_failed:{aux_name}:lcb={aux_res['lcb_one_sided']:.6f}<=-{max_drop:.6f}"
            )
    return trips


def _evaluate_candidate_guardrails(
    cand: dict[str, Any],
    cand_items: dict[str, dict[str, Any]] | None,
    rules: dict[str, Any],
    *,
    enforce_expected_strata: bool = True,
) -> list[str]:
    tripwires: list[str] = []
    cm_raw = cand.get("metrics", {})
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

    stratum_metrics_raw = cand.get("stratum_metrics", {})
    if rules.get("per_stratum_hard_floors") or rules.get("per_stratum_hard_ceilings"):
        derived_strata: dict[str, dict[str, list[float]]] = {}
        if cand_items:
            for r in cand_items.values():
                st = r["stratum_id"] or "__all__"
                for ak, av in r.get("aux", {}).items():
                    derived_strata.setdefault(st, {}).setdefault(ak, []).append(av)

        exp_st = {str(x) for x in rules.get("expected_strata", [])} if enforce_expected_strata else set()
        all_strata = sorted(set(stratum_metrics_raw.keys()) | set(derived_strata.keys()) | exp_st)
        if not all_strata:
            tripwires.append("missing_strata_for_per_stratum_guardrails")

        for m_name, floor_val in rules.get("per_stratum_hard_floors", {}).items():
            f_req = _require_finite(floor_val, f"per_stratum_hard_floors.{m_name}")
            for st in all_strata:
                if st in stratum_metrics_raw and isinstance(stratum_metrics_raw[st], dict) and m_name in stratum_metrics_raw[st]:
                    sv = _require_finite(stratum_metrics_raw[st][m_name], f"stratum_metrics[{st}].{m_name}")
                elif st in derived_strata and m_name in derived_strata[st]:
                    sv = statistics.fmean(derived_strata[st][m_name])
                else:
                    tripwires.append(f"missing_per_stratum_metric:{st}.{m_name}")
                    continue
                if sv < f_req:
                    tripwires.append(f"per_stratum_floor_failed:{st}.{m_name}={sv:.6f}<{f_req}")

        for m_name, ceil_val in rules.get("per_stratum_hard_ceilings", {}).items():
            c_req = _require_finite(ceil_val, f"per_stratum_hard_ceilings.{m_name}")
            for st in all_strata:
                if st in stratum_metrics_raw and isinstance(stratum_metrics_raw[st], dict) and m_name in stratum_metrics_raw[st]:
                    sv = _require_finite(stratum_metrics_raw[st][m_name], f"stratum_metrics[{st}].{m_name}")
                elif st in derived_strata and m_name in derived_strata[st]:
                    sv = statistics.fmean(derived_strata[st][m_name])
                else:
                    tripwires.append(f"missing_per_stratum_metric:{st}.{m_name}")
                    continue
                if sv > c_req:
                    tripwires.append(f"per_stratum_ceiling_failed:{st}.{m_name}={sv:.6f}>{c_req}")
    return tripwires


def _atomic_create_seal(seal_path: Path, payload: dict[str, Any]) -> None:
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(seal_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _replace_seal(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".seal.tmp", delete=False, mode="w",
                                     encoding="utf-8", newline="\n") as stream:
        tmp = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except (OSError, ValueError):
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def cmd_select_search_winner(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger).resolve() if args.ledger else None
    with _ledger_exclusive_lock(ledger_path):
        rows = _read_and_verify_ledger(ledger_path) if ledger_path else []
        completed = {r.get("confirm_look_id") for r in rows if r.get("confirm_look_outcome")}
        if any(
            r.get("halt_recommended") is True or r.get("decision_stage") == "holdout"
            or r.get("decision_stage") == "confirm_reservation" and r["entry_sha256"] not in completed
            for r in rows
        ):
            print(json.dumps({"decision": "ROLLBACK", "reason": "campaign_halted", "halt_recommended": True}))
            return 1
        return _select_search_winner(args, ledger_path, rows)


def _select_search_winner(
    args: argparse.Namespace, ledger_path: Path | None, prior_rows: list[dict[str, Any]],
) -> int:
    """
    Order-invariant batch Search selector: evaluates every candidate against a frozen incumbent
    and baseline (enforcing all global/per-stratum/ratio tripwires), ranks eligible passers by
    shrunk `lcb` (default when paired items exist) or `signed_gain`, breaks ties deterministically
    by (-wall_clock_sec, sha256(candidate_id)), and logs one auditable batch selection record.
    """
    inc_path = Path(args.incumbent).resolve()
    base_path = Path(args.baseline).resolve()
    rules_path = Path(args.rules).resolve()
    rules = _load_strict_json(rules_path)
    inc = _load_strict_json(inc_path)
    base = _load_strict_json(base_path)

    pm = str(rules["primary_metric"])
    direction = str(rules.get("direction", "maximize")).lower()
    sign = 1.0 if direction == "maximize" else -1.0
    stat_name = str(rules.get("primary_statistic", "mean"))
    weighting = str(rules.get("aggregation_weighting", "equal_item")).lower()
    bootstrap_mode = str(rules.get("bootstrap_mode", "auto")).lower()
    item_key = _item_cluster_key(args, rules)
    n_perm = _require_positive_int(rules.get("n_perm", DEFAULT_N_PERM), "n_perm")
    margin_mode = str(rules.get("margin_mode", "additive")).lower()
    metric_bound = _require_finite(rules["metric_bound"], "metric_bound") if "metric_bound" in rules else None
    claim_scope = str(rules.get("claim_scope", "training_recipe")).lower()
    alpha = _require_finite(rules.get("alpha_one_sided", 0.10), "alpha_one_sided")
    min_effect = _require_finite(rules.get("min_improvement", 0.0), "min_improvement")
    se_aa = _require_finite(rules.get("aa_noise_se", 0.0), "aa_noise_se")
    rank_by = str(args.rank_by or rules.get("search_rank_by", "auto")).lower()

    inc_items = _extract_structured_items(inc, "incumbent", stat_name, inc_path.parent, item_key)
    bm_raw = base.get("metrics", {})
    im_raw = inc.get("metrics", {})

    evaluated: list[dict[str, Any]] = []
    eligible: list[tuple[float, float, str, dict[str, Any]]] = []

    # Sort candidate paths by candidate_id hash first so evaluation itself has zero order dependence
    cand_loaded: list[tuple[str, Path, dict[str, Any]]] = []
    for raw_cp in args.candidates:
        cp = Path(raw_cp).resolve()
        c_obj = _load_strict_json(cp)
        cid = str(c_obj.get("candidate_id", cp.stem))
        cand_loaded.append((cid, cp, c_obj))
    cand_loaded.sort(key=lambda t: hashlib.sha256(t[0].encode("utf-8")).hexdigest())

    for cid, cp, cand in cand_loaded:
        cm_raw = cand.get("metrics", {})
        trips: list[str] = []
        if pm not in cm_raw or pm not in im_raw or pm not in bm_raw:
            trips.append(f"missing_primary_metric:{pm}")
        cand_items: dict[str, dict[str, Any]] | None = None
        try:
            cand_items = _extract_structured_items(cand, f"candidate[{cid}]", stat_name, cp.parent, item_key)
        except ValueError as exc:
            trips.append(f"invalid_item_records:{exc}")

        trips.extend(_evaluate_candidate_guardrails(cand, cand_items, rules, enforce_expected_strata=True))
        trips.extend(_evaluate_baseline_guardrails(cm_raw, bm_raw, rules, pm, direction))

        signed_gain = 0.0
        lcb_val = 0.0
        eff_min = min_effect
        if not trips:
            try:
                if cand_items is not None and inc_items is not None:
                    det_seed = int(hashlib.sha256(f"batch_search:{cid}".encode("utf-8")).hexdigest()[:8], 16)
                    p_res = _paired_bootstrap(
                        cand_items,
                        inc_items,
                        ref_label="incumbent",
                        stat_name=stat_name,
                        weighting=weighting,
                        bootstrap_mode=bootstrap_mode,
                        direction_sign=sign,
                        alpha_one_sided=alpha,
                        null_margin=min_effect,
                        margin_mode=margin_mode,
                        metric_bound=metric_bound,
                        claim_scope=claim_scope,
                        external_noise_se=se_aa,
                        seed=det_seed,
                        n_perm=n_perm,
                    )
                    signed_gain = p_res["signed_gain"]
                    lcb_val = p_res["lcb_one_sided"]
                    eff_min = p_res["effective_null_margin"]
                else:
                    c_pm = _require_finite(cm_raw[pm], f"{cid}.{pm}")
                    i_pm = _require_finite(im_raw[pm], f"incumbent.{pm}")
                    if margin_mode == "log_ratio":
                        c_pm, i_pm = math.log(c_pm), math.log(i_pm)
                    signed_gain = sign * (c_pm - i_pm)
                    lcb_val = signed_gain
                    eff_min = _compute_margin_requirement(i_pm, min_effect, margin_mode, sign, metric_bound)
            except ValueError as exc:
                trips.append(f"pairing_integrity_failed:{exc}")

        passed = (len(trips) == 0) and (signed_gain > max(0.0, eff_min))
        use_lcb = (rank_by == "lcb") or (rank_by == "auto" and cand_items is not None and inc_items is not None)
        rank_score = lcb_val if use_lcb else signed_gain
        wall_sec = float(cm_raw.get("eval_wall_clock_sec", cm_raw.get("guard_wall_clock_sec", 1.0e9)))
        tie_hash = hashlib.sha256(cid.encode("utf-8")).hexdigest()

        rec_summary = {
            "candidate_id": cid,
            "path": cp.as_posix(),
            "passed_search": passed,
            "signed_gain": signed_gain,
            "lcb_one_sided": lcb_val,
            "rank_score": rank_score,
            "wall_clock_sec": wall_sec,
            "tripwires": trips,
        }
        evaluated.append(rec_summary)
        if passed:
            eligible.append((round(rank_score, 9), -round(wall_sec, 6), tie_hash, rec_summary))

    eligible.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    winner = eligible[0][3] if eligible else None
    decision = "PROVISIONAL_KEEP" if winner is not None else "ROLLBACK"

    out_body: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trial_id": args.round_id,
        "decision_stage": "search",
        "selection_protocol": "batch_order_invariant",
        "candidate_id": winner["candidate_id"] if winner else "NONE",
        "selected_candidate_id": winner["candidate_id"] if winner else None,
        "selected_path": winner["path"] if winner else None,
        "decision": decision,
        "n_perm": n_perm,
        "p_floor": 1.0 / (1 + n_perm),
        "candidates_evaluated": len(evaluated),
        "eligible_count": len(eligible),
        "leaderboard": sorted(evaluated, key=lambda r: (r["passed_search"], round(r["rank_score"], 9)), reverse=True),
    }
    if ledger_path:
        reset_policy = str(rules.get("streak_reset_policy", "confirm_only")).lower()
        c_rb, u_rb = _compute_rollback_streak(prior_rows, decision, reset_policy)
        cap = int(rules.get("max_consecutive_rollbacks", 0))
        out_body.update(consecutive_rollbacks=c_rb, unresolved_rollbacks_since_promote=u_rb,
                        halt_recommended=cap > 0 and c_rb >= cap)
        out_body = _append_ledger_entry(ledger_path, prior_rows, out_body)

    print(json.dumps(out_body, indent=2))
    return 0 if winner is not None else 1


def cmd_gate_confirm_iut(args: argparse.Namespace) -> int:
    """Evaluate the declared comparator vector within one reserved IUT look."""
    rules_path = Path(args.rules).resolve()
    raw = rules_path.read_bytes()
    rules = _decode_json(raw, rules_path)
    if str(rules.get("decision_stage", "confirm")).lower() != "confirm":
        print(json.dumps({"status": "ERROR", "error": "gate-confirm-iut requires rules with decision_stage='confirm'"}))
        return 2

    comp_map = _parse_comparators(args.comparator, "--comparator")
    if len({Path(path).resolve() for path in comp_map.values()}) != len(comp_map):
        raise ValueError("Confirm comparator names must refer to distinct evidence files")
    req_comps = _required_comparators(rules)

    missing_req = [c for c in req_comps if c not in comp_map]
    if missing_req:
        print(
            json.dumps(
                {
                    "status": "MISSING_REQUIRED_COMPARATOR",
                    "iut_decision": "ROLLBACK",
                    "missing_comparators": missing_req,
                },
                indent=2,
            )
        )
        return 1

    ordered_comps = req_comps + [c for c in sorted(comp_map.keys()) if c not in req_comps]
    if not ordered_comps:
        print(json.dumps({"status": "ERROR", "error": "At least one --comparator NAME=PATH is required"}))
        return 2

    return _run_gate_look(args, rules, hashlib.sha256(raw).hexdigest(), {name: comp_map[name] for name in ordered_comps})


def _normalize_aux_margins(value: Any) -> dict[str, float | dict[str, float]]:
    if not isinstance(value, dict) or any(not name.strip() for name in value):
        raise ValueError("Auxiliary noninferiority margins must be an object with nonempty metric names")
    pairs: dict[str, float | dict[str, float]] = {}
    for name, margin in value.items():
        label = f"paired_aux_noninferiority.{name}"
        if isinstance(margin, dict):
            if margin.keys() != {"relative"}:
                raise ValueError(f"{label} must be an additive number or an object with a relative fraction")
            fraction = _require_nonneg_finite(margin["relative"], f"{label}.relative")
            if fraction > 1.0:
                raise ValueError(f"{label}.relative must be at most 1")
            pairs[name] = {"relative": fraction}
        else:
            pairs[name] = _require_nonneg_finite(margin, label)
    return pairs


def _resolve_aux_margins(declared: Any, cli: list[str] | None) -> dict[str, float | dict[str, float]]:
    pairs = _normalize_aux_margins(declared)
    if not cli:
        return pairs
    if len(cli) == 1 and cli[0].lstrip().startswith("{"):
        extra = _decode_json(cli[0].encode("utf-8"), Path("aux-noninferiority-margins.json"))
    else:
        extra = {name: float(value) for name, value in _parse_comparators(cli, "--aux-noninferiority-margins").items()}
    for name, limit in _normalize_aux_margins(extra).items():
        registered = pairs.get(name)
        compatible = (
            registered is None
            or isinstance(limit, dict) and isinstance(registered, dict) and limit["relative"] <= registered["relative"]
            or isinstance(limit, float) and isinstance(registered, float) and limit <= registered
        )
        if not compatible:
            raise ValueError(f"Cannot change units or loosen the frozen auxiliary margin for {name}")
        pairs[name] = limit
    return pairs


def cmd_gate_holdout(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger).resolve()
    with _ledger_exclusive_lock(ledger_path):
        return _gate_holdout_locked(args, ledger_path)


def _gate_holdout_locked(args: argparse.Namespace, ledger_path: Path) -> int:
    seal_path = Path(args.seal_file).resolve()
    manifest_path = Path(args.manifest).resolve()
    rules_path = Path(args.rules).resolve()

    def report(status: str, rc: int, **fields: Any) -> int:
        print(json.dumps({"status": status, **fields}, indent=2, allow_nan=False))
        return rc

    ok_man, man_rep = _verify_manifest_internal(manifest_path, None, args.expected_root_sha256)
    if not ok_man:
        return report("BLOCKED_MANIFEST_TAMPERED", 1, manifest_report=man_rep)
    manifest_sha = man_rep["manifest_root_sha256"]
    protected_root = Path(_load_strict_json(manifest_path)["root"]).resolve()
    receipt_path = protected_root / ".governed-holdout" / "HOLDOUT_SEAL.json"
    try:
        prior_rows = _read_and_verify_ledger(ledger_path)
    except (OSError, ValueError, KeyError) as exc:
        return report("BLOCKED_LEDGER_TAMPERED", 1, error=str(exc))
    holdout_rows = [
        r for r in prior_rows if r.get("decision_stage") == "holdout"
        and r.get("manifest_root_sha256") in (None, manifest_sha)
    ]
    if args.phase != "finalize" and (receipt_path.exists() or seal_path.exists() or holdout_rows):
        return report("HOLDOUT_ALREADY_CONSUMED", 2, error="This protected Holdout has already been reserved or evaluated.")
    if args.phase == "finalize":
        if not receipt_path.exists() or not holdout_rows:
            return report("ERROR", 2, error="Cannot finalize without the canonical Holdout reservation and ledger receipt")
        if holdout_rows[-1].get("decision") != "HOLDOUT_RESERVED":
            return report("HOLDOUT_ALREADY_CONSUMED", 2, error="The Holdout ledger is terminal")
        if not seal_path.exists():
            return report("SEAL_BINDING_MISMATCH", 1, error="Cannot find the original reserved seal")
        seal_data = _load_strict_json(seal_path)
        if seal_data.get("status") != "RESERVED":
            return report("HOLDOUT_ALREADY_CONSUMED", 2, error=f"Seal is in state {seal_data.get('status')!r}")

    raw_rules = rules_path.read_bytes()
    rules = _decode_json(raw_rules, rules_path)
    rules_sha = hashlib.sha256(raw_rules).hexdigest()
    alpha, _ = _confirm_alpha_plan(rules)
    n_perm = _require_positive_int(rules.get("n_perm", DEFAULT_N_PERM), "n_perm")
    req_comps = _required_comparators(rules, args.required_comparators, default=("incumbent",))
    cand_id = str(args.candidate_id)
    identities = _artifact_fingerprints(args.candidate_artifacts)

    latest_by_comp: dict[str, dict[str, Any]] = {}
    look_outcomes = {r.get("confirm_look_id"): r["confirm_look_outcome"] for r in prior_rows if "confirm_look_outcome" in r}
    for row in prior_rows:
        if row.get("decision_stage") == "confirm" and row.get("candidate_id") == cand_id:
            latest_by_comp[str(row.get("comparator_id") or "incumbent")] = row
    missing_promotes, sha_mismatches = [], []
    for comp in req_comps:
        row = latest_by_comp.get(comp, {})
        recorded_decision = row.get("decision", "MISSING")
        if recorded_decision != "PROMOTE":
            missing_promotes.append(f"{comp}:{recorded_decision}")
        elif row.get("confirm_look_id") and look_outcomes.get(row["confirm_look_id"]) != "PROMOTE":
            missing_promotes.append(f"{comp}:IUT_NOT_PROMOTED")
        if row and (
            row.get("rules_sha256") != rules_sha
            or row.get("candidate_artifact_sha256") != identities["candidate_artifact_sha256"]
            or row.get("artifact_semantic_sha256") not in (None, identities["artifact_semantic_sha256"])
        ):
            sha_mismatches.append(f"{comp}:artifact_or_rules_binding_mismatch")
    if missing_promotes or sha_mismatches:
        return report("BLOCKED_NOT_CONFIRMED", 1, candidate_id=cand_id,
                      missing_or_failed_comparators=missing_promotes, artifact_sha_mismatches=sha_mismatches)
    selected = [latest_by_comp[name] for name in req_comps]
    frozen_aux = selected[0].get("paired_aux_noninferiority", rules.get("paired_aux_noninferiority", {}))
    if any(r.get("paired_aux_noninferiority", frozen_aux) != frozen_aux for r in selected):
        return report("BLOCKED_NOT_CONFIRMED", 1, error="Confirm auxiliary margins differ across required comparators")
    aux_margins = _resolve_aux_margins(frozen_aux, args.aux_noninferiority_margins)
    look_ids = {r.get("confirm_look_id") for r in selected}
    if len(look_ids) > 1:
        return report("BLOCKED_NOT_CONFIRMED", 1, error="Required comparators must all promote in the same registered Confirm look")
    key = _item_cluster_key(args, rules)
    recorded_keys = {r.get("item_cluster_key") for r in selected if r.get("confirm_look_id")}
    if recorded_keys and (len(recorded_keys) != 1 or (key is not None and key not in recorded_keys)):
        return report("BLOCKED_NOT_CONFIRMED", 1, error="Holdout item-cluster key differs from Confirm")
    item_key = next(iter(recorded_keys)) if recorded_keys else key
    references = {name: latest_by_comp[name].get("reference_id") for name in req_comps}
    ref_ids = [value for value in references.values() if value is not None]
    if len(ref_ids) != len(set(ref_ids)):
        return report("BLOCKED_NOT_CONFIRMED", 1, error="Two comparator names refer to the same registered reference identity")
    baseline_ids = {r["baseline_id"] for r in selected if r.get("baseline_id") is not None}
    baseline_id = rules.get("holdout_baseline_id") or (next(iter(baseline_ids)) if len(baseline_ids) == 1 else None)
    if len(baseline_ids) > 1 or (baseline_ids and baseline_id not in baseline_ids):
        return report("BLOCKED_NOT_CONFIRMED", 1, error="Holdout baseline identity differs from Confirm")
    needs_baseline = (
        rules.get("decision_mode") == "noninferiority_efficiency"
        or any(rules.get(k) for k in ("max_ratio_vs_baseline", "min_ratio_vs_baseline", "max_degradation_vs_baseline", "max_drop_vs_baseline"))
        or "max_cumulative_primary_regression" in rules
    )
    refs: dict[str, str] = {}
    baseline_path = args.baseline_holdout
    if args.phase != "reserve":
        refs = _parse_comparators(args.comparator_holdout, "--comparator-holdout")
        missing, extra = set(req_comps) - refs.keys(), refs.keys() - set(req_comps)
        if not args.candidate_holdout or missing or extra:
            return report("MISSING_REQUIRED_COMPARATOR", 1, missing_comparators=sorted(missing),
                          unregistered_comparators=sorted(extra), candidate_supplied=bool(args.candidate_holdout))
        ref_paths = {Path(path).resolve() for path in refs.values()}
        if len(ref_paths) != len(refs) or Path(args.candidate_holdout).resolve() in ref_paths:
            return report("ERROR", 2, error="Candidate and Holdout comparators require distinct evidence paths")
        if baseline_path is None and baseline_id is not None:
            baseline_path = next((refs[name] for name, ref in references.items() if (ref or name) == baseline_id), None)
        if needs_baseline and (not baseline_path or not baseline_id):
            return report("MISSING_HOLDOUT_BASELINE", 1, error="Supply --baseline-holdout for the registered baseline")

    binding = {
        "candidate_id": cand_id, **identities, "manifest_root_sha256": manifest_sha, "rules_sha256": rules_sha,
        "ledger_file": ledger_path.as_posix(), "seal_file": seal_path.as_posix(),
        "required_comparators_verified": req_comps, "reference_ids": references, "baseline_id": baseline_id,
        "item_cluster_key": item_key, "paired_aux_noninferiority": aux_margins,
    }
    if args.phase in ("reserve", "atomic"):
        reservation = {
            **binding, "schema_version": GUARD_VERSION, "status": "RESERVED",
            "reserved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ledger_head_at_reserve_sha256": prior_rows[-1]["entry_sha256"] if prior_rows else "GENESIS",
        }
        try:
            _atomic_create_seal(receipt_path, reservation)
            _atomic_create_seal(seal_path, reservation)
        except FileExistsError:
            return report("HOLDOUT_ALREADY_CONSUMED", 2, error="A competing reservation already consumed this Holdout")
        prior_rows.append(_append_ledger_entry(ledger_path, prior_rows, {
            **binding, "timestamp_utc": reservation["reserved_at_utc"],
            "decision_stage": "holdout", "decision": "HOLDOUT_RESERVED",
            "reservation_seal_sha256": _sha256_file(seal_path),
        }))
        if args.phase == "reserve":
            return report("RESERVED", 0, **{k: v for k, v in reservation.items() if k != "status"})
        seal_data = reservation
    if (
        any(seal_data.get(k) != v for k, v in binding.items())
        or _load_strict_json(receipt_path) != seal_data
        or prior_rows[-1].get("reservation_seal_sha256") != _sha256_file(seal_path)
    ):
        return report("SEAL_BINDING_MISMATCH", 1, error="The manifest, seal, comparator vector, policy or artifact binding changed")

    # A failed or interrupted finalization is terminal, even if no public verdict is emitted.
    _replace_seal(seal_path, {**seal_data, "status": "FINALIZING", "holdout_primary_replicated": False, "holdout_accepted": False})
    comparisons: dict[str, Any] = {}
    holdout_tripwires: list[str] = []
    failure = None
    try:
        if args.eval_cmd:
            proc = subprocess.run(_format_cmd_tokens(args.eval_cmd, {}), capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise ValueError(f"Holdout evaluator failed ({proc.returncode}): {(proc.stderr or '')[-1000:]}")
        post_ok, _ = _verify_manifest_internal(manifest_path, None, args.expected_root_sha256)
        if not post_ok or _sha256_file(rules_path) != rules_sha or _artifact_fingerprints(args.candidate_artifacts) != identities:
            raise ValueError("Immutable policy or finalist changed after Holdout reservation")

        cand_h = _load_strict_json(Path(args.candidate_holdout))
        if cand_h.get("candidate_id", cand_id) != cand_id:
            raise ValueError("Candidate Holdout evidence identity differs from the confirmed finalist")
        baseline_h = _load_strict_json(Path(baseline_path)) if baseline_path else None
        if baseline_h is not None and baseline_id is not None and baseline_h.get("candidate_id") != baseline_id:
            raise ValueError("Holdout baseline evidence identity differs from the confirmed baseline")
        hold_rules = {**rules, "decision_stage": "confirm", "allow_aggregate_only_confirm": False,
                      "paired_aux_noninferiority": aux_margins}
        for name in req_comps:
            path = refs[name]
            ref = _load_strict_json(Path(path))
            if ref.get("candidate_id") != (references[name] or name):
                raise ValueError(f"Holdout comparator identity differs from the confirmed reference: {name}")
            comp_args = argparse.Namespace(candidate=args.candidate_holdout, incumbent=path,
                                           baseline=baseline_path or path, comparator_id=name)
            score = _score_gate(
                comp_args, hold_rules, cand_h, baseline_h if baseline_h is not None else ref, alpha,
                f"holdout:{manifest_sha}:{identities['artifact_semantic_sha256']}", item_key,
                holdout=True, aux_alpha=alpha / len(req_comps), preloaded_reference=ref,
            )
            if not score.get("paired_summary"):
                reasons = "; ".join(score.get("tripwires", []))
                raise ValueError(f"Missing or invalid paired Holdout evidence for {name}: {reasons}")
            holdout_tripwires.extend(score["tripwires"])
            comparisons[name] = {
                **score["paired_summary"], "decision": score["decision"],
                "decision_reason": score["decision_reason"], "primary_replicated": score["primary_replicated"],
            }
    except (OSError, ValueError, KeyError) as exc:
        failure = str(exc)
        holdout_tripwires.append(f"invalid_holdout_evidence:{failure}")

    holdout_tripwires = list(dict.fromkeys(holdout_tripwires))
    holdout_guardrails_passed = not holdout_tripwires
    holdout_primary_replicated = failure is None and len(comparisons) == len(req_comps) and all(v["primary_replicated"] for v in comparisons.values())
    holdout_accepted = holdout_guardrails_passed and holdout_primary_replicated
    final_seal = {
        **seal_data,
        "status": "FINALIZED" if failure is None else "FAILED_DURING_EVAL",
        "finalized_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "primary_metric": rules["primary_metric"], "primary_statistic": rules.get("primary_statistic", "mean"),
        "direction": rules.get("direction", "maximize"), "aggregation_weighting": rules.get("aggregation_weighting", "equal_item"),
        "holdout_guardrails_passed": holdout_guardrails_passed,
        "holdout_primary_replicated": bool(holdout_primary_replicated),
        "holdout_accepted": holdout_accepted,
        "holdout_tripwires": holdout_tripwires,
        "comparisons": comparisons,
        "n_perm": n_perm, "p_floor": 1.0 / (1 + n_perm),
        "paired_aux_alpha_one_sided": alpha / (len(req_comps) * len(aux_margins)) if aux_margins else None,
    }
    _replace_seal(seal_path, final_seal)
    seal_sha = _sha256_file(seal_path)
    ledger_rec = _append_ledger_entry(ledger_path, prior_rows, {
        **binding, "timestamp_utc": final_seal["finalized_at_utc"], "trial_id": f"holdout_{cand_id}",
        "decision_stage": "holdout", "decision": "HOLDOUT_SEALED" if holdout_guardrails_passed else "HOLDOUT_GUARDRAIL_FAILED",
        "holdout_guardrails_passed": holdout_guardrails_passed,
        "holdout_primary_replicated": bool(holdout_primary_replicated),
        "holdout_accepted": holdout_accepted,
        "holdout_tripwires": holdout_tripwires, "primary_metric": rules["primary_metric"],
        "candidate_primary": next(iter(comparisons.values()), {}).get("obs_candidate"),
        "holdout_seal_sha256": seal_sha,
        "comparisons_summary": {name: {
            "obs_reference": v["obs_reference"], "signed_gain": v["signed_gain"],
            "ci95": [v["ci95_low"], v["ci95_high"]], "lcb_one_sided": v["lcb_one_sided"],
        } for name, v in comparisons.items()},
    })
    return report(
        final_seal["status"], 0 if failure is None else 1, seal_file=seal_path.as_posix(),
        holdout_seal_sha256=seal_sha, ledger_entry_sha256=ledger_rec["entry_sha256"],
        holdout_guardrails_passed=holdout_guardrails_passed,
        holdout_primary_replicated=bool(holdout_primary_replicated),
        holdout_accepted=holdout_accepted,
        n_perm=n_perm, p_floor=1.0 / (1 + n_perm),
        paired_aux_alpha_one_sided=final_seal["paired_aux_alpha_one_sided"],
        holdout_tripwires=holdout_tripwires, comparisons=comparisons,
    )


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Hardened Executable Governance Guard for Autonomous Optimization Loops (v{GUARD_VERSION})")
    p.add_argument("--version", action="version", version=GUARD_VERSION)
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

    p_leak = sub.add_parser("audit-leaks", help="Static AST, allowlist, binary/hook, test-skip & Lean proof scanner")
    p_leak.add_argument("--paths", nargs="+", required=True, help="Files, diffs, or directories to scan")
    p_leak.add_argument("--locked-constant", action="append", default=[], help="Locked KEY=VALUE protocol constant")
    p_leak.add_argument("--allowlist", nargs="*", default=[], help="Allowed relative paths/globs from program.md mutable surface")
    p_leak.add_argument("--base-dir", default=None, help="Clean reference directory to enforce allowlist against candidate dir")
    p_leak.add_argument("--fail-on-warn", action="store_true", help="Treat WARN findings as blocking")
    p_leak.set_defaults(func=cmd_audit_leaks)

    p_splits = sub.add_parser("audit-splits", help="Verify zero duplicate feature rows, group overlap & time lookahead across splits")
    p_splits.add_argument("--train", required=True, help="Train/Search split file (.json, .jsonl, or .csv)")
    p_splits.add_argument("--val", default=None, help="Optional validation split file (.json, .jsonl, or .csv)")
    p_splits.add_argument("--confirm", required=True, help="Confirmation split file (.json, .jsonl, or .csv)")
    p_splits.add_argument("--holdout", default=None, help="Optional Holdout split file (.json, .jsonl, or .csv)")
    p_splits.add_argument("--feature-keys", nargs="*", default=[], help="Explicit feature column names to hash")
    p_splits.add_argument("--exclude-keys", nargs="*", default=["label", "target", "y"], help="Label/metadata keys excluded from feature hash")
    p_splits.add_argument("--group-key", default=None, help="Optional entity/user/problem ID column that must be disjoint across splits")
    p_splits.add_argument("--require-disjoint-users", action="store_true", help="Require complete user IDs and zero intersections")
    p_splits.add_argument("--user-partition", default="NONE", help="Any declared partition other than NONE requires disjoint users")
    p_splits.add_argument("--user-key", default="user_id", help="User ID column for intersection and M/U diagnostics")
    p_splits.add_argument("--item-key", default="item_id", help="Item ID column for M/U diagnostics")
    p_splits.add_argument("--raw-counts", help="JSON file with per-split n_users and n_items before preprocessing")
    p_splits.add_argument("--timestamp-key", default=None, help="Optional numeric timestamp column enforcing train < val < confirm < holdout")
    p_splits.add_argument("--embargo-gap", type=float, default=0.0, help="Minimum timestamp gap required between consecutive splits")
    p_splits.set_defaults(func=cmd_audit_splits)

    p_iso = sub.add_parser("run-isolated", help="Run candidate in scrubbed out-of-process predict/causal/build/train sandbox")
    p_iso.add_argument(
        "--mode",
        choices=["predict", "causal_online", "build_test_bench", "train_lock_score"],
        default="predict",
    )
    p_iso.add_argument("--candidate-script", default=None, help="Path to candidate Python worker (predict / causal_online)")
    p_iso.add_argument("--input-json", default=None, help="Features-only input JSON or streams JSON (predict / causal_online)")
    p_iso.add_argument("--output-json", required=True, help="Validated output JSON envelope")
    p_iso.add_argument("--workspace-root", default=None, help="Repo root to snapshot without .git or forbidden paths")
    p_iso.add_argument("--base-dir", default=None, help="Optional clean base directory for pre-run allowlist check")
    p_iso.add_argument("--allowlist", nargs="*", default=[], help="Optional mutable allowlist patterns")
    p_iso.add_argument("--expected-ids-json", default=None, help="Optional JSON list of required prediction IDs")
    p_iso.add_argument("--forbid-paths", nargs="*", default=[], help="Protected label/holdout directories forbidden to candidate")
    p_iso.add_argument("--build-cmd", default=None, help="Build command inside clean snapshot (build_test_bench)")
    p_iso.add_argument("--artifact-glob", default=None, help="Built binary/library glob to pin by SHA-256 (build_test_bench)")
    p_iso.add_argument("--test-cmd", default=None, help="Test command inside clean snapshot (build_test_bench)")
    p_iso.add_argument("--trusted-tests-dir", default=None, help="Immutable test directory overlaid before running --test-cmd")
    p_iso.add_argument("--trusted-tests-relpath", default=None, help="Relative destination path for --trusted-tests-dir")
    p_iso.add_argument("--min-tests-executed", type=int, default=0, help="Minimum executed test count required to block silent test skipping")
    p_iso.add_argument("--bench-cmd", default=None, help="Benchmark command writing --output-json / {output_json} (build_test_bench)")
    p_iso.add_argument("--train-cmd", default=None, help="Training command inside Confirm/Holdout-free snapshot (train_lock_score)")
    p_iso.add_argument("--checkpoint-relpath", default=None, help="Relative checkpoint file/dir produced by training (train_lock_score)")
    p_iso.add_argument("--score-cmd", default=None, help="Immutable scorer command using {checkpoint_path} & {output_json} (train_lock_score)")
    p_iso.add_argument("--allow-unsafe-pickle-checkpoints", action="store_true", help="Allow .pt/.pkl checkpoints (disabled by default)")
    p_iso.add_argument("--timeout-sec", type=float, default=120.0, help="Hard wall-clock timeout in seconds")
    p_iso.add_argument("--python-exe", default=None, help="Optional Python interpreter override")
    p_iso.add_argument("--skip-static-audit", action="store_true", help="Skip AST leak check before execution")
    p_iso.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra CLI args forwarded to candidate")
    p_iso.set_defaults(func=cmd_run_isolated)

    p_sel = sub.add_parser("select-search-winner", help="Order-invariant batch Search selector with deterministic tie-break")
    p_sel.add_argument("--candidates", nargs="+", required=True, help="Candidate Search result JSON files")
    p_sel.add_argument("--incumbent", required=True, help="Frozen round-start incumbent JSON")
    p_sel.add_argument("--baseline", required=True, help="Initial B_0 / M_0 baseline JSON")
    p_sel.add_argument("--rules", required=True, help="Search gate_rules.json")
    p_sel.add_argument("--round-id", default="search_round_01", help="Batch round identifier")
    p_sel.add_argument("--rank-by", choices=["auto", "lcb", "signed_gain"], default="auto", help="Ranking criterion across eligible candidates")
    p_sel.add_argument("--ledger", default=None, help="Optional trial_ledger.jsonl path")
    p_sel.add_argument("--item-cluster-key", help="Item-block column with user blocks in cluster_id")
    p_sel.set_defaults(func=cmd_select_search_winner)

    p_gate = sub.add_parser("gate-decision", aliases=["gate-confirm"], help="Run search/confirm/frontier gate with paired bootstrap & ledger")
    p_gate.add_argument("--candidate", required=True, help="Candidate metrics JSON (with keyed items)")
    p_gate.add_argument("--incumbent", required=True, help="Incumbent metrics JSON (re-measured or paired)")
    p_gate.add_argument("--baseline", required=True, help="Initial B_0 / M_0 metrics JSON")
    p_gate.add_argument("--rules", required=True, help="Gate, direction & tripwire rules JSON")
    p_gate.add_argument("--ledger", default=None, help="Optional hash-chained trial_ledger.jsonl path")
    p_gate.add_argument("--trial-id", default="trial_01", help="Trial or round identifier")
    p_gate.add_argument("--comparator-id", default=None, help="Optional comparator name when confirming against multiple baselines")
    p_gate.add_argument("--candidate-artifacts", nargs="*", default=None, help="Candidate code/weights/config paths to bind by SHA-256")
    p_gate.add_argument("--item-cluster-key", help="Item-block column with user blocks in cluster_id")
    p_gate.add_argument("--alpha-schedule", nargs="+", help="Predeclare positive Confirm alpha allocations (space- or comma-separated)")
    p_gate.add_argument("--allow-registered-retest", action="store_true", help="Request a new registered look after INCONCLUSIVE_RETEST")
    p_gate.add_argument("--aux-noninferiority-margins", nargs="+", help="Registered auxiliary tolerances NAME=DROP (or a JSON object)")
    p_gate.add_argument("--proposer-out", default=None, help="Optional path for redacted proposer-safe verdict JSON")
    p_gate.add_argument("--hypothesis", default="", help="Falsifiable hypothesis tested by this trial")
    p_gate.set_defaults(func=cmd_gate_decision)

    p_iut = sub.add_parser(
        "gate-confirm-iut",
        help="Batch M-comparator Intersection-Union Confirmation with locked hash-chained ledger appends",
    )
    p_iut.add_argument("--candidate", required=True, help="Candidate Confirm metrics JSON / .json.gz")
    p_iut.add_argument("--baseline", required=True, help="Initial B_0 / M_0 Confirm metrics JSON / .json.gz")
    p_iut.add_argument("--comparator", action="append", required=True, help="NAME=PATH Confirm JSON for each comparator")
    p_iut.add_argument("--rules", required=True, help="Confirm gate_rules.json")
    p_iut.add_argument("--ledger", default=None, help="Optional hash-chained trial_ledger.jsonl path")
    p_iut.add_argument("--trial-prefix", default="confirm_iut", help="Trial ID prefix (each row becomes <prefix>_<comp>)")
    p_iut.add_argument("--candidate-artifacts", nargs="*", default=None, help="Candidate code/weights/config paths to bind by SHA-256")
    p_iut.add_argument("--item-cluster-key", help="Item-block column with user blocks in cluster_id")
    p_iut.add_argument("--alpha-schedule", nargs="+", help="Predeclare positive Confirm alpha allocations (space- or comma-separated)")
    p_iut.add_argument("--allow-registered-retest", action="store_true", help="Request a new registered look after INCONCLUSIVE_RETEST")
    p_iut.add_argument("--aux-noninferiority-margins", nargs="+", help="Registered auxiliary tolerances NAME=DROP (or a JSON object)")
    p_iut.add_argument("--proposer-out", default=None, help="Optional path for redacted proposer-safe IUT summary JSON")
    p_iut.add_argument("--hypothesis", default="", help="Falsifiable hypothesis tested by this confirmation vector")
    p_iut.set_defaults(func=cmd_gate_confirm_iut)

    p_hold = sub.add_parser("gate-holdout", help="Atomic write-once single-touch Holdout seal & evaluation")
    p_hold.add_argument("--phase", choices=["reserve", "finalize", "atomic"], default="atomic")
    p_hold.add_argument("--seal-file", required=True, help="Write-once HOLDOUT_SEAL.json path")
    p_hold.add_argument("--manifest", required=True, help="Path to IMMUTABLE_MANIFEST.json")
    p_hold.add_argument("--expected-root-sha256", required=True, help="Externally pinned manifest root SHA-256")
    p_hold.add_argument("--ledger", required=True, help="Path to trial_ledger.jsonl")
    p_hold.add_argument("--rules", required=True, help="Path to confirm/holdout gate_rules.json")
    p_hold.add_argument("--candidate-id", required=True, help="Confirmed finalist candidate_id")
    p_hold.add_argument("--candidate-artifacts", nargs="+", required=True, help="Candidate files/weights/configs bound to promotion")
    p_hold.add_argument("--required-comparators", nargs="*", default=None, help="Mandatory comparators that must all have PROMOTE")
    p_hold.add_argument("--item-cluster-key", help="Item-block column with user blocks in cluster_id; must match Confirm")
    p_hold.add_argument("--baseline-holdout", help="Holdout JSON for the pinned baseline, if not also a named comparator")
    p_hold.add_argument("--aux-noninferiority-margins", nargs="+", help="Additional or stricter Holdout tolerances NAME=DROP (or a JSON object)")
    p_hold.add_argument("--eval-cmd", default=None, help="Optional shell command to generate holdout JSONs after reservation")
    p_hold.add_argument("--candidate-holdout", default=None, help="Candidate holdout metrics JSON")
    p_hold.add_argument("--comparator-holdout", action="append", default=[], help="NAME=PATH holdout JSON for each comparator")
    p_hold.set_defaults(func=cmd_gate_holdout)

    p_vled = sub.add_parser("verify-ledger", help="Verify SHA-256 hash chain of trial_ledger.jsonl")
    p_vled.add_argument("--ledger", required=True, help="Path to trial_ledger.jsonl")
    p_vled.add_argument("--expected-head-sha256", default=None, help="Optional externally pinned ledger head SHA-256")
    p_vled.set_defaults(func=cmd_verify_ledger)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd in {"run-isolated", "audit-leaks", "freeze-manifest", "verify-manifest", "verify-ledger"}:
        return int(args.func(args))
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())

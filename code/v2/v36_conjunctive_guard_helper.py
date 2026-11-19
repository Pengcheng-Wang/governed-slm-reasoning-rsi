#!/usr/bin/env python3
"""
Conjunctive Guard v3.6 Governance, Full-Shard Weight Hashing, Local-NVMe Seal &
Stratified Rademacher Sign-Flip Permutation Engine (`v36_conjunctive_guard_helper.py`).

Implements pre-Holdout governance and statistical safeguards from external frontier-model reviews:
  1. Full-checkpoint canonical weight manifest (`canonical_checkpoint_manifest`):
     lstat-checks every file in `ckpt_dir` (rejecting symlinks and verifying exact
     `model.safetensors.index.json` shard coverage) and SHA-256 hashes every `*.safetensors`
     shard + config/tokenizer file.
  2. Timestamped hash-linked `AMENDMENT_V3_6.json` (`write_v36_amendment`) pinned on local
     NVMe before any Protected Holdout access (`v3.6` can ONLY veto a `v2` PROMOTE, never rescue).
  3. Exact Paired McNemar / Binomial Test for sparse discrete formal theorem outcomes and
     20,000-draw Stratified Rademacher Sign-Flip Permutation Test (`stratified_sign_flip_test`)
     for unequal-sized 4-contest Holdout strata (`100 / 40 / 30 / 30`).
  4. Comparator-first SHA-256 pinning (`COMPARATOR_PIN_*.json`) + `governed_loop_guard_v3.py`
     `gate-decision` (`--proposer-out` redaction) -> `gate-holdout --phase reserve`
     (`O_CREAT | O_EXCL` + `os.fsync` on local NVMe) -> `gate-holdout --phase finalize`.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fsync_file_and_dir(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())
    if os.name != "nt":
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def canonical_checkpoint_manifest(ckpt_dir: Path, out_manifest_path: Path | None = None) -> dict[str, Any]:
    """Hash every regular file in `ckpt_dir` (all *.safetensors shards, configs, tokenizers).
    Fails closed on symlinks, missing files, or unlisted *.safetensors shards vs index.json.
    """
    ckpt_dir = ckpt_dir.resolve()
    if not ckpt_dir.is_dir():
        raise RuntimeError(f"Checkpoint directory missing: {ckpt_dir}")

    entries: dict[str, dict[str, Any]] = {}
    found_safetensors: set[str] = set()

    for root_s, dirs, files in os.walk(str(ckpt_dir), followlinks=False):
        dirs.sort()
        files.sort()
        root_p = Path(root_s)
        for dname in dirs:
            dp = root_p / dname
            if os.path.islink(dp):
                raise RuntimeError(f"Symlink directory forbidden inside checkpoint: {dp}")
        for fname in files:
            fp = root_p / fname
            st = os.lstat(fp)
            if os.path.islink(fp):
                raise RuntimeError(f"Symlink file forbidden inside checkpoint: {fp}")
            rel = fp.relative_to(ckpt_dir).as_posix()
            if rel.endswith(".safetensors"):
                found_safetensors.add(rel)
            entries[rel] = {
                "size_bytes": int(st.st_size),
                "sha256": sha256_file(fp),
            }

    if not found_safetensors:
        raise RuntimeError(f"No *.safetensors weight files found in {ckpt_dir}")

    idx_p = ckpt_dir / "model.safetensors.index.json"
    if idx_p.is_file():
        idx_data = json.loads(idx_p.read_text(encoding="utf-8"))
        wmap = idx_data.get("weight_map", {})
        expected_shards = {str(v) for v in wmap.values()}
        if expected_shards != found_safetensors:
            raise RuntimeError(
                f"Safetensors shard mismatch vs index.json in {ckpt_dir}: "
                f"index={sorted(expected_shards)} disk={sorted(found_safetensors)}"
            )

    canonical_blob = json.dumps({"entries": entries}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    root_digest = hashlib.sha256(canonical_blob).hexdigest()
    payload = {
        "ckpt_dir": ckpt_dir.as_posix(),
        "file_count": len(entries),
        "safetensors_shard_count": len(found_safetensors),
        "weights_root_sha256": root_digest,
        "entries": entries,
    }
    if out_manifest_path is not None:
        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        out_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fsync_file_and_dir(out_manifest_path)
    return payload


def exact_sign_test_one_sided(diffs: list[float], margin: float = 0.0) -> dict[str, float]:
    """Exact binomial sign test on non-zero paired differences d_i - margin."""
    shifted = [d - margin for d in diffs]
    n_pos = sum(1 for x in shifted if x > 1e-12)
    n_neg = sum(1 for x in shifted if x < -1e-12)
    n_disc = n_pos + n_neg
    if n_disc == 0:
        return {"n_pos": 0, "n_neg": 0, "n_discordant": 0, "p_exact_one_sided": 1.0}
    # P(Binom(n_disc, 0.5) >= n_pos) in log-space safe sum
    tail = 0.0
    for k in range(n_pos, n_disc + 1):
        tail += math.comb(n_disc, k) * (0.5 ** n_disc)
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_discordant": n_disc,
        "p_exact_one_sided": min(1.0, max(0.0, tail)),
    }


def stratified_sign_flip_test(
    cand_by_stratum: dict[str, dict[str, float]],
    ref_by_stratum: dict[str, dict[str, float]],
    *,
    expected_counts: dict[str, int] | None = None,
    margin: float = 0.0,
    n_perm: int = 20000,
    n_boot: int = 10000,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Exact-size stratified Rademacher sign-flip permutation test + within-stratum cluster bootstrap
    for both item-pooled (`N=200`) and equal-stratum (`4 x 1/4`) paired Holdout comparisons.
    """
    if set(cand_by_stratum.keys()) != set(ref_by_stratum.keys()):
        raise ValueError(f"Stratum mismatch: {sorted(cand_by_stratum)} vs {sorted(ref_by_stratum)}")
    strata = sorted(cand_by_stratum.keys())
    if expected_counts is not None:
        actual_counts = {s: len(cand_by_stratum[s]) for s in strata}
        if actual_counts != expected_counts:
            raise ValueError(f"Holdout stratum counts {actual_counts} != expected {expected_counts}")

    diffs_by_s: dict[str, list[float]] = {}
    all_diffs: list[float] = []
    stratum_obs: dict[str, dict[str, float]] = {}

    for s in strata:
        ca = cand_by_stratum[s]
        rf = ref_by_stratum[s]
        if set(ca.keys()) != set(rf.keys()):
            raise ValueError(f"Item ID mismatch in stratum {s}: {len(ca)} vs {len(rf)}")
        keys = sorted(ca.keys())
        ds = [float(ca[k]) - float(rf[k]) for k in keys]
        diffs_by_s[s] = ds
        all_diffs.extend(ds)
        ns = len(ds)
        stratum_obs[s] = {
            "n": ns,
            "cand_mean": sum(ca.values()) / max(ns, 1),
            "ref_mean": sum(rf.values()) / max(ns, 1),
            "delta": sum(ds) / max(ns, 1),
        }

    n_total = len(all_diffs)
    n_strata = len(strata)
    obs_pooled = sum(all_diffs) / max(n_total, 1)
    obs_equal_stratum = sum(stratum_obs[s]["delta"] for s in strata) / max(n_strata, 1)

    rng = random.Random(seed)
    ge_pooled = 1  # Phipson-Smyth (b + 1) / (M + 1) exact Monte Carlo validity
    ge_eq = 1
    obs_p_adj = obs_pooled - margin
    obs_eq_adj = obs_equal_stratum - margin

    shifted_by_s = {s: [d - margin for d in diffs_by_s[s]] for s in strata}
    for _ in range(n_perm):
        tot_s = 0.0
        eq_s = 0.0
        for s in strata:
            arr = shifted_by_s[s]
            ns = len(arr)
            # Independent Rademacher sign flips within each stratum
            ssum = 0.0
            bits = rng.getrandbits(ns)
            for idx, val in enumerate(arr):
                if (bits >> idx) & 1:
                    ssum += val
                else:
                    ssum -= val
            tot_s += ssum
            eq_s += ssum / ns
        if (tot_s / n_total) >= obs_p_adj - 1e-15:
            ge_pooled += 1
        if (eq_s / n_strata) >= obs_eq_adj - 1e-15:
            ge_eq += 1

    p_perm_pooled = ge_pooled / float(n_perm + 1)
    p_perm_equal_stratum = ge_eq / float(n_perm + 1)

    # Stratified cluster bootstrap (resampling paired items within each contest stratum)
    boot_pooled: list[float] = []
    boot_eq: list[float] = []
    for _ in range(n_boot):
        tot_b = 0.0
        eq_b = 0.0
        for s in strata:
            arr = diffs_by_s[s]
            ns = len(arr)
            bsum = sum(arr[rng.randrange(ns)] for _ in range(ns))
            tot_b += bsum
            eq_b += bsum / ns
        boot_pooled.append(tot_b / n_total)
        boot_eq.append(eq_b / n_strata)

    boot_pooled.sort()
    boot_eq.sort()
    se_pooled = math.sqrt(sum((x - obs_pooled) ** 2 for x in boot_pooled) / max(n_boot - 1, 1))
    se_eq = math.sqrt(sum((x - obs_equal_stratum) ** 2 for x in boot_eq) / max(n_boot - 1, 1))

    return {
        "n_total": n_total,
        "n_strata": n_strata,
        "per_stratum_descriptive": stratum_obs,
        "item_pooled": {
            "cand_mean": sum(stratum_obs[s]["cand_mean"] * stratum_obs[s]["n"] for s in strata) / n_total,
            "ref_mean": sum(stratum_obs[s]["ref_mean"] * stratum_obs[s]["n"] for s in strata) / n_total,
            "delta": obs_pooled,
            "se_stratified_boot": se_pooled,
            "ci95_lo": boot_pooled[int(0.025 * n_boot)],
            "ci95_hi": boot_pooled[int(0.975 * n_boot)],
            "p_sign_flip_one_sided": p_perm_pooled,
        },
        "equal_stratum": {
            "cand_mean": sum(stratum_obs[s]["cand_mean"] for s in strata) / n_strata,
            "ref_mean": sum(stratum_obs[s]["ref_mean"] for s in strata) / n_strata,
            "delta": obs_equal_stratum,
            "se_stratified_boot": se_eq,
            "ci95_lo": boot_eq[int(0.025 * n_boot)],
            "ci95_hi": boot_eq[int(0.975 * n_boot)],
            "p_sign_flip_one_sided": p_perm_equal_stratum,
        },
    }


def build_combined_4contest_metrics_json(
    contest_metrics_paths: dict[str, Path],
    out_path: Path,
    *,
    candidate_id: str,
    expected_counts: dict[str, int],
) -> Path:
    """Combine the 4 contest `*_metrics.json` files into a single `governed_loop_guard_v3.py`-compatible
    Holdout JSON with `stratum_id`, `stratum_metrics`, and exact 100/40/30/30 count verification.
    """
    items_out: list[dict[str, Any]] = []
    stratum_metrics: dict[str, dict[str, float]] = {}
    actual_counts: dict[str, int] = {}

    for bench in sorted(expected_counts.keys()):
        mp = contest_metrics_paths[bench]
        d = json.loads(mp.read_text(encoding="utf-8"))
        raw_items = d["items"]
        actual_counts[bench] = len(raw_items)
        stratum_metrics[bench] = {
            "math_avg_at_8": float(d["metrics"]["math_avg_at_8"]),
            "math_pass_at_8": float(d["metrics"]["math_pass_at_8"]),
            "stop_rate": float(d["metrics"]["stop_rate"]),
            "natural_stop_parse_failure_rate": float(d["metrics"].get("natural_stop_parse_failure_rate", 0.0)),
        }
        for it in raw_items:
            uid = f"{bench}::{it['id']}"
            items_out.append(
                {
                    "id": uid,
                    "cluster_id": uid,
                    "stratum_id": bench,
                    "value": float(it["value"]),
                }
            )

    if actual_counts != expected_counts:
        raise RuntimeError(f"Contest item count mismatch for {candidate_id}: {actual_counts} != {expected_counts}")

    n_tot = len(items_out)
    pooled_avg = sum(x["value"] for x in items_out) / float(n_tot)
    pooled_stop = sum(stratum_metrics[b]["stop_rate"] * actual_counts[b] for b in actual_counts) / float(n_tot)
    pooled_parse_fail = sum(
        stratum_metrics[b]["natural_stop_parse_failure_rate"] * actual_counts[b] for b in actual_counts
    ) / float(n_tot)
    equal_avg = sum(stratum_metrics[b]["math_avg_at_8"] for b in actual_counts) / float(len(actual_counts))

    first_d = json.loads(next(iter(contest_metrics_paths.values())).read_text(encoding="utf-8"))
    out_obj = {
        "schema_version": "3.0",
        "candidate_id": candidate_id,
        "env_fingerprint": first_d.get("env_fingerprint", {}),
        "protocol_fingerprint": {
            "track": "math",
            "protocol_id": "v2_math_holdout_4contest_suite",
            "expected_k": 8,
            "strata_counts": actual_counts,
        },
        "metrics": {
            "math_avg_at_8": pooled_avg,
            "math_equal_stratum_avg_at_8": equal_avg,
            "stop_rate": pooled_stop,
            "natural_stop_parse_failure_rate": pooled_parse_fail,
        },
        "stratum_metrics": stratum_metrics,
        "items": items_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fsync_file_and_dir(out_path)
    return out_path


def write_v36_amendment(
    v2_root: Path,
    v1_root: Path,
    *,
    manifest_v2_sha256: str,
    guard_v36_path: Path,
) -> tuple[Path, str]:
    """Write and fsync timestamped mid-campaign Conjunctive v3.6 Governance Amendment on local NVMe
    before any Protected Holdout evaluation occurs.
    """
    gov_dir = v2_root / "results/v3_governance"
    gov_dir.mkdir(parents=True, exist_ok=True)
    amend_p = gov_dir / "AMENDMENT_V3_6.json"
    if amend_p.is_file():
        return amend_p, sha256_file(amend_p)

    holdout_splits = {
        "lean_proofnet_test": v2_root / "eval_v2/sealed_holdout/v2_lean_holdout_proofnet174.jsonl",
        "math_beyondaime": v2_root / "eval_v2/sealed_holdout/v2_math_holdout_beyondaime.jsonl",
        "math_cmimc2025": v2_root / "eval_v2/sealed_holdout/v2_math_holdout_cmimc2025.jsonl",
        "math_brumo2025": v2_root / "eval_v2/sealed_holdout/v2_math_holdout_brumo2025.jsonl",
        "math_hmmt_nov2025": v2_root / "eval_v2/sealed_holdout/v2_math_holdout_hmmt_nov2025.jsonl",
        "math_retention_math500_r150": v1_root / "eval/sealed_holdout/math_holdout_retention150.jsonl",
    }
    split_digests = {k: sha256_file(p) for k, p in holdout_splits.items() if p.is_file()}

    payload = {
        "amendment_id": "AMENDMENT_V3_6_CONJUNCTIVE_DUAL_LEDGER",
        "status": "MID_CAMPAIGN_AMENDMENT_PRE_HOLDOUT",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "anchored_v2_manifest_root_sha256": manifest_v2_sha256,
        "guard_v36_script_sha256": sha256_file(guard_v36_path),
        "sealed_holdout_split_sha256": split_digests,
        "conjunctive_law": {
            "rule": "v3.6 can ONLY veto a v2 Confirmation PROMOTE, never rescue a v2 ROLLBACK.",
            "holm_bonferroni_indexing": "0-based step-down alpha / (K - rank_0) across ALL K tested challengers in track, stopping at first non-rejection.",
            "unpromoted_holdout_policy": "Any challenger failing Confirmation retains B_0 and is NEVER evaluated on Protected Holdout.",
            "holdout_math_strata_counts": {
                "beyondaime": 100,
                "cmimc2025": 40,
                "brumo2025": 30,
                "hmmt_nov2025": 30,
            },
            "holdout_conjunctive_inference": (
                "Both item-pooled (N=200) and equal_stratum (4 x 1/4) stratified Rademacher sign-flip "
                "permutation p_one_sided must pass Holm-Bonferroni across all promoted models in the track."
            ),
            "math500_retention_role": "Evaluated outside the 4-contest average (N=150, mean@8 per item) as a non-degradation retention check.",
            "comparator_first_pinning": (
                "Zero-shot comparators (B_0, Qwen3.8-27B, Goedel-Prover-V2-8B) are evaluated and SHA-256 "
                "pinned BEFORE reserving HOLDOUT_SEAL_*.json (O_CREAT | O_EXCL + fsync on local NVMe) or "
                "evaluating any promoted A* candidate."
            ),
        },
    }
    amend_p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fsync_file_and_dir(amend_p)
    return amend_p, sha256_file(amend_p)

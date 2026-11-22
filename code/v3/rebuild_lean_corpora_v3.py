#!/usr/bin/env python3
"""
Builds full `#print axioms`-verified Track L Corpora (`Stage1_SFT ~ 3,800`, `Stage2_Pool ~ 1,400`,
`RL_Pool = 500`) from `Goedel-LM/SFT_dataset_v2` (extracting the final complete ```lean4 proof block
after `### Complete Lean 4 Proof`), standalone `Mathlib4` multi-step proofs, and 4-layer-decontaminated
verified `Lean-Workbook` + compiler-repair proofs, then runs `governed_loop_guard_v37.py audit-splits`.
"""
from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

V1_ROOT = Path("/scratch/governed_slm_rsi_v1")
V2_ROOT = Path("/scratch/governed_slm_rsi_v2")
V3_ROOT = Path("/scratch/governed_slm_rsi_v3")
CODE_V3_DIR = V3_ROOT / "code/v3"
EVAL_V3_DIR = V3_ROOT / "eval_v3"
for p in (CODE_V3_DIR, EVAL_V3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lean4_verifier import split_frozen_statement  # noqa: E402
from prepare_v3_splits_and_corpora import (  # noqa: E402
    _classify_lean_stratum,
    _det_hash,
    _norm_lean_stmt_sha,
    _split_lean_hdr_and_decl,
    _verify_one_goedel_or_mathlib_sft,
    _write_jsonl,
    collect_prior_v1_v2_blocklist,
)


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    splits_dir = V3_ROOT / "data/splits_v3"
    holdout_dir = V3_ROOT / "eval_v3/sealed_holdout"
    corpora_dir = V3_ROOT / "data/corpora_v3"

    decon, _, _ = collect_prior_v1_v2_blocklist()
    for ep in [
        splits_dir / "v3_lean_search140.jsonl",
        splits_dir / "v3_lean_confirm180.jsonl",
        holdout_dir / "v3_lean_holdout.jsonl",
    ]:
        for r in _read_jsonl(ep):
            fs = str(r.get("formal_statement", ""))
            if fs:
                decon.register_lean_eval(str(r.get("id", "")), fs, str(r.get("informal_statement", "")))

    # Keep any already-verified standalone Mathlib4 proofs
    existing_s1 = _read_jsonl(corpora_dir / "v3_lean_stage1_sft.jsonl")
    seen_shas = {_norm_lean_stmt_sha(str(r.get("formal_statement", ""))) for r in existing_s1 if r.get("formal_statement")}

    # 1. Sample diverse candidates from `Goedel-LM/SFT_dataset_v2`, extracting ONLY the LAST ```lean4 block!
    sft_goedel = load_dataset("Goedel-LM/SFT_dataset_v2", split="train")
    stride = max(1, len(sft_goedel) // 9000)
    workers = min(72, os.cpu_count() or 32)

    def _collect_goedel_wave(start_offset: int, max_tasks: int, cache_fn: str) -> list[dict[str, Any]]:
        cache_p = corpora_dir / cache_fn
        if cache_p.exists():
            cached_rows = _read_jsonl(cache_p)
            for cr in cached_rows:
                seen_shas.add(_norm_lean_stmt_sha(str(cr.get("formal_statement", ""))))
            print(f"Loaded {len(cached_rows)} cached verified proofs from {cache_p}", flush=True)
            return cached_rows
        wave_tasks: list[tuple[str, str, str, str, str]] = []
        for idx in range(start_offset, len(sft_goedel), stride):
            if len(wave_tasks) >= max_tasks:
                break
            msgs = sft_goedel[idx]["messages"]
            if isinstance(msgs, str):
                try:
                    msgs = ast.literal_eval(msgs)
                except Exception:
                    continue
            if not msgs or len(msgs) < 2:
                continue
            u_txt = str(msgs[0].get("content", ""))
            a_txt = str(msgs[-1].get("content", ""))
            u_blocks = re.findall(r"```(?:lean4|lean)?\s*\n(.*?)```", u_txt, flags=re.DOTALL)
            a_blocks = re.findall(r"```(?:lean4|lean)?\s*\n(.*?)```", a_txt, flags=re.DOTALL)
            if not u_blocks or not a_blocks:
                continue
            hdr, decl = _split_lean_hdr_and_decl(u_blocks[-1])
            try:
                _, decl_prefix = split_frozen_statement(decl)
            except Exception:
                continue
            stmt = decl_prefix.strip() + " := by"
            if decon.is_lean_contaminated(stmt):
                continue
            lsha = _norm_lean_stmt_sha(stmt)
            if lsha in seen_shas:
                continue
            final_code_block = a_blocks[-1].strip()
            if "sorry" in final_code_block or len(final_code_block) > 4500:
                continue
            seen_shas.add(lsha)
            wrapped_single = f"```lean4\n{final_code_block}\n```"
            wave_tasks.append((f"goedel_sft_v2:{idx}", hdr, stmt, wrapped_single, "goedel_sft_v2"))

        print(f"Verifying {len(wave_tasks)} single-block Goedel-SFT-v2 proofs (offset={start_offset}, workers={workers}) via Lean 4.24 `#print axioms`...", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            res_list = list(ex.map(_verify_one_goedel_or_mathlib_sft, wave_tasks, chunksize=8))
        verified = [x for x in res_list if x is not None]
        _write_jsonl(cache_p, verified)
        print(f"Verified single-block Goedel-SFT-v2 proofs (offset={start_offset}): {len(verified)} / {len(wave_tasks)}", flush=True)
        return verified

    verified_goedel = _collect_goedel_wave(0, 4200, ".cached_goedel_verified_w1.jsonl")

    # 2. Also gather already-#print-axioms-verified decontaminated proofs from V1/V2 banks
    verified_wb: list[dict[str, Any]] = []
    for bank_p in [
        V1_ROOT / "data/verified_banks/lean_r2_cumulative_verified.jsonl",
        V2_ROOT / "data/verified_banks_v2/lean_qwen35_9b_r2_direct.jsonl",
        V2_ROOT / "data/verified_banks_v2/lean_qwen3_4b_r2_direct.jsonl",
    ]:
        for r in _read_jsonl(bank_p):
            fs = str(r.get("formal_statement", ""))
            msgs = r.get("messages", [])
            if not fs or not msgs or len(msgs) < 2:
                continue
            if decon.is_lean_contaminated(fs):
                continue
            lsha = _norm_lean_stmt_sha(fs)
            if lsha in seen_shas:
                continue
            asst = str(msgs[-1].get("content", ""))
            if "<think>" in asst or "</think>" in asst or len(asst) > 4500:
                continue
            seen_shas.add(lsha)
            # Extract proof body inside ```lean4
            m_b = re.findall(r"```(?:lean4|lean)?\s*\n(.*?)```", asst, flags=re.DOTALL)
            pbody = m_b[-1].strip() if m_b else asst.strip()
            rid = f"verified_wb:{len(verified_wb)}"
            verified_wb.append(
                {
                    "id": rid,
                    "problem_id": rid,
                    "source": "lean_workbook_verified",
                    "lean_header": str(r.get("lean_header", "import Mathlib\nimport Aesop\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n")),
                    "formal_statement": fs,
                    "proof_body": pbody,
                    "finish_reason": "stop",
                    "messages": msgs,
                }
            )

    all_clean_lean = verified_goedel + verified_wb
    if len(all_clean_lean) < 3400:
        # Clean-cluster standalone fallback when optional V1/V2 banks are absent: verify a second disjoint Goedel-SFT-v2 wave
        w2 = _collect_goedel_wave(max(1, stride // 2), 4400, ".cached_goedel_verified_w2.jsonl")
        all_clean_lean = verified_goedel + verified_wb + w2
    all_clean_lean.sort(key=lambda x: _det_hash("v3_lean_all:" + x["id"]))
    print(f"Total combined `#print axioms`-verified disjoint Lean proofs: {len(all_clean_lean) + len(existing_s1)}", flush=True)
    if len(all_clean_lean) < 1800:
        raise RuntimeError(f"Expected >= 1800 verified Lean items, got {len(all_clean_lean)}")

    rl_src = all_clean_lean[:500]
    s2_src = all_clean_lean[500:1500]
    s1_src = existing_s1 + all_clean_lean[1500:]

    lean_rl_pool500: list[dict[str, Any]] = []
    for x in rl_src:
        strat = _classify_lean_stratum("", x["formal_statement"])
        ref_p = x["messages"][-1]["content"]
        lean_rl_pool500.append(
            {
                "id": x["id"],
                "cluster_id": x["id"],
                "stratum": strat,
                "source": x["source"],
                "lean_header": x["lean_header"],
                "formal_statement": x["formal_statement"],
                "reference_proof": ref_p,
                "messages": [x["messages"][0]],
            }
        )

    # Include decontaminated compiler-repair pairs in Stage-II Pool alongside `s2_src`
    repair_rows: list[dict[str, Any]] = []
    for r_idx, r in enumerate(_read_jsonl(V2_ROOT / "data/verified_banks_v2/lean_qwen35_9b_r2_rep5050_pairs_only.jsonl")):
        if r.get("formal_statement") and not decon.is_lean_contaminated(str(r["formal_statement"])):
            rc = dict(r)
            rc["id"] = f"lean_repair_v2:{r_idx:04d}"
            rc.setdefault(
                "lean_header",
                "import Mathlib\nimport Aesop\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n",
            )
            repair_rows.append(rc)
    lean_stage2_pool = s2_src + repair_rows

    m_s1 = _write_jsonl(corpora_dir / "v3_lean_stage1_sft.jsonl", s1_src)
    m_s2 = _write_jsonl(corpora_dir / "v3_lean_stage2_frontier_pool.jsonl", lean_stage2_pool)
    m_rl = _write_jsonl(corpora_dir / "v3_lean_rl_pool500.jsonl", lean_rl_pool500)

    # Re-run `audit-splits` on Track L
    guard_py = EVAL_V3_DIR / "governed_loop_guard_v37.py"
    for label, tr_p, cf_p, ho_p in [
        (
            "lean_stage1_vs_eval",
            corpora_dir / "v3_lean_stage1_sft.jsonl",
            splits_dir / "v3_lean_confirm180.jsonl",
            holdout_dir / "v3_lean_holdout.jsonl",
        ),
        (
            "lean_rl_vs_search_and_confirm",
            corpora_dir / "v3_lean_rl_pool500.jsonl",
            splits_dir / "v3_lean_search140.jsonl",
            splits_dir / "v3_lean_confirm180.jsonl",
        ),
    ]:
        r_aud = subprocess.run(
            [
                sys.executable, "-I", "-B", str(guard_py), "audit-splits",
                "--train", str(tr_p),
                "--confirm", str(cf_p),
                "--holdout", str(ho_p),
                "--group-key", "id",
            ],
            capture_output=True,
            text=True,
        )
        if r_aud.returncode != 0:
            raise RuntimeError(f"audit-splits failed on {label}:\n{r_aud.stdout}\n{r_aud.stderr}")

    # Update `v3_split_audit_manifest.json`
    man_p = splits_dir / "v3_split_audit_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    man["files"]["v3_lean_stage1_sft"] = m_s1
    man["files"]["v3_lean_stage2_frontier_pool"] = m_s2
    man["files"]["v3_lean_rl_pool500"] = m_rl
    man_p.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "LEAN_CORPORA_REBUILT_AND_AUDITED",
                "v3_lean_stage1_sft": m_s1,
                "v3_lean_stage2_frontier_pool": m_s2,
                "v3_lean_rl_pool500": m_rl,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

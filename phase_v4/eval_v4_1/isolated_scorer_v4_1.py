#!/usr/bin/env python3
"""
Isolated Out-of-Process Scorer v4.1 (`python -I -B eval_v4_1/isolated_scorer_v4_1.py`)
Peer-review hardened for Wave-2 (`V4.1`) under `governed_loop_guard_v39.py` (v3.9.0):
  1. Preserves 100% Lean 4.24 (`#print axioms` whitelist) & SymPy mathematical verification from `eval_v4/`.
  2. Adds Markdown-bold-tolerant `**Final Answer:**` extraction fallback when `\boxed{}` is absent
     (fixing Gemma-4-E4B's 67.5% natural-stop parse failure without altering boxed extraction).
  3. Emits `item["aux"]` alongside `item["aux_metrics"]` (`stop_rate`, `{track}_pass_at_8`, `has_thinking_rate`)
     for native `governed_loop_guard_v39.py` `paired_aux_noninferiority` enforcement.
  4. Logs complete protocol provenance (`temperature`, `max_tokens`, `thinking_mode`, `eval_backend`, `seed`)
     and `has_thinking_rate` directly inside the scored JSON envelope (`R2/R3/R4` remediation).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_V4_DIR = ROOT / "eval_v4"
if str(EVAL_V4_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_V4_DIR))

import channel_normalizer_v4 as cn4  # noqa: E402
import isolated_scorer_v4 as is4  # noqa: E402
import math_verifier_v4 as mv4  # noqa: E402

_ORIG_LENIENT = mv4.extract_answer_lenient_v4
_BOLD_FINAL_RE = re.compile(
    r"(?:^|\n)\s*(?:\*+\s*)?(?:Final\s+)?Answer(?:\s*\*+)?\s*:\s*(?:\*+\s*)?([^\n*]+)",
    re.IGNORECASE,
)


def extract_answer_lenient_v4_1(
    raw_text: str,
    *,
    channel_family: str = "auto",
    thinking: bool = False,
) -> str | None:
    ans = _ORIG_LENIENT(raw_text, channel_family=channel_family, thinking=thinking)
    if ans is not None:
        return ans
    norm = cn4.normalize_reasoning_channel(raw_text, channel_family=channel_family, expect_thinking=thinking)
    if norm.has_unclosed_thinking or not norm.final_text:
        return None
    matches = list(_BOLD_FINAL_RE.finditer(norm.final_text))
    if matches:
        cand = matches[-1].group(1).strip().rstrip(".").strip("$").strip()
        if cand:
            return cand
    return None


mv4.extract_answer_lenient_v4 = extract_answer_lenient_v4_1


def _compute_thinking_rates(gen_path: Path, channel_family: str, thinking: bool) -> dict[str, float]:
    opener = gzip.open if gen_path.suffix == ".gz" else open
    by_id: dict[str, list[float]] = {}
    with opener(gen_path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            pid = str(r["id"])
            txt = str(r.get("text", ""))
            norm = cn4.normalize_reasoning_channel(txt, channel_family=channel_family, expect_thinking=thinking)
            by_id.setdefault(pid, []).append(1.0 if norm.has_thinking else 0.0)
    return {pid: statistics.fmean(vs) for pid, vs in by_id.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-of-process isolated scorer v4.1")
    ap.add_argument("--track", choices=["math", "lean"], required=True)
    ap.add_argument("--channel-family", choices=list(cn4.VALID_CHANNEL_FAMILIES), default="auto")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--expected-k", type=int, default=8)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--env-fingerprint", default="a100_80gb_bf16_v4_1")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--thinking-mode-label", default="default")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--eval-backend", default="auto")
    ap.add_argument("--eval-seed", type=int, default=20260325)
    ap.add_argument("--lean-project", default="/scratch/governed_crossfamily_v4/lean/mathlib_env")
    ap.add_argument("--elan-bin", default="/scratch/governed_crossfamily_v4/lean/elan/bin")
    ap.add_argument("--lean-timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--math-chunk-timeout", type=float, default=20.0)
    ap.add_argument("--math-item-timeout", type=float, default=5.0)
    ap.add_argument("--scored-details-output", default="")
    ap.add_argument("--extra-metrics-json", default="")
    ap.add_argument("--compare-baseline", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rc = is4.score_dataset(args)
    if rc != 0:
        return rc

    out_p = Path(args.output)
    payload = json.loads(out_p.read_text(encoding="utf-8"))
    think_by_id = _compute_thinking_rates(Path(args.generations), args.channel_family, args.thinking)

    for it in payload.get("items", []):
        pid = str(it["id"])
        th_rate = float(think_by_id.get(pid, 0.0))
        aux_d = dict(it.get("aux_metrics") or {})
        aux_d["has_thinking_rate"] = th_rate
        it["aux_metrics"] = aux_d
        it["aux"] = dict(aux_d)
        it["has_thinking_rate"] = th_rate

    all_th = [float(it.get("has_thinking_rate", 0.0)) for it in payload.get("items", [])]
    payload["schema_version"] = "4.1"
    payload["protocol_metadata"] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking_flag": bool(args.thinking),
        "thinking_mode_label": args.thinking_mode_label,
        "eval_backend": args.eval_backend,
        "eval_seed": args.eval_seed,
        "lean_timeout_s": args.lean_timeout if args.track == "lean" else None,
    }
    payload["metrics"]["has_thinking_rate"] = statistics.fmean(all_th) if all_th else 0.0

    tmp = out_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

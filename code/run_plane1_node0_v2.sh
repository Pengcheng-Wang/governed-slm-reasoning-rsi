#!/bin/bash
set -euo pipefail
ROOT=/scratch/governed_slm_rsi_v1
export PYTHONUSERBASE=$ROOT/pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=$ROOT/tmp/hf_cache
mkdir -p $ROOT/results/plane1/lean $ROOT/logs

echo "[$(date -u +%FT%TZ)] Resuming node-0 Plane-1 Lean calibration"

# 1. Ensure CPU seed-bank live Lean 4.24 re-verification is running or completed
if [ ! -s "$ROOT/data/verified_banks/lean_seed_verified.jsonl" ]; then
  if ! pgrep -f "verify_lean_seed_bank.py" >/dev/null 2>&1; then
    nohup python -I -B $ROOT/code/verify_lean_seed_bank.py --workers 36 > $ROOT/logs/verify_lean_seed_bank.log 2>&1 &
  fi
fi

run_lean_eval() {
  local MODEL_DIR="$1"
  local TAG="$2"
  local CAND_ID="$3"
  local MAX_TOK="$4"
  local SEED="$5"
  local EXTRA_GEN="$6"
  local EXTRA_SCORE="$7"

  local GEN_OUT="$ROOT/results/plane1/lean/${TAG}.jsonl.gz"
  local GEN_SUM="$ROOT/results/plane1/lean/${TAG}_gen.json"
  local MET_OUT="$ROOT/results/plane1/lean/${TAG}_metrics.json"
  local DET_OUT="$ROOT/results/plane1/lean/${TAG}_details.jsonl.gz"

  if [ ! -s "$GEN_OUT" ]; then
    python -I -B $ROOT/code/vllm_eval_engine.py \
      --model-path "$MODEL_DIR" \
      --split-file $ROOT/data/splits/lean_search128.jsonl \
      --output "$GEN_OUT" \
      --summary-output "$GEN_SUM" \
      --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.8 --max-tokens "$MAX_TOK" --seed "$SEED" $EXTRA_GEN
  fi

  if [ ! -s "$MET_OUT" ]; then
    python -I -B $ROOT/eval/isolated_scorer.py \
      --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
      --generations "$GEN_OUT" \
      --expected-k 8 --candidate-id "$CAND_ID" --protocol-id plane1_lean_search \
      --workers 32 --output "$MET_OUT" --scored-details-output "$DET_OUT" $EXTRA_SCORE
  fi
}

run_lean_eval "$ROOT/models/qwen35_4b" "q4b_direct_b0"  "q4b_lean_direct_b0"  2048 20261118 "" ""
run_lean_eval "$ROOT/models/qwen35_4b" "q4b_direct_aa1" "q4b_lean_direct_aa1" 2048 20261119 "" ""
run_lean_eval "$ROOT/models/qwen35_4b" "q4b_think_b0"   "q4b_lean_think_b0"   8192 20261118 "--thinking --presence-penalty 1.0" "--thinking"
run_lean_eval "$ROOT/models/qwen35_9b" "q9b_direct_b0"  "q9b_lean_direct_b0"  2048 20261118 "" ""
run_lean_eval "$ROOT/models/qwen35_9b" "q9b_think_b0"   "q9b_lean_think_b0"   8192 20261118 "--thinking --presence-penalty 1.0" "--thinking"

# Wait for seed-bank verification if still running
while pgrep -f "verify_lean_seed_bank.py" >/dev/null 2>&1; do
  sleep 5
done

echo "[$(date -u +%FT%TZ)] PLANE1_NODE0_DONE"

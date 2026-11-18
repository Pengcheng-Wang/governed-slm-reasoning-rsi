#!/bin/bash
set -euo pipefail
ROLE="$1"
ROOT=/scratch/governed_slm_rsi_v1
export PYTHONUSERBASE=$ROOT/pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=$ROOT/tmp/hf_cache

eval_direct12k() {
  local MODEL_DIR="$1"
  local TAG="$2"
  local CAND_ID="$3"
  local TP="$4"
  local SEED="$5"

  local GEN_OUT="$ROOT/results/plane1/math/${TAG}.jsonl.gz"
  local GEN_SUM="$ROOT/results/plane1/math/${TAG}_gen.json"
  local MET_OUT="$ROOT/results/plane1/math/${TAG}_metrics.json"
  local DET_OUT="$ROOT/results/plane1/math/${TAG}_details.jsonl.gz"

  if [ ! -s "$GEN_OUT" ]; then
    python -I -B $ROOT/code/vllm_eval_engine.py \
      --model-path "$MODEL_DIR" \
      --split-file $ROOT/data/splits/math_search128.jsonl \
      --output "$GEN_OUT" \
      --summary-output "$GEN_SUM" \
      --k 8 --tp "$TP" --gpus 0,1,2,3,4,5,6,7 \
      --temperature 0.7 --max-tokens 12288 --max-model-len 16384 --seed "$SEED"
  fi
  if [ ! -s "$MET_OUT" ]; then
    python -I -B $ROOT/eval/isolated_scorer.py \
      --track math --split-file $ROOT/data/splits/math_search128.jsonl \
      --generations "$GEN_OUT" \
      --expected-k 8 --candidate-id "$CAND_ID" --protocol-id plane1_math_search_direct12k \
      --output "$MET_OUT" --scored-details-output "$DET_OUT"
  fi
}

if [ "$ROLE" = "node1" ]; then
  eval_direct12k "$ROOT/models/qwen35_4b" "q4b_direct12k_b0"  "q4b_math_direct12k_b0"  1 20261118
  eval_direct12k "$ROOT/models/qwen35_4b" "q4b_direct12k_aa1" "q4b_math_direct12k_aa1" 1 20261119
elif [ "$ROLE" = "node2" ]; then
  eval_direct12k "$ROOT/models/qwen35_9b"  "q9b_direct12k_b0"  "q9b_math_direct12k_b0"  1 20261118
  eval_direct12k "$ROOT/models/qwen38_27b" "q27b_direct12k_b0" "q27b_math_direct12k_b0" 2 20261118
fi
echo "[$(date -u +%FT%TZ)] DIRECT12K_${ROLE}_DONE"

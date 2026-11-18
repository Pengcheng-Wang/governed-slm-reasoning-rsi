#!/bin/bash
set -euo pipefail
ROOT=/scratch/governed_slm_rsi_v1
export PYTHONUSERBASE=$ROOT/pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=$ROOT/tmp/hf_cache
mkdir -p $ROOT/results/plane1/math $ROOT/logs

echo "[$(date -u +%FT%TZ)] Resuming node-1 Plane-1 Competition Math calibration on Qwen3.5-4B"

run_math_eval() {
  local MODEL_DIR="$1"
  local TAG="$2"
  local CAND_ID="$3"
  local TEMP="$4"
  local MAX_TOK="$5"
  local MAX_LEN="$6"
  local SEED="$7"
  local EXTRA_GEN="$8"
  local EXTRA_SCORE="$9"

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
      --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature "$TEMP" --max-tokens "$MAX_TOK" --max-model-len "$MAX_LEN" --seed "$SEED" $EXTRA_GEN
  fi

  if [ ! -s "$MET_OUT" ]; then
    python -I -B $ROOT/eval/isolated_scorer.py \
      --track math --split-file $ROOT/data/splits/math_search128.jsonl \
      --generations "$GEN_OUT" \
      --expected-k 8 --candidate-id "$CAND_ID" --protocol-id plane1_math_search \
      --output "$MET_OUT" --scored-details-output "$DET_OUT" $EXTRA_SCORE
  fi
}

run_math_eval "$ROOT/models/qwen35_4b" "q4b_direct4k_b0"     "q4b_math_direct4k"      0.7 4096  32768 20261118 "" ""
run_math_eval "$ROOT/models/qwen35_4b" "q4b_think16k_pp0"    "q4b_math_think16k_pp0"  0.8 16384 32768 20261118 "--thinking --presence-penalty 0.0" "--thinking"
run_math_eval "$ROOT/models/qwen35_4b" "q4b_think16k_pp1"    "q4b_math_think16k_pp1"  0.8 16384 32768 20261118 "--thinking --presence-penalty 1.0" "--thinking"
run_math_eval "$ROOT/models/qwen35_4b" "q4b_think16k_pp1_aa1" "q4b_math_think16k_pp1_aa1" 0.8 16384 32768 20261119 "--thinking --presence-penalty 1.0" "--thinking"
run_math_eval "$ROOT/models/qwen35_4b" "q4b_think24k_pp1"    "q4b_math_think24k_pp1"  0.8 24576 32768 20261118 "--thinking --presence-penalty 1.0" "--thinking"

echo "[$(date -u +%FT%TZ)] PLANE1_NODE1_DONE"

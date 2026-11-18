#!/bin/bash
set -euo pipefail
ROOT=/scratch/governed_slm_rsi_v1
export PYTHONUSERBASE=$ROOT/pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=$ROOT/tmp/hf_cache
mkdir -p $ROOT/results/plane1/math $ROOT/results/plane1/lean $ROOT/logs

echo "[$(date -u +%FT%TZ)] Resuming node-2 Plane-1 Qwen3.5-9B Math + Qwen3.8-27B Comparator"

if [ ! -s "$ROOT/results/plane1/math/q9b_direct4k_b0.jsonl.gz" ]; then
  python -I -B $ROOT/code/vllm_eval_engine.py \
    --model-path $ROOT/models/qwen35_9b \
    --split-file $ROOT/data/splits/math_search128.jsonl \
    --output $ROOT/results/plane1/math/q9b_direct4k_b0.jsonl.gz \
    --summary-output $ROOT/results/plane1/math/q9b_direct4k_b0_gen.json \
    --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.7 --max-tokens 4096 --seed 20261118
fi
if [ ! -s "$ROOT/results/plane1/math/q9b_direct4k_b0_metrics.json" ]; then
  python -I -B $ROOT/eval/isolated_scorer.py \
    --track math --split-file $ROOT/data/splits/math_search128.jsonl \
    --generations $ROOT/results/plane1/math/q9b_direct4k_b0.jsonl.gz \
    --expected-k 8 --candidate-id q9b_math_direct4k --protocol-id plane1_math_search \
    --output $ROOT/results/plane1/math/q9b_direct4k_b0_metrics.json \
    --scored-details-output $ROOT/results/plane1/math/q9b_direct4k_b0_details.jsonl.gz
fi

if [ ! -s "$ROOT/results/plane1/math/q9b_think16k_pp1.jsonl.gz" ]; then
  python -I -B $ROOT/code/vllm_eval_engine.py \
    --model-path $ROOT/models/qwen35_9b \
    --split-file $ROOT/data/splits/math_search128.jsonl \
    --output $ROOT/results/plane1/math/q9b_think16k_pp1.jsonl.gz \
    --summary-output $ROOT/results/plane1/math/q9b_think16k_pp1_gen.json \
    --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 16384 --seed 20261118
fi
if [ ! -s "$ROOT/results/plane1/math/q9b_think16k_pp1_metrics.json" ]; then
  python -I -B $ROOT/eval/isolated_scorer.py \
    --track math --split-file $ROOT/data/splits/math_search128.jsonl \
    --generations $ROOT/results/plane1/math/q9b_think16k_pp1.jsonl.gz \
    --expected-k 8 --candidate-id q9b_math_think16k_pp1 --protocol-id plane1_math_search \
    --thinking --output $ROOT/results/plane1/math/q9b_think16k_pp1_metrics.json \
    --scored-details-output $ROOT/results/plane1/math/q9b_think16k_pp1_details.jsonl.gz
fi

if [ -e $ROOT/models/qwen38_27b/config.json ]; then
  if [ ! -s "$ROOT/results/plane1/lean/q27b_direct_b0.jsonl.gz" ]; then
    python -I -B $ROOT/code/vllm_eval_engine.py \
      --model-path $ROOT/models/qwen38_27b \
      --split-file $ROOT/data/splits/lean_search128.jsonl \
      --output $ROOT/results/plane1/lean/q27b_direct_b0.jsonl.gz \
      --summary-output $ROOT/results/plane1/lean/q27b_direct_b0_gen.json \
      --k 8 --tp 2 --gpus 0,1,2,3,4,5,6,7 --temperature 0.8 --max-tokens 2048 --seed 20261118
  fi
  if [ -d "$ROOT/lean/mathlib_env" ] && [ ! -s "$ROOT/results/plane1/lean/q27b_direct_b0_metrics.json" ]; then
    python -I -B $ROOT/eval/isolated_scorer.py \
      --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
      --generations $ROOT/results/plane1/lean/q27b_direct_b0.jsonl.gz \
      --expected-k 8 --candidate-id q27b_lean_direct_b0 --protocol-id plane1_lean_search \
      --workers 32 --output $ROOT/results/plane1/lean/q27b_direct_b0_metrics.json \
      --scored-details-output $ROOT/results/plane1/lean/q27b_direct_b0_details.jsonl.gz
  fi

  if [ ! -s "$ROOT/results/plane1/math/q27b_think24k_pp1.jsonl.gz" ]; then
    python -I -B $ROOT/code/vllm_eval_engine.py \
      --model-path $ROOT/models/qwen38_27b \
      --split-file $ROOT/data/splits/math_search128.jsonl \
      --output $ROOT/results/plane1/math/q27b_think24k_pp1.jsonl.gz \
      --summary-output $ROOT/results/plane1/math/q27b_think24k_pp1_gen.json \
      --k 8 --tp 2 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 24576 --max-model-len 32768 --seed 20261118
  fi
  if [ ! -s "$ROOT/results/plane1/math/q27b_think24k_pp1_metrics.json" ]; then
    python -I -B $ROOT/eval/isolated_scorer.py \
      --track math --split-file $ROOT/data/splits/math_search128.jsonl \
      --generations $ROOT/results/plane1/math/q27b_think24k_pp1.jsonl.gz \
      --expected-k 8 --candidate-id q27b_math_think24k_pp1 --protocol-id plane1_math_search \
      --thinking --output $ROOT/results/plane1/math/q27b_think24k_pp1_metrics.json \
      --scored-details-output $ROOT/results/plane1/math/q27b_think24k_pp1_details.jsonl.gz
  fi
fi

echo "[$(date -u +%FT%TZ)] PLANE1_NODE2_DONE"

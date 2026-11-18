#!/bin/bash
set -euo pipefail
ROOT=/scratch/governed_slm_rsi_v1
export PYTHONUSERBASE=$ROOT/pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=$ROOT/tmp/hf_cache
mkdir -p $ROOT/results/plane1/math $ROOT/logs

echo "[$(date -u +%FT%TZ)] Starting node-1 Plane-1 Competition Math calibration on Qwen3.5-4B"

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/math_search128.jsonl \
  --output $ROOT/results/plane1/math/q4b_direct4k_b0.jsonl.gz \
  --summary-output $ROOT/results/plane1/math/q4b_direct4k_b0_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.7 --max-tokens 4096 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track math --split-file $ROOT/data/splits/math_search128.jsonl \
  --generations $ROOT/results/plane1/math/q4b_direct4k_b0.jsonl.gz \
  --expected-k 8 --candidate-id q4b_math_direct4k --protocol-id plane1_math_search \
  --output $ROOT/results/plane1/math/q4b_direct4k_b0_metrics.json \
  --scored-details-output $ROOT/results/plane1/math/q4b_direct4k_b0_details.jsonl

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/math_search128.jsonl \
  --output $ROOT/results/plane1/math/q4b_think16k_pp0.jsonl.gz \
  --summary-output $ROOT/results/plane1/math/q4b_think16k_pp0_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 0.0 --max-tokens 16384 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track math --split-file $ROOT/data/splits/math_search128.jsonl \
  --generations $ROOT/results/plane1/math/q4b_think16k_pp0.jsonl.gz \
  --expected-k 8 --candidate-id q4b_math_think16k_pp0 --protocol-id plane1_math_search \
  --thinking --output $ROOT/results/plane1/math/q4b_think16k_pp0_metrics.json \
  --scored-details-output $ROOT/results/plane1/math/q4b_think16k_pp0_details.jsonl

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/math_search128.jsonl \
  --output $ROOT/results/plane1/math/q4b_think16k_pp1.jsonl.gz \
  --summary-output $ROOT/results/plane1/math/q4b_think16k_pp1_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 16384 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track math --split-file $ROOT/data/splits/math_search128.jsonl \
  --generations $ROOT/results/plane1/math/q4b_think16k_pp1.jsonl.gz \
  --expected-k 8 --candidate-id q4b_math_think16k_pp1 --protocol-id plane1_math_search \
  --thinking --output $ROOT/results/plane1/math/q4b_think16k_pp1_metrics.json \
  --scored-details-output $ROOT/results/plane1/math/q4b_think16k_pp1_details.jsonl

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/math_search128.jsonl \
  --output $ROOT/results/plane1/math/q4b_think16k_pp1_aa1.jsonl.gz \
  --summary-output $ROOT/results/plane1/math/q4b_think16k_pp1_aa1_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 16384 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track math --split-file $ROOT/data/splits/math_search128.jsonl \
  --generations $ROOT/results/plane1/math/q4b_think16k_pp1_aa1.jsonl.gz \
  --expected-k 8 --candidate-id q4b_math_think16k_pp1_aa1 --protocol-id plane1_math_search \
  --thinking --output $ROOT/results/plane1/math/q4b_think16k_pp1_aa1_metrics.json

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/math_search128.jsonl \
  --output $ROOT/results/plane1/math/q4b_think24k_pp1.jsonl.gz \
  --summary-output $ROOT/results/plane1/math/q4b_think24k_pp1_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 24576 --max-model-len 32768 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track math --split-file $ROOT/data/splits/math_search128.jsonl \
  --generations $ROOT/results/plane1/math/q4b_think24k_pp1.jsonl.gz \
  --expected-k 8 --candidate-id q4b_math_think24k_pp1 --protocol-id plane1_math_search \
  --thinking --output $ROOT/results/plane1/math/q4b_think24k_pp1_metrics.json

echo "[$(date -u +%FT%TZ)] PLANE1_NODE1_DONE"

#!/bin/bash
set -euo pipefail
ROOT=/scratch/governed_slm_rsi_v1
export PYTHONUSERBASE=$ROOT/pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=$ROOT/tmp/hf_cache
mkdir -p $ROOT/results/plane1/lean $ROOT/logs

echo "[$(date -u +%FT%TZ)] Starting node-0 Plane-1 Lean calibration + live seed-bank re-verification"

# 1. Start CPU seed-bank live Lean 4.24 re-verification (36 CPU cores) in background while GPUs run vLLM
python -I -B $ROOT/code/verify_lean_seed_bank.py --workers 36 > $ROOT/logs/verify_lean_seed_bank.log 2>&1 &
SEED_PID=$!

# 2. Qwen3.5-4B Lean Search128: direct (2048) vs thinking (8192, pp=1.0) + A/A repeat
python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/lean_search128.jsonl \
  --output $ROOT/results/plane1/lean/q4b_direct_b0.jsonl.gz \
  --summary-output $ROOT/results/plane1/lean/q4b_direct_b0_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.8 --max-tokens 2048 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
  --generations $ROOT/results/plane1/lean/q4b_direct_b0.jsonl.gz \
  --expected-k 8 --candidate-id q4b_lean_direct_b0 --protocol-id plane1_lean_search \
  --workers 32 --output $ROOT/results/plane1/lean/q4b_direct_b0_metrics.json \
  --scored-details-output $ROOT/results/plane1/lean/q4b_direct_b0_details.jsonl

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/lean_search128.jsonl \
  --output $ROOT/results/plane1/lean/q4b_direct_aa1.jsonl.gz \
  --summary-output $ROOT/results/plane1/lean/q4b_direct_aa1_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.8 --max-tokens 2048 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
  --generations $ROOT/results/plane1/lean/q4b_direct_aa1.jsonl.gz \
  --expected-k 8 --candidate-id q4b_lean_direct_aa1 --protocol-id plane1_lean_search \
  --workers 32 --output $ROOT/results/plane1/lean/q4b_direct_aa1_metrics.json

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_4b \
  --split-file $ROOT/data/splits/lean_search128.jsonl \
  --output $ROOT/results/plane1/lean/q4b_think_b0.jsonl.gz \
  --summary-output $ROOT/results/plane1/lean/q4b_think_b0_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 8192 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
  --generations $ROOT/results/plane1/lean/q4b_think_b0.jsonl.gz \
  --expected-k 8 --candidate-id q4b_lean_think_b0 --protocol-id plane1_lean_search \
  --thinking --workers 32 --output $ROOT/results/plane1/lean/q4b_think_b0_metrics.json \
  --scored-details-output $ROOT/results/plane1/lean/q4b_think_b0_details.jsonl

# 3. Qwen3.5-9B Lean Search128: direct (2048) vs thinking (8192, pp=1.0)
python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_9b \
  --split-file $ROOT/data/splits/lean_search128.jsonl \
  --output $ROOT/results/plane1/lean/q9b_direct_b0.jsonl.gz \
  --summary-output $ROOT/results/plane1/lean/q9b_direct_b0_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --temperature 0.8 --max-tokens 2048 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
  --generations $ROOT/results/plane1/lean/q9b_direct_b0.jsonl.gz \
  --expected-k 8 --candidate-id q9b_lean_direct_b0 --protocol-id plane1_lean_search \
  --workers 32 --output $ROOT/results/plane1/lean/q9b_direct_b0_metrics.json \
  --scored-details-output $ROOT/results/plane1/lean/q9b_direct_b0_details.jsonl

python -I -B $ROOT/code/vllm_eval_engine.py \
  --model-path $ROOT/models/qwen35_9b \
  --split-file $ROOT/data/splits/lean_search128.jsonl \
  --output $ROOT/results/plane1/lean/q9b_think_b0.jsonl.gz \
  --summary-output $ROOT/results/plane1/lean/q9b_think_b0_gen.json \
  --k 8 --tp 1 --gpus 0,1,2,3,4,5,6,7 --thinking --temperature 0.8 --presence-penalty 1.0 --max-tokens 8192 --seed 20261118

python -I -B $ROOT/eval/isolated_scorer.py \
  --track lean --split-file $ROOT/data/splits/lean_search128.jsonl \
  --generations $ROOT/results/plane1/lean/q9b_think_b0.jsonl.gz \
  --expected-k 8 --candidate-id q9b_lean_think_b0 --protocol-id plane1_lean_search \
  --thinking --workers 32 --output $ROOT/results/plane1/lean/q9b_think_b0_metrics.json

wait $SEED_PID
echo "[$(date -u +%FT%TZ)] PLANE1_NODE0_DONE"

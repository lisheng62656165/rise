#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
POOL=/home/lisheng/.nvidia_api_key_pool_stride10
MODEL=nemotron-3.5-lightning-30b-a3b
export DEEPPLANNING_MODEL="$MODEL" DEEPPLANNING_API_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
export DEEPPLANNING_API_KEY_ENV=NVIDIA_API_KEY DEEPPLANNING_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_REQUEST_TIMEOUT=90 DEEPPLANNING_MAX_RETRIES=2 DEEPPLANNING_RETRY_BACKOFF=2
export HTTP_PROXY=http://127.0.0.1:17897 HTTPS_PROXY=http://127.0.0.1:17897 ALL_PROXY=http://127.0.0.1:17897 PYTHONUNBUFFERED=1
key_index=0
for cohort in travel-zh travel-en; do
  for shard in 0 1; do
    key_index=$((key_index + 1))
    key=$(sed -n "${key_index}p" "$POOL")
    nohup env NVIDIA_API_KEY="$key" PYTHONPATH="$ROOT:$ROOT/travelplanning" \
      "$PY" "$ROOT/run_deepplanning_eds_eca.py" --root "$ROOT" --output "$OUT" \
      --cohort "$cohort" --workers 1 --max-llm-calls 400 --proposal-seed 64639 \
      --selector-seed 77113 --anchor-tag nemotron_vanilla_a_trial1 \
      --anchor-model-slug "$MODEL" --deepplanning-adapter \
      --job-shard-index "$shard" --job-shard-count 10 \
      >"$OUT/logs/missing_travel_20/lowconc_${cohort}_${shard}.log" 2>&1 &
  done
done

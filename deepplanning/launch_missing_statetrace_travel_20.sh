#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
POOL=/home/lisheng/.nvidia_api_key_pool_stride10
LOG="$OUT/logs/missing_travel_20"
MODEL=nemotron-3.5-lightning-30b-a3b

mkdir -p "$LOG"
export DEEPPLANNING_MODEL="$MODEL"
export DEEPPLANNING_API_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
export DEEPPLANNING_API_KEY_ENV=NVIDIA_API_KEY
export DEEPPLANNING_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_REQUEST_TIMEOUT=180
export DEEPPLANNING_MAX_RETRIES=4
export DEEPPLANNING_RETRY_BACKOFF=2
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

: >"$LOG/pids.tsv"
key_count=$(awk 'NF' "$POOL" | wc -l)
key_offset=${DEEPPLANNING_KEY_OFFSET:-0}
key_slot=0
for cohort in travel-zh travel-en; do
  for shard in $(seq 0 9); do
    key_slot=$((key_slot + 1))
    key_index=$(( (key_offset + key_slot - 1) % key_count + 1 ))
    key=$(sed -n "${key_index}p" "$POOL")
    test -n "$key"
    nohup env NVIDIA_API_KEY="$key" PYTHONPATH="$ROOT:$ROOT/travelplanning" \
      "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
        --root "$ROOT" --output "$OUT" --cohort "$cohort" --workers 1 \
        --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
        --anchor-tag nemotron_vanilla_a_trial1 --anchor-model-slug "$MODEL" \
        --deepplanning-adapter --job-shard-index "$shard" --job-shard-count 10 \
        >"$LOG/${cohort}_shard_${shard}.log" 2>&1 &
    printf '%s\t%s\t%s\t%s\n' "$!" "$cohort" "$shard" "$key_index" >>"$LOG/pids.tsv"
  done
done

cat "$LOG/pids.tsv"

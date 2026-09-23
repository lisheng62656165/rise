#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
POOL=/home/lisheng/.nvidia_api_key_pool_stride10
LOG="$OUT/logs/missing_travel_20"
PID_FILE="$LOG/pids.tsv"
MODEL=nemotron-3.5-lightning-30b-a3b
KEY_OFFSET=${DEEPPLANNING_KEY_OFFSET:-1}

export DEEPPLANNING_MODEL="$MODEL"
export DEEPPLANNING_API_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
export DEEPPLANNING_API_KEY_ENV=NVIDIA_API_KEY
export DEEPPLANNING_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_REQUEST_TIMEOUT=${DEEPPLANNING_REQUEST_TIMEOUT:-90}
export DEEPPLANNING_MAX_RETRIES=${DEEPPLANNING_MAX_RETRIES:-2}
export DEEPPLANNING_RETRY_BACKOFF=${DEEPPLANNING_RETRY_BACKOFF:-2}
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

key_count=$(awk 'NF' "$POOL" | wc -l)
tmp=$(mktemp "$LOG/pids.tsv.XXXXXX")
refilled=0
while IFS=$'\t' read -r pid cohort shard old_key_index; do
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf '%s\t%s\t%s\t%s\n' "$pid" "$cohort" "$shard" "$old_key_index" >> "$tmp"
    continue
  fi

  cohort_offset=0
  [[ "$cohort" == "travel-en" ]] && cohort_offset=10
  slot=$((cohort_offset + shard + 1))
  key_index=$(( (KEY_OFFSET + slot - 1) % key_count + 1 ))
  key=$(sed -n "${key_index}p" "$POOL")
  nohup env NVIDIA_API_KEY="$key" PYTHONPATH="$ROOT:$ROOT/travelplanning" \
    "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
      --root "$ROOT" --output "$OUT" --cohort "$cohort" --workers 1 \
      --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
      --anchor-tag nemotron_vanilla_a_trial1 --anchor-model-slug "$MODEL" \
      --deepplanning-adapter --job-shard-index "$shard" --job-shard-count 10 \
      >> "$LOG/${cohort}_shard_${shard}.log" 2>&1 &
  printf '%s\t%s\t%s\t%s\n' "$!" "$cohort" "$shard" "$key_index" >> "$tmp"
  refilled=$((refilled + 1))
done < "$PID_FILE"
mv "$tmp" "$PID_FILE"
printf '%s refilled=%d active_slots=%d\n' \
  "$(date --iso-8601=seconds)" "$refilled" "$(wc -l < "$PID_FILE")" \
  >> "$LOG/refill.log"

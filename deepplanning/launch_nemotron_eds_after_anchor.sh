#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
API_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
SHOP_ANCHOR=nemotron_vanilla_a_trial1_merged
TRAVEL_ANCHOR=nemotron_vanilla_a_trial1
LOG="$OUT/logs"

export DEEPPLANNING_MODEL="$MODEL"
export DEEPPLANNING_API_MODEL="$API_MODEL"
export DEEPPLANNING_API_KEY_ENV=NVIDIA_API_KEY
export DEEPPLANNING_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export HTTP_PROXY='http://127.0.0.1:17897'
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1
mkdir -p "$LOG"

count_files() { { find "$1" -type f -name "$2" 2>/dev/null || true; } | wc -l; }
count_substantive_plans() {
  { find "$1" -maxdepth 1 -type f -name '*.txt' -size +199c 2>/dev/null || true; } | wc -l
}
while true; do
  s1=$(count_files "$ROOT/shoppingplanning/database_infered/database_${SHOP_ANCHOR}_L1" messages.json)
  s2=$(count_files "$ROOT/shoppingplanning/database_infered/database_${SHOP_ANCHOR}_L2" messages.json)
  s3=$(count_files "$ROOT/shoppingplanning/database_infered/database_${SHOP_ANCHOR}_L3" messages.json)
  tz=$(count_substantive_plans "$ROOT/travel_runs/$TRAVEL_ANCHOR/${MODEL}_zh/reports")
  te=$(count_substantive_plans "$ROOT/travel_runs/$TRAVEL_ANCHOR/${MODEL}_en/reports")
  printf '%s anchors shopping=%s/%s/%s travel=%s/%s\n' "$(date -Is)" "$s1" "$s2" "$s3" "$tz" "$te" >>"$LOG/chain.log"
  if [[ "$s1" == 50 && "$s2" == 50 && "$s3" == 20 && "$tz" == 120 && "$te" == 120 ]]; then break; fi
  sleep 120
done

run_eds_shard() {
  local cohort=$1 shard_index=$2 shard_count=$3 anchor=$4 key_index=$5
  local domain=travelplanning extra=(--anchor-model-slug "$MODEL" --deepplanning-adapter)
  local key
  if [[ "$cohort" == shopping ]]; then domain=shoppingplanning; extra=(); fi
  key=$(awk 'NF' /home/lisheng/.nvidia_api_key_pool_eds_eca | sed -n "${key_index}p")
  [[ -n "$key" ]] || { echo "Missing API key index $key_index" >&2; exit 1; }
  nohup env NVIDIA_API_KEY="$key" \
  PYTHONPATH="$ROOT:$ROOT/$domain" "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
    --root "$ROOT" --output "$OUT" --cohort "$cohort" --workers 1 \
    --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
    --anchor-tag "$anchor" --job-shard-index "$shard_index" --job-shard-count "$shard_count" \
    "${extra[@]}" >>"$LOG/${cohort}_shard_${shard_index}.log" 2>&1 &
  LAST_PID=$!
}

pids=()
key_indexes=(1 2 3 4 5 6 9 10)
key_position=0
for shard in $(seq 0 7); do
  key_index=${key_indexes[$((key_position % ${#key_indexes[@]}))]}
  run_eds_shard shopping "$shard" 8 "$SHOP_ANCHOR" "$key_index"
  pids+=("$LAST_PID")
  key_position=$((key_position + 1))
done
for cohort in travel-zh travel-en; do
  for shard in $(seq 0 5); do
    key_index=${key_indexes[$((key_position % ${#key_indexes[@]}))]}
    run_eds_shard "$cohort" "$shard" 6 "$TRAVEL_ANCHOR" "$key_index"
    pids+=("$LAST_PID")
    key_position=$((key_position + 1))
  done
done
printf '%s\n' "${pids[@]}" >"$LOG/eds_pids.txt"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf '%s eds_status=%s\n' "$(date -Is)" "$status" >>"$LOG/chain.log"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
API_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
LOG="$OUT/logs"
KEY_FILE=/home/lisheng/.nvidia_api_key_pool_eds_eca

export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export DEEPPLANNING_API_KEY_ENV=NVIDIA_API_KEY
export DEEPPLANNING_MODEL="$MODEL"
export DEEPPLANNING_API_MODEL="$API_MODEL"
export DEEPPLANNING_BASE_URL=https://integrate.api.nvidia.com/v1
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export PYTHONUNBUFFERED=1
mkdir -p "$LOG"

key_indexes=(1 2 3 4 5 6 7 9 10 11)
pids=()
pos=0
run_shard() {
  local cohort=$1 idx=$2 key_index=$3
  local key
  key=$(awk 'NF' "$KEY_FILE" | sed -n "${key_index}p")
  [[ -n "$key" ]] || { echo "missing key $key_index" >&2; exit 1; }
  nohup env NVIDIA_API_KEY="$key" \
    PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
    --root "$ROOT" --output "$OUT" --cohort "$cohort" --workers 1 \
    --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
    --anchor-tag nemotron_vanilla_a_trial1 \
    --anchor-model-slug "$MODEL" --deepplanning-adapter \
    --job-shard-index "$idx" --job-shard-count 10 \
    </dev/null >>"$LOG/${cohort}_resume20_shard_${idx}.log" 2>&1 &
  pids+=("$!")
}

for cohort in travel-zh travel-en; do
  for idx in $(seq 0 9); do
    key_index=${key_indexes[$((pos % ${#key_indexes[@]}))]}
    run_shard "$cohort" "$idx" "$key_index"
    pos=$((pos + 1))
  done
done
printf '%s\n' "${pids[@]}" >"$LOG/travel_resume20_pids.txt"
printf '%s launched travel_resume20 workers=%s\n' "$(date -Is)" "${#pids[@]}" >>"$LOG/recovery.log"
wait_status=0
for pid in "${pids[@]}"; do wait "$pid" || wait_status=1; done
printf '%s travel_resume20_status=%s\n' "$(date -Is)" "$wait_status" >>"$LOG/recovery.log"
zh=$(find "$OUT/travel_zh/tasks" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
en=$(find "$OUT/travel_en/tasks" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
if [[ "$zh" -ne 120 || "$en" -ne 120 ]]; then
  nohup bash "$ROOT/recover_nemotron_eds_until_complete.sh" >>"$LOG/recovery.log" 2>&1 &
fi
exit "$wait_status"

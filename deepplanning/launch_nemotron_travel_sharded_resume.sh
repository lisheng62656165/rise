#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
ANCHOR=nemotron_vanilla_a_trial1
POOL=/home/lisheng/.nvidia_api_key_pool_eds_eca
BASE="$ROOT/travel_runs/$ANCHOR"
LOG="$ROOT/logs/$ANCHOR/sharded_timeout600"

mkdir -p "$LOG"

archive_orphans() {
  local language=$1
  local output="$BASE/${MODEL}_${language}"
  local archive="$BASE/infra_invalid_empty_${language}_sharded"
  mkdir -p "$archive"
  for trajectory in "$output"/trajectories/id_*.json; do
    [[ -e "$trajectory" ]] || continue
    local id
    id=$(basename "$trajectory" .json)
    if [[ ! -s "$output/reports/$id.txt" ]]; then
      mv "$trajectory" "$archive/"
    fi
  done
}

launch_language() {
  local language=$1
  shift
  local -a key_indexes=("$@")
  local output="$BASE/${MODEL}_${language}"
  local -a missing=()
  local id shard key_index key csv pid

  for id in $(seq 0 119); do
    [[ -s "$output/reports/id_${id}.txt" ]] || missing+=("$id")
  done

  for shard in $(seq 0 4); do
    csv=""
    for ((id=shard; id<${#missing[@]}; id+=5)); do
      csv+="${csv:+,}${missing[$id]}"
    done
    [[ -n "$csv" ]] || continue
    key_index=${key_indexes[$shard]}
    key=$(awk 'NF' "$POOL" | sed -n "${key_index}p")
    [[ -n "$key" ]] || { echo "Missing API key index $key_index" >&2; exit 1; }

    nohup env \
      NVIDIA_API_KEY="$key" \
      DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1 \
      HTTP_PROXY='http://127.0.0.1:17897' \
      HTTPS_PROXY='http://127.0.0.1:17897' \
      ALL_PROXY='http://127.0.0.1:17897' \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="$ROOT:$ROOT/travelplanning" \
      "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" \
        --travel-root "$ROOT/travelplanning" \
        --model "$MODEL" \
        --language "$language" \
        --workers 1 \
        --max-llm-calls 400 \
        --seed 53403 \
        --rerun-ids "$csv" \
        --output-root "$BASE" \
        >"$LOG/${language}_shard_${shard}.log" 2>&1 &
    pid=$!
    printf '%s %s %s %s\n' "$pid" "$language" "$shard" "$key_index" \
      >>"$LOG/pids.tsv"
  done
}

: >"$LOG/pids.tsv"
archive_orphans zh
archive_orphans en
launch_language zh 1 2 3 4 5
launch_language en 6 7 6 9 10
printf 'started=%s\n' "$(date -Is)" >"$LOG/status.txt"

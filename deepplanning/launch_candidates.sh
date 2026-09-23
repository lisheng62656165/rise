#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
SHOPPING_ROOT="$ROOT/shoppingplanning"
RUNNER="$ROOT/run_deepplanning_shopping_subset.py"

export MIMO_API_KEY="$(tr -d '\r\n' </home/lisheng/.mimo_api_key)"
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONPATH="$ROOT:$SHOPPING_ROOT"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/logs"

launch() {
  local arm=$1 level=$2 workers=$3 ids=$4
  local run_name="full_${arm}_L${level}_20260828"
  local log="$ROOT/logs/${run_name}.log"
  "$PY" "$RUNNER" \
    --shopping-root "$SHOPPING_ROOT" \
    --model mimo-v2.5-pro \
    --level "$level" \
    --case-ids $ids \
    --run-name "$run_name" \
    --workers "$workers" \
    --max-llm-calls 400 \
    --trial 1 \
    --orchestration-seed 20260828 \
    --allow-inference-failures >"$log" 2>&1 &
  pids+=("$!")
}

pids=()
for arm in vanilla_a generic_g; do
  launch "$arm" 1 4 "$(seq -s ' ' 1 50)"
  launch "$arm" 2 4 "$(seq -s ' ' 1 50)"
  launch "$arm" 3 2 "$(seq -s ' ' 1 20)"
done

printf '%s\n' "${pids[@]}" >"$ROOT/candidate_pids.txt"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
date -Is >"$ROOT/candidates_finished_at.txt"
exit "$status"

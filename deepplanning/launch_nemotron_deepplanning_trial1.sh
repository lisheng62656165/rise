#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
ANCHOR=nemotron_vanilla_a_trial1
LOG="$ROOT/logs/$ANCHOR"

export NVIDIA_API_KEY="$(awk 'NF' /home/lisheng/.nvidia_api_key_pool_eds_eca | tail -4 | head -1)"
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export HTTP_PROXY='http://127.0.0.1:17897'
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1
mkdir -p "$LOG" "$ROOT/travel_runs/$ANCHOR"

run_shop() {
  local level=$1 workers=$2 count=$3
  PYTHONPATH="$ROOT:$ROOT/shoppingplanning" "$PY" "$ROOT/run_deepplanning_shopping_subset.py" \
    --shopping-root "$ROOT/shoppingplanning" --model "$MODEL" --level "$level" \
    --case-ids $(seq 1 "$count") --run-name "${ANCHOR}_L${level}" \
    --workers "$workers" --max-llm-calls 400 --trial 1 --orchestration-seed 53403 \
    --allow-inference-failures >"$LOG/shopping_L${level}.log" 2>&1 &
  echo $!
}

run_travel() {
  local language=$1 workers=$2
  PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" \
    --travel-root "$ROOT/travelplanning" --model "$MODEL" --language "$language" \
    --workers "$workers" --max-llm-calls 400 --seed 53403 \
    --output-root "$ROOT/travel_runs/$ANCHOR" >"$LOG/travel_${language}.log" 2>&1 &
  echo $!
}

pids=(
  "$(run_shop 1 1 50)"
  "$(run_shop 2 1 50)"
  "$(run_shop 3 2 20)"
  "$(run_travel zh 3)"
  "$(run_travel en 3)"
)
printf '%s\n' "${pids[@]}" >"$LOG/pids.txt"
printf 'started=%s\n' "$(date -Is)" >"$LOG/status.txt"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf 'finished=%s status=%s\n' "$(date -Is)" "$status" >>"$LOG/status.txt"
exit "$status"

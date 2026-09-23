#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}
SHOP=$ROOT/shoppingplanning
OUT=$ROOT/results/deepseek_dp_compare
LOG=$OUT/logs/vanilla
MODEL=deepseek-v4.1-flash

mkdir -p "$LOG" "$OUT/travel_runs"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo 'Set DEEPSEEK_API_KEY before running.' >&2
  exit 2
fi
export DEEPPLANNING_OPENAI_BASE_URL=https://hgapi.dieqiyun.top/v1
export DEEPPLANNING_REQUEST_TIMEOUT=300
export DEEPPLANNING_MAX_RETRIES=3
export DEEPPLANNING_RETRY_BACKOFF=2
if [[ -n "${DEEPPLANNING_HTTP_PROXY:-}" ]]; then
  export HTTP_PROXY="$DEEPPLANNING_HTTP_PROXY"
  export HTTPS_PROXY="$DEEPPLANNING_HTTP_PROXY"
  export ALL_PROXY="$DEEPPLANNING_HTTP_PROXY"
fi
export PYTHONUNBUFFERED=1

run_shop() {
  local level=$1 workers=$2 count=$3
  PYTHONPATH="$ROOT:$SHOP" "$PY" "$ROOT/run_deepplanning_shopping_subset.py" \
    --shopping-root "$SHOP" --model "$MODEL" --level "$level" \
    --case-ids $(seq 1 "$count") --run-name "deepseek_vanilla_L${level}" \
    --workers "$workers" --max-llm-calls 400 --trial 1 \
    --orchestration-seed 53403 --allow-inference-failures \
    >"$LOG/shopping_L${level}.log" 2>&1
}

run_travel() {
  local language=$1 workers=$2
  PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" \
    --travel-root "$ROOT/travelplanning" --model "$MODEL" --language "$language" \
    --workers "$workers" --max-llm-calls 400 --seed 53403 \
    --output-root "$ROOT/travel_runs/deepseek_vanilla" \
    >"$LOG/travel_${language}.log" 2>&1
}

printf 'started=%s model=%s total_workers=100\n' "$(date -Is)" "$MODEL" >"$LOG/status.txt"
pids=()
run_shop 1 18 50 & pids+=("$!")
run_shop 2 18 50 & pids+=("$!")
run_shop 3 8 20 & pids+=("$!")
run_travel zh 28 & pids+=("$!")
run_travel en 28 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf 'finished=%s status=%s\n' "$(date -Is)" "$status" >>"$LOG/status.txt"
exit "$status"

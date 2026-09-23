#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}
SHOP=$ROOT/shoppingplanning
OUT=$ROOT/results/deepseek_dp_compare
MODEL=deepseek-v4.1-flash

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo 'Set DEEPSEEK_API_KEY before running.' >&2
  exit 2
fi
export DEEPPLANNING_OPENAI_BASE_URL=https://hgapi.dieqiyun.top/v1
export DEEPPLANNING_REQUEST_TIMEOUT=300
export DEEPPLANNING_MAX_RETRIES=3
export DEEPPLANNING_RETRY_BACKOFF=2
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

run_candidate() {
  local c=$1
  local seed=$((53403 + c))
  local log="$OUT/logs/best4_c${c}"
  local travel_out="$OUT/best4/c${c}/travel_runs"
  mkdir -p "$log" "$travel_out"
  PYTHONPATH="$ROOT:$SHOP" "$PY" "$ROOT/run_deepplanning_shopping_subset.py" --shopping-root "$SHOP" --model "$MODEL" --level 1 --case-ids $(seq 1 50) --run-name "deepseek_best4_c${c}_L1" --workers 5 --max-llm-calls 400 --trial 1 --orchestration-seed "$seed" --allow-inference-failures >"$log/shopping_L1.log" 2>&1 &
  p1=$!
  PYTHONPATH="$ROOT:$SHOP" "$PY" "$ROOT/run_deepplanning_shopping_subset.py" --shopping-root "$SHOP" --model "$MODEL" --level 2 --case-ids $(seq 1 50) --run-name "deepseek_best4_c${c}_L2" --workers 5 --max-llm-calls 400 --trial 1 --orchestration-seed "$seed" --allow-inference-failures >"$log/shopping_L2.log" 2>&1 &
  p2=$!
  PYTHONPATH="$ROOT:$SHOP" "$PY" "$ROOT/run_deepplanning_shopping_subset.py" --shopping-root "$SHOP" --model "$MODEL" --level 3 --case-ids $(seq 1 20) --run-name "deepseek_best4_c${c}_L3" --workers 3 --max-llm-calls 400 --trial 1 --orchestration-seed "$seed" --allow-inference-failures >"$log/shopping_L3.log" 2>&1 &
  p3=$!
  PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" --travel-root "$ROOT/travelplanning" --model "$MODEL" --language zh --workers 6 --max-llm-calls 400 --seed "$seed" --output-root "$travel_out" >"$log/travel_zh.log" 2>&1 &
  p4=$!
  PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" --travel-root "$ROOT/travelplanning" --model "$MODEL" --language en --workers 6 --max-llm-calls 400 --seed "$seed" --output-root "$travel_out" >"$log/travel_en.log" 2>&1 &
  p5=$!
  local status=0
  for pid in "$p1" "$p2" "$p3" "$p4" "$p5"; do wait "$pid" || status=1; done
  printf 'candidate=%s finished=%s status=%s\n' "$c" "$(date -Is)" "$status" >"$log/status.txt"
  return "$status"
}

mkdir -p "$OUT/logs"
printf 'started=%s candidates=4 total_workers=100\n' "$(date -Is)" >"$OUT/logs/best4_status.txt"
pids=()
for c in 0 1 2 3; do run_candidate "$c" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf 'finished=%s status=%s\n' "$(date -Is)" "$status" >>"$OUT/logs/best4_status.txt"
exit "$status"

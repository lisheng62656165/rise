#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}
OUT=$ROOT/results/deepseek_dp_compare/eds_eca
MODEL=deepseek-v4.1-flash
mkdir -p "$OUT/logs"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo 'Set DEEPSEEK_API_KEY before running.' >&2
  exit 2
fi
export DEEPPLANNING_MODEL="$MODEL"
export DEEPPLANNING_API_MODEL="$MODEL"
export DEEPPLANNING_API_KEY_ENV=DEEPSEEK_API_KEY
export DEEPPLANNING_BASE_URL=https://hgapi.dieqiyun.top/v1
export DEEPPLANNING_OPENAI_BASE_URL=https://hgapi.dieqiyun.top/v1
export DEEPPLANNING_REQUEST_TIMEOUT=300
export DEEPPLANNING_MAX_RETRIES=3
export DEEPPLANNING_RETRY_BACKOFF=2
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

run_cohort() {
  local cohort=$1 workers=$2 extra=()
  if [[ "$cohort" == travel-* ]]; then extra+=(--anchor-model-slug "$MODEL" --deepplanning-adapter); fi
  local domain=shoppingplanning
  [[ "$cohort" == travel-* ]] && domain=travelplanning
  PYTHONPATH="$ROOT:$ROOT/$domain" "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
    --root "$ROOT" --output "$OUT" --cohort "$cohort" --workers "$workers" \
    --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
    --anchor-tag "$ANCHOR_TAG" "${extra[@]}" \
    >"$OUT/logs/${cohort}.log" 2>&1
}

ANCHOR_TAG=deepseek_vanilla
printf 'started=%s model=%s total_workers=100\n' "$(date -Is)" "$MODEL" >"$OUT/logs/status.txt"
run_cohort shopping 34 & p1=$!
run_cohort travel-zh 33 & p2=$!
run_cohort travel-en 33 & p3=$!
status=0
for pid in "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
printf 'finished=%s status=%s\n' "$(date -Is)" "$status" >>"$OUT/logs/status.txt"
exit "$status"

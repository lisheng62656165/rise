#!/usr/bin/env bash
set -u

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_full_seed53403
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
WORKERS=${WORKERS:-10}

export MIMO_API_KEY="$(tr -d '\r\n' </home/lisheng/.mimo_api_key)"
export MIMO_BASE_URL=https://api.xiaomimimo.com/v1
export DEEPPLANNING_OPENAI_BASE_URL="$MIMO_BASE_URL"
export HTTP_PROXY='http://127.0.0.1:17897'
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs"

run_cohort() {
  local cohort=$1
  local domain=shoppingplanning
  [[ "$cohort" == travel-* ]] && domain=travelplanning
  PYTHONPATH="$ROOT:$ROOT/$domain" "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
    --root "$ROOT" --output "$OUT" --cohort "$cohort" \
    --workers "$WORKERS" --max-llm-calls 400 \
    --proposal-seed 64639 --selector-seed 77113 \
    >>"$OUT/logs/${cohort}.log" 2>&1
}

run_cohort shopping
run_cohort travel-zh
run_cohort travel-en

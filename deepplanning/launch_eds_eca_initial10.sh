#!/usr/bin/env bash
set -u

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_full_seed53403
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python

export MIMO_API_KEY="$(tr -d '\r\n' </home/lisheng/.mimo_api_key)"
export MIMO_BASE_URL=https://api.xiaomimimo.com/v1
export DEEPPLANNING_OPENAI_BASE_URL="$MIMO_BASE_URL"
export HTTP_PROXY='http://127.0.0.1:17897'
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs"

PYTHONPATH="$ROOT:$ROOT/shoppingplanning" "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
  --root "$ROOT" --output "$OUT" --cohort shopping --workers 4 \
  --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
  >>"$OUT/logs/shopping.log" 2>&1 &
shopping_pid=$!

PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
  --root "$ROOT" --output "$OUT" --cohort travel-zh --workers 3 \
  --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
  >>"$OUT/logs/travel-zh.log" 2>&1 &
zh_pid=$!

PYTHONPATH="$ROOT:$ROOT/travelplanning" "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" \
  --travel-root "$ROOT/travelplanning" --model mimo-v2.5-pro --language en \
  --workers 3 --max-llm-calls 400 --seed 88001 \
  --output-root "$ROOT/travel_runs/vanilla_a" \
  >>"$OUT/logs/vanilla-en.log" 2>&1 &
en_anchor_pid=$!

printf '%s\n' "$shopping_pid" "$zh_pid" "$en_anchor_pid" >"$OUT/initial10.pids"
wait "$shopping_pid" "$zh_pid" "$en_anchor_pid"

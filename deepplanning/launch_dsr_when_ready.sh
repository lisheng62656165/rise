#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python

while [[ ! -f "$ROOT/candidates_finished_at.txt" ]]; do
  date -Is
  sleep 60
done

count=$(find "$ROOT/shoppingplanning" \
  -path '*/database_run_full_vanilla_a_L*_20260828/case_*/messages.json' \
  -o -path '*/database_run_full_generic_g_L*_20260828/case_*/messages.json' | wc -l)
if [[ "$count" -ne 240 ]]; then
  echo "candidate coverage incomplete: $count/240" >&2
  exit 2
fi

export MIMO_API_KEY="$(tr -d '\r\n' </home/lisheng/.mimo_api_key)"
export MIMO_BASE_URL=https://api.xiaomimimo.com/v1
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONPATH="$ROOT:$ROOT/shoppingplanning"
export PYTHONUNBUFFERED=1

exec "$PY" "$ROOT/run_deepplanning_dsr.py" \
  --root "$ROOT" --workers 20 --max-llm-calls 400

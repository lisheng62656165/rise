#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
PIDS="$OUT/logs/eds_pids.txt"

old=$(sed -n '5p' "$PIDS")
if kill -0 "$old" 2>/dev/null; then
  echo "Refusing to overlap live shard PID $old" >&2
  exit 1
fi

key=$(awk 'NF' /home/lisheng/.nvidia_api_key_pool_eds_eca | sed -n '7p')
[[ -n "$key" ]]

nohup env \
  NVIDIA_API_KEY="$key" \
  DEEPPLANNING_MODEL=nemotron-3.5-lightning-30b-a3b \
  DEEPPLANNING_API_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b \
  DEEPPLANNING_API_KEY_ENV=NVIDIA_API_KEY \
  DEEPPLANNING_BASE_URL=https://integrate.api.nvidia.com/v1 \
  DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1 \
  HTTP_PROXY=http://127.0.0.1:17897 \
  HTTPS_PROXY=http://127.0.0.1:17897 \
  ALL_PROXY=http://127.0.0.1:17897 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$ROOT:$ROOT/shoppingplanning" \
  "$PY" "$ROOT/run_deepplanning_eds_eca.py" \
    --root "$ROOT" --output "$OUT" --cohort shopping --workers 1 \
    --max-llm-calls 400 --proposal-seed 64639 --selector-seed 77113 \
    --anchor-tag nemotron_vanilla_a_trial1_merged \
    --job-shard-index 4 --job-shard-count 8 \
    >>"$OUT/logs/shopping_shard_4.log" 2>&1 &
new=$!

awk -v n=5 -v p="$new" 'NR == n { print p; next } { print }' "$PIDS" >"$PIDS.tmp"
mv "$PIDS.tmp" "$PIDS"
printf '%s replaced shopping shard=4 old_pid=%s new_pid=%s key_index=7\n' \
  "$(date -Is)" "$old" "$new" >>"$OUT/logs/recovery.log"

sleep 3
kill -0 "$new"
echo "RELAUNCHED old=$old new=$new"

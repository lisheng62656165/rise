#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
SHOP="$ROOT/shoppingplanning"
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
POOL=/home/lisheng/.nvidia_api_key_pool_eds_eca
LOG="$ROOT/logs/nemotron_vanilla_a_trial1/shopping_13shards"
mkdir -p "$LOG"

ids=(45 47 48 50)
key_indexes=(1 2 3 4)
for position in "${!ids[@]}"; do
  case_id=${ids[$position]}
  key_index=${key_indexes[$position]}
  session="dp_shop_retry_L2_$case_id"
  run_name="nemotron_vanilla_a_trial1r9retry${position}_L2"
  if tmux has-session -t "$session" 2>/dev/null; then
    continue
  fi
  key=$(awk 'NF' "$POOL" | sed -n "${key_index}p")
  [[ -n "$key" ]] || { echo "Missing API key index $key_index" >&2; exit 1; }
  printf -v command '%q ' env \
    NVIDIA_API_KEY="$key" \
    DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1 \
    HTTP_PROXY=http://127.0.0.1:17897 \
    HTTPS_PROXY=http://127.0.0.1:17897 \
    ALL_PROXY=http://127.0.0.1:17897 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$ROOT:$SHOP" \
    "$PY" "$ROOT/run_deepplanning_shopping_subset.py" \
    --shopping-root "$SHOP" \
    --model nemotron-3.5-lightning-30b-a3b \
    --level 2 \
    --case-ids "$case_id" \
    --run-name "$run_name" \
    --workers 1 \
    --max-llm-calls 400 \
    --trial 1 \
    --orchestration-seed 53403 \
    --allow-inference-failures
  command+=" >$(printf '%q' "$LOG/L2_retry_case${case_id}.log") 2>&1"
  tmux new-session -d -s "$session" "$command"
  printf '%s id=%s key_index=%s\n' "$session" "$case_id" "$key_index"
done

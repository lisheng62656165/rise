#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
ANCHOR=nemotron_vanilla_a_trial1
POOL=/home/lisheng/.nvidia_api_key_pool_eds_eca
BASE="$ROOT/travel_runs/$ANCHOR"
OUTPUT="$BASE/${MODEL}_en"
LOG="$ROOT/logs/$ANCHOR/tmux_recovery"
EXCLUDE_IDS=${EXCLUDE_IDS:-76}
KEY_INDEXES=(1 2 3 4 5)

mkdir -p "$LOG"

is_excluded() {
  local candidate=$1 excluded
  IFS=',' read -ra excluded <<< "$EXCLUDE_IDS"
  for excluded in "${excluded[@]}"; do
    [[ "$candidate" == "$excluded" ]] && return 0
  done
  return 1
}

missing=()
for id in $(seq 0 119); do
  [[ -s "$OUTPUT/reports/id_${id}.txt" ]] && continue
  is_excluded "$id" || missing+=("$id")
done

printf 'missing_without_exclusions=%s\n' "${missing[*]}"
printf 'excluded=%s\n' "$EXCLUDE_IDS"

for shard in $(seq 0 4); do
  csv=""
  for ((position=shard; position<${#missing[@]}; position+=5)); do
    csv+="${csv:+,}${missing[$position]}"
  done
  [[ -n "$csv" ]] || continue

  session="dp_en_recovery_${shard}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session"
    continue
  fi

  key_index=${KEY_INDEXES[$shard]}
  key=$(awk 'NF' "$POOL" | sed -n "${key_index}p")
  [[ -n "$key" ]] || { echo "Missing API key index $key_index" >&2; exit 1; }

  printf -v command '%q ' env \
    NVIDIA_API_KEY="$key" \
    DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1 \
    HTTP_PROXY=http://127.0.0.1:17897 \
    HTTPS_PROXY=http://127.0.0.1:17897 \
    ALL_PROXY=http://127.0.0.1:17897 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$ROOT:$ROOT/travelplanning" \
    "$PY" "$ROOT/run_deepplanning_travel_inference_only.py" \
    --travel-root "$ROOT/travelplanning" \
    --model "$MODEL" \
    --language en \
    --workers 1 \
    --max-llm-calls 400 \
    --seed 53403 \
    --rerun-ids "$csv" \
    --output-root "$BASE"
  command+=" >$(printf '%q' "$LOG/shard_${shard}.log") 2>&1"

  tmux new-session -d -s "$session" "$command"
  printf '%s shard=%s ids=%s key_index=%s\n' "$session" "$shard" "$csv" "$key_index" \
    | tee -a "$LOG/sessions.tsv"
done

tmux list-sessions -F '#{session_name} #{session_created_string}' \
  | grep '^dp_en_recovery_' || true

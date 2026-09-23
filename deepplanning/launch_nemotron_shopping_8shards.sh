#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
SHOP="$ROOT/shoppingplanning"
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
POOL=/home/lisheng/.nvidia_api_key_pool_eds_eca
LOG="$ROOT/logs/nemotron_vanilla_a_trial1/shopping_13shards"

mkdir -p "$LOG"

launch_level() {
  local level=$1 shard_count=$2
  shift 2
  local -a key_indexes=("$@")
  local count=50
  [[ "$level" == 3 ]] && count=20
  local manifest="$SHOP/result_report/database_nemotron_vanilla_a_trial1r1_L${level}/subset_manifest.json"

  mapfile -t valid < <(
    "$PY" -c \
      'import json,sys; print(*json.load(open(sys.argv[1]))["execution_valid_case_ids"], sep="\n")' \
      "$manifest"
  )
  declare -A completed=()
  for id in "${valid[@]}"; do completed[$id]=1; done
  missing=()
  for id in $(seq 1 "$count"); do
    [[ -n "${completed[$id]:-}" ]] || missing+=("$id")
  done
  unset completed

  local shard position csv session key_index key run_name command
  for shard in $(seq 0 $((shard_count - 1))); do
    csv=()
    for ((position=shard; position<${#missing[@]}; position+=shard_count)); do
      csv+=("${missing[$position]}")
    done
    [[ ${#csv[@]} -gt 0 ]] || continue
    session="dp_shop_L${level}_shard${shard}"
    run_name="nemotron_vanilla_a_trial1r4s${shard}_L${level}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "session already exists: $session"
      continue
    fi
    if [[ -e "$SHOP/database_run_${run_name}" || -e "$SHOP/database_infered/database_${run_name}" ]]; then
      echo "run output already exists: $run_name" >&2
      continue
    fi

    key_index=${key_indexes[$shard]}
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
      --model "$MODEL" \
      --level "$level" \
      --case-ids "${csv[@]}" \
      --run-name "$run_name" \
      --workers 1 \
      --max-llm-calls 400 \
      --trial 1 \
      --orchestration-seed 53403 \
      --allow-inference-failures
    command+=" >$(printf '%q' "$LOG/L${level}_shard${shard}.log") 2>&1"
    tmux new-session -d -s "$session" "$command"
    printf '%s level=%s ids=%s key_index=%s\n' \
      "$session" "$level" "${csv[*]}" "$key_index" | tee -a "$LOG/sessions.tsv"
  done
}

launch_level 1 5 1 2 3 4 5
launch_level 2 5 6 9 10 1 2
launch_level 3 3 3 4 5

tmux list-sessions -F '#{session_name} #{session_created_string}' \
  | grep '^dp_shop_L' || true

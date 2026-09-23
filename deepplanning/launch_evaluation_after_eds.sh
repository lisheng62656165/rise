#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
SHOP="$ROOT/shoppingplanning"
TRAVEL="$ROOT/travelplanning"
MODEL=nemotron-3.5-lightning-30b-a3b
POOL=/home/lisheng/.nvidia_api_key_pool_stride10
LOG="$OUT/logs/evaluation"

mkdir -p "$LOG"

count_files() { { find "$1" -maxdepth 1 -type f -name "$2" 2>/dev/null || true; } | wc -l; }
while true; do
  shopping=$(count_files "$OUT/shopping/tasks" '*.json')
  zh=$(count_files "$OUT/travel_zh/tasks" '*.json')
  en=$(count_files "$OUT/travel_en/tasks" '*.json')
  printf '%s StateTrace tasks shopping=%s zh=%s en=%s\n' "$(date -Is)" "$shopping" "$zh" "$en" >>"$LOG/wait.log"
  [[ "$shopping" == 120 && "$zh" == 120 && "$en" == 120 ]] && break
  sleep 120
done

"$PY" "$ROOT/prepare_deepplanning_evaluation_views.py" \
  --root "$ROOT" --method-output "$OUT" >"$LOG/prepare_views.log" 2>&1

"$PY" "$SHOP/evaluation/evaluation_pipeline.py" \
  --database_dir "$OUT/evaluation_views/shopping_vanilla" \
  --output_dir nemotron_trial1_vanilla >"$LOG/shopping_vanilla.log" 2>&1 &
shop_vanilla_pid=$!
"$PY" "$SHOP/evaluation/evaluation_pipeline.py" \
  --database_dir "$OUT/evaluation_views/shopping_statetrace" \
  --output_dir nemotron_trial1_statetrace >"$LOG/shopping_statetrace.log" 2>&1 &
shop_method_pid=$!

export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export TRAVEL_CONVERSION_MODEL="$MODEL"
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

conversion_dirs=(
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_zh"
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_en"
  "$OUT/travel_zh/final"
  "$OUT/travel_en/final"
)
conversion_langs=(zh en zh en)
conversion_names=(vanilla_zh vanilla_en statetrace_zh statetrace_en)
count_converted() {
  count_files "$1/converted_plans" 'id_*_converted.json'
}

conversion_round=0
key_count=$(awk 'NF' "$POOL" | wc -l)
while true; do
  conversion_complete=true
  for result_dir in "${conversion_dirs[@]}"; do
    [[ $(count_converted "$result_dir") -eq 120 ]] || conversion_complete=false
  done
  $conversion_complete && break

  conversion_round=$((conversion_round + 1))
  conversion_pids=()
  for index in 0 1 2 3; do
    key_index=$(( ((conversion_round - 1) * 4 + index) % key_count + 1 ))
    key=$(awk 'NF' "$POOL" | sed -n "${key_index}p")
    env NVIDIA_API_KEY="$key" PYTHONPATH="$ROOT:$TRAVEL" \
      "$PY" "$ROOT/run_deepplanning_travel_conversion.py" \
        --travel-root "$TRAVEL" --result-dir "${conversion_dirs[$index]}" \
        --language "${conversion_langs[$index]}" --workers 5 --seed 53403 \
        --max-tokens 8192 --max-parse-retries 5 --request-timeout 120 --allow-partial \
        >"$LOG/convert_${conversion_names[$index]}_round_${conversion_round}.log" 2>&1 &
    conversion_pids+=("$!")
  done
  for pid in "${conversion_pids[@]}"; do wait "$pid" || true; done
  printf '%s conversion_round=%d counts=%s/%s/%s/%s\n' \
    "$(date -Is)" "$conversion_round" \
    "$(count_converted "${conversion_dirs[0]}")" \
    "$(count_converted "${conversion_dirs[1]}")" \
    "$(count_converted "${conversion_dirs[2]}")" \
    "$(count_converted "${conversion_dirs[3]}")" >>"$LOG/conversion_progress.log"
done

wait "$shop_vanilla_pid"
wait "$shop_method_pid"

evaluation_pids=()
for index in 0 1 2 3; do
  "$PY" "$ROOT/run_deepplanning_travel_official_evaluation.py" \
    --travel-root "$TRAVEL" --result-dir "${conversion_dirs[$index]}" \
    --language "${conversion_langs[$index]}" --workers 10 \
    >"$LOG/eval_${conversion_names[$index]}.log" 2>&1 &
  evaluation_pids+=("$!")
done
for pid in "${evaluation_pids[@]}"; do wait "$pid"; done

"$PY" "$ROOT/summarize_deepplanning_trial.py" \
  --root "$ROOT" --method-output "$OUT" >"$LOG/summary.log" 2>&1

printf 'completed=%s\n' "$(date -Is)" >"$LOG/status.txt"

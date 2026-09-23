#!/usr/bin/env bash
set -u

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
BASE="$ROOT/travel_runs/nemotron_vanilla_a_trial1"
SHOP_LOG="$ROOT/logs/nemotron_vanilla_a_trial1/shopping_13shards"

printf 'TIME=%s\n' "$(date +%F_%T)"
for language in en zh; do
  result="$BASE/nemotron-3.5-lightning-30b-a3b_$language"
  valid=$(find "$result/reports" -maxdepth 1 -type f -size +199c 2>/dev/null | wc -l)
  converted=$(find "$result/converted_plans" -maxdepth 1 -type f -size +10c 2>/dev/null | wc -l)
  missing=()
  for task_id in $(seq 0 119); do
    report="$result/reports/id_$task_id.txt"
    if [[ ! -f "$report" ]] || (( $(stat -c%s "$report") < 200 )); then
      missing+=("$task_id")
    fi
  done
  printf 'TRAVEL_%s_VALID=%s CONVERTED=%s MISSING=%s\n' \
    "${language^^}" "$valid" "$converted" "${missing[*]:-none}"
done

shopping_l1=$(find "$ROOT/shoppingplanning/database_infered/database_nemotron_vanilla_a_trial1_merged_L1" -type f -name messages.json 2>/dev/null | wc -l)
shopping_l2=$(find "$ROOT/shoppingplanning/database_infered/database_nemotron_vanilla_a_trial1_merged_L2" -type f -name messages.json 2>/dev/null | wc -l)
shopping_l3=$(find "$ROOT/shoppingplanning/database_infered/database_nemotron_vanilla_a_trial1_merged_L3" -type f -name messages.json 2>/dev/null | wc -l)
shopping_active=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^dp_shop_L' || true)
travel_active=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^dp_en' || true)
convert_active=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^dp_convert' || true)
eds_sessions=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^dp_eds_' || true)
manifests=$(find "$ROOT/shoppingplanning/result_report" -type f -name subset_manifest.json -path '*nemotron_vanilla_a_trial1r4*' 2>/dev/null | wc -l)
recent_shop_logs=$(find "$SHOP_LOG" -type f -mmin -5 2>/dev/null | wc -l)
eds_root=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
eds_pid_file="$eds_root/logs/eds_pids.txt"
eds_pid_count=0
eds_alive=0
if [[ -f "$eds_pid_file" ]]; then
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    eds_pid_count=$((eds_pid_count + 1))
    kill -0 "$pid" 2>/dev/null && eds_alive=$((eds_alive + 1))
  done <"$eds_pid_file"
fi

printf 'SHOP_ANCHORS_L1=%s L2=%s L3=%s ACTIVE=%s MANIFESTS=%s\n' \
  "$shopping_l1" "$shopping_l2" "$shopping_l3" "$shopping_active" "$manifests"
printf 'TRAVEL_ACTIVE=%s CONVERT_ACTIVE=%s EDS_SESSIONS=%s\n' "$travel_active" "$convert_active" "$eds_sessions"
printf 'EDS_PIDS=%s ALIVE=%s OUTPUTS_SHOP=%s OUTPUTS_ZH=%s OUTPUTS_EN=%s\n' \
  "$eds_pid_count" "$eds_alive" \
  "$(find "$eds_root/shopping/tasks" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)" \
  "$(find "$eds_root/travel_zh/tasks" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)" \
  "$(find "$eds_root/travel_en/tasks" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
for cohort in shopping travel_zh travel_en; do
  printf 'EDS_STAGE_%s_B=%s C=%s D=%s ERRORS=%s\n' "${cohort^^}" \
    "$(find "$eds_root/$cohort/checkpoints" -maxdepth 1 -type f -name '*_B.json' 2>/dev/null | wc -l)" \
    "$(find "$eds_root/$cohort/checkpoints" -maxdepth 1 -type f -name '*_C.json' 2>/dev/null | wc -l)" \
    "$(find "$eds_root/$cohort/checkpoints" -maxdepth 1 -type f -name '*_D.json' 2>/dev/null | wc -l)" \
    "$(find "$eds_root/$cohort/errors" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
done
printf 'RECENT_SHOP_LOGS=%s ACTIVE_NAMES=' "$recent_shop_logs"
tmux list-sessions -F '#{session_name}' 2>/dev/null \
  | grep -E '^dp_(shop_L|en|zh)' \
  | sort \
  | tr '\n' ','
printf '\n'
now=$(date +%s)
for session in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^dp_shop_L' | sort); do
  log_name=${session#dp_shop_}.log
  log="$SHOP_LOG/$log_name"
  if [[ -f "$log" ]]; then
    age=$((now - $(stat -c%Y "$log")))
    printf 'SHOP_LOG_AGE %s=%ss\n' "$session" "$age"
  fi
done

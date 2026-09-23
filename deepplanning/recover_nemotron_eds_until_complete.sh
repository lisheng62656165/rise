#!/usr/bin/env bash
set -u

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
LOG="$OUT/logs"
LAUNCHER="$ROOT/launch_nemotron_eds_after_anchor.sh"
PIDS="$LOG/eds_pids.txt"
RECOVERY_LOG="$LOG/recovery.log"

count_tasks() {
  find "$OUT/$1/tasks" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

alive_count() {
  local alive=0 pid
  if [[ -f "$PIDS" ]]; then
    while read -r pid; do
      kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done <"$PIDS"
  fi
  printf '%s' "$alive"
}

mkdir -p "$LOG"
round=0
while true; do
  shop=$(count_tasks shopping)
  zh=$(count_tasks travel_zh)
  en=$(count_tasks travel_en)
  alive=$(alive_count)
  printf '%s recovery round=%s tasks=%s/%s/%s alive=%s\n' \
    "$(date -Is)" "$round" "$shop" "$zh" "$en" "$alive" >>"$RECOVERY_LOG"

  if [[ "$shop" -eq 120 && "$zh" -eq 120 && "$en" -eq 120 ]]; then
    exit 0
  fi
  if [[ "$alive" -gt 0 ]]; then
    sleep 120
    continue
  fi

  round=$((round + 1))
  printf '%s launching recovery round=%s\n' "$(date -Is)" "$round" >>"$RECOVERY_LOG"
  bash "$LAUNCHER" >>"$RECOVERY_LOG" 2>&1 || true
  sleep 10
done

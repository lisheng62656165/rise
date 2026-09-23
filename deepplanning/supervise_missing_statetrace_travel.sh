#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
LAUNCHER="$ROOT/launch_missing_statetrace_travel_20.sh"
PID_FILE="$OUT/logs/missing_travel_20/pids.tsv"
SUPERVISOR_LOG="$OUT/logs/missing_travel_supervisor.log"

count_reports() {
  find "$OUT/travel_$1/final/reports" -maxdepth 1 -type f -name 'id_*.txt' | wc -l
}

wait_for_round() {
  local alive pid
  while true; do
    alive=0
    while IFS=$'\t' read -r pid _; do
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done < "$PID_FILE"
    printf '%s alive=%d zh=%s en=%s\n' \
      "$(date --iso-8601=seconds)" "$alive" "$(count_reports zh)" "$(count_reports en)" \
      >> "$SUPERVISOR_LOG"
    ((alive == 0)) && return
    sleep 60
  done
}

round=2
while true; do
  wait_for_round
  zh=$(count_reports zh)
  en=$(count_reports en)
  if ((zh == 120 && en == 120)); then
    printf '%s complete zh=%d en=%d\n' "$(date --iso-8601=seconds)" "$zh" "$en" \
      >> "$SUPERVISOR_LOG"
    exit 0
  fi
  round=$((round + 1))
  printf '%s launching_round=%d zh=%d en=%d\n' \
    "$(date --iso-8601=seconds)" "$round" "$zh" "$en" >> "$SUPERVISOR_LOG"
  DEEPPLANNING_KEY_OFFSET=$((round - 2)) bash "$LAUNCHER" \
    >> "$OUT/logs/launch_missing_travel_round_${round}.out" 2>&1
done

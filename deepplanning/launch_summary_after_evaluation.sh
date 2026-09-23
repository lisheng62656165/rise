#!/usr/bin/env bash
set -u

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
LOG="$OUT/logs/evaluation"

summaries=(
  "$ROOT/shoppingplanning/result_report/nemotron_trial1_vanilla/summary_report.json"
  "$ROOT/shoppingplanning/result_report/nemotron_trial1_statetrace/summary_report.json"
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_zh/evaluation/evaluation_summary.json"
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_en/evaluation/evaluation_summary.json"
  "$OUT/travel_zh/final/evaluation/evaluation_summary.json"
  "$OUT/travel_en/final/evaluation/evaluation_summary.json"
)

mkdir -p "$LOG"
while true; do
  ready=0
  for path in "${summaries[@]}"; do
    [[ -s "$path" ]] && ready=$((ready + 1))
  done
  printf '%s official summaries=%s/6\n' "$(date -Is)" "$ready" >>"$LOG/summary_wait.log"
  [[ "$ready" -eq 6 ]] && break
  sleep 120
done

"$PY" "$ROOT/summarize_deepplanning_trial.py" \
  --root "$ROOT" --method-output "$OUT" >"$LOG/summary.log" 2>&1

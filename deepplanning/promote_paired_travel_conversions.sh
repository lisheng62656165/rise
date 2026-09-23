#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PAIR="$OUT/evaluation_views/paired_partial_fast_v1"
PAIR_OLD="$OUT/evaluation_views/paired_partial_20260911"
MODEL=nemotron-3.5-lightning-30b-a3b

promote() {
  local source=$1 target=$2 promoted=0 skipped=0 converted report_name
  mkdir -p "$target/converted_plans"
  for converted in "$source"/converted_plans/id_*_converted.json; do
    [[ -f "$converted" ]] || continue
    report_name=$(basename "$converted" _converted.json).txt
    if [[ -f "$source/reports/$report_name" ]] \
      && [[ -f "$target/reports/$report_name" ]] \
      && cmp -s "$source/reports/$report_name" "$target/reports/$report_name"; then
      cp -n "$converted" "$target/converted_plans/"
      promoted=$((promoted + 1))
    else
      skipped=$((skipped + 1))
    fi
  done
  printf '%s -> %s promoted=%d skipped=%d total=%d\n' \
    "$source" "$target" "$promoted" "$skipped" \
    "$(find "$target/converted_plans" -maxdepth 1 -type f -name 'id_*_converted.json' | wc -l)"
}

promote "$PAIR/zh_vanilla" \
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_zh"
promote "$PAIR/en_vanilla" \
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_en"
promote "$PAIR/zh_statetrace" "$OUT/travel_zh/final"
promote "$PAIR/en_statetrace" "$OUT/travel_en/final"

promote "$PAIR_OLD/zh_vanilla" \
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_zh"
promote "$PAIR_OLD/en_vanilla" \
  "$ROOT/travel_runs/nemotron_vanilla_a_trial1/${MODEL}_en"
promote "$PAIR_OLD/zh_statetrace" "$OUT/travel_zh/final"
promote "$PAIR_OLD/en_statetrace" "$OUT/travel_en/final"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
TRAVEL=$ROOT/travelplanning
WRAPPER=$ROOT/run_deepplanning_travel_inference_only.py
DSR=$ROOT/run_travel_dsr.py

while [[ ! -f "$ROOT/run_summary.json" ]]; do
  date -Is
  sleep 60
done

export MIMO_API_KEY="$(tr -d '\r\n' </home/lisheng/.mimo_api_key)"
export DEEPPLANNING_OPENAI_BASE_URL=https://api.xiaomimimo.com/v1
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONPATH="$ROOT:$TRAVEL"
export PYTHONUNBUFFERED=1
export TRAVEL_CONVERSION_MODEL=mimo-v2.5-pro

mkdir -p "$ROOT/logs/travel"

run_candidates() {
  local lang=$1
  local arm=$2
  local seed=$3
  local out="$ROOT/travel_runs/$arm"
  "$PY" "$WRAPPER" --travel-root "$TRAVEL" --model mimo-v2.5-pro \
    --language "$lang" --workers 5 --max-llm-calls 400 --seed "$seed" \
    --output-root "$out" >"$ROOT/logs/travel/${arm}_${lang}.log" 2>&1
}

for lang in zh en; do
  run_candidates "$lang" vanilla_a 88001 &
  run_candidates "$lang" generic_g 88002 &
  wait

  A="$ROOT/travel_runs/vanilla_a/mimo-v2.5-pro_${lang}"
  G="$ROOT/travel_runs/generic_g/mimo-v2.5-pro_${lang}"
  OUT="$ROOT/travel_dsr/$lang"
  "$PY" "$DSR" --language "$lang" --a "$A" --g "$G" --out "$OUT" \
    --workers 20 --max-llm-calls 400 >"$ROOT/logs/travel/dsr_${lang}.log" 2>&1

  for label in baseline dsr; do
    if [[ "$label" == baseline ]]; then
      SRC="$A"
    else
      SRC="$OUT/final"
    fi
    "$PY" -c "import sys; from pathlib import Path; sys.path.insert(0, '$TRAVEL'); from evaluation.convert_report import convert_reports; print(convert_reports(result_dir=Path('$SRC'), language='$lang', workers=20, skip_existing=True, verbose=False))" \
      >"$ROOT/logs/travel/convert_${label}_${lang}.log" 2>&1
    "$PY" "$ROOT/run_deepplanning_travel_official_evaluator.py" \
      --travel-root "$TRAVEL" --result-dir "$SRC" --language "$lang" --workers 20 \
      >"$ROOT/logs/travel/eval_${label}_${lang}.json" 2>&1
  done
done

date -Is >"$ROOT/travel_finished_at.txt"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MAIN="$ROOT/travel_runs/nemotron_vanilla_a_trial1/nemotron-3.5-lightning-30b-a3b_en"
suffix=${1:-alt}
key_index=${2:-1}
ALT_ROOT="$ROOT/travel_runs/nemotron_vanilla_a_trial1_retry93_$suffix"
ALT="$ALT_ROOT/nemotron-3.5-lightning-30b-a3b_en"
LOG="$ROOT/logs/nemotron_vanilla_a_trial1/tmux_recovery/retry93_$suffix.log"
exec >"$LOG" 2>&1

export NVIDIA_API_KEY="$(sed -n "${key_index}p" /home/lisheng/.nvidia_api_key_pool_eds_eca)"
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/travelplanning"

"$PY" "$ROOT/run_deepplanning_travel_inference_only.py" \
  --travel-root "$ROOT/travelplanning" \
  --model nemotron-3.5-lightning-30b-a3b \
  --language en \
  --workers 1 \
  --max-llm-calls 400 \
  --seed 53403 \
  --rerun-ids 93 \
  --output-root "$ALT_ROOT"

report="$ALT/reports/id_93.txt"
trajectory="$ALT/trajectories/id_93.json"
if [[ -f "$report" ]] && (( $(stat -c%s "$report") >= 200 )) && [[ -s "$trajectory" ]]; then
  mkdir -p "$MAIN/reports" "$MAIN/trajectories"
  cp --no-clobber "$report" "$MAIN/reports/id_93.txt"
  cp --no-clobber "$trajectory" "$MAIN/trajectories/id_93.json"
  echo "Published valid id_93 only if the main result was still absent."
else
  echo "Alternate id_93 run did not produce a substantive report." >&2
  exit 2
fi

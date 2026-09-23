#!/usr/bin/env bash
set -u

ROOT=/home/lisheng/project/ICLR/experiment/deepplanning_dsr_full_20260828
OUT=/home/lisheng/project/ICLR/experiment/deepplanning_eds_eca_dp_nemotron_trial1
PAIR="$OUT/evaluation_views/paired_partial_fast_v1"
LOG="$OUT/logs/paired_partial_controller"
PY=/home/lisheng/project/ICLR/.venv-frd/bin/python
MODEL=nemotron-3.5-lightning-30b-a3b
POOL=/home/lisheng/.nvidia_api_key_pool_eds_eca

mkdir -p "$LOG"
export DEEPPLANNING_OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export TRAVEL_CONVERSION_MODEL="$MODEL"
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export PYTHONUNBUFFERED=1

count_converted() {
  find "$PAIR/$1/converted_plans" -maxdepth 1 -type f -name 'id_*_converted.json' 2>/dev/null | wc -l
}

targets_complete() {
  [[ $(count_converted zh_vanilla) -eq 93 ]] &&
  [[ $(count_converted zh_statetrace) -eq 93 ]] &&
  [[ $(count_converted en_vanilla) -eq 94 ]] &&
  [[ $(count_converted en_statetrace) -eq 94 ]]
}

round=0
while ! targets_complete; do
  round=$((round + 1))
  round_dir="$LOG/round_$round"
  mkdir -p "$round_dir"
  pids=()
  offset=$(( (round - 1) * 4 ))
  slot=0
  for lang in zh en; do
    for variant in vanilla statetrace; do
      slot=$((slot + 1))
      key_count=$(awk 'NF' "$POOL" | wc -l)
      key_index=$(( (offset + slot - 1) % key_count + 1 ))
      key=$(awk 'NF' "$POOL" | sed -n "${key_index}p")
      nohup env NVIDIA_API_KEY="$key" PYTHONPATH="$ROOT:$ROOT/travelplanning" \
        "$PY" "$ROOT/run_deepplanning_travel_conversion.py" \
          --travel-root "$ROOT/travelplanning" \
          --result-dir "$PAIR/${lang}_${variant}" --language "$lang" \
          --workers 10 --seed 53403 --max-tokens 4096 \
          --max-parse-retries 5 --request-timeout 60 --allow-partial \
          >"$round_dir/${lang}_${variant}.log" 2>&1 &
      pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  printf '%s round=%s counts=%s/%s/%s/%s\n' "$(date -Is)" "$round" \
    "$(count_converted zh_vanilla)" "$(count_converted zh_statetrace)" \
    "$(count_converted en_vanilla)" "$(count_converted en_statetrace)" \
    >>"$LOG/progress.log"
  sleep 5
done

"$PY" - "$ROOT" "$OUT" "$PAIR" <<'PY'
import json
import sys
from pathlib import Path

root, out, pair = map(Path, sys.argv[1:])
for lang, count in (("zh", 93), ("en", 94)):
    ids = sorted(
        int(path.stem.removeprefix("id_"))
        for path in (out / f"travel_{lang}" / "final" / "reports").glob("id_*.txt")
    )
    assert len(ids) == count
    source = json.loads((root / "travelplanning" / "data" / f"travelplanning_query_{lang}.json").read_text(encoding="utf-8"))
    subset = [row for row in source if int(row["id"]) in set(ids)]
    assert len(subset) == count
    (pair / f"test_{lang}.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
PY

eval_pids=()
for lang in zh en; do
  for variant in vanilla statetrace; do
    "$PY" "$ROOT/run_deepplanning_travel_official_evaluator.py" \
      --travel-root "$ROOT/travelplanning" --result-dir "$PAIR/${lang}_${variant}" \
      --test-data "$PAIR/test_${lang}.json" \
      --database-dir "$ROOT/travelplanning/database/database_${lang}" --workers 10 \
      >"$LOG/eval_${lang}_${variant}.log" 2>&1 &
    eval_pids+=("$!")
  done
done
for pid in "${eval_pids[@]}"; do wait "$pid"; done

"$PY" - "$PAIR" <<'PY'
import json
import sys
from pathlib import Path

pair = Path(sys.argv[1])
result = {}
for lang in ("zh", "en"):
    result[lang] = {}
    for variant in ("vanilla", "statetrace"):
        path = pair / f"{lang}_{variant}" / "evaluation" / "evaluation_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[lang][variant] = {
            "total_test_samples": payload["total_test_samples"],
            "plan_files_found": payload["plan_files_found"],
            **payload["metrics"],
        }
    result[lang]["delta"] = {
        key: result[lang]["statetrace"][key] - result[lang]["vanilla"][key]
        for key in ("delivery_rate", "commonsense_score", "personalized_score", "composite_score", "case_acc")
    }
(pair / "paired_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

date -Is >"$LOG/complete.txt"

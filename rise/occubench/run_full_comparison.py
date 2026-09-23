#!/usr/bin/env python3
"""Run the complete 382-task OccuBench comparison from this directory.

The script keeps all credentials in the process environment. It runs:
1) StateBench-EDS-ECA Original (A -> B -> C -> D),
2) OAgents Best-of-4 using the same A source,
3) paired-score the EDS-ECA A anchor (Vanilla) and accepted FINAL,
4) score deferred Best-of-4 outputs and write category aggregates.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print(">", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def completed_task_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(int(json.loads(line)["task_id"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return ids


def run_eds_until_complete(command: list[str], result_path: Path, target: int) -> None:
    for _ in range(5):
        if len(completed_task_ids(result_path)) >= target:
            return
        run(command)
    missing = target - len(completed_task_ids(result_path))
    raise RuntimeError(f"EDS-ECA still has {missing} missing tasks; rerun this command to resume.")


def run_oagents_until_complete(command: list[str], out: Path, target: int) -> None:
    for _ in range(5):
        if len(list(out.glob("*/result.json"))) >= target:
            return
        run(command)
    missing = target - len(list(out.glob("*/result.json")))
    raise RuntimeError(f"OAgents Best-of-4 still has {missing} missing tasks; rerun this command to resume.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--selector-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=53403,
                        help="Cohort and agent rollout seed.")
    parser.add_argument("--selector-seed", type=int, default=77113,
                        help="Selector candidate-order/API seed.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=382)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "full_comparison")
    args = parser.parse_args()
    if not os.environ.get(args.api_key_env):
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    eds = output / "eds_eca"
    oagents = output / "oagents_bon4"
    model_args = [
        "--agent-model", args.model, "--world-model", args.model,
        "--verifier-model", args.model, "--agent-base-url", args.base_url,
        "--world-base-url", args.base_url, "--verifier-base-url", args.base_url,
        "--agent-api-key-env", args.api_key_env, "--world-api-key-env", args.api_key_env,
        "--verifier-api-key-env", args.api_key_env,
    ]
    eds_command = [
        sys.executable, "run_occubench_eds_eca_mimo.py",
        "--dataset-root", str(ROOT), "--output-dir", str(eds),
        "--run-id", "statebench-eds-eca-original",
        "--seed", str(args.seed), "--selector-seed", str(args.selector_seed),
        "--selector-model", args.model, "--selector-base-url", args.base_url,
        "--selector-api-key-env", args.api_key_env,
        "--selector-policy", "statebench-eds-eca", "--selector-design", "original",
        "--workers", str(args.workers), "--selector-active-requests", str(args.selector_workers),
        "--max-steps", str(args.max_steps), "--max-tokens", str(args.max_tokens),
        "--verifier-votes", "1", "--limit", str(args.limit), "--resume",
        *model_args,
    ]
    eds_dir = eds / "statebench-eds-eca-original"
    run_eds_until_complete(eds_command, eds_dir / "results.jsonl", args.limit)
    paired = eds_dir / f"paired_{args.limit}"
    paired_command = [
        sys.executable, "score_occubench_paired.py",
        "--run-dir", str(eds_dir), "--dataset-root", str(ROOT),
        "--model", args.model, "--base-url", args.base_url,
        "--api-key-env", args.api_key_env, "--workers", str(args.workers),
        "--count", str(args.limit), "--non-stream",
    ]
    for _ in range(5):
        run(paired_command)
        summary_path = paired / "summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("paired_valid") == args.limit:
            break
    else:
        raise RuntimeError("EDS A/FINAL still has invalid verifier labels; rerun to resume scoring.")
    cohort_ids = [
        int(row["task_id"]) for row in
        (json.loads(line) for line in (paired / "cohort.jsonl").read_text(encoding="utf-8").splitlines())
    ]
    task_ids_file = output / "cohort_task_ids.txt"
    task_ids_file.parent.mkdir(parents=True, exist_ok=True)
    task_ids_file.write_text("\n".join(str(tid) for tid in cohort_ids) + "\n", encoding="utf-8")
    oagents_command = [
        sys.executable, "run_occubench_bestof4.py",
        "--source", str(eds_dir), "--out", str(oagents),
        "--dataset-root", str(ROOT), "--base-url", args.base_url,
        "--api-key-env", args.api_key_env, "--workers", str(args.workers),
        "--active-requests", str(args.selector_workers), "--count", str(args.limit),
        "--seed", str(args.seed),
        "--match-source-config", "--selector-policy", "oagents",
        "--short-initial-context", "0",
        "--task-ids-file", str(task_ids_file),
        "--scores-dir", str(paired), "--allow-missing-scores", "--defer-verification",
    ]
    run_oagents_until_complete(oagents_command, oagents, args.limit)
    scoring_command = [
        sys.executable, "score_bestof4.py",
        "--run-dir", str(oagents), "--dataset-root", str(ROOT),
        "--source-dir", str(eds_dir),
        "--model", args.model, "--base-url", args.base_url,
        "--api-key-env", args.api_key_env, "--workers", str(args.workers),
    ]
    for _ in range(4):
        run(scoring_command)
        valid = sum(
            json.loads(path.read_text(encoding="utf-8")).get("bestof4", {}).get("verification_valid") is True
            for path in oagents.glob("*/result.json")
        )
        if valid == args.limit:
            break
    else:
        raise RuntimeError("Some Best-of-4 verifier labels remain invalid; rerun to retry pending labels.")
    normalized = output / "scores"
    run([
        sys.executable, "collect_results.py", "--paired-dir", str(paired),
        "--oagents-dir", str(oagents),
        "--out", str(normalized),
    ])
    run([
        sys.executable, "aggregate_occubench_metrics.py",
        "--data-root", str(ROOT / "data"), "--scores-dir", str(normalized),
        "--out", str(normalized / "category_metrics.json"),
    ])


if __name__ == "__main__":
    main()

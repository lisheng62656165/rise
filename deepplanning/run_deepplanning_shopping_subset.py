from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepplanning_model_policy import PRIMARY_MODEL_CONFIG, resolve_model_identity
from src.deepplanning_run_audit import install_call_audit


DEFAULT_SHOPPING_ROOT = Path(
    "/home/lisheng/project/baseline/agent_eval/data_downloard/deepplanning/"
    "Qwen-Agent-main/benchmark/deepplanning/shoppingplanning"
)


def parse_case_ids(values: list[str]) -> list[int]:
    case_ids = []
    for value in values:
        normalized = value[5:] if value.startswith("case_") else value
        case_id = int(normalized)
        if case_id not in case_ids:
            case_ids.append(case_id)
    return case_ids


def prepare_database(shopping_root: Path, level: int, case_ids: list[int], run_name: str) -> Path:
    source = shopping_root / f"database_level{level}"
    target = shopping_root / f"database_run_{run_name}"
    if target.exists():
        raise FileExistsError(f"Run directory already exists: {target}")
    source_cases = [source / f"case_{case_id}" for case_id in case_ids]
    missing = [path for path in source_cases if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source cases: {missing}")
    target.mkdir(parents=True)
    for source_case in source_cases:
        shutil.copytree(source_case, target / source_case.name)
    return target


def completed_resume_case_ids(database_dir: Path, case_ids: list[int]) -> list[int]:
    expected = {f"case_{case_id}" for case_id in case_ids}
    actual = {path.name for path in database_dir.glob("case_*") if path.is_dir()}
    if actual != expected:
        raise ValueError(
            f"Partial run case set mismatch for {database_dir}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    completed = []
    for case_id in case_ids:
        messages_path = database_dir / f"case_{case_id}" / "messages.json"
        if not messages_path.is_file():
            continue
        try:
            messages = json.loads(messages_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        message_rows = messages.get("messages") if isinstance(messages, dict) else messages
        if isinstance(message_rows, list) and message_rows:
            completed.append(case_id)
    return completed


def run_subset(
    shopping_root: Path,
    model: str,
    level: int,
    case_ids: list[int],
    database_dir: Path,
    workers: int,
    max_llm_calls: int,
    usage_path: Path,
) -> dict:
    sys.path.insert(0, str(shopping_root))
    from agent.prompts import prompt_lib
    from agent import shopping_agent

    install_call_audit(shopping_agent, usage_path)

    return shopping_agent.run_agent_inference(
        model=model,
        test_data_path=shopping_root / f"data/level_{level}_query_meta.json",
        database_dir=database_dir,
        tool_schema_path=shopping_root / "tools/shopping_tool_schema.json",
        system_prompt=getattr(prompt_lib, f"SYSTEM_PROMPT_level{level}"),
        workers=min(workers, len(case_ids)),
        max_llm_calls=max_llm_calls,
        rerun_ids=case_ids,
    )


def evaluate_subset(shopping_root: Path, final_dir: Path) -> Path:
    final_dir = final_dir.resolve()
    command = [
        sys.executable,
        str(shopping_root / "evaluation/evaluation_pipeline.py"),
        "--database_dir",
        str(final_dir),
    ]
    subprocess.run(command, check=True)
    return shopping_root / "result_report" / final_dir.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shopping-root", type=Path, default=DEFAULT_SHOPPING_ROOT)
    parser.add_argument("--model", default=PRIMARY_MODEL_CONFIG)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-llm-calls", type=int, default=400)
    parser.add_argument("--trial", type=int, choices=(1, 2, 3, 4, 5, 6), required=True)
    parser.add_argument("--orchestration-seed", type=int, required=True)
    parser.add_argument("--resume-existing-inference", action="store_true")
    parser.add_argument("--min-execution-valid-rate", type=float, default=0.95)
    parser.add_argument(
        "--allow-inference-failures",
        action="store_true",
        help="Evaluate completed cases and record failed IDs instead of aborting the whole subset.",
    )
    args = parser.parse_args()

    model_identity = resolve_model_identity(args.shopping_root.parent / "models_config.json", args.model)
    case_ids = parse_case_ids(args.case_ids)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = args.run_name or f"{args.model}_level{args.level}_subset_{timestamp}"
    if args.resume_existing_inference:
        database_dir = args.shopping_root / f"database_run_{run_name}"
        if not database_dir.exists():
            raise FileNotFoundError(database_dir)
        completed_ids = completed_resume_case_ids(database_dir, case_ids)
        pending_ids = [case_id for case_id in case_ids if case_id not in completed_ids]
        resumed = run_subset(
            args.shopping_root,
            args.model,
            args.level,
            pending_ids,
            database_dir,
            args.workers,
            args.max_llm_calls,
            database_dir / "agent_usage.jsonl",
        ) if pending_ids else {"results": []}
        resumed_rows = {int(row["id"]): row for row in resumed.get("results", [])}
        result_rows = [
            {"id": case_id, "success": True, "error": None, "resumed_existing": True}
            if case_id in completed_ids
            else resumed_rows.get(case_id, {
                "id": case_id, "success": False, "error": "missing resumed result"
            })
            for case_id in case_ids
        ]
        result = {
            "total": len(result_rows),
            "success": sum(1 for row in result_rows if row["success"]),
            "failed": sum(1 for row in result_rows if not row["success"]),
            "results": result_rows,
        }
    else:
        database_dir = prepare_database(args.shopping_root, args.level, case_ids, run_name)
        result = run_subset(
            args.shopping_root,
            args.model,
            args.level,
            case_ids,
            database_dir,
            args.workers,
            args.max_llm_calls,
            database_dir / "agent_usage.jsonl",
        )
        result_rows = result.get("results") or []
    failed_rows = [row for row in result_rows if not row.get("success")]
    failed_case_ids = [str(row.get("id")) for row in failed_rows]
    if result["failed"] and not args.allow_inference_failures:
        raise RuntimeError(f"Subset inference failed; active database retained at {database_dir}: {result}")
    final_dir = args.shopping_root / "database_infered" / f"database_{run_name}"
    if final_dir.exists():
        raise FileExistsError(f"Final directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    database_dir.rename(final_dir)
    report_dir = evaluate_subset(args.shopping_root, final_dir)
    manifest = {
        "model": args.model,
        "api_model": model_identity["api_model"],
        "model_config_sha256": model_identity["model_config_sha256"],
        "level": args.level,
        "case_ids": case_ids,
        "workers": min(args.workers, len(case_ids)),
        "max_llm_calls": args.max_llm_calls,
        "controller_enabled": False,
        "prototype_store_count": 0,
        "old_artifact_access_count": 0,
        "split": "prototype_train",
        "trial": args.trial,
        "trial_cap": 6,
        "orchestration_seed": args.orchestration_seed,
        "api_seed_supported": False,
        "native_function_calling": True,
        "resumed_existing_inference": args.resume_existing_inference,
        "resumed_completed_case_ids": [
            str(value) for value in completed_ids
        ] if args.resume_existing_inference else [],
        "database_dir": str(final_dir),
        "report_dir": str(report_dir),
        "result": {key: value for key, value in result.items() if key != "results"},
        "allow_inference_failures": args.allow_inference_failures,
        "inference_failure_ids": failed_case_ids,
        "execution_valid_case_ids": [
            str(row.get("id")) for row in result_rows if row.get("success")
        ],
        "usage_path": str(final_dir / "agent_usage.jsonl"),
    }
    if failed_rows:
        failure_path = report_dir / "inference_failures.json"
        failure_path.write_text(
            json.dumps({"count": len(failed_rows), "rows": failed_rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["inference_failures_path"] = str(failure_path)
    manifest_path = report_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    valid_rate = len(manifest["execution_valid_case_ids"]) / len(case_ids)
    if valid_rate < args.min_execution_valid_rate:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

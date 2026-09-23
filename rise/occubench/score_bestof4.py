#!/usr/bin/env python3
"""Verifier-only scoring for the selected OAgents Best-of-4 trajectories."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path,
                        help="EDS-ECA source directory; defaults to the path recorded in the run manifest.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verifier-max-tokens", type=int, default=4096)
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    import sys
    sys.path.insert(0, str(args.dataset_root.resolve()))
    from occubench.lwm import WorldModelRegistry, create_client
    from occubench.verifier import Verifier
    os.environ["OCCUBENCH_REQUEST_TIMEOUT"] = "180"
    os.environ["OCCUBENCH_STREAM"] = "0"
    os.environ["OCCUBENCH_VERIFIER_MAX_TOKENS"] = str(args.verifier_max_tokens)
    os.environ["OCCUBENCH_VERIFIER_EXTRA_BODY_JSON"] = '{"chat_template_kwargs":{"enable_thinking":false}}'

    tasks = {int(r["task_id"]): r for r in read_jsonl(
        args.dataset_root / "data" / "eval_benchmark_solvable.jsonl")}
    registry = WorldModelRegistry(str(args.dataset_root / "data" / "world_model_configs"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    source_dir = args.source_dir or Path(manifest["source"])
    source_rows = {
        int(r["task_id"]): r for r in read_jsonl(source_dir / "results.jsonl")
    }
    paired_labels = {
        (int(r["task_id"]), r["stage"]): r
        for r in read_jsonl(source_dir / f"paired_{len(manifest['task_ids'])}" / "scores.jsonl")
        if r.get("verification_valid") is True and type(r.get("is_correct")) is bool
        and r.get("verifier_model") == args.model
    }
    paths = sorted(args.run_dir.glob("*/result.json"))
    rows = []
    pending = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("bestof4", {}).get("verification_valid") is True:
            rows.append(row)
            continue
        tid = int(row["task_id"])
        candidates = [json.loads((args.run_dir / str(tid) / f"candidate_{i}.json").read_text(encoding="utf-8"))
                      for i in (1, 2, 3)]
        source = source_rows.get(tid)
        if source is None:
            # The anchor is copied into the Best-of-4 output only by reference.
            raise ValueError(f"Missing source anchor for task {tid}")
        candidates.insert(0, next(s for s in source["stages"] if s["stage"] == "A"))
        selected = candidates[int(row["decision"]["selected"])]
        if selected["trajectory"] == candidates[0]["trajectory"]:
            verdict = paired_labels.get((tid, "A"))
            if verdict is not None:
                row["bestof4"] = dict(verdict, verifier_model=args.model,
                                      verification_valid=True, source="paired_vanilla_anchor")
                path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                continue
        task = tasks[tid]
        config = registry.get(task["env_name"])
        initial = config.get("task_initial_state", "{}")
        if isinstance(initial, dict):
            initial = json.dumps(initial, ensure_ascii=False)
        pending.append((path, row, task, config, initial, selected))

    write_lock = threading.Lock()

    def score_one(item: tuple) -> dict:
        path, row, task, config, initial, selected = item
        verdict = {"is_correct": None, "feedback": "Verification error"}
        error = "invalid_verifier_response"
        for attempt in range(8):
            client = None
            try:
                client = create_client(key, args.base_url)
                verdict = Verifier(args.model, client=client, num_votes=1).check(
                    task_scenario_name=task["task_scenario_name"],
                    task_initial_state=initial,
                    state_description=config.get("state_description", ""),
                    agent_instruction=task["agent_instruction"],
                    verification_plan=task["verification_plan"],
                    trajectory=selected["trajectory"],
                )
                if verdict.get("feedback") != "Verification error" and type(verdict.get("is_correct")) is bool:
                    error = ""
                    break
            except Exception as exc:
                error = type(exc).__name__
            finally:
                if client is not None:
                    client.close()
            time.sleep(min(20, 2 ** attempt))
        valid = not error
        row["bestof4"] = dict(
            verdict, verifier_model=args.model, verification_valid=valid,
            **({"scoring_error": error} if not valid else {}),
        )
        with write_lock:
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_one, item) for item in pending]
        for future in as_completed(futures):
            rows.append(future.result())
    print(json.dumps({"scored": len(rows), "target": len(list(args.run_dir.glob("*/result.json")))}, indent=2))


if __name__ == "__main__":
    main()

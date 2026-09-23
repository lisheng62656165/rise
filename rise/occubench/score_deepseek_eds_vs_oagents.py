"""Score the currently completed DeepSeek EDS-ECA/OAgents cohort pairwise."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from score_occubench_stages_nvidia import load_keys, load_tasks, read_jsonl


def valid(row: dict, model: str) -> bool:
    return (row.get("verification_valid") is True
            and row.get("verifier_model") == model
            and type(row.get("is_correct")) is bool)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--eds-dir", type=Path, required=True)
    p.add_argument("--oagents-dir", type=Path, required=True)
    p.add_argument("--key-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--base-url", default="https://hgapi.dieqiyun.top/v1")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--key-back", type=int, default=1)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eds_rows = {int(r["task_id"]): r for r in read_jsonl(args.eds_dir / "results.jsonl")}
    ids = sorted(int(d.name) for d in args.oagents_dir.iterdir()
                 if d.is_dir() and d.name.isdigit() and (d / "result.json").exists()
                 and int(d.name) in eds_rows)
    pairs: dict[int, dict[str, str]] = {}
    for tid in ids:
        eds = eds_rows[tid]
        final_stage = eds["final_stage"]
        final = next(s for s in eds["stages"]
                     if s["stage"] == final_stage and s["trajectory"] == eds["final_trajectory"])
        folder = args.oagents_dir / str(tid)
        decision = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        selected = int(decision["selected"])
        if selected == 0:
            anchor = next(s for s in eds["stages"] if s["stage"] == "A")
            oagent_trajectory = anchor["trajectory"]
        else:
            oagent_trajectory = json.loads(
                (folder / f"candidate_{selected}.json").read_text(encoding="utf-8"))["trajectory"]
        pairs[tid] = {"EDS-ECA": final["trajectory"], "OAgents-Best-of-4": oagent_trajectory}

    score_path = args.out / "scores.jsonl"
    lock = Lock()
    labels = {(int(r["task_id"]), r["method"]): r
              for r in read_jsonl(score_path) if valid(r, args.model)}
    tasks = load_tasks(args.dataset_root)
    registry = None
    keys = load_keys(args.key_file, args.key_back, 10)
    sys.path.insert(0, str(args.dataset_root))
    from occubench.lwm import WorldModelRegistry, create_client
    from occubench.verifier import Verifier
    registry = WorldModelRegistry(str(args.dataset_root / "data" / "world_model_configs"))
    os.environ.update(OCCUBENCH_REQUEST_TIMEOUT="180", OCCUBENCH_STREAM="1",
                      OCCUBENCH_VERIFIER_MAX_TOKENS="4096",
                      OCCUBENCH_VERIFIER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}')

    def report() -> None:
        paired = [tid for tid in ids if (tid, "EDS-ECA") in labels and (tid, "OAgents-Best-of-4") in labels]
        e = sum(labels[tid, "EDS-ECA"]["is_correct"] for tid in paired)
        o = sum(labels[tid, "OAgents-Best-of-4"]["is_correct"] for tid in paired)
        wins = sum(not labels[tid, "EDS-ECA"]["is_correct"] and labels[tid, "OAgents-Best-of-4"]["is_correct"] for tid in paired)
        losses = sum(labels[tid, "EDS-ECA"]["is_correct"] and not labels[tid, "OAgents-Best-of-4"]["is_correct"] for tid in paired)
        summary = {"cohort": len(ids), "paired_valid": len(paired),
                   "pending": 2 * len(ids) - sum((tid, m) in labels for tid in ids for m in pairs[tid]),
                   "eds_pass": e, "oagents_pass": o,
                   "eds_pass_at_1": e / len(paired) if paired else None,
                   "oagents_pass_at_1": o / len(paired) if paired else None,
                   "oagents_wins": wins, "eds_wins": losses,
                   "ties": len(paired) - wins - losses,
                   "delta_pp_oagents_minus_eds": 100 * (o - e) / len(paired) if paired else None,
                   "verifier_model": args.model}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)

    jobs = [(tid, method, trajectory) for tid, methods in pairs.items()
            for method, trajectory in methods.items() if (tid, method) not in labels]
    report()
    print(json.dumps({"unique_verifier_jobs": len(jobs), "workers": args.workers}), flush=True)

    def score(index: int, tid: int, method: str, trajectory: str) -> dict:
        task = tasks[tid]
        config = registry.get(task["env_name"])
        initial = config.get("task_initial_state", "{}")
        if isinstance(initial, dict):
            initial = json.dumps(initial, ensure_ascii=False)
        last_error = "Verification error"
        for attempt in range(5):
            key_back, key = keys[(index + attempt) % len(keys)]
            client = create_client(key, args.base_url)
            try:
                verdict = Verifier(args.model, client=client, num_votes=1).check(
                    task_scenario_name=task["task_scenario_name"], task_initial_state=initial,
                    state_description=config.get("state_description", ""),
                    agent_instruction=task["agent_instruction"],
                    verification_plan=task["verification_plan"], trajectory=trajectory)
                if verdict.get("feedback") != "Verification error" and type(verdict.get("is_correct")) is bool:
                    return {"task_id": tid, "method": method, "verifier_model": args.model,
                            "is_correct": verdict["is_correct"], "feedback": verdict.get("feedback", ""),
                            "verification_valid": True, "attempt": attempt + 1}
                last_error = verdict.get("feedback", "Verification error")
            except Exception as exc:
                last_error = type(exc).__name__
            finally:
                client.close()
        return {"task_id": tid, "method": method, "verifier_model": args.model,
                "is_correct": None, "feedback": last_error, "verification_valid": False}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score, i, tid, method, trajectory): (tid, method)
                   for i, (tid, method, trajectory) in enumerate(jobs)}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                with score_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                if valid(row, args.model):
                    labels[row["task_id"], row["method"]] = row
            report()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Normalize completed runner outputs into the files consumed by aggregation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--oagents-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/occubench_382"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eds_rows, oagent_rows, vanilla_rows = [], [], []
    for row in read_jsonl(args.paired_dir / "scores.jsonl"):
        if row.get("verification_valid") is not True or type(row.get("is_correct")) is not bool:
            continue
        if row.get("stage") == "FINAL":
            eds_rows.append({"task_id": int(row["task_id"]), "method": "EDS-ECA",
                             "is_correct": row["is_correct"], "verification_valid": True,
                             "verifier_model": row.get("verifier_model", "")})
        elif row.get("stage") == "A":
            vanilla_rows.append({"task_id": int(row["task_id"]), "method": "Vanilla-A",
                                 "is_correct": row["is_correct"], "verification_valid": True,
                                 "verifier_model": row.get("verifier_model", "")})
    for path in sorted(args.oagents_dir.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        verdict = row.get("bestof4", {})
        if verdict.get("verification_valid") is True and type(verdict.get("is_correct")) is bool:
            oagent_rows.append({"task_id": int(row["task_id"]), "method": "OAgents-Best-of-4",
                                "is_correct": verdict["is_correct"],
                                "verification_valid": True,
                                "verifier_model": verdict.get("verifier_model", "")})
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eds_rows + oagent_rows),
        encoding="utf-8")
    (args.out / "vanilla_scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in vanilla_rows),
        encoding="utf-8")
    print(json.dumps({"eds": len(eds_rows), "oagents": len(oagent_rows),
                      "vanilla": len(vanilla_rows), "target": 382}, indent=2))


if __name__ == "__main__":
    main()

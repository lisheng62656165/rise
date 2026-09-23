#!/usr/bin/env python3
"""Merge successful Shopping retries into the Vanilla anchor used by EDS-ECA."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_successes(shopping_root: Path, level: int) -> dict[int, Path]:
    report_root = shopping_root / "result_report"
    inferred_root = shopping_root / "database_infered"
    sources: dict[int, Path] = {}
    pattern = f"database_nemotron_vanilla_a_trial1r*_L{level}"
    for report_dir in sorted(report_root.glob(pattern)):
        manifest_path = report_dir / "subset_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        database_dir = inferred_root / report_dir.name
        for raw_id in manifest.get("execution_valid_case_ids", []):
            case_id = int(raw_id)
            case_dir = database_dir / f"case_{case_id}"
            messages_path = case_dir / "messages.json"
            if case_dir.is_dir() and messages_path.is_file() and messages_path.stat().st_size > 0:
                sources[case_id] = case_dir
    return sources


def merge_level(shopping_root: Path, level: int, replace: bool) -> tuple[int, list[int]]:
    expected_count = 20 if level == 3 else 50
    sources = load_successes(shopping_root, level)
    missing = [case_id for case_id in range(1, expected_count + 1) if case_id not in sources]
    if missing:
        return len(sources), missing

    target = (
        shopping_root
        / "database_infered"
        / f"database_nemotron_vanilla_a_trial1_merged_L{level}"
    )
    if target.exists():
        if not replace:
            complete = all(
                (target / f"case_{case_id}" / "messages.json").is_file()
                and (target / f"case_{case_id}" / "messages.json").stat().st_size > 0
                for case_id in range(1, expected_count + 1)
            )
            if complete:
                return expected_count, []
            raise FileExistsError(f"Incomplete existing merge target: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for case_id in range(1, expected_count + 1):
        shutil.copytree(sources[case_id], target / f"case_{case_id}")
    return expected_count, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shopping-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    incomplete = False
    for level in (1, 2, 3):
        count, missing = merge_level(args.shopping_root, level, args.replace)
        print(json.dumps({"level": level, "valid": count, "missing": missing}))
        incomplete = incomplete or bool(missing)
    if incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

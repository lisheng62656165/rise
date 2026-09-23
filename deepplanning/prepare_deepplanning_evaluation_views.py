#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def link_case(view: Path, name: str, source: Path) -> None:
    if not source.is_dir() or not (source / "messages.json").is_file():
        raise FileNotFoundError(source)
    target = view / name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Existing link has a different source: {target}")
        return
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(source, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method-output", type=Path, required=True)
    args = parser.parse_args()

    shopping = args.root / "shoppingplanning"
    views = args.method_output / "evaluation_views"
    vanilla_view = views / "shopping_vanilla"
    method_view = views / "shopping_statetrace"
    vanilla_view.mkdir(parents=True, exist_ok=True)
    method_view.mkdir(parents=True, exist_ok=True)

    for level, count in ((1, 50), (2, 50), (3, 20)):
        vanilla_source = (
            shopping
            / "database_infered"
            / f"database_nemotron_vanilla_a_trial1_merged_L{level}"
        )
        method_source = args.method_output / "shopping" / "final_database"
        for case_id in range(1, count + 1):
            name = f"case_L{level}_{case_id}"
            link_case(vanilla_view, name, vanilla_source / f"case_{case_id}")
            link_case(method_view, name, method_source / name)

    for view in (vanilla_view, method_view):
        count = sum(1 for path in view.glob("case_*") if path.is_dir())
        if count != 120:
            raise RuntimeError(f"Evaluation view coverage is {count}/120: {view}")
        print(f"{view}: {count}/120")


if __name__ == "__main__":
    main()

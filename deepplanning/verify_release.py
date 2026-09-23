"""Check that the bundled benchmark has all three full test cohorts."""
from pathlib import Path
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root.resolve()
    shopping = root / "shoppingplanning"
    levels = {level: len(list((shopping / f"database_level{level}").glob("case_*"))) for level in (1, 2, 3)}
    travel = {}
    for language in ("zh", "en"):
        database = root / "travelplanning" / "database" / f"database_{language}"
        travel[language] = len(list(database.glob("id_*")))
    result = {"shopping_levels": levels, "shopping_total": sum(levels.values()), "travel": travel}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if levels != {1: 50, 2: 50, 3: 20} or travel != {"zh": 120, "en": 120}:
        raise SystemExit("release is incomplete")


if __name__ == "__main__":
    main()

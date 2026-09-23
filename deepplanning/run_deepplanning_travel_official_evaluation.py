from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--travel-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    travel_root = args.travel_root.resolve()
    sys.path.insert(0, str(travel_root))
    from evaluation.eval_converted import evaluate_plans

    result = evaluate_plans(
        result_dir=args.result_dir.resolve(),
        test_data_path=travel_root / f"data/travelplanning_query_{args.language}.json",
        database_dir=travel_root / f"database/database_{args.language}",
        workers=args.workers,
    )
    summary = {
        "success": result["success"],
        "failed": result["failed"],
        "total": result["total"],
        "metrics": result["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    if result["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

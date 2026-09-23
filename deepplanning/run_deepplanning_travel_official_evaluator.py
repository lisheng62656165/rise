from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--travel-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    travel_root = args.travel_root.resolve()
    sys.path.insert(0, str(travel_root))
    from evaluation.eval_converted import evaluate_plans

    result = evaluate_plans(
        result_dir=args.result_dir.resolve(),
        test_data_path=args.test_data.resolve(),
        database_dir=args.database_dir.resolve(),
        workers=args.workers,
        verbose=False,
    )
    print(
        json.dumps(
            {
                "success": result.get("success"),
                "total": result.get("total"),
                "metrics": result.get("metrics"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

import json
import re
import sys
from pathlib import Path


root, pair = map(Path, sys.argv[1:])
for language in ("zh", "en"):
    converted_ids = []
    for variant in ("vanilla", "statetrace"):
        ids = {
            int(re.match(r"id_(\d+)_converted", path.stem).group(1))
            for path in (pair / f"{language}_{variant}" / "converted_plans").glob(
                "id_*_converted.json"
            )
        }
        converted_ids.append(ids)
    common = converted_ids[0] & converted_ids[1]
    source = json.loads(
        (root / "travelplanning" / "data" / f"travelplanning_query_{language}.json").read_text(
            encoding="utf-8"
        )
    )
    subset = [row for row in source if int(row["id"]) in common]
    assert len(subset) == len(common)
    (pair / f"test_common_{language}.json").write_text(
        json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(language, len(common))

    report_ids = {
        int(re.match(r"id_(\d+)", path.stem).group(1))
        for path in (pair / f"{language}_statetrace" / "reports").glob("id_*.txt")
    }
    partial = [row for row in source if int(row["id"]) in report_ids]
    assert len(partial) == len(report_ids)
    (pair / f"test_partial_{language}.json").write_text(
        json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
    )

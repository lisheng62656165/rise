"""Small release preflight that does not contact a model provider."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phase0_runner import DOMAINS, read_split_ids
from oagents_official_orm import PROMPT_PATH, load_prompt

ROOT = Path(__file__).resolve().parent


def main():
    expected = {s: {d: len(read_split_ids(ROOT / "benchmark", d, s)) for d in DOMAINS}
                for s in ("train", "dev", "test")}
    assert expected == {s: {d: 70 if s == "train" else 30 if s == "dev" else 50 for d in DOMAINS}
                        for s in ("train", "dev", "test")}
    assert load_prompt().strip()
    prompt_sha256 = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    result = {"split_sizes": expected, "test_tasks": 150,
              "official_orm_prompt_bytes": PROMPT_PATH.stat().st_size,
              "official_orm_prompt_sha256": prompt_sha256,
              "no_api_calls": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

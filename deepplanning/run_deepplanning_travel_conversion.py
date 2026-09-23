#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--travel-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("zh", "en"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=53403)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--reasoning-budget", type=int, default=2048)
    parser.add_argument("--max-parse-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.travel_root.resolve()))
    sys.path.insert(0, str((args.travel_root / "agent").resolve()))
    import call_llm as call_llm_module

    original_load = call_llm_module.load_model_config

    def seeded_load(model_name: str):
        config = dict(original_load(model_name))
        extra_body = dict(config.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        extra_body["seed"] = args.seed
        extra_body.pop("reasoning_budget", None)
        config["extra_body"] = extra_body
        config["temperature"] = 0.0
        config["top_p"] = 1.0
        config["max_tokens"] = args.max_tokens
        config["request_timeout"] = args.request_timeout
        config["max_retries"] = 1
        return config

    call_llm_module.load_model_config = seeded_load
    from evaluation.convert_report import convert_reports

    result = convert_reports(
        result_dir=args.result_dir.resolve(),
        language=args.language,
        workers=args.workers,
        skip_existing=True,
        max_parse_retries=args.max_parse_retries,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))
    converted = len(list((args.result_dir / "converted_plans").glob("id_*_converted.json")))
    if converted != 120 and not args.allow_partial:
        raise RuntimeError(f"Converted-plan coverage is {converted}/120 for {args.result_dir}")


if __name__ == "__main__":
    main()

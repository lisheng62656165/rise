#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def is_substantive_plan(plan: str) -> bool:
    normalized = re.sub(r"\s+", " ", plan or "").strip()
    semantic_chars = re.findall(r"[A-Za-z0-9\u3400-\u9fff]", normalized)
    placeholders = {"...", "…", "tbd", "todo", "n/a", "none"}
    return (
        len(normalized) >= 200
        and len(semantic_chars) >= 80
        and normalized.lower() not in placeholders
    )


def extract_plan(content: str) -> str:
    matches = re.findall(r"<plan>(.*?)</plan>", content or "", flags=re.DOTALL | re.IGNORECASE)
    plan = "\n\n".join(part.strip() for part in matches if part.strip())
    return plan if is_substantive_plan(plan) else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--seed", type=int, default=53403)
    args = parser.parse_args()

    if args.report.exists():
        existing = args.report.read_text(encoding="utf-8", errors="replace")
        if is_substantive_plan(existing):
            print(f"Substantive report already exists: {args.report}")
            return
        raise RuntimeError(f"Existing report is not substantive: {args.report}")

    travel_root = args.project_root / "travelplanning"
    sys.path.insert(0, str(travel_root))
    from agent import call_llm as call_llm_module

    original_load = call_llm_module.load_model_config

    def seeded_load(model_name: str):
        config = dict(original_load(model_name))
        extra_body = dict(config.get("extra_body") or {})
        extra_body["seed"] = args.seed
        config["extra_body"] = extra_body
        return config

    call_llm_module.load_model_config = seeded_load

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    messages = list(trajectory.get("messages") or [])
    if args.language == "zh":
        finalization = (
            "公开工具证据收集现已结束。不要调用任何工具，也不要只输出分析。"
            "请基于上方已有证据，立即将完整、可提交的最终行程放在 <plan>...</plan> 中。"
        )
    else:
        finalization = (
            "Public tool-evidence collection is now finished. Do not call any tools and do not "
            "output analysis alone. Using the evidence above, immediately return the complete, "
            "submittable final itinerary inside <plan>...</plan>."
        )
    messages.append({"role": "user", "content": finalization})

    started = time.time()
    plan = ""
    for attempt in range(2):
        response = call_llm_module.call_llm(args.model, messages, tools=None)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        plan = extract_plan(message.content or "")
        if plan:
            break
        messages.append({
            "role": "user",
            "content": (
                "只输出一个完整的 <plan>...</plan>，不要输出分析或其他文字。"
                if args.language == "zh"
                else "Output exactly one complete <plan>...</plan> block now, with no analysis or other text."
            ),
        })
    if not plan:
        raise RuntimeError("Two finalization responses did not contain a substantive <plan> block")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(plan, encoding="utf-8")
    trajectory["messages"] = messages
    trajectory["final_plan"] = plan
    trajectory["success"] = True
    trajectory["finalization_recovery"] = {
        "seed": args.seed,
        "tools_enabled": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_trajectory = args.report.parent.parent / "trajectories" / f"{args.report.stem}.json"
    output_trajectory.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "chars": len(plan)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

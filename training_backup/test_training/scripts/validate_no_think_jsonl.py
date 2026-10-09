#!/usr/bin/env python3
"""Validate that a JSONL file contains native no-think samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EMPTY_THINKING_VALUES = {None, "", "none", "disabled"}


def is_empty_thinking_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_THINKING_VALUES
    return value in EMPTY_THINKING_VALUES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()

    counters: Counter[str] = Counter()
    budgets: Counter[str] = Counter()
    thinking_types: Counter[str] = Counter()

    with args.jsonl.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            counters["total"] += 1
            lowered = raw_line.lower()
            if "<think" in lowered or "</think" in lowered:
                counters["think_tag"] += 1
            obj = json.loads(raw_line)
            budgets[str(obj.get("thinking_budget"))] += 1
            thinking_types[str(obj.get("thinking_type"))] += 1
            if not is_empty_thinking_value(obj.get("thinking_type")):
                counters["top_thinking_type_bad"] += 1
            if not is_empty_thinking_value(obj.get("thinking_budget")):
                counters["top_thinking_budget_bad"] += 1
            for message in obj.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                if not is_empty_thinking_value(message.get("thinking_type")):
                    counters["message_thinking_type_bad"] += 1
                if not is_empty_thinking_value(message.get("thinking_budget")):
                    counters["message_thinking_budget_bad"] += 1
            if line_no <= 5:
                counters[f"preview_line_{line_no}_messages"] = len(obj.get("messages") or [])

    report = {
        "file": str(args.jsonl),
        "counters": dict(counters),
        "thinking_budget_distribution": budgets.most_common(),
        "thinking_type_distribution": thinking_types.most_common(),
        "valid": (
            counters["total"] > 0
            and counters["think_tag"] == 0
            and counters["top_thinking_type_bad"] == 0
            and counters["top_thinking_budget_bad"] == 0
            and counters["message_thinking_type_bad"] == 0
            and counters["message_thinking_budget_bad"] == 0
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

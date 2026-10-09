#!/usr/bin/env python3
"""Create a deterministic native no-think JSONL sample from a training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMPTY_THINKING_VALUES = {None, "", "none", "disabled"}


def is_empty_thinking_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_THINKING_VALUES
    return value in EMPTY_THINKING_VALUES


def is_native_no_think(obj: dict[str, Any], raw_line: str) -> tuple[bool, str]:
    lowered = raw_line.lower()
    if "<think" in lowered or "</think" in lowered:
        return False, "think_tag"

    if not is_empty_thinking_value(obj.get("thinking_type")):
        return False, "top_thinking_type"
    if not is_empty_thinking_value(obj.get("thinking_budget")):
        return False, "top_thinking_budget"

    for message in obj.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if not is_empty_thinking_value(message.get("thinking_type")):
            return False, "message_thinking_type"
        if not is_empty_thinking_value(message.get("thinking_budget")):
            return False, "message_thinking_budget"

    return True, "eligible"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset-name", default="no-think-sampled")
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{args.dataset_name}.train.jsonl"
    summary_json = args.output_dir / "summary.json"
    manifest_json = args.output_dir / "manifest.json"

    rng = random.Random(args.seed)
    reservoir: list[str] = []
    counters: dict[str, int] = {
        "total": 0,
        "eligible_native_no_think": 0,
        "sampled": 0,
        "decode_error": 0,
        "think_tag": 0,
        "top_thinking_type": 0,
        "top_thinking_budget": 0,
        "message_thinking_type": 0,
        "message_thinking_budget": 0,
    }

    with args.source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            counters["total"] += 1
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                counters["decode_error"] += 1
                continue

            eligible, reason = is_native_no_think(obj, raw_line)
            if not eligible:
                counters[reason] = counters.get(reason, 0) + 1
                continue

            counters["eligible_native_no_think"] += 1
            seen = counters["eligible_native_no_think"]
            line = raw_line.rstrip("\n")
            if len(reservoir) < args.sample_size:
                reservoir.append(line)
            else:
                replace_idx = rng.randrange(seen)
                if replace_idx < args.sample_size:
                    reservoir[replace_idx] = line

    if counters["eligible_native_no_think"] < args.sample_size:
        raise RuntimeError(
            f"Only found {counters['eligible_native_no_think']} eligible rows, "
            f"less than requested sample size {args.sample_size}"
        )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for line in reservoir:
            handle.write(line)
            handle.write("\n")

    counters["sampled"] = len(reservoir)
    output_sha256 = sha256_file(output_jsonl)
    created_at = datetime.now(timezone.utc).isoformat()

    summary = {
        "dataset_name": args.dataset_name,
        "created_at_utc": created_at,
        "source": str(args.source),
        "output_jsonl": str(output_jsonl),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "selection": "native no-think: no thinking_type/budget enabled at top/message level and no <think> tag in raw JSONL line",
        "counters": counters,
        "output_sha256": output_sha256,
    }
    manifest = {
        "name": args.dataset_name,
        "train_file": str(output_jsonl),
        "num_records": len(reservoir),
        "format": "jsonl",
        "source_file": str(args.source),
        "sha256": output_sha256,
        "created_at_utc": created_at,
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

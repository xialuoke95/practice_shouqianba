#!/usr/bin/env python3
"""Version Arm C by normalizing non-training metadata for Arrow loading."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any


NORMALIZED_PATH = "dedup_stats.trainable_turn_ratio"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_row(row: dict[str, Any]) -> bool:
    stats = row.get("dedup_stats")
    if not isinstance(stats, dict) or "trainable_turn_ratio" not in stats:
        return False
    value = stats["trainable_turn_ratio"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(
            f"{NORMALIZED_PATH} must be numeric, got {type(value).__name__}"
        )
    if isinstance(value, float):
        return False
    stats["trainable_turn_ratio"] = float(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    parent_sha256 = sha256_file(args.input)
    rows = 0
    changed_rows = 0
    trace_ids: collections.Counter[str] = collections.Counter()
    trainable_messages = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with args.input.open(encoding="utf-8") as source, temp.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            rows += 1
            trace_id = row.get("trace_id")
            if not isinstance(trace_id, str) or not trace_id:
                raise AssertionError(f"missing trace_id at line {line_number}")
            trace_ids[trace_id] += 1
            changed_rows += int(normalize_row(row))
            trainable_messages += sum(
                1
                for message in row.get("messages", [])
                if message.get("role") in {"assistant", "tool_call"}
                and message.get("loss", 1) == 1
            )
            target.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        target.flush()
        os.fsync(target.fileno())
    os.replace(temp, args.output)

    multiplicities = collections.Counter(trace_ids.values())
    output_sha256 = sha256_file(args.output)
    manifest = {
        "schema_version": 1,
        "parent": {"path": str(args.input), "sha256": parent_sha256},
        "output": {"path": str(args.output), "sha256": output_sha256},
        "transformation": {
            "name": "normalize_arrow_numeric_metadata_v1",
            "path": NORMALIZED_PATH,
            "rule": "convert integer values to numerically equivalent floats",
            "changes_training_content": False,
            "changed_rows": changed_rows,
        },
        "rows": rows,
        "unique_trace_ids": len(trace_ids),
        "singleton_trace_ids": multiplicities[1],
        "duplicated_trace_ids": multiplicities[2],
        "max_trace_multiplicity": max(trace_ids.values()),
        "trainable_messages": trainable_messages,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = args.manifest.with_suffix(
        args.manifest.suffix + f".tmp.{os.getpid()}"
    )
    temp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_manifest, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

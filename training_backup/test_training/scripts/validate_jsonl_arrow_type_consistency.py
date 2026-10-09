#!/usr/bin/env python3
"""Reject JSONL scalar type drift that can break HuggingFace/PyArrow loading."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterator


def scalar_values(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield from scalar_values(child, child_path)
    elif isinstance(value, list):
        for child in value:
            yield from scalar_values(child, f"{path}[]")
    else:
        yield path, value


def scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    observed: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    examples: dict[tuple[str, str], dict[str, Any]] = {}
    rows = 0
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            rows += 1
            for path, value in scalar_values(row):
                kind = scalar_type(value)
                observed[path][kind] += 1
                examples.setdefault(
                    (path, kind),
                    {
                        "line": line_number,
                        "trace_id": row.get("trace_id"),
                        "value": repr(value)[:200],
                    },
                )

    mixed_numeric = []
    for path, counts in sorted(observed.items()):
        if counts["int"] and counts["float"]:
            mixed_numeric.append(
                {
                    "path": path,
                    "counts": dict(counts),
                    "int_example": examples[(path, "int")],
                    "float_example": examples[(path, "float")],
                }
            )
    result = {
        "input": str(args.input),
        "rows": rows,
        "mixed_numeric_paths": mixed_numeric,
        "valid": not mixed_numeric,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if mixed_numeric:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

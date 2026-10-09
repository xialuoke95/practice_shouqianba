#!/usr/bin/env python3
"""Render every requested row with the exact Qwen3.6 Swift template contract."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path


def disable_broken_login_node_transformer_engine() -> None:
    """Template-only validation does not need the login node's TE binary."""
    module = types.ModuleType("transformer_engine")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "transformer_engine", loader=None
    )
    sys.modules["transformer_engine"] = module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        default="/pfs-verdent/guoyinuo/coding_agent_models/models/Qwen3.6-27B",
    )
    args = parser.parse_args()

    disable_broken_login_node_transformer_engine()
    from swift import get_processor
    from swift.template import get_template

    processor = get_processor(
        args.model, model_type="qwen3_5", task_type="causal_lm"
    )
    template = get_template(
        processor,
        template_type="qwen3_5",
        max_length=262144,
        padding_free=True,
        sequence_parallel_size=2,
        template_backend="swift",
        add_non_thinking_prefix=True,
    )
    template.set_mode("train")

    results = []
    failures = []
    for path in args.datasets:
        lengths: list[int] = []
        active_labels: list[int] = []
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                rows += 1
                row = json.loads(line)
                try:
                    encoded = template.encode(row, return_length=True)
                    input_ids = encoded.get("input_ids") or []
                    labels = encoded.get("labels") or []
                    active = sum(label != -100 for label in labels)
                    if not input_ids or active == 0:
                        raise AssertionError(
                            f"empty render input_ids={len(input_ids)} active={active}"
                        )
                    lengths.append(len(input_ids))
                    active_labels.append(active)
                except Exception as exc:
                    failures.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "trace_id": row.get("trace_id"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        results.append(
            {
                "path": str(path),
                "rows": rows,
                "rendered_rows": len(lengths),
                "min_length": min(lengths) if lengths else None,
                "max_length": max(lengths) if lengths else None,
                "mean_length": sum(lengths) / len(lengths) if lengths else None,
                "min_active_labels": min(active_labels) if active_labels else None,
                "max_active_labels": max(active_labels) if active_labels else None,
                "mean_active_labels": (
                    sum(active_labels) / len(active_labels)
                    if active_labels
                    else None
                ),
            }
        )

    payload = {
        "valid": not failures,
        "template": "qwen3_5",
        "max_length": 262144,
        "datasets": results,
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

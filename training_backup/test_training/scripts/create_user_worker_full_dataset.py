#!/usr/bin/env python3
"""Create a user_worker-only JSONL from the May day_01_29 cleaned data.

This intentionally does not apply the no-think filter. It follows the same
post-cleaning baseline used by the recent no-think runs:

- source clean JSONL after L4/L5, trunc-marker, long-think, and bad-toolcall drops
- metadata JSONL after the long-think drop, used only to recover agent_source
- drop the same two excluded models used by the no-think drop_sonnet45_sonnet46_1m data
- keep rows with metadata agent_source == user_worker
- write the original clean training schema, without metadata fields
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DROP_MODELS = {
    "claude-sonnet-4-5@20250929",
    "claude-sonnet-4-6-1m",
}


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                yield line_no, line, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def key_for(obj: dict[str, Any]) -> tuple[Any, ...]:
    return (
        obj.get("trace_id"),
        obj.get("session_id"),
        obj.get("model"),
        obj.get("day"),
        obj.get("start_time"),
    )


def short_obj(obj: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "trace_id",
        "session_id",
        "model",
        "day",
        "start_time",
        "thinking_type",
        "thinking_budget",
        "has_persist",
        "msg_count",
        "trainable_asst_count",
        "masked_count",
        "compact_type",
        "tools",
        "messages",
    ]
    return {k: obj[k] for k in keep_keys if k in obj}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--drop-model", action="append", default=[])
    args = parser.parse_args()

    drop_models = set(DEFAULT_DROP_MODELS)
    drop_models.update(args.drop_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata_agent_source: dict[tuple[Any, ...], str] = {}
    metadata_duplicates = 0
    metadata_agent_counts: Counter[str] = Counter()
    metadata_total = 0
    for _, _, obj in load_jsonl(args.metadata):
        metadata_total += 1
        key = key_for(obj)
        if key in metadata_agent_source:
            metadata_duplicates += 1
        agent_source = obj.get("agent_source")
        metadata_agent_source[key] = agent_source
        metadata_agent_counts[str(agent_source)] += 1

    counts: Counter[str] = Counter()
    kept_models: Counter[str] = Counter()
    kept_days: Counter[str] = Counter()
    kept_thinking_type: Counter[str] = Counter()
    kept_thinking_budget: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    sha = hashlib.sha256()
    with args.output.open("w", encoding="utf-8") as out:
        for _, raw_line, obj in load_jsonl(args.clean):
            counts["input_total"] += 1
            model = obj.get("model")
            if model in drop_models:
                counts["removed_excluded_model"] += 1
                continue

            key = key_for(obj)
            agent_source = metadata_agent_source.get(key)
            if agent_source is None:
                counts["metadata_missing"] += 1
                continue
            if agent_source != "user_worker":
                counts["removed_not_user_worker"] += 1
                continue

            row = json.dumps(short_obj(obj), ensure_ascii=False, separators=(",", ":"))
            out.write(row + "\n")
            sha.update(row.encode("utf-8"))
            sha.update(b"\n")

            counts["kept"] += 1
            kept_models[str(model)] += 1
            kept_days[str(obj.get("day"))] += 1
            kept_thinking_type[str(obj.get("thinking_type"))] += 1
            kept_thinking_budget[str(obj.get("thinking_budget"))] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "trace_id": obj.get("trace_id"),
                        "session_id": obj.get("session_id"),
                        "model": model,
                        "day": obj.get("day"),
                        "thinking_type": obj.get("thinking_type"),
                        "thinking_budget": obj.get("thinking_budget"),
                        "msg_count": obj.get("msg_count"),
                    }
                )

    size_bytes = args.output.stat().st_size
    line_count = counts["kept"]
    now = datetime.now(timezone.utc).isoformat()
    summary = {
        "created_at_utc": now,
        "source": str(args.clean),
        "metadata_source": str(args.metadata),
        "output": str(args.output),
        "rules": {
            "keep_agent_source": "user_worker",
            "drop_models": sorted(drop_models),
            "no_think_filter": False,
            "schema": "metadata stripped; training keys only",
        },
        "counts": dict(counts),
        "metadata": {
            "total": metadata_total,
            "duplicates_by_key": metadata_duplicates,
            "agent_source_counts": dict(metadata_agent_counts),
        },
        "kept_models": dict(kept_models),
        "kept_days": dict(kept_days),
        "kept_thinking_type": dict(kept_thinking_type),
        "kept_thinking_budget": dict(kept_thinking_budget),
        "examples": examples,
        "sha256": sha.hexdigest(),
        "size_bytes": size_bytes,
        "line_count": line_count,
    }
    manifest = {
        "created_at_utc": now,
        "dataset": str(args.output),
        "source": str(args.clean),
        "metadata_source": str(args.metadata),
        "line_count": line_count,
        "size_bytes": size_bytes,
        "sha256": sha.hexdigest(),
        "rules": summary["rules"],
    }
    write_json(args.summary, summary)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

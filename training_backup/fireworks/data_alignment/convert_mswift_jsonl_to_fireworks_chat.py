#!/usr/bin/env python3
"""Convert the ms-swift-friendly agent JSONL into Fireworks/Tinker chat JSONL.

The May training pipeline emits a schema tailored for ms-swift's qwen3_5 swift
backend:

  - model tool calls are standalone ``{"role": "tool_call"}`` messages whose
    content is ``{"name": ..., "arguments": ...}``
  - top-level ``tools`` is a JSON string
  - ``loss=0`` marks masked assistant/tool_call turns

Fireworks cookbook SFT expects standard chat roles and OpenAI-style
``assistant.tool_calls``. This script performs that structural conversion and
maps ``loss=0`` to per-message ``trainable=false`` so cookbook renderers select
the customized loss mask path.

Multimodal rows are skipped by default. With ``--include-images``, ms-swift's
top-level ``images`` + inline ``<image>`` placeholders are converted into
OpenAI-style multimodal content parts:
``{"type": "image_url", "image_url": {"url": "data:image/..."}}``.

Rows whose consecutive ``tool_call`` group mixes trainable and non-trainable
calls are skipped by default. Fireworks/Tinker currently marks the whole
assistant message trainable/non-trainable, so such a row cannot be represented
without changing the loss mask.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_PLACEHOLDER = "<image>"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _parse_json_maybe(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return fallback
    return value


def _is_trainable(message: dict[str, Any]) -> bool:
    return message.get("loss") != 0


def _normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return _json_dumps(arguments)


def _parse_tool_call(content: Any) -> tuple[str, str]:
    payload = _parse_json_maybe(content, {})
    if not isinstance(payload, dict):
        payload = {"name": "unknown_tool", "arguments": {"raw": content}}

    name = payload.get("name") or payload.get("tool_name") or "unknown_tool"
    arguments = payload.get("arguments", {})
    return str(name), _normalize_tool_arguments(arguments)


def _content_with_image_parts(
    content: Any,
    *,
    images: list[str],
    image_cursor: list[int],
    stats: Counter[str],
) -> Any:
    if not isinstance(content, str) or IMAGE_PLACEHOLDER not in content:
        return content

    parts: list[dict[str, Any]] = []
    text_parts = content.split(IMAGE_PLACEHOLDER)
    for index, text in enumerate(text_parts):
        if text:
            parts.append({"type": "text", "text": text})
        if index == len(text_parts) - 1:
            continue
        if image_cursor[0] >= len(images):
            stats["missing_images_for_placeholders"] += 1
            parts.append({"type": "text", "text": IMAGE_PLACEHOLDER})
            continue
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": images[image_cursor[0]]},
            }
        )
        image_cursor[0] += 1
        stats["image_placeholders_converted"] += 1

    return parts


def _convert_messages(
    messages: list[dict[str, Any]],
    *,
    row_index: int,
    images: list[str] | None,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending_tool_call_ids: list[str] = []
    image_values = images or []
    image_cursor = [0]
    i = 0

    while i < len(messages):
        message = messages[i]
        role = message.get("role")
        content = message.get("content", "") or ""
        converted_content = _content_with_image_parts(
            content,
            images=image_values,
            image_cursor=image_cursor,
            stats=stats,
        )

        if role == "tool_call":
            group: list[dict[str, Any]] = []
            j = i
            while j < len(messages) and messages[j].get("role") == "tool_call":
                group.append(messages[j])
                j += 1

            trainable_values = {_is_trainable(m) for m in group}
            if len(trainable_values) > 1:
                stats["mixed_tool_call_loss_groups"] += 1

            tool_calls = []
            for offset, tool_message in enumerate(group):
                name, arguments = _parse_tool_call(tool_message.get("content"))
                call_id = f"call_{row_index:06d}_{i:05d}_{offset:02d}"
                pending_tool_call_ids.append(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                )

            converted.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                    "trainable": any(trainable_values),
                }
            )
            stats["tool_call_messages"] += len(group)
            stats["tool_call_groups"] += 1
            i = j
            continue

        if role in {"tool", "tool_response"}:
            if pending_tool_call_ids:
                tool_call_id = pending_tool_call_ids.pop(0)
            else:
                tool_call_id = f"call_orphan_{row_index:06d}_{i:05d}"
                stats["orphan_tool_results"] += 1
            converted.append(
                {
                    "role": "tool",
                    "content": converted_content,
                    "tool_call_id": tool_call_id,
                    "trainable": False,
                }
            )
            if role == "tool_response":
                stats["tool_response_renamed"] += 1
            i += 1
            continue

        if role in {"system", "user"}:
            converted.append(
                {"role": role, "content": converted_content, "trainable": False}
            )
            i += 1
            continue

        if role == "assistant":
            converted.append(
                {
                    "role": "assistant",
                    "content": converted_content,
                    "trainable": _is_trainable(message),
                }
            )
            i += 1
            continue

        converted.append(
            {
                "role": "user",
                "content": f"[{role}]\n{content}",
                "trainable": False,
            }
        )
        stats["unknown_roles"] += 1
        i += 1

    if pending_tool_call_ids:
        stats["unconsumed_tool_call_ids"] += len(pending_tool_call_ids)
    if image_cursor[0] < len(image_values):
        stats["unused_images"] += len(image_values) - image_cursor[0]

    return converted


def convert_row(
    row: dict[str, Any],
    *,
    row_index: int,
    stats: Counter[str],
    skip_images: bool,
    skip_mixed_tool_call_loss: bool,
) -> dict[str, Any] | None:
    if skip_images and row.get("images"):
        stats["skipped_image_rows"] += 1
        return None
    images = row.get("images") or []
    if images:
        stats["image_rows_converted"] += 1

    tools = _parse_json_maybe(row.get("tools"), [])
    if not isinstance(tools, list):
        stats["bad_tools_field"] += 1
        tools = []

    out = {
        k: v
        for k, v in row.items()
        if k not in {"messages", "tools", "images"}
    }
    row_stats: Counter[str] = Counter()
    out["messages"] = _convert_messages(
        row.get("messages") or [],
        row_index=row_index,
        images=images,
        stats=row_stats,
    )
    stats.update(row_stats)
    if skip_mixed_tool_call_loss and row_stats["mixed_tool_call_loss_groups"]:
        stats["skipped_mixed_tool_call_loss_rows"] += 1
        return None
    if tools:
        out["tools"] = tools
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ms-swift agent JSONL to Fireworks/Tinker chat JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no cap")
    parser.add_argument(
        "--include-images",
        action="store_true",
        help=(
            "Convert rows with images by replacing <image> placeholders with "
            "OpenAI-style image_url content parts. Default skips image rows."
        ),
    )
    parser.add_argument(
        "--allow-mixed-tool-call-loss",
        action="store_true",
        help=(
            "Keep rows where one assistant tool_call group mixes loss=0 and "
            "trainable calls. By default these rows are skipped because "
            "Fireworks trainable is message-level."
        ),
    )
    args = parser.parse_args()

    stats: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.input.open(encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for row_index, line in enumerate(src):
            if args.max_rows and written >= args.max_rows:
                break
            row = json.loads(line)
            stats["input_rows_seen"] += 1
            converted = convert_row(
                row,
                row_index=row_index,
                stats=stats,
                skip_images=not args.include_images,
                skip_mixed_tool_call_loss=not args.allow_mixed_tool_call_loss,
            )
            if converted is None:
                continue
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
            stats["written_rows"] += 1

    print(_json_dumps(dict(stats)))
    print(f"wrote={written} output={args.output}")


if __name__ == "__main__":
    main()

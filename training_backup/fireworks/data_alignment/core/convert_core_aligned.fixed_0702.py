#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


IMAGE_PLACEHOLDER = "<image>"


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_maybe(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return fallback
    return value


def jdump(value: Any, *, unicode_args: bool) -> str:
    return json.dumps(value, ensure_ascii=not unicode_args, separators=(",", ":"))


def is_trainable(message: Mapping[str, Any], *, loss_to_trainable: bool) -> bool:
    if not loss_to_trainable:
        return True
    return message.get("loss") != 0


def normalize_think_whitespace(content: str) -> str:
    if "<think>" not in content or "</think>" not in content:
        return content
    content = re.sub(r"<think>(?!\n)", "<think>\n", content)
    content = re.sub(r"(?<!\n)</think>", "\n</think>", content)
    content = re.sub(r"</think>\n(?!\n)", "</think>\n\n", content)
    return content


def content_with_image_parts(content: Any, images: list[str], cursor: list[int], stats: Counter[str]) -> Any:
    if not isinstance(content, str) or IMAGE_PLACEHOLDER not in content:
        return content
    parts: list[dict[str, Any]] = []
    text_parts = content.split(IMAGE_PLACEHOLDER)
    for index, text in enumerate(text_parts):
        if text:
            parts.append({"type": "text", "text": text})
        if index == len(text_parts) - 1:
            continue
        if cursor[0] >= len(images):
            stats["missing_images_for_placeholders"] += 1
            parts.append({"type": "text", "text": IMAGE_PLACEHOLDER})
            continue
        parts.append({"type": "image_url", "image_url": {"url": images[cursor[0]]}})
        cursor[0] += 1
        stats["image_placeholders_converted"] += 1
    return parts


def parse_tool_call(content: Any) -> tuple[str, Any]:
    payload = parse_json_maybe(content, {})
    if not isinstance(payload, dict):
        payload = {"name": "unknown_tool", "arguments": {"raw": content}}
    name = payload.get("name") or payload.get("tool_name") or "unknown_tool"
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    return str(name), arguments


def has_mixed_tool_call_loss(messages: list[dict[str, Any]], *, loss_to_trainable: bool) -> bool:
    i = 0
    while i < len(messages):
        if messages[i].get("role") != "tool_call":
            i += 1
            continue
        group = []
        j = i
        while j < len(messages) and messages[j].get("role") == "tool_call":
            group.append(messages[j])
            j += 1
        if len({is_trainable(m, loss_to_trainable=loss_to_trainable) for m in group}) > 1:
            return True
        i = j
    return False


def row_skip_reason(row: dict[str, Any], messages: list[dict[str, Any]], args: argparse.Namespace) -> str | None:
    if args.skip_image_rows and row.get("images"):
        return "image_row"
    if args.mixed_tool_call_loss_policy == "skip" and has_mixed_tool_call_loss(messages, loss_to_trainable=args.loss_to_trainable):
        return "mixed_tool_call_loss"
    return None


def record_skip(stats: Counter[str], reason: str) -> None:
    if reason == "image_row":
        stats["skipped_image_rows"] += 1
    elif reason == "mixed_tool_call_loss":
        stats["mixed_tool_call_loss_groups"] += 1
        stats["skipped_mixed_tool_call_loss_rows"] += 1
    else:
        stats["skipped_rows"] += 1


def adapter_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "parse_tools": args.parse_tools,
        "loss_to_trainable": args.loss_to_trainable,
        "tool_call_to_assistant_tool_calls": args.tool_call_to_assistant_tool_calls,
        "tool_response_linking": args.tool_response_linking,
        "tool_response_name_linking": args.tool_response_name_linking,
        "include_images": args.include_images,
        "mixed_tool_call_loss_policy": args.mixed_tool_call_loss_policy,
        "no_user_turn_split": args.no_user_turn_split,
        "preserve_tools_wrapper": args.preserve_tools_wrapper,
        "merge_adjacent_assistant_tool_call": args.merge_adjacent_assistant_tool_call,
        "merge_consecutive_tool_responses": args.merge_consecutive_tool_responses,
        "ms_swift_tool_call_whitespace": args.ms_swift_tool_call_whitespace,
        "tool_call_only_empty_think": args.tool_call_only_empty_think,
        "unicode_argument_serialization": args.unicode_argument_serialization,
        "think_block_newline_style": args.think_block_newline_style,
        "max_rows": args.max_rows,
        "skip_image_rows": args.skip_image_rows,
    }


def finalize_stats(stats: Counter[str], args: argparse.Namespace) -> dict[str, Any]:
    for key in [
        "bad_tools_field",
        "missing_tools",
        "mixed_tool_call_loss_groups",
        "skipped_rows",
        "unknown_roles",
        "missing_images_for_placeholders",
        "unused_images",
        "orphan_tool_results",
        "unconsumed_tool_call_ids",
        "tool_response_renamed",
        "tool_response_names_linked",
        "skipped_image_rows",
        "skipped_tool_call_groups",
        "skipped_mixed_tool_call_loss_rows",
        "image_rows_converted",
        "consecutive_tool_response_groups_merged",
        "consecutive_tool_response_messages_merged",
    ]:
        stats.setdefault(key, 0)
    report = dict(stats)
    report["adapter_modules"] = adapter_config(args)
    return report


def prepare_content(content: Any, *, images: list[str], image_cursor: list[int], args: argparse.Namespace, stats: Counter[str]) -> Any:
    if args.include_images:
        content = content_with_image_parts(content, images, image_cursor, stats)
    if isinstance(content, str):
        content = content.rstrip()
        if args.think_block_newline_style:
            content = normalize_think_whitespace(content)
    return content


def collect_tool_call_group(messages: list[dict[str, Any]], start_index: int) -> tuple[list[dict[str, Any]], int]:
    end_index = start_index + 1
    while end_index < len(messages) and messages[end_index].get("role") == "tool_call":
        end_index += 1
    return messages[start_index:end_index], end_index


def build_tool_calls(
    group: list[dict[str, Any]],
    *,
    row_index: int,
    start_index: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]], set[bool]]:
    tool_calls: list[dict[str, Any]] = []
    pending_calls: list[tuple[str, str]] = []
    trainable_values = {is_trainable(m, loss_to_trainable=args.loss_to_trainable) for m in group}
    for offset, tool_msg in enumerate(group):
        name, arguments = parse_tool_call(tool_msg.get("content"))
        call_id = f"call_{row_index:06d}_{start_index:05d}_{offset:02d}"
        pending_calls.append((call_id, name))
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else jdump(arguments, unicode_args=args.unicode_argument_serialization),
                },
            }
        )
    return tool_calls, pending_calls, trainable_values


def append_tool_call_turn(
    converted: list[dict[str, Any]],
    *,
    tool_calls: list[dict[str, Any]],
    trainable_values: set[bool],
    args: argparse.Namespace,
) -> None:
    trainable = any(trainable_values)
    if args.merge_adjacent_assistant_tool_call and converted and converted[-1].get("role") == "assistant":
        converted[-1].setdefault("tool_calls", [])
        converted[-1]["tool_calls"].extend(tool_calls)
        converted[-1]["trainable"] = bool(converted[-1].get("trainable")) or trainable
        return

    assistant_content = "<think>\n\n</think>\n\n" if args.tool_call_only_empty_think else ""
    converted.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls, "trainable": trainable})


def append_tool_response(
    converted: list[dict[str, Any]],
    *,
    role: str,
    content: Any,
    row_index: int,
    message_index: int,
    pending_tool_call_ids: list[tuple[str, str]],
    args: argparse.Namespace,
    stats: Counter[str],
) -> None:
    if pending_tool_call_ids:
        tool_call_id, tool_name = pending_tool_call_ids.pop(0)
    else:
        tool_call_id = f"call_orphan_{row_index:06d}_{message_index:05d}"
        tool_name = ""
        stats["orphan_tool_results"] += 1

    tool_message = {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id if args.tool_response_linking else f"call_unlinked_{row_index:06d}_{message_index:05d}",
        "trainable": False,
    }
    if getattr(args, "tool_response_name_linking", True) and tool_name:
        tool_message["name"] = tool_name
        stats["tool_response_names_linked"] += 1
    converted.append(tool_message)
    if role == "tool_response":
        stats["tool_response_renamed"] += 1


def append_regular_message(
    converted: list[dict[str, Any]],
    *,
    role: str,
    content: Any,
    msg: dict[str, Any],
    args: argparse.Namespace,
    stats: Counter[str],
) -> None:
    if role in {"system", "user"}:
        converted.append({"role": role, "content": content, "trainable": False})
    elif role == "assistant":
        converted.append({"role": "assistant", "content": content, "trainable": is_trainable(msg, loss_to_trainable=args.loss_to_trainable)})
    else:
        converted.append({"role": "user", "content": f"[{role}]\n{content}", "trainable": False})
        stats["unknown_roles"] += 1


def append_tool_call_group(
    converted: list[dict[str, Any]],
    pending_tool_call_ids: list[tuple[str, str]],
    *,
    group: list[dict[str, Any]],
    row_index: int,
    start_index: int,
    args: argparse.Namespace,
    stats: Counter[str],
) -> None:
    tool_calls, pending_calls, trainable_values = build_tool_calls(group, row_index=row_index, start_index=start_index, args=args)
    if len(trainable_values) > 1:
        stats["mixed_tool_call_loss_groups"] += 1
        if args.mixed_tool_call_loss_policy == "skip":
            stats["skipped_tool_call_groups"] += 1
            return
    pending_tool_call_ids.extend(pending_calls)
    append_tool_call_turn(converted, tool_calls=tool_calls, trainable_values=trainable_values, args=args)
    stats["tool_call_messages"] += len(group)
    stats["tool_call_groups"] += 1


def merge_consecutive_tool_responses(
    converted: list[dict[str, Any]],
    *,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    """Collapse runs of role="tool" messages into a single tool message so that
    the Qwen3 renderer emits one user turn with multiple <tool_response> blocks
    (matching ms-swift template behavior).

    Merge strategy: content parts are joined with the sentinel
    "\\n</tool_response>\\n<tool_response>\\n" so that the renderer's outer
    wrapping `<tool_response>\\n{content}\\n</tool_response>` produces the exact
    ms-swift byte layout. Only string-content messages participate; if any
    message in a run has non-string content (e.g. image parts), that run is
    left untouched.
    """
    if not converted:
        return converted
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(converted):
        msg = converted[i]
        if msg.get("role") != "tool":
            merged.append(msg)
            i += 1
            continue
        j = i + 1
        while j < len(converted) and converted[j].get("role") == "tool":
            j += 1
        group = converted[i:j]
        # Skip merging if any content is non-string (list of image parts, etc.)
        if len(group) == 1 or any(not isinstance(m.get("content"), str) for m in group):
            merged.extend(group)
            i = j
            continue
        joined_content = "\n</tool_response>\n<tool_response>\n".join(
            m["content"] for m in group
        )
        merged.append({
            "role": "tool",
            "content": joined_content,
            "tool_call_id": group[0].get("tool_call_id"),
            "trainable": False,
        })
        stats["consecutive_tool_response_groups_merged"] += 1
        stats["consecutive_tool_response_messages_merged"] += len(group)
        i = j
    return merged


def convert_messages(messages: list[dict[str, Any]], *, row_index: int, images: list[str], args: argparse.Namespace, stats: Counter[str]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending_tool_call_ids: list[tuple[str, str]] = []
    image_cursor = [0]
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        content = msg.get("content", "") or ""
        content = prepare_content(content, images=images, image_cursor=image_cursor, args=args, stats=stats)

        if role == "tool_call" and args.tool_call_to_assistant_tool_calls:
            group, j = collect_tool_call_group(messages, i)
            append_tool_call_group(
                converted,
                pending_tool_call_ids,
                group=group,
                row_index=row_index,
                start_index=i,
                args=args,
                stats=stats,
            )
            i = j
            continue

        if role in {"tool", "tool_response"}:
            append_tool_response(
                converted,
                role=role,
                content=content,
                row_index=row_index,
                message_index=i,
                pending_tool_call_ids=pending_tool_call_ids,
                args=args,
                stats=stats,
            )
            i += 1
            continue

        append_regular_message(converted, role=role, content=content, msg=msg, args=args, stats=stats)
        i += 1

    if args.include_images and image_cursor[0] < len(images):
        stats["unused_images"] += len(images) - image_cursor[0]
    if pending_tool_call_ids:
        stats["unconsumed_tool_call_ids"] += len(pending_tool_call_ids)
    if args.merge_consecutive_tool_responses:
        converted = merge_consecutive_tool_responses(converted, stats=stats)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ms-swift JSONL rows into Fireworks chat rows with switchable alignment adapters.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no cap")
    parser.add_argument("--skip-image-rows", action="store_true", help="Skip rows with top-level images instead of converting them")
    parser.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parse-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-to-trainable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tool-call-to-assistant-tool-calls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tool-response-linking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tool-response-name-linking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-user-turn-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-tools-wrapper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-adjacent-assistant-tool-call", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-consecutive-tool-responses", action=argparse.BooleanOptionalAction, default=True,
                        help="Fold runs of consecutive tool messages into one tool message so the Qwen3 renderer emits a single user turn with multiple <tool_response> blocks (ms-swift parity).")
    parser.add_argument("--ms-swift-tool-call-whitespace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tool-call-only-empty-think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unicode-argument-serialization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--think-block-newline-style", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mixed-tool-call-loss-policy", choices=["report", "skip"], default="report")
    args = parser.parse_args()

    stats: Counter[str] = Counter()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for row_index, line in enumerate(inp):
            if not line.strip():
                continue
            if args.max_rows and stats["written_rows"] >= args.max_rows:
                break
            row = json.loads(line)
            stats["input_rows_seen"] += 1
            messages = row.get("messages") or []
            skip_reason = row_skip_reason(row, messages, args)
            if skip_reason:
                record_skip(stats, skip_reason)
                continue
            images = row.get("images") or []
            if images:
                stats["image_rows_seen"] += 1
                stats["image_rows_converted"] += 1
            tools = parse_json_maybe(row.get("tools"), []) if args.parse_tools else row.get("tools")
            if tools and not isinstance(tools, list):
                stats["bad_tools_field"] += 1
                tools = []
            if not tools:
                stats["missing_tools"] += 1
            row_stats: Counter[str] = Counter()
            converted = {k: v for k, v in row.items() if k not in {"messages", "tools", "images"}}
            converted["messages"] = convert_messages(messages, row_index=row_index, images=images, args=args, stats=row_stats)
            if tools:
                converted["tools"] = tools
            converted["_alignment_adapters"] = adapter_config(args)
            stats.update(row_stats)
            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["written_rows"] += 1

    report = finalize_stats(stats, args)
    dump_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

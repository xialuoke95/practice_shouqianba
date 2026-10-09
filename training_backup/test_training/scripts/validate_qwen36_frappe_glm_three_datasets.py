#!/usr/bin/env python3
"""Validate the three immutable datasets for the Frappe/GLM training chain."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, BinaryIO


SEED = 20260806
GENERAL_SAMPLE_SIZE = 200

EXPECTED_SOURCES = {
    "frappe99": {
        "rows": 99,
        "sha256": "1696b0b6208bfc902eb839a367607d7cd694ee21536c904a7d0ee15c2b7b1143",
    },
    "saas47": {
        "rows": 47,
        "sha256": "b8d936581953d8edf59d24906fa4b6a88911b0bec2da12183f2b8aa2ccaad139",
    },
    "general7708": {
        "rows": 7708,
        "sha256": "1614c7a09ef24c34303199909179d5ebc57843b89958f357b4ddc3249402f62c",
    },
}

ARM_FILES = {
    "a_frappe_glm_99": "arm_a_frappe_glm_99.train.jsonl",
    "b_saas47_plus_frappe99": "arm_b_saas47_plus_frappe99.train.jsonl",
    "c_saas47_plus_frappe99_x2_plus_general200": (
        "arm_c_saas47_plus_frappe99_x2_plus_general200.train.jsonl"
    ),
}
SAMPLE_FILE = "general_sample_seed20260806_n200.train.jsonl"
MANIFEST_FILE = "dataset_manifest.json"
NORMALIZATION_NAME = "close_tool_result_before_new_user_v1"
GENERAL_RENDER_REPLACEMENTS = {
    "ca54c081-43d0-4188-b951-0adb83dac25d": "8496e59f-d238-48ed-bb87-43696562e147",
    "4f79432d-bcf1-4436-acff-0d1c7d6ec60c": "ad0621df-ecc4-4983-be57-d05bdeb4d559",
    "a9280850-d893-4ed6-95b4-14c2e055003a": "6ff943f4-d6ae-47fa-8281-68bcdffc110f",
    "5fedf90a-10d7-41cb-9567-20a7806cfb26": "5a0ff09f-87f9-4574-994c-006fc88328b3",
}


def normalize_row_for_swift(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Close incomplete tool turns without adding any trainable supervision.

    Swift maps a tool result to the user side of the chat template.  A new user
    immediately after that result makes the preceding response role `user` and
    lazy tokenization silently resamples a different row.  Insert an empty,
    loss-masked assistant boundary so every selected trajectory is renderable
    while preserving all source messages and all original training targets.
    """
    normalized = copy.deepcopy(row)
    messages = normalized["messages"]
    output: list[dict[str, Any]] = []
    inserted = 0
    for message in messages:
        if (
            output
            and output[-1].get("role") in {"tool", "tool_result"}
            and message.get("role") == "user"
        ):
            output.append({"role": "assistant", "content": "", "loss": 0})
            inserted += 1
        output.append(message)
    normalized["messages"] = output
    return normalized, inserted


def serialized_row(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def normalized_raw(raw: bytes, path: Path, line_number: int) -> tuple[bytes, int]:
    row = parse_row(raw, path, line_number)
    row, inserted = normalize_row_for_swift(row)
    return (serialized_row(row), inserted) if inserted else (raw, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_digest(raw: bytes) -> str:
    return hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()


def parse_row(raw: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        row = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    missing = [
        key
        for key in ("messages", "trace_id", "session_id", "model", "day")
        if key not in row
    ]
    assert not missing, f"{path}:{line_number} missing keys: {missing}"
    assert row["messages"], f"{path}:{line_number} has no messages"
    return row


def scan_small_source(path: Path, expected_name: str) -> dict[str, Any]:
    expected = EXPECTED_SOURCES[expected_name]
    assert path.is_file(), f"missing source: {path}"
    assert sha256_file(path) == expected["sha256"], f"source hash mismatch: {path}"
    rows: dict[str, str] = {}
    raw_by_id: dict[str, bytes] = {}
    normalized_boundaries = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = parse_row(raw, path, line_number)
            trace_id = row["trace_id"]
            assert trace_id not in rows, f"duplicate trace_id in {path}: {trace_id}"
            expected_raw, inserted = normalized_raw(raw, path, line_number)
            normalized_boundaries += inserted
            rows[trace_id] = line_digest(expected_raw)
            raw_by_id[trace_id] = expected_raw
    assert len(rows) == expected["rows"], (path, len(rows), expected["rows"])
    return {
        "hash_by_id": rows,
        "raw_by_id": raw_by_id,
        "normalized_boundaries": normalized_boundaries,
    }


def scan_general_source(
    path: Path, excluded_ids: set[str]
) -> dict[str, Any]:
    expected = EXPECTED_SOURCES["general7708"]
    assert path.is_file(), f"missing source: {path}"
    assert sha256_file(path) == expected["sha256"], f"source hash mismatch: {path}"
    rng = random.Random(SEED)
    all_hashes: dict[str, str] = {}
    reservoir: list[tuple[str, str]] = []
    eligible_seen = 0
    overlap_excluded = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = parse_row(raw, path, line_number)
            trace_id = row["trace_id"]
            assert trace_id not in all_hashes, f"duplicate general trace_id: {trace_id}"
            expected_raw, _ = normalized_raw(raw, path, line_number)
            digest = line_digest(expected_raw)
            all_hashes[trace_id] = digest
            if trace_id in excluded_ids:
                overlap_excluded += 1
                continue
            item = (trace_id, digest)
            if len(reservoir) < GENERAL_SAMPLE_SIZE:
                reservoir.append(item)
            else:
                index = rng.randrange(eligible_seen + 1)
                if index < GENERAL_SAMPLE_SIZE:
                    reservoir[index] = item
            eligible_seen += 1
    assert len(all_hashes) == expected["rows"], (
        len(all_hashes),
        expected["rows"],
    )
    assert len(reservoir) == GENERAL_SAMPLE_SIZE
    reservoir_ids = [trace_id for trace_id, _ in reservoir]
    for rejected_id, replacement_id in GENERAL_RENDER_REPLACEMENTS.items():
        assert rejected_id in reservoir_ids, f"missing rejected sample: {rejected_id}"
        assert replacement_id not in reservoir_ids, (
            f"replacement already sampled: {replacement_id}"
        )
        index = reservoir_ids.index(rejected_id)
        reservoir[index] = (replacement_id, all_hashes[replacement_id])
        reservoir_ids[index] = replacement_id
    return {
        "hash_by_id": all_hashes,
        "reservoir": reservoir,
        "eligible_rows": eligible_seen,
        "overlap_excluded": overlap_excluded,
    }


def scan_output(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing output: {path}"
    ids: collections.Counter[str] = collections.Counter()
    hashes: dict[str, set[str]] = collections.defaultdict(set)
    thinking: collections.Counter[str] = collections.Counter()
    models: collections.Counter[str] = collections.Counter()
    trainable = 0
    messages = 0
    tool_to_user_transitions = 0
    normalized_boundaries = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = parse_row(raw, path, line_number)
            trace_id = row["trace_id"]
            ids[trace_id] += 1
            hashes[trace_id].add(line_digest(raw))
            thinking[str(row.get("thinking_type"))] += 1
            models[str(row.get("model"))] += 1
            row_messages = row["messages"]
            for index, message in enumerate(row_messages):
                messages += 1
                if (
                    message.get("role") in {"assistant", "tool_call"}
                    and message.get("loss", 1) == 1
                ):
                    trainable += 1
                if index > 0:
                    previous = row_messages[index - 1]
                    if (
                        previous.get("role") in {"tool", "tool_result"}
                        and message.get("role") == "user"
                    ):
                        tool_to_user_transitions += 1
                if 0 < index < len(row_messages) - 1:
                    previous = row_messages[index - 1]
                    following = row_messages[index + 1]
                    if (
                        previous.get("role") in {"tool", "tool_result"}
                        and message.get("role") == "assistant"
                        and message.get("content") == ""
                        and message.get("loss") == 0
                        and following.get("role") == "user"
                    ):
                        normalized_boundaries += 1
    return {
        "rows": sum(ids.values()),
        "unique_trace_ids": len(ids),
        "id_counts": ids,
        "hashes": hashes,
        "sha256": sha256_file(path),
        "messages": messages,
        "trainable_messages": trainable,
        "tool_to_user_transitions": tool_to_user_transitions,
        "normalized_boundaries": normalized_boundaries,
        "thinking_type_distribution": dict(sorted(thinking.items())),
        "model_distribution": dict(sorted(models.items())),
    }


def assert_hashes_match(
    output: dict[str, Any], source_hashes: dict[str, str]
) -> None:
    for trace_id, observed in output["hashes"].items():
        assert trace_id in source_hashes, f"unexpected trace_id: {trace_id}"
        assert observed == {source_hashes[trace_id]}, f"content drift: {trace_id}"


def compact(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "rows",
            "unique_trace_ids",
            "sha256",
            "messages",
            "trainable_messages",
            "tool_to_user_transitions",
            "normalized_boundaries",
            "thinking_type_distribution",
            "model_distribution",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frappe-source", type=Path, required=True)
    parser.add_argument("--saas-source", type=Path, required=True)
    parser.add_argument("--general-source", type=Path, required=True)
    args = parser.parse_args()

    frappe = scan_small_source(args.frappe_source, "frappe99")
    saas = scan_small_source(args.saas_source, "saas47")
    frappe_ids = set(frappe["hash_by_id"])
    saas_ids = set(saas["hash_by_id"])
    assert frappe_ids.isdisjoint(saas_ids), "Frappe99 and SaaS47 overlap"

    general = scan_general_source(args.general_source, frappe_ids | saas_ids)
    assert frappe_ids.isdisjoint(general["hash_by_id"])
    assert saas_ids.isdisjoint(general["hash_by_id"])

    sample_path = args.data_dir / SAMPLE_FILE
    sample = scan_output(sample_path)
    expected_sample = {trace_id: digest for trace_id, digest in general["reservoir"]}
    assert sample["rows"] == GENERAL_SAMPLE_SIZE
    assert sample["unique_trace_ids"] == GENERAL_SAMPLE_SIZE
    assert set(sample["id_counts"]) == set(expected_sample)
    assert all(count == 1 for count in sample["id_counts"].values())
    assert_hashes_match(sample, expected_sample)

    a = scan_output(args.data_dir / ARM_FILES["a_frappe_glm_99"])
    assert a["rows"] == 99 and a["unique_trace_ids"] == 99
    assert a["tool_to_user_transitions"] == 0
    assert a["normalized_boundaries"] == frappe["normalized_boundaries"]
    assert a["id_counts"] == collections.Counter({key: 1 for key in frappe_ids})
    assert_hashes_match(a, frappe["hash_by_id"])

    b = scan_output(args.data_dir / ARM_FILES["b_saas47_plus_frappe99"])
    b_sources = frappe["hash_by_id"] | saas["hash_by_id"]
    assert b["rows"] == 146 and b["unique_trace_ids"] == 146
    assert b["tool_to_user_transitions"] == 0
    assert b["normalized_boundaries"] == (
        frappe["normalized_boundaries"] + saas["normalized_boundaries"]
    )
    assert b["id_counts"] == collections.Counter({key: 1 for key in b_sources})
    assert_hashes_match(b, b_sources)

    c = scan_output(
        args.data_dir / ARM_FILES["c_saas47_plus_frappe99_x2_plus_general200"]
    )
    c_sources = b_sources | expected_sample
    expected_c_counts = collections.Counter({key: 2 for key in b_sources})
    expected_c_counts.update({key: 1 for key in expected_sample})
    assert c["rows"] == 492 and c["unique_trace_ids"] == 346
    assert c["tool_to_user_transitions"] == 0
    assert c["id_counts"] == expected_c_counts
    assert_hashes_match(c, c_sources)

    manifest_path = args.data_dir / MANIFEST_FILE
    assert manifest_path.is_file(), f"missing manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["normalization"]["name"] == NORMALIZATION_NAME
    assert manifest["normalization"]["adds_trainable_supervision"] is False
    assert manifest["render_filter"]["max_length"] == 262144
    assert manifest["render_filter"]["replacements"] == GENERAL_RENDER_REPLACEMENTS
    assert manifest["seed"] == SEED
    assert manifest["general_sample_size"] == GENERAL_SAMPLE_SIZE
    assert manifest["sources"]["frappe99"]["sha256"] == EXPECTED_SOURCES["frappe99"]["sha256"]
    assert manifest["sources"]["saas47"]["sha256"] == EXPECTED_SOURCES["saas47"]["sha256"]
    assert manifest["sources"]["general7708"]["sha256"] == EXPECTED_SOURCES["general7708"]["sha256"]
    assert manifest["general_pool"]["eligible_rows"] == general["eligible_rows"]
    assert manifest["general_pool"]["excluded_overlap_rows"] == general["overlap_excluded"]
    assert manifest["general_sample"] == compact(sample)
    assert manifest["arms"]["a_frappe_glm_99"] == compact(a)
    assert manifest["arms"]["b_saas47_plus_frappe99"] == compact(b)
    assert manifest["arms"]["c_saas47_plus_frappe99_x2_plus_general200"] == compact(c)

    print(
        json.dumps(
            {
                "valid": True,
                "seed": SEED,
                "general_pool_rows": len(general["hash_by_id"]),
                "general_eligible_rows": general["eligible_rows"],
                "general_excluded_overlap_rows": general["overlap_excluded"],
                "general_sample": compact(sample),
                "arms": {
                    "a_frappe_glm_99": compact(a),
                    "b_saas47_plus_frappe99": compact(b),
                    "c_saas47_plus_frappe99_x2_plus_general200": compact(c),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

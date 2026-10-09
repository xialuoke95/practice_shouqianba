#!/usr/bin/env python3
"""Build the three requested Frappe/GLM Qwen3.6 training datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import validate_qwen36_frappe_glm_three_datasets as contract


def normalized(raw: bytes) -> bytes:
    return raw if raw.endswith(b"\n") else raw + b"\n"


def load_small_source(
    path: Path, expected_name: str
) -> tuple[list[bytes], set[str], int]:
    expected = contract.EXPECTED_SOURCES[expected_name]
    if contract.sha256_file(path) != expected["sha256"]:
        raise SystemExit(f"source SHA256 mismatch: {path}")
    rows: list[bytes] = []
    ids: set[str] = set()
    normalized_boundaries = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = contract.parse_row(raw, path, line_number)
            trace_id = row["trace_id"]
            if trace_id in ids:
                raise SystemExit(f"duplicate trace_id in {path}: {trace_id}")
            ids.add(trace_id)
            output_raw, inserted = contract.normalized_raw(raw, path, line_number)
            normalized_boundaries += inserted
            rows.append(normalized(output_raw))
    if len(rows) != expected["rows"]:
        raise SystemExit(
            f"expected {expected['rows']} rows in {path}, got {len(rows)}"
        )
    return rows, ids, normalized_boundaries


def reservoir_sample_general(
    path: Path, excluded_ids: set[str]
) -> tuple[list[bytes], int, int, int, int]:
    expected = contract.EXPECTED_SOURCES["general7708"]
    digest = hashlib.sha256()
    rng = random.Random(contract.SEED)
    reservoir: list[tuple[str, bytes, int]] = []
    replacement_rows: dict[str, tuple[bytes, int]] = {}
    all_ids: set[str] = set()
    eligible_seen = 0
    overlap_excluded = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            row = contract.parse_row(raw, path, line_number)
            trace_id = row["trace_id"]
            if trace_id in all_ids:
                raise SystemExit(f"duplicate general trace_id: {trace_id}")
            all_ids.add(trace_id)
            if trace_id in excluded_ids:
                overlap_excluded += 1
                continue
            output_raw, inserted = contract.normalized_raw(raw, path, line_number)
            normalized_output = normalized(output_raw)
            item = (trace_id, normalized_output, inserted)
            if trace_id in contract.GENERAL_RENDER_REPLACEMENTS.values():
                replacement_rows[trace_id] = (normalized_output, inserted)
            if len(reservoir) < contract.GENERAL_SAMPLE_SIZE:
                reservoir.append(item)
            else:
                index = rng.randrange(eligible_seen + 1)
                if index < contract.GENERAL_SAMPLE_SIZE:
                    reservoir[index] = item
            eligible_seen += 1
    if len(all_ids) != expected["rows"]:
        raise SystemExit(
            f"expected {expected['rows']} rows in {path}, got {len(all_ids)}"
        )
    if digest.hexdigest() != expected["sha256"]:
        raise SystemExit(f"source SHA256 mismatch: {path}")
    if len(reservoir) != contract.GENERAL_SAMPLE_SIZE:
        raise SystemExit(
            f"expected {contract.GENERAL_SAMPLE_SIZE} sampled rows, got {len(reservoir)}"
        )
    reservoir_ids = [trace_id for trace_id, _, _ in reservoir]
    for rejected_id, replacement_id in contract.GENERAL_RENDER_REPLACEMENTS.items():
        if rejected_id not in reservoir_ids:
            raise SystemExit(f"missing rejected sample: {rejected_id}")
        if replacement_id in reservoir_ids:
            raise SystemExit(f"replacement already sampled: {replacement_id}")
        if replacement_id not in replacement_rows:
            raise SystemExit(f"replacement row missing from source: {replacement_id}")
        index = reservoir_ids.index(rejected_id)
        replacement_raw, replacement_inserted = replacement_rows[replacement_id]
        reservoir[index] = (
            replacement_id,
            replacement_raw,
            replacement_inserted,
        )
        reservoir_ids[index] = replacement_id
    sampled_rows = [raw for _, raw, _ in reservoir]
    sampled_normalized_boundaries = sum(
        inserted for _, _, inserted in reservoir
    )
    return (
        sampled_rows,
        len(all_ids),
        eligible_seen,
        overlap_excluded,
        sampled_normalized_boundaries,
    )


def write_atomic(path: Path, rows: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle:
        handle.writelines(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frappe-source", type=Path, required=True)
    parser.add_argument("--saas-source", type=Path, required=True)
    parser.add_argument("--general-source", type=Path, required=True)
    args = parser.parse_args()

    frappe_rows, frappe_ids, frappe_normalized_boundaries = load_small_source(
        args.frappe_source, "frappe99"
    )
    saas_rows, saas_ids, saas_normalized_boundaries = load_small_source(
        args.saas_source, "saas47"
    )
    if not frappe_ids.isdisjoint(saas_ids):
        raise SystemExit("Frappe99 and SaaS47 trace IDs overlap")

    (
        general_sample,
        general_rows,
        eligible_rows,
        overlap_excluded,
        general_sample_normalized_boundaries,
    ) = reservoir_sample_general(args.general_source, frappe_ids | saas_ids)

    arm_a = list(frappe_rows)
    arm_b = list(saas_rows) + list(frappe_rows)
    random.Random(contract.SEED + 1).shuffle(arm_b)
    arm_c = (
        list(saas_rows)
        + list(frappe_rows)
        + list(saas_rows)
        + list(frappe_rows)
        + list(general_sample)
    )
    random.Random(contract.SEED + 2).shuffle(arm_c)

    if len(arm_a) != 99 or len(arm_b) != 146 or len(arm_c) != 492:
        raise SystemExit(
            f"unexpected arm sizes: A={len(arm_a)} B={len(arm_b)} C={len(arm_c)}"
        )

    output_paths = {
        "general_sample": args.output_dir / contract.SAMPLE_FILE,
        **{
            arm_name: args.output_dir / filename
            for arm_name, filename in contract.ARM_FILES.items()
        },
    }
    write_atomic(output_paths["general_sample"], general_sample)
    write_atomic(output_paths["a_frappe_glm_99"], arm_a)
    write_atomic(output_paths["b_saas47_plus_frappe99"], arm_b)
    write_atomic(
        output_paths["c_saas47_plus_frappe99_x2_plus_general200"], arm_c
    )

    sample_stats = contract.scan_output(output_paths["general_sample"])
    arm_stats = {
        name: contract.scan_output(output_paths[name])
        for name in contract.ARM_FILES
    }
    manifest = {
        "schema_version": 2,
        "seed": contract.SEED,
        "general_sample_size": contract.GENERAL_SAMPLE_SIZE,
        "normalization": {
            "name": contract.NORMALIZATION_NAME,
            "rule": "insert an empty loss-masked assistant boundary between a tool result and a following user message",
            "adds_trainable_supervision": False,
            "source_boundaries_inserted": {
                "frappe99": frappe_normalized_boundaries,
                "saas47": saas_normalized_boundaries,
                "general_sample": general_sample_normalized_boundaries,
            },
        },
        "render_filter": {
            "template": "qwen3_5",
            "max_length": 262144,
            "rule": "replace fixed-seed reservoir rows that fail exact full-template rendering with deterministic validated rows",
            "replacements": contract.GENERAL_RENDER_REPLACEMENTS,
        },
        "sources": {
            "frappe99": {
                "path": str(args.frappe_source),
                **contract.EXPECTED_SOURCES["frappe99"],
            },
            "saas47": {
                "path": str(args.saas_source),
                **contract.EXPECTED_SOURCES["saas47"],
            },
            "general7708": {
                "path": str(args.general_source),
                **contract.EXPECTED_SOURCES["general7708"],
            },
        },
        "general_pool": {
            "rows": general_rows,
            "eligible_rows": eligible_rows,
            "excluded_overlap_rows": overlap_excluded,
            "exclusion_rule": "exclude Frappe99 and SaaS47 trace IDs before fixed-seed reservoir sampling",
        },
        "general_sample": contract.compact(sample_stats),
        "arms": {
            name: contract.compact(stats) for name, stats in arm_stats.items()
        },
        "composition": {
            "a_frappe_glm_99": {
                "frappe99_multiplier": 1,
                "saas47_multiplier": 0,
                "general_sample_multiplier": 0,
            },
            "b_saas47_plus_frappe99": {
                "frappe99_multiplier": 1,
                "saas47_multiplier": 1,
                "general_sample_multiplier": 0,
            },
            "c_saas47_plus_frappe99_x2_plus_general200": {
                "frappe99_multiplier": 2,
                "saas47_multiplier": 2,
                "general_sample_multiplier": 1,
            },
        },
        "files": {name: str(path) for name, path in output_paths.items()},
    }
    manifest_path = args.output_dir / contract.MANIFEST_FILE
    temp_manifest = manifest_path.with_suffix(
        manifest_path.suffix + f".tmp.{os.getpid()}"
    )
    temp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the exact-47 BytePlus dataset and launch assets before submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


EXPECTED_ROWS = 47
EXPECTED_SHA256 = "b8d936581953d8edf59d24906fa4b6a88911b0bec2da12183f2b8aa2ccaad139"
EXPECTED_TRAINABLE_MESSAGES = 2621


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path: Path) -> dict[str, int | str]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if "<think" in raw.lower() or "reasoning_content" in raw:
                raise AssertionError(f"thinking residue at line {line_number}")
            rows.append(json.loads(raw))

    assert len(rows) == EXPECTED_ROWS, (len(rows), EXPECTED_ROWS)
    trace_ids = [row["trace_id"] for row in rows]
    assert len(set(trace_ids)) == EXPECTED_ROWS, "trace IDs must be unique"
    assert all(row.get("thinking_type") == "none" for row in rows)
    assert all(row.get("thinking_budget") is None for row in rows)
    assert all(row.get("messages") for row in rows)

    trainable = sum(
        1
        for row in rows
        for message in row["messages"]
        if message.get("role") in {"assistant", "tool_call"}
        and message.get("loss", 1) == 1
    )
    assert trainable == EXPECTED_TRAINABLE_MESSAGES, (
        trainable,
        EXPECTED_TRAINABLE_MESSAGES,
    )
    digest = sha256(path)
    assert digest == EXPECTED_SHA256, (digest, EXPECTED_SHA256)
    return {"rows": len(rows), "trainable_messages": trainable, "sha256": digest}


def validate_summary(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["output_rows"] == EXPECTED_ROWS
    assert summary["output_sha256"] == EXPECTED_SHA256
    assert summary["fireworks_ready_rows"] == EXPECTED_ROWS


def validate_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = [
        "EXPECTED_DATASET_LINES=${EXPECTED_DATASET_LINES:-47}",
        f"EXPECTED_DATASET_SHA256=${{EXPECTED_DATASET_SHA256:-{EXPECTED_SHA256}}}",
        "EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-16}",
        "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}",
        "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}",
        "TP_SIZE=4",
        "PP_SIZE=2",
        "CP_SIZE=2",
        "--optimizer_cpu_offload true",
    ]
    missing = [value for value in required if value not in text]
    assert not missing, f"training script is missing: {missing}"


def validate_task_yaml(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["TaskName"] == "train_saas_sft_v0_exact47_qwen36_27b_16gpu"
    assert "train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2.sh" in config[
        "Entrypoint"
    ]
    roles = config["TaskRoleSpecs"]
    assert roles == [
        {
            "RoleName": "worker",
            "RoleReplicas": 2,
            "Flavor": "ml.hpcpni3h.42xlarge",
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "data" / "saas_sft_v0_volc_exact47"
    report = validate_dataset(
        data_dir / "saas_sft_v0_fireworks_exact47.train.jsonl"
    )
    validate_summary(data_dir / "summary.json")
    validate_script(
        root
        / "scripts"
        / "train_saas_sft_v0_exact47_qwen36_27b_16gpu_gbs2_epoch2.sh"
    )
    validate_task_yaml(
        root
        / "docs"
        / "generated"
        / "byteplus_saas_sft_v0_exact47_qwen36_27b_16gpu_task.yaml"
    )
    print(json.dumps({"valid": True, **report}, indent=2))


if __name__ == "__main__":
    main()

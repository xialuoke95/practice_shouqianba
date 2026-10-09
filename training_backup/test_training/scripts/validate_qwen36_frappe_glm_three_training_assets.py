#!/usr/bin/env python3
"""Validate scripts and task YAMLs for the three-run BytePlus chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ARMS = {
    "a": {
        "task_name": "train_qwen36_frappe_glm_a99_16gpu_v3",
        "yaml": "byteplus_qwen36_frappe_glm_a99_16gpu_task.yaml",
        "rows": 99,
        "unique": 99,
        "trainable": 8999,
        "sha256": "4c315dcebf9e2b00bcab14075478852bddcc1077fabadfb555f52f1170101b44",
        "steps": 99,
        "save_steps": 50,
        "thinking_enabled": 99,
        "thinking_none": 0,
        "singletons": 99,
        "duplicates": 0,
        "max_multiplicity": 1,
        "dataset_dir": "qwen36_frappe_glm_three_runs_20260806_v3",
    },
    "b": {
        "task_name": "train_qwen36_saas47_frappe99_b146_16gpu_v3",
        "yaml": "byteplus_qwen36_saas47_frappe99_b146_16gpu_task.yaml",
        "rows": 146,
        "unique": 146,
        "trainable": 11620,
        "sha256": "8cb425c7cb234d43292f6b439d38576fe50295bdfa3757794300b196816e2d93",
        "steps": 146,
        "save_steps": 73,
        "thinking_enabled": 99,
        "thinking_none": 47,
        "singletons": 146,
        "duplicates": 0,
        "max_multiplicity": 1,
        "dataset_dir": "qwen36_frappe_glm_three_runs_20260806_v3",
    },
    "c": {
        "task_name": "train_qwen36_saas47_frappe99_x2_general200_c492_16gpu_v4",
        "yaml": "byteplus_qwen36_saas47_frappe99_x2_general200_c492_16gpu_task.yaml",
        "rows": 492,
        "unique": 346,
        "trainable": 36656,
        "sha256": "851ab2a30f50b4b5e5aea4fdba1639acf626afdc5f8343daafd2672f77fba425",
        "steps": 492,
        "save_steps": 246,
        "thinking_enabled": 398,
        "thinking_none": 94,
        "singletons": 200,
        "duplicates": 146,
        "max_multiplicity": 2,
        "dataset_dir": "qwen36_frappe_glm_three_runs_20260806_v4",
    },
}

GENERIC_SCRIPT = "train_qwen36_27b_16gpu_gbs2_epoch2_dataset_arm.sh"
CHAIN_SCRIPT = "run_qwen36_frappe_glm_three_task_chain.py"
REMOTE_SCRIPT = (
    "/pfs-verdent/subingtian/coding_agent_models/scripts/saas/" + GENERIC_SCRIPT
)


def require(text: str, values: list[str], label: str) -> None:
    missing = [value for value in values if value not in text]
    assert not missing, f"{label} is missing required values: {missing}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()

    manifest = json.loads(
        (
            root
            / "data"
            / "qwen36_frappe_glm_three_runs_20260806_v3"
            / "dataset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["arms"]["a_frappe_glm_99"]["sha256"] == ARMS["a"]["sha256"]
    assert manifest["arms"]["b_saas47_plus_frappe99"]["sha256"] == ARMS["b"]["sha256"]
    c_manifest = json.loads(
        (
            root
            / "data"
            / "qwen36_frappe_glm_three_runs_20260806_v4"
            / "arm_c_arrow_type_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert c_manifest["parent"]["sha256"] == (
        manifest["arms"]["c_saas47_plus_frappe99_x2_plus_general200"]["sha256"]
    )
    assert c_manifest["output"]["sha256"] == ARMS["c"]["sha256"]
    assert c_manifest["transformation"] == {
        "changed_rows": 34,
        "changes_training_content": False,
        "name": "normalize_arrow_numeric_metadata_v1",
        "path": "dedup_stats.trainable_turn_ratio",
        "rule": "convert integer values to numerically equivalent floats",
    }

    generic_path = root / "scripts" / GENERIC_SCRIPT
    generic_text = generic_path.read_text(encoding="utf-8")
    require(
        generic_text,
        [
            'EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-16}',
            'GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}',
            'NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}',
            "TP_SIZE=4",
            "PP_SIZE=2",
            "CP_SIZE=2",
            "--optimizer_cpu_offload true",
            "--logging_steps 1",
            "TRAINING_POSTFLIGHT_OK",
            "EXPECTED_TOTAL_STEPS",
            "EXPECTED_DATASET_SHA256",
            "EXPECTED_DUPLICATED_TRACE_IDS",
        ],
        GENERIC_SCRIPT,
    )

    yaml_paths: list[str] = []
    for arm, spec in ARMS.items():
        path = root / "docs" / "generated" / spec["yaml"]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["TaskName"] == spec["task_name"]
        assert config["TaskRoleSpecs"] == [
            {
                "RoleName": "worker",
                "RoleReplicas": 2,
                "Flavor": "ml.hpcpni3h.42xlarge",
            }
        ]
        assert config["Framework"] == "PyTorchDDP"
        assert config["ResourceQueueID"] == "q-20260128181205-z9l2h"
        entrypoint = config["Entrypoint"]
        require(
            entrypoint,
            [
                REMOTE_SCRIPT,
                spec["dataset_dir"],
                f"EXPECTED_DATASET_LINES={spec['rows']}",
                f"EXPECTED_DATASET_SHA256={spec['sha256']}",
                f"EXPECTED_TRAINABLE_MESSAGES={spec['trainable']}",
                f"EXPECTED_UNIQUE_TRACE_IDS={spec['unique']}",
                f"EXPECTED_SINGLETON_TRACE_IDS={spec['singletons']}",
                f"EXPECTED_DUPLICATED_TRACE_IDS={spec['duplicates']}",
                f"EXPECTED_MAX_TRACE_MULTIPLICITY={spec['max_multiplicity']}",
                f"EXPECTED_THINKING_ENABLED={spec['thinking_enabled']}",
                f"EXPECTED_THINKING_NONE={spec['thinking_none']}",
                f"EXPECTED_TOTAL_STEPS={spec['steps']}",
                f"SAVE_STEPS={spec['save_steps']}",
            ],
            spec["yaml"],
        )
        yaml_paths.append(str(path.relative_to(root)))

    chain_path = root / "scripts" / CHAIN_SCRIPT
    chain_text = chain_path.read_text(encoding="utf-8")
    require(
        chain_text,
        [
            "verify_remote_checkpoint",
            "fatal_matches",
            "missing_shards",
            "SUCCESS_STATUS",
            "FAILURE_STATUSES",
            *yaml_paths,
        ],
        CHAIN_SCRIPT,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "generic_script": str(generic_path),
                "chain_script": str(chain_path),
                "task_yamls": yaml_paths,
                "arms": ARMS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

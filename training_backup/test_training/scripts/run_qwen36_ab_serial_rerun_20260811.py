#!/usr/bin/env python3
"""Run and verify another fresh Arm A then Arm B reproducibility sequence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTROLLER = ROOT / "scripts" / "run_qwen36_frappe_glm_three_task_chain.py"


def load_controller():
    spec = importlib.util.spec_from_file_location("qwen36_ab_serial_20260811_base", BASE_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load controller: {BASE_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    controller = load_controller()
    controller.ARMS = [
        {
            "name": "a_frappe_glm_99_rerun3_20260811",
            "config": (
                "docs/generated/"
                "byteplus_qwen36_frappe_glm_a99_16gpu_rerun3_20260811_task.yaml"
            ),
            "output_dir": (
                "/pfs-verdent/guoyinuo/coding_agent_models/output/"
                "train_qwen36_frappe_glm_a99_16gpu_gbs2_epoch2_20260811_rerun3"
            ),
            "expected_steps": 99,
        },
        {
            "name": "b_saas47_plus_frappe99_rerun4_20260811",
            "config": (
                "docs/generated/"
                "byteplus_qwen36_saas47_frappe99_b146_16gpu_rerun4_20260811_task.yaml"
            ),
            "output_dir": (
                "/pfs-verdent/guoyinuo/coding_agent_models/output/"
                "train_qwen36_saas47_frappe99_b146_16gpu_gbs2_epoch2_20260811_rerun4"
            ),
            "expected_steps": 146,
        },
    ]
    return int(controller.main())


if __name__ == "__main__":
    raise SystemExit(main())

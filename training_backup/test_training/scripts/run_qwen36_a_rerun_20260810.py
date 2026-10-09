#!/usr/bin/env python3
"""Run and verify the Qwen3.6 Arm A reproducibility training for 20260810."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTROLLER = ROOT / "scripts" / "run_qwen36_frappe_glm_three_task_chain.py"


def load_controller():
    spec = importlib.util.spec_from_file_location("qwen36_a_base_controller", BASE_CONTROLLER)
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
            "name": "a_frappe_glm_99_rerun_20260810",
            "config": (
                "docs/generated/"
                "byteplus_qwen36_frappe_glm_a99_16gpu_rerun_20260810_task.yaml"
            ),
            "output_dir": (
                "/pfs-verdent/guoyinuo/coding_agent_models/output/"
                "train_qwen36_frappe_glm_a99_16gpu_gbs2_epoch2_20260810_rerun1"
            ),
            "expected_steps": 99,
        }
    ]
    return int(controller.main())


if __name__ == "__main__":
    raise SystemExit(main())

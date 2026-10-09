#!/usr/bin/env python3
"""Reset only failed Arm C in the sequential chain for the Arrow-safe retry."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ARM = "c_saas47_plus_frappe99_x2_plus_general200"
NEW_CONFIG = (
    "docs/generated/"
    "byteplus_qwen36_saas47_frappe99_x2_general200_c492_16gpu_task.yaml"
)
NEW_OUTPUT = (
    "/pfs-verdent/guoyinuo/coding_agent_models/output/"
    "train_qwen36_saas47_frappe99_x2_general200_c492_16gpu_gbs2_epoch2_20260806_v4"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    assert state["arms"]["a_frappe_glm_99"].get("verified") is True
    assert state["arms"]["b_saas47_plus_frappe99"].get("verified") is True
    old = state["arms"][ARM]
    assert old.get("task_id") == "t-20260807075948-t4jss"
    assert old.get("status") == "Killed"
    assert state.get("terminal_status") == f"{ARM}:Killed"

    prior_attempt = copy.deepcopy(old)
    prior_attempt["failure_reason"] = (
        "HuggingFace/PyArrow JSON schema inference rejected mixed int/float "
        "dedup_stats.trainable_turn_ratio metadata"
    )
    replacement = {
        "config": NEW_CONFIG,
        "output_dir": NEW_OUTPUT,
        "expected_steps": 492,
        "attempts": [prior_attempt],
        "retry_reason": "normalize_arrow_numeric_metadata_v1",
        "retry_prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    state["arms"][ARM] = replacement
    for key in ("terminal_status", "stopped_at", "completed_at"):
        state.pop(key, None)

    print(json.dumps(state, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    temp = args.state.with_suffix(args.state.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, args.state)


if __name__ == "__main__":
    main()

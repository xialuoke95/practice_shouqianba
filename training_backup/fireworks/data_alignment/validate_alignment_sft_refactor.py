#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DATA_ALIGNMENT = Path(__file__).resolve().parent
DEFAULT_OUT = Path("/home/ubuntu/.remotelab/workspace/fireworks-runs/alignment-sft-refactor-validation")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], cwd: Path) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - start, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def stable_summary(run_dir: Path) -> dict[str, Any]:
    cfg = read_json(run_dir / "generated_sft_config.json")
    result = read_json(run_dir / "dry_run_result.json")
    converter = read_json(run_dir / "converter_used.json")
    precheck = read_json(run_dir / "local_renderer_precheck_summary.json")
    conversion_report = read_json(run_dir / "a100_conversion_report.json")
    return {
        "selected_sample_ids": result["selected_sample_ids"],
        "dataset_sha256": result["dataset_sha256"],
        "converter_sha256": converter["converter_sha256"],
        "generated_config": {
            key: value
            for key, value in cfg.items()
            if key not in {"dataset"}
        },
        "precheck": {
            "count": precheck["count"],
            "status_counts": precheck["status_counts"],
            "split_counts": precheck["split_counts"],
            "total_input_ids": precheck["total_input_ids"],
            "total_active_targets": precheck["total_active_targets"],
            "errors": precheck["errors"],
        },
        "conversion_core": {
            key: conversion_report.get(key)
            for key in [
                "rows_read",
                "rows_written",
                "image_rows_seen",
                "image_rows_converted",
                "image_placeholders_converted",
                "skipped_image_rows",
                "missing_images_for_placeholders",
                "unused_images",
                "bad_tools_field",
                "mixed_tool_call_loss_groups",
            ]
            if key in conversion_report
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate run_alignment_sft refactor without submitting Training API jobs.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generic_dir = root / "generic"
    wrapper_dir = root / "qwen_wrapper"
    for path in [generic_dir, wrapper_dir]:
        path.mkdir(parents=True, exist_ok=True)

    common = ["--limit", str(args.limit), "--dry-run"]
    generic_cmd = [sys.executable, str(DATA_ALIGNMENT / "run_alignment_sft.py"), "--run-dir", str(generic_dir), *common]
    wrapper_cmd = [
        sys.executable,
        str(DATA_ALIGNMENT / "run_qwen36_27b_full_sft_preserve_thinking_official.py"),
        "--run-dir",
        str(wrapper_dir),
        *common,
    ]

    runs = {
        "generic": run_cmd(generic_cmd, DATA_ALIGNMENT),
        "qwen_wrapper": run_cmd(wrapper_cmd, DATA_ALIGNMENT),
    }
    errors = []
    for name, result in runs.items():
        if result["returncode"] != 0:
            errors.append(f"{name} exited {result['returncode']}")

    generic_stable = stable_summary(generic_dir) if not errors else {}
    wrapper_stable = stable_summary(wrapper_dir) if not errors else {}
    if not errors and generic_stable != wrapper_stable:
        errors.append("stable dry-run artifacts differ between generic runner and qwen wrapper")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "limit": args.limit,
        "generic_dir": str(generic_dir),
        "wrapper_dir": str(wrapper_dir),
        "runs": runs,
        "generic_stable": generic_stable,
        "wrapper_stable": wrapper_stable,
    }
    report_path = root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "errors": errors}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

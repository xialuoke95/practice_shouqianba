#!/usr/bin/env python3
"""Record read-only progress snapshots for the Volc chained training goal."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_PATH = Path("logs/watch_chain_user_worker_32gpu.state.json")
DEFAULT_JSONL = Path("logs/watch_chain_user_worker_32gpu.progress.jsonl")
DEFAULT_LATEST = Path("logs/watch_chain_user_worker_32gpu.progress.latest.json")
REMOTE_OUTPUT_ROOT = "/pfs-verdent/zhangyu/train_verdent_online/output"

OUTPUT_SUBDIRS = {
    "sampled_16gpu_prereq": "train_full_16gpu_full_sampled_1000_35a3b_te_fused_ce_pinned",
    "debug_32gpu_prereq": "train_debug_32gpu_no_think_full_35a3b_te_fused_ce_gbs72_max180k_iters80",
    "first_user_worker_32gpu": "train_full_32gpu_no_think_user_worker_35a3b_te_fused_ce_gbs72_max180k",
    "second_no_persist_32gpu": "train_full_32gpu_no_think_user_worker_no_persist_35a3b_te_fused_ce_gbs72_max180k",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_last_json(text: str) -> Any:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON payload found in output tail: {text[-1000:]}")


def run_volc(args: list[str]) -> Any:
    volc_args = " ".join(shlex.quote(part) for part in args)
    cmd = [
        "/bin/bash",
        "-lc",
        (
            "eval \"$(grep '^export HuoShan' ~/.bashrc)\" "
            "&& export VOLC_ACCESSKEY=\"$HuoShanAccessKeyID\" "
            "VOLC_SECRETKEY=\"$HuoShanSecretAccessKey\" "
            f"&& volc {volc_args}"
        ),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    if proc.returncode != 0:
        raise RuntimeError(f"volc exited {proc.returncode}: {output[-1500:]}")
    return parse_last_json(output)


def get_task(task_id: str) -> dict[str, Any]:
    payload = run_volc(["ml_task", "get", "--id", task_id, "--output", "json"])
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError(f"unexpected get payload for {task_id}: {payload!r}")
    return payload[0]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def remote_output_summary(subdir: str) -> dict[str, Any]:
    base = f"{REMOTE_OUTPUT_ROOT}/{subdir}"
    remote_script = r'''
import json
import os
import sys
from pathlib import Path


def tail_line(path: Path, window: int = 262144) -> str:
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - window), os.SEEK_SET)
        data = fh.read().decode("utf-8", errors="replace").splitlines()
    return data[-1] if data else ""


base = Path(sys.argv[1])
out = {"base": str(base), "exists": base.exists(), "runs": []}
if base.exists():
    for run in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name):
        logging = run / "logging.jsonl"
        checkpoints = sorted(
            [p for p in run.rglob("checkpoint-*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        item = {
            "run": str(run),
            "logging_exists": logging.exists(),
            "checkpoints": [str(p) for p in checkpoints[-5:]],
        }
        if logging.exists():
            stat = logging.stat()
            item["logging_mtime"] = stat.st_mtime
            item["logging_size"] = stat.st_size
            raw = tail_line(logging)
            item["last_metric_raw"] = raw
            try:
                item["last_metric"] = json.loads(raw)
            except Exception:
                item["last_metric"] = None
        out["runs"].append(item)
print(json.dumps(out, ensure_ascii=False, sort_keys=True))
'''
    proc = subprocess.run(
        ["ssh", "gpu_dev_huoshan", "python3", "-", base],
        input=remote_script,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "base": base,
            "exists": None,
            "error": (proc.stderr or proc.stdout)[-1500:],
        }
    return json.loads(proc.stdout)


def build_snapshot(state_path: Path) -> dict[str, Any]:
    state = load_state(state_path)
    task_ids: list[str] = []
    for task_id in state.get("wait_tasks", []):
        if task_id not in task_ids:
            task_ids.append(str(task_id))
    for key in ["first_task_id", "second_task_id"]:
        task_id = state.get(key)
        if task_id and task_id not in task_ids:
            task_ids.append(str(task_id))

    snapshot: dict[str, Any] = {
        "checked_at": utc_now(),
        "state_path": str(state_path),
        "state": state,
        "tasks": {},
        "outputs": {},
    }

    for task_id in task_ids:
        try:
            record = get_task(task_id)
            snapshot["tasks"][task_id] = {
                "JobId": record.get("JobId"),
                "JobName": record.get("JobName"),
                "Creator": record.get("Creator"),
                "Status": record.get("Status"),
                "Start": record.get("Start"),
                "End": record.get("End"),
                "Elapsed": record.get("Elapsed"),
                "TaskRoleSpecs": record.get("TaskRoleSpecs"),
            }
        except Exception as exc:
            snapshot["tasks"][task_id] = {"error": str(exc)}

    for label, subdir in OUTPUT_SUBDIRS.items():
        try:
            snapshot["outputs"][label] = remote_output_summary(subdir)
        except Exception as exc:
            snapshot["outputs"][label] = {"subdir": subdir, "error": str(exc)}

    return snapshot


def write_snapshot(snapshot: dict[str, Any], jsonl_path: Path, latest_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    with jsonl_path.open("a") as fh:
        fh.write(line + "\n")
    tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(latest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=str(STATE_PATH))
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--latest", default=str(DEFAULT_LATEST))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()

    while True:
        try:
            snapshot = build_snapshot(Path(args.state))
            write_snapshot(snapshot, Path(args.jsonl), Path(args.latest))
            print(f"{utc_now()} wrote {args.latest}", flush=True)
        except Exception as exc:
            print(f"{utc_now()} monitor error: {exc}", file=sys.stderr, flush=True)
            if not args.loop:
                return 1
        if not args.loop:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

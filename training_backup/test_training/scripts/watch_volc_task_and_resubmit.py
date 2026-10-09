#!/usr/bin/env python3
"""Watch a Volc MLP task and resubmit a config once the task succeeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUCCESS_STATUS = "Success"
FAILURE_STATUSES = {"Failed", "Killed", "Killing"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def parse_last_json(text: str) -> Any:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON payload found in command output: {text[-500:]}")


def run_volc(args: list[str]) -> Any:
    cmd = [
        "/bin/bash",
        "-lc",
        (
            "eval \"$(grep '^export HuoShan' ~/.bashrc)\" "
            "&& export VOLC_ACCESSKEY=\"$HuoShanAccessKeyID\" "
            "VOLC_SECRETKEY=\"$HuoShanSecretAccessKey\" "
            "&& volc "
            + " ".join(args)
        ),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    if proc.returncode != 0:
        raise RuntimeError(f"volc exited {proc.returncode}: {output[-1000:]}")
    return parse_last_json(output)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def get_task_status(task_id: str) -> str:
    payload = run_volc(["ml_task", "get", "--id", task_id, "--output", "json"])
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"unexpected get payload: {payload!r}")
    status = payload[0].get("Status")
    if not status:
        raise RuntimeError(f"task status missing in payload: {payload!r}")
    return str(status)


def submit(conf: str) -> str:
    payload = run_volc(["ml_task", "submit", "--conf", conf, "--output", "json"])
    if not isinstance(payload, dict) or not payload.get("Id"):
        raise RuntimeError(f"unexpected submit payload: {payload!r}")
    return str(payload["Id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--conf", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    state_path = Path(args.state)
    state = load_state(state_path)
    if state.get("submitted_task_id"):
        log(f"already submitted {state['submitted_task_id']}; exiting")
        return 0

    log(f"watching task_id={args.task_id} conf={args.conf}")
    while True:
        status = get_task_status(args.task_id)
        state.update(
            {
                "watched_task_id": args.task_id,
                "conf": args.conf,
                "last_status": status,
                "last_checked_at": utc_now(),
            }
        )
        save_state(state_path, state)
        log(f"status={status}")

        if status == SUCCESS_STATUS:
            new_task_id = submit(args.conf)
            state.update(
                {
                    "terminal_status": status,
                    "submitted_task_id": new_task_id,
                    "submitted_at": utc_now(),
                }
            )
            save_state(state_path, state)
            log(f"submitted_task_id={new_task_id}")
            return 0

        if status in FAILURE_STATUSES:
            state.update({"terminal_status": status, "stopped_at": utc_now()})
            save_state(state_path, state)
            log(f"terminal non-success status={status}; not resubmitting")
            return 2

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())

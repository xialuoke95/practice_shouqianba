#!/usr/bin/env python3
"""Watch prerequisite Volc MLP tasks, then submit two configs in sequence."""

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
    raise ValueError(f"no JSON payload found in command output: {text[-1000:]}")


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


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def get_task(task_id: str) -> dict[str, Any]:
    payload = run_volc(["ml_task", "get", "--id", task_id, "--output", "json"])
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"unexpected get payload for {task_id}: {payload!r}")
    if not isinstance(payload[0], dict):
        raise RuntimeError(f"unexpected task record for {task_id}: {payload[0]!r}")
    return payload[0]


def get_task_status(task_id: str) -> str:
    status = get_task(task_id).get("Status")
    if not status:
        raise RuntimeError(f"task status missing for {task_id}")
    return str(status)


def submit(conf: str) -> str:
    payload = run_volc(["ml_task", "submit", "--conf", conf, "--output", "json"])
    if not isinstance(payload, dict) or not payload.get("Id"):
        raise RuntimeError(f"unexpected submit payload for {conf}: {payload!r}")
    return str(payload["Id"])


def update_wait_statuses(state: dict[str, Any], wait_tasks: list[str]) -> tuple[bool, bool]:
    statuses: dict[str, str] = {}
    all_success = True
    any_failed = False
    for task_id in wait_tasks:
        status = get_task_status(task_id)
        statuses[task_id] = status
        if status != SUCCESS_STATUS:
            all_success = False
        if status in FAILURE_STATUSES:
            any_failed = True
    state["wait_tasks"] = wait_tasks
    state["wait_statuses"] = statuses
    state["last_checked_at"] = utc_now()
    return all_success, any_failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-task", action="append", default=[])
    parser.add_argument("--first-conf", required=True)
    parser.add_argument("--second-conf", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()

    state_path = Path(args.state)
    state = load_state(state_path)
    state.setdefault("first_conf", args.first_conf)
    state.setdefault("second_conf", args.second_conf)
    save_state(state_path, state)

    log(
        "watching prerequisites="
        f"{','.join(args.wait_task) if args.wait_task else '<none>'} "
        f"first_conf={args.first_conf} second_conf={args.second_conf}"
    )

    while True:
        state = load_state(state_path)

        if not state.get("first_task_id"):
            all_success, any_failed = update_wait_statuses(state, args.wait_task)
            save_state(state_path, state)
            log(f"wait_statuses={state.get('wait_statuses')}")
            if any_failed:
                state["terminal_status"] = "PrerequisiteFailed"
                state["stopped_at"] = utc_now()
                save_state(state_path, state)
                log("a prerequisite task reached a failure status; not submitting first config")
                return 2
            if not all_success:
                time.sleep(args.poll_seconds)
                continue

            first_task_id = submit(args.first_conf)
            state["first_task_id"] = first_task_id
            state["first_submitted_at"] = utc_now()
            save_state(state_path, state)
            log(f"first_task_id={first_task_id}")

        first_status = get_task_status(str(state["first_task_id"]))
        state["first_status"] = first_status
        state["last_checked_at"] = utc_now()
        save_state(state_path, state)
        log(f"first_status={first_status}")
        if first_status in FAILURE_STATUSES:
            state["terminal_status"] = f"First{first_status}"
            state["stopped_at"] = utc_now()
            save_state(state_path, state)
            log("first task failed; not submitting second config")
            return 3
        if first_status != SUCCESS_STATUS:
            time.sleep(args.poll_seconds)
            continue

        if not state.get("second_task_id"):
            second_task_id = submit(args.second_conf)
            state["second_task_id"] = second_task_id
            state["second_submitted_at"] = utc_now()
            save_state(state_path, state)
            log(f"second_task_id={second_task_id}")

        second_status = get_task_status(str(state["second_task_id"]))
        state["second_status"] = second_status
        state["last_checked_at"] = utc_now()
        save_state(state_path, state)
        log(f"second_status={second_status}")
        if second_status in FAILURE_STATUSES:
            state["terminal_status"] = f"Second{second_status}"
            state["stopped_at"] = utc_now()
            save_state(state_path, state)
            return 4
        if second_status == SUCCESS_STATUS:
            state["terminal_status"] = SUCCESS_STATUS
            state["completed_at"] = utc_now()
            save_state(state_path, state)
            log("chain completed successfully")
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())

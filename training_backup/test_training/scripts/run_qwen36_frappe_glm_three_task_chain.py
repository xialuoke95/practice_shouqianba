#!/usr/bin/env python3
"""Submit and verify the three Qwen3.6 BytePlus tasks strictly in sequence."""

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
FAILURE_STATUSES = {"Failed", "Killed", "Killing", "Stopped", "Error"}
ROOT = Path(__file__).resolve().parents[1]

ARMS = [
    {
        "name": "a_frappe_glm_99",
        "config": "docs/generated/byteplus_qwen36_frappe_glm_a99_16gpu_task.yaml",
        "output_dir": "/pfs-verdent/guoyinuo/coding_agent_models/output/train_qwen36_frappe_glm_a99_16gpu_gbs2_epoch2_20260806_v3",
        "expected_steps": 99,
    },
    {
        "name": "b_saas47_plus_frappe99",
        "config": "docs/generated/byteplus_qwen36_saas47_frappe99_b146_16gpu_task.yaml",
        "output_dir": "/pfs-verdent/guoyinuo/coding_agent_models/output/train_qwen36_saas47_frappe99_b146_16gpu_gbs2_epoch2_20260806_v3",
        "expected_steps": 146,
    },
    {
        "name": "c_saas47_plus_frappe99_x2_plus_general200",
        "config": "docs/generated/byteplus_qwen36_saas47_frappe99_x2_general200_c492_16gpu_task.yaml",
        "output_dir": "/pfs-verdent/guoyinuo/coding_agent_models/output/train_qwen36_saas47_frappe99_x2_general200_c492_16gpu_gbs2_epoch2_20260806_v4",
        "expected_steps": 492,
    },
]

REMOTE_VERIFY_SCRIPT = r'''
import json
import re
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
runs = sorted(
    (path for path in output.iterdir() if path.is_dir() and path.name.startswith("v")),
    key=lambda path: path.stat().st_mtime,
)
if not runs:
    raise SystemExit(f"no versioned run directory under {output}")
run = runs[-1]
logging_path = run / "logging.jsonl"
records = [json.loads(line) for line in logging_path.read_text().splitlines()]
loss_records = [record for record in records if "loss" in record]
if len(loss_records) != expected_steps:
    raise SystemExit(f"expected {expected_steps} loss steps, got {len(loss_records)}")
if loss_records[-1].get("iteration") != f"{expected_steps}/{expected_steps}":
    raise SystemExit(f"bad final iteration: {loss_records[-1].get('iteration')}")

checkpoint = run / f"checkpoint-{expected_steps}"
latest = checkpoint / "latest_checkpointed_iteration.txt"
if latest.read_text().strip() != str(expected_steps):
    raise SystemExit(f"bad latest marker: {latest}")
index_path = checkpoint / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
missing_shards = [name for name in shards if not (checkpoint / name).is_file()]
empty_shards = [name for name in shards if (checkpoint / name).stat().st_size == 0]
if missing_shards or empty_shards:
    raise SystemExit(
        f"checkpoint shard failure missing={missing_shards} empty={empty_shards}"
    )

logs = sorted(output.glob("train_*_node*.log"))
if len(logs) < 2:
    raise SystemExit(f"expected at least two node logs, got {len(logs)}")
fatal_pattern = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|RuntimeError|ERROR:"
)
fatal_matches = []
postflight_logs = 0
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), 1):
        if fatal_pattern.search(line):
            fatal_matches.append({"log": str(path), "line": line_number, "text": line[:500]})
    if "TRAINING_POSTFLIGHT_OK" in text:
        postflight_logs += 1
if fatal_matches:
    raise SystemExit(f"fatal_matches={json.dumps(fatal_matches[:20])}")
if postflight_logs < 2:
    raise SystemExit(f"only {postflight_logs} node logs contain TRAINING_POSTFLIGHT_OK")

losses = [record["loss"] for record in loss_records]
result = {
    "run_dir": str(run),
    "logging_path": str(logging_path),
    "checkpoint": str(checkpoint),
    "steps": len(loss_records),
    "first_loss": losses[0],
    "final_loss": losses[-1],
    "mean_loss": sum(losses) / len(losses),
    "min_loss": min(losses),
    "max_loss": max(losses),
    "peak_memory_gib": max(record.get("memory(GiB)", 0) for record in loss_records),
    "final_train_speed_seconds_per_iteration": loss_records[-1].get("train_speed(s/it)"),
    "final_learning_rate": loss_records[-1].get("learning_rate"),
    "safetensors_shards": len(shards),
    "indexed_tensors": len(index["weight_map"]),
    "indexed_weight_bytes": index.get("metadata", {}).get("total_size"),
    "checkpoint_directory_bytes": sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    ),
    "fatal_matches": 0,
    "missing_shards": 0,
    "postflight_logs": postflight_logs,
}
print(json.dumps(result, sort_keys=True))
'''


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
    raise ValueError(f"no JSON payload found: {text[-1500:]}")


def run_volc(args: list[str]) -> Any:
    volc_args = " ".join(shlex.quote(part) for part in args)
    command = [
        "/bin/bash",
        "-lc",
        (
            "eval \"$(grep '^export HuoShan' ~/.bashrc)\" "
            "&& export VOLC_ACCESSKEY=\"$HuoShanAccessKeyID\" "
            "VOLC_SECRETKEY=\"$HuoShanSecretAccessKey\" "
            f"&& volc {volc_args}"
        ),
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        raise RuntimeError(f"volc exited {proc.returncode}: {output[-2000:]}")
    return parse_last_json(output)


def submit(config: Path) -> str:
    payload = run_volc(
        ["ml_task", "submit", "--conf", str(config), "--output", "json"]
    )
    if not isinstance(payload, dict) or not payload.get("Id"):
        raise RuntimeError(f"unexpected submit payload: {payload!r}")
    return str(payload["Id"])


def get_task(task_id: str) -> dict[str, Any]:
    payload = run_volc(["ml_task", "get", "--id", task_id, "--output", "json"])
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError(f"unexpected task payload for {task_id}: {payload!r}")
    return payload[0]


def verify_remote_checkpoint(arm: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ssh",
            "gpu_dev_huoshan",
            "python3",
            "-",
            arm["output_dir"],
            str(arm["expected_steps"]),
        ],
        input=REMOTE_VERIFY_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        raise RuntimeError(
            f"remote checkpoint verification failed for {arm['name']}: {output[-3000:]}"
        )
    result = parse_last_json(output)
    if result.get("fatal_matches") != 0 or result.get("missing_shards") != 0:
        raise RuntimeError(f"invalid verification result: {result!r}")
    return result


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "created_at": utc_now(), "arms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "logs" / "qwen36_frappe_glm_three_task_chain_v3.state.json",
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    state = load_state(args.state)
    save_state(args.state, state)
    log(f"starting/resuming chain state={args.state}")

    for arm in ARMS:
        arm_state = state["arms"].setdefault(
            arm["name"],
            {
                "config": arm["config"],
                "output_dir": arm["output_dir"],
                "expected_steps": arm["expected_steps"],
            },
        )
        save_state(args.state, state)

        if arm_state.get("verified"):
            log(f"arm={arm['name']} already verified; continuing")
            continue

        config = ROOT / arm["config"]
        if not config.is_file():
            raise FileNotFoundError(config)

        if not arm_state.get("task_id"):
            task_id = submit(config)
            arm_state["task_id"] = task_id
            arm_state["submitted_at"] = utc_now()
            arm_state["status"] = "Submitted"
            save_state(args.state, state)
            log(f"arm={arm['name']} submitted task_id={task_id}")

        task_id = str(arm_state["task_id"])
        while True:
            task = get_task(task_id)
            status = str(task.get("Status", "Unknown"))
            arm_state["status"] = status
            arm_state["task"] = task
            arm_state["last_checked_at"] = utc_now()
            save_state(args.state, state)
            log(f"arm={arm['name']} task_id={task_id} status={status}")
            if status in FAILURE_STATUSES:
                state["terminal_status"] = f"{arm['name']}:{status}"
                state["stopped_at"] = utc_now()
                save_state(args.state, state)
                return 2
            if status == SUCCESS_STATUS:
                break
            time.sleep(args.poll_seconds)

        try:
            verification = verify_remote_checkpoint(arm)
        except Exception as exc:
            arm_state["verification_error"] = str(exc)
            state["terminal_status"] = f"{arm['name']}:VerificationFailed"
            state["stopped_at"] = utc_now()
            save_state(args.state, state)
            raise

        arm_state["verification"] = verification
        arm_state["verified"] = True
        arm_state["verified_at"] = utc_now()
        save_state(args.state, state)
        log(
            f"arm={arm['name']} verified checkpoint={verification['checkpoint']} "
            f"steps={verification['steps']}"
        )

    state["terminal_status"] = SUCCESS_STATUS
    state["completed_at"] = utc_now()
    save_state(args.state, state)
    log("all three arms completed and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

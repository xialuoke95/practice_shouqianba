#!/usr/bin/env bash
# Keep the three-task sequential controller alive across transient CLI/session
# failures. The controller itself never retries a failed training or checkpoint.

set -euo pipefail

WORKDIR=/home/ubuntu/workspace/test_training
STATE=$WORKDIR/logs/qwen36_frappe_glm_three_task_chain_v3.state.json
CHAIN_LOG=$WORKDIR/logs/qwen36_frappe_glm_three_task_chain_v3.log
SUPERVISOR_LOG=$WORKDIR/logs/qwen36_frappe_glm_three_task_chain_v3.supervisor.log
NOTIFIED_MARKER=$WORKDIR/logs/qwen36_frappe_glm_three_task_chain_v4.terminal_notified
PATTERN="python3 scripts/run_qwen36_frappe_glm_three_task_chain.py --state logs/qwen36_frappe_glm_three_task_chain_v3.state.json"

cd "$WORKDIR"
mkdir -p "$WORKDIR/logs"

terminal_status() {
    python3 - "$STATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit
try:
    print(json.loads(path.read_text()).get("terminal_status", ""))
except Exception:
    print("")
PY
}

while true; do
    status=$(terminal_status)
    if [ -n "$status" ]; then
        if [ ! -e "$NOTIFIED_MARKER" ]; then
            if remotelab trigger create --in 1s \
                --text "火山三连训 v3 控制器已到达终态 ${status}。立即检查 logs/qwen36_frappe_glm_three_task_chain_v3.state.json 与远端日志；成功则完成三组 checkpoint 验收和最终汇报，失败则诊断并在既有授权范围内修复后继续。" \
                --json >> "$SUPERVISOR_LOG" 2>&1; then
                : > "$NOTIFIED_MARKER"
            else
                printf '%s failed to schedule terminal wake-up; will retry\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
                sleep 60
                continue
            fi
        fi
        printf '%s terminal_status=%s; supervisor exiting\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$SUPERVISOR_LOG"
        exit 0
    fi

    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
        printf '%s chain controller not found; starting/resuming\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
        setsid -f /bin/bash -lc \
            "cd '$WORKDIR' && exec python3 scripts/run_qwen36_frappe_glm_three_task_chain.py --state logs/qwen36_frappe_glm_three_task_chain_v3.state.json --poll-seconds 120 >> '$CHAIN_LOG' 2>&1"
    fi
    sleep 300
done

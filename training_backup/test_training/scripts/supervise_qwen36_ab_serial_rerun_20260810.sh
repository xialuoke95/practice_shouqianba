#!/usr/bin/env bash
# Keep the A -> B serial reproducibility controller alive through both verifications.

set -euo pipefail

WORKDIR=/home/ubuntu/workspace/test_training
STATE=$WORKDIR/logs/qwen36_ab_serial_rerun_20260810.state.json
CHAIN_LOG=$WORKDIR/logs/qwen36_ab_serial_rerun_20260810.log
SUPERVISOR_LOG=$WORKDIR/logs/qwen36_ab_serial_rerun_20260810.supervisor.log
NOTIFIED_MARKER=$WORKDIR/logs/qwen36_ab_serial_rerun_20260810.terminal_notified
PATTERN="python3 scripts/run_qwen36_ab_serial_rerun_20260810.py --state logs/qwen36_ab_serial_rerun_20260810.state.json"

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
                --text "火山 A→B 串行复现控制器已到达终态 ${status}。检查 logs/qwen36_ab_serial_rerun_20260810.state.json、两个远端任务日志及 checkpoint；成功则汇报 A/B 指标与历次复现误差，失败则诊断修复后继续。" \
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
        printf '%s controller not found; starting/resuming\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
        setsid -f /bin/bash -lc \
            "cd '$WORKDIR' && exec python3 scripts/run_qwen36_ab_serial_rerun_20260810.py --state logs/qwen36_ab_serial_rerun_20260810.state.json --poll-seconds 120 >> '$CHAIN_LOG' 2>&1"
    fi
    sleep 300
done

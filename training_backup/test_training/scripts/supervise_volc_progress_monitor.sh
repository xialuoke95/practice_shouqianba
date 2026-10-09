#!/bin/bash
# Keep the read-only Volc progress monitor alive while the chain is active.

set -euo pipefail

WORKDIR=/home/ubuntu/workspace/test_training
STATE=$WORKDIR/logs/watch_chain_user_worker_32gpu.state.json
SUPERVISOR_LOG=$WORKDIR/logs/supervise_volc_progress_monitor.log
MONITOR_LOG=$WORKDIR/logs/monitor_volc_chain_progress.log
PATTERN="python3 scripts/monitor_volc_chain_progress.py --state logs/watch_chain_user_worker_32gpu.state.json"

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

start_monitor() {
    setsid -f /bin/bash -lc "cd '$WORKDIR' && exec python3 scripts/monitor_volc_chain_progress.py --state logs/watch_chain_user_worker_32gpu.state.json --jsonl logs/watch_chain_user_worker_32gpu.progress.jsonl --latest logs/watch_chain_user_worker_32gpu.progress.latest.json --loop --interval-seconds 600 >> '$MONITOR_LOG' 2>&1"
}

while true; do
    status="$(terminal_status)"
    if [ -n "$status" ]; then
        printf '%s terminal_status=%s; monitor supervisor exiting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$SUPERVISOR_LOG"
        exit 0
    fi

    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
        printf '%s progress monitor not found; restarting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
        start_monitor
    fi

    sleep 300
done

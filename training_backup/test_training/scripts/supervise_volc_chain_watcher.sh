#!/bin/bash
# Keep the Volc chain watcher alive until it reaches a terminal state.

set -euo pipefail

WORKDIR=/home/ubuntu/workspace/test_training
STATE=$WORKDIR/logs/watch_chain_user_worker_32gpu.state.json
SUPERVISOR_LOG=$WORKDIR/logs/supervise_watch_chain_user_worker_32gpu.log
WATCHER_LOG=$WORKDIR/logs/watch_chain_user_worker_32gpu.log
PATTERN="python3 scripts/watch_volc_chain_submit.py --wait-task t-20260620040947-kxlcd --wait-task t-20260620184158-j2k6t"

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
    status="$(terminal_status)"
    if [ -n "$status" ]; then
        printf '%s terminal_status=%s; supervisor exiting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$SUPERVISOR_LOG"
        exit 0
    fi

    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
        printf '%s watcher not found; restarting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
        setsid -f /bin/bash -lc "cd '$WORKDIR' && exec python3 scripts/watch_volc_chain_submit.py --wait-task t-20260620040947-kxlcd --wait-task t-20260620184158-j2k6t --first-conf docs/generated/byteplus_no_think_user_worker_32gpu_te_fused_ce_gbs72_max180k_task.yaml --second-conf docs/generated/byteplus_no_think_user_worker_no_persist_32gpu_te_fused_ce_gbs72_max180k_task.yaml --state logs/watch_chain_user_worker_32gpu.state.json --poll-seconds 180 >> '$WATCHER_LOG' 2>&1"
    fi

    sleep 300
done

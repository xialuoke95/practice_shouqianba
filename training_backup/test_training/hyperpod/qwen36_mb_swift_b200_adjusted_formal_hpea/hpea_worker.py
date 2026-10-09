#!/opt/venv/bin/python3
"""Small HPEA-compatible lifecycle wrapper for the Qwen3.6 training script."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


STATE_READY = "ready"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAULTED = "faulted"
STATE_STOPPED = "stopped"
STATE_STOPPING = "stopping"


class WorkerState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.status = STATE_READY
        self.message = "ready for podmanager start"
        self.process: subprocess.Popen[bytes] | None = None
        self.exit_code: int | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.stop_requested = False

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "ready": "True" if self.status in {STATE_READY, STATE_RUNNING} else "False",
                "message": self.message,
                "pid": self.process.pid if self.process and self.process.poll() is None else None,
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "timestamp": time.time(),
            }

    def start_training(self) -> dict[str, Any]:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.status = STATE_RUNNING
                self.message = "training process already running"
                return self.payload()
            if self.status == STATE_COMPLETED:
                self.message = "training process already completed"
                return self.payload()

            env = os.environ.copy()
            env["JOB_NAME"] = env.get("QWEN36_JOB_NAME", env.get("JOB_NAME", "qwen36_hpea"))
            env["NNODES"] = env.get("QWEN36_NNODES", env.get("NNODES", "1"))
            env["NPROC_PER_NODE"] = env.get("QWEN36_NPROC_PER_NODE", env.get("NPROC_PER_NODE", "8"))

            command = env.get("HPEA_TRAINING_COMMAND", "/scripts/run_qwen36_overlay_sft_job.sh")
            argv = ["/bin/bash", command]
            print(f"[hpea-wrapper] starting training: {' '.join(argv)}", flush=True)
            self.process = subprocess.Popen(argv, env=env, start_new_session=True)
            self.status = STATE_RUNNING
            self.message = "training process running"
            self.exit_code = None
            self.started_at = time.time()
            self.finished_at = None
            self.stop_requested = False
            threading.Thread(target=self._watch_child, daemon=True).start()
            return self.payload()

    def stop_training(self) -> dict[str, Any]:
        with self.lock:
            self.stop_requested = True
            proc = self.process
            if not proc or proc.poll() is not None:
                self.status = STATE_STOPPED
                self.message = "no active training process"
                self.finished_at = self.finished_at or time.time()
                return self.payload()

            self.status = STATE_STOPPING
            self.message = "stop requested"
            print(f"[hpea-wrapper] stopping training process group pid={proc.pid}", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            threading.Thread(target=self._force_kill_later, args=(proc,), daemon=True).start()
            return self.payload()

    def _watch_child(self) -> None:
        proc = self.process
        if proc is None:
            return
        rc = proc.wait()
        with self.lock:
            self.exit_code = rc
            self.finished_at = time.time()
            if self.stop_requested:
                self.status = STATE_STOPPED
                self.message = f"training stopped with exit code {rc}"
            elif rc == 0:
                self.status = STATE_COMPLETED
                self.message = "training completed successfully"
            else:
                self.status = STATE_FAULTED
                self.message = f"training failed with exit code {rc}"
        print(f"[hpea-wrapper] training exited rc={rc}", flush=True)

    def _force_kill_later(self, proc: subprocess.Popen[bytes]) -> None:
        timeout = int(os.environ.get("HPEA_STOP_GRACE_SECONDS", "120"))
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        print(f"[hpea-wrapper] SIGKILL training process group pid={proc.pid}", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


STATE = WorkerState()


class Handler(BaseHTTPRequestHandler):
    server_version = "qwen36-hpea-wrapper/1.0"

    def do_GET(self) -> None:  # noqa: N802
        self._route()

    def do_POST(self) -> None:  # noqa: N802
        self._route()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[hpea-wrapper] {self.address_string()} {fmt % args}", flush=True)

    def _route(self) -> None:
        if self.path == "/status":
            self._send_json(STATE.payload())
        elif self.path == "/start":
            self._discard_body()
            self._send_json(STATE.start_training())
        elif self.path == "/stop":
            self._discard_body()
            self._send_json(STATE.stop_training())
        elif self.path in {"/health", "/readyz"}:
            self._send_json({"ok": True, **STATE.payload()})
        elif self.path == "/update":
            self._discard_body()
            self._send_json({"ok": True, "message": "elastic update not implemented", **STATE.payload()})
        else:
            self._send_json({"error": f"unknown path {self.path}", **STATE.payload()}, status=404)

    def _discard_body(self) -> None:
        length = int(self.headers.get("content-length") or "0")
        if length:
            self.rfile.read(length)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _shutdown(signum: int, _frame: Any) -> None:
    print(f"[hpea-wrapper] received signal {signum}", flush=True)
    STATE.stop_training()
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    port = int(os.environ.get("HPEA_PORT", "8080"))
    host = os.environ.get("HPEA_HOST", "0.0.0.0")
    print(f"[hpea-wrapper] serving lifecycle API on {host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()

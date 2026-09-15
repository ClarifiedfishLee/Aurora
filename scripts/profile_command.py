#!/usr/bin/env python3
"""Run a command and append timing, status, and peak GPU memory to JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


def gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        return sum(values) if values else 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--metrics", type=Path, default=Path("runs/smoke/metrics.jsonl"))
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log or args.metrics.parent / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    peak_memory: int | None = gpu_memory_mib()
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak_memory
        while not stop.wait(1):
            current = gpu_memory_mib()
            if current is not None:
                peak_memory = current if peak_memory is None else max(peak_memory, current)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())
    stop.set()
    thread.join(timeout=2)
    duration = time.monotonic() - started
    record = {
        "name": args.name,
        "started_at_utc": started_at.isoformat(),
        "duration_seconds": round(duration, 3),
        "exit_code": process.returncode,
        "success": process.returncode == 0,
        "peak_gpu_memory_mib": peak_memory,
        "command": shlex.join(command),
        "log_path": str(log_path),
    }
    with args.metrics.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())

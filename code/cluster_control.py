#!/usr/bin/env python3
"""PID-identity-verified GPU keepalive quiesce & restore helper (zero secrets/AML metadata)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

KEEPALIVE = Path("/scratch/copilot-mi300x-keepalive")
TMUX_SESSION = "copilot-mi300x-keepalive"


def _proc_identity(pid: int) -> dict | None:
    proc = Path("/proc") / str(pid)
    if not (proc / "stat").is_file():
        return None
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        start_ticks = stat_text.rsplit(") ", 1)[1].split()[19]
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        return {"pid": pid, "start_ticks": start_ticks, "cmdline": cmdline}
    except Exception:
        return None


def quiesce() -> dict:
    stopped = []
    sup_pid_file = KEEPALIVE / "supervisor.pid"
    sup_start_file = KEEPALIVE / "supervisor.starttime"
    if sup_pid_file.is_file() and sup_start_file.is_file():
        pid = int(sup_pid_file.read_text().strip())
        expected_ticks = sup_start_file.read_text().strip()
        ident = _proc_identity(pid)
        if ident and ident["start_ticks"] == expected_ticks and str(KEEPALIVE / "remote_supervisor.sh") in ident["cmdline"]:
            os.kill(pid, signal.SIGTERM)
            for _ in range(40):
                if _proc_identity(pid) is None:
                    break
                time.sleep(0.25)
            stopped.append(ident)
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
    for _ in range(25):
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
        if not apps:
            break
        time.sleep(1.0)
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "node": os.uname().nodename,
        "event": "QUIESCED",
        "stopped_keepalive": stopped,
        "remaining_compute_apps": apps,
    }


def restore(profile: str = "medium") -> dict:
    if not (KEEPALIVE / "remote_supervisor.sh").is_file():
        raise RuntimeError(f"Missing keepalive supervisor at {KEEPALIVE / 'remote_supervisor.sh'}")
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
    cmd = [
        "tmux", "new-session", "-d", "-s", TMUX_SESSION,
        f"exec env KEEPALIVE_IDLE_PROFILE={profile} KEEPALIVE_INITIAL_IDLE_GRACE_SECONDS=5 "
        f"KEEPALIVE_IDLE_GRACE_SECONDS=5 bash '{KEEPALIVE}/remote_supervisor.sh'",
    ]
    subprocess.run(cmd, check=True, timeout=15)
    gpu_rows = []
    for _ in range(30):
        time.sleep(2.0)
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip().splitlines()
        gpu_rows = [
            {"gpu": int(p[0].strip()), "memory_mib": int(p[1].strip()), "util_pct": int(p[2].strip())}
            for line in smi for p in [line.split(",")]
        ]
        if len(gpu_rows) == 8 and all(g["memory_mib"] >= 1000 for g in gpu_rows):
            break
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "node": os.uname().nodename,
        "event": "RESTORED",
        "gpus": gpu_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["quiesce", "restore"])
    ap.add_argument("--profile", default="medium")
    args = ap.parse_args()
    out = quiesce() if args.action == "quiesce" else restore(args.profile)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import signal
from pathlib import Path


def boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def process_start_time(pid: int) -> int | None:
    """Return Linux /proc starttime ticks, robust to spaces in process names."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        after_name = stat[stat.rfind(")") + 2 :].split()
        return int(after_name[19])  # field 22; field 3 is index 0 here
    except (OSError, ValueError, IndexError):
        return None


def process_identity_alive(pid: int | None, expected_start_time: int | None) -> bool:
    if pid is None:
        return False
    actual = process_start_time(pid)
    if actual is None:
        return False
    return expected_start_time is None or actual == expected_start_time


def process_group_members(pgid: int) -> set[int]:
    members: set[int] = set()
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) == pgid:
                members.add(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return members


def managed_process_group_alive(
    pid: int | None, pgid: int | None, expected_start_time: int | None
) -> bool:
    """Recognize the saved group while preventing a recycled group-leader PID.

    If the original leader exited but workers remain, Linux keeps the original
    process group alive without a member whose PID equals PGID. A newly recycled
    group necessarily has a new leader at that PID, whose start time will not
    match the persisted leader identity.
    """
    if pid is None or pgid is None:
        return False
    if process_identity_alive(pid, expected_start_time):
        return True
    members = process_group_members(pgid)
    if not members:
        return False
    if pgid in members:
        # A process now leads this numeric group. Only accept the persisted leader.
        return pid == pgid and process_identity_alive(pid, expected_start_time)
    return True


def pid_in_process_group(pid: int, pgid: int | None) -> bool:
    if pgid is None:
        return False
    try:
        return os.getpgid(pid) == pgid
    except (ProcessLookupError, PermissionError):
        return False


def signal_managed_group(
    pid: int | None,
    pgid: int | None,
    expected_start_time: int | None,
    sig: signal.Signals,
) -> bool:
    """Signal only after checking the saved leader identity to avoid PID reuse."""
    if pgid is None or not managed_process_group_alive(pid, pgid, expected_start_time):
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False

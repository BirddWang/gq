# Troubleshooting

Paths below assume default XDG locations. If configured, inspect
`$GQ_DATA_DIR`, `$GQ_RUNTIME_DIR`, and `$GQ_DATABASE` instead.

## The daemon will not start

```bash
gq daemon status
tail -n 100 ~/.local/share/gq/gq-daemon.log
ls -la ~/.local/run/gq/
```

Common causes are an unavailable NVIDIA driver/NVML library, an unwritable XDG
directory, or another installed `gq` daemon. Do not delete the lock while its PID is
live. If no daemon holds the lock, the next start safely removes a stale socket.

## Refusing to run as root

`gq` is a single-user scheduler, and jobs inherit the daemon's privileges, so running
it as root on a shared host gives every queued command more power than it needs.

Inside a container this reasoning does not apply — Docker and Jupyter images commonly
have uid 0 as the only account — so `gq` detects containers (`/.dockerenv`,
`/run/.containerenv`, the `container` variable, cgroup markers) and starts normally,
noting it in the daemon log.

If detection fails on a setup that is genuinely isolated, override it:

```bash
GQ_ALLOW_ROOT=1 gq daemon start
```

On a real multi-user host, prefer running `gq` as your own user instead.

## NVML is unavailable / GPUs are UNKNOWN

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
python -c 'import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())'
tail -n 100 ~/.local/share/gq/gq-daemon.log
```

Check the NVIDIA kernel driver, container device passthrough, and installed
`nvidia-ml-py` package. The daemon will not allocate while observation is unknown.
Transient per-device errors are retried at the next reconciliation.

## A GPU is stuck as EXTERNAL

Use `gq gpu` to see the PID, then inspect only—`gq` never kills it:

```bash
ps -fp PID
cat /proc/PID/cgroup
nvidia-smi
```

A Jupyter kernel commonly retains a CUDA context at 0% utilization. Shut down the
kernel (not just the browser tab) or make the application release CUDA. The state
changes after NVML stops reporting the compute PID.

## A GPU is stuck as UNKNOWN

`UNKNOWN` indicates an NVML error, a disappeared device UUID, a stale/orphaned
allocation, or a foreign PID conflicting with a managed reservation. Inspect:

```bash
gq gpu
gq ps
gq show JOB
nvidia-smi
tail -n 100 ~/.local/share/gq/gq-daemon.log
```

Restarting the daemon triggers full identity reconciliation and does not kill the
job. Do not edit SQLite while the daemon is running.

## A job remains WAITING

Compare its request with allocatable devices:

```bash
gq show JOB
gq gpu
gq ps
```

Only `FREE` GPUs count. A request may wait behind running work, an external CUDA
context, or an NVML failure. Smaller younger jobs may backfill; the MVP has no
starvation reservation for large jobs.

## A job failed to launch

`gq show JOB` reports failures such as a missing executable, permission error, or
deleted working directory. Remember that executable lookup uses the captured
submission `PATH`, while the command runs later from the captured cwd. Review
`gq logs JOB` and the daemon log.

## Cancellation does not finish

The daemon sends `SIGTERM`, waits 10 seconds by default, then sends `SIGKILL` to the
saved process group. Check `gq show JOB` for PID/PGID and the daemon log. Processes
stuck in uninterruptible kernel sleep (`D` state) cannot be killed until the kernel
operation returns. For testing, shorten the grace period with
`GQ_CANCEL_GRACE_SECONDS`, set in the daemon environment before startup.

## Stale socket or PID file

First run `gq daemon status`. A new daemon holds an exclusive file lock and, once
that succeeds, removes stale socket/PID state itself. Manual removal is rarely
needed. If startup still fails and no listed PID is alive, preserve the daemon log
and remove only the specific files under `~/.local/run/gq/`; never remove the data
directory or database as a stale-socket fix.

## Database recovery

Stop the daemon first, then back up the database and its WAL files:

```bash
gq daemon stop
cp -a ~/.local/share/gq/gq.sqlite3* /path/to/backup/
sqlite3 ~/.local/share/gq/gq.sqlite3 'PRAGMA integrity_check;'
```

Do not discard the database casually: it is the durable ownership record used to
avoid double allocation after a crash. Restore a known-good backup or repair with
SQLite tooling only after preserving all files. Starting with a new database while
old managed jobs are alive makes their CUDA PIDs appear `EXTERNAL`, which is safe
but loses history.

## Jupyter still owns GPU memory

Restart or shut down the specific kernel. Closing a notebook tab or finishing a
cell does not necessarily destroy its CUDA context. Memory quantity and 0%
utilization do not change `gq` ownership decisions.

## CUDA_VISIBLE_DEVICES looks wrong

`gq show JOB` displays physical indices assigned at launch. Inside the job, CUDA
renumbers those visible devices from zero. Physical `2,3` therefore appear as
logical `cuda:0,cuda:1`. Use `CUDA_VISIBLE_DEVICES` inside the log/job and UUIDs in
`gq show` when correlating with `nvidia-smi`.

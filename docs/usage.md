# Using gq

## Starting and inspecting the daemon

```bash
gq daemon start
gq daemon status
gq daemon stop
```

Normal commands auto-start the daemon if its Unix socket is unavailable. Startup
waits for a successful handshake and reports the daemon log path on failure. A
file lock prevents duplicate daemons. `daemon stop` stops scheduling and leaves
managed process groups alive; restart the daemon to reconcile and resume queue
management.

Runtime paths follow XDG conventions:

- database: `$XDG_DATA_HOME/gq/gq.sqlite3`, normally
  `~/.local/share/gq/gq.sqlite3`;
- job logs: `~/.local/share/gq/logs/JOB.log`;
- daemon log: `~/.local/share/gq/gq-daemon.log`;
- socket/PID/lock: `$XDG_RUNTIME_DIR/gq/`, or `~/.local/run/gq/` when
  `XDG_RUNTIME_DIR` is unset.

For isolated tests, `GQ_DATA_DIR`, `GQ_RUNTIME_DIR`, and `GQ_DATABASE` override
those paths. `GQ_RECONCILE_INTERVAL` and `GQ_CANCEL_GRACE_SECONDS` override the
2-second scan interval and 10-second cancellation grace period.

## Submitting commands

The short and explicit forms are equivalent:

```bash
gq -g 1 python train.py --epochs 10
gq run -g 1 python train.py --epochs 10
gq -g 2 --name evaluation torchrun --nproc-per-node=2 eval.py
```

`-g/--gpus` must be at least one and cannot exceed the NVML-visible device count.
The command is saved as an argument vector, not interpolated through a shell. The
working directory and submission environment are captured so a queued job runs
with the same context after a daemon restart.

## What happens to your environment

Because a queued job may not start until long after submission, the environment is
persisted in SQLite. That makes any captured credential a durable one, so variables
whose names look like secrets are dropped rather than stored:

- anything containing `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `API_KEY`,
  `APIKEY`, or ending in `_KEY`;
- anything beginning with `AWS_`, `AZURE_`, `GCP_`, or `GOOGLE_APPLICATION_`.

The dropped names are printed at submission, never removed silently, because a job
that needs a credential at runtime would otherwise fail much later in a way that
looks like an ordinary bad run:

```text
gq: not persisting 1 secret-looking variable(s): HF_TOKEN
    use --env-all (or GQ_ENV_ALL=1) to keep them, or --env-keep NAME
```

If the job genuinely needs one — `HF_TOKEN` for a private Hugging Face dataset, say —
keep it explicitly:

```bash
gq -g 1 --env-keep HF_TOKEN uv run train.py   # this variable only
gq -g 1 --env-all uv run train.py             # everything, as before
export GQ_ENV_ALL=1                           # everything, for this shell
```

Whatever is kept is stored in the user-only SQLite database: protect the data
directory and do not share the database. Normal CLI responses never display it.

When physical indices 2 and 3 are allocated, the process receives:

```text
CUDA_VISIBLE_DEVICES=2,3
```

CUDA software inside the job sees those devices as logical `cuda:0` and `cuda:1`.
This remapping is intentional. All other submitted environment values are
preserved.

Jobs are considered in submission order, but a job that does not currently fit is
skipped so smaller jobs can backfill otherwise idle GPUs. The MVP does not yet
prevent starvation of a large job.

## Submitting scripts

```bash
gq submit train.sh
gq submit -g 4 --name override train.sh
```

Scripts run as `/bin/bash /absolute/path/to/train.sh` from the directory in which
they were submitted. These directives are supported:

```bash
#gq --gpus=2
#gq --name=pythia-410m
```

CLI options override directives. With no GPU directive or CLI option, a script
requests one GPU. Unknown or malformed `#gq` directives are rejected rather than
silently ignored.

## Jobs and states

```bash
gq ps
gq ps --limit 20
gq show 42
```

`gq ps`, `gq gpu`, and `gq show` accept `--json`, which emits the daemon's own
records unchanged and is the supported way to script against `gq`. The table layout
is for humans and may change between releases; the JSON field names will not, within
a protocol version.

The normal lifecycle is `WAITING → STARTING → RUNNING → DONE/FAILED`. `STARTING`
means GPUs have already been durably reserved but the process has not been fully
recorded yet. Cancellation temporarily uses `CANCELLING`, then `CANCELLED`.
`ORPHANED` represents a live recovered process whose saved GPU identity disappeared;
its surviving saved reservations fail closed as `UNKNOWN`.

- `DONE`: command exited zero.
- `FAILED`: launch failed, command exited nonzero, or a recovered process vanished
  without an observable exit status.
- `CANCELLED`: cancelled before launch or its process group was terminated.
- `ORPHANED`: restart recovery could not safely reconstruct the allocation.

`gq show` includes timestamps, saved argv, directory, physical allocation, PID,
PGID, exit code, log path, and any failure reason.

## GPU status and external programs

```bash
gq gpu
```

GPU states are:

- `FREE`: no reservation and no NVML compute PID;
- `RESERVED`: atomically assigned while a job starts;
- `RUNNING`: assigned to a live managed process group;
- `EXTERNAL`: one or more NVML compute PIDs do not belong to `gq`;
- `UNKNOWN`: NVML failed or logical and physical ownership conflict.

`EXTERNAL` and `UNKNOWN` are never allocatable. A Jupyter/IPython kernel often
retains a CUDA context after a cell ends, even at 0% utilization. It remains
`EXTERNAL` until NVML no longer reports its PID. The next periodic scan then frees
the GPU and immediately attempts queued work. `gq` never kills external processes.

## Logs

```bash
gq logs 42
gq logs -f 42
```

Standard output and standard error share one chronological binary-safe file.
`-f` waits through `WAITING`, streams appended bytes, and exits after the job
reaches a terminal state. Press Ctrl-C to stop following without changing the job.

## Cancellation

```bash
gq cancel 42
```

A waiting job is marked `CANCELLED` immediately. A running job receives `SIGTERM`
on its entire saved process group. If the identity-checked leader remains alive
after the grace period, the daemon sends `SIGKILL` to the same group. PID start
time and boot ID checks reduce the risk of signaling a reused PID. Cancellation
never targets an `EXTERNAL` PID.

## Removing old jobs and logs

Job history and logs grow without bound: a week of active use can leave tens of
megabytes of logs behind. Two commands reclaim it.

```bash
gq rm 41 42 43                          # specific jobs, with their logs
gq clean --older-than 30d               # everything terminal older than 30 days
gq clean --older-than 7d --state FAILED # only failed runs from the last week
gq clean --older-than 30d -y            # skip the confirmation prompt
```

Both refuse to touch a job that is not in a terminal state; cancel it first. `gq rm`
rejects the entire batch if any id is unknown or still active, so a mistyped id
cannot partially destroy history. Deleting a job also deletes its GPU assignment
rows and its log file. Log removal is confined to the managed log directory, so a
hand-edited `log_path` cannot turn cleanup into arbitrary file deletion.

Neither command is reversible. Back up `gq.sqlite3` first if the history matters.

## Restart behavior

Queued jobs resume normally because argv, directory, and environment are in
SQLite. Live jobs retain their saved UUID reservations when PID, Linux process
start time, process group, boot ID, and current GPU inventory agree. Because a new
daemon cannot `wait(2)` a process created by the old daemon, a reattached command
whose process later exits is recorded as `FAILED` with “exit status unavailable”;
its log remains intact. A boot change marks prior active jobs failed/interrupted.


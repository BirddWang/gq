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
`GQ_FAIL_FAST_COUNT` (default 3, `0` disables) and `GQ_FAIL_FAST_SECONDS` (default
60) control when a failing group is held; see [Held groups](#held-groups).

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
#gq --group=layer-depth
#gq --key=layer-depth/bart/seed1
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

## Sweeps: groups, keys, and retries

A sweep is usually a script that submits dozens of near-identical jobs. Three things
make those manageable: a **group** names the batch, a **key** names each cell, and
**retry** reruns what failed.

```bash
#!/bin/bash
export GQ_GROUP=xsum-ablation          # every submission below joins this group
for SEED in 1 2 3; do
  for CELL in kl kl-enc_hddn kl-enc_self; do
    gq -g 1 --key "xsum/$CELL/seed$SEED" uv run trainer.py --losses ... --seed "$SEED"
  done
done
```

### Groups

`--group NAME` (or `GQ_GROUP`, or `#gq --group=NAME` in a script) tags a job. A group
has no setup and no cost; it exists as soon as a job uses it.

```bash
gq groups                        # every group, with its progress
gq ps --group xsum-ablation      # its jobs, plus a one-line summary
```

`gq groups` counts **each cell's latest attempt**: a job that failed and then
succeeded on retry counts once, as `DONE`. A `RETRIED` column shows how many earlier
attempts were replaced.

### Keys make a sweep script safe to re-run

`--key KEY` (or `#gq --key=KEY`) names a job's cell. A submission is **skipped** when
another job with the same key is `WAITING`, `STARTING`, `RUNNING`, `CANCELLING`,
`ORPHANED`, or `DONE`:

```text
Skipped: key 'xsum/kl/seed1' is already DONE as job 212
```

A `FAILED` or `CANCELLED` job frees its key. So re-running a sweep script after a
crash, a cancellation, or a partial run submits exactly the cells that are missing or
broken, and skipping exits successfully. The check and the insert happen in one
transaction, so two scripts racing on the same key cannot both submit it.

The daemon decides "done" from the recorded exit status, which catches a run that was
killed half-way. That is not the same as your program having written its output. If
a cell can exit 0 without producing a checkpoint, keep a file check in the script too.

Keys are free-form up to 200 characters; `dataset/cell/seed` reads well. Deleting a
job with `gq rm` frees its key.

### Selecting many jobs

`gq ps`, `gq cancel`, and `gq retry` accept job ids and inclusive ranges, combined
with `--group` and a state filter:

```bash
gq ps 300-440
gq ps --group xsum-ablation --state FAILED
gq cancel 300-440 --waiting
gq cancel --group xsum-ablation          # asks first if it would kill running jobs
gq cancel --group xsum-ablation -y       # ... or not
```

A bulk cancel lists what matches, asks for confirmation if any of those jobs are
running, then cancels exactly the jobs it listed, so a job that joins the group while
you read the prompt is not swept up. Without a terminal it requires `-y`. The whole
batch is cancelled in a single step inside the daemon, so a GPU freed by one of the
jobs cannot be handed to another job in the same batch in the meantime.

`gq cancel JOB` with a single id behaves exactly as before, including reporting an
error for a job that already finished.

### Retrying

```bash
gq retry 216-218
gq retry --group xsum-ablation --failed
```

Retry resubmits a `FAILED` or `CANCELLED` job as a new job with the same command,
working directory, stored environment, GPU count, name, group, and key. `--failed` or
`--cancelled` narrows the selection. Jobs named explicitly by id that cannot be
retried are listed with the reason, instead of silently skipped.

Attempts form a lineage: `gq show --json` reports `retry_of`, the first job of the
chain. Only the newest attempt of a lineage is ever retried, so running the same
`gq retry` twice queues nothing the second time:

```text
Skipped 3 jobs:
  216: was already retried as job 482
```

The stored environment is the one captured at the original submission, with the same
secret filtering. Retry is for rerunning a command after fixing the code or data it
uses. If the command itself was wrong, submit a new one.

### Held groups

When a batch has a bug, every job in it tends to fail the same way within seconds.
Rather than let the whole queue churn through that, gq **holds** a group once its
3 most recent finished jobs all failed within 60 seconds of starting. A held group's
queued jobs stay `WAITING`, shown as `HELD` in `gq ps`, and its running jobs are left
alone:

```text
gq: group xsum-ablation is held: 3 jobs in a row failed within 60s of starting (7, 8, 9)
    fix the cause, then 'gq retry --group xsum-ablation --failed', or release it as is
    with 'gq daemon resume --group xsum-ablation'
```

The rule is deliberately narrow:

- only grouped jobs are ever held;
- a `DONE` breaks the streak, and a cancellation neither counts nor breaks it;
- only a failure of a job gq launched and watched can trigger a hold; failures that
  restart recovery records never do;
- a mistyped executable fails inside the scheduling pass itself, so the hold is
  re-checked before every launch, and a broken group of 143 jobs stops at 3;
- after a group is released, only failures from then on count;
- holds are stored in the database and survive a daemon restart or `gq update`.

`gq retry` releases the hold on the group of anything it resubmits, since retrying says
the cause was dealt with. Tune the rule with `GQ_FAIL_FAST_COUNT` (`0` disables it) and
`GQ_FAIL_FAST_SECONDS`, set in the daemon's environment.

You can hold a group by hand too, for example while you look into something:

```bash
gq daemon pause --group xsum-ablation
gq daemon resume --group xsum-ablation
```

## Pausing the queue

```bash
gq daemon pause    # queued jobs stay WAITING; running jobs are untouched
gq daemon resume
```

Pausing is useful before maintenance: submissions are still accepted, but nothing
new starts, so the set of running jobs can only shrink. `gq ps` and `gq daemon
status` say when the queue is paused. The pause lives only in the daemon's memory,
so restarting the daemon always resumes the queue.

## Updating gq

```bash
gq update --check   # is there a newer release on PyPI?
gq update           # install it and restart the daemon
gq update --wait    # first let running jobs finish
```

`gq update` works out how it was installed and runs the matching upgrade:
`uv tool upgrade` for a uv tool, `pipx upgrade` for pipx, or the environment's own
`pip` (or `uv pip` in a virtualenv created without pip). If gq was installed from a
source checkout, a git URL, or in editable mode, it does not switch you to PyPI
behind your back; it prints the command to run instead. This is the only command in
`gq` that contacts the network, and only `pypi.org`.

Updating restarts the daemon, and **running jobs are the one thing to be careful
about**. A restarted daemon reattaches to running jobs, but Linux does not let it
collect the exit status of a process an earlier daemon started, so those jobs are
recorded as `FAILED` ("exit status unavailable") when they finish, even if they
succeeded. So:

- with nothing running, `gq update` pauses the queue, stops the daemon, upgrades, and
  starts the new daemon, which resumes the queue;
- with jobs running, it refuses and lists them;
- `--wait` pauses the queue and waits for running jobs to finish first, while queued
  jobs stay safely `WAITING`. Ctrl-C cancels the update and resumes the queue;
- `--force` updates immediately and accepts the mislabeled results.

Queued jobs are never at risk: they are stored in SQLite and start under the new
daemon. If the daemon was not running before the update, it is not started.

## Restart behavior

Queued jobs resume normally because argv, directory, and environment are in
SQLite. Live jobs retain their saved UUID reservations when PID, Linux process
start time, process group, boot ID, and current GPU inventory agree. Because a new
daemon cannot `wait(2)` a process created by the old daemon, a reattached command
whose process later exits is recorded as `FAILED` with “exit status unavailable”;
its log remains intact. A boot change marks prior active jobs failed/interrupted.


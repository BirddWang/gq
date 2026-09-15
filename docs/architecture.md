# Architecture

## Components

```text
 gq CLI ── JSON line / user-only Unix socket ──> daemon
                                                    │
                  ┌─────────────────────────────────┼──────────────┐
                  │                                 │              │
             SQLite jobs                     scheduler lock    NVML scan
             + UUID reservations             + backfilling     + /proc
                  │                                 │              │
                  └──────────────────── launch process group <────┘
                                           │
                                      combined log
```

The CLI only validates presentation-level input and sends structured JSON. It
never chooses a GPU. The daemon serializes submission, reservation, launch,
completion, cancellation, reconciliation, and scheduling through one `asyncio`
lock. SQLite is durable state, not a substitute for that serialization boundary.

## Persistent model and allocation invariant

Jobs store argv as JSON, cwd, environment, requested count, timestamps, process
identity, result, and log path. `job_gpus` stores GPU UUID plus its device index at
launch. UUID is the durable identity; current index is the value used for that
launch's `CUDA_VISIBLE_DEVICES`.

The invariant is:

> A GPU assigned to a managed job is never simultaneously allocatable.

For each scheduling pass, the daemon merges current NVML observations with all
durable active assignments. It selects the lowest-index `FREE` devices, persists
the complete assignment and `STARTING` transition in one `BEGIN IMMEDIATE`
transaction, and only then launches. No `await` or second allocation occurs in the
middle of the logical reservation. A launch failure becomes `FAILED` and releases
the full set.

## Physical and logical GPU states

NVML supplies UUID, index, memory, and compute PIDs. Utilization is deliberately
ignored. A process reported on an unreserved device makes it `EXTERNAL`. On an
assigned device, `/proc` and `getpgid` establish whether each reported PID belongs
to the managed process group. A foreign PID conflicting with a reservation makes
the state `UNKNOWN`, while retaining the reservation.

An empty NVML process list does not override `STARTING`/`RUNNING`; CUDA may not have
initialized yet or may temporarily be idle. Per-device observation failures and
global scan failures produce `UNKNOWN`, which is fail-closed.

## Scheduling and reconciliation

Submissions, cancellations, and managed exits trigger scheduling immediately. A
2-second loop additionally performs:

```text
NVML snapshot → merge reservations → reap recovered dead jobs → backfill queue
```

Waiting jobs are visited oldest first. If one cannot fit, the pass continues to
younger jobs. This is useful workstation backfilling, without MVP priority or
anti-starvation machinery.

Jobs in a held group are skipped. The set of held groups lives in the `group_holds`
table and is mirrored in memory, and the pass consults it before *every* launch, not
once per pass: a launch failure is recorded synchronously inside the pass, and can
itself be the failure that puts the group on hold. Only a failure of a job this
daemon launched and watched can trigger a hold; restart recovery never does.

Key uniqueness is enforced where the job row is inserted, inside the same `BEGIN
IMMEDIATE` transaction, rather than by a check before it.

## Process lifecycle

Jobs use direct argv execution with `start_new_session=True`; the new session's
process group is persisted. Standard output and error point to one job log. The
daemon waits for the command and, if a launcher exits early, retains resources
until the rest of the process group disappears. Exit zero is `DONE`; other codes
are `FAILED`.

Cancellation first checks `/proc/PID/stat` start time against the persisted value,
then signals the PGID with `SIGTERM`. It escalates to `SIGKILL` after the configured
grace period. It does not infer ownership merely from a numeric PID.

## IPC and single-instance behavior

Each connection carries one JSON object and one JSON response, newline-delimited.
The socket and runtime directory are mode 0600/0700. A held `flock` is the
single-instance authority; only after acquiring it may startup remove a stale
socket. No pickle or shell command string crosses the protocol.

## Crash and reboot recovery

At startup, unfinished rows are compared with:

- current Linux boot ID;
- saved PID and `/proc` start-time ticks;
- current GPU UUID inventory;
- NVML process ownership.

A matching live job keeps its reservation even if NVML reports no CUDA PID. A dead
or reused PID is failed and released. A changed boot ID means interrupted by
reboot. Missing allocated UUIDs make a live process `ORPHANED`; any saved UUID that
still exists remains unavailable through an `UNKNOWN` reservation. Ambiguity never
becomes `FREE` merely because SQLite is old.

Daemon shutdown cancels only its own monitor tasks. It intentionally does not
signal user jobs. On restart, those jobs are logically reattached. Linux does not
allow the new daemon to recover their exit status, so their eventual disappearance
is recorded conservatively as `FAILED` with an explanatory reason.


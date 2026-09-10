# Contributing to gq

Thanks for taking a look. `gq` is deliberately small, and the bar for adding to it is
correspondingly high — please read "Scope" before starting on a feature.

## Getting set up

```bash
git clone https://github.com/BirddWang/gq
cd gq
uv sync --all-groups
uv run pytest
```

Most of the suite runs against a fake NVML provider (`tests/conftest.py`), so you do
**not** need an NVIDIA GPU to develop or to run CI. If you do have one, the hardware
discovery tests are opt-in:

```bash
GQ_RUN_GPU_TESTS=1 uv run pytest tests/integration
```

Before opening a pull request:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy
uv run pytest
```

### Testing against a scratch instance

Never develop against your real queue. All state is redirectable, so run an isolated
daemon instead:

```bash
export GQ_DATA_DIR=/tmp/gq-dev/data
export GQ_RUNTIME_DIR=/tmp/gq-dev/run
export GQ_RECONCILE_INTERVAL=0.5
export GQ_CANCEL_GRACE_SECONDS=2
gq daemon start
```

A separate `GQ_RUNTIME_DIR` gets its own lock and socket, so it will not collide with
a daemon you already have running.

## Scope

The MVP intentionally omits multi-node scheduling, CPU/RAM allocation, preemption,
MIG, containers, and fractional GPU sharing. Some of these are on the roadmap and
some are permanent non-goals. Please open an issue to discuss a feature before
writing it — a rejected PR is a waste of your time, and this project would rather
stay small than grow every reasonable idea.

## The one rule that matters

> A GPU assigned to a managed job is never simultaneously allocatable.

Everything in `allocator.py`, `scheduler.py`, and `database.py` exists to hold that
line, and the design **fails closed**: when ownership is ambiguous, the GPU becomes
`UNKNOWN` and no one gets it. Three consequences to keep in mind when changing
allocation code:

- **Utilization is never an ownership signal.** A Jupyter kernel sitting at 0% still
  owns its CUDA context. Only NVML compute-process presence and `gq`'s own durable
  reservations decide ownership.
- **The reservation and the state transition are one transaction.** `Database.reserve`
  moves `WAITING → STARTING` and writes the full GPU assignment inside a single
  `BEGIN IMMEDIATE`. Do not introduce an `await` or a second allocation between them.
- **Process identity is checked before signalling.** Use the helpers in
  `processes.py`; never signal a bare PID. A recycled PID must not be mistaken for
  the original job.

If a change would weaken any of these, it needs a very good argument and a test that
demonstrates the new invariant holds.

## Compatibility

- **Database.** Never edit the baseline schema to change an existing table. Append a
  migration in `MIGRATIONS` in `database.py`, keyed by the version it produces, and
  bump `SCHEMA_VERSION`. Users have real job history; a release that discards it is a
  bug. Test your migration against a copy of a populated database.
- **IPC.** Adding a request type is backward compatible. Changing or removing the
  meaning of an existing field is not — bump `PROTOCOL_VERSION` in `protocol.py`.
- **`--json` output** is the supported scripting interface. Adding fields is fine;
  renaming or removing them requires a protocol bump. Table layouts are for humans
  and may change freely.

## Style

Match the surrounding code. It is plain, typed Python with no framework: standard
library plus `nvidia-ml-py`, and `from __future__ import annotations` everywhere.

**Adding a runtime dependency requires a strong justification.** Being installable
with one small dependency is a feature of this project, not an accident.

Comments explain *why*, especially where the reasoning is subtle — race conditions,
PID reuse, fail-closed choices. Do not add comments that restate the code.

## Commits and pull requests

Write commit subjects in the imperative mood ("add lease reattachment", not "added"
or "adds"). Keep the PR description focused on what changed and why; if it fixes an
issue, link it. A PR that changes behavior should update `CHANGELOG.md` under an
`## [Unreleased]` heading, and one that changes the interface should update the
relevant file in `docs/`.

## Reporting bugs

Include your `gq` version (`gq --version`), Linux distribution, Python version, and
NVIDIA driver version, plus:

```bash
gq gpu
gq ps
gq show JOB            # for a specific job
tail -n 100 ~/.local/share/gq/gq-daemon.log
```

Please skim the daemon log before attaching it — it records job commands and file
paths. **Never attach `gq.sqlite3`**; it contains the stored environment of every
job. See [SECURITY.md](SECURITY.md).

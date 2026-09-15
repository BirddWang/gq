# gq

[![CI](https://github.com/BirddWang/gq/actions/workflows/ci.yml/badge.svg)](https://github.com/BirddWang/gq/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gq-local)](https://pypi.org/project/gq-local/)
[![Python](https://img.shields.io/pypi/pyversions/gq-local)](https://pypi.org/project/gq-local/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A small, single-user GPU job scheduler for one Linux workstation. Queue commands,
get whole NVIDIA GPUs assigned atomically, and keep jobs running after you close the
terminal.

```bash
gq -g 1 uv run train.py          # queued, runs when a GPU is genuinely free
gq -g 2 torchrun --nproc-per-node=2 train.py
gq ps                            # what is queued and running
gq logs -f 12                    # follow a job's output
```

No cluster, no root, no config file. One `pip install`, one user-level daemon.

## The part that matters

Most of the time, "is this GPU free?" is answered by looking at utilization. That
answer is wrong in the case that bites hardest on a workstation:

> Your Jupyter kernel finished a cell an hour ago. It sits at **0% utilization** and
> still holds a CUDA context and several GB of memory. Start a training run on that
> GPU and you get an OOM — or worse, two jobs quietly fighting over one card.

`gq` treats a GPU as allocatable only when **both** of these hold:

1. `gq` itself has no reservation on it, and
2. NVML reports no compute process on it that `gq` does not own.

Utilization is never used as an ownership signal. A process `gq` does not recognise
makes the GPU `EXTERNAL` and off-limits — `gq` routes around it and never kills it.
When ownership is ambiguous, the GPU becomes `UNKNOWN` and stays unallocated. The
design fails closed: the worst case is an idle GPU, never a double-booked one.

That care extends to the parts that usually break:

- **Reservation is atomic.** The `WAITING → STARTING` transition and the full GPU
  assignment are written in one SQLite transaction, before the process launches. A
  crash between the two is not possible.
- **Restarts do not lose jobs.** Stopping the daemon leaves your jobs running. On
  restart it reconciles against saved PID, `/proc` start-time ticks, process group,
  Linux boot ID, and current GPU UUIDs, and reattaches what it can prove is the same
  process.
- **Cancellation cannot hit the wrong process.** `gq` signals a saved process group
  only after checking process identity, so a recycled PID is never mistaken for your
  job.

## Install

```bash
uv tool install gq-local     # or: pipx install gq-local, pip install gq-local
```

Requires Linux, Python 3.11+, an NVIDIA driver with NVML, and no root. The only
runtime dependency is `nvidia-ml-py`.

## Quick start

```bash
gq daemon start          # optional; any command starts it on demand
gq gpu                   # what each GPU is doing, and who owns it
gq -g 1 uv run train.py  # submit; returns immediately
gq ps
gq logs -f 1
```

`gq` sets `CUDA_VISIBLE_DEVICES` for the job, so physical GPUs 2,3 appear inside your
code as `cuda:0` and `cuda:1`.

Scripts can carry their own requirements:

```bash
#!/bin/bash
#gq --gpus=2
#gq --name=experiment

uv run train.py
```

```bash
gq submit experiment.sh
```

Day to day:

| Command | |
|---|---|
| `gq -g N CMD...` / `gq run -g N CMD...` | submit a command |
| `gq submit SCRIPT` | submit a shell script with `#gq` directives |
| `gq ps [--limit N] [--json]` | list jobs |
| `gq gpu [--json]` | physical and logical GPU state |
| `gq show JOB [--json]` | everything about one job |
| `gq logs [-f] JOB` | read or follow a job's output |
| `gq cancel JOB...` | cancel jobs; takes ranges (`300-440`), `--group`, `--waiting` |
| `gq retry JOB...` | resubmit failed or cancelled jobs; `--group NAME --failed` |
| `gq groups` | every group with its progress, and whether it is held |
| `gq rm JOB...` | delete terminal jobs and their logs |
| `gq clean --older-than 30d` | sweep old terminal jobs and their logs |
| `gq daemon start\|stop\|status` | manage the daemon |
| `gq daemon pause\|resume` | stop or restart launching queued jobs |
| `gq update [--check] [--wait]` | upgrade to the latest release and restart the daemon |

`--json` is the supported scripting interface; table layouts may change between
releases.

## Running a sweep

Give the batch a group and each cell a key, and the script becomes safe to re-run:

```bash
export GQ_GROUP=xsum-ablation
for SEED in 1 2 3; do
  for CELL in kl kl-enc_hddn kl-enc_self; do
    gq -g 1 --key "xsum/$CELL/seed$SEED" uv run trainer.py --losses ... --seed "$SEED"
  done
done
```

- Re-running it submits only cells that are missing, failed, or cancelled.
- `gq groups` shows the sweep's progress; `gq cancel --group xsum-ablation` stops it.
- If the batch is broken, gq **holds the group** after 3 jobs in a row fail within a
  minute, instead of letting every cell fail the same way. Fix it, then
  `gq retry --group xsum-ablation --failed`.

See [Sweeps](docs/usage.md#sweeps-groups-keys-and-retries) for the details.

## A note on secrets

A queued job may not start for hours, so `gq` saves your environment in order to run
it later with the same context. Variables that look like credentials are dropped
rather than stored, and `gq` tells you which:

```text
gq: not persisting 1 secret-looking variable(s): HF_TOKEN
    use --env-all (or GQ_ENV_ALL=1) to keep them, or --env-keep NAME
```

If a job genuinely needs one, `--env-keep HF_TOKEN` puts it back. Anything kept is
stored in plaintext in a `0600` SQLite database — see [SECURITY.md](SECURITY.md).

## Is this the right tool?

| You want | Use |
|---|---|
| Many machines, many users, fair-share, accounting | Slurm, or a Kubernetes scheduler |
| One workstation, your own jobs, whole GPUs, zero setup | **gq** |
| A generic job queue with no GPU awareness | [task-spooler](https://vicerveza.homeunix.net/~viric/soft/ts/) |
| Cloud or multi-node orchestration for ML | SkyPilot, Determined |

`gq` deliberately does **not** do multi-node scheduling, multiple users, CPU/RAM
allocation, priorities, preemption, MIG, containers, or fractional GPU sharing. Some
of those are on the roadmap; see [CONTRIBUTING.md](CONTRIBUTING.md#scope) before
proposing one.

## Documentation

- [Usage](docs/usage.md) — every command, `#gq` directives, environment handling, retention
- [Architecture](docs/architecture.md) — allocation invariant, IPC, crash and reboot recovery
- [Troubleshooting](docs/troubleshooting.md) — stuck GPUs, stuck jobs, daemon problems
- [Security](SECURITY.md) — threat model and how credentials are handled
- [Changelog](CHANGELOG.md)

## Development

```bash
git clone https://github.com/BirddWang/gq
cd gq
uv sync --all-groups
uv run pytest
```

The suite runs against a fake NVML provider, so **no GPU is required** to develop or
to run CI. Hardware tests are opt-in:

```bash
GQ_RUN_GPU_TESTS=1 uv run pytest tests/integration
```

Contributions are welcome — please read [CONTRIBUTING.md](CONTRIBUTING.md) first,
especially the section on the allocation invariant.

## License

MIT — see [LICENSE](LICENSE).

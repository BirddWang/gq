# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-15

### Added

- **`gq update`** upgrades gq in place. It detects whether gq came from `uv tool`,
  `pipx`, or `pip`, runs the matching upgrade, and restarts the daemon on the new
  version. Installs from a source checkout, git URL, or editable mode are never
  switched to PyPI silently; `gq update` prints the command to run instead.
  `--check` only reports whether a release is available.
- Because a restarted daemon cannot collect exit status from jobs it did not start,
  `gq update` **refuses while jobs are running**. `--wait` pauses the queue and lets
  them finish first; `--force` proceeds anyway.
- **`gq daemon pause` and `gq daemon resume`** stop and restart the launching of
  queued jobs without affecting running ones. `gq ps` and `gq daemon status` report a
  paused queue.
- Trove classifiers for Python 3.11–3.13, so PyPI and its badges show supported
  versions.

## [0.1.1] - 2026-09-10

A hardening release. No new scheduling behavior; everything here came from defects
and rough edges observed in the first week of real use.

### Fixed

- The daemon no longer logs a traceback every time a client hangs up mid-response.
  `gq ps | head` and Ctrl-C out of `gq logs -f` are ordinary events, but each one
  raised `ConnectionResetError`/`BrokenPipeError` out of the connection callback.
- Running as root inside a container is now allowed without `GQ_ALLOW_ROOT=1`.
  Docker and Jupyter images commonly have uid 0 as the only account, and the guard
  made `gq` unusable there. It still refuses on what looks like a regular host, where
  the warning is genuinely warranted.
- `PRAGMA foreign_keys = ON` is now issued as its own statement rather than inside
  the schema script, so `ON DELETE CASCADE` on `job_gpus` cannot silently be a no-op.

### Added

- **Schema migrations.** The database carries a `PRAGMA user_version` and applies
  ordered migrations, each in one transaction with its version bump. Databases
  created before this release are adopted in place, not rebuilt.
- **`gq rm JOB...`** deletes terminal jobs and their logs. Active jobs are refused,
  and one bad id rejects the whole batch rather than partially destroying history.
- **`gq clean --older-than AGE [--state STATE] [-y]`** sweeps old terminal jobs and
  their logs. Defaults to 30d and confirms before deleting.
- **`--json`** on `gq ps`, `gq gpu`, and `gq show`.
- **Protocol versioning.** Every request carries a protocol number; a mismatched
  daemon reports how to restart instead of failing on a missing field. `gq daemon
  status` warns when the running daemon predates the installed client.

### Changed

- **Secret-looking environment variables are no longer persisted by default.** The
  submission environment is stored verbatim in SQLite so queued jobs survive a daemon
  restart, which made every captured credential a durable one. Variables matching
  `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `*_KEY`, `API_KEY` or the `AWS_`,
  `AZURE_`, `GCP_`, `GOOGLE_APPLICATION_` prefixes are dropped, and the dropped names
  are printed at submit time rather than being removed silently.

  This is a **behavior change that can break jobs** which need a credential at
  runtime — `HF_TOKEN` for a private Hugging Face dataset, for instance. Restore the
  old behavior per-job with `--env-all`, globally with `GQ_ENV_ALL=1`, or for one
  variable with `--env-keep HF_TOKEN`.

## [0.1.0] - 2026-08-31

Initial MVP: queueing, atomic NVML-checked GPU allocation, `CUDA_VISIBLE_DEVICES`
assignment, detached execution, cancellation, and crash/reboot recovery.

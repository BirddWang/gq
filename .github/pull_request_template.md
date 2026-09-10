## What this changes

<!-- And why. Link the issue it addresses, if there is one. -->

## Checklist

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` and `uv run mypy` pass
- [ ] Tests cover the change
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`, if behavior changed
- [ ] `docs/` updated, if the interface changed
- [ ] Schema change appends a migration and bumps `SCHEMA_VERSION` (see CONTRIBUTING.md)
- [ ] IPC change bumps `PROTOCOL_VERSION`, if it is not backward compatible

## Allocation safety

<!-- Delete if this does not touch allocation, cancellation, or recovery.
     Otherwise: which invariant does this rely on, and what test shows it holds? -->

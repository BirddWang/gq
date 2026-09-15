# Security

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/BirddWang/gq/security/advisories/new)
rather than in a public issue. Include the `gq` version, your Python and NVIDIA
driver versions, and the smallest reproduction you have.

## Threat model

`gq` is a **single-user** scheduler. It runs unprivileged, as you, and it assumes
everything that can already act as your user is trusted. It is not a sandbox and not
an access-control system:

- it does not isolate jobs from each other or from your account;
- it does not stop anyone from using a GPU without going through `gq`;
- it cannot prevent a job from reading anything your user can read.

What it *does* guarantee is that `gq` will not hand the same GPU to two jobs it
manages, and will not allocate a GPU that any process outside `gq` is computing on.

Because it is single-user, everything it owns is private to you:

| Path | Mode |
|---|---|
| data directory, log directory, runtime directory | `0700` |
| Unix socket | `0600` |
| PID file | `0600` |

The daemon refuses to run as root on what appears to be a regular host, since jobs
inherit its privileges. It permits root inside a container, where uid 0 is often the
only account available. `GQ_ALLOW_ROOT=1` overrides the check.

The IPC protocol carries one JSON object per connection. No pickle and no shell
command string ever crosses it; commands are stored and executed as argument vectors,
never interpolated through a shell.

## Credentials in the job database

A queued job may not start for hours or days, so `gq` persists its submission
environment in SQLite in order to run it later with the same context. Anything
captured is therefore stored **at rest, in plaintext**, in
`~/.local/share/gq/gq.sqlite3`.

Since 0.1.1, variables whose names look like credentials are dropped at submission
rather than stored — names containing `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`,
`API_KEY`, `APIKEY`, or ending in `_KEY`, and names beginning with `AWS_`, `AZURE_`,
`GCP_`, or `GOOGLE_APPLICATION_`. `gq` prints which names it dropped.

This is a **name-based heuristic and it will miss things.** A credential in a
variable named `MY_THING` is stored like any other value. If a job needs a secret and
you re-add it with `--env-all`, `--env-keep NAME`, or `GQ_ENV_ALL=1`, that secret
becomes durable in the database.

Consequences worth planning around:

- treat `gq.sqlite3` as sensitive; do not commit it, copy it to shared storage, or
  attach it to a bug report;
- `gq show` and `--json` never return the stored environment, but anyone who can read
  the file can read it directly;
- prefer having jobs read credentials from a file or a secret manager at run time
  over passing them through the environment;
- `gq rm` and `gq clean` delete the rows, but SQLite may retain freed pages. Use
  `VACUUM` if you need the bytes gone.

## Network access

`gq` makes no network connections, with one exception: `gq update` fetches
`https://pypi.org/pypi/gq-local/json` over HTTPS to learn the latest version, then
runs your existing package manager (`uv`, `pipx`, or `pip`), which downloads from
whatever index that tool is configured to use. The daemon never touches the network.

## Cancellation and process identity

Cancellation signals a saved process group, never a bare PID. Before signalling,
`gq` checks the saved `/proc/PID/stat` start-time ticks and the Linux boot ID, so a
recycled PID is not mistaken for the original job. `gq` never signals a process it
did not start.

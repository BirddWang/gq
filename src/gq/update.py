"""Work out how gq was installed, and what upgrading it means.

gq can reach a machine several ways (uv tool, pipx, pip into a virtualenv, or a
source checkout), and each has a different upgrade command. Guessing wrong is worse
than refusing: `pip install --upgrade` inside a uv tool environment, for instance,
leaves uv's receipt out of step with what is installed. So detection only returns a
command when it is confident, and otherwise explains what to run by hand.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
from typing import Any

from . import __version__

DISTRIBUTION = "gq-local"
PYPI_JSON_URL = f"https://pypi.org/pypi/{DISTRIBUTION}/json"


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Installation:
    kind: str
    description: str
    # The argv that upgrades this installation, or None when gq should not run one
    # itself. `advice` then says what to do instead.
    command: list[str] | None
    advice: str | None = None


def _read_direct_url() -> dict[str, Any] | None:
    """PEP 610 metadata, present only when gq was installed from a path, VCS, or URL."""
    try:
        text = metadata.distribution(DISTRIBUTION).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def detect_installation() -> Installation:
    """Classify the installation this process is running from."""
    try:
        metadata.distribution(DISTRIBUTION)
        installed = True
    except metadata.PackageNotFoundError:
        installed = False
    return classify_installation(
        prefix=Path(sys.prefix),
        executable=sys.executable,
        installed=installed,
        direct_url=_read_direct_url(),
        has_pip=util.find_spec("pip") is not None,
        which=shutil.which,
    )


def classify_installation(
    *,
    prefix: Path,
    executable: str,
    installed: bool,
    direct_url: dict[str, Any] | None,
    has_pip: bool,
    which: Callable[[str], str | None],
) -> Installation:
    uv_tool = (prefix / "uv-receipt.toml").exists()
    pipx = (prefix / "pipx_metadata.json").exists()

    if not installed:
        return Installation(
            "source",
            "running from a source tree without an installed distribution",
            None,
            "update the checkout with 'git pull'",
        )

    # Key presence, not truthiness: PEP 610 commonly writes `"archive_info": {}`.
    if direct_url is not None and "archive_info" not in direct_url:
        # Installed from a local directory or a VCS URL rather than from PyPI. An
        # upgrade from PyPI would silently switch the source, so describe the choice.
        dir_info = direct_url.get("dir_info")
        editable = isinstance(dir_info, dict) and bool(dir_info.get("editable"))
        where = str(direct_url.get("url", "a local source"))
        if uv_tool:
            to_pypi = f"uv tool install --force {DISTRIBUTION}"
        elif pipx:
            to_pypi = f"pipx install --force {DISTRIBUTION}"
        else:
            to_pypi = f"{executable} -m pip install --upgrade {DISTRIBUTION}"
        if editable:
            advice = f"this is an editable install; update it with 'git pull' in {where}"
        else:
            advice = (
                f"gq was installed from {where}, not from PyPI. To follow PyPI releases "
                f"from now on, run: {to_pypi}"
            )
        return Installation("source", f"installed from {where}", None, advice)

    if uv_tool:
        uv = which("uv")
        if uv is None:
            return Installation(
                "uv-tool",
                "a uv tool environment",
                None,
                f"'uv' is not on PATH; run 'uv tool upgrade {DISTRIBUTION}' where it is",
            )
        return Installation(
            "uv-tool", "a uv tool environment", [uv, "tool", "upgrade", DISTRIBUTION]
        )

    if pipx:
        pipx_path = which("pipx")
        if pipx_path is None:
            return Installation(
                "pipx",
                "a pipx environment",
                None,
                f"'pipx' is not on PATH; run 'pipx upgrade {DISTRIBUTION}' where it is",
            )
        return Installation("pipx", "a pipx environment", [pipx_path, "upgrade", DISTRIBUTION])

    if has_pip:
        return Installation(
            "pip",
            f"the Python environment at {prefix}",
            [executable, "-m", "pip", "install", "--upgrade", DISTRIBUTION],
        )
    uv = which("uv")
    if uv is not None:
        # uv-created virtualenvs ship without pip.
        return Installation(
            "pip",
            f"the Python environment at {prefix}",
            [uv, "pip", "install", "--python", executable, "--upgrade", DISTRIBUTION],
        )
    return Installation(
        "pip",
        f"the Python environment at {prefix}",
        None,
        f"this environment has neither pip nor uv; upgrade {DISTRIBUTION} with the tool "
        "that installed it",
    )


def latest_release(timeout: float = 10.0) -> str:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"gq/{__version__}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        version = payload["info"]["version"]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"could not reach PyPI to check for a new release: {exc}") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise UpdateError("PyPI returned a response gq did not understand") from exc
    if not isinstance(version, str) or not version:
        raise UpdateError("PyPI returned a response gq did not understand")
    return version


_RELEASE_RE = re.compile(r"\d+(?:\.\d+)*")


def _release_tuple(version: str) -> tuple[int, ...] | None:
    if not _RELEASE_RE.fullmatch(version.strip()):
        return None
    parts = [int(part) for part in version.strip().split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()  # 1.0 and 1.0.0 are the same release
    return tuple(parts)


def is_newer(candidate: str, current: str) -> bool:
    """True when `candidate` is a later plain release than `current`.

    Deliberately avoids a dependency on `packaging`. Anything that is not a plain
    dotted release (a pre-release or local build on either side) compares only for
    inequality, so a development build is always offered the published release.
    """
    left, right = _release_tuple(candidate), _release_tuple(current)
    if left is None or right is None:
        return candidate.strip() != current.strip()
    return left > right

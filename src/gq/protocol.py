from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

# Bump whenever the request/response contract changes incompatibly. Every request
# carries it so a daemon left running across an upgrade reports a clear mismatch
# instead of failing on a missing field.
# Version 2 (gq 0.3.0) added job groups and keys. A version 1 daemon would silently
# ignore a submission's key and queue duplicates, hence the bump.
PROTOCOL_VERSION = 2


class ProtocolError(RuntimeError):
    pass


def request(socket_path: Path, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(socket_path))
            client.sendall(data)
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise ProtocolError("daemon response was unexpectedly large")
                if b"\n" in chunk:
                    break
    except (OSError, TimeoutError) as exc:
        raise ProtocolError(str(exc)) from exc
    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise ProtocolError("daemon closed the connection without a response")
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("daemon returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ProtocolError("daemon returned an invalid response")
    return result

"""Small subprocess boundary used by the desktop GUI."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence


def stream_command(
    command: Sequence[str],
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run a command, stream non-empty merged output lines, and return its exit code."""
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        return process.wait()

    try:
        for line in process.stdout:
            message = line.strip()
            if message and on_line is not None:
                on_line(message)
    finally:
        process.stdout.close()

    return process.wait()

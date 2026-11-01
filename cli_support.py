"""Small helpers shared by the command-line scripts."""

from __future__ import annotations

import sys


def use_utf8_output() -> None:
    """Write stdout/stderr as UTF-8 so Chinese text and emoji never crash a script.

    On Windows, output redirected to a pipe or file defaults to the ANSI code page
    (GBK, cp1252 …), where characters such as "✅" raise UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # None under pythonw.exe
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

from __future__ import annotations

import importlib
import importlib.util
import sys


def process_runner_module():
    assert importlib.util.find_spec("process_runner") is not None
    return importlib.import_module("process_runner")


def test_stream_command_returns_exit_code_and_output() -> None:
    process_runner = process_runner_module()
    lines: list[str] = []

    returncode = process_runner.stream_command(
        [sys.executable, "-c", "print('hello'); raise SystemExit(7)"],
        lines.append,
    )

    assert returncode == 7
    assert lines == ["hello"]


def test_stream_command_handles_success() -> None:
    process_runner = process_runner_module()

    assert process_runner.stream_command([sys.executable, "-c", "pass"]) == 0

"""Lightweight CLI smoke tests that do not require datasets or model weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script_name), "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_help_succeeds(script_name: str, expected_text: str) -> None:
    result = run_help(script_name)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "usage:" in output.lower(), output
    assert expected_text in output, output


def test_train_help() -> None:
    assert_help_succeeds("train.py", "--device")


def test_predict_help() -> None:
    assert_help_succeeds("predict.py", "--source")


def test_predict_device_help() -> None:
    assert_help_succeeds("predict.py", "--device")


def test_val_help() -> None:
    assert_help_succeeds("val.py", "--model")


def test_video_predict_help() -> None:
    assert_help_succeeds("video_predict.py", "source")


if __name__ == "__main__":
    test_train_help()
    test_predict_help()
    test_val_help()
    test_video_predict_help()
    print("CLI smoke tests passed.")

"""Smoke-test command-line help without datasets, weights, or GPU hardware."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "train.py": "--data",
    "predict.py": "--source",
    "val.py": "--model",
    "video_predict.py": "--output",
}


def main() -> int:
    for script, expected_option in COMMANDS.items():
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        if "usage:" not in result.stdout.lower() or expected_option not in result.stdout:
            print(f"{script} help output did not include expected usage for {expected_option}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

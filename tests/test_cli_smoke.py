from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_is_dependency_light() -> None:
    expected = {
        "train.py": "--device",
        "predict.py": "--source",
        "val.py": "--model",
        "video_predict.py": "source",
    }

    for script, marker in expected.items():
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / script), "--help"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert marker in output


def test_cli_rejects_missing_local_inputs_without_loading_runtime(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "missing.pt"
    missing_data = tmp_path / "missing.yaml"
    missing_source = tmp_path / "missing.jpg"
    commands = (
        ["train.py", "--model", str(missing_model), "--data", str(missing_data)],
        ["predict.py", "--model", str(missing_model), "--source", str(missing_source)],
        ["val.py", "--model", str(missing_model), "--data", str(missing_data)],
        ["video_predict.py", str(missing_model), str(missing_source)],
    )

    for script, *arguments in commands:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / script), *arguments],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 2, output
        assert "找不到模型权重" in output

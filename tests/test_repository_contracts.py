from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_ultralytics_version_source_exists() -> None:
    assert (REPO_ROOT / "ultralytics" / "__init__.py").is_file()


def test_readme_local_images_exist() -> None:
    missing: list[str] = []
    image_pattern = re.compile(r"!\[[^]]*]\(([^)]+)\)")

    for readme_name in ("README.md", "README.zh-CN.md"):
        readme = REPO_ROOT / readme_name
        for target in image_pattern.findall(readme.read_text(encoding="utf-8")):
            if "://" not in target and not (REPO_ROOT / target).is_file():
                missing.append(f"{readme_name}: {target}")

    assert missing == []


def test_tests_directory_can_be_tracked() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "tests/probe.py"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr


def test_repository_hygiene_script_passes() -> None:
    script = REPO_ROOT / "scripts" / "check_repository_hygiene.py"
    assert script.is_file()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_project_metadata_targets_this_repository() -> None:
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "YfengJ/steel-defect-detection" in contributing
    assert 'title: "Steel Surface Defect Detection with YOLOv8"' in citation
    assert 'url: "https://github.com/YfengJ/steel-defect-detection"' in citation

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def requirements(path: str) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def test_vendored_ultralytics_uses_compatible_torch_range() -> None:
    runtime = requirements("requirements.txt")

    assert "torch>=1.8.0,<2.6.0" in runtime
    assert "torchvision>=0.9.0,<0.21.0" in runtime
    assert not any(item.startswith("ultralytics") for item in runtime)


def test_development_checks_are_declared() -> None:
    development = requirements("requirements-dev.txt")

    assert "pytest>=9.1.1,<10.0" in development
    assert "ruff>=0.12,<1.0" in development


def test_dependabot_uses_conservative_runtime_updates() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    pip_update = next(
        update for update in config["updates"] if update["package-ecosystem"] == "pip"
    )

    assert pip_update["versioning-strategy"] == "increase-if-necessary"
    ignored = {rule["dependency-name"] for rule in pip_update["ignore"]}
    assert {"torch", "torchvision", "opencv-python", "numpy"} <= ignored

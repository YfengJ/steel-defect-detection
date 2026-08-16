from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def runtime_requirements() -> list[str]:
    return [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_torch_stays_below_weights_only_default_change() -> None:
    requirements = runtime_requirements()

    assert "torch>=1.8.0,<2.6.0" in requirements
    assert "torchvision>=0.9.0,<0.21.0" in requirements


def test_external_ultralytics_is_not_installed_over_vendored_source() -> None:
    requirements = runtime_requirements()

    assert not any(item.startswith("ultralytics") for item in requirements)


def test_numpy_keeps_legacy_trapz_api() -> None:
    requirements = runtime_requirements()

    assert any(item.startswith("numpy>=1.22.2,<2.4") for item in requirements)


def test_pytest_uses_patched_tmpdir_handling() -> None:
    requirements = [
        line.strip()
        for line in (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "pytest>=9.1.1,<10.0" in requirements


def test_dependabot_avoids_floor_only_updates() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    pip_update = next(
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "pip"
    )

    assert pip_update["versioning-strategy"] == "increase-if-necessary"
    assert not any(
        rule["dependency-name"] == "pytest" for rule in pip_update["ignore"]
    )


def test_ci_includes_weekly_health_check() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )

    assert config["on"]["schedule"] == [{"cron": "17 3 * * 1"}]
    assert "workflow_dispatch" in config["on"]


def test_ruff_stays_on_validated_minor_line() -> None:
    requirements = [
        line.strip()
        for line in (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "ruff>=0.12,<0.13" in requirements

from pathlib import Path


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

from pathlib import Path

from scripts.check_repository_hygiene import BANNED_SUFFIXES, validate_github_yaml


def test_repository_hygiene_blocks_dataset_archives() -> None:
    assert ".zip" in BANNED_SUFFIXES


def test_validate_github_yaml_accepts_valid_files(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: CI\non:\n  push:\n", encoding="utf-8")

    failures = validate_github_yaml(tmp_path, [Path(".github/workflows/ci.yml")])

    assert failures == []


def test_validate_github_yaml_reports_invalid_files(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: [CI\n", encoding="utf-8")

    failures = validate_github_yaml(tmp_path, [Path(".github/workflows/ci.yml")])

    assert len(failures) == 1
    assert failures[0].startswith("GitHub YAML syntax error in .github/workflows/ci.yml:")


def test_validate_github_yaml_ignores_non_github_yaml(tmp_path: Path) -> None:
    config = tmp_path / "mkdocs.yml"
    config.write_text("site_name: docs\n", encoding="utf-8")

    failures = validate_github_yaml(tmp_path, [Path("mkdocs.yml")])

    assert failures == []


def test_validate_github_yaml_ignores_deleted_file(tmp_path: Path) -> None:
    failures = validate_github_yaml(
        tmp_path,
        [Path(".github/obsolete.yml")],
    )

    assert failures == []

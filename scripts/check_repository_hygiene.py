"""Check repository hygiene for common open-source maintenance mistakes."""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path

import yaml

MAX_TRACKED_FILE_BYTES = 20 * 1024 * 1024
ARTIFACT_DIRS = {"datasets", "runs", "weights", "wandb"}
ARTIFACT_SUFFIXES = {
    ".engine",
    ".h5",
    ".mlmodel",
    ".mlpackage",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".tflite",
    ".torchscript",
    ".weights",
}
TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "MANIFEST.in",
}
MAINTAINED_MARKDOWN = {
    Path("CONTRIBUTING.md"),
    Path("README.md"),
    Path("README.zh-CN.md"),
    Path("RELEASE_NOTES.md"),
    Path("ROADMAP.md"),
    Path("SECURITY.md"),
    Path("SUPPORT.md"),
    Path("docs/dataset.md"),
    Path("docs/macos.md"),
    Path("docs/troubleshooting.md"),
}
MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def run_git(args: list[str]) -> bytes:
    return subprocess.check_output(["git", *args])


def tracked_files() -> list[Path]:
    raw = run_git(["ls-files", "-z"])
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]


def is_text_like(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def validate_github_yaml(repo_root: Path, paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for relative_path in paths:
        if (
            not relative_path.parts
            or relative_path.parts[0] != ".github"
            or relative_path.suffix.lower() not in {".yaml", ".yml"}
        ):
            continue
        path = repo_root / relative_path
        if not path.is_file():
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            failures.append(f"GitHub YAML syntax error in {relative_path}: {exc}")
    return failures


def validate_local_markdown_links(repo_root: Path) -> list[str]:
    failures: list[str] = []
    for relative_path in sorted(MAINTAINED_MARKDOWN):
        path = repo_root / relative_path
        if not path.is_file():
            failures.append(f"Missing maintained documentation: {relative_path}")
            continue
        for raw_target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if (
                not target
                or target.startswith("#")
                or "://" in target
                or target.startswith("mailto:")
            ):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not (path.parent / target).exists():
                failures.append(f"Broken local link in {relative_path}: {raw_target}")
    return failures


def main() -> int:
    repo_root = Path(run_git(["rev-parse", "--show-toplevel"]).decode().strip())
    files = tracked_files()
    failures: list[str] = []

    for relative_path in files:
        path = repo_root / relative_path
        if set(relative_path.parts) & ARTIFACT_DIRS:
            failures.append(f"Tracked local artifact directory: {relative_path}")
        if relative_path.suffix.lower() in ARTIFACT_SUFFIXES:
            failures.append(f"Tracked model or exported weight file: {relative_path}")

        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            failures.append(
                f"Tracked file exceeds 20 MB: {relative_path} ({size} bytes)"
            )
        if is_text_like(relative_path):
            data = path.read_bytes()
            if b"\r" in data:
                failures.append(f"Text file contains CR line endings: {relative_path}")

    failures.extend(validate_github_yaml(repo_root, files))
    failures.extend(validate_local_markdown_links(repo_root))

    if failures:
        print("Repository hygiene check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

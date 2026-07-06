"""Check repository hygiene for small open source maintenance mistakes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

MAX_TRACKED_FILE_BYTES = 20 * 1024 * 1024

BANNED_DIRS = {"datasets", "runs", "weights", "wandb"}
BANNED_SUFFIXES = {
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


def run_git(args: list[str]) -> bytes:
    return subprocess.check_output(["git", *args])


def tracked_files() -> list[Path]:
    raw = run_git(["ls-files", "-z"])
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]


def is_text_like(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def github_yaml_files(paths: list[Path]) -> list[Path]:
    return [
        path for path in paths
        if path.parts and path.parts[0] == ".github" and path.suffix.lower() in {".yaml", ".yml"}
    ]


def validate_github_yaml(repo_root: Path, paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for relative_path in github_yaml_files(paths):
        path = repo_root / relative_path
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            failures.append(f"GitHub YAML syntax error in {relative_path}: {exc}")
    return failures


def main() -> int:
    repo_root = Path(run_git(["rev-parse", "--show-toplevel"]).decode().strip())
    failures: list[str] = []
    files = tracked_files()

    for relative_path in files:
        path = repo_root / relative_path
        parts = set(relative_path.parts)
        suffix = relative_path.suffix.lower()

        if parts & BANNED_DIRS:
            failures.append(f"Tracked local artifact directory: {relative_path}")

        if suffix in BANNED_SUFFIXES:
            failures.append(f"Tracked model or exported weight file: {relative_path}")

        if path.exists() and path.is_file():
            size = path.stat().st_size
            if size > MAX_TRACKED_FILE_BYTES:
                failures.append(
                    f"Tracked file exceeds 20 MB: {relative_path} ({size} bytes)"
                )

            if is_text_like(relative_path):
                data = path.read_bytes()
                if b"\r\n" in data or b"\r" in data:
                    failures.append(f"Text file contains CR line endings: {relative_path}")

    failures.extend(validate_github_yaml(repo_root, files))

    if failures:
        print("Repository hygiene check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

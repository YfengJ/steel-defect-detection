from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gui_connects_local_path_validation() -> None:
    source = (REPO_ROOT / "ui.py").read_text(encoding="utf-8")

    for validator in (
        "validate_model_path",
        "validate_dataset_config",
        "validate_file_source",
        "validate_directory_source",
        "validate_video_source",
    ):
        assert validator in source


def test_gui_uses_exit_aware_runner_and_main_thread_callbacks() -> None:
    source = (REPO_ROOT / "ui.py").read_text(encoding="utf-8")

    assert "stream_command" in source
    assert "returncode == 0" in source
    assert "self.master.after(0, lambda message=msg: log_callback(message))" in source


def test_gui_video_worker_releases_capture_in_finally() -> None:
    tree = ast.parse((REPO_ROOT / "ui.py").read_text(encoding="utf-8"))
    video_worker = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "video_thread"
    )

    assert any(
        isinstance(node, ast.Try) and node.finalbody for node in ast.walk(video_worker)
    )


def test_gui_video_preview_releases_capture_in_finally() -> None:
    tree = ast.parse((REPO_ROOT / "ui.py").read_text(encoding="utf-8"))
    preview = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "browse_video_and_preview"
    )

    assert any(
        isinstance(node, ast.Try) and node.finalbody for node in ast.walk(preview)
    )

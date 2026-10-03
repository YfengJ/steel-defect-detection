"""Exercise GUI task callbacks without requiring a display or Tk installation."""

import ast
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gui():
    # Compile the actual task methods only, bypassing heavy GUI imports/layout.
    source = ast.parse((REPO_ROOT / "ui.py").read_text(encoding="utf-8"))
    gui_class = next(node for node in source.body if isinstance(node, ast.ClassDef))
    methods = {
        "run_subprocess", "start_training", "start_validation",
        "start_prediction", "start_batch_prediction", "run_video_inference",
    }
    gui_class.body = [node for node in gui_class.body if node.name in methods]
    callbacks = []

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

    namespace = {
        "subprocess": subprocess,
        "sys": sys,
        "threading": SimpleNamespace(Thread=ImmediateThread),
        "Path": Path,
        "datetime": datetime,
        "tk": SimpleNamespace(END="end"),
        "X": "x",
    }
    for validator in (
        "validate_model_path", "validate_dataset_config", "validate_file_source",
        "validate_directory_source", "validate_video_source",
    ):
        namespace[validator] = lambda *args: None
    exec(compile(ast.Module(body=[gui_class], type_ignores=[]), "ui.py", "exec"), namespace)
    instance = namespace["YOLOv8_GUI"]()
    instance.master = SimpleNamespace(after=lambda delay, func, *args: callbacks.append((func, args)))
    instance.log = Mock()
    instance.update_status = Mock()
    instance.show_toast = Mock()
    instance.callbacks = callbacks
    instance.namespace = namespace
    instance.video_worker_active = False
    return instance


@pytest.mark.parametrize("exit_code", [0, 7])
def test_subprocess_reports_exit_status_and_queues_ui_callbacks(gui, exit_code):
    log_callback = Mock()
    finish_callback = Mock()
    gui.run_subprocess(
        [sys.executable, "-c", f"print('first'); print('second'); exit({exit_code})"],
        log_callback=log_callback,
        finish_callback=finish_callback,
    )

    log_callback.assert_not_called()
    finish_callback.assert_not_called()
    for callback, args in gui.callbacks:
        callback(*args)
    assert [call.args[0] for call in log_callback.call_args_list] == ["first", "second"]
    finish_callback.assert_called_once_with(exit_code)


def test_missing_executable_still_finishes(gui, tmp_path):
    finish_callback = Mock()
    gui.run_subprocess([str(tmp_path / "missing-executable")], finish_callback=finish_callback)
    for callback, args in gui.callbacks:
        callback(*args)
    finish_callback.assert_called_once_with(1)


@pytest.mark.parametrize("task", ["training", "validation", "prediction", "batch_prediction"])
def test_failed_task_does_not_report_success(gui, task):
    for name in (
        "train_model", "train_data", "train_epochs", "train_batch", "train_imgsz",
        "val_model", "val_data", "predict_model", "predict_source", "predict_conf",
        "batch_model", "batch_data",
    ):
        setattr(gui, name, SimpleNamespace(get=lambda: "placeholder"))
    gui.train_gauge = Mock()
    gui.val_text = Mock()
    gui.run_subprocess = Mock()

    getattr(gui, f"start_{task}")()
    gui.run_subprocess.call_args.kwargs["finish_callback"](1)

    assert gui.update_status.call_args.args[1] == "danger"
    gui.show_toast.assert_not_called()
    if task == "training":
        gui.train_gauge.stop.assert_called_once()
        gui.train_gauge.pack_forget.assert_called_once()


def test_video_open_failure_releases_capture_and_resets_state(gui):
    capture = Mock()
    capture.isOpened.return_value = False
    gui.namespace["cv2"] = SimpleNamespace(VideoCapture=Mock(return_value=capture))
    gui.namespace["YOLO"] = Mock()
    gui.video_loop_running = False
    gui.video_model = SimpleNamespace(get=lambda: "local.pt")
    gui.video_status = Mock()

    gui.run_video_inference("1")

    assert gui.video_loop_running is False
    gui.namespace["cv2"].VideoCapture.assert_called_once_with(1)
    capture.release.assert_called_once()
    for callback, args in gui.callbacks:
        callback(*args)
    assert gui.update_status.call_args.args[1] == "danger"


def test_active_video_does_not_start_another_worker(gui):
    gui.video_loop_running = True
    gui.video_worker_active = True
    gui.run_video_inference("0")
    gui.update_status.assert_not_called()


def test_stopping_video_does_not_start_another_worker(gui):
    gui.video_loop_running = False
    gui.video_worker_active = True
    gui.run_video_inference("0")
    gui.update_status.assert_not_called()


def test_video_inference_failure_releases_capture(gui):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, object())
    model = Mock(side_effect=RuntimeError("device unavailable"))
    gui.namespace["cv2"] = SimpleNamespace(VideoCapture=Mock(return_value=capture))
    gui.namespace["YOLO"] = Mock(return_value=model)
    gui.video_loop_running = False
    gui.video_model = SimpleNamespace(get=lambda: "local.pt")
    gui.video_status = Mock()

    gui.run_video_inference("input.mp4")

    assert gui.video_loop_running is False
    assert gui.video_worker_active is False
    capture.release.assert_called_once()
    for callback, args in gui.callbacks:
        callback(*args)
    assert gui.update_status.call_args.args[1] == "danger"


def test_subprocess_non_utf8_logs_do_not_skip_completion(gui):
    finish_callback = Mock()
    gui.run_subprocess(
        [sys.executable, "-c", "import os; os.write(1, bytes([255, 10]))"],
        finish_callback=finish_callback,
    )
    for callback, args in gui.callbacks:
        callback(*args)
    finish_callback.assert_called_once_with(0)

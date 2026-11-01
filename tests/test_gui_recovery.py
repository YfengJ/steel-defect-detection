"""Exercise GUI task callbacks without requiring a display or Tk installation."""

import ast
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gui_logic


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gui():
    # Compile the actual task methods only, bypassing heavy GUI imports/layout.
    source = ast.parse((REPO_ROOT / "ui.py").read_text(encoding="utf-8"))
    gui_class = next(node for node in source.body if isinstance(node, ast.ClassDef))
    methods = {
        "run_subprocess", "start_training", "start_validation",
        "start_prediction", "start_batch_prediction", "run_video_inference",
        "show_video_frame", "_post", "_task_failed", "stop_task",
    }
    gui_class.body = [node for node in gui_class.body if getattr(node, "name", None) in methods]
    callbacks = []

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

    namespace = {
        **{name: getattr(gui_logic, name) for name in dir(gui_logic) if not name.startswith("_")},
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "time": time,
        "threading": SimpleNamespace(Thread=ImmediateThread),
        "Path": Path,
        "datetime": datetime,
        "tk": SimpleNamespace(END="end"),
        "ROOT": REPO_ROOT,
        "RUNS_DIR": REPO_ROOT / "runs" / "detect",
        "PYTHON": sys.executable,
    }
    for validator in (
        "validate_model_path", "validate_dataset_config", "validate_file_source",
        "validate_directory_source", "validate_video_source",
    ):
        namespace[validator] = lambda *args: None
    exec(compile(ast.Module(body=[gui_class], type_ignores=[]), "ui.py", "exec"), namespace)
    instance = namespace["YOLOv8_GUI"]()
    instance.master = SimpleNamespace(after=lambda delay, func, *args: callbacks.append((func, args)))
    instance.ui_queue = SimpleNamespace(put=callbacks.append)
    instance.log = Mock()
    instance.update_status = Mock()
    instance.show_toast = Mock()
    instance.callbacks = callbacks
    instance.namespace = namespace
    instance.video_worker_active = False
    instance.training_active = False
    instance.validation_active = False
    instance._closing = False
    instance._busy = set()
    instance._processes = {}
    instance._stop_requested = set()
    for view in (
        "show_log", "show_validation_error", "on_training_started", "on_training_finished",
        "on_validation_started", "on_validation_finished", "show_validation_metrics",
        "show_prediction_result", "show_batch_summary", "on_video_started", "on_video_finished",
        "update_video_stats",
    ):
        setattr(instance, view, Mock())
    return instance


def _set_inputs(gui, value="placeholder"):
    for name in (
        "train_model", "train_data", "train_imgsz", "val_model", "val_data",
        "predict_model", "predict_source", "batch_model", "batch_data",
    ):
        setattr(gui, name, SimpleNamespace(get=lambda: value))
    for name, default in (
        ("train_epochs", "3"), ("train_batch", "4"), ("train_workers", "0"),
        ("predict_conf", 0.25), ("batch_conf", 0.25), ("val_split", "val"),
        ("train_device", "自动"), ("val_device", "cpu"), ("predict_device", "自动"),
        ("batch_device", "自动"),
    ):
        setattr(gui, name, SimpleNamespace(get=lambda default=default: default))


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
    _set_inputs(gui)
    gui.train_gauge = Mock()
    gui.val_text = Mock()
    gui.run_subprocess = Mock()

    getattr(gui, f"start_{task}")()
    gui.run_subprocess.call_args.kwargs["finish_callback"](1)

    assert gui.update_status.call_args.args[1] == "danger"
    gui.show_toast.assert_not_called()
    gui.show_log.assert_called_once_with(True)
    if task == "training":
        assert gui.on_training_finished.call_args.args[0] == 1
        assert gui.training_active is False


def test_video_open_failure_releases_capture_and_resets_state(gui):
    capture = Mock()
    capture.isOpened.return_value = False
    gui.namespace["open_video_capture"] = Mock(return_value=capture)
    gui.namespace["YOLO"] = Mock()
    gui.video_loop_running = False
    gui.video_model = SimpleNamespace(get=lambda: "local.pt")
    gui.video_conf = SimpleNamespace(get=lambda: 0.25)
    gui.video_device = SimpleNamespace(get=lambda: "自动")

    gui.run_video_inference("1")

    assert gui.video_loop_running is False
    gui.namespace["open_video_capture"].assert_called_once_with("1")
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
    gui.namespace["open_video_capture"] = Mock(return_value=capture)
    gui.namespace["YOLO"] = Mock(return_value=model)
    gui.video_loop_running = False
    gui.video_model = SimpleNamespace(get=lambda: "local.pt")
    gui.video_conf = SimpleNamespace(get=lambda: 0.25)
    gui.video_device = SimpleNamespace(get=lambda: "自动")

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


@pytest.mark.parametrize("task, script", [
    ("training", "train.py"), ("validation", "val.py"),
    ("prediction", "predict.py"), ("batch_prediction", "predict.py"),
])
def test_tasks_use_repository_scripts_independent_of_cwd(gui, task, script):
    _set_inputs(gui)
    gui.train_gauge = Mock()
    gui.val_text = Mock()
    gui.run_subprocess = Mock()

    getattr(gui, f"start_{task}")()

    cmd = gui.run_subprocess.call_args.args[0]
    assert cmd[1] == str(REPO_ROOT / script)
    if "--project" in cmd:
        assert Path(cmd[cmd.index("--project") + 1]).is_absolute()


@pytest.mark.parametrize("task", ["training", "validation"])
def test_long_running_task_cannot_start_twice(gui, task):
    _set_inputs(gui)
    gui.train_gauge = Mock()
    gui.val_text = Mock()
    gui.run_subprocess = Mock()

    getattr(gui, f"start_{task}")()
    getattr(gui, f"start_{task}")()
    assert gui.run_subprocess.call_count == 1

    gui.run_subprocess.call_args.kwargs["finish_callback"](0)
    getattr(gui, f"start_{task}")()
    assert gui.run_subprocess.call_count == 2


def test_prediction_report_finds_result_with_original_suffix(gui, tmp_path):
    runs_dir = tmp_path / "runs"
    gui.namespace["RUNS_DIR"] = runs_dir
    _set_inputs(gui, str(tmp_path / "sample.bmp"))
    gui.run_subprocess = Mock()

    gui.start_prediction()
    cmd = gui.run_subprocess.call_args.args[0]
    save_dir = runs_dir / cmd[cmd.index("--name") + 1]
    (save_dir / "labels").mkdir(parents=True)
    (save_dir / "sample.bmp").write_bytes(b"result")
    (save_dir / "labels" / "sample.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")
    gui.run_subprocess.call_args.kwargs["finish_callback"](0)

    image_path, detections, reported_dir = gui.show_prediction_result.call_args.args
    assert image_path == save_dir / "sample.bmp"
    assert reported_dir == save_dir
    assert [(d.name, d.conf) for d in detections] == [("龟裂", 0.9)]
    assert gui.update_status.call_args.args[1] == "success"


def test_batch_without_detections_still_reports(gui, tmp_path):
    gui.namespace["RUNS_DIR"] = tmp_path / "runs"
    _set_inputs(gui)
    gui.run_subprocess = Mock()

    gui.start_batch_prediction()
    gui.run_subprocess.call_args.kwargs["finish_callback"](0)

    summary, _output_dir = gui.show_batch_summary.call_args.args
    assert summary.total_defects == 0
    assert gui.update_status.call_args.args[1] == "success"


def test_video_drops_frames_while_ui_is_busy(gui):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, "frame-1"), (True, "frame-2"), (False, None)]
    result = Mock(boxes=[object(), object()])
    model = Mock(return_value=[result])
    gui.namespace["open_video_capture"] = Mock(return_value=capture)
    gui.namespace["cv2"] = SimpleNamespace(cvtColor=Mock(return_value="rgb"), COLOR_BGR2RGB=4)
    gui.namespace["Image"] = SimpleNamespace(
        fromarray=Mock(return_value="pil"), Resampling=SimpleNamespace(BILINEAR="bilinear"),
    )
    gui.namespace["YOLO"] = Mock(return_value=model)
    gui.video_model = SimpleNamespace(get=lambda: "local.pt")
    gui.video_conf = SimpleNamespace(get=lambda: 0.4)
    gui.video_device = SimpleNamespace(get=lambda: "cpu")
    gui.video_canvas = object()
    gui.show_image_on_canvas = Mock()

    gui.run_video_inference("input.mp4")

    assert model.call_count == 2
    assert model.call_args.kwargs == {"conf": 0.4, "verbose": False, "device": "cpu"}
    result.plot.assert_called_once()
    frame_callbacks = [(func, args) for func, args in gui.callbacks if func == gui.show_video_frame]
    assert len(frame_callbacks) == 1
    func, args = frame_callbacks[0]
    func(*args)
    assert gui.video_frame_pending is False
    gui.show_image_on_canvas.assert_called_once_with("pil", gui.video_canvas, "bilinear")
    stats = gui.update_video_stats.call_args.args[0]
    assert stats["boxes"] == 2 and stats["frames"] == 1


def test_subprocess_runs_in_repository_root_with_unbuffered_output(gui, tmp_path):
    lines = []
    gui.run_subprocess(
        [sys.executable, "-c", "import os, sys; print(os.getcwd()); print(sys.flags.dont_write_bytecode, os.environ['PYTHONUNBUFFERED'])"],
        log_callback=lines.append,
    )
    for callback, args in gui.callbacks:
        callback(*args)
    assert Path(lines[0]).resolve() == REPO_ROOT
    assert lines[1].endswith("1")


def test_task_paths_are_absolute_and_options_forwarded(gui, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _set_inputs(gui, "relative/model.pt")
    gui.run_subprocess = Mock()

    gui.start_validation()

    cmd = gui.run_subprocess.call_args.args[0]
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "relative" / "model.pt")
    assert cmd[cmd.index("--split") + 1] == "val"
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_training_uses_unique_run_directory(gui):
    _set_inputs(gui)
    gui.run_subprocess = Mock()

    gui.start_training()

    cmd = gui.run_subprocess.call_args.args[0]
    name = cmd[cmd.index("--name") + 1]
    assert name.startswith("train_") and name != "train_result"
    assert cmd[cmd.index("--workers") + 1] == "0"
    assert "--device" not in cmd
    run_dir, epochs = gui.on_training_started.call_args.args
    assert run_dir.name == name and epochs == 3


@pytest.mark.parametrize("field, value", [("train_epochs", "0"), ("train_batch", "x"), ("train_workers", "-1")])
def test_training_rejects_invalid_numbers(gui, field, value):
    _set_inputs(gui)
    setattr(gui, field, SimpleNamespace(get=lambda: value))
    gui.run_subprocess = Mock()

    gui.start_training()

    gui.run_subprocess.assert_not_called()
    gui.show_validation_error.assert_called_once()
    assert gui.training_active is False


def test_stopped_training_reports_stop_instead_of_failure(gui):
    _set_inputs(gui)
    gui.run_subprocess = Mock()
    process = Mock()
    process.poll.return_value = None
    gui._processes["train"] = process

    gui.start_training()
    gui.stop_task("train")
    process.terminate.assert_called_once()
    gui.run_subprocess.call_args.kwargs["finish_callback"](-15)

    assert gui.update_status.call_args.args[1] == "warning"
    gui.show_log.assert_not_called()
    assert "train" not in gui._stop_requested


def test_validation_metrics_line_updates_view(gui):
    _set_inputs(gui)
    gui.run_subprocess = Mock()

    gui.start_validation()
    on_line = gui.run_subprocess.call_args.kwargs["log_callback"]
    on_line("mAP50 : 0.7")
    on_line('METRICS_JSON {"precision": 0.7, "classes": []}')

    gui.show_validation_metrics.assert_called_once_with({"precision": 0.7, "classes": []})


def test_post_is_ignored_after_close(gui):
    gui._closing = True
    gui._post(print, "late")
    assert gui.callbacks == []


def test_worker_updates_never_touch_tk_from_other_threads(gui):
    # Non-threaded Tcl builds silently drop after() calls made from worker threads.
    gui.master = SimpleNamespace(after=Mock())
    func = Mock()
    gui._post(func, 1)
    gui.master.after.assert_not_called()
    assert gui.callbacks == [(func, (1,))]

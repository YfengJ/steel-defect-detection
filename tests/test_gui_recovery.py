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
        "show_video_frame",
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
        "ROOT": REPO_ROOT,
        "RUNS_DIR": REPO_ROOT / "runs" / "detect",
        "CLASS_MAP": {0: "龟裂"},
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
    instance.training_active = False
    instance.validation_active = False
    return instance


def _set_inputs(gui, value="placeholder"):
    for name in (
        "train_model", "train_data", "train_epochs", "train_batch", "train_imgsz",
        "val_model", "val_data", "predict_model", "predict_source", "predict_conf",
        "batch_model", "batch_data",
    ):
        setattr(gui, name, SimpleNamespace(get=lambda: value))


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
    gui.show_image_on_canvas = Mock()
    gui.predict_canvas = object()
    gui.predict_report = Mock()

    gui.start_prediction()
    cmd = gui.run_subprocess.call_args.args[0]
    save_dir = runs_dir / cmd[cmd.index("--name") + 1]
    (save_dir / "labels").mkdir(parents=True)
    (save_dir / "sample.bmp").write_bytes(b"result")
    (save_dir / "labels" / "sample.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")
    gui.run_subprocess.call_args.kwargs["finish_callback"](0)

    gui.show_image_on_canvas.assert_called_once_with(save_dir / "sample.bmp", gui.predict_canvas)
    report = gui.predict_report.insert.call_args.args[1]
    assert "发现目标数量: 1" in report
    assert "龟裂: 90.0%" in report
    assert gui.update_status.call_args.args[1] == "success"


def test_batch_without_detections_still_reports(gui, tmp_path):
    gui.namespace["RUNS_DIR"] = tmp_path / "runs"
    _set_inputs(gui)
    gui.run_subprocess = Mock()
    gui.analyze_and_report_batch = Mock()

    gui.start_batch_prediction()
    gui.run_subprocess.call_args.kwargs["finish_callback"](0)

    gui.analyze_and_report_batch.assert_called_once()
    assert gui.update_status.call_args.args[1] == "success"


def test_video_drops_frames_while_ui_is_busy(gui):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, "frame-1"), (True, "frame-2"), (False, None)]
    result = Mock()
    model = Mock(return_value=[result])
    gui.namespace["cv2"] = SimpleNamespace(
        VideoCapture=Mock(return_value=capture), cvtColor=Mock(return_value="rgb"), COLOR_BGR2RGB=4,
    )
    gui.namespace["Image"] = SimpleNamespace(
        fromarray=Mock(return_value="pil"), Resampling=SimpleNamespace(BILINEAR="bilinear"),
    )
    gui.namespace["YOLO"] = Mock(return_value=model)
    gui.video_model = SimpleNamespace(get=lambda: "local.pt")
    gui.video_status = Mock()
    gui.video_canvas = object()
    gui.show_image_on_canvas = Mock()

    gui.run_video_inference("input.mp4")

    assert model.call_count == 2
    result.plot.assert_called_once()
    frame_callbacks = [(func, args) for func, args in gui.callbacks if func == gui.show_video_frame]
    assert len(frame_callbacks) == 1
    func, args = frame_callbacks[0]
    func(*args)
    assert gui.video_frame_pending is False
    gui.show_image_on_canvas.assert_called_once_with("pil", gui.video_canvas, "bilinear")

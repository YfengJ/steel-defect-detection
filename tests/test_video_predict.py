import ast
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from video_predict import VideoPredictor


@pytest.fixture
def video_runtime(monkeypatch):
    model = Mock(return_value=[])
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Mock(return_value=model)))
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = [64, 64, 25]
    capture.read.return_value = (False, None)
    writer = Mock()
    writer.isOpened.return_value = True
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr(cv2, "VideoWriter", Mock(return_value=writer))
    return capture, writer, model


def test_unopened_source_is_an_error_and_releases_capture(video_runtime):
    capture, writer, _ = video_runtime
    capture.isOpened.return_value = False
    with pytest.raises(ValueError, match="无法打开视频源"):
        VideoPredictor.run("local.pt", "missing.mp4")
    capture.release.assert_called_once()
    writer.release.assert_not_called()


def test_unopened_writer_is_an_error_and_releases_resources(video_runtime):
    capture, writer, _ = video_runtime
    writer.isOpened.return_value = False
    with pytest.raises(OSError, match="无法创建输出视频"):
        VideoPredictor.run("local.pt", "input.mp4", "missing/output.mp4")
    capture.release.assert_called_once()
    writer.release.assert_called_once()


def test_inference_failure_releases_resources(video_runtime):
    capture, writer, model = video_runtime
    capture.read.return_value = (True, object())
    model.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        VideoPredictor.run("local.pt", "input.mp4")
    capture.release.assert_called_once()
    writer.release.assert_called_once()


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), -1, 0])
def test_invalid_fps_uses_fallback(video_runtime, fps):
    capture, writer, _ = video_runtime
    capture.get.side_effect = [64, 64, fps]
    VideoPredictor.run("local.pt", "input.mp4")
    assert cv2.VideoWriter.call_args.args[2] == 25
    writer.release.assert_called_once()
    capture.release.assert_called_once()


def test_invalid_dimensions_are_rejected(video_runtime):
    capture, _, _ = video_runtime
    capture.get.side_effect = [0, 64, 25]
    with pytest.raises(ValueError, match="视频尺寸无效"):
        VideoPredictor.run("local.pt", "input.mp4")
    capture.release.assert_called_once()


def test_cli_exits_unsuccessfully_on_video_failure(monkeypatch):
    source_path = Path(__file__).resolve().parents[1] / "video_predict.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    entrypoint = module.body[-1]
    monkeypatch.setattr(sys, "argv", ["video_predict.py", "local.pt", "bad.mp4"])
    namespace = {
        "__name__": "__main__",
        "argparse": argparse,
        "VideoPredictor": SimpleNamespace(run=Mock(side_effect=OSError("writer failed"))),
    }
    with pytest.raises(SystemExit) as error:
        exec(compile(ast.Module(body=[entrypoint], type_ignores=[]), str(source_path), "exec"), namespace)
    assert error.value.code == 1

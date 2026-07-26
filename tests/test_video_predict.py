from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

from video_predict import VideoPredictor


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeCapture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _property: int) -> float:
        return 25.0

    def read(self):
        return True, object()

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, _frame) -> None:
        pass

    def release(self) -> None:
        self.released = True


def install_video_fakes(monkeypatch, capture: FakeCapture, writer: FakeWriter) -> None:
    ultralytics = types.ModuleType("ultralytics")

    class FailingYOLO:
        names = {0: "defect"}

        def __init__(self, _model_path: str) -> None:
            pass

        def __call__(self, _frame, **_kwargs):
            raise RuntimeError("inference failed")

    ultralytics.YOLO = FailingYOLO

    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FRAME_WIDTH = 1
    cv2.CAP_PROP_FRAME_HEIGHT = 2
    cv2.CAP_PROP_FPS = 3
    cv2.VideoCapture = lambda _source: capture
    cv2.VideoWriter_fourcc = lambda *_args: 0
    cv2.VideoWriter = lambda *_args: writer

    numpy = types.ModuleType("numpy")
    pil = types.ModuleType("PIL")
    pil.Image = object()
    pil.ImageDraw = object()

    class FakeImageFont:
        @staticmethod
        def truetype(_name: str, _size: int):
            return object()

        @staticmethod
        def load_default():
            return object()

    pil.ImageFont = FakeImageFont

    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "PIL", pil)


def test_run_releases_video_resources_when_inference_fails(monkeypatch) -> None:
    capture = FakeCapture()
    writer = FakeWriter()
    install_video_fakes(monkeypatch, capture, writer)

    result = VideoPredictor.run("model.pt", "video.mp4", "output.mp4")

    assert result is False
    assert capture.released
    assert writer.released


def test_cli_returns_failure_when_processing_fails(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "video_predict.py"),
            str(tmp_path / "missing.pt"),
            str(tmp_path / "missing.mp4"),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout + result.stderr

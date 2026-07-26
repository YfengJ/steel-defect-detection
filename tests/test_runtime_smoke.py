from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings(
    "ignore:The .grad attribute of a Tensor:UserWarning"
)


def test_cpu_prediction_pipeline_without_external_weights() -> None:
    model = YOLO(REPO_ROOT / "ultralytics/cfg/models/v8/yolov8.yaml")
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    results = model.predict(source=image, imgsz=64, device="cpu", verbose=False)

    assert len(results) == 1
    assert tuple(results[0].orig_shape) == (64, 64)

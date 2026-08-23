"""Minimal CPU integration coverage for YOLO training and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from path_validation import resolved_dataset_config_path


def _write_synthetic_dataset(root: Path) -> Path:
    dataset_root = root / "dataset"
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        for index in range(2):
            pixels = np.full((64, 64, 3), 40 + index * 20, dtype=np.uint8)
            pixels[20:44, 18:46] = (190, 190, 190)
            Image.fromarray(pixels).save(image_dir / f"sample-{index}.png")
            (label_dir / f"sample-{index}.txt").write_text(
                "0 0.5 0.5 0.4375 0.375\n",
                encoding="utf-8",
            )

    config = root / "dataset.yaml"
    config.write_text(
        "\n".join(
            (
                "path: dataset",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: synthetic_defect",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config


def test_cpu_training_and_validation_smoke(tmp_path: Path) -> None:
    from ultralytics import YOLO

    config = _write_synthetic_dataset(tmp_path)
    output_root = tmp_path / "runs"

    with resolved_dataset_config_path(config) as resolved_config:
        model = YOLO("yolov8n.yaml")
        training = model.train(
            data=str(resolved_config),
            epochs=1,
            imgsz=64,
            batch=2,
            workers=0,
            device="cpu",
            project=str(output_root),
            name="train-smoke",
            exist_ok=True,
            plots=False,
            verbose=False,
        )

        best_weight = Path(training.save_dir) / "weights" / "best.pt"
        assert best_weight.is_file()

        metrics = YOLO(str(best_weight)).val(
            data=str(resolved_config),
            split="val",
            imgsz=64,
            batch=2,
            workers=0,
            device="cpu",
            project=str(output_root),
            name="val-smoke",
            plots=False,
            verbose=False,
        )

    assert metrics.box.map >= 0
    assert metrics.box.map50 >= 0
    assert not (Path.cwd() / "runs" / "detect" / "train-smoke").exists()

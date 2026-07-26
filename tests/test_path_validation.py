from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def validation_module():
    assert importlib.util.find_spec("path_validation") is not None
    return importlib.import_module("path_validation")


def test_model_and_source_paths(tmp_path: Path) -> None:
    validation = validation_module()
    model = tmp_path / "best.pt"
    image = tmp_path / "sample.jpg"
    folder = tmp_path / "images"
    model.touch()
    image.touch()
    folder.mkdir()

    assert validation.validate_model_path(model) is None
    assert validation.validate_file_source(image, "图片") is None
    assert validation.validate_directory_source(folder, "图片目录") is None
    assert "模型权重" in validation.validate_model_path(tmp_path / "missing.pt")
    assert "图片" in validation.validate_file_source(tmp_path / "missing.jpg", "图片")
    assert "图片目录" in validation.validate_directory_source(
        tmp_path / "missing", "图片目录"
    )


def test_empty_directory_source_is_not_current_working_directory() -> None:
    validation = validation_module()

    message = validation.validate_directory_source("", "图片目录")

    assert message is not None
    assert "图片目录" in message


def test_video_source_accepts_camera_and_checks_files(tmp_path: Path) -> None:
    validation = validation_module()

    assert validation.validate_video_source("0") is None
    assert "视频源" in validation.validate_video_source(tmp_path / "missing.mp4")


def test_prediction_source_accepts_files_and_directories(tmp_path: Path) -> None:
    validation = validation_module()
    image = tmp_path / "sample.jpg"
    folder = tmp_path / "images"
    image.touch()
    folder.mkdir()

    assert validation.validate_prediction_source(image) is None
    assert validation.validate_prediction_source(folder) is None
    assert "预测源" in validation.validate_prediction_source(tmp_path / "missing")


def test_dataset_config_validates_images_and_labels(tmp_path: Path) -> None:
    validation = validation_module()
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )

    assert validation.validate_dataset_config(config) is None
    assert validation.resolve_dataset_config_path(config) == config.resolve()

    (dataset / "labels" / "val").rmdir()
    message = validation.validate_dataset_config(config)
    assert message is not None
    assert "标签目录" in message


def test_dataset_config_reports_malformed_yaml(tmp_path: Path) -> None:
    validation = validation_module()
    config = tmp_path / "dataset.yaml"
    config.write_text("path: [\n", encoding="utf-8")

    message = validation.validate_dataset_config(config)

    assert message is not None
    assert "无法读取数据集配置" in message


def test_dataset_config_rejects_image_file_instead_of_directory(tmp_path: Path) -> None:
    validation = validation_module()
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "train").touch()
    (dataset / "images" / "val").mkdir()
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )

    message = validation.validate_dataset_config(config)

    assert message is not None
    assert "图像路径" in message


def test_dataset_config_requires_class_metadata(tmp_path: Path) -> None:
    validation = validation_module()
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    message = validation.validate_dataset_config(config)

    assert message is not None
    assert "names 或 nc" in message


def test_dataset_config_accepts_multiple_split_directories(tmp_path: Path) -> None:
    validation = validation_module()
    dataset = tmp_path / "dataset"
    for split in ("train-a", "train-b", "val-a", "val-b"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        """path: dataset
train: [images/train-a, images/train-b]
val: [images/val-a, images/val-b]
names: [defect]
""",
        encoding="utf-8",
    )

    assert validation.validate_dataset_config(config) is None


def test_vendored_runtime_uses_dataset_yaml_directory(
    monkeypatch, tmp_path: Path
) -> None:
    from ultralytics.data import utils as data_utils

    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(data_utils, "DATASETS_DIR", config.parent.resolve())
    monkeypatch.setattr(data_utils, "check_font", lambda *_args, **_kwargs: None)

    loaded = data_utils.check_det_dataset(str(config.resolve()), autodownload=False)

    assert Path(loaded["path"]) == dataset.resolve()

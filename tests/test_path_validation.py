from pathlib import Path

from path_validation import (
    validate_dataset_config,
    validate_directory_source,
    validate_file_source,
    validate_model_path,
    validate_video_source,
)


def test_validate_model_path_reports_missing_weight(tmp_path: Path) -> None:
    message = validate_model_path(tmp_path / "missing.pt")

    assert message is not None
    assert "模型权重" in message
    assert "本地" in message


def test_validate_model_path_reports_empty_value() -> None:
    message = validate_model_path("")

    assert message is not None
    assert "模型权重" in message


def test_validate_model_path_accepts_existing_file(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.touch()

    assert validate_model_path(model) is None


def test_validate_file_source_accepts_existing_file(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    image.touch()

    assert validate_file_source(image, "图片") is None


def test_validate_file_source_reports_missing_file(tmp_path: Path) -> None:
    message = validate_file_source(tmp_path / "missing.jpg", "图片")

    assert message is not None
    assert "图片" in message


def test_validate_directory_source_accepts_existing_directory(tmp_path: Path) -> None:
    assert validate_directory_source(tmp_path, "图片目录") is None


def test_validate_directory_source_reports_missing_directory(tmp_path: Path) -> None:
    message = validate_directory_source(tmp_path / "missing", "图片目录")

    assert message is not None
    assert "图片目录" in message


def test_validate_video_source_accepts_camera_index() -> None:
    assert validate_video_source("0") is None


def test_validate_video_source_reports_missing_file(tmp_path: Path) -> None:
    message = validate_video_source(tmp_path / "missing.mp4")

    assert message is not None
    assert "视频源" in message


def test_validate_dataset_config_reports_missing_file(tmp_path: Path) -> None:
    message = validate_dataset_config(tmp_path / "missing.yaml")

    assert message is not None
    assert "数据集配置" in message


def test_validate_dataset_config_reports_malformed_yaml(tmp_path: Path) -> None:
    config = tmp_path / "dataset.yaml"
    config.write_text("path: [\n", encoding="utf-8")

    message = validate_dataset_config(config)

    assert message is not None
    assert "无法读取数据集配置" in message


def test_validate_dataset_config_reports_missing_split(tmp_path: Path) -> None:
    config = tmp_path / "dataset.yaml"
    config.write_text("path: dataset\ntrain: images/train\n", encoding="utf-8")

    message = validate_dataset_config(config)

    assert message is not None
    assert "val" in message


def test_validate_dataset_config_reports_missing_images(tmp_path: Path) -> None:
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    message = validate_dataset_config(config)

    assert message is not None
    assert "图像路径" in message
    assert "train" in message


def test_validate_dataset_config_reports_missing_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "images" / "val").mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    message = validate_dataset_config(config)

    assert message is not None
    assert "标签目录" in message
    assert "train" in message


def test_validate_dataset_config_accepts_complete_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True)
        (dataset_root / "labels" / split).mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    assert validate_dataset_config(config) is None

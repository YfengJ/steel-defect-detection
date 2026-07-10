"""Dependency-light validation for local files selected in the desktop GUI."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import yaml

PathLike: TypeAlias = str | Path


def _path(value: PathLike) -> Path:
    return Path(value).expanduser()


def _resolve(base: Path, value: object) -> Path:
    path = _path(str(value))
    return path if path.is_absolute() else base / path


def _label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts)


def validate_model_path(value: PathLike) -> str | None:
    path = _path(value)
    if not path.is_file():
        return f"找不到模型权重：{path}。请先在本地准备 .pt 权重文件。"
    return None


def validate_file_source(value: PathLike, label: str = "输入文件") -> str | None:
    path = _path(value)
    if not path.is_file():
        return f"找不到{label}：{path}。请重新选择本地文件。"
    return None


def validate_directory_source(value: PathLike, label: str = "输入目录") -> str | None:
    path = _path(value)
    if not path.is_dir():
        return f"找不到{label}：{path}。请重新选择本地目录。"
    return None


def validate_video_source(value: PathLike) -> str | None:
    text = str(value).strip()
    if text.isdecimal():
        return None
    return validate_file_source(text, "视频源")


def validate_dataset_config(value: PathLike) -> str | None:
    config_path = _path(value)
    if not config_path.is_file():
        return (
            f"找不到数据集配置：{config_path}。"
            "请先准备 dataset.yaml，并参考 docs/dataset.md。"
        )

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return f"无法读取数据集配置：{config_path}（{exc.__class__.__name__}）。"

    if not isinstance(config, dict):
        return f"无法读取数据集配置：{config_path}。YAML 顶层必须是键值映射。"

    root_value = config.get("path", ".")
    dataset_root = _resolve(config_path.parent, root_value)

    for split in ("train", "val"):
        split_value = config.get(split)
        if not isinstance(split_value, str) or not split_value.strip():
            return f"数据集配置缺少有效的 {split} 路径：{config_path}。"

        image_path = _resolve(dataset_root, split_value)
        if not image_path.exists():
            return f"找不到 {split} 图像路径：{image_path}。请检查 dataset.yaml。"

        label_path = _label_path(image_path)
        if label_path is None:
            return (
                f"无法推断 {split} 标签目录：{image_path}。"
                "请使用 images/<split> 与 labels/<split> 的目录结构。"
            )
        if not label_path.exists():
            return f"找不到 {split} 标签目录：{label_path}。请先准备 YOLO 标签。"

    return None

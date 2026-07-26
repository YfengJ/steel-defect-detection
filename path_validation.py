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


def _split_paths(value: object) -> list[str] | None:
    if isinstance(value, str) and value.strip():
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        return value
    return None


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
    if not str(value).strip() or not path.is_dir():
        return f"找不到{label}：{path}。请重新选择本地目录。"
    return None


def validate_prediction_source(value: PathLike) -> str | None:
    path = _path(value)
    if not str(value).strip() or not (path.is_file() or path.is_dir()):
        return f"找不到预测源：{path}。请选择本地图片、视频或图片目录。"
    return None


def validate_video_source(value: PathLike) -> str | None:
    text = str(value).strip()
    if text.isdecimal():
        return None
    return validate_file_source(text, "视频源")


def resolve_dataset_config_path(value: PathLike) -> Path:
    """Resolve a dataset YAML so the CLI and vendored runtime share one base."""
    return _path(value).resolve()


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

    root_value = config.get("path") or "."
    if not isinstance(root_value, str):
        return f"数据集配置中的 path 必须是字符串：{config_path}。"
    if "names" not in config and "nc" not in config:
        return f"数据集配置缺少 names 或 nc 类别信息：{config_path}。"
    dataset_root = _resolve(config_path.parent, root_value)
    for split in ("train", "val"):
        split_paths = _split_paths(config.get(split))
        if split_paths is None:
            return f"数据集配置缺少有效的 {split} 路径：{config_path}。"

        for split_value in split_paths:
            image_path = _resolve(dataset_root, split_value)
            if not image_path.is_dir():
                return f"找不到 {split} 图像路径：{image_path}。请检查 dataset.yaml。"

            label_path = _label_path(image_path)
            if label_path is None:
                return (
                    f"无法推断 {split} 标签目录：{image_path}。"
                    "请使用 images/<split> 与 labels/<split> 的目录结构。"
                )
            if not label_path.is_dir():
                return f"找不到 {split} 标签目录：{label_path}。请先准备 YOLO 标签。"

    return None

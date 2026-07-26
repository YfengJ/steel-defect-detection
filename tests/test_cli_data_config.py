from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path

import train
import val


def make_dataset_config(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: dataset\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )
    return config, dataset


def install_fake_ultralytics(monkeypatch, captured: dict[str, object]) -> None:
    module = types.ModuleType("ultralytics")
    data_module = types.ModuleType("ultralytics.data")
    data_utils = types.ModuleType("ultralytics.data.utils")
    data_utils.DATASETS_DIR = None
    data_module.utils = data_utils
    captured["data_utils"] = data_utils

    class FakeBoxMetrics:
        map = 0.5
        map50 = 0.6
        map75 = 0.4
        maps = [0.5]

    class FakeYOLO:
        names = {0: "defect"}

        def __init__(self, _model_path: str) -> None:
            pass

        def train(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(save_dir="runs/detect/train_result")

        def val(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(box=FakeBoxMetrics())

    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setitem(sys.modules, "ultralytics.data", data_module)
    monkeypatch.setitem(sys.modules, "ultralytics.data.utils", data_utils)


def test_train_passes_absolute_dataset_root_to_ultralytics(
    monkeypatch, tmp_path: Path
) -> None:
    config, dataset = make_dataset_config(tmp_path)
    model = tmp_path / "model.pt"
    model.touch()
    captured: dict[str, object] = {}
    install_fake_ultralytics(monkeypatch, captured)
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: Namespace(
            model=str(model),
            data=str(config),
            epochs=1,
            batch=1,
            imgsz=32,
            device="cpu",
        ),
    )

    assert train.main() == 0
    assert captured["data"] == str(config.resolve())
    assert captured["data_utils"].DATASETS_DIR == dataset.parent.resolve()


def test_val_passes_absolute_dataset_root_to_ultralytics(
    monkeypatch, tmp_path: Path
) -> None:
    config, dataset = make_dataset_config(tmp_path)
    model = tmp_path / "model.pt"
    model.touch()
    captured: dict[str, object] = {}
    install_fake_ultralytics(monkeypatch, captured)
    monkeypatch.setattr(
        val,
        "parse_args",
        lambda: Namespace(model=str(model), data=str(config)),
    )

    assert val.main() == 0
    assert captured["data"] == str(config.resolve())
    assert captured["data_utils"].DATASETS_DIR == dataset.parent.resolve()

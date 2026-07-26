from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def load_translate(monkeypatch):
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda values: values
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)
    sys.modules.pop("translate", None)
    return importlib.import_module("translate")


def test_convert_xml_accepts_missing_difficult_field(
    tmp_path: Path, monkeypatch
) -> None:
    translate = load_translate(monkeypatch)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "sample.xml").write_text(
        """<annotation>
  <size><width>100</width><height>50</height></size>
  <object>
    <name>scratches</name>
    <bndbox><xmin>10</xmin><xmax>30</xmax><ymin>5</ymin><ymax>15</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )

    translate.convert_xml_to_txt("sample", tmp_path)

    label = (tmp_path / "labels" / "train" / "sample.txt").read_text(encoding="utf-8")
    assert label.startswith("5 ")
    assert label.endswith("\n")


def test_convert_xml_rejects_out_of_bounds_box(tmp_path: Path, monkeypatch) -> None:
    translate = load_translate(monkeypatch)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "sample.xml").write_text(
        """<annotation>
  <size><width>100</width><height>50</height></size>
  <object>
    <name>scratches</name>
    <bndbox><xmin>10</xmin><xmax>130</xmax><ymin>5</ymin><ymax>15</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )

    try:
        translate.convert_xml_to_txt("sample", tmp_path)
    except ValueError as exc:
        assert "边界框" in str(exc)
    else:
        raise AssertionError("invalid bounding box was accepted")

    assert not (tmp_path / "labels" / "train" / "sample.txt").exists()

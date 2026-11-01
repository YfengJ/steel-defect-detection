from pathlib import Path

import pytest

from translate import convert_xml_to_txt, main


def _write_annotation(root: Path, image_id: str, body: str) -> None:
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / f"{image_id}.xml").write_text(body, encoding="utf-8")


def _voc(objects: str, width: int = 200, height: int = 100) -> str:
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height></size>"
        f"{objects}</annotation>"
    )


BOX = "<bndbox><xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>30</ymax></bndbox>"


def test_missing_difficult_defaults_to_not_difficult(tmp_path: Path) -> None:
    _write_annotation(tmp_path, "scratches_1", _voc(f"<object><name>scratches</name>{BOX}</object>"))

    assert convert_xml_to_txt("scratches_1", tmp_path)

    line = (tmp_path / "labels" / "train" / "scratches_1.txt").read_text(encoding="utf-8").split()
    assert line[0] == "5"
    assert [float(value) for value in line[1:]] == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_difficult_and_unknown_objects_are_skipped(tmp_path: Path) -> None:
    objects = (
        f"<object><name>patches</name><difficult>1</difficult>{BOX}</object>"
        f"<object><name>unknown</name><difficult>0</difficult>{BOX}</object>"
    )
    _write_annotation(tmp_path, "patches_1", _voc(objects))

    assert convert_xml_to_txt("patches_1", tmp_path)
    assert (tmp_path / "labels" / "train" / "patches_1.txt").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("body", ["<annotation>", _voc("", width=0)])
def test_invalid_annotation_is_skipped_without_label(tmp_path: Path, body: str) -> None:
    _write_annotation(tmp_path, "broken", body)

    assert not convert_xml_to_txt("broken", tmp_path)
    assert not (tmp_path / "labels" / "train" / "broken.txt").exists()


def test_main_converts_every_existing_split(tmp_path: Path) -> None:
    for split, image_id in (("train", "crazing_1"), ("val", "crazing_2")):
        image_dir = tmp_path / "images" / split
        image_dir.mkdir(parents=True)
        (image_dir / f"{image_id}.JPG").write_bytes(b"")
        _write_annotation(tmp_path, image_id, _voc(f"<object><name>crazing</name>{BOX}</object>"))

    assert main(tmp_path) == 0
    assert (tmp_path / "labels" / "train" / "crazing_1.txt").is_file()
    assert (tmp_path / "labels" / "val" / "crazing_2.txt").is_file()


def test_main_reports_missing_dataset(tmp_path: Path) -> None:
    assert main(tmp_path / "missing") == 1

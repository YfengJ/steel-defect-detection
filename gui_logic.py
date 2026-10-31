"""Display-free helpers behind the desktop GUI.

Everything here works on plain files and strings so it can be unit tested
without Tk, a model, or a display.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CLASS_MAP = {0: "龟裂", 1: "夹杂", 2: "斑块", 3: "麻点", 4: "氧化铁皮", 5: "划痕"}
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
METRICS_PREFIX = "METRICS_JSON "
DEVICE_AUTO = "自动"
# tqdm 进度条刷新行，例如 " 45%|████▌     | 9/20 [00:03<00:04]"
_PROGRESS_RE = re.compile(r"\d{1,3}%\|")


@dataclass(frozen=True)
class Detection:
    cls_id: int
    name: str
    x: float
    y: float
    w: float
    h: float
    conf: float | None

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class ImageResult:
    name: str
    detections: list[Detection]

    @property
    def top_class(self) -> str:
        if not self.detections:
            return "—"
        return Counter(d.name for d in self.detections).most_common(1)[0][0]

    @property
    def max_conf(self) -> float | None:
        confs = [d.conf for d in self.detections if d.conf is not None]
        return max(confs) if confs else None


@dataclass
class BatchSummary:
    image_count: int = 0
    images: list[ImageResult] = field(default_factory=list)

    @property
    def defect_images(self) -> int:
        return sum(1 for image in self.images if image.detections)

    @property
    def detections(self) -> list[Detection]:
        return [d for image in self.images for d in image.detections]

    @property
    def total_defects(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(d.name for d in self.detections).most_common())

    @property
    def confidences(self) -> list[float]:
        return [d.conf for d in self.detections if d.conf is not None]

    @property
    def mean_conf(self) -> float | None:
        confs = self.confidences
        return sum(confs) / len(confs) if confs else None


def class_name(cls_id: int, class_map: dict[int, str] = CLASS_MAP) -> str:
    return class_map.get(cls_id, f"类别 {cls_id}")


def parse_label_file(path: Path, class_map: dict[int, str] = CLASS_MAP) -> list[Detection]:
    """Parse a YOLO txt file written with ``save_conf`` (cls x y w h [conf])."""
    detections = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = (float(value) for value in parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else None
        except ValueError:
            continue
        detections.append(Detection(cls_id, class_name(cls_id, class_map), x, y, w, h, conf))
    return detections


def find_result_image(save_dir: Path, source: Path) -> Path | None:
    """Ultralytics keeps the source filename, including suffixes such as .bmp."""
    save_dir, source = Path(save_dir), Path(source)
    direct = save_dir / source.name
    if direct.is_file():
        return direct
    return next((p for p in sorted(save_dir.glob(f"{source.stem}.*")) if p.is_file()), None)


def list_images(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def summarize_batch(label_dir: Path, source_dir: Path | None = None,
                    class_map: dict[int, str] = CLASS_MAP) -> BatchSummary:
    """Combine the source image list with the label files of one batch run.

    Images without detections have no label file, so the source folder is the
    only reliable count of processed images.
    """
    label_dir = Path(label_dir)
    labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
    sources = list_images(source_dir) if source_dir is not None else []

    images = []
    seen = set()
    for image in sources:
        seen.add(image.stem)
        label = labels.get(image.stem)
        images.append(ImageResult(image.name, parse_label_file(label, class_map) if label else []))
    for stem in sorted(set(labels) - seen):
        images.append(ImageResult(stem, parse_label_file(labels[stem], class_map)))
    return BatchSummary(image_count=len(images), images=images)


def batch_verdict(summary: BatchSummary) -> tuple[str, str]:
    """Return a short verdict and a bootstyle-like level for the batch."""
    if summary.image_count == 0:
        return "目录中没有可识别的图片", "warning"
    if summary.total_defects == 0:
        return "未检出缺陷", "success"
    mean = summary.mean_conf
    if mean is None:
        return "已检出缺陷，置信度未知", "info"
    if mean >= 0.8:
        return "置信度高，结果较可靠", "success"
    if mean < 0.5:
        return "置信度偏低，建议人工复核", "warning"
    return "置信度中等，建议抽样复核", "info"


def parse_metrics_line(line: str) -> dict | None:
    """Return the JSON payload printed by ``val.py`` or None for other lines."""
    if not line.startswith(METRICS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(METRICS_PREFIX):])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def read_training_progress(results_csv: Path) -> list[dict[str, float]]:
    """Read Ultralytics ``results.csv`` (headers are space padded)."""
    path = Path(results_csv)
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row = {}
                for key, value in raw.items():
                    if key is None or value is None:
                        continue
                    try:
                        row[key.strip()] = float(value)
                    except ValueError:
                        continue
                if "epoch" not in row:
                    continue
                # exist_ok 复用目录时新一轮会追加到旧文件末尾；epoch 回退说明换了一轮，只保留最新一轮
                if rows and row["epoch"] <= rows[-1]["epoch"]:
                    rows = []
                rows.append(row)
    except (OSError, csv.Error):
        return []
    return rows


def parse_confidence(value: object) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return conf if 0.0 < conf <= 1.0 else None


def parse_positive_int(value: object, allow_zero: bool = False) -> int | None:
    try:
        number = int(str(value).strip())
    except ValueError:
        return None
    return number if number > 0 or (allow_zero and number == 0) else None


def device_arg(choice: object) -> str | None:
    """Map the GUI device choice to a CLI ``--device`` value (None = auto)."""
    text = str(choice or "").strip()
    return None if not text or text == DEVICE_AUTO else text


def absolute_path(value: object) -> str:
    """Resolve user input before handing it to a subprocess running in the repo root."""
    return str(Path(str(value).strip()).expanduser().resolve())


def is_progress_noise(line: str) -> bool:
    """Intermediate tqdm refreshes flood the log; keep only completed bars."""
    return bool(_PROGRESS_RE.search(line)) and "100%|" not in line


def format_conf(conf: float | None) -> str:
    return "—" if conf is None else f"{conf:.1%}"

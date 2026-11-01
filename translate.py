import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from cli_support import use_utf8_output

# 定义数据集类别
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def convert_box(size, box):
    """转换坐标为 YOLO xywh 格式"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def convert_xml_to_txt(image_id, root_dir, split="train"):
    """把单个 VOC XML 转换为 labels/<split>/<image_id>.txt，成功返回 True。"""
    in_file = root_dir / 'annotations' / f'{image_id}.xml'
    out_file_path = root_dir / 'labels' / split / f'{image_id}.txt'

    try:
        root = ET.parse(in_file).getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w <= 0 or h <= 0:
            raise ValueError(f"图像尺寸无效: {w}x{h}")

        lines = []
        for obj in root.iter('object'):
            # 许多 VOC 标注没有 difficult 字段，缺省视为 0
            difficult = obj.findtext('difficult', default='0').strip() or '0'
            cls = obj.findtext('name', default='').strip()
            if cls not in CLASSES or int(difficult) == 1:
                continue

            cls_id = CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))

            bb = convert_box((w, h), b)
            lines.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    except FileNotFoundError:
        print(f"Warning: Missing file {in_file}")
        return False
    except (ET.ParseError, AttributeError, TypeError, ValueError) as exc:
        # 单个损坏的标注不应中断整个数据集的转换
        print(f"Warning: Skipping invalid annotation {in_file}: {exc}")
        return False

    # 解析成功后再写文件，避免损坏的 XML 留下空标签误导训练
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    out_file_path.write_text(''.join(lines), encoding='utf-8')
    return True


def main(root_dir=Path('datasets/NEU-DET')):
    if not root_dir.exists():
        print(f"错误: 找不到目录 {root_dir}，请确保在正确的位置运行脚本。")
        return 1

    # 转换 images/ 下已存在的每个划分（train/val/test）
    split_dirs = [(split, root_dir / 'images' / split) for split in SPLITS]
    split_dirs = [(split, path) for split, path in split_dirs if path.is_dir()]
    if not split_dirs:
        print(f"错误: 图片目录不存在 {root_dir / 'images' / 'train'}")
        return 1

    failures = 0
    for split, img_dir in split_dirs:
        image_ids = sorted({f.stem for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES})
        print(f"[{split}] 找到 {len(image_ids)} 个样本，开始转换...")
        for image_id in tqdm(image_ids):
            if not convert_xml_to_txt(image_id, root_dir, split):
                failures += 1

    if failures:
        print(f"⚠️ 转换完成，但有 {failures} 个样本未能转换，请检查上方警告。")
    else:
        print("✅ 转换完成！")
    return 0


if __name__ == "__main__":
    use_utf8_output()
    sys.exit(main())

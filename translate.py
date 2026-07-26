import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

# 定义数据集类别
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def convert_box(size, box):
    """转换坐标为 YOLO xywh 格式"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def convert_xml_to_txt(image_id, root_dir):
    in_file = root_dir / "annotations" / f"{image_id}.xml"
    out_file_path = root_dir / "labels" / "train" / f"{image_id}.txt"

    try:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"标注缺少图像尺寸: {in_file}")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w <= 0 or h <= 0:
            raise ValueError(f"图像尺寸必须为正数: {in_file}")

        labels = []
        for obj in root.iter("object"):
            difficult_node = obj.find("difficult")
            difficult = (
                difficult_node.text
                if difficult_node is not None and difficult_node.text
                else "0"
            )
            name_node = obj.find("name")
            cls = name_node.text if name_node is not None else None
            if cls not in CLASSES or int(difficult) == 1:
                continue

            xmlbox = obj.find("bndbox")
            if xmlbox is None:
                raise ValueError(f"对象缺少边界框: {in_file}")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            if not (0 <= b[0] < b[1] <= w and 0 <= b[2] < b[3] <= h):
                raise ValueError(f"边界框超出图像范围: {in_file} ({b})")

            cls_id = CLASSES.index(cls)
            bb = convert_box((w, h), b)
            labels.append(
                str(cls_id) + " " + " ".join(str(value) for value in bb) + "\n"
            )

        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        out_file_path.write_text("".join(labels), encoding="utf-8")
    except FileNotFoundError:
        print(f"Warning: Missing file {in_file}")


if __name__ == "__main__":
    # 配置你的数据集根目录
    ROOT_DIR = Path("datasets/NEU-DET")

    if not ROOT_DIR.exists():
        print(f"错误: 找不到目录 {ROOT_DIR}，请确保在正确的位置运行脚本。")
        exit(1)

    # 假设 images/train 下存放图片
    img_dir = ROOT_DIR / "images" / "train"
    if not img_dir.exists():
        print(f"错误: 图片目录不存在 {img_dir}")
        exit(1)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    image_ids = sorted(
        {f.stem for f in img_dir.iterdir() if f.suffix.lower() in image_suffixes}
    )

    print(f"找到 {len(image_ids)} 个样本，开始转换...")

    for image_id in tqdm(image_ids):
        convert_xml_to_txt(image_id, ROOT_DIR)

    print("✅ 转换完成！")

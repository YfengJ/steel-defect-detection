# 基于 YOLOv8 的钢铁表面缺陷检测

一个面向训练、验证、推理和 GUI 演示的 YOLOv8 钢铁表面缺陷检测开源项目。

[English](README.md) | [简体中文](README.zh-CN.md)

![Python](https://img.shields.io/badge/Python-3.10%20recommended-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![GUI](https://img.shields.io/badge/GUI-ttkbootstrap-green)
![Dataset](https://img.shields.io/badge/Dataset-NEU--DET-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)

---

## 项目简介

本项目是一个基于 **YOLOv8** 的 **钢铁表面缺陷检测开源项目**，面向 NEU-DET 等钢铁表面缺陷数据集，覆盖模型训练、验证、图片/视频推理和 GUI 演示等基础流程。

项目适合学生课程设计、深度学习和目标检测初学者，以及想了解工业视觉缺陷检测流程的学习者。当前目标不是工业生产级闭环系统，而是提供一个真实、可运行、可持续维护的 YOLOv8 学习和实验仓库。

> 仓库不包含 NEU-DET 数据集、训练结果、`.pt/.pth` 权重或其他大文件。请按照 [docs/dataset.md](docs/dataset.md) 自行准备数据集和模型权重。

## 核心功能

| 功能 | 描述 |
| --- | --- |
| 单图检测 | 加载单张图片并可视化缺陷检测框。 |
| 批量检测 | 处理图片文件夹并生成统计报告。 |
| 视频检测 | 对视频文件或摄像头画面进行缺陷检测。 |
| 模型训练 | 通过命令行或 GUI 训练 YOLOv8 模型。 |
| 模型验证 | 评估训练权重并查看 mAP 等指标。 |
| GUI 演示 | 使用 `ttkbootstrap` 桌面界面完成常见流程。 |

## 项目文档

- [macOS / Apple Silicon 运行说明](docs/macos.md)
- [数据集准备说明](docs/dataset.md)
- [常见问题排查](docs/troubleshooting.md)
- [Roadmap](ROADMAP.md)
- [Support](SUPPORT.md)
- [Security Policy](SECURITY.md)

## 界面展示

### 单图检测

![单图检测](screenshots/predict.png)

### 批量检测

![批量检测](screenshots/batch.png)

### 视频检测

![视频检测](screenshots/video.png)

### 模型训练

![模型训练](screenshots/train.png)

### 模型验证

![模型验证](screenshots/val.png)

## 项目结构

```text
steel-defect-detection/
├── ui.py                  # ttkbootstrap 桌面 GUI
├── train.py               # YOLOv8 训练入口
├── predict.py             # 图片和批量预测入口
├── val.py                 # 验证入口
├── video_predict.py       # 视频推理模块
├── translate.py           # VOC XML 转 YOLO TXT 工具
├── dataset.yaml           # 数据集配置
├── requirements.txt       # Python 依赖
├── docs/                  # 项目运行和维护文档
├── .github/               # CI、Dependabot 和 Issue 模板
├── datasets/              # 本地数据集目录，不提交
├── runs/                  # 本地训练/推理输出，不提交
└── weights/ 或 *.pt        # 本地模型权重，不提交
```

## 检测类别

默认配置遵循 NEU-DET 的 6 类钢铁表面缺陷：

| ID | 类别 | 含义 |
| --- | --- | --- |
| 0 | `crazing` | 龟裂 |
| 1 | `inclusion` | 夹杂 |
| 2 | `patches` | 斑块 |
| 3 | `pitted_surface` | 麻点 |
| 4 | `rolled-in_scale` | 氧化铁皮压入 |
| 5 | `scratches` | 划痕 |

## 快速开始

### 1. 环境要求

- 推荐 Python 3.10
- CPU、Apple Silicon MPS 或 NVIDIA CUDA
- Windows、Linux 或 macOS

macOS 用户建议先阅读 [docs/macos.md](docs/macos.md)。

### 2. 安装依赖

```bash
git clone https://github.com/YfengJ/steel-defect-detection.git
cd steel-defect-detection

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

请在本地自行准备：

- 数据集：参考 [docs/dataset.md](docs/dataset.md)
- 模型权重：例如官方 `yolov8n.pt`，或自己训练得到的 `best.pt`

### 3. 启动 GUI

```bash
python ui.py
```

GUI 包含图片预测、批量预测、视频推理、训练和验证等功能页。

## 命令行使用

### 训练

```bash
# CPU
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device cpu

# Apple Silicon MPS
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 8 --device mps

# NVIDIA CUDA
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device cuda
```

### 验证

```bash
python val.py --model runs/detect/train_result/weights/best.pt --data dataset.yaml
```

### 预测

```bash
python predict.py --model runs/detect/train_result/weights/best.pt --source path/to/image.jpg
```

## 数据集准备

本项目可使用 [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/) 或相同类别格式的数据集。

推荐本地结构：

```text
datasets/NEU-DET/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
```

YOLO 标注格式为：

```text
class_id x_center y_center width height
```

所有坐标都需要归一化到 `0.0` 到 `1.0`。

如果你的标注是 VOC XML 格式，请放到 `datasets/NEU-DET/annotations/` 后运行：

```bash
python translate.py
```

## 模型权重

模型权重不会提交到 git。大文件请保存在本地、云存储或 GitHub Releases。

| 权重 | 适用场景 |
| --- | --- |
| `yolov8n.pt` | CPU 或小实验的快速基线 |
| `yolov8s.pt` | 速度和精度更平衡 |
| `yolov8m.pt` | 硬件允许时的中等规模实验 |
| `best.pt` | 你自己训练得到的缺陷检测模型 |

## 依赖更新

仓库已启用 Dependabot，用于 Python 依赖和 GitHub Actions 更新。由于 PyTorch、OpenCV、NumPy 和 Ultralytics 的跨平台兼容性比较敏感，大版本依赖更新会保守处理。

## Release Notes

v0.1.1 的更新内容、已知限制和后续计划请查看 [RELEASE_NOTES.md](RELEASE_NOTES.md)。

## License

本项目基于 [AGPL-3.0 License](LICENSE) 开源。

YOLOv8 来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，遵循其上游开源协议。

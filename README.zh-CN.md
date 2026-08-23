# 基于 YOLOv8 的钢铁表面缺陷检测

一个面向训练、验证、推理和 GUI 演示的 YOLOv8 钢铁表面缺陷检测开源项目。

[English](README.md) | [简体中文](README.zh-CN.md)

![Python](https://img.shields.io/badge/Python-3.10%20recommended-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![GUI](https://img.shields.io/badge/GUI-ttkbootstrap-green)
![Dataset](https://img.shields.io/badge/Dataset-NEU--DET-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)
[![CI](https://github.com/YfengJ/steel-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YfengJ/steel-defect-detection/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/YfengJ/steel-defect-detection)](https://github.com/YfengJ/steel-defect-detection/releases)

---

## 项目简介

本项目是一个基于 **YOLOv8** 的 **钢铁表面缺陷检测开源项目**，面向 NEU-DET
等钢铁表面缺陷数据集，覆盖模型训练、验证、图片/视频推理和 GUI 演示等基础流程。

项目适合学生课程设计、深度学习和目标检测初学者，以及想了解工业视觉缺陷检测流程的学习者。
当前目标不是工业生产级闭环系统，而是提供一个真实、可运行、可持续维护的 YOLOv8 学习和实验仓库。

> 仓库不包含 NEU-DET 数据集、训练结果、`.pt/.pth` 权重或其他大文件。
> 请按照 [docs/dataset.md](docs/dataset.md) 自行准备数据集和模型权重。

![钢铁表面缺陷检测结果](screenshots/predict.png)

## 从这里开始

```bash
python -m pip install -r requirements.txt
python predict.py --model /path/to/trusted-best.pt --source /path/to/image.jpg --device cpu
```

渲染结果保存在 `runs/detect/`。请准备来源可信的本地权重和有权使用的图片；
仓库不会捆绑这两类文件。CPU、Apple Silicon MPS 和 NVIDIA CUDA 的完整示例请参考
[示例推理指南](docs/sample_inference.md)。

## 当前状态

- 当前版本：`v0.1.3`。
- CI 使用 Python 3.10 执行 Ruff 检查、单元测试、CLI smoke tests、合成数据 CPU
  训练/验证集成测试、仓库卫生检查和 Python 编译。
- 相同的健康检查会在每周一自动运行，也可以在 GitHub Actions 页面手动触发。
- 已有 macOS、数据集、模型卡、示例推理、故障排查、支持、安全和路线图文档。
- 数据集和模型权重不包含在仓库中，需要用户在本地自行准备。
- GUI 会在启动任务前检查模型、数据集、图片、目录和视频路径。
- Apple Silicon MPS 已使用 PyTorch 2.5.1 完成张量运算和一次临时图片推理实测。
- v0.2.0 将重点完善可复现训练参数、更广泛的设备检查和实验输出管理。
- 新一轮 50 epochs YOLOv8s/NEU-DET 复现实验已如实记录在
  [实验日志](docs/experiments/yolov8s-neu-det-baseline.md) 中。由于设备迁移时遗失了原始
  checkpoint 和结果文件，仓库不会把回忆中的历史指标当作已验证结果发布。

## 已复现基线

v0.1.3 在 Apple Silicon 上使用本地 1,440 / 360 图片划分训练 YOLOv8s 50 轮。
随后使用 CPU 对本地 `best.pt` 独立验证，结果如下：

| Precision | Recall | mAP50 | mAP50-95 | mAP75 |
| ---: | ---: | ---: | ---: | ---: |
| 0.719 | 0.721 | 0.7637 | 0.4455 | 0.4530 |

这些数值只代表一个公开数据集风格划分，不代表工厂现场性能。`crazing` 是最弱类别，
mAP50-95 为 0.1861。每类结果、环境、限制和本地权重校验值请查看
[实验日志](docs/experiments/yolov8s-neu-det-baseline.md) 与
[模型卡](docs/experiments/yolov8s-neu-det-v0.1.3-model-card.md)。数据集和权重均未提交或发布。

| 缺陷类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `crazing` | 0.625 | 0.383 | 0.472 | 0.1861 |
| `inclusion` | 0.716 | 0.767 | 0.816 | 0.4821 |
| `patches` | 0.741 | 0.891 | 0.904 | 0.6045 |
| `pitted_surface` | 0.825 | 0.788 | 0.847 | 0.5253 |
| `rolled-in_scale` | 0.611 | 0.567 | 0.630 | 0.2883 |
| `scratches` | 0.795 | 0.928 | 0.913 | 0.5864 |

**复现实验配置：** YOLOv8s、50 个 epoch、640 px 输入尺寸、batch size 16，
使用 Apple M5 的 MPS 完成训练，随后在 CPU 上对 360 张验证图像中的 822 个
标注缺陷实例进行独立验证。

## 核心功能

| 功能 | 描述 |
| --- | --- |
| 单图检测 | 加载单张图片并可视化缺陷检测框。 |
| 批量检测 | 处理图片文件夹并生成统计报告。 |
| 视频检测 | 对视频文件或摄像头画面进行缺陷检测。 |
| 模型训练 | 通过命令行或 GUI 训练 YOLOv8 模型。 |
| 模型验证 | 评估训练权重并查看 mAP 等指标。 |
| GUI 演示 | 使用 `ttkbootstrap` 桌面界面完成常见流程。 |
| 设备选择 | CLI 推理可选择 CPU、Apple Silicon MPS 或 NVIDIA CUDA。 |

## 项目文档

- [macOS / Apple Silicon 运行说明](docs/macos.md)
- [使用本地文件完成示例推理](docs/sample_inference.md)
- [数据集准备说明](docs/dataset.md)
- [模型卡模板](docs/model_card.md)
- [YOLOv8s 基线实验日志](docs/experiments/yolov8s-neu-det-baseline.md)
- [YOLOv8s v0.1.3 模型卡](docs/experiments/yolov8s-neu-det-v0.1.3-model-card.md)
- [常见问题排查](docs/troubleshooting.md)
- [Roadmap](ROADMAP.md)
- [贡献指南](CONTRIBUTING.md)
- [社区行为准则](CODE_OF_CONDUCT.md)
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
├── path_validation.py     # 不依赖深度学习运行时的 GUI 路径检查
├── dataset.yaml           # 数据集配置
├── requirements.txt       # Python 依赖
├── requirements-dev.txt   # 轻量代码检查和测试依赖
├── docs/                  # 项目运行和维护文档
├── .github/               # CI、Dependabot 和 Issue 模板
├── datasets/              # 本地数据集目录，不提交
├── runs/                  # 本地训练/推理输出，不提交
└── weights/ 或 *.pt        # 本地模型权重，不提交
```

## 维护范围与上游关系

为了保留原项目的可复现运行环境，本仓库内置了 **Ultralytics 8.0.182**。
其中 `ultralytics/`、大部分通用 `docs/` 和 `examples/` 内容来自采用 AGPL-3.0
协议的 [Ultralytics 上游项目](https://github.com/ultralytics/ultralytics)。

本项目重点维护：

- `train.py`、`val.py`、`predict.py` 和 `video_predict.py`；
- `ui.py` 中基于 `ttkbootstrap` 的桌面工作流；
- 数据集转换与路径检查；
- CPU、CUDA 和 Apple Silicon MPS 环境兼容；
- 项目测试、CI、版本发布文档和贡献者支持。

对内置 YOLO 核心的修改会保持保守。未来切换到固定版本的外部依赖属于兼容性工程，
不会被描述为本项目原创的模型架构工作。

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

如果想先用自己的本地图片和可信权重完成一次不依赖数据集的运行，请参考
[docs/sample_inference.md](docs/sample_inference.md)。

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
python predict.py \
  --model runs/detect/train_result/weights/best.pt \
  --source path/to/image.jpg \
  --device cpu
```

## 数据集准备

本项目可使用 [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)
或相同类别格式的数据集。

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

只加载来源可信的模型权重。PyTorch checkpoint 可能包含序列化的 Python 对象。

| 权重 | 适用场景 |
| --- | --- |
| `yolov8n.pt` | CPU 或小实验的快速基线 |
| `yolov8s.pt` | 速度和精度更平衡 |
| `yolov8m.pt` | 硬件允许时的中等规模实验 |
| `best.pt` | 你自己训练得到的缺陷检测模型 |

## 依赖更新

仓库已启用 Dependabot，用于 Python 依赖和 GitHub Actions 更新。由于 PyTorch、OpenCV、
NumPy 和 Ultralytics 的跨平台兼容性比较敏感，大版本依赖更新会保守处理。本仓库内置
Ultralytics `8.0.182` 源码；由于更高版本的 `torch.load` 默认行为与该版本权重加载器不兼容，
PyTorch 当前限制在 `2.6` 以下。
NumPy 当前限制在 `2.4` 以下，因为内置运行时仍会调用 `numpy.trapz`。

## Release Notes

v0.1.3 的更新内容、已知限制和后续计划请查看 [RELEASE_NOTES.md](RELEASE_NOTES.md)。

## 社区参与

欢迎提交真实的错误报告、复现记录和范围清晰的 Pull Request。如果这个项目对你的学习
或研究有帮助，也可以在
[复现报告 Discussion](https://github.com/YfengJ/steel-defect-detection/discussions/57)
分享经过测试的环境和结果，并通过 Star 支持项目继续维护。请不要使用互刷或其他人为制造的 Star。

## License

本项目基于 [AGPL-3.0 License](LICENSE) 开源。

YOLOv8 来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，遵循其上游开源协议。

<p align="center">
  <h1 align="center">🛡️ 基于 YOLOv8 的智能钢铁表面缺陷检测系统</h1>
  <p align="center">
    <b>Steel Surface Defect Detection System Based on YOLOv8</b>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=yolo" alt="YOLOv8">
    <img src="https://img.shields.io/badge/GUI-ttkbootstrap-green" alt="GUI">
    <img src="https://img.shields.io/badge/Dataset-NEU--DET-orange" alt="Dataset">
    <img src="https://img.shields.io/badge/License-AGPL--3.0-red" alt="License">
  </p>
</p>

---

## 📖 项目简介

本项目是一个基于 **YOLOv8** 深度学习目标检测算法的 **钢铁表面缺陷智能检测系统**，针对工业生产中钢铁表面质量检测需求，实现了从模型训练、验证到推理部署的全流程解决方案。

系统提供了一套 **图形化操作界面（GUI）**，支持单图检测、批量检测、实时视频流检测等多种检测模式，并能自动生成包含缺陷分类统计、置信度分布等信息的可视化分析报告。

### ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🖼️ **单图检测** | 上传单张图片，快速检测并可视化标注缺陷区域 |
| 📂 **批量检测** | 一键处理整个文件夹，自动生成分析报告与统计图表 |
| 📹 **视频流检测** | 支持视频文件与摄像头的实时缺陷检测 |
| ⚙️ **模型训练** | 在 GUI 中直接配置参数并启动模型训练流程 |
| 📊 **模型验证** | 加载训练好的模型进行数据集验证，查看 mAP 等指标 |
| 📈 **可视化报告** | 自动统计缺陷类别占比（饼图）和置信度分布（直方图） |

---

## 🏗️ 系统架构

```
项目根目录/
├── ui.py                  # 🖥️ GUI 主程序（ttkbootstrap 界面）
├── train.py               # ⚙️ 模型训练脚本
├── predict.py             # 🔮 推理预测脚本（单图/批量）
├── val.py                 # 📊 模型验证脚本
├── video_predict.py       # 📹 视频推理脚本
├── translate.py           # 🔄 VOC XML → YOLO TXT 标注格式转换工具
├── dataset.yaml           # 📋 数据集配置文件
├── requirements.txt       # 📦 项目依赖
├── datasets/
│   └── NEU-DET/           # 🗂️ NEU 钢铁缺陷数据集
│       ├── images/        #    原始图片 (train/val/test)
│       ├── labels/        #    YOLO 格式标注
│       └── annotations/   #    VOC XML 原始标注
├── runs/                  # 📁 训练与推理结果输出目录
├── ultralytics/           # 🧠 YOLOv8 核心库（Ultralytics v8.0.182）
└── *.pt                   # 🏋️ 预训练模型权重文件
```

---

## 🎯 检测目标

基于 **NEU-DET（东北大学钢铁表面缺陷数据集）**，系统可识别以下 **6 类** 常见钢铁表面缺陷：

| 编号 | 英文名称 | 中文名称 | 描述 |
|:---:|---------|---------|------|
| 0 | Crazing | 龟裂 | 表面细小裂纹网状分布 |
| 1 | Inclusion | 夹杂 | 金属内部异物夹杂 |
| 2 | Patches | 斑块 | 表面不规则色斑 |
| 3 | Pitted Surface | 麻点 | 表面点蚀坑洞 |
| 4 | Rolled-in Scale | 氧化铁皮压入 | 轧制过程中氧化皮压入 |
| 5 | Scratches | 划痕 | 表面线性划伤 |

---

## 🚀 快速开始

### 1. 环境要求

- **Python** >= 3.8
- **CUDA**（推荐，用于 GPU 加速训练与推理）
- **操作系统**：Windows / Linux / macOS

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/steel-defect-detection.git
cd steel-defect-detection

# 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 额外安装 GUI 所需库
pip install ttkbootstrap
```

### 3. 启动 GUI 系统

```bash
python ui.py
```

启动后将打开图形化操作界面，包含以下功能选项卡：

- **🖼️ 单图检测** — 选择模型和图片，一键分析
- **📂 批量检测** — 选择模型和图片目录，批量处理
- **📹 视频流检测** — 支持视频文件或摄像头实时检测
- **⚙️ 模型训练** — 配置训练参数并启动训练
- **📊 模型验证** — 验证模型性能指标

---

## 📝 使用说明

### 单图检测

1. 在"单图检测"选项卡中，选择 `.pt` 模型文件
2. 点击"选择图片"加载待检测图片
3. 点击"开始分析"，右侧面板将显示检测报告

### 批量检测

1. 在"批量检测"选项卡中，选择模型文件和图片目录
2. 点击"启动批量处理"
3. 系统将自动生成：
   - 📊 详细分析报告（缺陷数量、类别统计、置信度）
   - 🥧 缺陷类别占比饼图
   - 📊 置信度分布直方图

### 视频流检测

1. 在"视频流检测"选项卡中，选择模型
2. 选择视频文件或点击"摄像头"使用实时画面
3. 系统将逐帧进行缺陷检测并实时显示结果

### 模型训练

```bash
# 命令行方式
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16

# 或通过 GUI 界面操作
python ui.py  # 切换到"模型训练"选项卡
```

### 模型验证

```bash
# 命令行方式
python val.py --model runs/detect/train_result/weights/best.pt --data dataset.yaml
```

---

## 🔧 数据集准备

### NEU-DET 数据集

本项目使用 [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)，数据集结构如下：

```
datasets/NEU-DET/
├── images/
│   ├── train/    # 训练集图片
│   ├── val/      # 验证集图片
│   └── test/     # 测试集图片
├── labels/
│   └── train/    # YOLO 格式标注 (cls x y w h)
└── annotations/  # VOC XML 原始标注
```

### 标注格式转换

如果你的数据集为 VOC XML 格式，可使用内置转换工具：

```bash
python translate.py
```

该脚本会自动将 `annotations/` 下的 XML 文件转换为 YOLO 格式的 TXT 标注文件。

---

## 📦 预训练模型

项目提供多种 YOLOv8 预训练权重，可根据硬件条件和精度需求选择：

| 模型 | 参数量 | 适用场景 |
|------|--------|---------|
| `yolov8n.pt` | 3.2M | 轻量级，适合边缘设备 |
| `yolov8s.pt` | 11.2M | 小型模型，速度与精度平衡 |
| `yolov8m.pt` | 25.9M | 中等模型，推荐通用场景 |
| `yolov8x.pt` | 68.2M | 大型模型，追求最高精度 |

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 目标检测模型 | YOLOv8 (Ultralytics v8.0.182) |
| GUI 框架 | ttkbootstrap (基于 Tkinter) |
| 图像处理 | OpenCV, Pillow |
| 数据可视化 | Matplotlib |
| 数据处理 | NumPy |

---

## 📁 项目文件说明

| 文件 | 功能说明 |
|------|---------|
| `ui.py` | GUI 主程序，提供完整的图形化操作界面 |
| `train.py` | 模型训练脚本，支持命令行参数配置 |
| `predict.py` | 推理预测脚本，支持单图与批量推理 |
| `val.py` | 模型验证脚本，输出 mAP 等评估指标 |
| `video_predict.py` | 视频推理处理模块，支持中文标签绘制 |
| `translate.py` | VOC XML 到 YOLO TXT 标注格式转换工具 |
| `dataset.yaml` | 数据集路径与类别配置文件 |

---

## 📄 License

本项目基于 [AGPL-3.0 License](LICENSE) 开源。

YOLOv8 核心库来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，遵循其开源协议。

---

<p align="center">
  <i>⭐ 如果这个项目对你有帮助，请给一个 Star！</i>
</p>

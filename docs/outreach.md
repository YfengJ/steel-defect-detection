# Project Outreach Notes

Use these notes when sharing a release. Keep every claim tied to a public file
or reproducible command; do not claim broad adoption or verified metrics that
the repository does not have.

## English Post

I maintain an open-source YOLOv8 steel surface defect detection project aimed
at students, computer vision beginners, and industrial vision learners. It
includes CLI and desktop GUI workflows for training, validation, image/video
inference, plus tested Apple Silicon MPS guidance. The repository does not
bundle datasets or weights, and the current baseline is being reproduced with
its environment and limitations documented publicly.

Repository: https://github.com/YfengJ/steel-defect-detection

Feedback from people reproducing the workflow on Windows, Linux, CUDA, or
Apple Silicon is especially useful.

## 中文发布文案

我正在维护一个面向学生、计算机视觉初学者和工业视觉学习者的 YOLOv8 钢铁表面缺陷
检测开源项目。项目包含训练、验证、图片/视频推理和桌面 GUI，并提供经过实际检查的
Apple Silicon MPS 运行说明。仓库不捆绑数据集和权重，目前正在公开记录新一轮基线
实验的环境、命令和限制。

项目地址：https://github.com/YfengJ/steel-defect-detection

欢迎在 Windows、Linux、CUDA 或 Apple Silicon 环境复现后提交真实反馈。

## Suggested Article Outline

1. Why steel surface defect detection is a useful learning problem.
2. Dataset licensing and the six NEU-DET classes.
3. Converting VOC XML labels to YOLO format.
4. Creating a Python environment on Apple Silicon.
5. Training YOLOv8s for 50 epochs with MPS.
6. Reporting mAP50, mAP50-95, precision, recall, and per-class behavior.
7. Common path, checkpoint, and dependency failures.
8. Reproducing the GUI and CLI workflows from a clean clone.

Good channels include a detailed GitHub Discussion, Zhihu, CSDN, Juejin,
Dev.to, LinkedIn, and relevant computer-vision communities. Share useful
technical content first; never buy stars or participate in star exchanges.

import sys
import os
import threading
import queue
import subprocess
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import tkinter as tk
from tkinter import filedialog

# 引入现代化UI库
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# --- 兼容性导入 Toast ---
try:
    from ttkbootstrap.toast import ToastNotification
except ImportError:
    try:
        from ttkbootstrap.widgets import ToastNotification
    except ImportError:
        ToastNotification = None

from ttkbootstrap.dialogs import Messagebox

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont

# 引入YOLO
from ultralytics import YOLO

# ==================== 配色常量 ====================
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_card': '#16213e',
    'bg_input': '#0f3460',
    'accent_blue': '#00adb5',
    'accent_green': '#00e676',
    'accent_orange': '#ff9800',
    'accent_red': '#ff5252',
    'accent_purple': '#7c4dff',
    'text_primary': '#e8e8e8',
    'text_secondary': '#a0a0b0',
    'text_muted': '#6c6c80',
    'border': '#2a2a4a',
    'canvas_bg': '#0d1117',
    'log_bg': '#0d1117',
    'success': '#00e676',
    'warning': '#ffab40',
    'error': '#ff5252',
    'header_gradient_start': '#00adb5',
    'header_gradient_end': '#7c4dff',
}


class YOLOv8_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("🛡️ 智能钢铁缺陷检测系统 Pro | YOLOv8")
        self.master.geometry("1400x920")
        self.master.minsize(1200, 800)

        # 变量初始化
        self.setup_variables()

        # 界面布局
        self.create_ui()

        # 日志队列处理
        self.log_queue = queue.Queue()
        self.process_log_queue()

        # 关闭协议
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_toast(self, title, message, bootstyle="success"):
        """兼容性 Toast 显示函数"""
        if ToastNotification:
            ToastNotification(title=title, message=message, bootstyle=bootstyle).show_toast()
        else:
            self.log(f"[{title}] {message}")

    def setup_variables(self):
        """初始化所有StringVar和状态变量"""
        # 训练页变量
        self.train_model = ttk.StringVar()
        self.train_data = ttk.StringVar()
        self.train_epochs = ttk.StringVar(value="50")
        self.train_batch = ttk.StringVar(value="16")
        self.train_imgsz = ttk.StringVar(value="640")

        # 验证页变量
        self.val_model = ttk.StringVar()
        self.val_data = ttk.StringVar()
        # 预测页变量
        self.predict_model = ttk.StringVar()
        self.predict_source = ttk.StringVar()
        self.predict_conf = ttk.StringVar(value="0.25")

        # 批量页变量
        self.batch_model = ttk.StringVar()
        self.batch_data = ttk.StringVar()

        # 视频页变量
        self.video_model = ttk.StringVar()
        self.video_source = ttk.StringVar()

        # 状态控制
        self.video_loop_running = False
        self.original_img = None
        self.current_theme = 'superhero'

        # 配置Matplotlib字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def create_ui(self):
        """构建主界面 - 现代化设计"""
        # 主容器
        main_frame = ttk.Frame(self.master, padding=0)
        main_frame.pack(fill=BOTH, expand=YES)

        # 1. 顶部标题栏
        self._build_header(main_frame)

        # 2. 内容区域（带内边距）
        content_wrapper = ttk.Frame(main_frame, padding=(15, 10, 15, 10))
        content_wrapper.pack(fill=BOTH, expand=YES)

        # 3. 全局垂直分割面板
        self.main_paned = ttk.Panedwindow(content_wrapper, orient=VERTICAL)
        self.main_paned.pack(fill=BOTH, expand=YES)

        # --- 上半部分：功能选项卡 ---
        self.notebook = ttk.Notebook(self.main_paned, bootstyle="info")
        self.main_paned.add(self.notebook, weight=5)

        # 创建各个功能Tab
        self.setup_predict_tab()
        self.setup_batch_tab()
        self.setup_video_tab()
        self.setup_train_tab()
        self.setup_val_tab()

        # --- 下半部分：日志区域 ---
        self.setup_log_area(self.main_paned)

        # 4. 底部状态栏
        self._build_statusbar(main_frame)

    # ------------------ 顶部标题栏 ------------------

    def _build_header(self, parent):
        """构建顶部标题栏"""
        header = ttk.Frame(parent, bootstyle="info")
        header.pack(fill=X)

        header_inner = ttk.Frame(header, padding=(20, 12, 20, 12), bootstyle="info")
        header_inner.pack(fill=X)

        # 左侧：图标 + 标题
        left = ttk.Frame(header_inner, bootstyle="info")
        left.pack(side=LEFT)

        ttk.Label(left, text="🛡️",
                  font=("Segoe UI Emoji", 22),
                  bootstyle="inverse-info").pack(side=LEFT, padx=(0, 10))

        title_frame = ttk.Frame(left, bootstyle="info")
        title_frame.pack(side=LEFT)

        ttk.Label(title_frame, text="智能钢铁缺陷检测系统",
                  font=("Microsoft YaHei", 16, "bold"),
                  bootstyle="inverse-info").pack(anchor=W)

        ttk.Label(title_frame, text="Steel Defect Detection System · YOLOv8",
                  font=("Segoe UI", 9),
                  bootstyle="inverse-info").pack(anchor=W)

        # 右侧：版本信息 + 状态指示
        right = ttk.Frame(header_inner, bootstyle="info")
        right.pack(side=RIGHT)

        # GPU 状态检测
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
        except:
            gpu_available = False
            gpu_name = "CPU"

        status_text = f"{'🟢 GPU: ' + gpu_name if gpu_available else '🟡 CPU 模式'}"
        ttk.Label(right, text=status_text,
                  font=("Segoe UI", 9),
                  bootstyle="inverse-info").pack(side=RIGHT, padx=(15, 0))

        ttk.Label(right, text="v2.5",
                  font=("Segoe UI", 10, "bold"),
                  bootstyle="inverse-info").pack(side=RIGHT)

        ttk.Label(right, text="版本  ",
                  font=("Segoe UI", 9),
                  bootstyle="inverse-info").pack(side=RIGHT)

    # ------------------ 底部状态栏 ------------------

    def _build_statusbar(self, parent):
        """构建底部状态栏"""
        statusbar = ttk.Frame(parent, padding=(15, 5, 15, 5))
        statusbar.pack(fill=X, side=BOTTOM)

        self.status_label = ttk.Label(statusbar, text="✅ 系统就绪，请选择功能开始使用",
                                      font=("Microsoft YaHei", 9),
                                      bootstyle="secondary")
        self.status_label.pack(side=LEFT)

        ttk.Label(statusbar, text="基于 Ultralytics YOLOv8 | NEU-DET 数据集",
                  font=("Segoe UI", 8),
                  bootstyle="secondary").pack(side=RIGHT)

    def update_status(self, text, style="secondary"):
        """更新底部状态栏"""
        self.status_label.config(text=text, bootstyle=style)

    # ------------------ UI 组件构建区域 ------------------

    def setup_log_area(self, parent_paned):
        """底部日志控制台"""
        log_frame = ttk.Labelframe(parent_paned, text="📟 系统运行日志", padding=5, bootstyle="dark")
        parent_paned.add(log_frame, weight=1)

        # 日志文本区域
        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill=BOTH, expand=YES)

        self.log_text = tk.Text(log_inner, height=6, bg=COLORS['log_bg'], fg=COLORS['text_primary'],
                                font=('Cascadia Code', 9), state='disabled', relief='flat',
                                padx=10, pady=5, insertbackground='white', selectbackground='#264f78')
        self.log_text.pack(side=LEFT, fill=BOTH, expand=YES)

        self.log_text.tag_config('INFO', foreground=COLORS['accent_green'])
        self.log_text.tag_config('WARNING', foreground=COLORS['accent_orange'])
        self.log_text.tag_config('ERROR', foreground=COLORS['accent_red'])
        self.log_text.tag_config('SYSTEM', foreground=COLORS['accent_blue'])

        vsb = ttk.Scrollbar(log_inner, command=self.log_text.yview, bootstyle="round")
        vsb.pack(side=RIGHT, fill=Y)
        self.log_text.configure(yscrollcommand=vsb.set)

        # 日志底部工具栏
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=X, pady=(5, 0))
        ttk.Button(log_toolbar, text="🗑️ 清空日志", command=self.clear_log,
                   bootstyle="secondary-outline", padding=(10, 2)).pack(side=RIGHT)

    def clear_log(self):
        """清空日志"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.log("日志已清空", "SYSTEM")

    def setup_predict_tab(self):
        """单图预测界面 - 优化版"""
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  🖼️ 单图检测  ")

        # 顶部工具栏（卡片式）
        tools_card = ttk.Labelframe(tab, text="检测配置", bootstyle="info", padding=10)
        tools_card.pack(fill=X, pady=(0, 8))

        tools_inner = ttk.Frame(tools_card)
        tools_inner.pack(fill=X)

        # 模型选择
        model_group = ttk.Frame(tools_inner)
        model_group.pack(side=LEFT, fill=X, expand=YES)

        ttk.Label(model_group, text="🧠 模型:", font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Entry(model_group, textvariable=self.predict_model, width=35).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(model_group, text="📂", command=lambda: self.browse_file(self.predict_model),
                   bootstyle="info-outline", padding=(8, 2)).pack(side=LEFT, padx=(5, 0))

        ttk.Separator(tools_inner, orient=VERTICAL).pack(side=LEFT, padx=15, fill=Y)

        # 置信度
        conf_group = ttk.Frame(tools_inner)
        conf_group.pack(side=LEFT)
        ttk.Label(conf_group, text="置信度:", font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Spinbox(conf_group, from_=0.1, to=1.0, increment=0.05,
                     textvariable=self.predict_conf, width=6).pack(side=LEFT)

        ttk.Separator(tools_inner, orient=VERTICAL).pack(side=LEFT, padx=15, fill=Y)

        # 操作按钮
        btn_group = ttk.Frame(tools_inner)
        btn_group.pack(side=LEFT)

        ttk.Button(btn_group, text="📷 选择图片", command=self.browse_predict_img,
                   bootstyle="info-outline", padding=(12, 5)).pack(side=LEFT, padx=3)
        ttk.Button(btn_group, text="▶  开始分析", command=self.start_prediction,
                   bootstyle="success", padding=(12, 5)).pack(side=LEFT, padx=3)

        # 内容区
        content = ttk.Panedwindow(tab, orient=HORIZONTAL)
        content.pack(fill=BOTH, expand=YES)

        # 左侧：图像显示
        img_container = ttk.Labelframe(content, text="📷 可视化结果", bootstyle="dark", padding=5)
        content.add(img_container, weight=3)

        self.predict_canvas = tk.Canvas(img_container, bg=COLORS['canvas_bg'], highlightthickness=0)
        self.predict_canvas.pack(fill=BOTH, expand=YES)

        # 欢迎提示（居中）
        self.predict_canvas.create_text(300, 200, text="📷  请选择图片开始检测",
                                         font=("Microsoft YaHei", 14),
                                         fill=COLORS['text_muted'], anchor=tk.CENTER)

        # 右侧：结果面板
        res_container = ttk.Labelframe(content, text="📋 检测报告", bootstyle="warning", padding=8)
        content.add(res_container, weight=1)

        self.predict_report = tk.Text(res_container, width=28, bg=COLORS['canvas_bg'],
                                      fg=COLORS['text_primary'],
                                      font=('Microsoft YaHei', 10), relief='flat',
                                      padx=10, pady=10, wrap=tk.WORD)
        self.predict_report.pack(fill=BOTH, expand=YES)

        # 默认显示提示文字
        self.predict_report.insert(tk.END, "💡 操作指引\n\n", "title")
        self.predict_report.insert(tk.END, "1. 选择训练好的模型文件 (.pt)\n\n")
        self.predict_report.insert(tk.END, "2. 点击「选择图片」加载待检测图片\n\n")
        self.predict_report.insert(tk.END, "3. 点击「开始分析」执行检测\n\n")
        self.predict_report.insert(tk.END, "━━━━━━━━━━━━━━━\n\n")
        self.predict_report.insert(tk.END, "📊 检测结果将在此处显示")
        self.predict_report.tag_config("title", font=("Microsoft YaHei", 12, "bold"),
                                       foreground=COLORS['accent_blue'])

    def setup_batch_tab(self):
        """批量预测界面 - 优化版"""
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  📂 批量检测  ")

        # 输入配置卡片
        config_card = ttk.Labelframe(tab, text="批量处理配置", bootstyle="info", padding=10)
        config_card.pack(fill=X, pady=(0, 8))

        # 第一行
        row1 = ttk.Frame(config_card)
        row1.pack(fill=X, pady=3)

        ttk.Label(row1, text="🧠 模型路径:", font=("Microsoft YaHei", 9), width=12).pack(side=LEFT)
        ttk.Entry(row1, textvariable=self.batch_model).pack(side=LEFT, fill=X, expand=YES, padx=5)
        ttk.Button(row1, text="📂 浏览", command=lambda: self.browse_file(self.batch_model),
                   bootstyle="info-outline", padding=(10, 2)).pack(side=LEFT)

        # 第二行
        row2 = ttk.Frame(config_card)
        row2.pack(fill=X, pady=3)

        ttk.Label(row2, text="📁 图片目录:", font=("Microsoft YaHei", 9), width=12).pack(side=LEFT)
        ttk.Entry(row2, textvariable=self.batch_data).pack(side=LEFT, fill=X, expand=YES, padx=5)
        ttk.Button(row2, text="📂 浏览", command=lambda: self.browse_dir(self.batch_data),
                   bootstyle="info-outline", padding=(10, 2)).pack(side=LEFT)

        # 启动按钮
        btn_frame = ttk.Frame(config_card)
        btn_frame.pack(fill=X, pady=(8, 0))
        ttk.Button(btn_frame, text="🚀 启动批量处理", command=self.start_batch_prediction,
                   bootstyle="success", padding=(20, 8)).pack(side=RIGHT)

        # 内容分割区
        content_pane = ttk.Panedwindow(tab, orient=HORIZONTAL)
        content_pane.pack(fill=BOTH, expand=YES, pady=5)

        # 左侧：详细分析报告
        report_frame = ttk.Labelframe(content_pane, text="📊 详细分析报告", bootstyle="dark", padding=8)
        content_pane.add(report_frame, weight=1)

        self.batch_report_text = tk.Text(report_frame, width=30, bg=COLORS['canvas_bg'],
                                         fg=COLORS['text_primary'],
                                         font=('Microsoft YaHei', 9), relief='flat',
                                         padx=10, pady=10, wrap=tk.WORD)
        self.batch_report_text.pack(fill=BOTH, expand=YES)

        # 右侧：图表区
        charts_frame = ttk.Frame(content_pane)
        content_pane.add(charts_frame, weight=2)

        self.chart_pie_frame = ttk.Labelframe(charts_frame, text="🥧 缺陷类别占比",
                                               bootstyle="dark", padding=3)
        self.chart_pie_frame.pack(side=TOP, fill=BOTH, expand=YES, pady=(0, 4))

        self.chart_hist_frame = ttk.Labelframe(charts_frame, text="📊 置信度分布",
                                                bootstyle="dark", padding=3)
        self.chart_hist_frame.pack(side=BOTTOM, fill=BOTH, expand=YES)

    def setup_video_tab(self):
        """视频检测界面 - 优化版"""
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  📹 视频流检测  ")

        # 控制栏卡片
        ctrl_card = ttk.Labelframe(tab, text="视频检测配置", bootstyle="info", padding=10)
        ctrl_card.pack(fill=X, pady=(0, 8))

        ctrl_inner = ttk.Frame(ctrl_card)
        ctrl_inner.pack(fill=X)

        # 模型选择
        model_grp = ttk.Frame(ctrl_inner)
        model_grp.pack(side=LEFT, fill=X, expand=YES)

        ttk.Label(model_grp, text="🧠 模型:", font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Entry(model_grp, textvariable=self.video_model, width=25).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(model_grp, text="📂", command=lambda: self.browse_file(self.video_model),
                   bootstyle="info-outline", padding=(8, 2)).pack(side=LEFT, padx=(5, 0))

        ttk.Separator(ctrl_inner, orient=VERTICAL).pack(side=LEFT, padx=12, fill=Y)

        # 视频源
        src_grp = ttk.Frame(ctrl_inner)
        src_grp.pack(side=LEFT, fill=X, expand=YES)

        ttk.Label(src_grp, text="🎥 视频源:", font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Entry(src_grp, textvariable=self.video_source, width=25).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(src_grp, text="📂", command=self.browse_video_and_preview,
                   bootstyle="info-outline", padding=(8, 2)).pack(side=LEFT, padx=(5, 0))

        ttk.Separator(ctrl_inner, orient=VERTICAL).pack(side=LEFT, padx=12, fill=Y)

        # 按钮组
        btn_grp = ttk.Frame(ctrl_inner)
        btn_grp.pack(side=LEFT)

        ttk.Button(btn_grp, text="▶ 开始检测", command=self.start_video_prediction,
                   bootstyle="success", padding=(12, 5)).pack(side=LEFT, padx=3)
        ttk.Button(btn_grp, text="📷 摄像头", command=self.start_camera_prediction,
                   bootstyle="warning", padding=(12, 5)).pack(side=LEFT, padx=3)
        ttk.Button(btn_grp, text="⏹ 停止", command=self.stop_video_prediction,
                   bootstyle="danger", padding=(12, 5)).pack(side=LEFT, padx=3)

        # 视频显示区
        video_container = ttk.Labelframe(tab, text="📺 视频画面", bootstyle="dark", padding=3)
        video_container.pack(fill=BOTH, expand=YES, pady=(0, 5))

        self.video_canvas = tk.Canvas(video_container, bg=COLORS['canvas_bg'], highlightthickness=0)
        self.video_canvas.pack(fill=BOTH, expand=YES)

        self.video_canvas.create_text(400, 200, text="📹  请选择视频源或点击摄像头",
                                       font=("Microsoft YaHei", 14),
                                       fill=COLORS['text_muted'], anchor=tk.CENTER)

        # 状态提示
        status_frame = ttk.Frame(tab)
        status_frame.pack(fill=X)
        self.video_status = ttk.Label(status_frame, text="⏸️ 等待操作 — 请选择视频源或点击摄像头开始",
                                       font=("Microsoft YaHei", 9), bootstyle="secondary")
        self.video_status.pack(side=LEFT)

    def setup_train_tab(self):
        """训练界面 - 优化版"""
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  ⚙️ 模型训练  ")

        # 居中配置卡片
        center_frame = ttk.Frame(tab)
        center_frame.pack(fill=X, pady=15, padx=40)

        card = ttk.Labelframe(center_frame, text="🔧 训练参数配置", padding=20, bootstyle="info")
        card.pack(fill=X)

        # 模型选择
        self.create_grid_input(card, 0, "🧠 预训练模型 (.pt):", self.train_model)

        # 数据集选择
        self.create_grid_input(card, 1, "📁 数据集配置 (.yaml):", self.train_data)

        # 训练参数行
        param_frame = ttk.Frame(card)
        param_frame.grid(row=2, column=0, columnspan=3, sticky=EW, padx=5, pady=10)

        # Epochs
        ttk.Label(param_frame, text="📈 训练轮数:",
                  font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Spinbox(param_frame, from_=1, to=3000, textvariable=self.train_epochs,
                     width=8).pack(side=LEFT, padx=(0, 20))

        # Batch Size
        ttk.Label(param_frame, text="📦 批大小:",
                  font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Spinbox(param_frame, from_=1, to=128, textvariable=self.train_batch,
                     width=8).pack(side=LEFT, padx=(0, 20))

        # Image Size
        ttk.Label(param_frame, text="📐 图像尺寸:",
                  font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Combobox(param_frame, textvariable=self.train_imgsz,
                      values=["320", "416", "512", "640", "800", "1024"],
                      width=8, state="readonly").pack(side=LEFT)

        card.columnconfigure(1, weight=1)

        # 启动按钮
        btn_container = ttk.Frame(center_frame)
        btn_container.pack(pady=20)

        ttk.Button(btn_container, text="🔥 开始训练", command=self.start_training,
                   bootstyle="danger", padding=(30, 10)).pack(side=LEFT, padx=5)

        # 进度条
        self.train_gauge = ttk.Floodgauge(tab, bootstyle="success",
                                          font=(None, 11, 'bold'),
                                          mask="训练中... {}%",
                                          orient=HORIZONTAL)

        # 提示信息
        tips_frame = ttk.Labelframe(tab, text="💡 训练提示", bootstyle="dark", padding=12)
        tips_frame.pack(fill=X, padx=40, pady=(0, 10))

        tips = [
            "• 首次训练建议使用 yolov8n.pt（轻量级）进行快速验证",
            "• Windows 下如遇多进程报错，可将 workers 设为 0",
            "• 训练结果将保存在 runs/detect/train_result/ 目录下",
            "• 建议 GPU 显存 ≥ 4GB，否则请降低 batch 大小",
        ]
        for tip in tips:
            ttk.Label(tips_frame, text=tip, font=("Microsoft YaHei", 9),
                      bootstyle="secondary").pack(anchor=W, pady=1)

    def setup_val_tab(self):
        """验证界面 - 优化版"""
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  📊 模型验证  ")

        # 配置卡片
        config_card = ttk.Labelframe(tab, text="验证配置", bootstyle="info", padding=12)
        config_card.pack(fill=X, pady=(0, 8))

        # 第一行：选择模型
        row1 = ttk.Frame(config_card)
        row1.pack(fill=X, pady=3)

        ttk.Label(row1, text="🧠 验证模型:", font=("Microsoft YaHei", 9), width=12).pack(side=LEFT)
        ttk.Entry(row1, textvariable=self.val_model).pack(side=LEFT, fill=X, expand=YES, padx=5)
        ttk.Button(row1, text="📂 浏览", command=lambda: self.browse_file(self.val_model),
                   bootstyle="info-outline", padding=(10, 2)).pack(side=LEFT)

        # 第二行：选择数据集
        row2 = ttk.Frame(config_card)
        row2.pack(fill=X, pady=3)

        ttk.Label(row2, text="📁 数据集配置:", font=("Microsoft YaHei", 9), width=12).pack(side=LEFT)
        ttk.Entry(row2, textvariable=self.val_data).pack(side=LEFT, fill=X, expand=YES, padx=5)
        ttk.Button(row2, text="📂 浏览", command=lambda: self.browse_file(self.val_data),
                   bootstyle="info-outline", padding=(10, 2)).pack(side=LEFT)

        # 按钮
        btn_frame = ttk.Frame(config_card)
        btn_frame.pack(fill=X, pady=(8, 0))
        ttk.Button(btn_frame, text="🚀 开始验证", command=self.start_validation,
                   bootstyle="success", padding=(20, 8)).pack(side=RIGHT)

        # 结果显示区
        result_frame = ttk.Labelframe(tab, text="📊 验证结果", bootstyle="dark", padding=8)
        result_frame.pack(fill=BOTH, expand=YES, pady=5)

        self.val_text = tk.Text(result_frame, font=('Cascadia Code', 10),
                                bg=COLORS['canvas_bg'], fg=COLORS['text_primary'],
                                padx=10, pady=10, relief='flat', wrap=tk.WORD)
        self.val_text.pack(fill=BOTH, expand=YES)

        # 默认提示
        self.val_text.insert(tk.END, "📊 验证指标说明\n\n", "title")
        self.val_text.insert(tk.END, "• mAP50-95：平均检测精度（IoU 0.5~0.95）\n")
        self.val_text.insert(tk.END, "• mAP50：IoU=0.5 时的检测精度\n")
        self.val_text.insert(tk.END, "• mAP75：IoU=0.75 时的检测精度\n\n")
        self.val_text.insert(tk.END, "选择模型和数据集后点击「开始验证」")
        self.val_text.tag_config("title", font=("Microsoft YaHei", 12, "bold"),
                                  foreground=COLORS['accent_blue'])

    # ------------------ 辅助 UI 构建函数 ------------------

    def create_file_input(self, parent, label, variable, width=40, btn_text="📂"):
        ttk.Label(parent, text=label, font=("Microsoft YaHei", 9)).pack(side=LEFT, padx=(0, 5))
        ttk.Entry(parent, textvariable=variable, width=width).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(parent, text=btn_text, command=lambda: self.browse_file(variable),
                   bootstyle="info-outline", padding=(8, 2)).pack(side=LEFT, padx=5)

    def create_grid_input(self, parent, row, label, variable, is_dir=False):
        cmd = lambda: self.browse_dir(variable) if is_dir else self.browse_file(variable)
        ttk.Label(parent, text=label, font=("Microsoft YaHei", 9)).grid(
            row=row, column=0, padx=5, pady=10, sticky=E)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, padx=5, pady=10, sticky=EW)
        ttk.Button(parent, text="📂 浏览", command=cmd,
                   bootstyle="info-outline", padding=(10, 2)).grid(row=row, column=2, padx=5)

    # ------------------ 逻辑功能实现 ------------------

    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.log_queue.put((full_msg, level))
        print(full_msg)

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg, level = self.log_queue.get()
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, msg + "\n", level)
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
        self.master.after(100, self.process_log_queue)

    def browse_file(self, variable):
        path = filedialog.askopenfilename()
        if path: variable.set(path)

    def browse_dir(self, variable):
        path = filedialog.askdirectory()
        if path: variable.set(path)

    def browse_video_and_preview(self):
        """选择视频并显示第一帧预览"""
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov")])
        if path:
            self.video_source.set(path)
            self.video_status.config(text="✅ 视频已加载，点击【开始检测】运行", bootstyle="info")
            try:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    self.show_image_on_canvas(img_pil, self.video_canvas)
                cap.release()
            except Exception as e:
                self.log(f"预览视频失败: {e}", "WARNING")

    def run_subprocess(self, cmd, log_callback=None, finish_callback=None):
        def thread_target():
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, encoding='utf-8', bufsize=1
                )
                for line in process.stdout:
                    msg = line.strip()
                    if msg:
                        self.log(msg)
                        if log_callback: log_callback(msg)
                process.wait()
                if finish_callback: self.master.after(0, finish_callback)
            except Exception as e:
                self.log(f"进程异常: {str(e)}", "ERROR")

        threading.Thread(target=thread_target, daemon=True).start()

    # --- 训练逻辑 ---
    def start_training(self):
        if not self.train_model.get() or not self.train_data.get():
            Messagebox.show_error("请填写模型和数据集路径", "参数错误")
            return
        self.train_gauge.pack(fill=X, pady=10)
        self.train_gauge.start()
        self.update_status("🔥 训练进行中...", "danger")

        cmd = [sys.executable, "train.py",
               "--model", self.train_model.get(),
               "--data", self.train_data.get(),
               "--epochs", self.train_epochs.get(),
               "--batch", self.train_batch.get(),
               "--imgsz", self.train_imgsz.get()]
        self.log(f"🚀 启动训练进程...", "INFO")

        def on_finish():
            self.train_gauge.stop()
            self.train_gauge.pack_forget()
            self.update_status("✅ 训练完成", "success")
            self.show_toast(title="训练完成", message="模型训练已结束", bootstyle="success")

        self.run_subprocess(cmd, finish_callback=on_finish)

    # --- 验证逻辑 ---
    def start_validation(self):
        if not self.val_model.get():
            Messagebox.show_warning("请选择模型文件 (.pt)")
            return
        if not self.val_data.get():
            Messagebox.show_warning("请选择数据集配置文件 (.yaml)")
            return

        self.val_text.delete(1.0, tk.END)
        self.val_text.insert(tk.END, "⏳ 正在初始化验证进程...\n\n")
        self.update_status("📊 验证进行中...", "info")

        cmd = [
            sys.executable, "val.py",
            "--model", self.val_model.get(),
            "--data", self.val_data.get()
        ]

        def on_val_finish():
            self.update_status("✅ 验证完成", "success")
            self.show_toast("验证完成", "结果已输出")

        self.run_subprocess(
            cmd,
            log_callback=lambda m: self.val_text.insert(tk.END, m + "\n"),
            finish_callback=on_val_finish
        )

    # --- 单图预测逻辑 ---
    def browse_predict_img(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg;*.bmp")])
        if path:
            self.predict_source.set(path)
            self.show_image_on_canvas(path, self.predict_canvas)

    def show_image_on_canvas(self, img_path_or_pil, canvas):
        try:
            if isinstance(img_path_or_pil, (str, Path)):
                pil_img = Image.open(img_path_or_pil)
            else:
                pil_img = img_path_or_pil
            pil_img = ImageOps.exif_transpose(pil_img)

            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()
            if canvas_w < 10: canvas_w, canvas_h = 600, 400

            ratio = min(canvas_w / pil_img.width, canvas_h / pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))

            resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized)

            canvas.delete("all")
            canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=tk_img)
            canvas.image = tk_img
        except Exception as e:
            self.log(f"显示图片错误: {e}", "ERROR")

    def start_prediction(self):
        if not self.predict_model.get() or not self.predict_source.get():
            Messagebox.show_error("请选择模型和图片")
            return
        self.update_status("🔮 检测进行中...", "info")
        exp_name = f"single_{datetime.now().strftime('%H%M%S')}"
        cmd = [sys.executable, "predict.py",
               "--model", self.predict_model.get(),
               "--source", self.predict_source.get(),
               "--conf", self.predict_conf.get(),
               "--name", exp_name, "--save", "--project", "runs/detect"]

        def on_predict_finish():
            save_dir = Path("runs/detect") / exp_name
            found_imgs = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png")) + list(save_dir.glob("*.jpeg"))
            if found_imgs:
                res_path = found_imgs[0]
                self.show_image_on_canvas(res_path, self.predict_canvas)
                txt_path = save_dir / "labels" / f"{Path(self.predict_source.get()).stem}.txt"
                report_text = f"✅ 检测完成\n\n"
                report_text += f"📂 保存路径:\n{res_path}\n\n"
                report_text += f"━━━━━━━━━━━━━━━\n\n"
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        report_text += f"📊 发现目标数量: {len(lines)}\n\n"
                        class_map = {0: "龟裂", 1: "夹杂", 2: "斑块", 3: "麻点", 4: "氧化铁皮", 5: "划痕"}
                        report_text += "📋 详细检测结果:\n\n"
                        for line in lines:
                            parts = line.split()
                            cls_id = int(parts[0])
                            cls_name = class_map.get(cls_id, f"Class {cls_id}")
                            conf = float(parts[-1]) if len(parts) > 5 else 0.0
                            report_text += f"  • {cls_name}: {conf:.1%}\n"
                else:
                    report_text += "⚠️ 未检测到明显缺陷"

                self.predict_report.delete(1.0, tk.END)
                self.predict_report.insert(tk.END, report_text)
                self.update_status("✅ 单图检测完成", "success")
                self.show_toast("检测成功", "结果已更新", bootstyle="success")
            else:
                self.log("未找到结果图片", "WARNING")

        self.run_subprocess(cmd, finish_callback=on_predict_finish)

    # --- 批量预测逻辑 ---
    def start_batch_prediction(self):
        if not self.batch_model.get() or not self.batch_data.get():
            Messagebox.show_error("请完善信息")
            return
        self.update_status("🚀 批量处理进行中...", "info")
        exp_name = f"batch_{datetime.now().strftime('%H%M%S')}"
        cmd = [sys.executable, "predict.py",
               "--model", self.batch_model.get(),
               "--source", self.batch_data.get(),
               "--name", exp_name, "--save", "--save_txt", "--project", "runs/detect"]

        def on_batch_finish():
            self.log("批量处理完成，开始生成分析报告...")
            save_dir = Path("runs/detect") / exp_name / "labels"
            if save_dir.exists():
                self.analyze_and_report_batch(save_dir, str(Path("runs/detect") / exp_name))
                self.update_status("✅ 批量处理完成，报告已生成", "success")
                self.show_toast("批量完成", "报告与图表已生成", bootstyle="success")
            else:
                self.log("未找到标签目录，可能未检测到任何目标", "WARNING")

        self.run_subprocess(cmd, finish_callback=on_batch_finish)

    def analyze_and_report_batch(self, label_dir, output_path):
        """生成详细报告并绘制图表"""
        for widget in self.chart_pie_frame.winfo_children(): widget.destroy()
        for widget in self.chart_hist_frame.winfo_children(): widget.destroy()

        stats = {
            'total_files': len(list(label_dir.glob("*.txt"))),
            'total_defects': 0,
            'classes': defaultdict(int),
            'confidences': [],
            'areas': []
        }

        class_map = {0: "龟裂", 1: "夹杂", 2: "斑块", 3: "麻点", 4: "氧化铁皮", 5: "划痕"}

        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 1:
                        cls_id = int(parts[0])
                        stats['classes'][class_map.get(cls_id, str(cls_id))] += 1
                        stats['total_defects'] += 1
                    if len(parts) >= 6:
                        stats['confidences'].append(float(parts[5]))
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        stats['areas'].append(w * h)

        # 文本报告
        avg_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0

        report_text = f"📋 批量检测分析报告\n"
        report_text += f"{'━' * 24}\n\n"
        report_text += f"📂 输出目录:\n{output_path}\n\n"
        report_text += f"🖼️ 包含缺陷文件数: {stats['total_files']}\n"
        report_text += f"⚠️ 检出缺陷总数: {stats['total_defects']}\n"
        report_text += f"🎯 平均置信度: {avg_conf:.2%}\n\n"
        report_text += f"{'━' * 24}\n\n"

        report_text += f"📊 各类缺陷统计:\n\n"
        for k, v in sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True):
            ratio = v / stats['total_defects'] if stats['total_defects'] else 0
            bar = '█' * int(ratio * 15)
            report_text += f"  {k:<6} {bar} {v}个 ({ratio:.1%})\n"

        report_text += f"\n{'━' * 24}\n\n"
        report_text += f"📏 缺陷尺寸分析:\n\n"
        if stats['areas']:
            report_text += f"  最大: {max(stats['areas']):.4f}\n"
            report_text += f"  最小: {min(stats['areas']):.4f}\n"
            large_count = sum(1 for a in stats['areas'] if a > 0.1)
            report_text += f"  大型缺陷(>10%): {large_count}个\n"
        else:
            report_text += "  暂无尺寸数据\n"

        report_text += f"\n{'━' * 24}\n\n"
        report_text += f"📝 综合评价:\n\n"
        if avg_conf > 0.8:
            report_text += "  ✅ 置信度高，结果可靠。\n"
        elif avg_conf < 0.5:
            report_text += "  ⚠️ 置信度偏低，建议人工复核。\n"
        if stats['total_defects'] == 0:
            report_text += "  ✅ 批次质量极佳，未发现缺陷。\n"

        self.batch_report_text.delete(1.0, tk.END)
        self.batch_report_text.insert(tk.END, report_text)

        # 绘制图表
        plt.style.use('dark_background')

        # 饼图
        if stats['classes']:
            fig1, ax1 = plt.subplots(figsize=(5, 3), dpi=100)
            fig1.patch.set_facecolor('#0d1117')
            ax1.set_facecolor('#0d1117')
            colors = ['#00adb5', '#7c4dff', '#ff9800', '#00e676', '#ff5252', '#448aff']
            ax1.pie(stats['classes'].values(), labels=stats['classes'].keys(),
                    autopct='%1.1f%%', startangle=90,
                    colors=colors[:len(stats['classes'])],
                    textprops={'fontsize': 9, 'color': 'white'})
            ax1.set_title("缺陷类别占比", fontsize=11, color='white', pad=10)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)

            canvas1 = FigureCanvasTkAgg(fig1, master=self.chart_pie_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=BOTH, expand=YES)

        # 直方图
        if stats['confidences']:
            fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=100)
            fig2.patch.set_facecolor('#0d1117')
            ax2.set_facecolor('#0d1117')
            ax2.hist(stats['confidences'], bins=10, color='#00adb5', alpha=0.85,
                     edgecolor='#1a1a2e', linewidth=1.5)
            ax2.set_title("置信度分布", fontsize=11, color='white', pad=10)
            ax2.set_xlabel("Confidence", fontsize=9, color='#a0a0b0')
            ax2.set_ylabel("Count", fontsize=9, color='#a0a0b0')
            ax2.tick_params(axis='both', which='major', labelsize=8, colors='#a0a0b0')
            ax2.grid(True, alpha=0.15, color='#2a2a4a')
            ax2.spines['bottom'].set_color('#2a2a4a')
            ax2.spines['left'].set_color('#2a2a4a')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)

            canvas2 = FigureCanvasTkAgg(fig2, master=self.chart_hist_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=BOTH, expand=YES)

    # --- 视频预测逻辑 ---
    def start_video_prediction(self):
        self.run_video_inference(source=self.video_source.get())

    def start_camera_prediction(self):
        self.run_video_inference(source="0")

    def run_video_inference(self, source):
        if not self.video_model.get():
            Messagebox.show_error("请选择模型")
            return
        self.video_loop_running = True
        self.video_status.config(text="🔥 正在推理中...", bootstyle="danger")
        self.update_status("📹 视频推理进行中...", "danger")

        def video_thread():
            try:
                model = YOLO(self.video_model.get())
                cap = cv2.VideoCapture(int(source) if source == "0" else source)
                while self.video_loop_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    results = model(frame, verbose=False)
                    res_plotted = results[0].plot()
                    img_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    self.master.after(0, lambda i=img_pil: self.show_image_on_canvas(i, self.video_canvas))
                cap.release()
                self.master.after(0, lambda: self.video_status.config(text="⏸️ 已停止", bootstyle="secondary"))
                self.master.after(0, lambda: self.update_status("✅ 视频处理完成", "success"))
            except Exception as e:
                self.log(f"视频流错误: {e}", "ERROR")

        threading.Thread(target=video_thread, daemon=True).start()

    def stop_video_prediction(self):
        self.video_loop_running = False
        self.video_status.config(text="⏸️ 正在停止...", bootstyle="warning")

    def on_close(self):
        self.video_loop_running = False
        self.master.destroy()


if __name__ == "__main__":
    app = ttk.Window(themename="superhero")
    gui = YOLOv8_GUI(app)
    app.mainloop()
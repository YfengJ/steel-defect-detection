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
from PIL import Image, ImageTk, ImageOps

# 引入YOLO
from ultralytics import YOLO


class YOLOv8_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("🛡️ 智能钢铁缺陷检测系统 Pro | YOLOv8")
        self.master.geometry("1300x900")
        self.master.minsize(1100, 768)

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

        # 验证页变量
        self.val_model = ttk.StringVar()
        self.val_data = ttk.StringVar()
        # 预测页变量
        self.predict_model = ttk.StringVar()
        self.predict_source = ttk.StringVar()

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
        """构建主界面 - 采用垂直分割布局"""
        # 主容器
        main_frame = ttk.Frame(self.master, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        # 1. 顶部标题栏
        header = ttk.Frame(main_frame)
        header.pack(fill=X, pady=(0, 5))
        ttk.Label(header, text="STEEL DEFECT DETECTION", font=("Impact", 20), bootstyle="secondary").pack(side=LEFT)
        ttk.Label(header, text="V2.2", font=("Arial", 10), bootstyle="success").pack(side=LEFT, padx=5, pady=(10, 0))

        # 2. 全局垂直分割面板
        self.main_paned = ttk.Panedwindow(main_frame, orient=VERTICAL)
        self.main_paned.pack(fill=BOTH, expand=YES)

        # --- 上半部分：功能选项卡 ---
        self.notebook = ttk.Notebook(self.main_paned, bootstyle="primary")
        self.main_paned.add(self.notebook, weight=4)

        # 创建各个功能Tab
        self.setup_predict_tab()
        self.setup_batch_tab()
        self.setup_video_tab()
        self.setup_train_tab()
        self.setup_val_tab()

        # --- 下半部分：日志区域 ---
        self.setup_log_area(self.main_paned)

    # ------------------ UI 组件构建区域 ------------------

    def setup_log_area(self, parent_paned):
        """底部日志控制台"""
        log_frame = ttk.Labelframe(parent_paned, text="📟 系统运行日志", padding=5, bootstyle="info")
        parent_paned.add(log_frame, weight=1)

        self.log_text = tk.Text(log_frame, height=8, bg='#2b2b2b', fg='white',
                                font=('Consolas', 9), state='disabled', relief='flat')
        self.log_text.pack(side=LEFT, fill=BOTH, expand=YES)

        self.log_text.tag_config('INFO', foreground='#00bc8c')
        self.log_text.tag_config('WARNING', foreground='#f39c12')
        self.log_text.tag_config('ERROR', foreground='#e74c3c')

        vsb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        vsb.pack(side=RIGHT, fill=Y)
        self.log_text.configure(yscrollcommand=vsb.set)

    def setup_predict_tab(self):
        """单图预测界面"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="🖼️ 单图检测")

        # 顶部工具栏
        tools = ttk.Frame(tab)
        tools.pack(fill=X, pady=(0, 5))

        self.create_file_input(tools, "模型路径:", self.predict_model)
        ttk.Separator(tools, orient=VERTICAL).pack(side=LEFT, padx=15, fill=Y)

        ttk.Button(tools, text="📷 选择图片", command=self.browse_predict_img, bootstyle="info-outline").pack(side=LEFT,
                                                                                                             padx=5)
        ttk.Button(tools, text="▶ 开始分析", command=self.start_prediction, bootstyle="warning").pack(side=LEFT, padx=5)

        # 内容区
        content = ttk.Panedwindow(tab, orient=HORIZONTAL)
        content.pack(fill=BOTH, expand=YES)

        # 左侧：图像显示
        img_container = ttk.Labelframe(content, text="可视化结果", bootstyle="secondary", padding=5)
        content.add(img_container, weight=3)

        self.predict_canvas = tk.Canvas(img_container, bg='#1e1e1e', highlightthickness=0)
        self.predict_canvas.pack(fill=BOTH, expand=YES)

        # 右侧：结果面板
        res_container = ttk.Labelframe(content, text="检测报告", bootstyle="warning", padding=5)
        content.add(res_container, weight=1)

        self.predict_report = tk.Text(res_container, width=30, bg='#2b2b2b', fg='#f0f0f0',
                                      font=('微软雅黑', 10), relief='flat', padx=5, pady=5)
        self.predict_report.pack(fill=BOTH, expand=YES)

    def setup_batch_tab(self):
        """批量预测界面"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📂 批量检测")

        # 1. 输入区
        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=X, pady=5)

        ttk.Label(input_frame, text="模型路径:").grid(row=0, column=0, sticky=E, padx=5)
        ttk.Entry(input_frame, textvariable=self.batch_model).grid(row=0, column=1, sticky=EW, padx=5)
        ttk.Button(input_frame, text="📂", command=lambda: self.browse_file(self.batch_model),
                   style="secondary-outline").grid(row=0, column=2)

        ttk.Label(input_frame, text="图片目录:").grid(row=0, column=3, sticky=E, padx=10)
        ttk.Entry(input_frame, textvariable=self.batch_data).grid(row=0, column=4, sticky=EW, padx=5)
        ttk.Button(input_frame, text="📂", command=lambda: self.browse_dir(self.batch_data),
                   style="secondary-outline").grid(row=0, column=5)

        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(4, weight=1)

        # 按钮
        ttk.Button(tab, text="🚀 启动批量处理", command=self.start_batch_prediction,
                   bootstyle="primary", width=20).pack(pady=5)

        # 2. 内容分割区
        content_pane = ttk.Panedwindow(tab, orient=HORIZONTAL)
        content_pane.pack(fill=BOTH, expand=YES, pady=5)

        # 左侧：详细分析报告
        report_frame = ttk.Labelframe(content_pane, text="📊 详细分析报告", bootstyle="info", padding=5)
        content_pane.add(report_frame, weight=1)

        self.batch_report_text = tk.Text(report_frame, width=30, bg='#2b2b2b', fg='#f0f0f0',
                                         font=('微软雅黑', 9), relief='flat', padx=10, pady=10)
        self.batch_report_text.pack(fill=BOTH, expand=YES)

        # 右侧：图表区
        charts_frame = ttk.Frame(content_pane)
        content_pane.add(charts_frame, weight=2)

        self.chart_pie_frame = ttk.Labelframe(charts_frame, text="缺陷类别占比", bootstyle="secondary", padding=2)
        self.chart_pie_frame.pack(side=TOP, fill=BOTH, expand=YES, pady=(0, 5))

        self.chart_hist_frame = ttk.Labelframe(charts_frame, text="置信度分布", bootstyle="secondary", padding=2)
        self.chart_hist_frame.pack(side=BOTTOM, fill=BOTH, expand=YES)

    def setup_video_tab(self):
        """视频检测界面 - 修复不显示预览和按钮名称问题"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📹 视频流检测")

        # 控制栏
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill=X, pady=5)

        # 模型选择
        self.create_file_input(ctrl, "模型:", self.video_model, width=20)
        ttk.Separator(ctrl, orient=VERTICAL).pack(side=LEFT, padx=10, fill=Y)

        # 视频源选择（修改：使用自定义的浏览函数以支持预览）
        ttk.Label(ctrl, text="视频源:").pack(side=LEFT, padx=(0, 5))
        ttk.Entry(ctrl, textvariable=self.video_source, width=20).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(ctrl, text="📂", command=self.browse_video_and_preview,
                   bootstyle="secondary-outline").pack(side=LEFT, padx=5)

        ttk.Separator(ctrl, orient=VERTICAL).pack(side=LEFT, padx=10, fill=Y)

        # 按钮组（修改：名称更直观）
        ttk.Button(ctrl, text="▶ 开始检测", command=self.start_video_prediction, bootstyle="success").pack(side=LEFT,
                                                                                                           padx=5)
        ttk.Button(ctrl, text="📷 摄像头", command=self.start_camera_prediction, bootstyle="warning").pack(side=LEFT,
                                                                                                          padx=5)
        ttk.Button(ctrl, text="⏹ 停止", command=self.stop_video_prediction, bootstyle="danger").pack(side=LEFT, padx=5)

        # 视频显示区
        self.video_canvas = tk.Canvas(tab, bg='black')
        self.video_canvas.pack(fill=BOTH, expand=YES, pady=5)

        self.video_status = ttk.Label(tab, text="请选择视频源或点击摄像头", bootstyle="secondary")
        self.video_status.pack(anchor=W)

    def setup_train_tab(self):
        """训练界面"""
        tab = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(tab, text="⚙️ 模型训练")

        center_frame = ttk.Frame(tab)
        center_frame.pack(fill=X, pady=20, padx=50)

        card = ttk.Labelframe(center_frame, text="训练参数配置", padding=20, bootstyle="primary")
        card.pack(fill=X)

        self.create_grid_input(card, 0, "预训练模型 (.pt):", self.train_model)
        self.create_grid_input(card, 1, "数据集配置 (.yaml):", self.train_data)

        ttk.Label(card, text="训练轮数 (Epochs):").grid(row=2, column=0, padx=5, pady=10, sticky=E)
        ttk.Spinbox(card, from_=1, to=3000, textvariable=self.train_epochs).grid(row=2, column=1, padx=5, pady=10,
                                                                                 sticky=W)
        card.columnconfigure(1, weight=1)

        ttk.Button(center_frame, text="🔥 开始训练", command=self.start_training,
                   bootstyle="danger", width=30).pack(pady=20)

        self.train_gauge = ttk.Floodgauge(tab, bootstyle="success",
                                          font=(None, 12, 'bold'),
                                          mask="训练中... {}%",
                                          orient=HORIZONTAL)

    def setup_val_tab(self):
        """验证界面 - 修改版"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📊 模型验证")

        # 创建一个容器来放输入框
        input_container = ttk.Frame(tab)
        input_container.pack(fill=X, pady=10)

        # 第一行：选择模型
        row1 = ttk.Frame(input_container)
        row1.pack(fill=X, pady=5)
        self.create_file_input(row1, "验证模型:", self.val_model, width=50)

        # 第二行：选择数据集 (新增)
        row2 = ttk.Frame(input_container)
        row2.pack(fill=X, pady=5)
        self.create_file_input(row2, "数据集(yaml):", self.val_data, width=50)

        # 按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=X, pady=5)
        ttk.Button(btn_frame, text="🚀 开始验证", command=self.start_validation, bootstyle="primary").pack(side=LEFT,
                                                                                                          padx=85)
        # padx=85 是为了让按钮大致对齐，你可以自己调

        # 结果显示区
        self.val_text = tk.Text(tab, font=('Consolas', 10), bg='#2b2b2b', fg='white', padx=5, pady=5)
        self.val_text.pack(fill=BOTH, expand=YES, pady=5)

    # ------------------ 辅助 UI 构建函数 ------------------

    def create_file_input(self, parent, label, variable, width=40, btn_text="📂"):
        ttk.Label(parent, text=label).pack(side=LEFT, padx=(0, 5))
        ttk.Entry(parent, textvariable=variable, width=width).pack(side=LEFT, fill=X, expand=YES)
        ttk.Button(parent, text=btn_text, command=lambda: self.browse_file(variable),
                   bootstyle="secondary-outline").pack(side=LEFT, padx=5)

    def create_grid_input(self, parent, row, label, variable, is_dir=False):
        cmd = lambda: self.browse_dir(variable) if is_dir else self.browse_file(variable)
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=5, pady=10, sticky=E)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, padx=5, pady=10, sticky=EW)
        ttk.Button(parent, text="📂", command=cmd, bootstyle="secondary-outline").grid(row=row, column=2, padx=5)

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
            self.video_status.config(text="视频已加载，点击【开始检测】运行", bootstyle="info")
            # 预览第一帧
            try:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                if ret:
                    # 转换颜色空间 BGR -> RGB
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    # 显示
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
        cmd = [sys.executable, "train.py", "--model", self.train_model.get(), "--data", self.train_data.get(),
               "--epochs", self.train_epochs.get()]
        self.log(f"🚀 启动训练进程...", "INFO")

        def on_finish():
            self.train_gauge.stop()
            self.train_gauge.pack_forget()
            self.show_toast(title="训练完成", message="模型训练已结束", bootstyle="success")

        self.run_subprocess(cmd, finish_callback=on_finish)

    # --- 验证逻辑 ---
    def start_validation(self):
        """验证逻辑 - 修改版"""
        # 1. 校验模型
        if not self.val_model.get():
            Messagebox.show_warning("请选择模型文件 (.pt)")
            return

        # 2. 校验数据集 (新增)
        if not self.val_data.get():
            Messagebox.show_warning("请选择数据集配置文件 (.yaml)")
            return

        self.val_text.delete(1.0, tk.END)
        self.val_text.insert(tk.END, "⏳ 正在初始化验证进程...\n")

        # 3. 组装命令，加入 --data 参数
        cmd = [
            sys.executable, "val.py",
            "--model", self.val_model.get(),
            "--data", self.val_data.get()  # <--- 新增这行，强制指定数据集
        ]

        self.run_subprocess(
            cmd,
            log_callback=lambda m: self.val_text.insert(tk.END, m + "\n"),
            finish_callback=lambda: self.show_toast("验证完成", "结果已输出")
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
        exp_name = f"single_{datetime.now().strftime('%H%M%S')}"
        cmd = [sys.executable, "predict.py", "--model", self.predict_model.get(), "--source", self.predict_source.get(),
               "--name", exp_name, "--save", "--project", "runs/detect"]

        def on_predict_finish():
            save_dir = Path("runs/detect") / exp_name
            found_imgs = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png")) + list(save_dir.glob("*.jpeg"))
            if found_imgs:
                res_path = found_imgs[0]
                self.show_image_on_canvas(res_path, self.predict_canvas)
                txt_path = save_dir / "labels" / f"{Path(self.predict_source.get()).stem}.txt"
                report_text = f"✅ 检测完成\n📂 保存路径: {res_path}\n\n"
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        report_text += f"📊 发现目标数量: {len(lines)}\n\n详细数据:\n"
                        class_map = {0: "裂纹", 1: "夹杂", 2: "气孔", 3: "划痕", 4: "氧化", 5: "脱碳"}
                        for line in lines:
                            parts = line.split()
                            cls_id = int(parts[0])
                            cls_name = class_map.get(cls_id, f"Class {cls_id}")
                            conf = float(parts[-1]) if len(parts) > 5 else 0.0
                            report_text += f"- {cls_name}: 置信度 {conf:.2f}\n"
                else:
                    report_text += "⚠️ 未检测到明显缺陷"
                self.predict_report.delete(1.0, tk.END)
                self.predict_report.insert(tk.END, report_text)
                self.show_toast("检测成功", "结果已更新", bootstyle="success")
            else:
                self.log("未找到结果图片", "WARNING")

        self.run_subprocess(cmd, finish_callback=on_predict_finish)

    # --- 批量预测逻辑 ---
    def start_batch_prediction(self):
        if not self.batch_model.get() or not self.batch_data.get():
            Messagebox.show_error("请完善信息")
            return
        exp_name = f"batch_{datetime.now().strftime('%H%M%S')}"
        cmd = [sys.executable, "predict.py", "--model", self.batch_model.get(), "--source", self.batch_data.get(),
               "--name", exp_name, "--save", "--save_txt", "--project", "runs/detect"]

        def on_batch_finish():
            self.log("批量处理完成，开始生成分析报告...")
            save_dir = Path("runs/detect") / exp_name / "labels"
            if save_dir.exists():
                self.analyze_and_report_batch(save_dir, str(Path("runs/detect") / exp_name))
                self.show_toast("批量完成", "报告与图表已生成", bootstyle="success")
            else:
                self.log("未找到标签目录，可能未检测到任何目标", "WARNING")

        self.run_subprocess(cmd, finish_callback=on_batch_finish)

    def analyze_and_report_batch(self, label_dir, output_path):
        """生成详细报告并绘制图表"""
        # 清除旧图表
        for widget in self.chart_pie_frame.winfo_children(): widget.destroy()
        for widget in self.chart_hist_frame.winfo_children(): widget.destroy()

        # 统计变量
        stats = {
            'total_files': len(list(label_dir.glob("*.txt"))),
            'total_defects': 0,
            'classes': defaultdict(int),
            'confidences': [],
            'areas': []
        }

        class_map = {0: "裂纹", 1: "夹杂", 2: "气孔", 3: "划痕", 4: "氧化", 5: "脱碳"}

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

        # 1. 生成文本报告
        avg_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0

        report_text = f"📋 批量检测分析报告\n"
        report_text += f"========================\n"
        report_text += f"📂 输出目录: {output_path}\n"
        report_text += f"🖼️ 包含缺陷文件数: {stats['total_files']}\n"
        report_text += f"⚠️ 检出缺陷总数: {stats['total_defects']}\n"
        report_text += f"🎯 平均置信度: {avg_conf:.2%}\n\n"

        report_text += f"📊 各类缺陷统计:\n"
        for k, v in sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True):
            ratio = v / stats['total_defects'] if stats['total_defects'] else 0
            report_text += f"  - {k}: {v}个 ({ratio:.1%})\n"

        report_text += f"\n📏 缺陷尺寸分析 (相对面积):\n"
        if stats['areas']:
            report_text += f"  - 最大缺陷: {max(stats['areas']):.4f}\n"
            report_text += f"  - 最小缺陷: {min(stats['areas']):.4f}\n"
            large_count = sum(1 for a in stats['areas'] if a > 0.1)
            report_text += f"  - 大型缺陷(>10%): {large_count}个\n"
        else:
            report_text += "  暂无尺寸数据\n"

        report_text += f"\n📝 综合评价:\n"
        if avg_conf > 0.8:
            report_text += "  模型检测置信度高，结果可靠。\n"
        elif avg_conf < 0.5:
            report_text += "  平均置信度偏低，建议人工复核。\n"
        if stats['total_defects'] == 0:
            report_text += "  批次质量极佳，未发现缺陷。\n"

        self.batch_report_text.delete(1.0, tk.END)
        self.batch_report_text.insert(tk.END, report_text)

        # 2. 绘制图表
        plt.style.use('dark_background')

        # 饼图
        if stats['classes']:
            fig1, ax1 = plt.subplots(figsize=(5, 3), dpi=100)
            ax1.pie(stats['classes'].values(), labels=stats['classes'].keys(),
                    autopct='%1.1f%%', startangle=90,
                    colors=plt.cm.Pastel1.colors,
                    textprops={'fontsize': 8})
            ax1.set_title("缺陷类别占比", fontsize=10)
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            canvas1 = FigureCanvasTkAgg(fig1, master=self.chart_pie_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=BOTH, expand=YES)

        # 直方图
        if stats['confidences']:
            fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=100)
            ax2.hist(stats['confidences'], bins=10, color='#00bc8c', alpha=0.7, edgecolor='white')
            ax2.set_title("置信度分布", fontsize=10)
            ax2.set_xlabel("Confidence", fontsize=8)
            ax2.set_ylabel("Count", fontsize=8)
            ax2.tick_params(axis='both', which='major', labelsize=8)
            ax2.grid(True, alpha=0.2)
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
                self.master.after(0, lambda: self.video_status.config(text="已停止", bootstyle="secondary"))
            except Exception as e:
                self.log(f"视频流错误: {e}", "ERROR")

        threading.Thread(target=video_thread, daemon=True).start()

    def stop_video_prediction(self):
        self.video_loop_running = False
        self.video_status.config(text="正在停止...", bootstyle="warning")

    def on_close(self):
        self.video_loop_running = False
        self.master.destroy()


if __name__ == "__main__":
    app = ttk.Window(themename="superhero")
    gui = YOLOv8_GUI(app)
    app.mainloop()
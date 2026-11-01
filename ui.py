"""Desktop GUI for steel surface defect detection built on ttkbootstrap."""

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import matplotlib
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageOps, ImageTk
from ttkbootstrap.style import ThemeDefinition

from gui_logic import (
    CLASS_MAP,
    DEVICE_AUTO,
    absolute_path,
    batch_verdict,
    device_arg,
    find_result_image,
    format_conf,
    is_progress_noise,
    is_windows,
    open_video_capture,
    parse_confidence,
    parse_label_file,
    parse_metrics_line,
    parse_positive_int,
    python_executable,
    read_training_progress,
    stop_process_tree,
    subprocess_options,
    summarize_batch,
    window_geometry,
)
from path_validation import (
    validate_dataset_config,
    validate_directory_source,
    validate_file_source,
    validate_model_path,
    validate_video_source,
)
from ultralytics import YOLO

APP_NAME = "钢材表面缺陷检测"
APP_VERSION = "0.2.1"

# 子进程脚本与输出目录以仓库根目录为准，避免从其他目录启动 GUI 时找不到脚本或结果
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "detect"
# 子进程解释器：从 pythonw.exe 启动 GUI 时改用同目录的 python.exe，保证能读取输出
PYTHON = python_executable()

# 日志控件最多保留的行数，防止长时间训练时内存与渲染开销持续增长
MAX_LOG_LINES = 2000

UI_FONTS = ("PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC",
            "Source Han Sans SC", "WenQuanYi Micro Hei", "Helvetica Neue", "Segoe UI", "DejaVu Sans")
MONO_FONTS = ("SF Mono", "Menlo", "Cascadia Mono", "Consolas", "JetBrains Mono",
              "DejaVu Sans Mono", "Courier New")
# matplotlib 中文字体候选（Windows / macOS / Linux）
CJK_FONTS = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti TC", "Arial Unicode MS",
             "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]

IMAGE_TYPES = [("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("所有文件", "*.*")]
MODEL_TYPES = [("PyTorch 权重", "*.pt"), ("所有文件", "*.*")]
YAML_TYPES = [("数据集配置", "*.yaml *.yml"), ("所有文件", "*.*")]
VIDEO_TYPES = [("视频", "*.mp4 *.avi *.mkv *.mov"), ("所有文件", "*.*")]

THEMES = {
    "dark": {
        "ttk": dict(primary="#4f8cff", secondary="#2b323c", success="#2fb67c", info="#3fa7d6",
                    warning="#e9a23b", danger="#e5534b", light="#c5ccd6", dark="#0b0e12",
                    bg="#161b22", fg="#e6edf3", selectbg="#2d4a7a", selectfg="#ffffff",
                    border="#2a313b", inputfg="#e6edf3", inputbg="#0f1318", active="#1f2630"),
        "page": "#0d1117", "sidebar": "#0a0d12", "surface": "#161b22", "raised": "#1c222b",
        "border": "#262d37", "text": "#e6edf3", "muted": "#8b949e", "subtle": "#5c6672",
        "canvas": "#0a0d12", "nav_hover": "#141a22", "nav_active": "#1a2230",
    },
    "light": {
        "ttk": dict(primary="#2563eb", secondary="#64748b", success="#16a34a", info="#0891b2",
                    warning="#d97706", danger="#dc2626", light="#f1f5f9", dark="#1e293b",
                    bg="#ffffff", fg="#0f172a", selectbg="#dbe7ff", selectfg="#0f172a",
                    border="#d5dbe3", inputfg="#0f172a", inputbg="#ffffff", active="#eef2f7"),
        "page": "#f3f5f8", "sidebar": "#eaeef3", "surface": "#ffffff", "raised": "#f6f8fa",
        "border": "#e1e6ed", "text": "#0f172a", "muted": "#5b6675", "subtle": "#94a3b8",
        "canvas": "#eef1f5", "nav_hover": "#e1e7ef", "nav_active": "#d8e1ee",
    },
}
CHART_COLORS = ["#4f8cff", "#2fb67c", "#e9a23b", "#e5534b", "#a371f7", "#3fa7d6"]

PAGES = (
    ("predict", "单图检测", "加载一张钢材表面图片，查看缺陷位置与置信度"),
    ("batch", "批量检测", "对整个文件夹运行检测，并生成缺陷统计报告"),
    ("video", "视频检测", "对视频文件或摄像头画面进行实时推理"),
    ("train", "模型训练", "在本地数据集上训练 YOLOv8，并实时查看训练曲线"),
    ("val", "模型验证", "评估权重在验证集上的精度、召回率与 mAP"),
)


def pick_font_family(candidates):
    families = set(tkfont.families())
    return next((name for name in candidates if name in families), "TkDefaultFont")


def detect_devices():
    """Return (device choices, human readable accelerator label)."""
    choices = [DEVICE_AUTO, "cpu"]
    label = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            choices.append("0")
            label = f"CUDA · {torch.cuda.get_device_name(0)}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            choices.append("mps")
            if label == "CPU":
                label = "Apple MPS"
    except Exception:
        pass
    return choices, label


def open_path(path):
    """Open a file or folder with the platform's default application."""
    path = str(path)
    if is_windows():
        os.startfile(path)  # noqa: S606 - local path chosen by the user
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class YOLOv8_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title(f"{APP_NAME} · YOLOv8")
        self._fit_window()

        self.style = master.style if hasattr(master, "style") else ttk.Style()
        self.theme_mode = "dark"
        self.palette = THEMES[self.theme_mode]
        self._painted = []
        self._register_themes()
        self._build_fonts()

        self.setup_variables()
        self.log_queue = queue.Queue()
        # 工作线程只往队列里放回调，由 Tk 主线程统一执行：非线程版 Tcl 下跨线程调用 after() 不会触发回调
        self.ui_queue = queue.Queue()
        self.create_ui()
        self.apply_theme(self.theme_mode)
        self.process_log_queue()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ setup

    def _fit_window(self):
        """按屏幕和 DPI 缩放决定窗口尺寸，避免 1366x768 等小屏幕上窗口超出屏幕。"""
        scale = self.master.winfo_fpixels("1i") / 96 if is_windows() else 1.0
        width, height, min_w, min_h = window_geometry(
            self.master.winfo_screenwidth(), self.master.winfo_screenheight(), scale)
        x = max((self.master.winfo_screenwidth() - width) // 2, 0)
        y = max((self.master.winfo_screenheight() - height) // 3, 0)
        self.master.geometry(f"{width}x{height}+{x}+{y}")
        self.master.minsize(min_w, min_h)

    def _register_themes(self):
        for mode, theme in THEMES.items():
            name = f"steel-{mode}"
            if name not in self.style.theme_names():
                # 第三个位置参数在 ttkbootstrap 1.x 为 themetype，2.x 为 mode
                self.style.register_theme(ThemeDefinition(name, theme["ttk"], mode))

    def _build_fonts(self):
        family = pick_font_family(UI_FONTS)
        mono = pick_font_family(MONO_FONTS)
        base = 13 if sys.platform == "darwin" else 10
        self.fonts = {
            "body": tkfont.Font(family=family, size=base),
            "body_bold": tkfont.Font(family=family, size=base, weight="bold"),
            "small": tkfont.Font(family=family, size=base - 2),
            "small_bold": tkfont.Font(family=family, size=base - 2, weight="bold"),
            "title": tkfont.Font(family=family, size=base + 8, weight="bold"),
            "h2": tkfont.Font(family=family, size=base + 1, weight="bold"),
            "brand": tkfont.Font(family=family, size=base + 3, weight="bold"),
            "metric": tkfont.Font(family=family, size=base + 10, weight="bold"),
            "nav": tkfont.Font(family=family, size=base + 1),
            "nav_icon": tkfont.Font(family=family, size=base + 2),
            "mono": tkfont.Font(family=mono, size=base - 1),
        }
        self.master.option_add("*Font", self.fonts["body"])

    def setup_variables(self):
        """初始化所有界面变量与运行状态"""
        self.device_choices, self.device_label = detect_devices()

        self.train_model = ttk.StringVar()
        self.train_data = ttk.StringVar(value=str(ROOT / "dataset.yaml") if (ROOT / "dataset.yaml").is_file() else "")
        self.train_epochs = ttk.StringVar(value="50")
        self.train_batch = ttk.StringVar(value="16")
        self.train_imgsz = ttk.StringVar(value="640")
        self.train_workers = ttk.StringVar(value="2")
        self.train_device = ttk.StringVar(value=DEVICE_AUTO)

        self.val_model = ttk.StringVar()
        self.val_data = ttk.StringVar(value=self.train_data.get())
        self.val_split = ttk.StringVar(value="val")
        self.val_device = ttk.StringVar(value=DEVICE_AUTO)

        self.predict_model = ttk.StringVar()
        self.predict_source = ttk.StringVar()
        self.predict_conf = ttk.DoubleVar(value=0.25)
        self.predict_device = ttk.StringVar(value=DEVICE_AUTO)

        self.batch_model = ttk.StringVar()
        self.batch_data = ttk.StringVar()
        self.batch_conf = ttk.DoubleVar(value=0.25)
        self.batch_device = ttk.StringVar(value=DEVICE_AUTO)

        self.video_model = ttk.StringVar()
        self.video_source = ttk.StringVar()
        self.video_camera = ttk.StringVar(value="0")
        self.video_conf = ttk.DoubleVar(value=0.25)
        self.video_device = ttk.StringVar(value=DEVICE_AUTO)

        # 选择任一模型后，自动填充其余为空的模型输入，减少重复操作
        self._model_vars = (self.predict_model, self.batch_model, self.video_model,
                            self.val_model)

        self.video_loop_running = False
        self.video_worker_active = False
        self.video_frame_pending = False
        self.training_active = False
        self.validation_active = False
        self._busy = set()
        self._processes = {}
        self._stop_requested = set()
        self._closing = False
        self._canvas_sources = {}
        self._train_run_dir = None
        self._train_epochs_total = 0
        self._train_started_at = None
        self._last_batch = None
        self._last_metrics = None
        self._toast_job = None

        matplotlib.rcParams["font.sans-serif"] = CJK_FONTS
        matplotlib.rcParams["axes.unicode_minus"] = False

    # ---------------------------------------------------------------- theming

    def paint(self, widget, **options):
        """Register palette-driven colors so they follow theme switches."""
        self._painted.append((widget, options))
        self._apply_paint(widget, options)
        return widget

    def _apply_paint(self, widget, options):
        try:
            widget.configure(**{key: self.palette[value] for key, value in options.items()})
        except tk.TclError:
            pass

    def apply_theme(self, mode):
        self.theme_mode = mode
        self.palette = THEMES[mode]
        theme = f"steel-{mode}"
        if self.style.theme_use() != theme:
            self.style.theme_use(theme)
        p = self.palette
        self.style.configure("Treeview", rowheight=int(self.fonts["body"].metrics("linespace") * 1.9),
                             font=self.fonts["body"], background=p["surface"],
                             fieldbackground=p["surface"], foreground=p["text"], borderwidth=0)
        self.style.configure("Treeview.Heading", font=self.fonts["small_bold"],
                             background=p["raised"], foreground=p["muted"], relief="flat")
        self.style.map("Treeview", background=[("selected", p["ttk"]["selectbg"])],
                       foreground=[("selected", p["ttk"]["selectfg"])])
        alive = []
        for widget, options in self._painted:
            if widget.winfo_exists():
                self._apply_paint(widget, options)
                alive.append((widget, options))
        self._painted = alive
        self._select_nav(getattr(self, "current_page", "predict"))
        self.theme_button.configure(text="浅色模式" if mode == "dark" else "深色模式")
        for canvas in (self.predict_canvas, self.video_canvas):
            self._render_canvas(canvas)
        self._restyle_charts()

    def toggle_theme(self):
        self.apply_theme("light" if self.theme_mode == "dark" else "dark")

    # ------------------------------------------------------------- components

    def label(self, parent, text="", font="body", color="text", bg="surface", **kwargs):
        widget = tk.Label(parent, text=text, font=self.fonts[font], anchor="w", **kwargs)
        return self.paint(widget, fg=color, bg=bg)

    def frame(self, parent, bg="surface", **kwargs):
        return self.paint(tk.Frame(parent, **kwargs), bg=bg)

    def card(self, parent, title=None, subtitle=None, padding=16):
        outer = tk.Frame(parent, highlightthickness=1, bd=0)
        self.paint(outer, bg="surface", highlightbackground="border", highlightcolor="border")
        body = self.frame(outer)
        body.pack(fill="both", expand=True, padx=padding, pady=padding)
        if title:
            header = self.frame(body)
            header.pack(fill="x", pady=(0, 12))
            self.label(header, title, font="h2").pack(side="left")
            if subtitle:
                self.label(header, subtitle, font="small", color="muted").pack(side="right")
            body.header = header
        return outer, body

    def field(self, parent, text, variable, kind="file", filetypes=None, on_pick=None):
        """Label above an entry with a browse button."""
        self.label(parent, text, font="small", color="muted").pack(fill="x", pady=(0, 4))
        row = self.frame(parent)
        row.pack(fill="x", pady=(0, 12))
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

        def browse():
            if kind == "dir":
                path = filedialog.askdirectory(parent=self.master)
            else:
                path = filedialog.askopenfilename(parent=self.master, filetypes=filetypes or [])
            if path:
                variable.set(path)
                if on_pick:
                    on_pick(path)

        ttk.Button(row, text="浏览", command=browse, bootstyle="secondary",
                   padding=(10, 4)).pack(side="left", padx=(8, 0))
        return row

    def model_field(self, parent, variable):
        def share(path):
            for other in self._model_vars:
                if not other.get():
                    other.set(path)
        return self.field(parent, "模型权重 (.pt)", variable, filetypes=MODEL_TYPES, on_pick=share)

    def choice(self, parent, text, variable, values, width=10):
        box = self.frame(parent)
        self.label(box, text, font="small", color="muted").pack(fill="x", pady=(0, 4))
        # 用 Menubutton 代替 Combobox：Tk 9 在 macOS 上把下拉框弹出层改成原生菜单，
        # ttkbootstrap 1.x 切换主题时重设该弹出层会直接报错
        button = ttk.Menubutton(box, textvariable=variable, bootstyle="secondary", width=width,
                                padding=(10, 6))
        menu = tk.Menu(button, tearoff=False)
        for value in values:
            menu.add_radiobutton(label=value, value=value, variable=variable)
        button.configure(menu=menu)
        button.pack(fill="x")
        return box

    def number(self, parent, text, variable, low, high, step=1, width=8):
        box = self.frame(parent)
        self.label(box, text, font="small", color="muted").pack(fill="x", pady=(0, 4))
        ttk.Spinbox(box, from_=low, to=high, increment=step, textvariable=variable,
                    width=width).pack(fill="x")
        return box

    def conf_slider(self, parent, variable):
        header = self.frame(parent)
        header.pack(fill="x", pady=(0, 4))
        self.label(header, "置信度阈值", font="small", color="muted").pack(side="left")
        value = self.label(header, f"{variable.get():.2f}", font="small_bold", color="text")
        value.pack(side="right")

        def changed(raw):
            snapped = round(float(raw) / 0.05) * 0.05
            variable.set(round(min(max(snapped, 0.05), 0.95), 2))
            value.configure(text=f"{variable.get():.2f}")

        ttk.Scale(parent, from_=0.05, to=0.95, value=variable.get(), command=changed,
                  bootstyle="primary").pack(fill="x", pady=(0, 12))

    def two_columns(self, parent, left, right):
        row = self.frame(parent)
        row.pack(fill="x", pady=(0, 12))
        row.columnconfigure((0, 1), weight=1, uniform="pair")
        left(row).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        right(row).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        return row

    def device_choice(self, parent, variable):
        return lambda row: self.choice(row, "推理设备", variable, self.device_choices)

    def stat_tile(self, parent, caption, value="—"):
        outer = tk.Frame(parent, highlightthickness=1, bd=0)
        self.paint(outer, bg="raised", highlightbackground="border", highlightcolor="border")
        inner = self.frame(outer, bg="raised")
        inner.pack(fill="both", expand=True, padx=14, pady=10)
        self.label(inner, caption, font="small", color="muted", bg="raised").pack(fill="x")
        value_label = self.label(inner, value, font="metric", bg="raised")
        value_label.pack(fill="x")
        return outer, value_label

    def tile_row(self, parent, captions):
        row = self.frame(parent, bg="page")
        values = []
        for index, caption in enumerate(captions):
            row.columnconfigure(index, weight=1, uniform="tiles")
            tile, value = self.stat_tile(row, caption)
            tile.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 6, 0))
            values.append(value)
        return row, values

    def table(self, parent, columns, height=8):
        """columns: (key, heading, width, anchor)"""
        box = self.frame(parent)
        tree = ttk.Treeview(box, columns=[c[0] for c in columns], show="headings", height=height)
        for index, (key, heading, width, anchor) in enumerate(columns):
            tree.heading(key, text=heading, anchor=anchor)
            tree.column(key, width=width, minwidth=width, anchor=anchor, stretch=index == 0)
        scroll = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return box, tree

    def image_canvas(self, parent, placeholder):
        canvas = tk.Canvas(parent, highlightthickness=0, bd=0)
        self.paint(canvas, bg="canvas")
        canvas.placeholder = placeholder
        self._canvas_sources[canvas] = None
        canvas.bind("<Configure>", lambda _e: self._render_canvas(canvas))
        return canvas

    def chart(self, parent, height=2.6):
        figure = Figure(figsize=(4, height), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, canvas

    # -------------------------------------------------------------- main layout

    def create_ui(self):
        """主界面：左侧导航 + 右侧页面 + 可折叠日志 + 状态栏"""
        root = self.frame(self.master, bg="page")
        root.pack(fill="both", expand=True)

        self._build_sidebar(root)

        main = self.frame(root, bg="page")
        main.pack(side="left", fill="both", expand=True)
        self.main_area = main

        self._build_statusbar(main)
        self.setup_log_area(main)

        self.page_host = self.frame(main, bg="page")
        self.page_host.pack(fill="both", expand=True, padx=24, pady=(20, 12))
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)

        self.pages = {}
        builders = {
            "predict": self.setup_predict_tab, "batch": self.setup_batch_tab,
            "video": self.setup_video_tab, "train": self.setup_train_tab, "val": self.setup_val_tab,
        }
        for key, title, subtitle in PAGES:
            page = self.frame(self.page_host, bg="page")
            page.grid(row=0, column=0, sticky="nsew")
            header = self.frame(page, bg="page")
            header.pack(fill="x", pady=(0, 16))
            self.label(header, title, font="title", bg="page").pack(fill="x")
            self.label(header, subtitle, font="body", color="muted", bg="page").pack(fill="x", pady=(2, 0))
            body = self.frame(page, bg="page")
            body.pack(fill="both", expand=True)
            builders[key](body)
            self.pages[key] = page

        self._build_toast(main)
        self.show_page("predict")

    def _build_sidebar(self, parent):
        side = self.frame(parent, bg="sidebar", width=232)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        brand = self.frame(side, bg="sidebar")
        brand.pack(fill="x", padx=20, pady=(24, 28))
        mark = tk.Canvas(brand, width=34, height=34, highlightthickness=0, bd=0)
        self.paint(mark, bg="sidebar")
        mark.create_rectangle(2, 2, 32, 32, fill=THEMES["dark"]["ttk"]["primary"], outline="")
        mark.create_rectangle(9, 9, 25, 25, outline="#ffffff", width=2)
        mark.create_line(9, 17, 25, 17, fill="#ffffff", width=2)
        mark.pack(side="left", padx=(0, 12))
        names = self.frame(brand, bg="sidebar")
        names.pack(side="left", fill="x")
        self.label(names, APP_NAME, font="brand", bg="sidebar").pack(fill="x")
        self.label(names, f"YOLOv8 · v{APP_VERSION}", font="small", color="muted", bg="sidebar").pack(fill="x")

        self.label(side, "工作台", font="small_bold", color="subtle", bg="sidebar").pack(fill="x", padx=22, pady=(0, 6))
        icons = {"predict": "◧", "batch": "▦", "video": "▶", "train": "◆", "val": "✓"}
        self.nav_items = {}
        for key, title, _subtitle in PAGES:
            item = tk.Frame(side, cursor="hand2")
            item.pack(fill="x", padx=10, pady=1)
            bar = tk.Frame(item, width=3)
            bar.pack(side="left", fill="y")
            icon = tk.Label(item, text=icons[key], font=self.fonts["nav_icon"], width=2)
            icon.pack(side="left", padx=(12, 6), pady=9)
            text = tk.Label(item, text=title, font=self.fonts["nav"], anchor="w")
            text.pack(side="left", fill="x", expand=True)
            for widget in (item, icon, text):
                widget.bind("<Button-1>", lambda _e, k=key: self.show_page(k))
                widget.bind("<Enter>", lambda _e, k=key: self._hover_nav(k, True))
                widget.bind("<Leave>", lambda _e, k=key: self._hover_nav(k, False))
            self.nav_items[key] = (item, bar, icon, text)

        footer = self.frame(side, bg="sidebar")
        footer.pack(side="bottom", fill="x", padx=20, pady=20)
        self.label(footer, "计算设备", font="small_bold", color="subtle", bg="sidebar").pack(fill="x")
        device_row = self.frame(footer, bg="sidebar")
        device_row.pack(fill="x", pady=(4, 14))
        dot = tk.Canvas(device_row, width=10, height=10, highlightthickness=0, bd=0)
        self.paint(dot, bg="sidebar")
        accelerated = self.device_label != "CPU"
        dot.create_oval(1, 1, 9, 9, fill="#2fb67c" if accelerated else "#e9a23b", outline="")
        dot.pack(side="left", padx=(0, 8))
        self.label(device_row, self.device_label, font="small", bg="sidebar").pack(side="left", fill="x")
        self.theme_button = ttk.Button(footer, text="浅色模式", command=self.toggle_theme,
                                       bootstyle="secondary", padding=(10, 6))
        self.theme_button.pack(fill="x")

    def _hover_nav(self, key, inside):
        if key == getattr(self, "current_page", None):
            return
        color = self.palette["nav_hover"] if inside else self.palette["sidebar"]
        item, bar, icon, text = self.nav_items[key]
        for widget in (item, bar, icon, text):
            widget.configure(bg=color)

    def _select_nav(self, active):
        p = self.palette
        for key, (item, bar, icon, text) in self.nav_items.items():
            selected = key == active
            bg = p["nav_active"] if selected else p["sidebar"]
            for widget in (item, icon, text):
                widget.configure(bg=bg)
            bar.configure(bg=p["ttk"]["primary"] if selected else bg)
            icon.configure(fg=p["ttk"]["primary"] if selected else p["muted"])
            text.configure(fg=p["text"] if selected else p["muted"],
                           font=self.fonts["body_bold"] if selected else self.fonts["nav"])

    def show_page(self, key):
        self.current_page = key
        self.pages[key].tkraise()
        self._select_nav(key)

    def _build_statusbar(self, parent):
        bar = self.frame(parent, bg="sidebar")
        bar.pack(side="bottom", fill="x")
        inner = self.frame(bar, bg="sidebar")
        inner.pack(fill="x", padx=24, pady=8)
        self.status_dot = tk.Canvas(inner, width=10, height=10, highlightthickness=0, bd=0)
        self.paint(self.status_dot, bg="sidebar")
        self.status_dot.pack(side="left", padx=(0, 8))
        self.status_label = self.label(inner, "就绪", font="small", bg="sidebar")
        self.status_label.pack(side="left")
        self.log_toggle = ttk.Button(inner, text="显示日志", command=self.toggle_log,
                                     bootstyle="primary-link", padding=(6, 0))
        self.log_toggle.pack(side="right")
        self.label(inner, "Ultralytics YOLOv8 · NEU-DET", font="small", color="subtle",
                   bg="sidebar").pack(side="right", padx=12)
        self.update_status("就绪 · 先选择模型权重和输入", "secondary")

    def update_status(self, text, style="secondary"):
        """更新底部状态栏；style 取 secondary/primary/success/warning/danger/info"""
        colors = self.palette["ttk"]
        color = colors.get(style, self.palette["muted"]) if style != "secondary" else self.palette["muted"]
        self.status_label.configure(text=text)
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 9, 9, fill=color, outline="")

    def _build_toast(self, parent):
        self.toast = tk.Frame(parent, highlightthickness=1, bd=0)
        self.paint(self.toast, bg="raised", highlightbackground="border", highlightcolor="border")
        self.toast_accent = tk.Frame(self.toast, width=4)
        self.toast_accent.pack(side="left", fill="y")
        body = self.frame(self.toast, bg="raised")
        body.pack(side="left", fill="both", padx=14, pady=10)
        self.toast_title = self.label(body, "", font="body_bold", bg="raised")
        self.toast_title.pack(fill="x")
        self.toast_message = self.label(body, "", font="small", color="muted", bg="raised")
        self.toast_message.pack(fill="x")

    def show_toast(self, title, message, bootstyle="success"):
        """应用内通知，3 秒后自动消失（不依赖 ttkbootstrap 的 Toast 窗口实现）"""
        self.toast_accent.configure(bg=self.palette["ttk"].get(bootstyle, self.palette["ttk"]["primary"]))
        self.toast_title.configure(text=title)
        self.toast_message.configure(text=message)
        self.toast.place(relx=1.0, x=-24, y=20, anchor="ne")
        self.toast.lift()
        if self._toast_job:
            self.master.after_cancel(self._toast_job)
        self._toast_job = self.master.after(3000, self.toast.place_forget)

    # --------------------------------------------------------------- log area

    def setup_log_area(self, parent):
        """底部可折叠日志控制台"""
        self.log_frame = self.frame(parent, bg="canvas")
        header = self.frame(self.log_frame, bg="canvas")
        header.pack(fill="x", padx=24, pady=(8, 0))
        self.label(header, "运行日志", font="small_bold", color="muted", bg="canvas").pack(side="left")
        ttk.Button(header, text="清空", command=self.clear_log, bootstyle="primary-link",
                   padding=(6, 0)).pack(side="right")
        body = self.frame(self.log_frame, bg="canvas")
        body.pack(fill="both", expand=True, padx=24, pady=(4, 10))
        self.log_text = tk.Text(body, height=9, font=self.fonts["mono"], state="disabled",
                                relief="flat", bd=0, wrap="none", padx=4, pady=4)
        self.paint(self.log_text, bg="canvas", fg="muted", insertbackground="text")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for tag, key in (("INFO", "muted"), ("SYSTEM", "text")):
            self.log_text.tag_configure(tag, foreground=self.palette[key])
        self.log_text.tag_configure("WARNING", foreground=THEMES["dark"]["ttk"]["warning"])
        self.log_text.tag_configure("ERROR", foreground=THEMES["dark"]["ttk"]["danger"])
        self.log_visible = False

    def toggle_log(self):
        self.show_log(not self.log_visible)

    def show_log(self, visible=True):
        if visible and not self.log_visible:
            self.log_frame.pack(side="bottom", fill="x", before=self.page_host)
        elif not visible and self.log_visible:
            self.log_frame.pack_forget()
        self.log_visible = visible
        self.log_toggle.configure(text="隐藏日志" if visible else "显示日志")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")
        self.log("日志已清空", "SYSTEM")

    # ------------------------------------------------------------ predict page

    def setup_predict_tab(self, body):
        """单图检测：左侧设置与结果，右侧图像对比"""
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = self.frame(body, bg="page", width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.grid_propagate(False)
        left.pack_propagate(False)

        card, form = self.card(left, "检测设置")
        card.pack(fill="x")
        self.model_field(form, self.predict_model)
        self.field(form, "待检测图片", self.predict_source, filetypes=IMAGE_TYPES,
                   on_pick=self._preview_predict_source)
        self.conf_slider(form, self.predict_conf)
        self.choice(form, "推理设备", self.predict_device, self.device_choices).pack(fill="x", pady=(0, 16))
        self.predict_button = ttk.Button(form, text="开始检测", command=self.start_prediction,
                                         bootstyle="primary", padding=(10, 9))
        self.predict_button.pack(fill="x")

        card, result = self.card(left, "检测结果")
        card.pack(fill="both", expand=True, pady=(16, 0))
        tiles = self.frame(result)
        tiles.pack(fill="x", pady=(0, 12))
        tiles.columnconfigure((0, 1), weight=1, uniform="t")
        tile, self.predict_count = self.stat_tile(tiles, "缺陷数量")
        tile.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tile, self.predict_best = self.stat_tile(tiles, "最高置信度")
        tile.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        box, self.predict_table = self.table(result, (
            ("cls", "类别", 90, "w"), ("conf", "置信度", 70, "e"), ("size", "面积占比", 80, "e"),
        ), height=5)
        box.pack(fill="both", expand=True)
        self.predict_output = self.label(result, "", font="small", color="muted", wraplength=290)
        self.predict_output.pack(fill="x", pady=(10, 0))

        viewer_card, viewer = self.card(body, "图像")
        viewer_card.grid(row=0, column=1, sticky="nsew")
        switch = self.frame(viewer.header)
        switch.pack(side="right")
        self.predict_view = tk.StringVar(value="result")
        self.predict_view_buttons = {}
        for key, text in (("source", "原图"), ("result", "检测结果")):
            button = ttk.Radiobutton(switch, text=text, value=key, variable=self.predict_view,
                                     command=self._switch_predict_view, bootstyle="primary-toolbutton",
                                     padding=(12, 4))
            button.pack(side="left")
            self.predict_view_buttons[key] = button
        self.predict_canvas = self.image_canvas(viewer, "选择一张图片开始检测")
        self.predict_canvas.pack(fill="both", expand=True)
        self._predict_images = {"source": None, "result": None}

    def _preview_predict_source(self, path):
        self._predict_images = {"source": path, "result": None}
        self.predict_view.set("source")
        self.show_image_on_canvas(path, self.predict_canvas)
        self._reset_prediction_result()

    def _switch_predict_view(self):
        image = self._predict_images.get(self.predict_view.get())
        if image is not None:
            self.show_image_on_canvas(image, self.predict_canvas)

    def _reset_prediction_result(self):
        self.predict_count.configure(text="—")
        self.predict_best.configure(text="—")
        self.predict_table.delete(*self.predict_table.get_children())
        self.predict_output.configure(text="")

    def show_prediction_result(self, image_path, detections, save_dir):
        self._predict_images["result"] = image_path
        self.predict_view.set("result")
        self.show_image_on_canvas(image_path, self.predict_canvas)
        self.predict_table.delete(*self.predict_table.get_children())
        for det in sorted(detections, key=lambda d: d.conf or 0, reverse=True):
            self.predict_table.insert("", "end", values=(det.name, format_conf(det.conf), f"{det.area:.1%}"))
        confs = [d.conf for d in detections if d.conf is not None]
        self.predict_count.configure(text=str(len(detections)))
        self.predict_best.configure(text=format_conf(max(confs)) if confs else "—")
        message = f"结果已保存到 {save_dir}" if detections else f"未检出缺陷 · 结果已保存到 {save_dir}"
        self.predict_output.configure(text=message)

    # -------------------------------------------------------------- batch page

    def setup_batch_tab(self, body):
        """批量检测：统计卡片 + 图表 + 逐图明细"""
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = self.frame(body, bg="page", width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.pack_propagate(False)
        card, form = self.card(left, "批量设置")
        card.pack(fill="x")
        self.model_field(form, self.batch_model)
        self.field(form, "图片文件夹", self.batch_data, kind="dir")
        self.conf_slider(form, self.batch_conf)
        self.choice(form, "推理设备", self.batch_device, self.device_choices).pack(fill="x", pady=(0, 16))
        self.batch_button = ttk.Button(form, text="开始批量检测", command=self.start_batch_prediction,
                                       bootstyle="primary", padding=(10, 9))
        self.batch_button.pack(fill="x")
        self.batch_open = ttk.Button(form, text="打开输出目录", state="disabled", bootstyle="secondary",
                                     padding=(10, 7), command=lambda: self._open_output(self._last_batch_dir))
        self.batch_open.pack(fill="x", pady=(8, 0))
        self._last_batch_dir = None

        card, verdict = self.card(left, "结论")
        card.pack(fill="x", pady=(16, 0))
        self.batch_verdict = self.label(verdict, "运行批量检测后显示结论", color="muted", wraplength=290)
        self.batch_verdict.pack(fill="x")

        right = self.frame(body, bg="page")
        right.grid(row=0, column=1, sticky="nsew")
        tiles, values = self.tile_row(right, ("处理图片", "含缺陷图片", "缺陷总数", "平均置信度"))
        tiles.pack(fill="x", pady=(0, 16))
        (self.batch_images, self.batch_defect_images, self.batch_total, self.batch_mean) = values

        charts = self.frame(right, bg="page")
        charts.pack(fill="x", pady=(0, 16))
        charts.columnconfigure((0, 1), weight=1, uniform="charts")
        card, inner = self.card(charts, "缺陷类别分布", padding=12)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.batch_class_fig, self.batch_class_canvas = self.chart(inner)
        card, inner = self.card(charts, "置信度分布", padding=12)
        card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.batch_conf_fig, self.batch_conf_canvas = self.chart(inner)

        card, inner = self.card(right, "逐图明细", subtitle="双击打开结果图")
        card.pack(fill="both", expand=True)
        box, self.batch_table = self.table(inner, (
            ("name", "文件", 220, "w"), ("count", "缺陷数", 70, "e"),
            ("top", "主要类别", 100, "w"), ("best", "最高置信度", 100, "e"),
        ))
        box.pack(fill="both", expand=True)
        self.batch_table.bind("<Double-1>", self._open_batch_row)

    def show_batch_summary(self, summary, output_dir):
        self._last_batch = summary
        self._last_batch_dir = output_dir
        self.batch_open.configure(state="normal" if Path(output_dir).is_dir() else "disabled")
        self.batch_images.configure(text=str(summary.image_count))
        self.batch_defect_images.configure(text=str(summary.defect_images))
        self.batch_total.configure(text=str(summary.total_defects))
        self.batch_mean.configure(text=format_conf(summary.mean_conf))
        verdict, level = batch_verdict(summary)
        self.batch_verdict.configure(text=verdict, fg=self.palette["ttk"].get(level, self.palette["text"]))
        self.batch_table.delete(*self.batch_table.get_children())
        for image in sorted(summary.images, key=lambda i: len(i.detections), reverse=True):
            self.batch_table.insert("", "end", values=(
                image.name, len(image.detections), image.top_class, format_conf(image.max_conf)))
        self._draw_batch_charts()

    def _open_batch_row(self, _event):
        selection = self.batch_table.selection()
        if not selection or self._last_batch_dir is None:
            return
        name = self.batch_table.item(selection[0], "values")[0]
        path = find_result_image(Path(self._last_batch_dir), Path(name))
        if path is not None:
            self._open_output(path)

    def _open_output(self, path):
        if path is None:
            return
        try:
            open_path(path)
        except OSError as exc:
            self.log(f"无法打开 {path}: {exc}", "WARNING")

    def _style_axes(self, figure, ax):
        p = self.palette
        figure.set_facecolor(p["surface"])
        ax.set_facecolor(p["surface"])
        ax.tick_params(colors=p["muted"], labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(p["border"])
        ax.grid(True, axis="x" if ax.get_label() == "barh" else "y", color=p["border"], linewidth=0.6)
        ax.set_axisbelow(True)

    def _empty_chart(self, figure, canvas, text):
        figure.clear()
        figure.set_facecolor(self.palette["surface"])
        figure.text(0.5, 0.5, text, ha="center", va="center", color=self.palette["subtle"], fontsize=9)
        canvas.draw_idle()

    def _draw_batch_charts(self):
        summary = self._last_batch
        if summary is None or not summary.class_counts:
            self._empty_chart(self.batch_class_fig, self.batch_class_canvas, "暂无缺陷数据")
        else:
            fig = self.batch_class_fig
            fig.clear()
            ax = fig.add_subplot(label="barh")
            names = list(summary.class_counts)[::-1]
            counts = [summary.class_counts[name] for name in names]
            ax.barh(names, counts, color=CHART_COLORS[0], height=0.6)
            self._style_axes(fig, ax)
            ax.tick_params(axis="y", colors=self.palette["text"], labelsize=9)
            fig.subplots_adjust(left=0.22, right=0.96, top=0.95, bottom=0.12)
            self.batch_class_canvas.draw_idle()

        if summary is None or not summary.confidences:
            self._empty_chart(self.batch_conf_fig, self.batch_conf_canvas, "暂无置信度数据")
        else:
            fig = self.batch_conf_fig
            fig.clear()
            ax = fig.add_subplot()
            ax.hist(summary.confidences, bins=[i / 20 for i in range(21)], color=CHART_COLORS[1],
                    edgecolor=self.palette["surface"], linewidth=1)
            ax.set_xlim(0, 1)
            self._style_axes(fig, ax)
            fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.14)
            self.batch_conf_canvas.draw_idle()

    # -------------------------------------------------------------- video page

    def setup_video_tab(self, body):
        """视频检测：左侧控制与实时统计，右侧画面"""
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = self.frame(body, bg="page", width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.pack_propagate(False)
        card, form = self.card(left, "视频设置")
        card.pack(fill="x")
        self.model_field(form, self.video_model)
        self.field(form, "视频文件", self.video_source, filetypes=VIDEO_TYPES,
                   on_pick=self._preview_video)
        self.conf_slider(form, self.video_conf)
        self.two_columns(
            form,
            lambda row: self.number(row, "摄像头编号", self.video_camera, 0, 9),
            self.device_choice(form, self.video_device),
        )
        buttons = self.frame(form)
        buttons.pack(fill="x", pady=(4, 0))
        buttons.columnconfigure((0, 1), weight=1, uniform="b")
        self.video_start = ttk.Button(buttons, text="检测视频", command=self.start_video_prediction,
                                      bootstyle="primary", padding=(10, 9))
        self.video_start.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.video_cam = ttk.Button(buttons, text="打开摄像头", command=self.start_camera_prediction,
                                    bootstyle="secondary", padding=(10, 9))
        self.video_cam.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.video_stop = ttk.Button(form, text="停止", command=self.stop_video_prediction,
                                     bootstyle="danger-outline", padding=(10, 7), state="disabled")
        self.video_stop.pack(fill="x", pady=(8, 0))

        card, stats = self.card(left, "实时统计")
        card.pack(fill="x", pady=(16, 0))
        tiles = self.frame(stats)
        tiles.pack(fill="x")
        tiles.columnconfigure((0, 1, 2), weight=1, uniform="v")
        self.video_stat_labels = {}
        for index, (key, caption) in enumerate((("fps", "FPS"), ("boxes", "当前缺陷"), ("frames", "已处理帧"))):
            tile, value = self.stat_tile(tiles, caption)
            tile.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 6, 0))
            self.video_stat_labels[key] = value
        self.video_status = self.label(stats, "等待开始", font="small", color="muted")
        self.video_status.pack(fill="x", pady=(12, 0))

        viewer_card, viewer = self.card(body, "画面")
        viewer_card.grid(row=0, column=1, sticky="nsew")
        self.video_canvas = self.image_canvas(viewer, "选择视频文件或打开摄像头")
        self.video_canvas.pack(fill="both", expand=True)

    def _preview_video(self, path):
        """选择视频后显示第一帧预览"""
        cap = cv2.VideoCapture(path)
        try:
            ret, frame = cap.read()
            if ret:
                self.show_image_on_canvas(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                          self.video_canvas)
                self.video_status.configure(text="视频已加载，点击「检测视频」开始")
            else:
                self.log(f"无法读取视频首帧: {path}", "WARNING")
        finally:
            cap.release()

    def on_video_started(self):
        self.video_start.configure(state="disabled")
        self.video_cam.configure(state="disabled")
        self.video_stop.configure(state="normal")
        self.video_status.configure(text="推理中…", fg=self.palette["ttk"]["primary"])
        for value in self.video_stat_labels.values():
            value.configure(text="—")

    def on_video_finished(self, error):
        self.video_start.configure(state="normal")
        self.video_cam.configure(state="normal")
        self.video_stop.configure(state="disabled")
        if error is not None:
            self.video_status.configure(text=f"失败：{error}", fg=self.palette["ttk"]["danger"])
        else:
            self.video_status.configure(text="已停止", fg=self.palette["muted"])

    def update_video_stats(self, stats):
        self.video_stat_labels["fps"].configure(text=f"{stats['fps']:.1f}")
        self.video_stat_labels["boxes"].configure(text=str(stats["boxes"]))
        self.video_stat_labels["frames"].configure(text=str(stats["frames"]))

    # -------------------------------------------------------------- train page

    def setup_train_tab(self, body):
        """训练：参数表单 + 进度 + 实时曲线"""
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = self.frame(body, bg="page", width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.pack_propagate(False)
        card, form = self.card(left, "训练参数")
        card.pack(fill="x")
        self.field(form, "预训练权重 (.pt)", self.train_model, filetypes=MODEL_TYPES)
        self.field(form, "数据集配置 (.yaml)", self.train_data, filetypes=YAML_TYPES)
        self.two_columns(form, lambda row: self.number(row, "训练轮数", self.train_epochs, 1, 3000),
                         lambda row: self.number(row, "批大小", self.train_batch, 1, 256))
        self.two_columns(form, lambda row: self.choice(row, "图像尺寸", self.train_imgsz,
                                                       ["320", "416", "512", "640", "800", "1024"]),
                         lambda row: self.number(row, "数据加载进程", self.train_workers, 0, 16))
        self.choice(form, "训练设备", self.train_device, self.device_choices).pack(fill="x", pady=(0, 16))
        buttons = self.frame(form)
        buttons.pack(fill="x")
        buttons.columnconfigure(0, weight=2)
        buttons.columnconfigure(1, weight=1)
        self.train_button = ttk.Button(buttons, text="开始训练", command=self.start_training,
                                       bootstyle="primary", padding=(10, 9))
        self.train_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.train_stop = ttk.Button(buttons, text="停止", command=lambda: self.stop_task("train"),
                                     bootstyle="danger-outline", padding=(10, 9), state="disabled")
        self.train_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        card, tips = self.card(left, "提示")
        card.pack(fill="x", pady=(16, 0))
        for tip in ("首次训练建议用 yolov8n.pt 快速跑通流程",
                    "Windows 出现多进程报错时，把数据加载进程设为 0",
                    "显存不足时降低批大小；Apple 芯片可选 mps",
                    "每次训练写入独立目录，不会覆盖旧结果"):
            self.label(tips, f"·  {tip}", font="small", color="muted", wraplength=290,
                       justify="left").pack(fill="x", pady=2)

        right = self.frame(body, bg="page")
        right.grid(row=0, column=1, sticky="nsew")
        card, progress = self.card(right, "训练进度")
        card.pack(fill="x", pady=(0, 16))
        top = self.frame(progress)
        top.pack(fill="x")
        self.train_epoch_label = self.label(top, "尚未开始", font="metric")
        self.train_epoch_label.pack(side="left")
        self.train_time_label = self.label(top, "", font="small", color="muted")
        self.train_time_label.pack(side="right", anchor="s")
        self.train_progress = ttk.Progressbar(progress, mode="determinate", maximum=100,
                                              bootstyle="primary")
        self.train_progress.pack(fill="x", pady=(10, 10))
        footer = self.frame(progress)
        footer.pack(fill="x")
        self.train_dir_label = self.label(footer, "", font="small", color="muted")
        self.train_dir_label.pack(side="left", fill="x", expand=True)
        self.train_open = ttk.Button(footer, text="打开目录", bootstyle="primary-link", padding=(6, 0),
                                     state="disabled", command=lambda: self._open_output(self._train_run_dir))
        self.train_open.pack(side="right")

        tiles, values = self.tile_row(right, ("Precision", "Recall", "mAP50", "mAP50-95"))
        tiles.pack(fill="x", pady=(0, 16))
        self.train_metric_labels = values

        card, inner = self.card(right, "训练曲线", subtitle="每个 epoch 结束后更新")
        card.pack(fill="both", expand=True)
        self.train_fig, self.train_canvas = self.chart(inner, height=3.2)
        self._draw_train_chart([])

    def on_training_started(self, run_dir, epochs):
        self._train_run_dir = run_dir
        self._train_epochs_total = epochs
        self._train_started_at = time.monotonic()
        self.train_button.configure(state="disabled")
        self.train_stop.configure(state="normal")
        self.train_open.configure(state="disabled")
        self.train_epoch_label.configure(text=f"准备中 · 0 / {epochs}")
        self.train_progress.configure(value=0)
        self.train_dir_label.configure(text=str(run_dir))
        for value in self.train_metric_labels:
            value.configure(text="—")
        self._draw_train_chart([])
        self._poll_training()

    def on_training_finished(self, returncode, run_dir):
        self.train_button.configure(state="normal")
        self.train_stop.configure(state="disabled")
        self._refresh_training(final=True)
        if Path(run_dir).is_dir():
            self.train_open.configure(state="normal")
        if returncode == 0:
            self.train_progress.configure(value=100)

    def _poll_training(self):
        if not self.training_active:
            return
        self._refresh_training()
        self.master.after(2000, self._poll_training)

    def _refresh_training(self, final=False):
        if self._train_run_dir is None:
            return
        rows = read_training_progress(self._train_run_dir / "results.csv")
        total = self._train_epochs_total or 1
        done = int(rows[-1]["epoch"]) if rows else 0  # results.csv 的 epoch 从 1 开始
        elapsed = time.monotonic() - (self._train_started_at or time.monotonic())
        if rows:
            last = rows[-1]
            keys = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
            for label, key in zip(self.train_metric_labels, keys):
                if key in last:
                    label.configure(text=f"{last[key]:.3f}")
        state = "已结束" if final else "训练中"
        self.train_epoch_label.configure(text=f"{state} · {min(done, total)} / {total}")
        self.train_progress.configure(value=min(done / total * 100, 100))
        text = f"已用 {self._format_duration(elapsed)}"
        if done and not final and done < total:
            text += f" · 预计剩余 {self._format_duration(elapsed / done * (total - done))}"
        self.train_time_label.configure(text=text)
        self._draw_train_chart(rows)

    @staticmethod
    def _format_duration(seconds):
        seconds = int(seconds)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    def _draw_train_chart(self, rows):
        self._train_rows = rows
        if not rows:
            self._empty_chart(self.train_fig, self.train_canvas, "开始训练后，这里会显示 mAP 与损失曲线")
            return
        fig = self.train_fig
        fig.clear()
        epochs = [row["epoch"] for row in rows]
        ax = fig.add_subplot(1, 2, 1)
        for key, name, color in (("metrics/mAP50(B)", "mAP50", CHART_COLORS[0]),
                                 ("metrics/mAP50-95(B)", "mAP50-95", CHART_COLORS[1])):
            ax.plot(epochs, [row.get(key, 0) for row in rows], label=name, color=color, linewidth=2)
        ax.set_title("验证 mAP", color=self.palette["text"], fontsize=10)
        self._style_axes(fig, ax)
        ax.legend(fontsize=8, frameon=False, labelcolor=self.palette["muted"])
        ax2 = fig.add_subplot(1, 2, 2)
        for key, name, color in (("train/box_loss", "box", CHART_COLORS[2]),
                                 ("train/cls_loss", "cls", CHART_COLORS[3]),
                                 ("train/dfl_loss", "dfl", CHART_COLORS[4])):
            ax2.plot(epochs, [row.get(key, 0) for row in rows], label=name, color=color, linewidth=2)
        ax2.set_title("训练损失", color=self.palette["text"], fontsize=10)
        self._style_axes(fig, ax2)
        ax2.legend(fontsize=8, frameon=False, labelcolor=self.palette["muted"])
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.22)
        self.train_canvas.draw_idle()

    # ---------------------------------------------------------------- val page

    def setup_val_tab(self, body):
        """验证：整体指标卡片 + 各类别明细"""
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = self.frame(body, bg="page", width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.pack_propagate(False)
        card, form = self.card(left, "验证设置")
        card.pack(fill="x")
        self.model_field(form, self.val_model)
        self.field(form, "数据集配置 (.yaml)", self.val_data, filetypes=YAML_TYPES)
        self.two_columns(form, lambda row: self.choice(row, "数据划分", self.val_split, ["val", "test"]),
                         self.device_choice(form, self.val_device))
        self.val_button = ttk.Button(form, text="开始验证", command=self.start_validation,
                                     bootstyle="primary", padding=(10, 9))
        self.val_button.pack(fill="x", pady=(4, 0))

        card, notes = self.card(left, "指标说明")
        card.pack(fill="x", pady=(16, 0))
        for line in ("Precision：检出结果中正确的比例",
                     "Recall：真实缺陷被检出的比例",
                     "mAP50：IoU=0.5 时的平均精度",
                     "mAP50-95：IoU 0.5~0.95 的平均精度"):
            self.label(notes, line, font="small", color="muted", wraplength=290).pack(fill="x", pady=2)

        right = self.frame(body, bg="page")
        right.grid(row=0, column=1, sticky="nsew")
        tiles, values = self.tile_row(right, ("Precision", "Recall", "mAP50", "mAP50-95"))
        tiles.pack(fill="x", pady=(0, 16))
        self.val_metric_labels = values
        card, inner = self.card(right, "各类别结果")
        card.pack(fill="both", expand=True)
        self.val_state = self.label(inner.header, "", font="small", color="muted")
        self.val_state.pack(side="right")
        box, self.val_table = self.table(inner, (
            ("name", "类别", 160, "w"), ("p", "Precision", 100, "e"), ("r", "Recall", 100, "e"),
            ("m50", "mAP50", 100, "e"), ("m", "mAP50-95", 100, "e"),
        ))
        box.pack(fill="both", expand=True)

    def on_validation_started(self):
        self.val_button.configure(state="disabled")
        self.val_state.configure(text="验证中…")
        self.val_table.delete(*self.val_table.get_children())
        for value in self.val_metric_labels:
            value.configure(text="—")

    def on_validation_finished(self, returncode):
        self.val_button.configure(state="normal")
        self.val_state.configure(text="完成" if returncode == 0 else "失败，请查看日志")

    def show_validation_metrics(self, metrics):
        self._last_metrics = metrics
        for label, key in zip(self.val_metric_labels, ("precision", "recall", "map50", "map")):
            label.configure(text=f"{metrics.get(key, 0):.3f}")
        self.val_table.delete(*self.val_table.get_children())
        for row in metrics.get("classes", []):
            zh = CLASS_MAP.get(row.get("id"))
            name = f"{row['name']}（{zh}）" if zh else row["name"]
            self.val_table.insert("", "end", values=(
                name, f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                f"{row['map50']:.3f}", f"{row['map']:.3f}"))

    def _restyle_charts(self):
        self._draw_batch_charts()
        self._draw_train_chart(getattr(self, "_train_rows", []))

    # ------------------------------------------------------------ image canvas

    def show_image_on_canvas(self, img_path_or_pil, canvas, resample=Image.Resampling.LANCZOS):
        try:
            if isinstance(img_path_or_pil, (str, Path)):
                # 读取后立即关闭文件句柄，避免结果文件被长期占用
                with Image.open(img_path_or_pil) as opened:
                    pil_img = ImageOps.exif_transpose(opened)
                    pil_img.load()
            else:
                pil_img = img_path_or_pil
            self._canvas_sources[canvas] = (pil_img, resample)
            self._render_canvas(canvas)
        except Exception as e:
            self.log(f"显示图片错误: {e}", "ERROR")

    def _render_canvas(self, canvas):
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 10 or height < 10:
            return
        canvas.delete("all")
        source = self._canvas_sources.get(canvas)
        if source is None:
            canvas.create_text(width // 2, height // 2, text=canvas.placeholder,
                               font=self.fonts["body"], fill=self.palette["subtle"])
            return
        pil_img, resample = source
        ratio = min(width / pil_img.width, height / pil_img.height)
        size = (max(1, int(pil_img.width * ratio)), max(1, int(pil_img.height * ratio)))
        tk_img = ImageTk.PhotoImage(pil_img.resize(size, resample))
        canvas.create_image(width // 2, height // 2, anchor="center", image=tk_img)
        canvas.image = tk_img

    # ------------------------------------------------------------------ logging

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.log_queue.put((full_msg, level))
        print(full_msg)

    def process_log_queue(self):
        # 每个周期批量写入，并裁剪旧日志，避免训练输出刷屏时界面卡顿
        entries = []
        while not self.log_queue.empty():
            entries.append(self.log_queue.get())
        if entries:
            self.log_text.configure(state="normal")
            for msg, level in entries:
                self.log_text.insert(tk.END, msg + "\n", level)
            line_count = int(self.log_text.index("end-1c").split(".")[0])
            if line_count > MAX_LOG_LINES:
                self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES + 1}.0")
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")
        while not self.ui_queue.empty():
            func, args = self.ui_queue.get()
            try:
                func(*args)
            except Exception as e:
                self.log(f"界面更新失败: {e}", "ERROR")
        if not self._closing:
            # 视频推理时更频繁地取帧，其余时间 100ms 足够
            self.master.after(30 if self.video_worker_active else 100, self.process_log_queue)

    def show_validation_error(self, message):
        """Show an actionable local setup error without starting background work."""
        self.log(message, "WARNING")
        messagebox.showerror("输入未就绪", message, parent=self.master)

    # ------------------------------------------------------------ task runners

    def _post(self, func, *args):
        """Hand func to the Tk main thread (safe from worker threads); ignored once closing."""
        if not self._closing:
            self.ui_queue.put((func, args))

    def run_subprocess(self, cmd, log_callback=None, finish_callback=None, task=None):
        def thread_target():
            returncode = 1
            try:
                # 子进程无缓冲输出，日志才能实时显示；工作目录固定为仓库根目录
                env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                with subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env,
                    universal_newlines=True, encoding="utf-8", errors="replace", bufsize=1,
                    **subprocess_options(),
                ) as process:
                    if task:
                        self._processes[task] = process
                    for line in process.stdout:
                        msg = line.strip()
                        if not msg:
                            continue
                        if not is_progress_noise(msg):
                            self.log(msg)
                        if log_callback:
                            self._post(log_callback, msg)
                    returncode = process.wait()
                if returncode and task not in self._stop_requested:
                    self.log(f"进程退出码: {returncode}。请查看上方错误日志。", "ERROR")
            except Exception as e:
                self.log(f"进程异常: {str(e)}", "ERROR")
            finally:
                if task:
                    self._processes.pop(task, None)
                if finish_callback:
                    self._post(finish_callback, returncode)

        threading.Thread(target=thread_target, daemon=True).start()

    def stop_task(self, task):
        process = self._processes.get(task)
        if process is None or process.poll() is not None:
            return
        self._stop_requested.add(task)
        self.log("正在停止任务…", "WARNING")
        stop_process_tree(process)

    def _task_failed(self, message):
        self.update_status(message, "danger")
        self.show_log(True)

    # --- 训练逻辑 ---
    def start_training(self):
        if self.training_active:
            self.log("训练正在进行中，请等待当前任务结束", "WARNING")
            return
        epochs = parse_positive_int(self.train_epochs.get())
        batch = parse_positive_int(self.train_batch.get())
        workers = parse_positive_int(self.train_workers.get(), allow_zero=True)
        for message in (
            validate_model_path(self.train_model.get()),
            validate_dataset_config(self.train_data.get()),
            None if epochs else "训练轮数需为正整数。",
            None if batch else "批大小需为正整数。",
            None if workers is not None else "数据加载进程需为 0 或正整数。",
        ):
            if message:
                self.show_validation_error(message)
                return
        self.training_active = True
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = RUNS_DIR / run_name
        cmd = [PYTHON, str(ROOT / "train.py"),
               "--model", absolute_path(self.train_model.get()),
               "--data", absolute_path(self.train_data.get()),
               "--epochs", str(epochs), "--batch", str(batch),
               "--imgsz", str(self.train_imgsz.get()), "--workers", str(workers),
               "--project", str(RUNS_DIR), "--name", run_name]
        device = device_arg(self.train_device.get())
        if device:
            cmd += ["--device", device]
        self.on_training_started(run_dir, epochs)
        self.update_status("训练进行中…", "primary")
        self.log("启动训练进程…", "SYSTEM")

        def on_finish(returncode):
            self.training_active = False
            stopped = "train" in self._stop_requested
            self._stop_requested.discard("train")
            self.on_training_finished(returncode, run_dir)
            if stopped:
                self.update_status("训练已停止", "warning")
                return
            if returncode:
                self._task_failed("训练失败，请查看日志")
                return
            self.update_status("训练完成", "success")
            self.show_toast("训练完成", f"权重已保存到 {run_dir / 'weights'}", bootstyle="success")

        self.run_subprocess(cmd, finish_callback=on_finish, task="train")

    # --- 验证逻辑 ---
    def start_validation(self):
        if self.validation_active:
            self.log("验证正在进行中，请等待当前任务结束", "WARNING")
            return
        for message in (
            validate_model_path(self.val_model.get()),
            validate_dataset_config(self.val_data.get()),
        ):
            if message:
                self.show_validation_error(message)
                return

        self.validation_active = True
        self.on_validation_started()
        self.update_status("验证进行中…", "primary")
        cmd = [PYTHON, str(ROOT / "val.py"),
               "--model", absolute_path(self.val_model.get()),
               "--data", absolute_path(self.val_data.get()),
               "--split", self.val_split.get()]
        device = device_arg(self.val_device.get())
        if device:
            cmd += ["--device", device]

        def on_line(msg):
            metrics = parse_metrics_line(msg)
            if metrics is not None:
                self.show_validation_metrics(metrics)

        def on_val_finish(returncode):
            self.validation_active = False
            self.on_validation_finished(returncode)
            if returncode:
                self._task_failed("验证失败，请查看日志")
                return
            self.update_status("验证完成", "success")
            self.show_toast("验证完成", "指标已更新", bootstyle="success")

        self.run_subprocess(cmd, log_callback=on_line, finish_callback=on_val_finish, task="val")

    # --- 单图预测逻辑 ---
    def start_prediction(self):
        if "predict" in self._busy:
            return
        conf = parse_confidence(self.predict_conf.get())
        for message in (
            validate_model_path(self.predict_model.get()),
            validate_file_source(self.predict_source.get(), "待检测图片"),
            None if conf else "置信度阈值需在 0 到 1 之间。",
        ):
            if message:
                self.show_validation_error(message)
                return
        self._busy.add("predict")
        self.update_status("检测中…", "primary")
        exp_name = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # 记录启动时的输入，避免检测期间用户修改输入框导致报告读错文件
        source = Path(absolute_path(self.predict_source.get()))
        cmd = [PYTHON, str(ROOT / "predict.py"),
               "--model", absolute_path(self.predict_model.get()),
               "--source", str(source), "--conf", f"{conf:g}",
               "--name", exp_name, "--project", str(RUNS_DIR)]
        device = device_arg(self.predict_device.get())
        if device:
            cmd += ["--device", device]

        def on_predict_finish(returncode):
            self._busy.discard("predict")
            if returncode:
                self._task_failed("单图检测失败，请查看日志")
                return
            save_dir = RUNS_DIR / exp_name
            res_path = find_result_image(save_dir, source)
            if res_path is None:
                self.log("未找到结果图片", "WARNING")
                self.update_status("未找到结果图片，请查看日志", "warning")
                return
            txt_path = save_dir / "labels" / f"{source.stem}.txt"
            detections = parse_label_file(txt_path) if txt_path.is_file() else []
            self.show_prediction_result(res_path, detections, save_dir)
            self.update_status(f"检测完成 · {len(detections)} 处缺陷", "success")
            self.show_toast("检测完成", f"发现 {len(detections)} 处缺陷", bootstyle="success")

        self.run_subprocess(cmd, finish_callback=on_predict_finish, task="predict")

    # --- 批量预测逻辑 ---
    def start_batch_prediction(self):
        if "batch" in self._busy:
            return
        conf = parse_confidence(self.batch_conf.get())
        for message in (
            validate_model_path(self.batch_model.get()),
            validate_directory_source(self.batch_data.get(), "待检测图片目录"),
            None if conf else "置信度阈值需在 0 到 1 之间。",
        ):
            if message:
                self.show_validation_error(message)
                return
        self._busy.add("batch")
        self.update_status("批量检测进行中…", "primary")
        exp_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source_dir = Path(absolute_path(self.batch_data.get()))
        cmd = [PYTHON, str(ROOT / "predict.py"),
               "--model", absolute_path(self.batch_model.get()),
               "--source", str(source_dir), "--conf", f"{conf:g}",
               "--name", exp_name, "--project", str(RUNS_DIR)]
        device = device_arg(self.batch_device.get())
        if device:
            cmd += ["--device", device]

        def on_batch_finish(returncode):
            self._busy.discard("batch")
            if returncode:
                self._task_failed("批量检测失败，请查看日志")
                return
            output_dir = RUNS_DIR / exp_name
            # 没有任何检出时 Ultralytics 不会创建 labels 目录，这是正常结果而非失败
            summary = summarize_batch(output_dir / "labels", source_dir)
            self.show_batch_summary(summary, output_dir)
            self.update_status(f"批量检测完成 · {summary.image_count} 张图片", "success")
            self.show_toast("批量检测完成", f"{summary.defect_images} 张图片检出缺陷", bootstyle="success")

        self.run_subprocess(cmd, finish_callback=on_batch_finish, task="batch")

    # --- 视频预测逻辑 ---
    def start_video_prediction(self):
        self.run_video_inference(source=self.video_source.get())

    def start_camera_prediction(self):
        self.run_video_inference(source=str(self.video_camera.get()).strip() or "0")

    def run_video_inference(self, source):
        if self.video_worker_active:
            return
        conf = parse_confidence(self.video_conf.get())
        for message in (
            validate_model_path(self.video_model.get()),
            validate_video_source(source),
            None if conf else "置信度阈值需在 0 到 1 之间。",
        ):
            if message:
                self.show_validation_error(message)
                return
        self.video_loop_running = True
        self.video_worker_active = True
        self.video_frame_pending = False
        self.on_video_started()
        self.update_status("视频推理进行中…", "primary")
        model_path = self.video_model.get()
        infer_kwargs = {"conf": conf, "verbose": False}
        device = device_arg(self.video_device.get())
        if device:
            infer_kwargs["device"] = device

        def video_thread():
            cap = None
            error = None
            frames = 0
            started = time.perf_counter()
            try:
                model = YOLO(model_path)
                cap = open_video_capture(source)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频源: {source}。请检查文件或摄像头权限。")
                while self.video_loop_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, **infer_kwargs)
                    frames += 1
                    # 界面尚未绘制完上一帧时丢弃当前帧，避免 after 队列积压导致延迟和内存增长
                    if self.video_frame_pending:
                        continue
                    res_plotted = results[0].plot()
                    img_pil = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                    stats = {"fps": frames / max(time.perf_counter() - started, 1e-6),
                             "frames": frames, "boxes": len(results[0].boxes)}
                    self.video_frame_pending = True
                    self._post(self.show_video_frame, img_pil, stats)
            except Exception as e:
                error = str(e)
                self.log(f"视频流错误: {error}", "ERROR")
            finally:
                if cap is not None:
                    cap.release()
                self.video_loop_running = False
                self.video_worker_active = False
                self._post(self.on_video_finished, error)
                if error is not None:
                    self._post(self.update_status, "视频检测失败，请查看日志", "danger")
                else:
                    self._post(self.update_status, f"视频处理结束 · 共 {frames} 帧", "success")

        threading.Thread(target=video_thread, daemon=True).start()

    def show_video_frame(self, img_pil, stats=None):
        self.video_frame_pending = False
        # 视频帧使用更快的双线性缩放，保证实时性
        self.show_image_on_canvas(img_pil, self.video_canvas, Image.Resampling.BILINEAR)
        if stats:
            self.update_video_stats(stats)

    def stop_video_prediction(self):
        if self.video_worker_active:
            self.video_loop_running = False
            self.video_status.configure(text="正在停止…")

    def on_close(self):
        running = [task for task, process in self._processes.items() if process.poll() is None]
        if running and not messagebox.askyesno(
                "任务仍在运行", "关闭窗口会终止正在运行的训练/验证/检测任务，确定退出吗？", parent=self.master):
            return
        self._closing = True
        self.video_loop_running = False
        for task in running:
            self._stop_requested.add(task)
            stop_process_tree(self._processes[task])
        self.master.destroy()


def enable_windows_dpi_awareness():
    """Render crisp text on scaled Windows displays instead of a blurry bitmap-stretched window."""
    if not is_windows():
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
    try:
        # 任务栏使用本应用自己的图标分组，而不是归到 python.exe 下
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("YfengJ.SteelDefectDetection")
    except (AttributeError, OSError):
        pass


def main():
    enable_windows_dpi_awareness()
    app = ttk.Window(themename="darkly")
    YOLOv8_GUI(app)
    app.mainloop()


if __name__ == "__main__":
    main()

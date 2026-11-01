"""Start the real Tk GUI once: builds every page in both themes, then closes cleanly.

Skipped where no display is available (for example a Linux CI runner without Xvfb).
"""

import os
import sys

import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("ttkbootstrap")

from ui import PAGES  # noqa: E402


@pytest.fixture
def app():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    import ttkbootstrap as ttk

    import ui

    ui.enable_windows_dpi_awareness()
    try:
        window = ttk.Window(themename="darkly")
    except tk.TclError as error:
        pytest.skip(f"Tk unavailable: {error}")
    gui = ui.YOLOv8_GUI(window)
    window.update()
    yield window, gui
    try:
        window.destroy()
    except tk.TclError:
        pass  # 测试里已经关闭


def test_every_page_builds_in_both_themes(app):
    window, gui = app
    for _ in range(2):
        for key, _title, _subtitle in PAGES:
            gui.show_page(key)
            window.update()
        gui.toggle_theme()
        window.update()
    assert set(gui.pages) == {"predict", "batch", "video", "train", "val"}


def test_window_fits_on_screen(app):
    window, _gui = app
    window.update_idletasks()
    assert window.winfo_width() <= window.winfo_screenwidth()
    assert window.winfo_height() <= window.winfo_screenheight()


def test_close_without_running_tasks_destroys_window(app):
    window, gui = app
    gui.on_close()
    with pytest.raises(tk.TclError):
        window.winfo_exists()

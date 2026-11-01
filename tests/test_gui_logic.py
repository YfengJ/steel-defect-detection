from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gui_logic
from gui_logic import (
    BatchSummary,
    Detection,
    ImageResult,
    absolute_path,
    batch_verdict,
    device_arg,
    find_result_image,
    is_progress_noise,
    parse_confidence,
    parse_label_file,
    parse_metrics_line,
    parse_positive_int,
    python_executable,
    read_training_progress,
    stop_process_tree,
    subprocess_options,
    summarize_batch,
    video_source,
    window_geometry,
)


def test_parse_label_file_reads_confidence_and_skips_bad_lines(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.1 0.91\n5 0.1 0.1 0.1 0.1\nbroken line\n9 a b c d\n", encoding="utf-8")

    detections = parse_label_file(label)

    assert [(d.cls_id, d.name, d.conf) for d in detections] == [(0, "龟裂", 0.91), (5, "划痕", None)]
    assert detections[0].area == pytest.approx(0.02)


def test_unknown_class_gets_readable_name(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("7 0.5 0.5 0.2 0.1 0.5\n", encoding="utf-8")
    assert parse_label_file(label)[0].name == "类别 7"


def test_find_result_image_keeps_original_suffix(tmp_path):
    (tmp_path / "plate.bmp").write_bytes(b"x")
    assert find_result_image(tmp_path, Path("/input/plate.bmp")) == tmp_path / "plate.bmp"
    (tmp_path / "other.jpg").write_bytes(b"x")
    assert find_result_image(tmp_path, Path("other.png")) == tmp_path / "other.jpg"
    assert find_result_image(tmp_path, Path("missing.png")) is None


def test_summarize_batch_counts_images_without_labels(tmp_path):
    source = tmp_path / "images"
    labels = tmp_path / "labels"
    source.mkdir()
    labels.mkdir()
    for name in ("a.jpg", "b.png", "c.bmp", "notes.txt"):
        (source / name).write_bytes(b"x")
    (labels / "a.txt").write_text("1 0.5 0.5 0.1 0.1 0.6\n1 0.2 0.2 0.1 0.1 0.8\n", encoding="utf-8")
    (labels / "c.txt").write_text("5 0.5 0.5 0.1 0.1 0.4\n", encoding="utf-8")

    summary = summarize_batch(labels, source)

    assert summary.image_count == 3
    assert summary.defect_images == 2
    assert summary.total_defects == 3
    assert summary.class_counts == {"夹杂": 2, "划痕": 1}
    assert summary.mean_conf == pytest.approx(0.6)
    a = next(image for image in summary.images if image.name == "a.jpg")
    assert a.top_class == "夹杂" and a.max_conf == 0.8


def test_summarize_batch_without_label_dir_is_empty_but_counts_sources(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    summary = summarize_batch(tmp_path / "missing", tmp_path)
    assert summary.image_count == 1
    assert summary.total_defects == 0
    assert batch_verdict(summary) == ("未检出缺陷", "success")


@pytest.mark.parametrize("confs, level", [([0.9, 0.85], "success"), ([0.3], "warning"), ([0.6], "info")])
def test_batch_verdict_levels(confs, level):
    detections = [Detection(0, "龟裂", 0.5, 0.5, 0.1, 0.1, c) for c in confs]
    summary = BatchSummary(image_count=1, images=[ImageResult("a.jpg", detections)])
    assert batch_verdict(summary)[1] == level


def test_batch_verdict_for_empty_folder():
    assert batch_verdict(BatchSummary())[1] == "warning"


def test_parse_metrics_line():
    assert parse_metrics_line('METRICS_JSON {"map50": 0.76}') == {"map50": 0.76}
    assert parse_metrics_line("METRICS_JSON {broken") is None
    assert parse_metrics_line("METRICS_JSON [1]") is None
    assert parse_metrics_line("mAP50: 0.7") is None


def test_read_training_progress_strips_padded_headers(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "                  epoch,         train/box_loss,       metrics/mAP50(B)\n"
        "                      1,                 1.5,                 0.1\n"
        "                      2,                 1.2,                 0.3\n",
        encoding="utf-8",
    )
    rows = read_training_progress(csv_path)
    assert [row["epoch"] for row in rows] == [1.0, 2.0]
    assert rows[-1]["metrics/mAP50(B)"] == pytest.approx(0.3)
    assert read_training_progress(tmp_path / "missing.csv") == []


def test_read_training_progress_keeps_only_latest_appended_run(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,metrics/mAP50(B)\n1,0.1\n2,0.2\n1,0.15\n2,0.3\n3,0.4\n", encoding="utf-8")
    rows = read_training_progress(csv_path)
    assert [(row["epoch"], row["metrics/mAP50(B)"]) for row in rows] == [(1, 0.15), (2, 0.3), (3, 0.4)]


@pytest.mark.parametrize("value, expected", [("0.25", 0.25), (1, 1.0), ("0", None), ("1.5", None), ("x", None)])
def test_parse_confidence(value, expected):
    assert parse_confidence(value) == expected


def test_parse_positive_int():
    assert parse_positive_int(" 16 ") == 16
    assert parse_positive_int("0") is None
    assert parse_positive_int("0", allow_zero=True) == 0
    assert parse_positive_int("-1", allow_zero=True) is None
    assert parse_positive_int("1.5") is None


def test_device_arg_maps_auto_to_none():
    assert device_arg(gui_logic.DEVICE_AUTO) is None
    assert device_arg("") is None
    assert device_arg("mps") == "mps"


def test_absolute_path_expands_relative_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert absolute_path(" weights/best.pt ") == str(tmp_path / "weights" / "best.pt")


def test_progress_noise_filter():
    assert is_progress_noise("  1/50  2.1G  1.2  45%|████▌     | 9/20 [00:03<00:04]")
    assert not is_progress_noise("  1/50  2.1G  1.2 100%|██████████| 20/20 [00:07<00:00]")
    assert not is_progress_noise("✅ 预测完成，耗时 1.20s")


def test_python_executable_prefers_console_interpreter_next_to_pythonw(tmp_path):
    (tmp_path / "pythonw.exe").write_bytes(b"")
    assert python_executable(str(tmp_path / "pythonw.exe")) == str(tmp_path / "pythonw.exe")
    (tmp_path / "python.exe").write_bytes(b"")
    assert python_executable(str(tmp_path / "pythonw.exe")) == str(tmp_path / "python.exe")
    assert python_executable("/usr/bin/python3") == "/usr/bin/python3"


def test_subprocess_options_hide_console_only_on_windows():
    assert subprocess_options("win32") == {"creationflags": gui_logic.CREATE_NO_WINDOW}
    assert subprocess_options("darwin") == {}
    assert subprocess_options("linux") == {}


def test_stop_process_tree_uses_taskkill_on_windows(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(gui_logic.subprocess, "run", run)
    process = Mock(pid=4321)
    process.poll.return_value = None

    stop_process_tree(process, "win32")

    assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    process.terminate.assert_not_called()


def test_stop_process_tree_falls_back_to_terminate(monkeypatch):
    monkeypatch.setattr(gui_logic.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=128)))
    process = Mock(pid=1)
    process.poll.return_value = None
    stop_process_tree(process, "win32")
    process.terminate.assert_called_once()

    posix = Mock()
    posix.poll.return_value = None
    stop_process_tree(posix, "darwin")
    posix.terminate.assert_called_once()

    finished = Mock()
    finished.poll.return_value = 0
    stop_process_tree(finished, "win32")
    finished.terminate.assert_not_called()


def test_video_source_turns_digits_into_camera_index():
    assert video_source(" 0 ") == 0
    assert video_source("1") == 1
    assert video_source(r"C:\videos\plate.mp4") == r"C:\videos\plate.mp4"


def test_open_video_capture_uses_directshow_for_windows_cameras(monkeypatch):
    import cv2

    capture = Mock()
    capture.isOpened.return_value = True
    factory = Mock(return_value=capture)
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    assert gui_logic.open_video_capture("0", "win32") is capture
    factory.assert_called_once_with(0, cv2.CAP_DSHOW)

    factory.reset_mock()
    gui_logic.open_video_capture("clip.mp4", "win32")
    factory.assert_called_once_with("clip.mp4")


def test_open_video_capture_falls_back_when_directshow_fails(monkeypatch):
    import cv2

    failed, fallback = Mock(), Mock()
    failed.isOpened.return_value = False
    factory = Mock(side_effect=[failed, fallback])
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    assert gui_logic.open_video_capture("0", "win32") is fallback
    failed.release.assert_called_once()
    assert factory.call_args.args == (0,)


@pytest.mark.parametrize("screen, scale, expected", [
    ((2560, 1440), 1.0, (1400, 900, 1180, 760)),
    ((1366, 768), 1.0, (1284, 675, 1180, 675)),
    ((1920, 1080), 1.5, (1804, 950, 1770, 950)),
])
def test_window_geometry_fits_screen_and_dpi(screen, scale, expected):
    assert window_geometry(*screen, scale=scale) == expected

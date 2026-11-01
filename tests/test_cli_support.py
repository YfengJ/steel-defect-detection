import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_use_utf8_output_survives_a_legacy_code_page():
    # 模拟 Windows 重定向输出时的 cp1252 编码：没有 use_utf8_output 时打印 emoji 会崩溃
    code = "from cli_support import use_utf8_output; use_utf8_output(); print('✅ 完成')"
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode("utf-8").strip() == "✅ 完成"


def test_cli_help_works_with_a_legacy_code_page():
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    result = subprocess.run([sys.executable, "predict.py", "--help"], cwd=REPO_ROOT, env=env,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

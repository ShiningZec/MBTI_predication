#!/usr/bin/env python3
"""
MBTI 性格预测系统 — 一键启动器
==============================
双击 launch.command（macOS）或终端运行:
  python launch.py
  conda run -n web python launch.py

自动完成:
  1. 检测/激活 conda web 环境
  2. 检查端口占用
  3. 启动 FastAPI 服务
  4. 打开浏览器访问前端页面
  5. Ctrl+C 停止服务
"""

import os
import sys
import time
import signal
import socket
import subprocess
import webbrowser
from pathlib import Path

# ---- 配置 ----
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
CONDA_ENV = "web"

# 项目根目录 (src/app/ → 上两级)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_MODULE = "src.app.api:app"


def find_conda_python(env_name: str = CONDA_ENV) -> str | None:
    """查找 conda 环境中可用的 Python。"""
    # 方法1: 直接尝试 conda run
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 方法2: 搜索常见 conda 路径
    for base in [Path.home() / "miniconda3", Path.home() / "anaconda3",
                 Path("/opt/miniconda3"), Path("/opt/anaconda3"),
                 Path("/usr/local/miniconda3")]:
        env_python = base / "envs" / env_name / "bin" / "python"
        if env_python.exists():
            return str(env_python)

    return None


def is_port_in_use(host: str, port: int) -> bool:
    """检查端口是否已被占用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            return False


def wait_for_server(host: str, port: int, timeout: float = 15.0) -> bool:
    """等待服务器就绪。"""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(host, port):
            return True
        time.sleep(0.3)
    return False


def open_browser():
    """打开默认浏览器。"""
    print(f"\n🌐 正在打开浏览器: {URL}")
    webbrowser.open(URL)


def main():
    os.chdir(str(PROJECT_ROOT))

    print("=" * 56)
    print("  🧠  MBTI 性格预测系统")
    print("=" * 56)
    print(f"  项目目录 : {PROJECT_ROOT}")
    print(f"  服务地址 : {URL}")
    print()

    # ---- 检查当前 Python 是否有 uvicorn ----
    try:
        import uvicorn
        print(f"  Python   : {sys.executable}")
    except ImportError:
        # 尝试切换到 conda web 环境
        print("  ⚠️  当前 Python 未安装 uvicorn，正在查找 conda web 环境...")
        conda_python = find_conda_python()
        if conda_python and conda_python != sys.executable:
            print(f"  ✅ 找到: {conda_python}")
            print(f"  正在重新启动...\n")
            # 用 conda 环境中的 Python 重新运行自己
            os.execv(conda_python, [conda_python, __file__] + sys.argv[1:])
        else:
            print("  ❌ 未找到 conda web 环境")
            print(f"\n  请先创建环境并安装依赖:")
            print(f"    conda create -n {CONDA_ENV} python=3.10 -y")
            print(f"    conda activate {CONDA_ENV}")
            print(f"    pip install fastapi uvicorn pydantic pyyaml")
            print()
            input("按回车键退出...")
            sys.exit(1)

    # ---- 检查端口 ----
    if is_port_in_use(HOST, PORT):
        print(f"⚠️  端口 {PORT} 已被占用 — 可能服务已在运行")
        print(f"   直接打开浏览器: {URL}")
        open_browser()
        print("\n按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 再见")
            return

    # ---- 启动服务（子进程方式，更可靠） ----
    print(f"🚀 正在启动服务...")
    print(f"   (按 Ctrl+C 停止服务)\n")

    # 使用当前 Python 的 uvicorn 模块启动
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", APP_MODULE,
         "--host", HOST, "--port", str(PORT),
         "--log-level", "info"],
        cwd=str(PROJECT_ROOT),
    )

    # ---- 等待就绪 ----
    print("⏳ 等待服务就绪...", end="", flush=True)
    if wait_for_server(HOST, PORT):
        print(" ✅")
        open_browser()
    else:
        print(" ❌")
        print(f"⚠️  服务启动超时，请手动打开: {URL}")

    # ---- 保持运行 ----
    print("\n✨ 服务运行中... 按 Ctrl+C 停止\n")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("👋 再见！")


if __name__ == "__main__":
    main()

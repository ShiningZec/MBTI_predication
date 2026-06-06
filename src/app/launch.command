#!/bin/bash
# ============================================================
# MBTI 性格预测系统 — macOS 一键启动
# ============================================================
# 双击此文件 → 自动启动服务 + 打开浏览器
# ============================================================

cd "$(dirname "$0")/../.."
PROJECT_DIR="$(pwd)"

echo "=========================================="
echo "  🧠  MBTI 性格预测系统"
echo "=========================================="
echo "  项目目录: $PROJECT_DIR"
echo ""

# 初始化 conda
__conda_setup="$("$HOME/miniconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda3/etc/profile.d/conda.sh"
    fi
fi

# 激活 web 环境
conda activate web 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  conda 环境 'web' 未找到"
    echo "   将使用默认 Python (需要已安装 fastapi/uvicorn)"
    echo ""
fi

# 启动
python "$PROJECT_DIR/src/app/launch.py"

# 保持终端窗口打开
echo ""
read -p "按回车键关闭窗口..."

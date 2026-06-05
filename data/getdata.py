"""
MBTI 数据集下载与解压脚本
========================
数据来源: Kaggle - MBTI Personality Types 500 Dataset
作者: Zeyad Khalid
URL: https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset

Prerequisites:
    pip install kagglehub
    # 或将 kaggle.json 放到 ~/.kaggle/ 目录下

Usage:
    python data/getdata.py                # 下载到默认 data/ 目录
    python data/getdata.py -o ./my_data   # 下载到指定目录
"""

import argparse
import shutil
import zipfile
from pathlib import Path

# ============================================================
# 配置
# ============================================================
DATASET_PATH = "zeyadkhalid/mbti-personality-types-500-dataset"
TARGET_NAME = "MBTI_500.csv"

# 原始数据集中可能的 CSV 文件名（不同版本可能不同）
CANDIDATE_NAMES = [
    "MBTI 500.csv",
    "MBTI_500.csv",
    "mbti_500.csv",
    "MBTI.csv",
    "mbti.csv",
    "posts.csv",
]


def get_output_dir(path: str | None = None) -> Path:
    """确定输出目录（默认为脚本同级 data/ 目录）。"""
    if path:
        out = Path(path).resolve()
    else:
        out = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_with_kagglehub(out_dir: Path) -> Path | None:
    """
    使用 kagglehub 下载数据集。
    kagglehub 不需要手动设置 API key（首次运行会引导认证）。
    返回下载目录路径。
    """
    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("[ERROR] 请先安装 kagglehub: pip install kagglehub")
        return None

    print(f"[下载] 正在从 Kaggle 下载 {DATASET_PATH} ...")
    download_dir = kagglehub.dataset_download(DATASET_PATH)
    print(f"[完成] 下载至: {download_dir}")
    return Path(download_dir)


def download_with_kaggle_api(out_dir: Path) -> Path | None:
    """
    使用 kaggle CLI API 下载数据集（备用方案）。
    需要先配置 ~/.kaggle/kaggle.json。
    """
    import subprocess
    import sys

    print(f"[下载] 使用 Kaggle API 下载 {DATASET_PATH} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", DATASET_PATH,
            "-p", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] Kaggle API 下载失败:\n{result.stderr}")
        return None

    # 查找下载的 ZIP 文件
    zip_files = list(out_dir.glob("*.zip"))
    if not zip_files:
        print("[ERROR] 未找到下载的 ZIP 文件")
        return None

    zip_path = zip_files[0]
    print(f"[完成] 下载至: {zip_path}")
    return zip_path


def find_and_extract(download_dir: Path, out_dir: Path) -> Path | None:
    """
    在下载目录中查找 CSV，支持 ZIP 解压和直接 CSV。
    返回最终 CSV 文件路径。
    """
    # 1. 如果是 ZIP 文件，先解压
    if download_dir.is_file() and download_dir.suffix == ".zip":
        print(f"[解压] {download_dir.name} → {out_dir}")
        with zipfile.ZipFile(download_dir, "r") as zf:
            zf.extractall(out_dir)
        # 在解压目录中继续查找
        search_dir = out_dir
    else:
        search_dir = download_dir

    # 2. 在目录中查找匹配的 CSV
    csv_files = list(search_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] 在 {search_dir} 中未找到 CSV 文件")
        return None

    # 按优先级匹配
    for name in CANDIDATE_NAMES:
        for f in csv_files:
            if f.name == name:
                return f

    # 退而求其次：取第一个 CSV（可能是未知文件名）
    print(f"[WARN] 未匹配已知文件名，使用: {csv_files[0].name}")
    return csv_files[0]


def rename_to_target(csv_path: Path, out_dir: Path) -> Path:
    """将 CSV 重命名为统一名称 MBTI_500.csv。"""
    target = out_dir / TARGET_NAME
    if csv_path.resolve() == target.resolve():
        print(f"[跳过] 已是目标文件名: {target}")
        return target

    if target.exists():
        backup = target.with_suffix(".csv.bak")
        print(f"[备份] {target.name} → {backup.name}")
        target.replace(backup)

    print(f"[重命名] {csv_path.name} → {TARGET_NAME}")
    shutil.move(str(csv_path), str(target))
    return target


def cleanup(temp_dir: Path, out_dir: Path):
    """清理临时文件（下载的 ZIP 等）。"""
    for item in out_dir.glob("*.zip"):
        item.unlink()
        print(f"[清理] 已删除: {item.name}")


def main(output: str | None = None):
    out_dir = get_output_dir(output)
    target_path = out_dir / TARGET_NAME

    # 如果目标文件已存在，询问是否跳过
    if target_path.exists():
        print(f"[存在] {target_path} 已存在 ({target_path.stat().st_size / 1e6:.1f} MB)")
        return

    # ---- 下载 ----
    download_dir = download_with_kagglehub(out_dir)
    if download_dir is None:
        # 备用：Kaggle API
        download_dir = download_with_kaggle_api(out_dir)
    if download_dir is None:
        print("[失败] 数据集下载失败，请检查网络或 Kaggle 认证。")
        print("       手动下载: https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset")
        return

    # ---- 查找 & 解压 ----
    csv_path = find_and_extract(download_dir, out_dir)
    if csv_path is None:
        return

    # ---- 重命名为标准名称 ----
    final_path = rename_to_target(csv_path, out_dir)

    # ---- 清理 ----
    cleanup(download_dir, out_dir)

    # ---- 验证 ----
    print(f"\n{'='*50}")
    print(f"[成功] 数据集准备完毕: {final_path}")
    print(f"       大小: {final_path.stat().st_size / 1e6:.1f} MB")

    # 快速验证 CSV 可读性
    try:
        import pandas as pd
        df = pd.read_csv(final_path)
        print(f"       行数: {len(df):,}")
        print(f"       列名: {df.columns.tolist()}")
    except ImportError:
        print("       (安装 pandas 后可显示数据概览)")
    except Exception as e:
        print(f"       [WARN] CSV 读取测试失败: {e}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="下载 MBTI 500 数据集并解压重命名为 MBTI_500.csv"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出目录（默认: data/）",
    )
    args = parser.parse_args()
    main(output=args.output)

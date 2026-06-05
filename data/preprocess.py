"""
MBTI 数据集预处理脚本
======================
1. 文本清洗（HTML标签、URL、特殊字符、空白规范化）
2. 标签拆分（16种 MBTI → 4维二分类标签）
3. 数据集划分（train/test = 8:2，按类型分层采样）

输出文件（保存到 data/ 目录）：
- train.csv  /  test.csv         原始格式（清洗后文本 + type + 四维标签）
- train.json /  test.json        JSON 格式（text + labels dict）
- label_map.json                  类型映射表

Usage:
    python data/preprocess.py                    # 默认：80/20 划分，seed=42
    python data/preprocess.py --seed 123         # 自定义随机种子
    python data/preprocess.py --val-ratio 0.1    # 划分 10% 验证集（可选）
"""

import re
import json
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "MBTI_500.csv"
OUT_DIR = BASE_DIR  # 输出到同目录

# ============================================================
# MBTI 标签映射
# ============================================================
# 16 种类型 → 4 个二分类维度
MBTI_TO_LABELS = {
    "INTJ": {"EI": 0, "SN": 1, "TF": 1, "JP": 1},  # I, N, T, J
    "INTP": {"EI": 0, "SN": 1, "TF": 1, "JP": 0},  # I, N, T, P
    "INFJ": {"EI": 0, "SN": 1, "TF": 0, "JP": 1},  # I, N, F, J
    "INFP": {"EI": 0, "SN": 1, "TF": 0, "JP": 0},  # I, N, F, P
    "ENTJ": {"EI": 1, "SN": 1, "TF": 1, "JP": 1},  # E, N, T, J
    "ENTP": {"EI": 1, "SN": 1, "TF": 1, "JP": 0},  # E, N, T, P
    "ENFJ": {"EI": 1, "SN": 1, "TF": 0, "JP": 1},  # E, N, F, J
    "ENFP": {"EI": 1, "SN": 1, "TF": 0, "JP": 0},  # E, N, F, P
    "ISTJ": {"EI": 0, "SN": 0, "TF": 1, "JP": 1},  # I, S, T, J
    "ISFJ": {"EI": 0, "SN": 0, "TF": 0, "JP": 1},  # I, S, F, J
    "ISTP": {"EI": 0, "SN": 0, "TF": 1, "JP": 0},  # I, S, T, P
    "ISFP": {"EI": 0, "SN": 0, "TF": 0, "JP": 0},  # I, S, F, P
    "ESTJ": {"EI": 1, "SN": 0, "TF": 1, "JP": 1},  # E, S, T, J
    "ESFJ": {"EI": 1, "SN": 0, "TF": 0, "JP": 1},  # E, S, F, J
    "ESTP": {"EI": 1, "SN": 0, "TF": 1, "JP": 0},  # E, S, T, P
    "ESFP": {"EI": 1, "SN": 0, "TF": 0, "JP": 0},  # E, S, F, P
}

DIMENSION_NAMES = ["EI", "SN", "TF", "JP"]
DIMENSION_LABELS = {
    "EI": {"E": 1, "I": 0},
    "SN": {"S": 0, "N": 1},
    "TF": {"T": 1, "F": 0},
    "JP": {"J": 1, "P": 0},
}


# ============================================================
# 文本清洗
# ============================================================

def clean_text(text: str) -> str:
    """清洗单条文本：去 HTML / URL / 特殊符号 / 空白规范化。"""
    if not isinstance(text, str):
        return ""

    # 1. 移除 URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 2. 移除 HTML 标签
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. 移除 @ 提及和 # 话题标签符号（保留文字）
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", lambda m: m.group(0).replace("#", " "), text)

    # 4. 移除 ||| 分隔符（某些数据集用这个分隔帖子）
    text = text.replace("|||", " ")

    # 5. 规范化空白（合并多个空格/换行/制表符）
    text = re.sub(r"\s+", " ", text)

    # 6. 移除首尾空白
    text = text.strip()

    return text


def get_text_stats(texts: pd.Series) -> dict:
    """计算文本长度统计。"""
    lengths = texts.str.len()
    return {
        "count": int(len(texts)),
        "mean_len": float(lengths.mean()),
        "median_len": float(lengths.median()),
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "std_len": float(lengths.std()),
    }


# ============================================================
# 标签处理
# ============================================================

def split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """将 type 列拆分为四个二分类标签列。"""
    labels_df = pd.DataFrame(
        df["type"].map(MBTI_TO_LABELS).tolist(), index=df.index
    )
    df["label_EI"] = labels_df["EI"].astype(int)
    df["label_SN"] = labels_df["SN"].astype(int)
    df["label_TF"] = labels_df["TF"].astype(int)
    df["label_JP"] = labels_df["JP"].astype(int)
    return df


# ============================================================
# 主流程
# ============================================================

def main(
    val_ratio: float = 0.0,
    test_ratio: float = 0.2,
    seed: int = 42,
    max_len: int | None = None,
):
    print("=" * 60)
    print("MBTI 数据集预处理")
    print("=" * 60)

    # -------- 1. 加载原始数据 --------
    print(f"\n[1/5] 加载原始数据: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"  总样本数: {len(df):,}")
    print(f"  列名: {df.columns.tolist()}")

    # -------- 2. 文本清洗 --------
    print("\n[2/5] 文本清洗...")
    before_stats = get_text_stats(df["posts"])
    print(f"  清洗前: mean={before_stats['mean_len']:.0f}, "
          f"min={before_stats['min_len']}, max={before_stats['max_len']}")

    df["text"] = df["posts"].apply(clean_text)

    # 清洗后可能产生空文本，过滤掉
    n_before = len(df)
    df = df[df["text"].str.strip().str.len() > 10].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  丢弃 {n_dropped} 条清洗后过短样本")

    after_stats = get_text_stats(df["text"])
    print(f"  清洗后: mean={after_stats['mean_len']:.0f}, "
          f"min={after_stats['min_len']}, max={after_stats['max_len']}")

    # 可选：截断过长文本
    if max_len is not None:
        df["text"] = df["text"].str[:max_len]
        print(f"  文本已截断至 {max_len} 字符")

    # -------- 3. 标签拆分 --------
    print("\n[3/5] MBTI 类型 → 四维二分类标签...")
    df = split_labels(df)

    print(f"  类型分布 (top-5):")
    for t, cnt in df["type"].value_counts().head(5).items():
        pct = cnt / len(df) * 100
        labels = MBTI_TO_LABELS[t]
        print(f"    {t}: {cnt:6d} ({pct:5.1f}%)  "
              f"EI={labels['EI']} SN={labels['SN']} "
              f"TF={labels['TF']} JP={labels['JP']}")

    print(f"\n  维度分布:")
    for dim in DIMENSION_NAMES:
        col = f"label_{dim}"
        pos = df[col].sum()
        neg = len(df) - pos
        print(f"    {dim}: 1={pos:6d} ({pos/len(df)*100:5.1f}%)  "
              f"0={neg:6d} ({neg/len(df)*100:5.1f}%)")

    # -------- 4. 数据集划分 --------
    print(f"\n[4/5] 数据集划分 (test={test_ratio:.0%}, seed={seed})...")

    # 按 MBTI 类型分层采样，保证 train/test 中各类比例一致
    df_train, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df["type"],
        shuffle=True,
    )

    # 可选：从训练集再切出验证集
    if val_ratio > 0:
        val_size = val_ratio / (1.0 - test_ratio)
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_size,
            random_state=seed,
            stratify=df_train["type"],
            shuffle=True,
        )

    print(f"  Train : {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    if val_ratio > 0:
        print(f"  Val   : {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")

    # -------- 5. 保存 --------
    print(f"\n[5/5] 保存处理结果到 {OUT_DIR} ...")

    # 输出列：原始 posts 保留用于对比，text 是清洗后的，type + 四维标签
    out_cols = ["text", "type", "label_EI", "label_SN", "label_TF", "label_JP"]

    # CSV 格式（便于 pandas 读取）
    train_csv = OUT_DIR / "train.csv"
    test_csv = OUT_DIR / "test.csv"
    df_train[out_cols].to_csv(train_csv, index=False)
    df_test[out_cols].to_csv(test_csv, index=False)
    print(f"  [CSV]  {train_csv}  ({len(df_train):,} 条)")
    print(f"  [CSV]  {test_csv}  ({len(df_test):,} 条)")

    if val_ratio > 0:
        val_csv = OUT_DIR / "val.csv"
        df_val[out_cols].to_csv(val_csv, index=False)
        print(f"  [CSV]  {val_csv}  ({len(df_val):,} 条)")

    # JSON 格式（便于直接加载为 dict）
    for split_name, split_df in [("train", df_train), ("test", df_test)]:
        records = []
        for _, row in split_df.iterrows():
            records.append({
                "text": row["text"],
                "mbti_type": row["type"],
                "labels": {
                    "EI": int(row["label_EI"]),
                    "SN": int(row["label_SN"]),
                    "TF": int(row["label_TF"]),
                    "JP": int(row["label_JP"]),
                },
            })
        json_path = OUT_DIR / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  [JSON] {json_path}  ({len(records):,} 条)")

    # 标签映射表（供后续推理时反向查表）
    label_map = {
        "dimensions": DIMENSION_NAMES,
        "dimension_labels": DIMENSION_LABELS,
        "mbti_to_labels": MBTI_TO_LABELS,
        "labels_to_mbti": {
            f"{v['EI']}{v['SN']}{v['TF']}{v['JP']}": k
            for k, v in MBTI_TO_LABELS.items()
        },
    }
    map_path = OUT_DIR / "label_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  [MAP]  {map_path}")

    # -------- 打印摘要 --------
    print(f"\n{'=' * 60}")
    print("预处理完成！输出文件:")
    print(f"  {OUT_DIR / 'train.csv'}")
    print(f"  {OUT_DIR / 'test.csv'}")
    print(f"  {OUT_DIR / 'train.json'}")
    print(f"  {OUT_DIR / 'test.json'}")
    print(f"  {OUT_DIR / 'label_map.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBTI 数据集预处理")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="测试集比例 (default: 0.2)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="验证集比例，从训练集中切 (default: 0)")
    parser.add_argument("--max-len", type=int, default=None,
                        help="文本截断长度（字符数），None 表示不截断")
    args = parser.parse_args()
    main(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_len=args.max_len,
    )

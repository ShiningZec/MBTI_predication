"""
数据集可视化 — 用于报告/论文展示
================================
生成以下图表到 data_viz/ 目录：

1. MBTI 16 类型分布（条形图）
2. 四维标签分布（堆叠条形图）
3. 文本长度分布（直方图）
4. Train/Test 划分对比
5. 常见词词云（按维度分组）
6. 四维相关性矩阵

Usage:
    python data_viz.py
"""

import sys
import warnings
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("data")
OUT_DIR = Path("data_viz")
OUT_DIR.mkdir(exist_ok=True)

DIMS = ["EI", "SN", "TF", "JP"]
DIM_LABELS = {"EI": ("E", "I"), "SN": ("S", "N"),
              "TF": ("T", "F"), "JP": ("J", "P")}
MBTI_ORDER = ["INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP",
              "ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"]

COLORS_16 = plt.cm.tab20(np.linspace(0, 1, 16))
COLORS_4 = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    full  = pd.read_csv(DATA_DIR / "MBTI_500.csv")
    return train, test, full


# ============================================================
# 图 1: MBTI 16 类型分布
# ============================================================
def plot_type_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = df["type"].value_counts()
    counts = counts.reindex(MBTI_ORDER)

    bars = ax.bar(range(16), counts.values, color=COLORS_16, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(16))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Samples", fontsize=12)
    ax.set_title("MBTI 16-Type Distribution (n=106,067)", fontsize=14, fontweight="bold")

    # 标注数值
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,}", ha="center", fontsize=7, color="gray")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_type_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[1/6] 16 类型分布 → 01_type_distribution.png")


# ============================================================
# 图 2: 四维标签分布
# ============================================================
def plot_dimension_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for i, dim in enumerate(DIMS):
        col = f"label_{dim}"
        pos = df[col].sum()
        neg = len(df) - pos
        labels = [DIM_LABELS[dim][1], DIM_LABELS[dim][0]]  # [pos_label, neg_label]
        sizes = [pos, neg]
        explode = (0.05, 0)

        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, autopct="%1.1f%%",
            colors=[COLORS_4[i], "#e0e0e0"], startangle=90,
            textprops={"fontsize": 11},
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax.set_title(f"{dim}  ({labels[0]}:{pos/len(df)*100:.1f}%  "
                     f"{labels[1]}:{neg/len(df)*100:.1f}%)", fontsize=12, fontweight="bold")

    fig.suptitle("MBTI 4-Dimension Label Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_dimension_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[2/6] 四维标签分布 → 02_dimension_distribution.png")


# ============================================================
# 图 3: 文本长度分布
# ============================================================
def plot_text_length(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    lens = df["posts"].str.len()

    ax.hist(lens, bins=80, color=COLORS_4[0], alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(lens.mean(), color="red", ls="--", lw=2, label=f"Mean = {lens.mean():.0f} ch")
    ax.axvline(lens.median(), color="orange", ls="--", lw=2, label=f"Median = {lens.median():.0f} ch")

    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Text Length Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    # 标注 BERT token 等价
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    bert_ticks = [0, 1000, 2000, 3000, 4000, 5000]
    ax2.set_xticks(bert_ticks)
    ax2.set_xticklabels([f"~{int(t/4)}" for t in bert_ticks], fontsize=8, color="gray")
    ax2.set_xlabel("Est. BPE Tokens (÷4)", fontsize=9, color="gray")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_text_length.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[3/6] 文本长度分布 → 03_text_length.png")


# ============================================================
# 图 4: Train/Test 划分
# ============================================================
def plot_train_test_split(train: pd.DataFrame, test: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 饼图
    ax = axes[0]
    sizes = [len(train), len(test)]
    labels = [f"Train\n{sizes[0]:,} ({sizes[0]/sum(sizes)*100:.0f}%)",
              f"Test\n{sizes[1]:,} ({sizes[1]/sum(sizes)*100:.0f}%)"]
    ax.pie(sizes, labels=labels, colors=[COLORS_4[0], COLORS_4[2]],
           autopct="", startangle=90, textprops={"fontsize": 11, "fontweight": "bold"})
    ax.set_title("Train/Test Split", fontsize=13, fontweight="bold")

    # 对比 16 型比例验证 stratification
    ax = axes[1]
    train_counts = train["type"].value_counts(normalize=True).reindex(MBTI_ORDER)
    test_counts = test["type"].value_counts(normalize=True).reindex(MBTI_ORDER)
    x = np.arange(16)
    w = 0.35
    ax.bar(x - w/2, train_counts.values, w, label="Train", color=COLORS_4[0], alpha=0.8)
    ax.bar(x + w/2, test_counts.values, w, label="Test", color=COLORS_4[2], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(train_counts.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Stratified Split Verification", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "04_train_test_split.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[4/6] Train/Test 划分 → 04_train_test_split.png")


# ============================================================
# 图 5: 常见词对比（按维度）
# ============================================================
def plot_keyword_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    stop_words = "the a an and or but in on at to for of with by from up about " \
                 "is are was were be been being have has had do does did will would " \
                 "shall should can could may might must i me my we us our you your " \
                 "he she it they them their this that these those not no just so if " \
                 "then than too very really also only".split()
    # RoBERTa stop tokens to filter
    stop_words += ["know", "like", "think", "people", "one", "get", "go", "make",
                   "time", "say", "see", "would", "could", "thing", "feel", "really"]

    for i, dim in enumerate(DIMS):
        col = f"label_{dim}"
        text_col = "text" if "text" in df.columns else "posts"
        pos_texts = df[df[col] == 1][text_col].dropna().astype(str)
        neg_texts = df[df[col] == 0][text_col].dropna().astype(str)

        pos_label, neg_label = DIM_LABELS[dim]

        vec = CountVectorizer(stop_words="english", max_features=200, ngram_range=(1, 2))
        vec.fit(pd.concat([pos_texts, neg_texts]))

        pos_vec = vec.transform(pos_texts).sum(axis=0).A1
        neg_vec = vec.transform(neg_texts).sum(axis=0).A1

        # 找区分度最大的词
        feature_names = vec.get_feature_names_out()
        diff_ratio = np.zeros(len(feature_names))
        for j in range(len(feature_names)):
            p = pos_vec[j] / (len(pos_texts) + 1)
            n = neg_vec[j] / (len(neg_texts) + 1)
            diff_ratio[j] = p - n

        top_pos_idx = np.argsort(diff_ratio)[-15:]
        top_neg_idx = np.argsort(diff_ratio)[:15]

        ax = axes[i]
        words = []
        scores = []
        colors_bar = []
        for idx in top_pos_idx:
            words.append(feature_names[idx])
            scores.append(abs(diff_ratio[idx]))
            colors_bar.append(COLORS_4[i])
        for idx in top_neg_idx[::-1]:
            words.append(feature_names[idx])
            scores.append(abs(diff_ratio[idx]))
            colors_bar.append("#aaaaaa")

        y_pos = range(len(words))
        ax.barh(y_pos, scores, color=colors_bar, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{dim}  —  {pos_label} (colored) vs {neg_label} (grey)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Discriminative Score", fontsize=9)

    fig.suptitle("Most Discriminative Words per Dimension", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "05_keyword_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[5/6] 关键词判别力对比 → 05_keyword_comparison.png")


# ============================================================
# 图 6: 四维相关性
# ============================================================
def plot_dimension_correlation(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    label_cols = [f"label_{d}" for d in DIMS]
    corr = df[label_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0,
                vmin=-0.3, vmax=0.3, mask=mask,
                xticklabels=DIMS, yticklabels=DIMS,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 14, "fontweight": "bold"},
                ax=ax)
    ax.set_title("Inter-Dimension Label Correlation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "06_dimension_correlation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[6/6] 四维相关性 → 06_dimension_correlation.png")


# ============================================================
def main():
    print("加载数据...")
    train, test, full = load_data()

    plot_type_distribution(full)
    plot_dimension_distribution(train)
    plot_text_length(full)
    plot_train_test_split(train, test)
    plot_keyword_comparison(train)
    plot_dimension_correlation(train)

    print(f"\n全部图表已保存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()

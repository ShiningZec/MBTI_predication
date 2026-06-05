"""
MBTI 模型评估 + 可视化
======================
加载训练好的模型，在测试集上输出多维度指标和图表。

指标:
    - 逐维度: Accuracy, Precision, Recall, F1, AUC-ROC, MCC
    - 整体: Exact Match, Hamming Loss

可视化:
    - 混淆矩阵 (4维 × 2)
    - ROC 曲线 (4维)
    - 置信度分布直方图
    - 四维雷达图

Usage:
    python eval.py                                    # 自动找最新 checkpoint
    python eval.py --ckpt output/xxx/best             # 指定 checkpoint
    python eval.py --ckpt output/xxx/best --no-plot   # 仅指标，不出图
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头模式，写文件不弹窗
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier
from src.data.dataset import MBTIDataset

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
DIMS = ["EI", "SN", "TF", "JP"]
DIM_LABELS = {"EI": ("E", "I"), "SN": ("S", "N"), "TF": ("T", "F"), "JP": ("J", "P")}
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]


# ============================================================
# 评估核心
# ============================================================

@torch.no_grad()
def evaluate_full(encoder, classifier, dataloader, device):
    all_logits = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        features = encoder(input_ids, attention_mask)
        logits_dict = classifier(features)

        logits = torch.stack([logits_dict[d].cpu() for d in DIMS], dim=1)
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)   # (N, 4)
    labels = np.concatenate(all_labels, axis=0)    # (N, 4)
    probs = 1 / (1 + np.exp(-logits))              # sigmoid
    preds = (probs >= 0.5).astype(int)

    return logits, probs, preds, labels


def compute_metrics(probs, preds, labels):
    """计算全部指标。"""
    results = {}
    for i, dim in enumerate(DIMS):
        y_t = labels[:, i]
        y_p = preds[:, i]
        y_pr = probs[:, i]

        results[dim] = {
            "accuracy":  round(accuracy_score(y_t, y_p), 4),
            "precision": round(precision_score(y_t, y_p, zero_division=0), 4),
            "recall":    round(recall_score(y_t, y_p, zero_division=0), 4),
            "f1":        round(f1_score(y_t, y_p, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_t, y_pr), 4),
            "mcc":       round(matthews_corrcoef(y_t, y_p), 4),
        }

    # 整体
    results["exact_match"] = round(float((preds == labels).all(axis=1).mean()), 4)
    results["hamming_loss"] = round(hamming_loss(labels, preds), 4)
    results["mean_acc"] = round(np.mean([results[d]["accuracy"] for d in DIMS]), 4)
    results["mean_f1"] = round(np.mean([results[d]["f1"] for d in DIMS]), 4)
    results["mean_auc"] = round(np.mean([results[d]["auc"] for d in DIMS]), 4)
    results["macro_mcc"] = round(np.mean([results[d]["mcc"] for d in DIMS]), 4)

    return results


# ============================================================
# 可视化
# ============================================================

def plot_confusion_matrices(labels, preds, save_path):
    """4 维混淆矩阵，2×2 子图。"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    for i, dim in enumerate(DIMS):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues", vmin=0)

        pos_l, neg_l = DIM_LABELS[dim]
        ax.set_xticks([0, 1])
        ax.set_xticklabels([neg_l, pos_l])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([neg_l, pos_l])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"[{dim}] {pos_l}/{neg_l}")

        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r,c]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[r,c] > cm.max()/2 else "gray")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 混淆矩阵 → {save_path}")


def plot_roc_curves(labels, probs, save_path):
    """4 维 ROC 曲线叠在一张图。"""
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, dim in enumerate(DIMS):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f"{dim} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — MBTI 4 Dimensions")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] ROC 曲线 → {save_path}")


def plot_confidence_distribution(probs, preds, labels, save_path):
    """每个维度的置信度分布，按正确/错误分组。"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    for i, dim in enumerate(DIMS):
        ax = axes[i]
        correct = preds[:, i] == labels[:, i]
        wrong = ~correct

        ax.hist(probs[correct, i], bins=30, alpha=0.6, color=COLORS[i],
                label=f"Correct ({correct.sum():,})", density=True)
        ax.hist(probs[wrong, i], bins=30, alpha=0.5, color="red",
                label=f"Wrong ({wrong.sum():,})", density=True)
        ax.axvline(x=0.5, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_title(dim)
        ax.legend(fontsize=7)

    fig.suptitle("Confidence Distribution — Correct vs Wrong", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 置信度分布 → {save_path}")


def plot_radar(metrics, save_path):
    """四维雷达图（Accuracy / F1 / AUC）。"""
    categories = DIMS
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for metric_name, color, ls in [("accuracy", COLORS[0], "-"),
                                     ("f1", COLORS[1], "--"),
                                     ("auc", COLORS[2], ":")]:
        values = [metrics[d][metric_name] for d in DIMS]
        values += values[:1]
        ax.plot(angles, values, color=color, lw=2, ls=ls, label=metric_name.upper())
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("MBTI 4-Dimension Radar", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 雷达图 → {save_path}")


# ============================================================
# 入口
# ============================================================

def find_latest_ckpt(output_dir="output"):
    out = Path(output_dir)
    if not out.exists():
        return None
    for d in sorted(out.iterdir(), reverse=True):
        best = d / "best"
        if (best / "encoder.pt").exists():
            return best
    return None


def load_model(ckpt_path, device):
    ckpt = Path(ckpt_path)
    info_path = ckpt.parent / "training_info.json"
    info = {}
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    model_name = info.get("model") or "D:/ML/MBTI_pred/models/roberta-base"
    pooling = info.get("pooling", "mean")
    max_len = info.get("config", {}).get("max_length", 512)
    hidden = info.get("hidden_size", 768)

    encoder = RoBERTaEncoder(model_name=model_name, pooling=pooling, max_length=max_len)
    encoder.load_state_dict(torch.load(ckpt / "encoder.pt", map_location=device, weights_only=True))
    classifier = MBTIClassifier(input_dim=hidden)
    classifier.load_state_dict(torch.load(ckpt / "classifier.pt", map_location=device, weights_only=True))
    encoder.to(device).eval()
    classifier.to(device).eval()
    return encoder, classifier, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default="data/test.csv")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 找 checkpoint ----
    ckpt = args.ckpt or find_latest_ckpt()
    if not ckpt:
        print("未找到 checkpoint"); return
    print(f"Checkpoint: {ckpt}")

    # ---- 加载模型 ----
    encoder, classifier, info = load_model(ckpt, device)
    print(f"模型: {encoder.model_name}, Pooling: {encoder.pooling_name}")

    # ---- 评估 ----
    ds = MBTIDataset(args.test_csv, encoder.tokenizer, args.max_len)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False)
    print(f"测试集: {len(ds):,}")

    logits, probs, preds, labels = evaluate_full(encoder, classifier, dl, device)
    metrics = compute_metrics(probs, preds, labels)

    # ---- 打印指标 ----
    print(f"\n{'='*65}")
    print(f"{'Dim':>5} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'MCC':>8}")
    print("-" * 65)
    for dim in DIMS:
        m = metrics[dim]
        print(f"{dim:>5} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f} {m['mcc']:>8.4f}")
    print("-" * 65)
    print(f"{'Mean':>5} {metrics['mean_acc']:>8.4f} {'':>8} {'':>8} {metrics['mean_f1']:>8.4f} {metrics['mean_auc']:>8.4f}")
    print(f"\n  Exact Match: {metrics['exact_match']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  Macro MCC: {metrics['macro_mcc']:.4f}")

    # ---- 保存指标 JSON ----
    json_path = out_dir / "metrics.json"
    if info.get("best_overall_acc"):
        metrics["best_train_acc"] = info["best_overall_acc"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[指标] 已保存 → {json_path}")

    # ---- 可视化 ----
    if not args.no_plot:
        plot_confusion_matrices(labels, preds, out_dir / "confusion_matrices.png")
        plot_roc_curves(labels, probs, out_dir / "roc_curves.png")
        plot_confidence_distribution(probs, preds, labels, out_dir / "confidence_dist.png")
        plot_radar(metrics, out_dir / "radar.png")


if __name__ == "__main__":
    main()

"""
超参数优化 — 两阶段搜索
========================

阶段 1：随机搜索（离散 + 连续参数），20-30 次
阶段 2：贝叶斯优化（固定最佳离散参数，调优连续参数）

每次搜索结果保存到 test/<timestamp>.json，包含完整参数和 6 项评测指标。

Usage:
    python hp_tune.py                     # 正式优化（全量数据）
    python hp_tune.py --test              # 快速测试（1 trial，5% 数据）
    python hp_tune.py --random 30         # 自定义随机搜索次数（默认 25）
    python hp_tune.py --bayes 30          # 自定义贝叶斯搜索次数（默认 25）
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.samplers import RandomSampler, TPESampler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef,
    hamming_loss,
)
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier, JointBCELoss
from src.data.dataset import MBTIDataset

# ============================================================
# 搜索空间
# ============================================================

DISCRETE_PARAMS = {
    "max_length":     [256, 512],
    "batch_size":     [8, 16],
    "pooling":        ["cls", "mean"],
    "freeze_layers":  [0, 6, 9],
    "head_hidden":    [128, 256, 512],
}

CONTINUOUS_PARAMS = {
    "dropout":        {"type": "float",      "low": 0.1,  "high": 0.4},
    "encoder_lr":     {"type": "loguniform", "low": 5e-6, "high": 3e-5},
    "classifier_lr":  {"type": "loguniform", "low": 5e-5, "high": 3e-4},
    "weight_decay":   {"type": "loguniform", "low": 1e-6, "high": 1e-2},
    "warmup_ratio":   {"type": "float",      "low": 0.0,  "high": 0.15},
}

FIXED = {
    "model_name":  "D:/ML/MBTI_pred/models/roberta-base",
    "train_csv":   "data/train.csv",
    "test_csv":    "data/test.csv",
    "num_epochs":  10,
    "seed":        42,
    "dim_weights":  {"EI": 0.25, "SN": 0.35, "TF": 0.20, "JP": 0.20},
    "pos_weights": {"SN": 10.0},
}

DIMS = ["EI", "SN", "TF", "JP"]

# ============================================================
# 工具函数
# ============================================================

def sample_all_params(trial: optuna.Trial) -> dict:
    """采样所有参数（离散 + 连续）。"""
    params = {}
    for name, choices in DISCRETE_PARAMS.items():
        params[name] = trial.suggest_categorical(name, choices)
    for name, spec in CONTINUOUS_PARAMS.items():
        log = spec["type"] == "loguniform"
        params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)
    return params


def sample_continuous(trial: optuna.Trial, fixed_discrete: dict) -> dict:
    """固定离散参数，仅采样连续参数。"""
    params = dict(fixed_discrete)
    for name, spec in CONTINUOUS_PARAMS.items():
        log = spec["type"] == "loguniform"
        params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)
    return params


def save_result(params: dict, metrics: dict, phase: str):
    """保存单次搜索结果到 test/<timestamp>.json。"""
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{phase}_{ts}.json"
    record = {
        "timestamp": ts,
        "phase": phase,
        "params": params,
        "metrics": metrics,
    }
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  [保存] {filename}")


# ============================================================
# 训练 + 评测
# ============================================================

def compute_metrics(probs, preds, labels):
    """计算全部指标（与 eval.py 一致）。"""
    m = {}
    for i, dim in enumerate(DIMS):
        y_t = labels[:, i]
        y_p = preds[:, i]
        y_pr = probs[:, i]
        m[dim] = {
            "accuracy":  round(accuracy_score(y_t, y_p), 4),
            "precision": round(precision_score(y_t, y_p, zero_division=0), 4),
            "recall":    round(recall_score(y_t, y_p, zero_division=0), 4),
            "f1":        round(f1_score(y_t, y_p, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_t, y_pr), 4),
            "mcc":       round(matthews_corrcoef(y_t, y_p), 4),
        }
    m["exact_match"]  = round(float((preds == labels).all(axis=1).mean()), 4)
    m["hamming_loss"] = round(hamming_loss(labels, preds), 4)
    m["mean_acc"]     = round(np.mean([m[d]["accuracy"] for d in DIMS]), 4)
    m["mean_f1"]      = round(np.mean([m[d]["f1"] for d in DIMS]), 4)
    m["mean_auc"]     = round(np.mean([m[d]["auc"] for d in DIMS]), 4)
    m["macro_mcc"]    = round(np.mean([m[d]["mcc"] for d in DIMS]), 4)
    return m


@torch.no_grad()
def evaluate_model(encoder, classifier, test_loader, device):
    """评估模型，返回 probs/preds/labels + 所有指标。"""
    encoder.eval()
    classifier.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        features = encoder(input_ids, attention_mask)
        logits = classifier(features)
        probs = torch.stack(
            [torch.sigmoid(logits[d]) for d in DIMS], dim=1
        ).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs)
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return probs, preds, labels


def train_and_eval(params: dict, test_mode: bool = False) -> dict:
    """训练 10 epoch，返回评测指标。"""
    # ---- encoder ----
    encoder = RoBERTaEncoder(
        model_name=FIXED["model_name"],
        pooling=params["pooling"],
        max_length=params["max_length"],
    )
    # 冻结层
    freeze_n = params["freeze_layers"]
    if freeze_n > 0:
        for i in range(min(freeze_n, 12)):
            for p in encoder.backbone.encoder.layer[i].parameters():
                p.requires_grad = False

    # ---- classifier ----
    classifier = MBTIClassifier(
        input_dim=encoder.hidden_size,
        head_hidden=params["head_hidden"],
        dropout=params["dropout"],
    )

    # ---- 数据 ----
    train_ds = MBTIDataset(FIXED["train_csv"], encoder.tokenizer, params["max_length"])
    test_ds  = MBTIDataset(FIXED["test_csv"],  encoder.tokenizer, params["max_length"])
    if test_mode:
        n_train = max(500, len(train_ds) // 20)
        n_test  = max(200, len(test_ds) // 20)
        train_ds = Subset(train_ds, np.random.choice(len(train_ds), n_train, replace=False))
        test_ds  = Subset(test_ds,  np.random.choice(len(test_ds),  n_test,  replace=False))
        FIXED["num_epochs"] = 1

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=min(params["batch_size"]*2, 32), shuffle=False)

    # ---- 训练 ----
    device = encoder.device
    classifier.to(device)

    criterion = JointBCELoss(
        ei_weight=FIXED["dim_weights"]["EI"],
        sn_weight=FIXED["dim_weights"]["SN"],
        tf_weight=FIXED["dim_weights"]["TF"],
        jp_weight=FIXED["dim_weights"]["JP"],
        pos_weights=FIXED["pos_weights"],
    )
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": params["encoder_lr"]},
        {"params": classifier.parameters(), "lr": params["classifier_lr"]},
    ], weight_decay=params["weight_decay"])

    total_steps = len(train_loader) * FIXED["num_epochs"]
    warmup_steps = int(total_steps * params["warmup_ratio"])

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7,
        )

    for epoch in range(FIXED["num_epochs"]):
        encoder.train()
        classifier.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            features = encoder(input_ids, attention_mask)
            logits = classifier(features)
            loss, _ = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            grads = [p for p in encoder.parameters() if p.requires_grad] + \
                    [p for p in classifier.parameters() if p.requires_grad]
            if grads:
                torch.nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            scheduler.step()

    # ---- 评测 ----
    probs, preds, labels = evaluate_model(encoder, classifier, test_loader, device)
    metrics = compute_metrics(probs, preds, labels)

    # 清理显存
    del encoder, classifier, optimizer
    torch.cuda.empty_cache()

    return metrics


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="快速测试模式")
    parser.add_argument("--random", type=int, default=25, help="随机搜索次数")
    parser.add_argument("--bayes", type=int, default=25, help="贝叶斯搜索次数")
    args = parser.parse_args()

    if args.test:
        args.random = 1
        args.bayes = 0
        FIXED["num_epochs"] = 1
        print("快速测试模式：1 trial, 5% 数据, 1 epoch")

    print(f"\n阶段 1 — 随机搜索 ({args.random} trials)")
    print(f"阶段 2 — 贝叶斯优化 ({args.bayes} trials)\n")

    # ============ 阶段 1：随机搜索 ============
    print("=" * 60)
    print("阶段 1：随机搜索（离散 + 连续参数）")
    print("=" * 60)

    sampler1 = RandomSampler(seed=FIXED["seed"])
    study1 = optuna.create_study(
        study_name="phase1_random",
        sampler=sampler1,
        direction="maximize",
    )

    def obj1(trial):
        params = sample_all_params(trial)
        print(f"\n[Phase1 Trial {trial.number}] {params}")
        metrics = train_and_eval(params, test_mode=args.test)
        save_result(params, metrics, "phase1")
        print(f"  Mean Acc={metrics['mean_acc']:.4f}  "
              f"Mean F1={metrics['mean_f1']:.4f}  "
              f"Exact Match={metrics['exact_match']:.4f}")
        return metrics["mean_acc"]

    study1.optimize(obj1, n_trials=args.random, show_progress_bar=False)

    best_discrete = {k: study1.best_params[k] for k in DISCRETE_PARAMS}
    print(f"\n阶段 1 最佳离散参数: {best_discrete}")
    print(f"最佳 Mean Acc: {study1.best_value:.4f}")

    # ============ 阶段 2：贝叶斯优化 ============
    if args.bayes == 0:
        print("\n跳过阶段 2（--bayes 0）")
        return

    print(f"\n{'=' * 60}")
    print("阶段 2：贝叶斯优化（固定最佳离散参数，调优连续参数）")
    print(f"{'=' * 60}")
    print(f"离散参数固定: {best_discrete}")

    sampler2 = TPESampler(seed=FIXED["seed"])
    study2 = optuna.create_study(
        study_name="phase2_bayes",
        sampler=sampler2,
        direction="maximize",
    )

    def obj2(trial):
        params = sample_continuous(trial, best_discrete)
        print(f"\n[Phase2 Trial {trial.number}] {params}")
        metrics = train_and_eval(params, test_mode=args.test)
        save_result(params, metrics, "phase2")
        print(f"  Mean Acc={metrics['mean_acc']:.4f}  "
              f"Mean F1={metrics['mean_f1']:.4f}  "
              f"Exact Match={metrics['exact_match']:.4f}")
        return metrics["mean_acc"]

    study2.optimize(obj2, n_trials=args.bayes, show_progress_bar=False)

    # ============ 汇总 ============
    sep = "=" * 60
    print(f"\n{sep}")
    print("优化完成")
    print(sep)
    print(f"\n阶段 1 最佳 (随机): Mean Acc = {study1.best_value:.4f}")
    for k, v in study1.best_params.items():
        print(f"  {k}: {v}")
    print(f"\n阶段 2 最佳 (贝叶斯): Mean Acc = {study2.best_value:.4f}")
    for k, v in study2.best_params.items():
        print(f"  {k}: {v}")

    # 保存最优配置
    best = {
        "best_mean_acc": study2.best_value,
        "best_params": study2.best_params,
        "phase1_best": study1.best_params,
        "phase1_value": study1.best_value,
    }
    with open("test/best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"\n最优配置已保存: test/best_config.json")


if __name__ == "__main__":
    main()

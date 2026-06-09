"""
超参数优化 — 两阶段随机搜索
========================

阶段 1（随机搜索）: 搜索离散参数 (pooling, head_hidden)
阶段 2（随机搜索）: 固定最佳离散，随机搜索连续参数 (dropout, lr, weight_decay, warmup)

结果保存到 test/<phase>_<timestamp>.json

Usage:
    python hp_tune.py                     # 正式优化
    python hp_tune.py --test              # 快速测试（1 trial, 1 epoch, 5% 数据）
    python hp_tune.py --p1 15 --p2 10     # 自定义各阶段 trial 数
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
from optuna.samplers import RandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef, hamming_loss,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier, JointBCELoss
from src.data.dataset import MBTIDataset

# ============================================================
# 固定配置（不参与搜索）
# ============================================================

CONFIG = {
    # 模型
    "model_name":     "D:/ML/MBTI_pred/models/roberta-base",
    "max_length":     512,
    "batch_size":     128,
    "pooling":        "mean",           # 阶段 1 搜索后覆盖

    # 训练
    "num_epochs":     5,
    "fp16":           True,             # 快 2x，精度损失可忽略
    "num_workers":    0,               # DataLoader 线程 (0=主进程)
    "freeze_layers":  0,
    "seed":           42,

    # 损失
    "dim_weights":  {"EI": 0.25, "SN": 0.35, "TF": 0.20, "JP": 0.20},
    "pos_weights":  {"SN": 10.0},

    # 数据
    "train_csv": "data/train.csv",
    "test_csv":  "data/test.csv",

    # 阶段 1 连续参数暂用值
    "dropout":        0.2,
    "encoder_lr":     2e-5,
    "classifier_lr":  1e-4,
    "weight_decay":   0.01,
    "warmup_ratio":   0.1,
}

# ============================================================
# 搜索空间
# ============================================================

# 阶段 1：离散参数
SEARCH_DISCRETE = {
    "pooling":      ["cls", "mean"],
    "head_hidden":  [64, 128, 256, 512],
}

# 阶段 2：连续参数
SEARCH_CONTINUOUS = {
    "dropout":        {"type": "float",      "low": 0.1,  "high": 0.4},
    "encoder_lr":     {"type": "loguniform", "low": 5e-6, "high": 3e-5},
    "classifier_lr":  {"type": "loguniform", "low": 5e-5, "high": 3e-4},
    "weight_decay":   {"type": "loguniform", "low": 1e-6, "high": 1e-2},
    "warmup_ratio":   {"type": "float",      "low": 0.0,  "high": 0.15},
}

DIMS = ["EI", "SN", "TF", "JP"]

# ============================================================
# 搜索用训练（支持数据子集比例）
# ============================================================

def train_and_eval(params: dict, data_ratio: float = 1.0, num_epochs: int = None):
    """训练并返回 (评测指标, 最终训练loss)。"""
    epochs = num_epochs or CONFIG["num_epochs"]

    # ---- encoder ----
    encoder = RoBERTaEncoder(
        model_name=CONFIG["model_name"],
        pooling=params.get("pooling", "mean"),
        max_length=params.get("max_length", 512),
    )
    # ---- classifier ----
    classifier = MBTIClassifier(
        input_dim=encoder.hidden_size,
        head_hidden=params.get("head_hidden", 64),
        dropout=params.get("dropout", 0.2),
    )

    # ---- 数据 ----
    train_ds = MBTIDataset(CONFIG["train_csv"], encoder.tokenizer,
                           params.get("max_length", 512))
    test_ds  = MBTIDataset(CONFIG["test_csv"],  encoder.tokenizer,
                           params.get("max_length", 512))
    if data_ratio < 1.0:
        n_train = max(2000, int(len(train_ds) * data_ratio))
        n_test  = max(1000, int(len(test_ds) * data_ratio))
        train_ds = Subset(train_ds, np.random.choice(len(train_ds), n_train, replace=False))
        test_ds  = Subset(test_ds,  np.random.choice(len(test_ds),  n_test,  replace=False))

    nw = CONFIG.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=params.get("batch_size", 16),
                              shuffle=True, num_workers=nw)
    test_loader  = DataLoader(test_ds, batch_size=min(params.get("batch_size", 16)*2, 32),
                              shuffle=False, num_workers=nw)

    # ---- 训练 ----
    device = encoder.device
    classifier.to(device)

    criterion = JointBCELoss(
        ei_weight=CONFIG["dim_weights"]["EI"],
        sn_weight=CONFIG["dim_weights"]["SN"],
        tf_weight=CONFIG["dim_weights"]["TF"],
        jp_weight=CONFIG["dim_weights"]["JP"],
        pos_weights=CONFIG["pos_weights"],
    )
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": params.get("encoder_lr", 2e-5)},
        {"params": classifier.parameters(), "lr": params.get("classifier_lr", 1e-4)},
    ], weight_decay=params.get("weight_decay", 0.01))

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * params.get("warmup_ratio", 0.1))
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7)

    scaler = torch.amp.GradScaler("cuda") if params.get("fp16", True) else None

    for epoch in range(epochs):
        encoder.train(); classifier.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False, ncols=80)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    features = encoder(ids, mask)
                    logits = classifier(features)
                    loss, _ = criterion(logits, labels)
                scaler.scale(loss).backward()
                grad_params = [p for n, p in encoder.named_parameters() if p.requires_grad] + \
                              [p for n, p in classifier.named_parameters() if p.requires_grad]
                if grad_params:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(grad_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                features = encoder(ids, mask)
                logits = classifier(features)
                loss, _ = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                grad_params = [p for n, p in encoder.named_parameters() if p.requires_grad] + \
                              [p for n, p in classifier.named_parameters() if p.requires_grad]
                if grad_params:
                    torch.nn.utils.clip_grad_norm_(grad_params, 1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # ---- 评测 ----
    encoder.eval(); classifier.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = encoder(ids, mask)
            logits = classifier(features)
            probs = torch.stack([torch.sigmoid(logits[d]) for d in DIMS], dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    labels = np.concatenate(all_labels)

    m = {}
    for i, dim in enumerate(DIMS):
        yt, yp, ypr = labels[:, i], preds[:, i], probs[:, i]
        m[dim] = {
            "accuracy":  round(accuracy_score(yt, yp), 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall":    round(recall_score(yt, yp, zero_division=0), 4),
            "f1":        round(f1_score(yt, yp, zero_division=0), 4),
            "auc":       round(roc_auc_score(yt, ypr), 4),
            "mcc":       round(matthews_corrcoef(yt, yp), 4),
        }
    m["exact_match"]  = round(float((preds == labels).all(axis=1).mean()), 4)
    m["hamming_loss"] = round(hamming_loss(labels, preds), 4)
    m["mean_acc"]     = round(np.mean([m[d]["accuracy"] for d in DIMS]), 4)
    m["mean_f1"]      = round(np.mean([m[d]["f1"] for d in DIMS]), 4)
    m["mean_auc"]     = round(np.mean([m[d]["auc"] for d in DIMS]), 4)
    m["macro_mcc"]    = round(np.mean([m[d]["mcc"] for d in DIMS]), 4)

    final_loss = total_loss / len(train_loader)

    del encoder, classifier, optimizer
    torch.cuda.empty_cache()
    return m, final_loss


def save_result(params: dict, metrics: dict, phase: str, trial_no: int = 0,
                avg_loss: float = 0.0):
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{phase}_{ts}.json"
    record = {
        "timestamp": ts, "phase": phase, "trial": trial_no,
        "params": params, "metrics": metrics, "avg_loss": avg_loss,
    }
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  [保存] test/{filename}")

    # 追加到汇总日志
    summary_path = out_dir / "trials_summary.jsonl"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="快速测试")
    parser.add_argument("--p1", type=int, default=8, help="阶段 1 trial 数")
    parser.add_argument("--p2", type=int, default=10, help="阶段 2 随机搜索 trial 数")
    args = parser.parse_args()

    if args.test:
        args.p1, args.p2 = 1, 0
        CONFIG["num_epochs"] = 1
        print("快速测试：1 trial, 5% 数据, 1 epoch\n")

    print(f"搜索配置：阶段 1={args.p1} 随机, 阶段 2={args.p2} 随机")
    print(f"离散参数：{list(SEARCH_DISCRETE.keys())}")
    print(f"连续参数（阶段 2）：{list(SEARCH_CONTINUOUS.keys())}\n")

    # ============ 阶段 1：随机搜索离散参数 ============
    print("=" * 55)
    print("阶段 1：随机搜索（固定连续，搜离散）")
    print("=" * 55)

    sampler1 = RandomSampler(seed=CONFIG["seed"])
    study1 = optuna.create_study(study_name="p1_discrete", sampler=sampler1,
                                  direction="maximize")

    def obj1(trial):
        params = dict(CONFIG)
        for name, choices in SEARCH_DISCRETE.items():
            params[name] = trial.suggest_categorical(name, choices)

        print(f"\n{'─'*50}")
        print(f"[Phase1 {trial.number+1}/{args.p1}] "
              f"pool={params['pooling']} hh={params['head_hidden']}")
        print(f"{'─'*50}")

        ratio = 0.05 if args.test else 0.3
        metrics, avg_loss = train_and_eval(params, data_ratio=ratio)
        save_result(params, metrics, "phase1", trial_no=trial.number + 1,
                    avg_loss=avg_loss)
        print(f"  Mean Acc={metrics['mean_acc']:.4f}  "
              f"MF1={metrics['mean_f1']:.4f}  "
              f"Exact={metrics['exact_match']:.4f}  "
              f"MCC={metrics['macro_mcc']:.4f}")
        return metrics["mean_acc"]

    def obj1_with_progress(trial):
        value = obj1(trial)
        try:
            best = study1.best_value
        except ValueError:
            best = value
        print(f"\n  >>> 阶段 1 进度: {trial.number+1}/{args.p1} "
              f"| 最佳 Mean Acc: {best:.4f}\n")
        return value

    study1.optimize(obj1_with_progress, n_trials=args.p1)

    # 合并最佳离散 + CONFIG 固定值 → 阶段 2 基座
    best_discrete = dict(CONFIG)
    for k in SEARCH_DISCRETE:
        best_discrete[k] = study1.best_params[k]
    print(f"\n阶段 1 最佳: pool={best_discrete['pooling']}, "
          f"hh={best_discrete['head_hidden']}")
    print(f"Mean Acc: {study1.best_value:.4f}")

    # ============ 阶段 2：贝叶斯搜索连续参数 ============
    if args.p2 == 0:
        print("\n--- 完成（仅阶段 1）---"); return

    print(f"\n{'=' * 55}")
    print("阶段 2：随机搜索（固定离散，搜连续）")
    print("=" * 55)

    sampler2 = RandomSampler(seed=CONFIG["seed"])
    study2 = optuna.create_study(study_name="p2_continuous", sampler=sampler2,
                                  direction="maximize")

    def obj2(trial):
        params = dict(best_discrete)  # 含全部 CONFIG 固定 + 最佳离散
        for name, spec in SEARCH_CONTINUOUS.items():
            log = spec["type"] == "loguniform"
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)

        print(f"\n{'─'*50}")
        print(f"[Phase2 {trial.number+1}/{args.p2}] "
              f"drop={params['dropout']:.3f} "
              f"e_lr={params['encoder_lr']:.1e} "
              f"c_lr={params['classifier_lr']:.1e} "
              f"wd={params['weight_decay']:.1e} "
              f"warm={params['warmup_ratio']:.3f}")
        print(f"{'─'*50}")

        ratio = 0.05 if args.test else 0.5
        metrics, avg_loss = train_and_eval(params, data_ratio=ratio)
        save_result(params, metrics, "phase2", trial_no=trial.number + 1,
                    avg_loss=avg_loss)
        print(f"  Mean Acc={metrics['mean_acc']:.4f}  "
              f"MF1={metrics['mean_f1']:.4f}  "
              f"Exact={metrics['exact_match']:.4f}  "
              f"MCC={metrics['macro_mcc']:.4f}")
        return metrics["mean_acc"]

    def obj2_with_progress(trial):
        value = obj2(trial)
        try:
            best = study2.best_value
        except ValueError:
            best = value
        print(f"\n  >>> 阶段 2 进度: {trial.number+1}/{args.p2} "
              f"| 最佳 Mean Acc: {best:.4f}\n")
        return value

    study2.optimize(obj2_with_progress, n_trials=args.p2)

    # ============ 汇总 ============
    sep = "=" * 55
    print(f"\n{sep}")
    print("优化完成")
    print(sep)
    print(f"\n阶段 1 最佳离散:")
    for k, v in study1.best_params.items():
        print(f"  {k}: {v}")
    print(f"  → Mean Acc = {study1.best_value:.4f}")
    print(f"\n阶段 2 最佳连续参数:")
    for k, v in study2.best_params.items():
        print(f"  {k}: {v}")
    print(f"  → Mean Acc = {study2.best_value:.4f}")

    best = {
        "best_mean_acc": study2.best_value,
        "best_params": study2.best_params,
        "phase1_best_discrete": study1.best_params,
        "phase1_value": study1.best_value,
    }
    with open("test/best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"\n最优配置: test/best_config.json")


if __name__ == "__main__":
    main()

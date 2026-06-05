"""
MBTI 模型训练器
==============
编排编码器 + 分类器的完整训练流程，包含：
- 训练循环 (前向 + 反向 + 优化)
- 每 epoch 验证与指标计算
- 模型保存 / 恢复
- 学习率调度 + 早停
- TensorBoard 日志

Usage:
    >>> from src.representation import RoBERTaEncoder
    >>> from src.model import MBTIClassifier, MBTITrainer, TrainingConfig
    >>> encoder = RoBERTaEncoder(pooling="mean")
    >>> classifier = MBTIClassifier(input_dim=encoder.hidden_size)
    >>> config = TrainingConfig(num_epochs=5, batch_size=16)
    >>> trainer = MBTITrainer(encoder, classifier, config)
    >>> trainer.fit("data/train.csv", "data/test.csv")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import MBTIDataset, LABEL_COLS
from src.model.classifier import MBTIClassifier, JointBCELoss
from src.representation.encoder import RoBERTaEncoder


# ============================================================
# 训练配置
# ============================================================

@dataclass
class TrainingConfig:
    """训练超参数配置"""

    # 基础参数
    num_epochs: int = 5
    batch_size: int = 16
    max_length: int = 512

    # 优化器
    learning_rate: float = 2e-5          # encoder 学习率
    classifier_lr: float = 1e-4          # 分类头学习率（通常比 encoder 大）
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 损失函数
    dim_weights: dict[str, float] = field(default_factory=lambda: {
        "EI": 0.25, "SN": 0.25, "TF": 0.25, "JP": 0.25,
    })
    pos_weights: dict[str, float] | None = None   # e.g. {"SN": 10.0}

    # 训练控制
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3

    # 保存与日志
    output_dir: str = "output"
    save_best: bool = True
    log_interval: int = 50          # 每 N 步打印训练 loss
    eval_interval: int = 0          # 每 N 步验证 (0 = 每 epoch)

    # 其他
    seed: int = 42
    fp16: bool = False              # 混合精度（需要 GPU + amp）
    num_workers: int = 0

    # 训练数据路径（可覆盖）
    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"
    label_map_path: str = "data/label_map.json"


# ============================================================
# 训练器
# ============================================================

class MBTITrainer:
    """
    MBTI 端到端训练器。

    负责:
    - 加载数据 & 创建 DataLoader
    - 编排 encoder → classifier 前向传播
    - 计算联合损失 & 反向传播
    - 评估指标 (Accuracy / F1 / AUC 逐维度 + 整体)
    - 保存最佳模型权重
    """

    def __init__(
        self,
        encoder: RoBERTaEncoder,
        classifier: MBTIClassifier,
        config: TrainingConfig | None = None,
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.config = config or TrainingConfig()

        self.device = encoder.device
        self.classifier.to(self.device)

        # ---- 损失函数 ----
        self.criterion = JointBCELoss(
            ei_weight=self.config.dim_weights["EI"],
            sn_weight=self.config.dim_weights["SN"],
            tf_weight=self.config.dim_weights["TF"],
            jp_weight=self.config.dim_weights["JP"],
            pos_weights=self.config.pos_weights,
        )

        # ---- 日志 ----
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config.output_dir) / self._timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # TensorBoard writer 按 epoch 创建，见 fit()

        # ---- 训练状态 ----
        self.global_step = 0
        self.current_epoch = 0
        self.best_score = 0.0
        self.best_epoch = 0
        self.best_epoch_dir = None
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

        # ---- 随机种子 ----
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        print(f"[Trainer] 输出目录: {self.output_dir}")
        print(f"[Trainer] 设备: {self.device}")

    # ================================================================
    # 公开接口
    # ================================================================

    def fit(
        self,
        train_csv: str | None = None,
        test_csv: str | None = None,
    ) -> dict[str, Any]:
        """完整训练流程：数据加载 → 训练循环 → 评估 → 保存。"""
        train_csv = train_csv or self.config.train_csv
        test_csv  = test_csv  or self.config.test_csv

        # ---- 1. 创建 DataLoader ----
        train_loader, test_loader = self._create_dataloaders(train_csv, test_csv)

        # ---- 2. 设置优化器 & 调度器 ----
        optimizer = self._build_optimizer()
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = self._build_scheduler(optimizer, total_steps)

        # ---- 3. 混合精度 ----
        scaler = torch.amp.GradScaler("cuda") if self.config.fp16 else None

        # ---- 4. 训练循环 ----
        print(f"\n{'='*60}")
        print(f"开始训练 — {self.config.num_epochs} epochs, "
              f"{len(train_loader)} batches/epoch")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            # 每个 epoch 独立的输出目录
            epoch_dir = self.output_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            epoch_writer = SummaryWriter(log_dir=str(epoch_dir / "logs"))

            # —— 训练一个 epoch ——
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, scaler)
            scheduler.step()  # 每个 epoch 结束后衰减一次

            # —— 验证 ——
            eval_metrics = self.evaluate(test_loader)
            eval_metrics["train_loss"] = train_loss
            eval_metrics["epoch"] = epoch
            eval_metrics["lr"] = optimizer.param_groups[0]["lr"]
            self.history.append(eval_metrics)

            # —— 日志 ——
            self._log_epoch(eval_metrics, epoch_writer)

            # —— 保存当前 epoch 模型 ——
            self.save(epoch_dir)

            # —— 更新最佳 ——
            self._checkpoint_best(eval_metrics, epoch_dir, epoch)

            # —— 早停 ——
            if self._check_early_stopping(eval_metrics["overall_acc"]):
                print(f"\n[早停] epoch {epoch}, 最佳 acc={self.best_score:.4f} "
                      f"(@epoch {self.best_epoch})")
                break

            epoch_writer.close()

        # ---- 5. 加载最佳模型 & 最终评估 ----
        self._load_best()
        final_metrics = self.evaluate(test_loader)
        print(f"\n{'='*60}")
        print("训练完成 — 测试集最终指标:")
        self._print_metrics(final_metrics)
        print(f"{'='*60}")

        # 保存配置
        self._save_config(final_metrics)

        return final_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """在给定 DataLoader 上评估模型。"""
        self.encoder.eval()
        self.classifier.eval()

        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            features = self.encoder(input_ids, attention_mask)
            logits_dict = self.classifier(features)

            loss, _ = self.criterion(logits_dict, labels)
            total_loss += loss.item()

            # 拼接概率: (B, 4) — logits → sigmoid → prob
            probs = torch.stack(
                [torch.sigmoid(logits_dict[d]) for d in self.classifier.DIMS], dim=1
            )
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)       # (N, 4)
        all_labels = np.concatenate(all_labels, axis=0)      # (N, 4)
        all_preds = (all_probs >= 0.5).astype(int)

        metrics = {
            "loss": total_loss / len(dataloader),
        }

        # 逐维度指标
        per_dim_acc = []
        per_dim_f1  = []
        per_dim_auc = []
        for i, dim in enumerate(self.classifier.DIMS):
            acc = accuracy_score(all_labels[:, i], all_preds[:, i])
            f1  = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            except ValueError:
                auc = 0.5
            metrics[f"{dim}_acc"] = acc
            metrics[f"{dim}_f1"]  = f1
            metrics[f"{dim}_auc"] = auc
            per_dim_acc.append(acc)
            per_dim_f1.append(f1)
            per_dim_auc.append(auc)

        # 整体指标
        metrics["overall_acc"] = float(
            (all_preds == all_labels).all(axis=1).mean()
        )
        metrics["mean_acc"] = float(np.mean(per_dim_acc))
        metrics["mean_f1"]  = float(np.mean(per_dim_f1))
        metrics["mean_auc"] = float(np.mean(per_dim_auc))

        return metrics

    def predict_text(self, text: str, threshold: float = 0.5) -> dict:
        """单条文本推理（供 FastAPI 等调用）。"""
        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            encoded = self.encoder.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            features = self.encoder(
                encoded["input_ids"], encoded["attention_mask"]
            )
            result = self.classifier.predict(features, threshold)

        return result

    def save(self, path: str | Path):
        """保存 encoder + classifier 权重。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path / "encoder.pt")
        torch.save(self.classifier.state_dict(), path / "classifier.pt")
        print(f"[保存] 模型已保存至 {path}")

    def load(self, path: str | Path):
        """加载权重。"""
        path = Path(path)
        self.encoder.load_state_dict(
            torch.load(path / "encoder.pt", map_location=self.device)
        )
        self.classifier.load_state_dict(
            torch.load(path / "classifier.pt", map_location=self.device)
        )
        print(f"[加载] 模型已从 {path} 加载")

    # ================================================================
    # 内部方法
    # ================================================================

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler,
    ) -> float:
        self.encoder.train()
        self.classifier.train()

        total_loss = 0.0
        n_batches = len(loader)

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # ---- 前向 + 反向 ----
            if scaler:
                with torch.amp.autocast("cuda"):
                    features = self.encoder(input_ids, attention_mask)
                    probs = self.classifier(features)
                    loss, _ = self.criterion(probs, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                features = self.encoder(input_ids, attention_mask)
                probs = self.classifier(features)
                loss, _ = self.criterion(probs, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            # ---- 梯度累积 ----
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self._trainable_params(), self.config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self._trainable_params(), self.config.max_grad_norm
                    )
                    optimizer.step()
                optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # 日志
            if batch_idx > 0 and batch_idx % self.config.log_interval == 0:
                print(f"  Epoch {self.current_epoch} | "
                      f"Batch {batch_idx}/{n_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # 中途评估
            if (self.config.eval_interval > 0
                    and self.global_step % self.config.eval_interval == 0):
                # 仅在没有 test_loader 时跳过（fit 中已有 per-epoch eval）
                pass

        return total_loss / n_batches

    def _create_dataloaders(
        self, train_csv: str, test_csv: str
    ) -> tuple[DataLoader, DataLoader]:
        train_ds = MBTIDataset(train_csv, self.encoder.tokenizer, self.config.max_length)
        test_ds  = MBTIDataset(test_csv,  self.encoder.tokenizer, self.config.max_length)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        print(f"[数据] Train={len(train_ds):,} Test={len(test_ds):,}")
        return train_loader, test_loader

    def _build_optimizer(self) -> AdamW:
        """分层学习率：encoder 较低，classifier 较高。"""
        return AdamW([
            {"params": self.encoder.parameters(),
             "lr": self.config.learning_rate},
            {"params": self.classifier.parameters(),
             "lr": self.config.classifier_lr},
        ], weight_decay=self.config.weight_decay)

    def _build_scheduler(
        self, optimizer: AdamW, total_steps: int
    ) -> CosineAnnealingLR:
        # 按 epoch 衰减：scheduler.step() 在 _train_epoch 结束时调用
        return CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs, eta_min=1e-7,
        )

    def _trainable_params(self):
        return [p for p in self.encoder.parameters() if p.requires_grad] + \
               [p for p in self.classifier.parameters() if p.requires_grad]

    def _checkpoint_best(
        self, metrics: dict, epoch_dir: Path, epoch: int
    ):
        score = metrics.get("overall_acc", 0)
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_dir = epoch_dir
            self.patience_counter = 0
            if self.config.save_best:
                self.save(self.output_dir / "best")
                print(f"  [✓] 新最佳模型 (acc={score:.4f}) → best/")
        else:
            self.patience_counter += 1

    def _check_early_stopping(self, score: float) -> bool:
        return self.patience_counter >= self.config.early_stopping_patience

    def _load_best(self):
        best_path = self.output_dir / "best"
        if best_path.exists():
            self.load(best_path)
        elif self.best_epoch_dir is not None:
            self.load(self.best_epoch_dir)

    def _save_config(self, final_metrics: dict):
        """保存训练配置和最终指标到 JSON。"""
        info = {
            "model": self.encoder.model_name,
            "pooling": self.encoder.pooling_name,
            "hidden_size": self.encoder.hidden_size,
            "config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "learning_rate": self.config.learning_rate,
                "classifier_lr": self.config.classifier_lr,
            },
            "best_epoch": self.best_epoch,
            "best_overall_acc": self.best_score,
            "final_metrics": {k: v for k, v in final_metrics.items()
                              if isinstance(v, (float, int))},
        }
        with open(self.output_dir / "training_info.json", "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    # ================================================================
    # 日志辅助
    # ================================================================

    def _log_epoch(self, metrics: dict, writer: SummaryWriter):
        """打印 epoch 级别指标 & 写 TensorBoard。"""
        print(f"\n--- Epoch {metrics['epoch']} ---")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Test Loss:  {metrics['loss']:.4f}")
        print(f"  Per-dim: ", end="")
        for dim in self.classifier.DIMS:
            print(f"{dim}_acc={metrics[f'{dim}_acc']:.3f} "
                  f"{dim}_f1={metrics[f'{dim}_f1']:.3f} "
                  f"{dim}_auc={metrics[f'{dim}_auc']:.3f}  ", end="")
        print(f"\n  Overall Acc: {metrics['overall_acc']:.4f}  "
              f"Mean Acc: {metrics['mean_acc']:.4f}")
        print()

        # TensorBoard
        step = metrics["epoch"]
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                writer.add_scalar(f"eval/{k}", v, step)

    @staticmethod
    def _print_metrics(metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

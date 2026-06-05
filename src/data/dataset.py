"""
MBTI 数据集 PyTorch Dataset 封装
================================
加载预处理后的 CSV，使用编码器的 tokenizer 进行动态分词，
输出模型可直接消费的 tensor 格式 (input_ids, attention_mask, labels)。

Usage:
    >>> from src.representation import RoBERTaEncoder
    >>> from src.data import MBTIDataset
    >>> encoder = RoBERTaEncoder()
    >>> train_ds = MBTIDataset("data/train.csv", encoder.tokenizer)
    >>> loader = DataLoader(train_ds, batch_size=16, shuffle=True)
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

LABEL_COLS = ["label_EI", "label_SN", "label_TF", "label_JP"]


class MBTIDataset(Dataset):
    """
    加载预处理后的 MBTI CSV 数据，在线 tokenize 并返回训练用 tensor。

    Args:
        csv_path: 预处理后 CSV 路径（含 text, type, label_EI/SN/TF/JP 列）
        tokenizer: HuggingFace RoBERTa tokenizer
        max_length: token 截断上限
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        df = pd.read_csv(csv_path)

        self.texts = df["text"].astype(str).tolist()
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (max_len,)
            "labels":         self.labels[idx],                      # (4,)
        }

    # ——— 标签统计 ———

    @property
    def num_samples(self) -> int:
        return len(self)

    def label_distribution(self) -> dict[str, dict[str, int]]:
        """返回每维度的正/负样本数。"""
        dist = {}
        for i, col in enumerate(LABEL_COLS):
            pos = int(self.labels[:, i].sum())
            neg = len(self) - pos
            dim = col.replace("label_", "")
            dist[dim] = {"pos": pos, "neg": neg}
        return dist


# ============================================================
# 工具函数
# ============================================================

def create_dataloaders(
    train_csv: str = "data/train.csv",
    test_csv: str = "data/test.csv",
    tokenizer: PreTrainedTokenizerBase | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    一键创建 train / test DataLoader。

    Args:
        train_csv: 训练集 CSV 路径
        test_csv:  测试集 CSV 路径
        tokenizer: RoBERTa tokenizer（若为 None 则自动加载 roberta-base）
        batch_size: 批大小
        max_length: token 截断上限
        num_workers: DataLoader 工作进程数

    Returns:
        (train_loader, test_loader)
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_ds = MBTIDataset(train_csv, tokenizer, max_length)
    test_ds  = MBTIDataset(test_csv,  tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {train_ds.num_samples:,} samples, "
          f"{len(train_loader)} batches (bs={batch_size})")
    print(f"Test : {test_ds.num_samples:,} samples, "
          f"{len(test_loader)} batches (bs={batch_size})")
    return train_loader, test_loader

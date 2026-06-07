"""
MBTI 四维分类模型
================
接收 768 维语义向量，直接分四路独立分类头，
各维度输出 [0,1] 间概率值。

架构:
    特征向量 (768-dim)
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
  EI    SN   TF   JP  Head  (768 → 64 → 1, Sigmoid)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 分类头
# ============================================================

class DimensionHead(nn.Module):
    """单维度二分类头: input_dim → 64 → 1 (输出 logits，兼容 FP16)"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)  # (batch,) logits


# ============================================================
# 四维分类器
# ============================================================

class MBTIClassifier(nn.Module):
    """
    MBTI 四维二分类器，接收编码器输出 (768-dim) 并输出四个概率。

    Attributes:
        dims (list[str]): 维度名称 ["EI", "SN", "TF", "JP"]
    """

    DIMS = ["EI", "SN", "TF", "JP"]

    def __init__(
        self,
        input_dim: int = 768,
        head_hidden: int = 64,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: 编码器输出维度 (roberta-base = 768)
            head_hidden: 各分类头隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()

        # ---- 四个独立分类头（直接从 768 维输入）----
        self.heads = nn.ModuleDict({
            dim: DimensionHead(input_dim, head_hidden, dropout)
            for dim in self.DIMS
        })

        self._input_dim = input_dim

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, input_dim) 编码器语义向量

        Returns:
            dict with keys "EI", "SN", "TF", "JP" → (batch,) probabilities
        """
        return {dim: head(features) for dim, head in self.heads.items()}

    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> dict:
        """
        推理接口：返回概率 + 离散预测 + MBTI 类型。

        Returns:
            {
                "probabilities": {"EI": float, ...},
                "predictions":  {"EI": 0|1, ...},
                "mbti_type": "INFP",
                "confidence": float  (四维置信度均值)
            }
        """
        with torch.no_grad():
            logits = self.forward(features)
            probs = {dim: torch.sigmoid(logits[dim]) for dim in self.DIMS}

        result = {"probabilities": {}, "predictions": {}}
        mbti_chars = ["E" if probs["EI"].item() >= threshold else "I",
                       "S" if probs["SN"].item() >= threshold else "N",
                       "T" if probs["TF"].item() >= threshold else "F",
                       "J" if probs["JP"].item() >= threshold else "P"]

        for i, dim in enumerate(self.DIMS):
            p = probs[dim].item()
            result["probabilities"][dim] = {
                "positive": round(p, 4),
                "negative": round(1 - p, 4),
            }
            result["predictions"][dim] = 1 if p >= threshold else 0

        result["mbti_type"] = "".join(mbti_chars)
        result["confidence"] = round(sum(
            abs(p - 0.5) * 2 for p in [probs[d].item() for d in self.DIMS]
        ) / 4, 4)

        return result

    @property
    def input_dim(self) -> int:
        return self._input_dim


# ============================================================
# 联合 BCE 损失函数
# ============================================================

class JointBCELoss(nn.Module):
    """
    四维加权 BCE 联合损失: L_total = Σ λ_i * BCE(prob_i, label_i)

    默认等权重 λ=0.25；可根据各维度样本不均衡程度调整，
    例如 SN 维度严重不均衡 (91% N)，可适当提高其权重。
    """

    def __init__(
        self,
        ei_weight: float = 0.25,
        sn_weight: float = 0.25,
        tf_weight: float = 0.25,
        jp_weight: float = 0.25,
        pos_weights: dict[str, float] | None = None,
    ):
        """
        Args:
            ei/sn/tf/jp_weight: 各维度损失权重（建议和为 1.0）
            pos_weights: 每维度正样本 BCE pos_weight（缓解类别不均衡）
                         e.g. {"SN": 10.0} 加大少数类 S(0) 的惩罚
        """
        super().__init__()
        self.dim_weights = {"EI": ei_weight, "SN": sn_weight,
                            "TF": tf_weight, "JP": jp_weight}
        self.pos_weights = pos_weights or {}

    def forward(
        self,
        probs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            probs: {"EI": (B,), "SN": (B,), ...} 预测概率
            labels: (B, 4) 真实标签, 列序: [EI, SN, TF, JP]

        Returns:
            (total_loss, per_dim_losses)
        """
        total = torch.tensor(0.0, device=labels.device)
        per_dim = {}

        for i, dim in enumerate(["EI", "SN", "TF", "JP"]):
            pos_w = self.pos_weights.get(dim, None)
            w = torch.tensor(pos_w, device=labels.device) if pos_w else None
            # FP16 下强制 loss 用 FP32 计算，避免溢出
            logit = probs[dim].float()
            label = labels[:, i].float()
            loss = F.binary_cross_entropy_with_logits(
                logit, label, pos_weight=w,
            )
            total += self.dim_weights[dim] * loss
            per_dim[dim] = loss.item()

        return total, per_dim

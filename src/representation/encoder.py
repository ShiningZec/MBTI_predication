"""
RoBERTa 语义编码器
=================

将清洗后的英文文本映射为 768 维稠密语义向量，作为下游分类器的输入特征。

支持的 Pooling 策略:
    - CLS Token: 取 <s> 位置的最后一层隐藏状态 (RoBERTa 没有 [CLS]，用 <s>)
    - Mean Pooling: 所有 token 取均值（含注意力掩码加权），信息利用最充分 [推荐]
    - Max Pooling: 每个维度取最大值，突出关键特征信号
    - Attention-Weighted: 使用最后一层注意力权重做加权平均

模型选型:
    - 主模型: roberta-base (12层, 768维, 125M参数)
    - 轻量备选: distilroberta-base (6层, 768维, 82M参数)

Usage:
    >>> encoder = RoBERTaEncoder(pooling="mean")
    >>> vectors = encoder.encode(["This is a sample text.", "Another one."])
    >>> print(vectors.shape)  # (2, 768)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# 项目根目录（用于定位本地模型缓存）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / "models")


# ============================================================
# Pooling 策略
# ============================================================

class PoolingStrategy(nn.Module):
    """Pooling 策略基类。"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) 最后一层隐藏状态
            attention_mask: (batch, seq_len) 注意力掩码, 1=有效token, 0=padding
        Returns:
            pooled: (batch, hidden_size) 池化后的句子向量
        """
        raise NotImplementedError


class CLSPooling(PoolingStrategy):
    """取序列第一个 token (<s>) 的隐藏状态作为句子表征。"""

    def forward(self, hidden_states, attention_mask, **kwargs):
        return hidden_states[:, 0, :]  # (batch, hidden_size)


class MeanPooling(PoolingStrategy):
    """对所有 token 取均值，padding 位置不参与计算。"""

    def forward(self, hidden_states, attention_mask, **kwargs):
        # 将 mask 扩展为与 hidden_states 相同的维度
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # 加权求和 / 有效token数
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class MaxPooling(PoolingStrategy):
    """每个维度取最大值（padding 位置置为极小值）。"""

    def forward(self, hidden_states, attention_mask, **kwargs):
        # padding 位置设为极小值，使其不会被 max 选中
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden = hidden_states.clone()
        hidden[mask_expanded == 0] = -1e9
        return torch.max(hidden, dim=1)[0]


class AttentionPooling(PoolingStrategy):
    """
    使用最后一层 self-attention 权重做加权平均。
    取所有 head 的注意力均值，对 token 维度加权求和。
    """

    def forward(self, hidden_states, attention_mask, attentions=None, **kwargs):
        if attentions is None:
            # 回退到 mean pooling
            return MeanPooling()(hidden_states, attention_mask)

        # attentions: tuple of (batch, num_heads, seq_len, seq_len)
        # 取最后一层，对所有 head 取平均，取 <s> 对所有 token 的注意力
        last_attn = attentions[-1]                     # (B, H, L, L)
        attn_weights = last_attn.mean(dim=1)[:, 0, :]  # (B, L) — <s> 关注的权重

        # 用 mask 清零 padding 位置
        attn_weights = attn_weights * attention_mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)

        # 加权求和
        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        return pooled


# ============================================================
# 策略注册表
# ============================================================

POOLING_REGISTRY: dict[str, type[PoolingStrategy]] = {
    "cls":       CLSPooling,
    "mean":      MeanPooling,
    "max":       MaxPooling,
    "attention": AttentionPooling,
}


def get_pooling(name: str) -> PoolingStrategy:
    """根据名称获取池化策略实例。"""
    name = name.lower()
    if name not in POOLING_REGISTRY:
        raise ValueError(f"未知 pooling: '{name}'，可选: {list(POOLING_REGISTRY.keys())}")
    return POOLING_REGISTRY[name]()


# ============================================================
# RoBERTa 编码器
# ============================================================

class RoBERTaEncoder(nn.Module):
    """
    RoBERTa 语义编码器。

    封装 HuggingFace RoBERTa 模型，输出指定 pooling 策略下的句子级向量。

    Attributes:
        model_name (str): HuggingFace 模型标识
        hidden_size (int): 输出向量维度 (768 for roberta-base)
        max_length (int): 最大输入 token 数
        pooling_name (str): 使用的 pooling 策略名称
    """

    # 推荐的英文 RoBERTa 模型（按计算量排序）
    AVAILABLE_MODELS = {
        "roberta-base":             {"params": "125M", "layers": 12, "dim": 768},
        "roberta-large":            {"params": "355M", "layers": 24, "dim": 1024},
        "distilroberta-base":       {"params": "82M",  "layers": 6,  "dim": 768},
    }

    def __init__(
        self,
        model_name: str = "roberta-base",
        pooling: str = "mean",
        max_length: int = 512,
        device: str | torch.device | None = None,
        freeze_backbone: bool = False,
        output_attentions: bool = False,
        cache_dir: str | None = None,
    ):
        """
        Args:
            model_name: HuggingFace RoBERTa 模型标识 (e.g. "roberta-base")
            pooling: pooling 策略 ("cls" | "mean" | "max" | "attention")
            max_length: tokenizer 最大截断长度 (RoBERTa 上限 514)
            device: 指定设备 ("cuda" | "cpu")，None 则自动选择
            freeze_backbone: 是否冻结 RoBERTa 权重（仅用作特征提取）
            output_attentions: 是否输出注意力权重（attention pooling 需要）
            cache_dir: 模型缓存目录，默认使用项目根目录下的 models/
        """
        super().__init__()

        self.model_name = model_name
        self.pooling_name = pooling
        self.max_length = max_length

        # ---- 缓存目录 ----
        if cache_dir is None:
            cache_dir = _DEFAULT_CACHE_DIR

        # ---- 设备 ----
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- 加载模型 & 分词器 ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # 确保 output_attentions 在 attention pooling 时开启
        self._need_attentions = output_attentions or (pooling == "attention")
        if self._need_attentions:
            self.backbone.config.output_attentions = True

        # ---- 冻结策略 ----
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Pooling ----
        self.pooling = get_pooling(pooling)

        # ---- 元信息 ----
        self.hidden_size = self.backbone.config.hidden_size
        self.num_params = sum(p.numel() for p in self.backbone.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )

        self.to(self.device)

    # -------- 公开接口 --------

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        将文本列表编码为语义向量（CPU tensor，适合下游使用）。

        Args:
            texts: 单个文本字符串或字符串列表
            batch_size: 分批推理时的 batch 大小
            show_progress: 是否显示进度条

        Returns:
            Tensor of shape (N, hidden_size)，位于 CPU 上
        """
        if isinstance(texts, str):
            texts = [texts]

        was_training = self.training
        self.eval()

        all_vectors = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.backbone(**{
                    k: v for k, v in encoded.items()
                    if k in ("input_ids", "attention_mask")
                })

                pooled = self.pooling(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                    attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
                )

                all_vectors.append(pooled.cpu())

        if was_training:
            self.train()

        return torch.cat(all_vectors, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播（供训练循环调用，保留梯度）。

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            pooled: (batch, hidden_size)，梯度保留
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self._need_attentions,
        )

        return self.pooling(
            outputs.last_hidden_state,
            attention_mask,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    # -------- 属性 --------

    @property
    def embedding_dim(self) -> int:
        return self.hidden_size

    @property
    def num_parameters(self) -> int:
        return self.num_params

    def __repr__(self) -> str:
        return (
            f"RoBERTaEncoder(\n"
            f"  model={self.model_name},\n"
            f"  pooling={self.pooling_name},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  max_length={self.max_length},\n"
            f"  params={self.num_params:,} (trainable={self.trainable_params:,}),\n"
            f"  device={self.device}\n"
            f")"
        )

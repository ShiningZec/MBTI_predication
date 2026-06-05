"""
注意力权重提取与可视化数据生成
==============================
从 RoBERTa 编码器最后一层提取多头注意力权重，
生成前端可视化所需的热力图数据结构。

Usage:
    >>> extractor = AttentionExtractor(encoder)
    >>> data = extractor.extract("I enjoy reading books alone.")
    >>> # data["layers"][-1]["attention"] → (num_heads, seq_len, seq_len)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class AttentionExtractor:
    """
    从 RoBERTa 编码器的 forward 中提取各层注意力权重。

    需要在初始化编码器时设置 `output_attentions=True`。
    """

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer
        self.device = next(encoder.parameters()).device

    @torch.no_grad()
    def extract(self, text: str, max_length: int = 512) -> dict:
        """
        提取文本在编码器中所有层的注意力权重。

        Args:
            text: 输入文本
            max_length: token 截断上限

        Returns:
            {
                "tokens": ["<s>", "I", " enjoy", ...],     # 分词结果
                "layers": [
                    { "attention": (num_heads, L, L) },    # 归一化后的注意力
                    ...
                ],
                "merged_attention": (L, L),                 # 所有层 & head 均值
                "cls_attention": (L,),                      # <s> 对各 token 的注意力
                "num_layers": int,
                "num_heads": int,
            }
        """
        self.encoder.eval()

        # 临时开启 attention 输出（需先切 eager 模式，SDPA 不支持 output_attentions）
        original_attn = self.encoder.backbone.config.output_attentions
        original_impl = self.encoder.backbone.config._attn_implementation
        self.encoder.backbone.config._attn_implementation = "eager"
        self.encoder.backbone.config.output_attentions = True

        # ---- Tokenize ----
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        # ---- Forward ----
        outputs = self.encoder.backbone(**{
            k: v for k, v in encoded.items()
            if k in ("input_ids", "attention_mask")
        })

        # 恢复原设置
        self.encoder.backbone.config.output_attentions = original_attn
        self.encoder.backbone.config._attn_implementation = original_impl

        # ---- 获取注意力 ----
        # attentions: tuple of (batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions

        if attentions is None:
            raise RuntimeError(
                "编码器未输出注意力权重。请在初始化时设 output_attentions=True。"
            )

        valid_len = encoded["attention_mask"].sum().item()
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0, :int(valid_len)]
        )

        # ---- 逐层整理 ----
        layers_data = []
        all_attns = []

        for layer_idx, attn in enumerate(attentions):
            # attn: (1, H, L, L) → (H, valid_len, valid_len)
            a = attn[0, :, :valid_len, :valid_len].cpu().numpy()
            all_attns.append(a)
            layers_data.append({
                "layer": layer_idx,
                "attention": a.tolist(),  # 转为 JSON 可序列化
            })

        # ---- 汇总 ----
        merged = np.mean(all_attns, axis=(0,)).tolist()  # 所有层 & head 均值

        # <s> (token 0) 对各 token 的注意力 (所有 head 均值)
        cls_to_all = np.mean(
            [a[:, 0, :].mean(axis=0) for a in all_attns], axis=0
        ).tolist()

        return {
            "tokens": tokens,
            "layers": layers_data,
            "merged_attention": merged,
            "cls_attention": cls_to_all,
            "num_layers": len(attentions),
            "num_heads": attentions[0].shape[1],
        }

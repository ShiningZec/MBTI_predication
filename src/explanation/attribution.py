"""
Integrated Gradients 特征归因
=============================
使用 Captum 计算每个 token 对各 MBTI 维度的预测贡献值。

原理: Integrated Gradients 从 baseline (零向量) 到输入插值，
      沿路径对梯度积分，得到每个输入特征的归因分数。

Usage:
    >>> analyzer = AttributionAnalyzer(encoder, classifier)
    >>> result = analyzer.analyze("我喜欢独处和思考")
    >>> for dim, tokens in result["keywords"].items():
    >>>     print(dim, tokens[:5])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

# Captum 为可选依赖 —— 未安装时降级为梯度归因
try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


class AttributionAnalyzer:
    """
    基于 Integrated Gradients 的 token 级归因分析器。

    Attributes:
        encoder: RoBERTa 编码器
        classifier: MBTI 四维分类器
        tokenizer: 对应的 tokenizer
        dims: ["EI", "SN", "TF", "JP"]
    """

    DIMS = ["EI", "SN", "TF", "JP"]

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.tokenizer = tokenizer or encoder.tokenizer
        self.device = next(encoder.parameters()).device

        # 获取 embedding 层引用（用于 Captum LayerIG）
        self._embedding_layer = None
        for module in encoder.modules():
            if isinstance(module, nn.Embedding):
                self._embedding_layer = module
                break

    @torch.enable_grad()
    def analyze(
        self,
        text: str,
        n_steps: int = 50,
        top_k: int = 10,
        internal_batch_size: int = 8,
    ) -> dict:
        """
        对单条文本进行归因分析。

        Args:
            text: 输入文本
            n_steps: IG 积分步数
            top_k: 每维度返回的 Top-K 关键词数
            internal_batch_size: 内部批大小

        Returns:
            {
                "tokens": [...],
                "keywords": { "EI": [...], ... },
                "attribution": { "EI": [...], ... },
                "probabilities": {...},
            }
        """
        self.encoder.eval()
        self.classifier.eval()
        # 确保 embedding 层梯度开启（IG 需要）
        if self._embedding_layer is not None:
            self._embedding_layer.weight.requires_grad_(True)

        # ---- Tokenize ----
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # ---- 解码 token 文本 ----
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # 仅保留有效 token（去掉 special + padding）
        valid_len = attention_mask.sum().item()
        valid_tokens = tokens[:int(valid_len)]

        # ---- 预测概率 ----
        features = self.encoder(input_ids, attention_mask)
        logits = self.classifier(features)
        probs = {dim: torch.sigmoid(logits[dim]) for dim in self.DIMS}

        # ---- Integrated Gradients ----
        if HAS_CAPTUM:
            attributions = self._compute_ig(
                input_ids, attention_mask, n_steps, internal_batch_size
            )
        else:
            attributions = self._compute_gradient(
                input_ids, attention_mask
            )

        # ---- 聚合 (取 embedding 维度的均值作为 token 级分数) ----
        token_scores = {}
        for i, dim in enumerate(self.DIMS):
            scores = attributions[dim][0, :valid_len].mean(dim=-1)
            # 转为 Python float 列表
            token_scores[dim] = scores.cpu().tolist()

        # ---- 提取 Top-K 关键词 ----
        keywords = {}
        for dim in self.DIMS:
            scored = [
                {"token": valid_tokens[j], "score": round(token_scores[dim][j], 4)}
                for j in range(len(valid_tokens))
                # 过滤特殊 token
                if not valid_tokens[j].startswith("<")
            ]
            scored.sort(key=lambda x: abs(x["score"]), reverse=True)
            keywords[dim] = scored[:top_k]

        probabilities = {
            dim: {
                "positive": round(probs[dim].item(), 4),
                "negative": round(1 - probs[dim].item(), 4),
            }
            for dim in self.DIMS
        }

        return {
            "tokens": valid_tokens,
            "keywords": keywords,
            "attribution": token_scores,
            "probabilities": probabilities,
        }

    # ============================================================
    # 内部
    # ============================================================

    def _compute_ig(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_steps: int,
        internal_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """使用 Captum Integrated Gradients 计算归因。"""

        # baseline: 全 PAD token embedding
        baseline_ids = torch.zeros_like(input_ids)
        baseline_ids[:, 0] = self.tokenizer.cls_token_id or 0
        baseline_ids[:, -1] = self.tokenizer.sep_token_id or 2

        if self._embedding_layer is not None:
            ig = LayerIntegratedGradients(self._forward_wrapper, self._embedding_layer)
            kwargs = {
                "inputs": input_ids,
                "baselines": baseline_ids,
                "additional_forward_args": (attention_mask,),
                "n_steps": n_steps,
                "internal_batch_size": internal_batch_size,
            }
        else:
            ig = IntegratedGradients(self._forward_wrapper)
            kwargs = {
                "inputs": input_ids.float(),
                "baselines": baseline_ids.float(),
                "additional_forward_args": (attention_mask,),
                "n_steps": n_steps,
            }

        attributions = {}
        for i, dim in enumerate(self.DIMS):
            attr = ig.attribute(
                target=i,  # 第 i 个输出维度
                **kwargs,
            )
            attributions[dim] = attr

        return attributions

    def _compute_gradient(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        降级方案：直接用梯度 × 输入作为近似归因
        (Saliency Maps, 不需要 Captum)
        """
        input_embeds = self._embedding_layer(input_ids) if self._embedding_layer is not None else input_ids.float()
        input_embeds.retain_grad()

        # 用 inputs_embeds 替代 input_ids，保持梯度链
        features = self.encoder.backbone(
            inputs_embeds=input_embeds.detach().requires_grad_(True),
            attention_mask=attention_mask,
        )
        # 手动 pooling（简化：mean pooling）
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (features.last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        probs = self.classifier(pooled)

        attributions = {}
        for i, dim in enumerate(self.DIMS):
            self.encoder.zero_grad()
            self.classifier.zero_grad()
            grad_out = torch.autograd.grad(
                probs[dim].sum(), input_embeds, retain_graph=True,
            )[0]
            attributions[dim] = (grad_out * input_embeds).detach()

        return attributions

    def _forward_wrapper(self, input_ids, attention_mask=None, **kwargs):
        """Captum 需要的前向函数签名。"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        features = self.encoder(input_ids, attention_mask)
        logits = self.classifier(features)
        return torch.stack([logits[d] for d in self.DIMS], dim=-1)

"""
表征层模块 — RoBERTa 语义编码器
==============================
将自然语言文本转换为 768 维稠密语义向量，供下游 MBTI 四维分类器使用。
"""

from .encoder import RoBERTaEncoder

__all__ = ["RoBERTaEncoder"]

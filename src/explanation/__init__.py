"""
解释层模块 — 模型预测可解释性
=============================
将黑盒预测转化为用户可理解的解释，包括：
- Integrated Gradients 特征归因
- 注意力权重可视化
- 模板化 NLG 人格解读
"""

from .attribution import AttributionAnalyzer
from .attention import AttentionExtractor
from .interpreter import MBTIInterpreter

__all__ = ["AttributionAnalyzer", "AttentionExtractor", "MBTIInterpreter"]

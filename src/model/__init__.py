"""
任务层模块 — MBTI 四维分类器与训练器
====================================
基于 768 维语义向量，通过共享表示层 + 四个独立分类头
输出 E/I、S/N、T/F、J/P 四维二分类概率。
"""

from .classifier import MBTIClassifier, JointBCELoss
from .trainer import MBTITrainer, TrainingConfig

__all__ = ["MBTIClassifier", "JointBCELoss", "MBTITrainer", "TrainingConfig"]

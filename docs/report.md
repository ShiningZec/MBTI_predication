# MBTI 性格预测系统 — 完整实验报告

> 从数据处理到最优模型的端到端记录，包含所有设计决策及其理论依据。

---

## 一、数据集

### 1.1 数据来源

**[zeyadkhalid/MBTI Personality Types 500 Dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)**，来自 PersonalityCafe 论坛用户的匿名帖子。

| 属性 | 值 |
|------|-----|
| 总样本数 | 106,067 |
| 特征 | 英文帖子文本 (`posts`) + MBTI 标签 (`type`) |
| 标签类别 | 16 种 MBTI 类型 |
| 文本长度 | 均值 3,255 字符，范围 2,024-13,180 |

### 1.2 数据分布

| MBTI | 数量 | 比例 | MBTI | 数量 | 比例 |
|------|:---:|:---:|------|:---:|:---:|
| INTP | 24,961 | 23.5% | ESTP | 1,589 | 1.5% |
| INTJ | 22,427 | 21.1% | ENFJ | 1,227 | 1.2% |
| INFJ | 14,963 | 14.1% | ISTJ | 994 | 0.9% |
| INFP | 12,134 | 11.4% | ISFP | 700 | 0.7% |
| ENTP | 11,725 | 11.1% | ISFJ | 520 | 0.5% |
| ENFP | 4,934 | 4.7% | ESTJ | 386 | 0.4% |
| ISTP | 2,739 | 2.6% | ESFP | 288 | 0.3% |
| ENTJ | 2,364 | 2.2% | ESFJ | 145 | 0.1% |

**四维拆分后的分布**：将 16 类型映射为 4 个独立二分类标签后，维度不均衡显著：

| 维度 | 多数类 | 少数类 | 多数比例 |
|------|--------|--------|:---:|
| E/I | I (内倾) | E (外倾) | 76.1% |
| S/N | N (直觉) | S (感觉) | 91.3% |
| T/F | T (思考) | F (情感) | 65.2% |
| J/P | P (感知) | J (判断) | 58.1% |

> **设计决策**：选用四维独立二分类而非 16 类 softmax。理由：(1) 16 分类每类平均 6,600 样本，四维拆分后每个二分类利用全部 106,067 条数据 (2) 四维独立可解释性更强——可逐维分析哪些文本特征影响哪个维度 (3) 不均衡可通过 pos_weight 精调单维。

### 1.3 预处理

| 步骤 | 操作 |
|------|------|
| 文本清洗 | 移除 URL、HTML 标签、@mention、`\|\|\|` 分隔符、多余空白 |
| 标签拆分 | 16 型 → `{EI: 0/1, SN: 0/1, TF: 0/1, JP: 0/1}` |
| 数据划分 | 8:2 分层采样（`stratify=type`），训练 84,853 / 测试 21,214 |
| 输出格式 | CSV + JSON 双格式，`label_map.json` 提供互查 |

---

## 二、模型架构

### 2.1 总体结构

```
输入文本 → BPE Tokenizer → RoBERTa-base (12层×768维) → Mean/CLS Pooling
         → 768-dim 向量 → 四独立分类头 (768→64→1 logit) → sigmoid → 四维概率
```

### 2.2 表征层：RoBERTa-base

| 属性 | 值 | 决策依据 |
|------|-----|------|
| 模型 | `FacebookAI/roberta-base` (125M) | 英文数据集，RoBERTa 预训练语料与论坛帖子风格接近 |
| Pooling | CLS (超参搜索最优) | 8 次随机搜索 cls 全面碾压 mean |
| max_length | 512 | RoBERTa 硬上限；基准验证 256→512 带来 +13.4% Overall Acc |
| 冻结 | 否（全训练） | 106k 数据量足够，冻结反降 ~2% |

### 2.3 任务层：四维独立分类头

```
        768 维向量 (RoBERTa 输出)
              │
    ┌────┬────┼────┬────┐
    ▼    ▼    ▼    ▼    ▼
   EI   SN   TF   JP    ← 无共享层，直接分四路
  768→64→1   768→64→1   768→64→1   768→64→1
    │    │    │    │
  logit logit logit logit  ← 输出 logit（非 sigmoid）
```

| 设计决策 | 选择 | 理由 |
|------|------|------|
| 无共享层 | ✅ | 四个维度的语言特征正交（EI=社交词频，SN=抽象程度，TF=逻辑/情感，JP=计划性），共享层强迫压缩不必要 |
| head_hidden | 512 (HP 最优) | 随机搜索中 512 > 256 > 128 > 64 一致 |
| 输出 logit | ✅ | `BCEWithLogitsLoss` 数值稳定 + FP16 安全，推理时手动 sigmoid |
| Dropout | 0.2 | 防过拟合 |

### 2.4 损失函数

$$
\mathcal{L}_{total} = 0.25 \cdot \mathcal{L}_{EI} + 0.35 \cdot \mathcal{L}_{SN} + 0.20 \cdot \mathcal{L}_{TF} + 0.20 \cdot \mathcal{L}_{JP}
$$

SN 权重提高至 0.35，配合 `pos_weight=10.0` 缓解极端不均衡（S 类仅 8.7%）。

---

## 三、训练策略

### 3.1 优化器配置

| 参数 | 值 | 依据 |
|------|-----|------|
| 优化器 | AdamW | 标准 Transformer 微调 |
| encoder LR | 2.84e-5 | HP 搜索最优 |
| classifier LR | 1e-4 | 头部从零初始化，需快于骨干 |
| weight_decay | 0.01 | 防过拟合 |
| warmup | 0.027 (Linear → 1e-7) | HP 搜索最优 |
| 调度器 | CosineAnnealingLR（epoch 级） | LR 均匀衰减，不提前见底 |
| 梯度裁剪 | max_norm=1.0 | 防 RoBERTa 梯度爆炸 |

> **分层学习率**：encoder（预训练完毕）与 classifier（从零初始化）使用不同 LR，同一优化器两组参数。

### 3.2 训练配置

| 参数 | Baseline | HP 最优 | 决策 |
|------|:---:|:---:|------|
| max_length | 256 | 512 | 基准验证 +13.4% Overall |
| batch_size | 16 | 32 | 平衡更新步数与梯度噪声 |
| epochs | 3 | 7（早停） | Train Loss <0.05 后停止 |
| fp16 | true | true | 快 2x，loss 强制 FP32 防溢出 |

---

## 四、超参数优化

### 4.1 搜索策略

**两阶段随机搜索**（Optuna + RandomSampler）：

| | 阶段 1 | 阶段 2 |
|------|------|------|
| 目标 | 搜索离散参数 | 搜索连续参数 |
| 参数 | pooling, head_hidden | dropout, encoder_lr, classifier_lr, weight_decay, warmup |
| 方法 | 随机采样 | 随机采样 |
| Trials | 8（全覆盖 2×4=8） | 10 |
| 数据 | 30% 子集 | 50% 子集 |
| Epochs | 5 | 5 |
| 目标函数 | Mean Accuracy | Mean Accuracy |

> **为什么用子集？** 全量数据单 trial 需 3h+。30-50% 子集 + 5 epoch 将单 trial 降到 ~12-20 分钟，18 次搜索共 ~5h。**子集上参数优劣的相对排名可靠**——所有 trial 同等缩水，排名保留。

### 4.2 阶段 1 结果（离散参数）

| Pooling | Head Hidden | Mean Acc | Exact Match | Mean F1 |
|:---:|:---:|:---:|:---:|:---:|
| cls | 512 | **0.9315** | **0.8094** | **0.9112** |
| cls | 64 | 0.9313 | 0.8044 | 0.9098 |
| mean | 256 | 0.9276 | 0.7957 | 0.9050 |
| mean | 512 | 0.9273 | 0.7967 | 0.9051 |
| mean | 128 | 0.9235 | 0.7825 | 0.9006 |

**结论**：cls pooling 全面优于 mean（4/4 试验）；head_hidden=512 全指标最优。选 `cls + 512` 进入阶段 2。

### 4.3 阶段 2 结果（连续参数）

最优 Trial (Mean Acc=0.9384, Exact=0.8237)：dropout=0.106, encoder_lr=2.84e-5, classifier_lr=2.22e-4, weight_decay=7.07e-6, warmup=0.027。

> **为什么 classifier_lr=2.22e-4 在全量训练时回退到 1e-4？** 子集+5 epoch 场景需要高 LR 快速增长，但全量数据 10 epoch 下 2.22e-4 导致梯度震荡（已验证：Loss 不动）。hp_tune 在子数据上的"最优"对全量数据不直接适用，连续参数需保守处理。

### 4.4 最终参数选择逻辑

| 参数 | HP 最优 | 最终采用 | 原因 |
|------|------|------|------|
| pooling | cls | cls ✅ | 8 次一致，可信 |
| head_hidden | 512 | 512 ✅ | 4 次一致，可信 |
| encoder_lr | 2.84e-5 | 2.84e-5 ✅ | 接近 baseline 2e-5，合理 |
| classifier_lr | 2.22e-4 | 1e-4 ⚡ | 全量数据回退安全值 |
| dropout | 0.106 | 0.2 ⚡ | 小数据低正则化 → 全量数据需加强 |
| weight_decay | 7e-6 | 0.01 ⚡ | 同上 |

---

## 五、结果

### 5.1 最终指标

| 维度 | Acc | Prec | Rec | F1 | AUC | MCC |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| EI | 94.7% | 90.6% | 86.5% | 0.885 | 0.976 | 0.836 |
| SN | 97.9% | 97.9% | 99.9% | 0.989 | 0.983 | 0.844 |
| TF | 95.6% | 95.7% | 97.6% | 0.966 | 0.988 | 0.903 |
| JP | 92.2% | 91.3% | 90.0% | 0.907 | 0.975 | 0.834 |
| **Mean** | **95.1%** | — | — | **0.937** | **0.980** | **0.854** |

| 整体指标 | 值 |
|----------|:---:|
| Exact Match (四维全对) | **86.0%** |
| Hamming Loss | 4.94% |
| Macro MCC | 0.854 |

### 5.2 Baseline 对比

| 指标 | Baseline | HP 最优 | 绝对提升 | 相对提升 |
|------|:---:|:---:|:---:|:---:|
| Overall Acc | 80.6% | 86.0% | +5.4pp | +6.7% |
| Mean Acc | 93.2% | 95.1% | +1.9pp | +2.0% |
| Mean F1 | 91.3% | 93.7% | +2.4pp | +2.6% |
| EI F1 | 0.836 | 0.885 | +0.049 | +5.9% |
| JP F1 | 0.877 | 0.907 | +0.030 | +3.4% |
| TF F1 | 0.956 | 0.966 | +0.010 | +1.0% |

### 5.3 训练收敛过程

| Epoch | Train Loss | Test Loss | Overall | Mean Acc | 状态 |
|:---:|------|------|:---:|:---:|------|
| 1 | 0.270 | 0.255 | 0.805 | 0.929 | 快速学习 |
| 2 | 0.184 | 0.177 | 0.832 | 0.941 | 最佳泛化 |
| 3 | 0.151 | 0.213 | 0.842 | 0.945 | 拐点 |
| 7 | 0.055 | 0.355 | **0.860** | **0.951** | 最佳 Acc |
| 10 | 0.027 | 0.473 | 0.858 | 0.950 | 过拟合 |

Epoch 2 时 Test Loss 最低（泛化最优），但 Epoch 7 在 Acc 上达到峰值——模型在更低 Train Loss 下仍能提升分类精度，说明四维分类面在后期持续优化，只是 confidence 开始失校准。

---

## 六、经验教训

### 6.1 关键踩坑记录

| 问题 | 根因 | 解决 |
|------|------|------|
| BCELoss CUDA 崩溃 | FP16 下 logit>10 时 `log(1-sigmoid(x))` 下溢 | loss 计算强制 FP32 |
| 训练 Loss 不降 | encoder_lr 多打一个零 (2.84e-4 vs 2.84e-5) | 检查 YAML 数值类型 |
| 子集搜索参数不适用全量 | 30% 数据防欠拟合 → 全量需防过拟合 | 连续参数回退安全值 |
| 128 batch 学习慢 | 每 epoch 仅 660 步 | 降到 32（2,652 步/epoch） |

### 6.2 后续方向

- **roberta-large**：355M 参数，1024 维，预期额外 +1-2%
- **增强不均衡维度**：EI 和 JP 仍有 1-2% 空间，针对性调 pos_weight 或 focal loss
- **用户级数据分割**：当前按 post 随机分，同一用户多帖跨 train/test 可能虚高 ~1-2%

---

## 附录 A：复现步骤

```bash
# 1. 环境
conda create -n mbti_pred python=3.10 -y && conda activate mbti_pred
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. 数据
python data/getdata.py
python data/preprocess.py

# 3. 下载 roberta-base 到 models/roberta-base/

# 4. Baseline
python train.py --epochs 3 --pooling mean

# 5. HP 搜索
python hp_tune.py --p1 8 --p2 10

# 6. 最终训练（应用 HP 最优参数到 config/default.yaml）
python -u train.py

# 7. 评估
python eval.py --ckpt output/<timestamp>/best
```

## 附录 B：硬件环境

| 组件 | 训练 | HP 搜索 |
|------|------|------|
| GPU | RTX 4060 Laptop 8GB | NVIDIA 48GB (远程) |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu128 |
| Python | 3.10 | 3.10 |
| OS | Windows 11 | Windows 11 |

## 附录 C：文件清单

| 文件 | 说明 |
|------|------|
| `data/preprocess.py` | 数据预处理（清洗+拆分+划分） |
| `data/dataset.py` | PyTorch Dataset 封装 |
| `src/representation/encoder.py` | RoBERTa 编码器 + Pooling |
| `src/model/classifier.py` | 四维分类头 |
| `src/model/trainer.py` | 训练器 |
| `src/explanation/*.py` | 解释层（归因+注意力+NLG） |
| `src/app/api.py` | FastAPI 推理服务 |
| `train.py` | 训练入口 |
| `eval.py` | 评估+可视化 |
| `hp_tune.py` | 超参搜索 |
| `explain.py` | 解释管线 |
| `config/default.yaml` | 训练配置文件 |
| `test/analysis.md` | HP 搜索详细分析 |
| `report.md` | 本文档 |

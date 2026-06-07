# 项目进度报告

> 更新日期：2025-06-05

---

## 一、项目概述

基于 RoBERTa-base 的英文文本 MBTI 人格类型预测系统。采用**四维独立二分类**架构（非 16 类 softmax），输入文本 → 输出 E/I、S/N、T/F、J/P 四维概率，附关键词归因、注意力数据和 NLG 解读。

---

## 二、完成进度

### 核心模块

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|:--:|
| 数据层 | `data/preprocess.py` | 文本清洗、16类→4维标签、8:2分层采样、CSV+JSON双格式输出 | ✅ |
| 数据层 | `src/data/dataset.py` | PyTorch Dataset，在线 tokenize，返回 input_ids/attention_mask/labels | ✅ |
| 表征层 | `src/representation/encoder.py` | RoBERTaEncoder + 4 种 Pooling(cls/mean/max/attention)，本地模型缓存 | ✅ |
| 任务层 | `src/model/classifier.py` | 四独立头(768→64→1 logit)，BCEWithLogitsLoss，FP16 兼容 | ✅ |
| 训练器 | `src/model/trainer.py` | 分层 LR、epoch 级 Cosine 调度、早停、每 epoch 独立目录、TensorBoard | ✅ |
| 解释层 | `src/explanation/attribution.py` | Integrated Gradients (Captum) token 归因 | ✅ |
| 解释层 | `src/explanation/attention.py` | 12层×12头注意力权重提取 | ✅ |
| 解释层 | `src/explanation/interpreter.py` | 中/英 NLG 模板解读，覆盖 16 种 MBTI | ✅ |
| 应用层 | `src/app/api.py` | FastAPI，启动加载模型，POST /api/predict | ✅ |

### 入口脚本

| 脚本 | 功能 | 状态 |
|------|------|:--:|
| `train.py` | YAML 配置 + CLI 覆盖，一键训练 | ✅ |
| `eval.py` | 6 项指标 + 4 张可视化图 | ✅ |
| `explain.py` | 完整解释管线 → JSON | ✅ |
| `api_server.py` | FastAPI 启动入口 | ✅ |

### 配置文件

| 文件 | 内容 | 状态 |
|------|------|:--:|
| `config/default.yaml` | 全部超参集中管理 | ✅ |
| `requirements.txt` | 依赖清单 | ✅ |

---

## 三、基准模型（Baseline）评测分析

> 模型位置：`checkpoints/baseline/`  
> 基准定义：roberta-base, mean pooling, max_length=256, 3 epochs  
> 后续所有实验均与此基准对比

### 3.1 基准训练配置

| 超参 | 值 | 说明 |
|------|-----|------|
| 模型 | `roberta-base` (125M, 768维) | 本地加载 `models/roberta-base/` |
| Pooling | mean | 所有 token 均值 |
| max_length | 256 | 训练截断 |
| batch_size | 16 | — |
| epochs | 3 | 未早停 |
| encoder LR | 2e-5 | — |
| classifier LR | 1e-4 | 头部快于骨干 |
| Dropout | 0.2 | — |
| SN pos_weight | 10.0 | 缓解 S 类不均衡 |
| SN dim_weight | 0.35 | 重点优化 SN 维 |
| 硬件 | RTX 4060 Laptop 8GB | — |

### 3.2 基准测试集指标

数据集：zeyadkhalid/MBTI-500，106,067 条 → 训练 84,853 / 测试 21,214

| 维度 | Acc | Prec | Rec | F1 | AUC | MCC |
|------|-----|------|-----|-----|-----|-----|
| **EI** | 91.7% | 79.0% | 88.8% | 83.6% | 96.9% | 0.783 |
| **SN** | 96.9% | 97.1% | 99.7% | 98.3% | 98.2% | 0.790 |
| **TF** | 94.3% | 95.3% | 96.0% | 95.6% | 98.4% | 0.874 |
| **JP** | 90.0% | 90.2% | 85.3% | 87.7% | 96.3% | 0.793 |
| **Mean** | **93.2%** | — | — | **91.3%** | **97.4%** | **0.810** |

| 整体指标 | 值 | 说明 |
|----------|-----|------|
| Exact Match | **80.6%** | 四维全对比例 |
| Hamming Loss | 6.8% | 平均每样本错 0.27 个维度 |
| Macro MCC | 0.810 | 强相关性 (随机=0, 完美=1) |

### 3.3 基准逐维度深度分析

#### EI（外向/内向）— 最难的维度

| 指标 | 值 | 解读 |
|------|-----|------|
| Accuracy | 91.7% | 看起来高，但受多数类(I:76%)抬升 |
| Precision | 79.0% | **最低** — 预测为 E 的样本中 21% 其实是 I |
| Recall | 88.8% | 真 E 中有 11% 被漏判 |
| F1 | 83.6% | Acc 与 F1 差距 8.1pp，说明类别分布扭曲了 Acc |
| AUC | 96.9% | 排序能力强，阈值选 0.5 不是最优 |
| MCC | 0.783 | **最低** — Precision-Recall 双双不理想 |

**诊断**：低 Precision + 高 Recall = 模型倾向判 E。假阳性主要来自 I 型用户使用外向词汇（"party"、"friends"、"talk"）但整体语境是内向的。建议针对 EI 维度调高分类阈值（如 0.6）或单独增加 pos_weight。

#### SN（感觉/直觉）— 极度不均衡下的虚假繁荣

| 指标 | 值 | 解读 |
|------|-----|------|
| Accuracy | 96.9% | 全维最高，但 N 占 91.3%，瞎猜 N 也有 91.3% |
| Precision | 97.1% | 高 — 判 N 基本都对 |
| Recall | 99.7% | **极高** — 几乎不漏判任何 N |
| F1 | 98.3% | 受 N 类主导，不代表 S 类分类能力 |
| AUC | 98.2% | 排序能力极强 |
| MCC | 0.790 | **暴露真相** — 仅 0.79，说明 S 类分类很差 |

**诊断**：Accuracy/F1/Precision/Recall 四项虚高，完全由多数类 N(91.3%) 主导。唯一诚实指标是 MCC=0.79。pos_weight=10 一定程度上缓解了不均衡，但 S 类样本仅 9,201 条，模型对 S 的特征学习不足。需增加 S 类数据增强或采用 focal loss。

#### TF（思考/情感）— 表现最好的维度

| 指标 | 值 | 解读 |
|------|-----|------|
| Accuracy | 94.3% | 较高 |
| Precision | 95.3% | **最高** — 判 T 几乎都对 |
| Recall | 96.0% | 真 T 中仅 4% 漏判 |
| F1 | 95.6% | **全维最优**，Precision 和 Recall 平衡 |
| AUC | 98.4% | **最高** — 区分能力最强 |
| MCC | 0.874 | **全维最优** — 六项指标全部领先 |

**诊断**：T/F 是四个维度中区分度最高的。原因在于文本特征明显 — "because"、"logic"、"analysis" vs "feel"、"value"、"care" 等关键词信号强。T:F 比例 65:35 也比其他维度均衡。当前模型在此维度已接近天花板。

#### JP（判断/感知）— 第二瓶颈

| 指标 | 值 | 解读 |
|------|-----|------|
| Accuracy | 90.0% | 四维最低（SN 虚高不计） |
| Precision | 90.2% | 尚可 — 判 J 的准确率 |
| Recall | 85.3% | **偏低** — 真 J 中 15% 被漏判为 P |
| F1 | 87.7% | Precision-Recall 差距 4.9pp，略失衡 |
| AUC | 96.3% | 排序能力好，阈值可调 |
| MCC | 0.793 | 仅优于 EI 和 SN |

**诊断**：J/P 的本质区别在于"行为模式"（plan vs go with flow），文本中的表达非常间接且多样化，不如 T/F 有明确关键词。Recall 偏低说明 J 类样本（41.9%）中有相当一部分被模型误判为 P。建议增加 head_hidden 维度或使用 attention pooling 增强文本特征捕获。

### 3.5 基准六指标综合排名

| 维度 | Acc | Prec | Rec | F1 | AUC | MCC | 综合 |
|------|:---:|:----:|:---:|:---:|:---:|:---:|:----:|
| TF | 2nd | 🥇 | 2nd | 🥇 | 🥇 | 🥇 | **🏆 最优** |
| JP | 4th | 2nd | 4th | 2nd | 4th | 2nd | 中等 |
| EI | 3rd | 4th | 3rd | 4th | 3rd | 4th | 偏弱 |
| SN | 1st | 🥇 | 🥇 | 🥇 | 🥈 | 3rd | ⚠️ 虚高 |

> 关键结论：**不均衡维度（SN、EI）的 Acc/F1 有欺骗性，MCC 是唯一不受类别分布影响的综合指标。**

### 3.6 可视化评估（`eval_output/`）

| 图表 | 关键发现 |
|------|---------|
| `confusion_matrices.png` | TF 混淆矩阵对角线最亮；EI 假阳性（I→E）偏高 |
| `roc_curves.png` | 四维 AUC 均 > 0.96，模型区分能力极强 |
| `confidence_dist.png` | 正确预测集中在 0.85+ 置信区，错误集中在 0.5 附近 — 模型"自知其不知" |
| `radar.png` | TF 维度 Accuracy/F1/AUC 最均衡，EI 的 F1 明显低于其他维 |

### 3.7 基准训练日志 vs 测试集评估

训练时在 `max_length=256` 下计算 loss，评估时可用更长上下文。两者差异揭示了截断的影响：

| 指标 | 训练日志 (train, len=256) | 正式评估 (test, len=512) | 提升 |
|------|--------------------------|--------------------------|------|
| EI Acc | 87.4% | 91.7% | +4.3% |
| SN Acc | 95.3% | 96.9% | +1.6% |
| TF Acc | 88.5% | 94.3% | +5.8% |
| JP Acc | 82.1% | 90.0% | +7.9% |
| Overall | 67.2% | 80.6% | +13.4% |
| Mean Acc | 88.3% | 93.2% | +4.9% |

> 基准模型训练时仅用 256 截断，评估时扩展到 512 获得更多上下文，提升显著。**说明 max_length=256 是当前最大瓶颈，用 512 重新训练预期带来同等幅度提升。**

---

## 四、HP 优化结果

### 4.1 搜索总结

| | Baseline | HP 最优 | 来源 |
|------|:---:|:---:|------|
| pooling | mean | **cls** | HP 搜索 |
| head_hidden | 64 | **512** | HP 搜索 |
| encoder_lr | 2e-5 | **2.84e-5** | HP 搜索 |
| max_length | 256 | **512** | 固定 |
| batch_size | 16 | **32** | 固定 |
| epochs | 3 | **7 (best)** | — |

### 4.2 HP 最优逐维度指标

| 维度 | Acc | F1 | AUC | MCC |
|------|:---:|:---:|:---:|:---:|
| EI | 94.7% | 0.885 | 0.976 | 0.836 |
| SN | 97.9% | 0.989 | 0.983 | 0.844 |
| TF | 95.6% | 0.966 | 0.988 | 0.903 |
| JP | 92.2% | 0.907 | 0.975 | 0.834 |
| **Mean** | **95.1%** | **0.937** | **0.980** | **0.854** |

| 整体指标 | 值 |
|----------|:---:|
| Exact Match | **86.0%** |
| Mean Acc | **95.1%** |
| Mean F1 | **93.7%** |

---

## 五、后续优化方向

| 优先级 | 事项 | 说明 |
|:------:|------|------|
| 🔴 | **roberta-large** | 355M 参数，预期 +1-2% |
| 🟡 | **全量数据 HP 搜索** | 当前用 30-50% 子集，全量可能翻出更好参数 |
| 🟡 | **针对性调优 EI/JP** | 两个最难维度的 pos_weight 精调 |
| 🟢 | **LoRA 微调实验** | 减少可训参数，加速训练 |
| 🟢 | **前端可视化增强** | 词云/热力图展示关键词归因 |
| 🟡 | **更换 pooling 策略** | 当前 mean，可试 attention-weighted |
| 🟢 | **LoRA 微调实验** | 减少可训参数，加速训练 |
| 🟢 | **前端页面开发** | 用户输入 → 雷达图 + 热力图 + 解读 |
| 🟢 | **AB 对比实验** | TF-IDF+LR baseline vs RoBERTa 系统对比 |

# MBTI 性格预测系统（设计阶段）

本项目构建一个基于自然语言处理与深度学习的 **MBTI 性格预测系统**，通过用户输入的文本内容预测其 MBTI 四维人格类型（E/I、S/N、T/F、J/P），并提供可解释的预测分析。

项目当前处于 **设计与规划阶段**，目标是搭建从数据输入、语义表征、人格预测、结果解释到前端展示的完整端到端系统。

---

## 总体架构

系统采用 **五层流水线架构**，层与层之间通过明确定义的数据契约解耦，各层可独立开发、测试和替换。

![pasted-image-1780583053050.webp](https://files.seeusercontent.com/2026/06/04/Xcs6/pasted-image-1780583053050.webp)
---

## 1. 数据层（Data Layer）

### 1.1 职责

负责原始数据的加载、清洗、预处理和标准化，向下游输出高质量的结构化数据。

### 1.2 数据源

| 数据来源 | 格式 | 说明 |
|----------|------|------|
| MBTI 语料数据集 | CSV（type, posts） | 用户论坛帖子与对应 MBTI 标签 |
| 用户实时输入 | 纯文本字符串 | 前端输入的待预测文本 |

### 1.3 预处理流程

```
原始文本 → HTML 标签移除 → 特殊符号/表情清洗 → URL/提及移除
         → 空白规范化 → 分词 → 长度截断/填充 → 输出
```

### 1.4 标签处理

MBTI 16 种类型拆分为四个独立二分类标签：

| MBTI 类型 | E/I | S/N | T/F | J/P |
|-----------|-----|-----|-----|-----|
| INFJ | 0 (I) | 1 (N) | 0 (F) | 1 (J) |
| ENTP | 1 (E) | 1 (N) | 1 (T) | 0 (P) |
| ... | ... | ... | ... | ... |

### 1.5 模块接口

```
输入：原始 CSV 文件路径 或 用户输入文本字符串
输出：{
    "text": str,           // 清洗后文本
    "tokens": List[str],   // 分词序列
    "labels": {            // 仅训练时有
        "EI": 0|1,
        "SN": 0|1,
        "TF": 0|1,
        "JP": 0|1
    }
}
```

---

## 2. 表征层（Representation Layer）

### 2.1 职责

将自然语言文本转换为稠密语义向量，供下游分类器使用。这是系统的核心能力层，表征质量直接决定预测上限。

### 2.2 技术选型

选用 **BERT/RoBERTa-base** 作为表征模型，平衡效果与训练/推理成本。

| 模型 | 向量维度 | 参数量 | 说明 |
|------|----------|--------|------|
| `bert-base-uncased` | 768 | 110M | 英文数据首选 |
| `bert-base-chinese` | 768 | 110M | 中文数据 |
| `RoBERTa-wwm-ext` | 768 | 110M | **中文推荐**，哈工大讯飞联合发布，全词掩码，中文 NLP 表现优于原生 BERT |

> 本项目以中文 MBTI 文本为主要处理对象，默认使用 **`RoBERTa-wwm-ext`**。

### 2.3 处理流程

```
输入文本序列
     │
     ▼
Tokenizer（WordPiece / BPE）
     │
     ▼
Token Embeddings + Segment Embeddings + Position Embeddings
     │
     ▼
Transformer Encoder（12 层 × 12 头自注意力）
     │
     ▼
Pooling Strategy（CLS Token / Mean Pooling / Max Pooling）
     │
     ▼
768-dim 语义向量 → 传入任务层
```

### 2.4 Pooling 策略对比

| 策略 | 做法 | 特点 |
|------|------|------|
| CLS Token | 取 `[CLS]` 位置的输出 | BERT 原生方式，适合分类 |
| Mean Pooling | 所有 token 取均值 | 信息利用充分，推荐实验 |
| Max Pooling | 每个维度取最大值 | 突出关键特征 |

### 2.5 模块接口

```
输入：清洗后的文本字符串
输出：numpy.ndarray / torch.Tensor, shape = (batch_size, 768)
```

---

## 3. 任务层（Task Layer）

### 3.1 职责

基于语义向量进行四维二分类，输出每个维度的预测概率。

### 3.2 设计思路

相比直接做 16 类 softmax 分类，**四维独立二分类**有以下优势：

- **稳定性**：每个维度只需学习两个类别，决策边界更清晰
- **可解释性**：可以独立分析每个维度受哪些文本特征影响
- **样本效率**：16 分类需要更多数据覆盖所有组合，四维拆分后每个二分类器都能利用全部数据
- **灵活性**：可以单独优化某一维度的分类器

### 3.3 模型结构

```text
                 特征向量 (768-dim)
                        |
                  +-------------------+
                  | Shared Dense       |  <-- 共享表示层 (Dropout)
                  | 768 -> 256         |
                  +-------------------+
                        |
        +---------------+---------------+-------------+
        |               |               |             |
   E/I Head        S/N Head       T/F Head       J/P Head
   256 -> 64       256 -> 64      256 -> 64       256 -> 64
    64 -> 1         64 -> 1        64 -> 1         64 -> 1
        |               |               |               |
    Sigmoid          Sigmoid         Sigmoid         Sigmoid
        |               |               |               |
   p(E) [0,1]      p(S) [0,1]     p(T) [0,1]     p(J) [0,1]
```

### 3.4 损失函数

每个维度使用 **二元交叉熵（BCE Loss）**，总损失为四个维度的加权和：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{EI} + \lambda_2 \mathcal{L}_{SN} + \lambda_3 \mathcal{L}_{TF} + \lambda_4 \mathcal{L}_{JP}$$

默认 $\lambda_i = 0.25$（等权），可根据各维度分类难度调整。

### 3.5 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy（逐维度） | 每个二分类的准确率 |
| Overall Accuracy | 四维全部正确才算正确 |
| F1 Score | 每维度的精确率-召回率调和均值 |
| AUC-ROC | 维度预测的区分能力 |

### 3.6 输出映射

```
p(E) ≥ 0.5 → "E" | p(E) < 0.5 → "I"
p(S) ≥ 0.5 → "S" | p(S) < 0.5 → "N"
p(T) ≥ 0.5 → "T" | p(T) < 0.5 → "F"
p(J) ≥ 0.5 → "J" | p(J) < 0.5 → "P"

组合 → 如 [I, N, F, P] → "INFP"
```

### 3.7 模块接口

```
输入：语义向量 (batch_size, 768)
输出：{
    "probabilities": {
        "EI": {"E": 0.32, "I": 0.68},
        "SN": {"S": 0.21, "N": 0.79},
        "TF": {"T": 0.45, "F": 0.55},
        "JP": {"J": 0.38, "P": 0.62}
    },
    "mbti_type": "INFP",
    "confidence": 0.73  // 四维平均置信度
}
```

---

## 4. 解释层（Explanation Layer）

### 4.1 职责

将模型的"黑盒预测"转化为用户可理解的解释——回答"**为什么**你被预测为 INFP 而不是 INFJ？"

### 4.2 解释技术

| 技术 | 粒度 | 说明 | 实现路径 |
|------|------|------|----------|
| 注意力权重提取 | Token 级 | 取 BERT 最后层多头注意力均值，高亮关键 tokens | 从模型 forward 中导出 `attentions` |
| Integrated Gradients | Token 级 | 计算每个 token 对各维度预测的贡献值 | Captum / Transformers Interpret |
| 模板化 NLG | 维度级 | 基于概率值映射到人格描述文本 | 规则引擎 |

### 4.3 实施方案

采用 **注意力热力图 + Integrated Gradients + NLG 模板解读** 的组合方案：

- **注意力热力图**：直观展示模型关注的文本区域
- **Integrated Gradients**：精确计算每个 token 对各维度的贡献值，标注 Top-K 关键词
- **NLG 模板解读**：将概率值和关键词贡献转化为自然语言人格描述

### 4.4 输出结构

```
输入：原始文本 + 四维概率
输出：{
    "keywords": {
        "EI": [{"token": "独处", "score": 0.23}, {"token": "安静", "score": 0.18}],
        "SN": [{"token": "具体", "score": 0.31}, {"token": "实际", "score": 0.25}],
        "TF": [{"token": "逻辑", "score": 0.28}, {"token": "分析", "score": 0.22}],
        "JP": [{"token": "计划", "score": 0.35}, {"token": "安排", "score": 0.19}]
    },
    "interpretation": {
        "EI": "你的文本显示出较高的内向倾向（I: 68%）。你更多使用与内心世界相关的词汇...",
        "SN": "你在直觉维度上倾向明显（N: 79%）。你偏好抽象概念和未来可能性...",
        ...
    },
    "summary": "综合分析，你的 MBTI 类型为 INFP（调停者）。你是一个充满理想主义的内向者..."
}
```

### 4.5 模块接口

```
输入：原始文本 + 任务层输出（概率 + 类型）
输出：InterpretationResult（上述 JSON 结构）
```

---

## 5. 应用层（Application Layer）

### 5.1 职责

提供推理 API 服务和用户交互界面，将预测和解释结果直观地呈现给用户。

### 5.2 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| 推理服务框架 | FastAPI | 异步支持好，自动生成 OpenAPI 文档 |
| 前端 | 原生 HTML/CSS/JS | 无框架依赖，单文件部署 |
| 可视化 | ECharts / Chart.js | 雷达图 + 条形图 |
| 模型服务 | PyTorch + Transformers | 加载训练好的 BERT + 分类头 |

### 5.3 API 设计

```
POST /api/predict
Request:
{
    "text": "我喜欢一个人安静地读书，思考人生的意义..."
}

Response:
{
    "mbti_type": "INFP",
    "probabilities": {
        "EI": {"E": 0.32, "I": 0.68},
        "SN": {"S": 0.21, "N": 0.79},
        "TF": {"T": 0.45, "F": 0.55},
        "JP": {"J": 0.38, "P": 0.62}
    },
    "explanation": {
        "keywords": { ... },
        "interpretation": { ... },
        "summary": "..."
    }
}
```

### 5.4 项目目录结构

```
MBTI_pred/
├── data/                    # 数据集
│   ├── mbti_train.csv       # 训练数据
│   ├── mbti_val.csv         # 验证数据
│   └── mbti_test.csv        # 测试数据
├── src/
│   ├── data/                # 数据层
│   │   ├── preprocess.py    # 文本清洗与预处理
│   │   └── dataset.py       # PyTorch Dataset 封装
│   ├── representation/      # 表征层
│   │   └── encoder.py       # BERT/RoBERTa 编码器
│   ├── model/               # 任务层
│   │   ├── classifier.py    # 四维分类头
│   │   └── trainer.py       # 训练循环与评估
│   ├── explanation/         # 解释层
│   │   ├── attribution.py   # Integrated Gradients 特征归因
│   │   ├── attention.py     # 注意力权重可视化数据
│   │   └── interpreter.py   # 模板化 NLG 解读生成
│   └── app/                 # 应用层
│       ├── api.py           # FastAPI 服务
│       └── static/
│           └── index.html   # 前端页面
├── experiments/             # 实验记录
│   └── configs/             # 训练配置 YAML
├── models/                  # 保存的模型权重
├── requirements.txt
└── readme.md
```

---

## 6. 技术方案总览

### 6.1 整体技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 数据层 | Python + Pandas | 文本清洗、标签处理、数据集划分 |
| 表征层 | `RoBERTa-wwm-ext` | 768 维语义向量，HuggingFace Transformers |
| 任务层 | PyTorch + Shared FC + 4 Heads | 四维独立二分类，BCE 联合损失 |
| 解释层 | Integrated Gradients + 注意力权重 | Captum / Transformers Interpret |
| 推理服务 | FastAPI | 异步推理，RESTful API |
| 前端 | HTML/CSS/JS + ECharts | 雷达图 + 条形图可视化 |

### 6.2 对比参考

以下方案作为实验对比参考，验证主方案效果提升：

| 维度 | Baseline（对照） | 主方案（采用） | 进阶探索 |
|------|-----------------|----------------|----------|
| 文本表征 | TF-IDF + SVD | BERT/RoBERTa-base | RoBERTa-large |
| 分类模型 | 逻辑回归 × 4 | Shared FC + 4 Heads | LoRA 微调 |
| 解释方法 | 词频统计 | IG + 注意力 | SHAP + LLM |
| 预测精度（预期） | 60~65% | 72~78% | 78~85% |
| 训练时间 | 分钟级 | 小时级（GPU） | 天级（GPU） |

---

## 7. 开发路线图

| 阶段 | 内容 | 预计产出 |
|------|------|----------|
| **Phase 1：数据准备** | 数据清洗、EDA、标签拆分、数据集划分 | 结构化训练/验证/测试集 |
| **Phase 2：Baseline 建立** | TF-IDF + 逻辑回归基线模型 | 基线指标（预测精度 60~65%） |
| **Phase 3：主模型训练** | RoBERTa-wwm-ext 表征 + 四分类头训练与调优 | 主模型（目标精度 72~78%） |
| **Phase 4：解释层开发** | Integrated Gradients 归因 + 注意力可视化 + NLG 解读 | 可解释的预测输出 |
| **Phase 5：服务化与前端** | FastAPI 推理 API + HTML 前端 + ECharts 可视化 | 可交互的 Web Demo |
| **Phase 6：对比实验（可选）** | 主方案 vs Baseline A/B 对比，撰写实验报告 | 论文/报告素材 |

---

## 附录 A：MBTI 维度说明

| 维度 | 字母 | 含义 |
|------|------|------|
| **E / I** | Extraversion / Introversion | 外向（关注外部世界） / 内向（关注内心世界） |
| **S / N** | Sensing / Intuition | 感觉（关注具体事实） / 直觉（关注抽象模式） |
| **T / F** | Thinking / Feeling | 思考（逻辑决策） / 情感（价值观决策） |
| **J / P** | Judging / Perceiving | 判断（计划有序） / 感知（灵活开放） |

## 附录 B：16 种 MBTI 类型速查

| 类型 | 别称 | 类型 | 别称 |
|------|------|------|------|
| INTJ | 建筑师 | INTP | 逻辑学家 |
| ENTJ | 指挥官 | ENTP | 辩论家 |
| INFJ | 提倡者 | INFP | 调停者 |
| ENFJ | 主人公 | ENFP | 竞选者 |
| ISTJ | 物流师 | ISFJ | 守卫者 |
| ESTJ | 总经理 | ESFJ | 执政官 |
| ISTP | 鉴赏家 | ISFP | 探险家 |
| ESTP | 企业家 | ESFP | 表演者 |

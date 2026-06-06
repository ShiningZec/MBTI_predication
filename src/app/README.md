# MBTI 前端 — 使用手册

## 快速启动

```bash
# 方式1：双击（macOS）
双击 src/app/launch.command

# 方式2：命令行
conda activate web
python src/app/launch.py

# 方式3：手动
conda activate web
uvicorn src.app.api:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

启动器会自动：检查端口 → 启动服务 → 打开浏览器。

---

## 训练完成后对接前端

训练产出在 `output/<时间戳>/` 下，结构类似：

```
output/20260606_120000/
├── epoch_1/        # 每轮 checkpoint
├── epoch_2/
├── epoch_3/
├── best/           # 最佳 epoch 副本
│   ├── encoder.pt
│   └── classifier.pt
└── training_info.json
```

### 你需要做的

| # | 事项 | 说明 |
|---|------|------|
| 1 | **下载 RoBERTa 模型** | 放到 `models/roberta-base/` 下（需包含 `config.json`、`pytorch_model.bin`、`vocab.json`、`merges.txt`） |
| 2 | **安装 ML 依赖** | `conda activate web && pip install torch transformers captum scikit-learn pandas` |
| 3 | **训练完成** | `python train.py` — checkpoint 自动写入 `output/<时间戳>/` |
| 4 | **放置评估结果** | `python eval.py` — 指标 JSON + 图片写入 `eval_output/` |
| 5 | **启动服务** | `python src/app/launch.py` |

### 后端自动做了什么

`api.py` 会在启动时自动扫描：

1. **模型权重**：`models/roberta-base/` → `models/` 下任意含 `config.json` 的目录
2. **Checkpoint**：`output/<最新>/best/` → `output/<最新>/epoch_N/` → `checkpoints/baseline/`
3. **训练信息**：`output/<最新>/training_info.json`
4. **评估指标**：`eval_output/metrics.json`
5. **评估图片**：`eval_output/*.png`

只要你把文件放在约定位置，**不需要改任何代码**。

---

## 目录约定

```
MBTI_predication/
├── models/
│   └── roberta-base/       # ← 放 RoBERTa 权重（下载后）
├── output/
│   └── <时间戳>/
│       ├── best/
│       │   ├── encoder.pt  # ← 训练产出的 checkpoint
│       │   └── classifier.pt
│       └── training_info.json
├── checkpoints/
│   └── baseline/           # ← 回退路径（可选）
├── eval_output/            # ← eval.py 产出的指标和图片
│   ├── metrics.json
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── radar.png
│   └── confidence_dist.png
└── src/app/
    ├── api.py              # FastAPI 后端
    ├── index.html          # Web 前端
    ├── launch.py           # 启动器
    └── launch.command      # macOS 双击启动
```

---

## 环境配置速查

```bash
# 创建环境
conda create -n web python=3.10 -y
conda activate web

# 基础（前端服务）
pip install fastapi uvicorn pydantic pyyaml

# ML 推理（模型预测需要）
pip install torch transformers captum scikit-learn pandas

# 训练（如需在 web 环境训练）
pip install tensorboard matplotlib seaborn tqdm datasets tokenizers accelerate safetensors
```

---

## API 端点一览

| 方法 | 路径 | 说明 | 需要模型 |
|------|------|------|----------|
| GET | `/health` | 服务状态 + 模型是否就绪 | — |
| GET | `/api/model` | 模型参数、超参、评估指标 | — |
| POST | `/api/predict` | 文本预测（`{"text":"..."}`） | ✅ |
| GET | `/static/eval/*.png` | 评估图表 | — |
| GET | `/` | 前端页面 | — |

---

## 故障排查

| 症状 | 可能原因 | 解决 |
|------|----------|------|
| 前端显示"服务离线" | 后端未启动 | `python api_server.py` |
| 预测返回 503 | 模型/checkpoint 未就绪 | 检查 `models/` 和 `output/` 目录 |
| 页面空白 | CDN 被墙 | 挂代理，或把 MWC 源码构建后用本地路径 |
| `ModuleNotFoundError: torch` | ML 依赖未安装 | `pip install torch transformers captum` |
| 预测超时 | CPU 推理太慢 | 检查 `health` 返回的 `device`，建议用 CUDA |

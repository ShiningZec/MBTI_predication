# 前端改进方案：MBTI 预测系统交互式前端

## Context

当前前端是一个 517 行纯 HTML 单文件，功能完整但缺少三个关键板块：
1. Baseline vs HP 最优模型的对比展示
2. 超参数搜索 20 条记录的交互式可视化
3. 静态 PNG 图片部分替换为交互式 Canvas/SVG 图表

用户要求保持纯 HTML 无框架、学术报告风格、动画过渡、适用于论文展示。

---

## Architecture: Single-File Interactive Canvas

在现有 `src/app/index.html` 基础上扩展至 ~2500 行单文件，所有图表用 Canvas/SVG 从 JSON 数据渲染，新增 3 个 API 端点和 3 个视图，保持零外部依赖。

### 导航结构（从 2 个扩展到 5 个视图）

侧边栏新增 3 个导航项：

| 视图 | 内容 | 数据来源 |
|------|------|------|
| Dialogue Analysis | 文本输入 → 预测 + 解释（已有，增强） | POST /api/predict |
| Model Comparison | Baseline vs Best 参数/指标/图表（新增） | GET /api/compare |
| Hyperparameter Search | HP 搜索 20 trial 可视化（新增） | GET /api/trials |
| Data Exploration | data_viz/ 图库（新增） | /static/data_viz |
| Model Details | 当前模型参数/指标（已有，增强） | GET /api/model |

---

## Backend Changes (`src/app/api.py`)

### 新增 3 个端点

**1. `GET /api/compare`** — 加载 baseline 和 best 的 metrics.json + training_info.json，返回结构化对比数据（指标、参数、deltas、图片 URL）

**2. `GET /api/trials`** — 读取 `test/trials_summary.jsonl`，返回全部 20 个 trial + phase 分组元数据 + best trial 索引

**3. `GET /api/best-config`** — 读取 `test/best_config.json`，返回最优超参

### 新增静态挂载
```python
/static/eval-baseline  → eval_output/baseline/
/static/eval-best      → eval_output/best/
/static/data_viz       → data_viz/
```

---

## Frontend Changes (`src/app/index.html`)

### CSS: 学术报告风格

在现有暗色主题基础上扩展：
- 图表色板：Tableau 10 学术配色，色彩低调克制
- 网格线 subtle（rgba(255,255,255,0.04)）
- 字体：等宽数字（tabular-nums）用于数据对比表格
- 动画：视图切换 200ms 淡入、Canvas 图表 600ms 交错入场、度量 delta 数字滚动、图表 hover 150ms 高亮

### JS: Chart4Lib（Canvas/SVG 图表库）

6 个可复用图表组件，纯函数式 Canvas/SVG 渲染：

| 图表 | 类型 | 功能 |
|------|------|------|
| ComparisonBar | Canvas | 两组并排柱状图（baseline vs best），hover 显示差异 |
| DualRadar | SVG | 双六边形雷达（Acc / F1 / AUC / MCC / Exact / 1-Hamming） |
| ConfidenceHist | Canvas | 双直方图叠加，置信度分桶 |
| TrialScatter | Canvas | 散点图（trial index × Mean Acc），phase 分色，点击弹详情 |
| ParallelCoords | Canvas | 5 轴平行坐标（Dropout / LR / WD / Warmup ratio × Mean Acc） |
| GaugeRing | SVG | 现有圆环仪表盘的增强版（已存在，微调） |

- 所有 Canvas 使用 `ResizeObserver` + `devicePixelRatio` 保证 HiDPI 清晰
- 共享 hover tooltip `<div>`，mousemove 定位
- 坐标轴标签使用无衬线字体 11px，图表标题 13px weight 600

### 交互式图表 vs 静态图片分配

- 新 Canvas 交互式：ComparisonBar、DualRadar、ConfidenceHist、TrialScatter、ParallelCoords
- 保留 PNG：confusion_matrices、roc_curves（per-class 数据不在 metrics.json 中，无法生成 Canvas 图表；且这两种图学术上更惯用静态位图）

---

## Implementation Order

| 阶段 | 内容 | 依赖 |
|------|------|------|
| 1 | 后端：新增 3 个 API 端点 + 静态挂载 | 无 |
| 2 | HTML：新增导航项 + 3 个视图容器 + CSS | 无 |
| 3 | Chart4Lib：6 个 Canvas/SVG 图表组件 | 无 |
| 4 | 对比视图：fetch `/api/compare` → 渲染 4 个图表 | 1,2,3 |
| 5 | HP 搜索视图：fetch `/api/trials` → 散点 + 平行坐标 | 1,2,3 |
| 6 | 数据探索视图：data_viz 灯箱画廊 | 1,2 |
| 7 | 模型详情增强：best 信息切换 | 1,2 |
| 8 | 动画优化：过渡/数字滚动/交错入场 | 4,5 |
| 9 | 响应式：新视图窄屏适配 | 4,5,6 |

---

## Verification

1. 启动 `python api_server.py` → 浏览器访问 `localhost:8000`
2. 点击每个导航项，确认 5 个视图全部渲染
3. 鼠标悬停在 Canvas 图表上，确认 tooltip 出现
4. 缩窄浏览器窗口到 500px，确认布局不破裂
5. 从 `/api/compare` 和 `/api/trials` 直接 curl 验证 JSON 正确

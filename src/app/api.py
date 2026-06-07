"""
MBTI 推理 API 服务
==================
FastAPI 服务，启动时加载模型，接收文本返回预测 + 解释。
同时提供模型信息查询和前端静态文件服务。

启动:
    python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000
    # 或
    python api_server.py

测试:
    curl -X POST http://localhost:8000/api/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "I enjoy spending quiet evenings alone..."}'
"""

from __future__ import annotations

import json
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

# ML 依赖延迟加载
_torch = None
_RoBERTaEncoder = None
_MBTIClassifier = None
_AttributionAnalyzer = None
_MBTIInterpreter = None


def _lazy_import_ml():
    """延迟加载 ML 依赖（仅在实际推理时需要）。"""
    global _torch, _RoBERTaEncoder, _MBTIClassifier, _AttributionAnalyzer, _MBTIInterpreter
    if _torch is not None:
        return True
    try:
        import torch as _t
        _torch = _t
        from src.representation import RoBERTaEncoder as _RE
        from src.model import MBTIClassifier as _MC
        from src.explanation import AttributionAnalyzer as _AA, MBTIInterpreter as _MI
        _RoBERTaEncoder = _RE
        _MBTIClassifier = _MC
        _AttributionAnalyzer = _AA
        _MBTIInterpreter = _MI
        return True
    except ImportError as e:
        global _model_available, _model_error
        _model_error = f"ML 依赖未安装: {e}"
        _model_available = False
        return False

# ============================================================
# 路径自动检测
# ============================================================
DEVICE = "cpu"  # 默认值，load_models 中延迟确定


def _find_model_path() -> str | None:
    """自动查找 RoBERTa 模型权重目录。
    搜索顺序：
      1. models/roberta-base/
      2. models/ 下任意包含 config.json 的目录
    """
    models_dir = _PROJECT_ROOT / "models"
    if not models_dir.is_dir():
        return None

    # 优先找 roberta-base
    for candidate in ["roberta-base", "roberta-base-local"]:
        p = models_dir / candidate
        if (p / "config.json").exists():
            return str(p)

    # 扫描 models/ 下所有子目录
    for sub in sorted(models_dir.iterdir()):
        if sub.is_dir() and (sub / "config.json").exists():
            return str(sub)

    return None


def _find_checkpoint_dir() -> str | None:
    """自动查找 checkpoint 目录。
    搜索顺序：
      1. output/ 下最新的 */best/ 子目录
      2. output/ 下最新的 */epoch_N/ 子目录（取最大 epoch）
      3. checkpoints/baseline/
    """
    # 1. 扫描 output/ 找最新训练的 best/
    output_dir = _PROJECT_ROOT / "output"
    if output_dir.is_dir():
        best_dirs = sorted(output_dir.glob("*/best"), reverse=True)
        for d in best_dirs:
            if (d / "encoder.pt").exists() or (d / "classifier.pt").exists():
                print(f"[启动] 找到 checkpoint: {d}")
                return str(d)

        # 2. 没有 best/，找最大 epoch 目录
        for run_dir in sorted(output_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            epoch_dirs = sorted(
                [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
                key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
                reverse=True,
            )
            for d in epoch_dirs:
                if (d / "encoder.pt").exists() or (d / "classifier.pt").exists():
                    print(f"[启动] 找到 checkpoint: {d}")
                    return str(d)

    # 3. 回退到 baseline
    baseline = _PROJECT_ROOT / "checkpoints" / "baseline"
    if baseline.is_dir():
        if (baseline / "encoder.pt").exists() or (baseline / "classifier.pt").exists():
            print(f"[启动] 使用 baseline checkpoint: {baseline}")
            return str(baseline)

    return None

# ============================================================
# 请求 / 响应模型
# ============================================================

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="待预测的英文文本",
                      examples=["I enjoy spending quiet evenings alone with a good book."])

class DimensionProb(BaseModel):
    positive: float
    negative: float

class KeywordItem(BaseModel):
    token: str
    score: float

class PredictionResult(BaseModel):
    mbti_type: str
    probabilities: dict[str, DimensionProb]
    confidence: float

class Interpretation(BaseModel):
    EI: str
    SN: str
    TF: str
    JP: str
    summary: str

class PredictResponse(BaseModel):
    prediction: PredictionResult
    keywords: dict[str, list[KeywordItem]]
    interpretation: Interpretation


# ============================================================
# 模型管理（全局单例）
# ============================================================

_model_cache: dict = {}
_model_available: bool = False
_model_error: str | None = None


def load_models():
    """启动时加载模型，全局复用。模型不可用时优雅降级。"""
    global _model_cache, _model_available, _model_error, DEVICE
    if _model_cache:
        return _model_cache

    print(f"[启动] 加载模型...")

    # 延迟加载 ML 依赖
    if not _lazy_import_ml():
        print("[启动] ML 依赖不可用，以降级模式运行")
        return {}

    # 确定设备
    DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    print(f"[启动] device={DEVICE}")

    # 自动查找模型路径
    model_path = _find_model_path()
    if model_path is None:
        _model_error = (
            "RoBERTa 模型未找到。请下载 roberta-base 权重到以下位置：\n"
            "  • models/roberta-base/\n"
            "  • models/<任意目录名>/\n"
            "下载地址: https://huggingface.co/FacebookAI/roberta-base"
        )
        print(f"[启动] {_model_error}")
        _model_available = False
        return {}

    print(f"[启动] 模型路径: {model_path}")

    # 自动查找 checkpoint
    ckpt_dir = _find_checkpoint_dir()
    if ckpt_dir is None:
        _model_error = (
            "Checkpoint 未找到。训练完成后将 checkpoint 放到以下位置：\n"
            "  • output/<时间戳>/best/\n"
            "  • checkpoints/baseline/\n"
            "训练命令: python train.py"
        )
        print(f"[启动] {_model_error}")
        _model_available = False
        return {}

    print(f"[启动] checkpoint: {ckpt_dir}")

    try:
        encoder = _RoBERTaEncoder(
            model_name=model_path, pooling="mean", max_length=512,
        )
        encoder.load_state_dict(_torch.load(
            f"{ckpt_dir}/encoder.pt", map_location=DEVICE, weights_only=True,
        ))
        encoder.to(DEVICE).eval()

        classifier = _MBTIClassifier()
        classifier.load_state_dict(_torch.load(
            f"{ckpt_dir}/classifier.pt", map_location=DEVICE, weights_only=True,
        ))
        classifier.to(DEVICE).eval()

        _model_cache = {"encoder": encoder, "classifier": classifier}
        _model_available = True
        print(f"[启动] ✅ 模型加载完成")
        return _model_cache
    except Exception as e:
        _model_error = str(e)
        print(f"[启动] ❌ 模型加载失败: {_model_error}")
        _model_available = False
        return {}


def predict(text: str, lang: str = "zh") -> dict:
    """核心推理逻辑（线程安全，模型只读）。"""
    models = _model_cache or load_models()
    encoder = models["encoder"]
    classifier = models["classifier"]

    # 归因分析（含预测概率 + 关键词）
    analyzer = _AttributionAnalyzer(encoder, classifier)
    attr = analyzer.analyze(text)

    # 构建预测结果
    probs = attr["probabilities"]
    prediction = {
        "mbti_type": _probs_to_type(probs),
        "probabilities": probs,
        "confidence": round(sum(
            abs(p["positive"] - 0.5) * 2 for p in probs.values()
        ) / 4, 4),
    }

    # 后处理关键词：清洗 BPE 标记 + 归一化分数
    keywords = {}
    for dim, tokens in attr["keywords"].items():
        cleaned = []
        for t in tokens:
            # 去掉 RoBERTa BPE 空格标记
            word = t["token"].lstrip("Ġ").lstrip("Ċ").lstrip("âĢĶ")
            if word in ("<s>", "</s>", "<pad>"):
                continue
            cleaned.append({"token": word, "score": round(t["score"], 6)})
        # 归一化分数到 [0, 1] 方便前端渲染
        if cleaned:
            abs_max = max(abs(c["score"]) for c in cleaned) or 1e-9
            for c in cleaned:
                c["intensity"] = round(abs(c["score"]) / abs_max, 4)
        keywords[dim] = cleaned[:12]  # 每维度最多 12 个

    # NLG 解读
    interpreter = _MBTIInterpreter()
    interp = interpreter.interpret(probs, attr["keywords"], lang=lang)

    return {
        "prediction": prediction,
        "keywords": keywords,
        "interpretation": {
            "EI": interp["interpretation"]["EI"],
            "SN": interp["interpretation"]["SN"],
            "TF": interp["interpretation"]["TF"],
            "JP": interp["interpretation"]["JP"],
            "summary": interp["summary"],
        },
    }


def _probs_to_type(probs: dict) -> str:
    ei = "E" if probs["EI"]["positive"] >= 0.5 else "I"
    sn = "S" if probs["SN"]["positive"] >= 0.5 else "N"
    tf = "T" if probs["TF"]["positive"] >= 0.5 else "F"
    jp = "J" if probs["JP"]["positive"] >= 0.5 else "P"
    return f"{ei}{sn}{tf}{jp}"


# ============================================================
# 模型信息加载
# ============================================================

def _find_latest_training_info() -> dict | None:
    """查找最新的 training_info.json。"""
    output_dir = _PROJECT_ROOT / "output"
    if not output_dir.exists():
        return None
    # 按时间戳排序，取最新的
    dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        reverse=True,
    )
    for d in dirs:
        info_path = d / "training_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_metrics() -> dict | None:
    """加载 eval_output/metrics.json。"""
    metrics_path = _PROJECT_ROOT / "eval_output" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _compute_checkpoint_params() -> dict:
    """从 checkpoint 文件统计参数量。需要 torch 可用。"""
    encoder_params = 0
    classifier_params = 0

    if _torch is None:
        return {"encoder": 0, "classifier": 0, "total": 0}

    # 尝试从自动检测的 checkpoint 路径读取
    ckpt_dir = _find_checkpoint_dir()
    if ckpt_dir is None:
        return {"encoder": 0, "classifier": 0, "total": 0}

    encoder_path = Path(ckpt_dir) / "encoder.pt"
    if encoder_path.exists():
        try:
            state = _torch.load(str(encoder_path), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                encoder_params = sum(v.numel() for v in state.values())
        except Exception:
            pass

    classifier_path = Path(ckpt_dir) / "classifier.pt"
    if classifier_path.exists():
        try:
            state = _torch.load(str(classifier_path), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                classifier_params = sum(v.numel() for v in state.values())
        except Exception:
            pass

    return {
        "encoder": encoder_params,
        "classifier": classifier_params,
        "total": encoder_params + classifier_params,
    }


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时清理。模型加载失败不阻止启动。"""
    try:
        load_models()
    except Exception as e:
        print(f"[启动] 模型加载异常（服务将以降级模式运行）: {e}")
    yield
    _model_cache.clear()


app = FastAPI(
    title="MBTI 性格预测 API",
    description="基于 RoBERTa 的英文文本 MBTI 人格类型预测与解释",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API 路由
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_available": _model_available,
        "model_error": _model_error,
    }


@app.get("/api/model")
def api_model():
    """返回模型元信息、超参数、评估指标和参数量统计。"""
    training_info = _find_latest_training_info()
    metrics = _load_metrics()
    params = _compute_checkpoint_params()

    result = {
        "model_name": "roberta-base",
        "pooling": "mean",
        "hidden_size": 768,
        "max_length": training_info.get("config", {}).get("max_length", 256) if training_info else 256,
        "device": str(DEVICE),
        "params": {
            "encoder": params["encoder"],
            "classifier": params["classifier"],
            "total": params["total"],
            "encoder_human": _format_params(params["encoder"]),
            "classifier_human": _format_params(params["classifier"]),
            "total_human": _format_params(params["total"]),
        },
        "training_config": training_info.get("config", {}) if training_info else {},
        "best_epoch": training_info.get("best_epoch") if training_info else None,
        "metrics": metrics,
    }
    return result


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    """预测 MBTI 类型并返回解释。"""
    if not _model_available:
        raise HTTPException(
            status_code=503,
            detail=f"模型未就绪。{_model_error or '请先下载模型权重和 checkpoint 文件。'}"
        )
    try:
        return predict(req.text, lang="zh")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 静态文件挂载（必须在路由之后）
# ============================================================

# 评估图片
eval_dir = _PROJECT_ROOT / "eval_output"
if eval_dir.exists():
    app.mount("/static/eval", StaticFiles(directory=str(eval_dir)), name="eval_static")

# 前端页面 - 挂载到根路径
frontend_dir = _PROJECT_ROOT / "src" / "app"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ============================================================
# 辅助函数
# ============================================================

def _format_params(n: int) -> str:
    """将参数数量格式化为人类可读字符串。"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    else:
        return str(n)

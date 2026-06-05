"""
MBTI 推理 API 服务
==================
FastAPI 服务，启动时加载模型，接收文本返回预测 + 解释。

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

import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier
from src.explanation import AttributionAnalyzer, MBTIInterpreter

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "D:/ML/MBTI_pred/models/roberta-base"
CKPT_DIR = "checkpoints/baseline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_models():
    """启动时加载模型，全局复用。"""
    global _model_cache
    if _model_cache:
        return _model_cache

    print(f"[启动] 加载模型... (device={DEVICE})")

    encoder = RoBERTaEncoder(
        model_name=MODEL_PATH, pooling="mean", max_length=512,
    )
    encoder.load_state_dict(torch.load(
        f"{CKPT_DIR}/encoder.pt", map_location=DEVICE, weights_only=True,
    ))
    encoder.to(DEVICE).eval()

    classifier = MBTIClassifier()
    classifier.load_state_dict(torch.load(
        f"{CKPT_DIR}/classifier.pt", map_location=DEVICE, weights_only=True,
    ))
    classifier.to(DEVICE).eval()

    _model_cache = {"encoder": encoder, "classifier": classifier}
    print(f"[启动] 模型加载完成")
    return _model_cache


def predict(text: str, lang: str = "zh") -> dict:
    """核心推理逻辑（线程安全，模型只读）。"""
    models = _model_cache or load_models()
    encoder = models["encoder"]
    classifier = models["classifier"]

    # 归因分析（含预测概率 + 关键词）
    analyzer = AttributionAnalyzer(encoder, classifier)
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

    # NLG 解读
    interpreter = MBTIInterpreter()
    interp = interpreter.interpret(probs, attr["keywords"], lang=lang)

    return {
        "prediction": prediction,
        "keywords": attr["keywords"],
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
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时清理。"""
    load_models()
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


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    """预测 MBTI 类型并返回解释。"""
    try:
        return predict(req.text, lang="zh")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

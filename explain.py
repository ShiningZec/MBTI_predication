"""
MBTI 解释脚本 — 输出结构化 JSON 供前端使用
============================================

Usage:
    python explain.py "I enjoy reading books alone..."
    python explain.py --text "Your text here"
    python explain.py --file input.txt
    python explain.py --text "..." --lang en --no-attention

输出 JSON 结构:
{
    "prediction":  { "mbti_type": "ISTJ", "confidence": 0.77, "probabilities": {...} },
    "keywords":    { "EI": [{"token": "...", "score": 0.23}, ...], ... },
    "attention":   { "tokens": [...], "cls_attention": [...], ... },
    "attribution": { "EI": [0.01, 0.23, ...], ... },
    "interpretation": { "EI": "...", "SN": "...", ..., "summary": "..." }
}
"""

import argparse
import json
import sys
import warnings

import torch

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier
from src.explanation import AttributionAnalyzer, AttentionExtractor, MBTIInterpreter

warnings.filterwarnings("ignore")

CKPT = "checkpoints/baseline"
MODEL_PATH = "D:/ML/MBTI_pred/models/roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pipeline():
    """加载已训练的 encoder + classifier。"""
    encoder = RoBERTaEncoder(
        model_name=MODEL_PATH, pooling="mean", max_length=512,
    )
    encoder.load_state_dict(torch.load(
        f"{CKPT}/encoder.pt", map_location=DEVICE, weights_only=True,
    ))
    encoder.to(DEVICE).eval()

    classifier = MBTIClassifier()
    classifier.load_state_dict(torch.load(
        f"{CKPT}/classifier.pt", map_location=DEVICE, weights_only=True,
    ))
    classifier.to(DEVICE).eval()

    return encoder, classifier


def explain(
    text: str,
    encoder: RoBERTaEncoder,
    classifier: MBTIClassifier,
    lang: str = "zh",
    include_attention: bool = True,
    include_attribution: bool = True,
) -> dict:
    """对单条文本执行完整解释管线，返回结构化 dict。"""

    result: dict = {}

    # ---- 1. 归因分析（含预测概率 + 关键词） ----
    if include_attribution:
        try:
            analyzer = AttributionAnalyzer(encoder, classifier)
            attr = analyzer.analyze(text)
            result["prediction"] = {
                "mbti_type": _prob_to_type(attr["probabilities"]),
                "probabilities": attr["probabilities"],
                "confidence": round(sum(
                    abs(p["positive"] - 0.5) * 2
                    for p in attr["probabilities"].values()
                ) / 4, 4),
            }
            result["keywords"] = attr["keywords"]
            result["attribution"] = attr["attribution"]
            result["tokens"] = attr["tokens"]
        except Exception as e:
            result["prediction"] = {"error": str(e)}
            result["keywords"] = {}

    # ---- 2. 注意力 ----
    if include_attention:
        try:
            extractor = AttentionExtractor(encoder)
            attn = extractor.extract(text)
            result["attention"] = {
                "tokens": attn["tokens"],
                "cls_attention": attn["cls_attention"],
                "num_layers": attn["num_layers"],
                "num_heads": attn["num_heads"],
            }
        except Exception as e:
            result["attention"] = {"error": str(e)}

    # ---- 3. NLG 解读 ----
    if "prediction" in result and "probabilities" in result["prediction"]:
        interpreter = MBTIInterpreter()
        interpretation = interpreter.interpret(
            result["prediction"]["probabilities"],
            result.get("keywords"),
            lang=lang,
        )
        result["interpretation"] = {
            "EI": interpretation["interpretation"]["EI"],
            "SN": interpretation["interpretation"]["SN"],
            "TF": interpretation["interpretation"]["TF"],
            "JP": interpretation["interpretation"]["JP"],
            "summary": interpretation["summary"],
        }

    return result


def _prob_to_type(probs: dict) -> str:
    ei = "E" if probs["EI"]["positive"] >= 0.5 else "I"
    sn = "S" if probs["SN"]["positive"] >= 0.5 else "N"
    tf = "T" if probs["TF"]["positive"] >= 0.5 else "F"
    jp = "J" if probs["JP"]["positive"] >= 0.5 else "P"
    return f"{ei}{sn}{tf}{jp}"


def main():
    parser = argparse.ArgumentParser(description="MBTI 解释管线")
    parser.add_argument("text", nargs="?", type=str, default=None,
                        help="输入文本")
    parser.add_argument("--text", "-t", dest="text_opt", type=str, default=None)
    parser.add_argument("--file", "-f", type=str, default=None)
    parser.add_argument("--lang", "-l", type=str, default="zh",
                        choices=["zh", "en"])
    parser.add_argument("--no-attention", action="store_true",
                        help="跳过注意力提取（加速）")
    parser.add_argument("--no-attribution", action="store_true",
                        help="跳过 IG 归因（加速）")
    args = parser.parse_args()

    raw = args.text or args.text_opt
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    if not raw:
        print("请提供输入文本: python explain.py 'your text here'", file=sys.stderr)
        sys.exit(1)

    print("加载模型...", file=sys.stderr)
    encoder, classifier = load_pipeline()

    print("分析中...", file=sys.stderr)
    result = explain(
        raw, encoder, classifier,
        lang=args.lang,
        include_attention=not args.no_attention,
        include_attribution=not args.no_attribution,
    )

    out_path = "explanation_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    main()

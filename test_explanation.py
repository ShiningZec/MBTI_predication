"""解释层测试：加载模型，对示例文本给出完整解释。"""
import sys
import torch

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier
from src.explanation import AttributionAnalyzer, AttentionExtractor, MBTIInterpreter

CKPT = "checkpoints/baseline"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}\n")

# ---- 加载模型 ----
encoder = RoBERTaEncoder(model_name="D:/ML/MBTI_pred/models/roberta-base", pooling="mean", max_length=512)
encoder.load_state_dict(torch.load(f"{CKPT}/encoder.pt", map_location=device, weights_only=True))
encoder.to(device).eval()

classifier = MBTIClassifier()
classifier.load_state_dict(torch.load(f"{CKPT}/classifier.pt", map_location=device, weights_only=True))
classifier.to(device).eval()
print("模型加载完成\n")

# ---- 测试文本 ----
text = (
    "I enjoy spending quiet evenings alone with a good book. "
    "Large social gatherings drain my energy, and I prefer deep "
    "one-on-one conversations. I tend to overthink decisions and "
    "plan everything in advance rather than going with the flow."
)
print(f"输入文本: {text}\n")

# ---- 1. 归因分析 ----
print("=" * 60)
print("Integrated Gradients 归因")
print("=" * 60)
analyzer = AttributionAnalyzer(encoder, classifier)
attr_result = analyzer.analyze(text)

print(f"预测: {attr_result['probabilities']}")
print(f"\n每维度 Top-5 关键词:")
for dim in ["EI", "SN", "TF", "JP"]:
    tokens = []
    for k in attr_result["keywords"][dim][:5]:
        t = k["token"].replace("Ġ", "").replace("Ċ", "")
        tokens.append(f"{t}({k['score']:.3f})")
    print(f"  {dim}: {', '.join(tokens)}")

# ---- 2. 注意力 ----
print(f"\n{'=' * 60}")
print("注意力热力图数据")
print("=" * 60)
extractor = AttentionExtractor(encoder)
attn = extractor.extract(text)
print(f"  层数: {attn['num_layers']}, 头数: {attn['num_heads']}, Token数: {len(attn['tokens'])}")
print(f"  注意力矩阵形状: {len(attn['merged_attention'])}x{len(attn['merged_attention'][0])}")

# CLS 关注度最高的 token
cls_scores = attn["cls_attention"]
top_cls_idx = sorted(range(len(cls_scores)), key=lambda i: cls_scores[i], reverse=True)[:8]
print(f"  <s> 最关注的 tokens: ", end="")
for idx in top_cls_idx:
    t = attn['tokens'][idx].replace("Ġ", "").replace("Ċ", "")
    print(f"{t}({cls_scores[idx]:.3f})", end=" ")
print()

# ---- 3. NLG 解读 ----
print(f"\n{'=' * 60}")
print("NLG 人格解读")
print("=" * 60)
interpreter = MBTIInterpreter()
interpretation = interpreter.interpret(
    attr_result["probabilities"],
    attr_result["keywords"],
)

print(f"\nMBTI 类型: {interpretation['mbti_type']}\n")
for dim in ["EI", "SN", "TF", "JP"]:
    print(f"  [{dim}] {interpretation['interpretation'][dim]}\n")
print(f"  总结: {interpretation['summary']}")

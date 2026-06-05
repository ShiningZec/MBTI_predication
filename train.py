"""
MBTI 模型训练入口
================
所有可调参数在 config/default.yaml 中配置。
CLI 参数会覆盖配置文件中的对应值。

Usage:
    python train.py                          # 使用 config/default.yaml
    python train.py --cfg config/exp1.yaml   # 指定配置文件
    python train.py --epochs 10 --lr 5e-5   # CLI 覆盖部分参数
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.representation import RoBERTaEncoder
from src.model import MBTIClassifier, MBTITrainer, TrainingConfig


def load_config(cfg_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli_args(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI 参数覆盖配置文件中的对应值。"""
    overrides = {
        "model": ["name", "pooling", "max_length", "freeze_backbone", "dropout"],
        "classifier": ["head_hidden"],
        "training": ["num_epochs", "batch_size", "gradient_accumulation_steps",
                     "max_grad_norm", "fp16", "num_workers", "seed",
                     "early_stopping_patience"],
        "optimizer": ["learning_rate", "classifier_lr", "weight_decay", "warmup_ratio"],
        "loss": ["dim_weights", "pos_weights"],
        "data": ["train_csv", "test_csv", "label_map"],
        "output": ["dir", "save_best", "log_interval"],
    }

    cli_map = {
        "name": "model", "pooling": "pooling", "max_length": "max_len",
        "freeze_backbone": "freeze", "dropout": "dropout",
        "num_epochs": "epochs", "batch_size": "bs",
        "learning_rate": "lr", "classifier_lr": "cls_lr",
        "fp16": "fp16", "seed": "seed",
        "dir": "output_dir", "train_csv": "train_csv", "test_csv": "test_csv",
    }

    for section, keys in overrides.items():
        for key in keys:
            cli_attr = cli_map.get(key, key)
            val = getattr(args, cli_attr, None)
            if val is not None and val != parser.get_default(cli_attr):
                cfg[section][key] = val

    return cfg


def build_parser():
    """构建 CLI 参数解析器（所有默认值来自配置文件，此处仅声明类型）。"""
    parser = argparse.ArgumentParser(description="MBTI 模型训练")
    parser.add_argument("--cfg", type=str, default="config/default.yaml",
                        help="配置文件路径")
    # 模型
    parser.add_argument("--model", type=str, default=None, help="模型名/路径")
    parser.add_argument("--pooling", type=str, default=None,
                        choices=["cls", "mean", "max", "attention"])
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--freeze", action="store_true", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    # 训练
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    # 优化器
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cls-lr", type=float, default=None)
    # 数据
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    # 输出
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---- 加载配置 ----
    cfg = load_config(args.cfg)
    cfg = merge_cli_args(cfg, args)

    m = cfg["model"]
    clf = cfg["classifier"]
    t = cfg["training"]
    opt = cfg["optimizer"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]
    out = cfg["output"]

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 编码器 ----
    print(f"\n加载编码器: {m['name']} (pooling={m['pooling']})...")
    encoder = RoBERTaEncoder(
        model_name=m["name"],
        pooling=m["pooling"],
        max_length=m["max_length"],
        freeze_backbone=m["freeze_backbone"],
    )
    print(encoder)

    # ---- 分类器 ----
    classifier = MBTIClassifier(
        input_dim=encoder.hidden_size,
        head_hidden=clf["head_hidden"],
        dropout=m["dropout"],
    )

    # ---- 训练配置 ----
    config = TrainingConfig(
        num_epochs=t["num_epochs"],
        batch_size=t["batch_size"],
        max_length=m["max_length"],
        learning_rate=opt["learning_rate"],
        classifier_lr=opt["classifier_lr"],
        weight_decay=opt["weight_decay"],
        warmup_ratio=opt["warmup_ratio"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_grad_norm=t["max_grad_norm"],
        fp16=t["fp16"],
        num_workers=t["num_workers"],
        seed=t["seed"],
        early_stopping_patience=t["early_stopping_patience"],
        dim_weights=loss_cfg.get("dim_weights", {"EI": 0.25, "SN": 0.25, "TF": 0.25, "JP": 0.25}),
        pos_weights={k: v for k, v in loss_cfg.get("pos_weights", {}).items() if v is not None},
        train_csv=data_cfg["train_csv"],
        test_csv=data_cfg["test_csv"],
        label_map_path=data_cfg.get("label_map", "data/label_map.json"),
        output_dir=out["dir"],
        save_best=out["save_best"],
        log_interval=out["log_interval"],
    )

    print(f"\n训练配置:")
    print(f"  epochs={config.num_epochs}, batch_size={config.batch_size}")
    print(f"  encoder_lr={config.learning_rate}, classifier_lr={config.classifier_lr}")
    print(f"  max_len={config.max_length}, fp16={config.fp16}")
    print(f"  freeze_backbone={m['freeze_backbone']}")
    print(f"  dim_weights={config.dim_weights}")
    if config.pos_weights:
        print(f"  pos_weights={config.pos_weights}")

    # ---- 训练 ----
    trainer = MBTITrainer(encoder, classifier, config)
    metrics = trainer.fit()

    print(f"\n最终测试集指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

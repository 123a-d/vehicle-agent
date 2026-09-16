"""
阶段三：Baseline 模型重新加载与验证分析
文件名：evaluate_baseline.py

作用：
1. 从磁盘重新加载训练阶段保存的 baseline_best.pth。
2. 重新构建 ResNet18 51 类模型。
3. 在完整 val 验证集上重新推理。
4. 重新计算 Accuracy、Macro-F1。
5. 输出每个类别的 Precision / Recall / F1 / 样本数。
6. 单独打印 unknown 类表现。
7. 生成混淆矩阵和最常见误判对。

运行命令：
    python evaluate_baseline.py
"""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = PROJECT_ROOT / "dataset_51" / "val"

BASELINE_DIR = PROJECT_ROOT / "outputs" / "baseline_resnet18"
CHECKPOINT_PATH = BASELINE_DIR / "baseline_best.pth"

EVAL_DIR = BASELINE_DIR / "evaluation"
SUMMARY_PATH = EVAL_DIR / "evaluation_summary.json"
PER_CLASS_PATH = EVAL_DIR / "per_class_metrics.csv"
CONFUSION_CSV_PATH = EVAL_DIR / "confusion_matrix.csv"
TOP_CONFUSIONS_PATH = EVAL_DIR / "top_confusions.csv"
CONFUSION_PNG_PATH = EVAL_DIR / "confusion_matrix.png"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_val_dataset():
    """使用与训练阶段兼容的预处理读取验证集。"""
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"找不到验证集目录：{VAL_DIR}")

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform,
    )


def load_model(device: torch.device):
    """
    从 checkpoint 恢复模型。

    checkpoint 保存的是参数，所以需要先建立同样的 ResNet18 + 51 分类头，
    再将 model_state_dict 加载进去。
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        checkpoint["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def run_inference(model, val_loader, device):
    """对整个验证集推理，返回真实索引和预测索引。"""
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred


def save_per_class_metrics(report, class_names):
    """保存每个类别的 Precision / Recall / F1 / support。"""
    rows = []

    for class_name in class_names:
        metrics = report[class_name]
        rows.append(
            {
                "class_name": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )

    with PER_CLASS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(matrix, class_names):
    """
    保存混淆矩阵。
    行 = 真实类别，列 = 预测类别。
    """
    with CONFUSION_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred"] + class_names)

        for class_name, row in zip(
            class_names,
            matrix,
        ):
            writer.writerow(
                [class_name] + row.tolist()
            )


def save_top_confusions(matrix, class_names, top_k=20):
    """保存最常见的误判方向，供下一阶段分析。"""
    rows = []

    for true_index in range(len(class_names)):
        for pred_index in range(len(class_names)):
            if true_index == pred_index:
                continue

            count = int(
                matrix[true_index, pred_index]
            )

            if count > 0:
                rows.append(
                    {
                        "true_class": class_names[true_index],
                        "predicted_class": class_names[pred_index],
                        "count": count,
                    }
                )

    rows.sort(
        key=lambda row: row["count"],
        reverse=True,
    )

    rows = rows[:top_k]

    with TOP_CONFUSIONS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "true_class",
                "predicted_class",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_figure(matrix):
    """保存混淆矩阵图片，用于观察整体错误分布。"""
    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(
        "Baseline ResNet18 Confusion Matrix"
    )
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(
        CONFUSION_PNG_PATH,
        dpi=160,
    )
    plt.close(fig)


def main():
    """Baseline 重新加载与验证分析主流程。"""

    print("=" * 72)
    print("阶段三：Baseline 重新加载与验证分析")
    print("=" * 72)

    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"计算设备：{device}")
    print(f"Checkpoint：{CHECKPOINT_PATH}")

    model, checkpoint = load_model(device)

    print(
        f"Checkpoint 保存 Epoch：{checkpoint['epoch']}"
    )
    print(
        "Checkpoint 原 Macro-F1："
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    val_dataset = build_val_dataset()
    class_names = val_dataset.classes

    if (
        checkpoint["class_to_idx"]
        != val_dataset.class_to_idx
    ):
        raise ValueError(
            "Checkpoint 中类别映射与当前 val 数据集不一致。"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"验证图片数：{len(val_dataset)}")
    print(f"类别数：{len(class_names)}")
    print("\n开始重新推理完整验证集...")

    y_true, y_pred = run_inference(
        model,
        val_loader,
        device,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    unknown_metrics = report["unknown"]

    print("\n验证结果")
    print("-" * 72)
    print(f"Accuracy：{accuracy:.4f}")
    print(f"Macro-F1：{macro_f1:.4f}")
    print(
        "unknown Precision："
        f"{unknown_metrics['precision']:.4f}"
    )
    print(
        "unknown Recall："
        f"{unknown_metrics['recall']:.4f}"
    )
    print(
        "unknown F1："
        f"{unknown_metrics['f1-score']:.4f}"
    )
    print(
        "unknown Support："
        f"{int(unknown_metrics['support'])}"
    )

    save_per_class_metrics(
        report,
        class_names,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    save_confusion_matrix_csv(
        matrix,
        class_names,
    )

    save_top_confusions(
        matrix,
        class_names,
        top_k=20,
    )

    save_confusion_matrix_figure(
        matrix,
    )

    summary = {
        "model": "resnet18_baseline",
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "num_validation_images": len(
            val_dataset
        ),
        "num_classes": len(
            class_names
        ),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "unknown": {
            "precision": float(
                unknown_metrics["precision"]
            ),
            "recall": float(
                unknown_metrics["recall"]
            ),
            "f1": float(
                unknown_metrics["f1-score"]
            ),
            "support": int(
                unknown_metrics["support"]
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n分析文件已生成：")
    print(f"- {SUMMARY_PATH}")
    print(f"- {PER_CLASS_PATH}")
    print(f"- {CONFUSION_CSV_PATH}")
    print(f"- {TOP_CONFUSIONS_PATH}")
    print(f"- {CONFUSION_PNG_PATH}")
    print("\n重新加载验证完成。")


if __name__ == "__main__":
    main()

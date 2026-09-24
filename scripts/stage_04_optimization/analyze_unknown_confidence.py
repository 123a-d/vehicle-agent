"""
阶段四 · Unknown 专项分析 4B：Confidence 分布 + Threshold 扫描
文件名：analyze_unknown_confidence.py

本脚本不训练模型，只分析当前最佳模型：
    outputs/unknown_weighted/best_macro_f1.pth

它会：
1. 记录验证集每张图片的 Top-1 / Top-2 confidence；
2. 比较 known 正确、known 错误、真实 unknown 等不同组的 confidence；
3. 扫描多个 threshold；
4. 观察 threshold 对 Macro-F1、Known-F1、Unknown P/R/F1 的影响。

运行：
    python analyze_unknown_confidence.py
"""

from pathlib import Path
import csv

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_DIR = PROJECT_ROOT / "dataset_51" / "val"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unknown_weighted"
    / "best_macro_f1.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "confidence_analysis_weighted"
)

PER_IMAGE_CSV = OUTPUT_DIR / "per_image_confidence.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "confidence_group_summary.csv"
THRESHOLD_SCAN_CSV = OUTPUT_DIR / "threshold_scan.csv"
SUMMARY_TXT = OUTPUT_DIR / "analysis_summary.txt"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

NUM_CLASSES = 51
UNKNOWN_INDEX = 50

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 从 0.00 扫到 0.95，步长 0.05
THRESHOLDS = [
    round(float(x), 2)
    for x in np.arange(0.00, 1.00, 0.05)
]


def build_val_dataset():
    """构建验证集，保持与之前评估完全一致的预处理。"""

    transform = transforms.Compose(
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

    dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=transform,
    )

    if len(dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"类别数应为 {NUM_CLASSES}，实际为 {len(dataset.classes)}。"
        )

    if dataset.class_to_idx.get("unknown") != UNKNOWN_INDEX:
        raise ValueError("unknown 索引不是 50。")

    return dataset


def load_model(device):
    """加载当前最佳 weighted-unknown checkpoint。"""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{CHECKPOINT_PATH}"
        )

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


def collect_predictions(model, dataset, device):
    """
    收集每张图片的原始预测和 confidence。

    Top-1 confidence：
        51个 softmax 概率中最大的那个。

    Top1-Top2 margin：
        第一名概率 - 第二名概率。
        margin 越小，模型通常越犹豫。
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    rows = []
    global_index = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            top2_values, top2_indices = probabilities.topk(
                k=2,
                dim=1,
            )

            for i in range(labels.size(0)):
                true_index = int(labels[i].item())

                pred_index = int(
                    top2_indices[i, 0].item()
                )

                top1_conf = float(
                    top2_values[i, 0].item()
                )

                top2_index = int(
                    top2_indices[i, 1].item()
                )

                top2_conf = float(
                    top2_values[i, 1].item()
                )

                image_path = dataset.samples[
                    global_index
                ][0]

                rows.append(
                    {
                        "image_path": image_path,
                        "true_index": true_index,
                        "true_label": dataset.classes[true_index],
                        "pred_index": pred_index,
                        "pred_label": dataset.classes[pred_index],
                        "top1_confidence": top1_conf,
                        "top2_index": top2_index,
                        "top2_label": dataset.classes[top2_index],
                        "top2_confidence": top2_conf,
                        "top1_top2_margin": top1_conf - top2_conf,
                        "is_correct": int(
                            pred_index == true_index
                        ),
                        "is_true_unknown": int(
                            true_index == UNKNOWN_INDEX
                        ),
                        "is_pred_unknown": int(
                            pred_index == UNKNOWN_INDEX
                        ),
                    }
                )

                global_index += 1

    return rows


def save_csv(path, rows):
    """通用 CSV 保存函数。"""

    if not rows:
        return

    with path.open(
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


def describe(values):
    """统计一组 confidence 的基本分布。"""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_group_summary(rows):
    """构建不同样本组的 confidence 分布统计。"""

    groups = {
        "known_correct": [
            row["top1_confidence"]
            for row in rows
            if (
                row["is_true_unknown"] == 0
                and row["is_correct"] == 1
            )
        ],
        "known_wrong": [
            row["top1_confidence"]
            for row in rows
            if (
                row["is_true_unknown"] == 0
                and row["is_correct"] == 0
            )
        ],
        "true_unknown_all": [
            row["top1_confidence"]
            for row in rows
            if row["is_true_unknown"] == 1
        ],
        "true_unknown_correct": [
            row["top1_confidence"]
            for row in rows
            if (
                row["is_true_unknown"] == 1
                and row["is_correct"] == 1
            )
        ],
        "true_unknown_missed": [
            row["top1_confidence"]
            for row in rows
            if (
                row["is_true_unknown"] == 1
                and row["is_correct"] == 0
            )
        ],
    }

    result = []

    for name, values in groups.items():
        result.append(
            {
                "group": name,
                **describe(values),
            }
        )

    return result


def calculate_metrics(y_true, y_pred):
    """统一计算总体、known、unknown 指标。"""

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    known_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(UNKNOWN_INDEX)),
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[UNKNOWN_INDEX],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "known_macro_f1": float(known_macro_f1),
        "unknown_precision": float(precision[0]),
        "unknown_recall": float(recall[0]),
        "unknown_f1": float(f1[0]),
        "unknown_support": int(support[0]),
    }


def scan_thresholds(rows):
    """
    扫描 threshold。

    规则：
        1. 原始预测若已经是 unknown，保持 unknown；
        2. 原始预测为 known 且 confidence < threshold，
           强制改成 unknown；
        3. 其余情况保持原始预测。
    """

    y_true = [
        row["true_index"]
        for row in rows
    ]

    true_known_count = sum(
        1
        for label in y_true
        if label != UNKNOWN_INDEX
    )

    results = []

    for threshold in THRESHOLDS:
        y_pred = []
        threshold_rejected_count = 0

        for row in rows:
            raw_pred = row["pred_index"]
            confidence = row["top1_confidence"]

            if raw_pred == UNKNOWN_INDEX:
                final_pred = UNKNOWN_INDEX

            elif confidence < threshold:
                final_pred = UNKNOWN_INDEX
                threshold_rejected_count += 1

            else:
                final_pred = raw_pred

            y_pred.append(final_pred)

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        known_rejected_count = sum(
            1
            for true_label, pred_label
            in zip(y_true, y_pred)
            if (
                true_label != UNKNOWN_INDEX
                and pred_label == UNKNOWN_INDEX
            )
        )

        known_rejection_rate = (
            known_rejected_count
            / true_known_count
        )

        results.append(
            {
                "threshold": threshold,
                **metrics,
                "threshold_rejected_count": threshold_rejected_count,
                "known_rejected_count": known_rejected_count,
                "known_rejection_rate": known_rejection_rate,
            }
        )

    return results


def find_group(summary_rows, group_name):
    """从分组统计中找到指定 group。"""

    for row in summary_rows:
        if row["group"] == group_name:
            return row

    raise KeyError(group_name)


def write_summary(
    checkpoint,
    group_summary,
    threshold_rows,
):
    """生成文本版总结，并返回重点结果。"""

    baseline = threshold_rows[0]

    best_macro = max(
        threshold_rows,
        key=lambda row: row["macro_f1"],
    )

    best_unknown = max(
        threshold_rows,
        key=lambda row: row["unknown_f1"],
    )

    known_correct = find_group(
        group_summary,
        "known_correct",
    )

    unknown_missed = find_group(
        group_summary,
        "true_unknown_missed",
    )

    lines = [
        "阶段四 Unknown Confidence 分析",
        "=" * 60,
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Checkpoint Epoch: {checkpoint.get('epoch')}",
        "",
        "原始模型（threshold=0.00）",
        f"Macro-F1: {baseline['macro_f1']:.4f}",
        f"Known Macro-F1: {baseline['known_macro_f1']:.4f}",
        f"Unknown Precision: {baseline['unknown_precision']:.4f}",
        f"Unknown Recall: {baseline['unknown_recall']:.4f}",
        f"Unknown F1: {baseline['unknown_f1']:.4f}",
        "",
        "Confidence 分布",
        (
            "Known 正确："
            f"mean={known_correct['mean']:.4f}, "
            f"median={known_correct['median']:.4f}"
        ),
        (
            "真实 unknown 但误判成 known："
            f"mean={unknown_missed['mean']:.4f}, "
            f"median={unknown_missed['median']:.4f}"
        ),
        "",
        "Macro-F1 最佳 threshold",
        f"threshold={best_macro['threshold']:.2f}",
        f"Macro-F1={best_macro['macro_f1']:.4f}",
        f"Known-F1={best_macro['known_macro_f1']:.4f}",
        f"Unknown F1={best_macro['unknown_f1']:.4f}",
        (
            "Unknown P/R="
            f"{best_macro['unknown_precision']:.4f}/"
            f"{best_macro['unknown_recall']:.4f}"
        ),
        (
            "Known rejection rate="
            f"{best_macro['known_rejection_rate']:.4f}"
        ),
        "",
        "Unknown F1 最佳 threshold",
        f"threshold={best_unknown['threshold']:.2f}",
        f"Macro-F1={best_unknown['macro_f1']:.4f}",
        f"Known-F1={best_unknown['known_macro_f1']:.4f}",
        f"Unknown F1={best_unknown['unknown_f1']:.4f}",
        (
            "Unknown P/R="
            f"{best_unknown['unknown_precision']:.4f}/"
            f"{best_unknown['unknown_recall']:.4f}"
        ),
        (
            "Known rejection rate="
            f"{best_unknown['known_rejection_rate']:.4f}"
        ),
        "",
        "注意：threshold 是在验证集上调出来的超参数。",
        "如果后续有独立 test，应只在 test 上做最终客观评估。",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return (
        baseline,
        best_macro,
        best_unknown,
        known_correct,
        unknown_missed,
    )


def main():
    """主流程。"""

    print("=" * 80)
    print(
        "阶段四 · Unknown专项分析4B："
        "Confidence分布 + Threshold扫描"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"设备：{device}")
    print(f"模型：{CHECKPOINT_PATH}")

    dataset = build_val_dataset()

    print(
        f"验证集：{len(dataset)} 张"
    )

    model, checkpoint = load_model(
        device
    )

    print(
        "\n正在收集每张图片的 confidence..."
    )

    rows = collect_predictions(
        model,
        dataset,
        device,
    )

    save_csv(
        PER_IMAGE_CSV,
        rows,
    )

    group_summary = build_group_summary(
        rows
    )

    save_csv(
        GROUP_SUMMARY_CSV,
        group_summary,
    )

    threshold_rows = scan_thresholds(
        rows
    )

    save_csv(
        THRESHOLD_SCAN_CSV,
        threshold_rows,
    )

    (
        baseline,
        best_macro,
        best_unknown,
        known_correct,
        unknown_missed,
    ) = write_summary(
        checkpoint,
        group_summary,
        threshold_rows,
    )

    print("\n" + "-" * 80)
    print("原始模型（threshold=0.00）")
    print(
        f"Macro-F1：{baseline['macro_f1']:.4f}"
    )
    print(
        f"Known-F1：{baseline['known_macro_f1']:.4f}"
    )
    print(
        "Unknown P/R/F1："
        f"{baseline['unknown_precision']:.4f} / "
        f"{baseline['unknown_recall']:.4f} / "
        f"{baseline['unknown_f1']:.4f}"
    )

    print("\nConfidence 分布重点：")
    print(
        "Known预测正确："
        f"mean={known_correct['mean']:.4f}, "
        f"median={known_correct['median']:.4f}"
    )
    print(
        "真实Unknown但误判Known："
        f"mean={unknown_missed['mean']:.4f}, "
        f"median={unknown_missed['median']:.4f}"
    )

    print("\nMacro-F1 最佳 threshold：")
    print(
        f"threshold = {best_macro['threshold']:.2f}"
    )
    print(
        f"Macro-F1 = {best_macro['macro_f1']:.4f}"
    )
    print(
        f"Known-F1 = {best_macro['known_macro_f1']:.4f}"
    )
    print(
        f"Unknown F1 = {best_macro['unknown_f1']:.4f}"
    )
    print(
        "Unknown P/R = "
        f"{best_macro['unknown_precision']:.4f} / "
        f"{best_macro['unknown_recall']:.4f}"
    )
    print(
        "Known rejection rate = "
        f"{best_macro['known_rejection_rate']:.4f}"
    )

    print("\nUnknown F1 最佳 threshold：")
    print(
        f"threshold = {best_unknown['threshold']:.2f}"
    )
    print(
        f"Macro-F1 = {best_unknown['macro_f1']:.4f}"
    )
    print(
        f"Known-F1 = {best_unknown['known_macro_f1']:.4f}"
    )
    print(
        f"Unknown F1 = {best_unknown['unknown_f1']:.4f}"
    )
    print(
        "Unknown P/R = "
        f"{best_unknown['unknown_precision']:.4f} / "
        f"{best_unknown['unknown_recall']:.4f}"
    )
    print(
        "Known rejection rate = "
        f"{best_unknown['known_rejection_rate']:.4f}"
    )

    print("\n输出文件：")
    print(f"- {PER_IMAGE_CSV}")
    print(f"- {GROUP_SUMMARY_CSV}")
    print(f"- {THRESHOLD_SCAN_CSV}")
    print(f"- {SUMMARY_TXT}")


if __name__ == "__main__":
    main()

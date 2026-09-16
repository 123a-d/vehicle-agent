"""
阶段四 · Unknown 专项实验 4F
文件名：analyze_feature_prototype_rejection.py

目标
====
不再继续修改 unknown 训练数据，而是换一种拒识思路：

    ResNet18 图片
        ↓
    512维视觉特征
        ↓
    与 50 个 known 类“特征原型（prototype）”比较
        ↓
    如果离所有 known 类都不够近
        ↓
    判为 unknown

当前基础模型仍然使用实验 4A 的最佳模型：
    outputs/unknown_weighted/best_macro_f1.pth

实验思路
========
1. 只使用训练集中的 50 个 known 类图片；
2. 每张图提取 ResNet18 的 512 维特征；
3. 对每个 known 类求平均特征，得到 50 个 prototype；
4. 对验证集每张图片提取 512 维特征；
5. 计算它和 50 个 prototype 的 cosine similarity；
6. 扫描不同 similarity threshold；
7. 如果：
       原模型已经预测 unknown
   或：
       max prototype similarity < threshold
   则最终输出 unknown。

为什么这样做
============
我们已经发现：
    很多真实 unknown 即使被误判成 known，
    Softmax confidence 仍然能高到 0.7、0.9 甚至接近 1。

所以仅靠 softmax confidence threshold 不够。

这次我们直接问：
    “这张图的视觉特征，真的和某个 known 类训练样本群足够接近吗？”

输出
====
outputs/feature_prototype_rejection/
    ├── class_prototypes.pt
    ├── per_image_feature_analysis.csv
    ├── similarity_group_summary.csv
    ├── similarity_threshold_scan.csv
    └── analysis_summary.txt

运行
====
    python analyze_feature_prototype_rejection.py

注意
====
本脚本：
    - 不训练模型
    - 不修改模型参数
    - 不修改 dataset_51
    - threshold 只在 val 上分析

如果课程最后有独立 test，
最终阈值应固定后再去 test 上做一次客观评估。
"""

from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset_51"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "dataset_51"
    / "val"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unknown_weighted"
    / "best_macro_f1.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "feature_prototype_rejection"
)

PROTOTYPE_PATH = (
    OUTPUT_DIR
    / "class_prototypes.pt"
)

PER_IMAGE_CSV = (
    OUTPUT_DIR
    / "per_image_feature_analysis.csv"
)

GROUP_SUMMARY_CSV = (
    OUTPUT_DIR
    / "similarity_group_summary.csv"
)

THRESHOLD_SCAN_CSV = (
    OUTPUT_DIR
    / "similarity_threshold_scan.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "analysis_summary.txt"
)


# ============================================================
# 2. 固定参数
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

NUM_CLASSES = 51
UNKNOWN_INDEX = 50
KNOWN_CLASS_COUNT = 50

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]

# Cosine similarity 的理论范围是 [-1, 1]。
# 深度视觉特征通常集中在较高区间，
# 这里先扫描 0.00 ~ 0.99，步长 0.01。
SIMILARITY_THRESHOLDS = [
    round(float(x), 2)
    for x in np.arange(
        0.00,
        1.00,
        0.01,
    )
]


# ============================================================
# 3. 数据预处理
# ============================================================

def build_transform():
    """
    保持与之前验证流程完全相同的预处理。
    """

    return transforms.Compose(
        [
            transforms.Resize(
                256
            ),
            transforms.CenterCrop(
                IMAGE_SIZE
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_datasets():
    """
    读取原始 dataset_51。

    prototype 只使用 train 中 0~49 的 known 类。
    train/unknown 不参与 prototype 构建。
    """

    transform = build_transform()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=transform,
    )

    if (
        train_dataset.class_to_idx
        != val_dataset.class_to_idx
    ):
        raise ValueError(
            "train 和 val 的类别映射不一致。"
        )

    if (
        len(train_dataset.classes)
        != NUM_CLASSES
    ):
        raise ValueError(
            f"应有 {NUM_CLASSES} 类，"
            f"实际为 {len(train_dataset.classes)}。"
        )

    if (
        train_dataset.class_to_idx.get(
            "unknown"
        )
        != UNKNOWN_INDEX
    ):
        raise ValueError(
            "unknown 索引不是 50。"
        )

    return (
        train_dataset,
        val_dataset,
    )


# ============================================================
# 4. 加载当前最佳模型
# ============================================================

def load_model(
    device: torch.device,
):
    """
    加载实验 4A 的 best_macro_f1.pth。

    这里需要两部分：
        1. ResNet18 backbone 提取 512维特征；
        2. 原来的 fc 继续输出 51 类 logits。
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        checkpoint[
            "num_classes"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# 5. 提取 ResNet18 的 512维特征
# ============================================================

def extract_features(
    model,
    images,
):
    """
    ResNet18 原始流程：

        conv1
        bn1
        relu
        maxpool
        layer1
        layer2
        layer3
        layer4
        avgpool
        flatten
        fc

    我们在 fc 之前截取：

        flatten 后的 512维向量

    这就是本实验使用的视觉 embedding / feature。
    """

    x = model.conv1(
        images
    )

    x = model.bn1(
        x
    )

    x = model.relu(
        x
    )

    x = model.maxpool(
        x
    )

    x = model.layer1(
        x
    )

    x = model.layer2(
        x
    )

    x = model.layer3(
        x
    )

    x = model.layer4(
        x
    )

    x = model.avgpool(
        x
    )

    features = torch.flatten(
        x,
        1,
    )

    return features


# ============================================================
# 6. 构建 50 个 known 类 Prototype
# ============================================================

def build_class_prototypes(
    model,
    train_dataset,
    device,
):
    """
    对训练集中的 50 个 known 类构建 prototype。

    做法：
        每张 known 图片
        -> 提取 512维 feature
        -> L2 normalize
        -> 同类累加求平均
        -> 再把类平均向量 normalize

    最终：
        prototypes.shape = [50, 512]

    第 i 行就是 known class index=i 的类别原型。
    """

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    feature_sums = torch.zeros(
        KNOWN_CLASS_COUNT,
        512,
        dtype=torch.float32,
        device=device,
    )

    class_counts = torch.zeros(
        KNOWN_CLASS_COUNT,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():

        for images, labels in loader:

            # prototype 只使用 0~49 known 类。
            known_mask = (
                labels
                != UNKNOWN_INDEX
            )

            if not known_mask.any():
                continue

            images = images[
                known_mask
            ].to(
                device
            )

            labels = labels[
                known_mask
            ].to(
                device
            )

            features = extract_features(
                model,
                images,
            )

            # 先把每张图的 feature 做 L2 normalize。
            features = F.normalize(
                features,
                p=2,
                dim=1,
            )

            for class_index in range(
                KNOWN_CLASS_COUNT
            ):

                class_mask = (
                    labels
                    == class_index
                )

                if not class_mask.any():
                    continue

                feature_sums[
                    class_index
                ] += features[
                    class_mask
                ].sum(
                    dim=0
                )

                class_counts[
                    class_index
                ] += int(
                    class_mask.sum().item()
                )

    if (
        class_counts
        == 0
    ).any():

        missing = torch.where(
            class_counts
            == 0
        )[0].tolist()

        raise ValueError(
            "某些 known 类没有训练样本："
            f"{missing}"
        )

    # 对每个类求平均。
    prototypes = (
        feature_sums
        / class_counts.unsqueeze(
            1
        )
    )

    # prototype 再做一次 L2 normalize。
    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=1,
    )

    return (
        prototypes,
        class_counts,
    )


# ============================================================
# 7. 保存 Prototype
# ============================================================

def save_prototypes(
    prototypes,
    class_counts,
    train_dataset,
):
    """
    保存 prototype，方便后续真正封装推理 Tool 时直接加载。

    保存内容：
        prototypes: [50, 512]
        class_counts
        idx_to_class
        checkpoint
    """

    idx_to_class = {
        index: class_name
        for class_name, index
        in train_dataset
        .class_to_idx
        .items()
        if index < UNKNOWN_INDEX
    }

    torch.save(
        {
            "prototypes": (
                prototypes.cpu()
            ),

            "class_counts": (
                class_counts.cpu()
            ),

            "idx_to_class": (
                idx_to_class
            ),

            "feature_dim": 512,

            "source_checkpoint": str(
                CHECKPOINT_PATH
            ),
        },
        PROTOTYPE_PATH,
    )


# ============================================================
# 8. 对验证集做逐图片特征分析
# ============================================================

def collect_val_results(
    model,
    prototypes,
    val_dataset,
    device,
):
    """
    对每张验证图片同时得到：

    A. 原分类器结果
        raw_pred_index
        softmax confidence

    B. Prototype similarity
        与 50 个 known prototype 的 cosine similarity

    重点：
        max_known_similarity
        = 与最接近 known prototype 的相似度

    如果它很低：
        说明这张图不像任何一个 known 类。
    """

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    rows = []

    global_index = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(
                device
            )

            # ------------------------------------------------
            # 提取 512维 feature
            # ------------------------------------------------
            features = extract_features(
                model,
                images,
            )

            normalized_features = F.normalize(
                features,
                p=2,
                dim=1,
            )

            # ------------------------------------------------
            # 原 51 类分类器 logits / softmax
            # ------------------------------------------------
            logits = model.fc(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            (
                raw_confidences,
                raw_pred_indices,
            ) = probabilities.max(
                dim=1
            )

            # ------------------------------------------------
            # feature 与 50个 prototype cosine similarity
            #
            # 因为双方都已 L2 normalize，
            # 矩阵乘法就等价于 cosine similarity。
            # ------------------------------------------------
            similarity_matrix = (
                normalized_features
                @ prototypes.T
            )

            (
                max_similarities,
                nearest_proto_indices,
            ) = similarity_matrix.max(
                dim=1
            )

            # top2 prototype：
            top2_similarities, top2_proto_indices = (
                similarity_matrix.topk(
                    k=2,
                    dim=1,
                )
            )

            batch_size = (
                labels.size(0)
            )

            for i in range(
                batch_size
            ):

                true_index = int(
                    labels[
                        i
                    ].item()
                )

                raw_pred_index = int(
                    raw_pred_indices[
                        i
                    ].item()
                )

                raw_confidence = float(
                    raw_confidences[
                        i
                    ].item()
                )

                nearest_proto_index = int(
                    nearest_proto_indices[
                        i
                    ].item()
                )

                max_similarity = float(
                    max_similarities[
                        i
                    ].item()
                )

                second_similarity = float(
                    top2_similarities[
                        i, 1
                    ].item()
                )

                similarity_margin = (
                    max_similarity
                    - second_similarity
                )

                image_path = (
                    val_dataset.samples[
                        global_index
                    ][0]
                )

                rows.append(
                    {
                        "image_path": (
                            image_path
                        ),

                        "true_index": (
                            true_index
                        ),

                        "true_label": (
                            val_dataset.classes[
                                true_index
                            ]
                        ),

                        "raw_pred_index": (
                            raw_pred_index
                        ),

                        "raw_pred_label": (
                            val_dataset.classes[
                                raw_pred_index
                            ]
                        ),

                        "raw_confidence": (
                            raw_confidence
                        ),

                        "nearest_proto_index": (
                            nearest_proto_index
                        ),

                        "nearest_proto_label": (
                            val_dataset.classes[
                                nearest_proto_index
                            ]
                        ),

                        "max_known_similarity": (
                            max_similarity
                        ),

                        "second_known_similarity": (
                            second_similarity
                        ),

                        "similarity_margin": (
                            similarity_margin
                        ),

                        "raw_is_correct": int(
                            raw_pred_index
                            == true_index
                        ),

                        "is_true_unknown": int(
                            true_index
                            == UNKNOWN_INDEX
                        ),

                        "raw_pred_unknown": int(
                            raw_pred_index
                            == UNKNOWN_INDEX
                        ),
                    }
                )

                global_index += 1

    return rows


# ============================================================
# 9. CSV 保存
# ============================================================

def save_csv(
    path,
    rows,
):
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
            fieldnames=rows[
                0
            ].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 10. 分组统计 similarity
# ============================================================

def describe(
    values,
):
    """统计一组数值的分布。"""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        len(values)
        == 0
    ):
        return {
            "count": 0,
            "mean": float(
                "nan"
            ),
            "median": float(
                "nan"
            ),
            "q25": float(
                "nan"
            ),
            "q75": float(
                "nan"
            ),
            "min": float(
                "nan"
            ),
            "max": float(
                "nan"
            ),
        }

    return {
        "count": int(
            len(values)
        ),

        "mean": float(
            np.mean(
                values
            )
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "q25": float(
            np.quantile(
                values,
                0.25,
            )
        ),

        "q75": float(
            np.quantile(
                values,
                0.75,
            )
        ),

        "min": float(
            np.min(
                values
            )
        ),

        "max": float(
            np.max(
                values
            )
        ),
    }


def build_group_summary(
    rows,
):
    """
    最重要的比较：

    known_correct：
        真实 known，且分类器预测正确。

    true_unknown_missed：
        真实 unknown，但分类器误判为 known。

    如果：
        known_correct similarity 明显高
        unknown_missed similarity 明显低

    说明 prototype distance 有拒识价值。
    """

    groups = {
        "known_correct": [
            row[
                "max_known_similarity"
            ]
            for row in rows
            if (
                row[
                    "is_true_unknown"
                ]
                == 0
                and row[
                    "raw_is_correct"
                ]
                == 1
            )
        ],

        "known_wrong": [
            row[
                "max_known_similarity"
            ]
            for row in rows
            if (
                row[
                    "is_true_unknown"
                ]
                == 0
                and row[
                    "raw_is_correct"
                ]
                == 0
            )
        ],

        "true_unknown_all": [
            row[
                "max_known_similarity"
            ]
            for row in rows
            if (
                row[
                    "is_true_unknown"
                ]
                == 1
            )
        ],

        "true_unknown_correct": [
            row[
                "max_known_similarity"
            ]
            for row in rows
            if (
                row[
                    "is_true_unknown"
                ]
                == 1
                and row[
                    "raw_is_correct"
                ]
                == 1
            )
        ],

        "true_unknown_missed": [
            row[
                "max_known_similarity"
            ]
            for row in rows
            if (
                row[
                    "is_true_unknown"
                ]
                == 1
                and row[
                    "raw_is_correct"
                ]
                == 0
            )
        ],
    }

    summary_rows = []

    for (
        group_name,
        values,
    ) in groups.items():

        summary_rows.append(
            {
                "group": (
                    group_name
                ),
                **describe(
                    values
                ),
            }
        )

    return (
        summary_rows
    )


# ============================================================
# 11. 指标计算
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """统一计算 overall / known / unknown 指标。"""

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(
            range(
                NUM_CLASSES
            )
        ),
        average="macro",
        zero_division=0,
    )

    known_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(
            range(
                UNKNOWN_INDEX
            )
        ),
        average="macro",
        zero_division=0,
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[
            UNKNOWN_INDEX
        ],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "known_macro_f1": float(
            known_macro_f1
        ),

        "unknown_precision": float(
            precision[0]
        ),

        "unknown_recall": float(
            recall[0]
        ),

        "unknown_f1": float(
            f1[0]
        ),

        "unknown_support": int(
            support[0]
        ),
    }


# ============================================================
# 12. 扫描 prototype similarity threshold
# ============================================================

def scan_similarity_thresholds(
    rows,
):
    """
    最终决策逻辑：

    1. 如果原模型已经预测 unknown：
           final = unknown

    2. 如果原模型预测 known，
       但 max_known_similarity < threshold：
           final = unknown

    3. 否则：
           final = 原模型预测

    注意：
        这里 prototype 只负责“拒识”，
        不负责替代原分类器重新决定具体 known 类。

    这样更容易隔离变量：
        known 分类仍交给已经表现不错的实验 4A 分类器；
        prototype 只负责发现“不像任何 known”的样本。
    """

    y_true = [
        row[
            "true_index"
        ]
        for row in rows
    ]

    true_known_count = sum(
        1
        for label in y_true
        if (
            label
            != UNKNOWN_INDEX
        )
    )

    scan_rows = []

    for threshold in (
        SIMILARITY_THRESHOLDS
    ):

        y_pred = []

        prototype_rejected_count = 0

        for row in rows:

            raw_pred = (
                row[
                    "raw_pred_index"
                ]
            )

            similarity = (
                row[
                    "max_known_similarity"
                ]
            )

            if (
                raw_pred
                == UNKNOWN_INDEX
            ):

                final_pred = (
                    UNKNOWN_INDEX
                )

            elif (
                similarity
                < threshold
            ):

                final_pred = (
                    UNKNOWN_INDEX
                )

                prototype_rejected_count += 1

            else:

                final_pred = (
                    raw_pred
                )

            y_pred.append(
                final_pred
            )

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        # true known 最终被拒绝为 unknown 的数量。
        known_rejected_count = sum(
            1
            for (
                true_label,
                pred_label,
            ) in zip(
                y_true,
                y_pred,
            )
            if (
                true_label
                != UNKNOWN_INDEX
                and pred_label
                == UNKNOWN_INDEX
            )
        )

        known_rejection_rate = (
            known_rejected_count
            / true_known_count
        )

        scan_rows.append(
            {
                "threshold": (
                    threshold
                ),

                **metrics,

                "prototype_rejected_count": (
                    prototype_rejected_count
                ),

                "known_rejected_count": (
                    known_rejected_count
                ),

                "known_rejection_rate": (
                    known_rejection_rate
                ),
            }
        )

    return (
        scan_rows
    )


# ============================================================
# 13. 辅助查找
# ============================================================

def find_group(
    summary_rows,
    group_name,
):
    """从分组统计里找到指定 group。"""

    for row in (
        summary_rows
    ):
        if (
            row[
                "group"
            ]
            == group_name
        ):
            return row

    raise KeyError(
        group_name
    )


# ============================================================
# 14. 生成文字总结
# ============================================================

def write_summary(
    checkpoint,
    group_summary,
    threshold_rows,
):
    """
    自动挑出几个值得看的 operating point：

    1. 原始模型
    2. Macro-F1 最佳 threshold
    3. Unknown F1 最佳 threshold
    4. known rejection <= 5% 条件下 Unknown F1 最佳 threshold

    第4个比较符合工程实际：
        我们不能为了抓 unknown，
        把大量 known 也拒掉。
    """

    # threshold=0.00 基本不会因为 prototype 被拒绝，
    # 因此作为原始模型参考。
    baseline = threshold_rows[
        0
    ]

    best_macro = max(
        threshold_rows,
        key=lambda row: (
            row[
                "macro_f1"
            ]
        ),
    )

    best_unknown = max(
        threshold_rows,
        key=lambda row: (
            row[
                "unknown_f1"
            ]
        ),
    )

    feasible_rows = [
        row
        for row in threshold_rows
        if (
            row[
                "known_rejection_rate"
            ]
            <= 0.05
        )
    ]

    if feasible_rows:
        best_under_5pct = max(
            feasible_rows,
            key=lambda row: (
                row[
                    "unknown_f1"
                ]
            ),
        )
    else:
        best_under_5pct = None

    known_correct = find_group(
        group_summary,
        "known_correct",
    )

    unknown_missed = find_group(
        group_summary,
        "true_unknown_missed",
    )

    lines = [
        "阶段四 · Feature Prototype Rejection 分析",
        "=" * 70,
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        (
            "Checkpoint Epoch: "
            f"{checkpoint.get('epoch')}"
        ),
        "",
        "1. Similarity 分布",
        (
            "Known 预测正确："
            f"mean={known_correct['mean']:.4f}, "
            f"median={known_correct['median']:.4f}, "
            f"q25={known_correct['q25']:.4f}"
        ),
        (
            "真实 Unknown 但误判 Known："
            f"mean={unknown_missed['mean']:.4f}, "
            f"median={unknown_missed['median']:.4f}, "
            f"q75={unknown_missed['q75']:.4f}"
        ),
        "",
        "2. 原始模型（prototype threshold=0.00）",
        (
            f"Macro-F1={baseline['macro_f1']:.4f}"
        ),
        (
            f"Known-F1={baseline['known_macro_f1']:.4f}"
        ),
        (
            f"Unknown F1={baseline['unknown_f1']:.4f}"
        ),
        (
            "Unknown P/R="
            f"{baseline['unknown_precision']:.4f}/"
            f"{baseline['unknown_recall']:.4f}"
        ),
        "",
        "3. Macro-F1 最佳 threshold",
        (
            f"threshold={best_macro['threshold']:.2f}"
        ),
        (
            f"Macro-F1={best_macro['macro_f1']:.4f}"
        ),
        (
            f"Known-F1={best_macro['known_macro_f1']:.4f}"
        ),
        (
            f"Unknown F1={best_macro['unknown_f1']:.4f}"
        ),
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
        "4. Unknown F1 最佳 threshold",
        (
            f"threshold={best_unknown['threshold']:.2f}"
        ),
        (
            f"Macro-F1={best_unknown['macro_f1']:.4f}"
        ),
        (
            f"Known-F1={best_unknown['known_macro_f1']:.4f}"
        ),
        (
            f"Unknown F1={best_unknown['unknown_f1']:.4f}"
        ),
        (
            "Unknown P/R="
            f"{best_unknown['unknown_precision']:.4f}/"
            f"{best_unknown['unknown_recall']:.4f}"
        ),
        (
            "Known rejection rate="
            f"{best_unknown['known_rejection_rate']:.4f}"
        ),
    ]

    if (
        best_under_5pct
        is not None
    ):
        lines.extend(
            [
                "",
                "5. Known rejection <= 5% 时 Unknown F1 最佳 threshold",
                (
                    f"threshold={best_under_5pct['threshold']:.2f}"
                ),
                (
                    f"Macro-F1={best_under_5pct['macro_f1']:.4f}"
                ),
                (
                    f"Known-F1={best_under_5pct['known_macro_f1']:.4f}"
                ),
                (
                    f"Unknown F1={best_under_5pct['unknown_f1']:.4f}"
                ),
                (
                    "Unknown P/R="
                    f"{best_under_5pct['unknown_precision']:.4f}/"
                    f"{best_under_5pct['unknown_recall']:.4f}"
                ),
                (
                    "Known rejection rate="
                    f"{best_under_5pct['known_rejection_rate']:.4f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "注意：",
            "阈值是在 val 上选择的。",
            "如果有独立 test，最终应固定阈值后只评估一次 test。",
        ]
    )

    SUMMARY_TXT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return (
        baseline,
        best_macro,
        best_unknown,
        best_under_5pct,
        known_correct,
        unknown_missed,
    )


# ============================================================
# 15. 主程序
# ============================================================

def main():
    print("=" * 86)

    print(
        "阶段四 · Unknown实验4F："
        "512维 Prototype + Cosine Similarity Reject"
    )

    print("=" * 86)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"设备：{device}"
    )

    print(
        f"模型：{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
    ) = build_datasets()

    print(
        f"\n训练集总图片：{len(train_dataset)}"
    )

    print(
        f"验证集总图片：{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    (
        model,
        checkpoint,
    ) = load_model(
        device
    )

    # --------------------------------------------------------
    # 构建 50 类 prototype
    # --------------------------------------------------------

    print(
        "\n正在用 train known 构建 50 个 class prototypes..."
    )

    (
        prototypes,
        class_counts,
    ) = build_class_prototypes(
        model=model,
        train_dataset=train_dataset,
        device=device,
    )

    print(
        "Prototype shape："
        f"{tuple(prototypes.shape)}"
    )

    print(
        "每类训练样本数量范围："
        f"{int(class_counts.min().item())}"
        " ~ "
        f"{int(class_counts.max().item())}"
    )

    save_prototypes(
        prototypes=prototypes,
        class_counts=class_counts,
        train_dataset=train_dataset,
    )

    # --------------------------------------------------------
    # 验证集逐图片分析
    # --------------------------------------------------------

    print(
        "\n正在分析验证集 feature similarity..."
    )

    rows = collect_val_results(
        model=model,
        prototypes=prototypes,
        val_dataset=val_dataset,
        device=device,
    )

    save_csv(
        PER_IMAGE_CSV,
        rows,
    )

    # --------------------------------------------------------
    # Similarity 分组统计
    # --------------------------------------------------------

    group_summary = (
        build_group_summary(
            rows
        )
    )

    save_csv(
        GROUP_SUMMARY_CSV,
        group_summary,
    )

    # --------------------------------------------------------
    # 扫描 threshold
    # --------------------------------------------------------

    threshold_rows = (
        scan_similarity_thresholds(
            rows
        )
    )

    save_csv(
        THRESHOLD_SCAN_CSV,
        threshold_rows,
    )

    # --------------------------------------------------------
    # 自动总结
    # --------------------------------------------------------

    (
        baseline,
        best_macro,
        best_unknown,
        best_under_5pct,
        known_correct,
        unknown_missed,
    ) = write_summary(
        checkpoint=checkpoint,
        group_summary=group_summary,
        threshold_rows=threshold_rows,
    )

    # --------------------------------------------------------
    # 终端重点输出
    # --------------------------------------------------------

    print(
        "\n" + "-" * 86
    )

    print(
        "Similarity 分布重点："
    )

    print(
        "Known预测正确："
        f"mean={known_correct['mean']:.4f}, "
        f"median={known_correct['median']:.4f}, "
        f"q25={known_correct['q25']:.4f}"
    )

    print(
        "真实Unknown但误判Known："
        f"mean={unknown_missed['mean']:.4f}, "
        f"median={unknown_missed['median']:.4f}, "
        f"q75={unknown_missed['q75']:.4f}"
    )

    print(
        "\n原始模型："
    )

    print(
        f"Macro-F1 = {baseline['macro_f1']:.4f}"
    )

    print(
        f"Known-F1 = {baseline['known_macro_f1']:.4f}"
    )

    print(
        f"Unknown F1 = {baseline['unknown_f1']:.4f}"
    )

    print(
        "\nMacro-F1 最佳 similarity threshold："
    )

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

    print(
        "\nUnknown F1 最佳 similarity threshold："
    )

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

    if (
        best_under_5pct
        is not None
    ):

        print(
            "\nKnown rejection <=5% 时的最佳方案："
        )

        print(
            f"threshold = {best_under_5pct['threshold']:.2f}"
        )

        print(
            f"Macro-F1 = {best_under_5pct['macro_f1']:.4f}"
        )

        print(
            f"Known-F1 = {best_under_5pct['known_macro_f1']:.4f}"
        )

        print(
            f"Unknown F1 = {best_under_5pct['unknown_f1']:.4f}"
        )

        print(
            "Unknown P/R = "
            f"{best_under_5pct['unknown_precision']:.4f} / "
            f"{best_under_5pct['unknown_recall']:.4f}"
        )

        print(
            "Known rejection rate = "
            f"{best_under_5pct['known_rejection_rate']:.4f}"
        )

    print(
        "\n输出文件："
    )

    print(
        f"- {PROTOTYPE_PATH}"
    )

    print(
        f"- {PER_IMAGE_CSV}"
    )

    print(
        f"- {GROUP_SUMMARY_CSV}"
    )

    print(
        f"- {THRESHOLD_SCAN_CSV}"
    )

    print(
        f"- {SUMMARY_TXT}"
    )


if __name__ == "__main__":
    main()

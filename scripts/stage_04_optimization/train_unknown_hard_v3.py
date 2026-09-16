"""
阶段四 · Unknown 专项实验 4E：Hard Negative v3 训练
文件名：train_unknown_hard_v3.py

实验目标
========
比较实验 4A 与实验 4E：

实验 4A：
    unknown train = 旧方案
    20 个来源 × 10 张 = 200 张
    unknown weight = 2.0
    最佳 Macro-F1 = 0.8417
    最佳 Unknown F1 = 0.2056

实验 4E：
    unknown train = Hard Negative v3
    40 个 Hard Negative 来源 × 4 张 = 160 张
    20 个旧普通 unknown 来源 × 2 张 = 40 张
    总计仍然 = 200 张

本实验其余训练条件与实验 4A 保持一致：
    - 从 baseline_best.pth 开始
    - ResNet18
    - 只训练 layer4 + fc
    - Learning Rate = 1e-4
    - Epoch = 5
    - Batch Size = 32
    - 不加入随机数据增强
    - unknown CrossEntropyLoss 权重 = 2.0

因此，这次真正想回答的问题是：

    “把随机/普通 unknown 换成模型最容易高置信度误认的
     Hard Negative，能不能显著改善对 unseen unknown 的拒识能力？”

重要：
    unknown val 的 12 个来源车型没有参与 Hard Negative Mining，
    也不在 v3 train 中，因此验证仍然是在测试 unseen unknown。

运行
====
    python train_unknown_hard_v3.py

输出
====
outputs/unknown_hard_v3/
    ├── best_macro_f1.pth
    ├── best_unknown_f1.pth
    └── training_history.csv
"""

from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
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

# 原始 51 类训练 / 验证数据。
# train 中仍然保留旧 unknown，但本脚本会在内存中把它替换成 v3。
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

# 新构建好的 Hard Negative v3 unknown。
UNKNOWN_V3_DIR = (
    PROJECT_ROOT
    / "unknown_train_v3_hard"
    / "unknown"
)

UNKNOWN_V3_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_train_sources_v3_hard.txt"
)

UNKNOWN_VAL_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_val_sources.txt"
)

# 为了与实验 4A 做公平对比，
# 仍然从同一个 Baseline checkpoint 开始训练。
BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_resnet18"
    / "baseline_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unknown_hard_v3"
)

BEST_MACRO_PATH = (
    OUTPUT_DIR
    / "best_macro_f1.pth"
)

BEST_UNKNOWN_PATH = (
    OUTPUT_DIR
    / "best_unknown_f1.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)


# ============================================================
# 2. 实验参数
# ============================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_WORKERS = 0

LEARNING_RATE = 1e-4

NUM_CLASSES = 51
UNKNOWN_INDEX = 50

# 与实验 4A 保持一致。
UNKNOWN_WEIGHT = 2.0

# v3 的预期规模。
EXPECTED_V3_IMAGES = 200
EXPECTED_V3_SOURCES = 60

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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 3. 随机种子
# ============================================================

def set_random_seed(seed: int) -> None:
    """固定随机种子，使实验尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 4. 读取 unknown 来源列表
# ============================================================

def read_source_ids(path: Path):
    """
    读取每行一个车型 ID 的 txt 文件。
    """

    if not path.exists():
        raise FileNotFoundError(
            f"找不到来源列表：{path}"
        )

    ids = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    # 去重并保持顺序。
    return list(
        dict.fromkeys(ids)
    )


# ============================================================
# 5. 再次检查 train / val unknown 来源隔离
# ============================================================

def verify_unknown_source_isolation():
    """
    检查：

        v3 unknown train 来源
        ∩
        unknown val 来源

        必须为空集。

    这是防止数据泄漏的关键工程检查。
    """

    train_source_ids = read_source_ids(
        UNKNOWN_V3_SOURCES_FILE
    )

    val_source_ids = read_source_ids(
        UNKNOWN_VAL_SOURCES_FILE
    )

    if (
        len(train_source_ids)
        != EXPECTED_V3_SOURCES
    ):
        raise ValueError(
            "v3 unknown 来源应为 60 个，"
            f"实际为 {len(train_source_ids)}。"
        )

    overlap = (
        set(train_source_ids)
        & set(val_source_ids)
    )

    if overlap:
        raise ValueError(
            "发现 v3 unknown train / val 来源泄漏："
            f"{sorted(overlap)}"
        )

    return (
        train_source_ids,
        val_source_ids,
        overlap,
    )


# ============================================================
# 6. 构建 Dataset
# ============================================================

def build_datasets():
    """
    构建训练集和验证集。

    与实验 4C 一样，这里不复制整个 50 类 known 数据。

    做法：
        1. 用 ImageFolder 读取 dataset_51/train；
        2. 保留其中 0~49 的 known 样本；
        3. 删除旧的 unknown 样本；
        4. 加入 v3 的 200 张 Hard Negative unknown；
        5. 验证集完全不变。

    ImageFolder 的核心数据结构之一就是：
        samples = [(图片路径, 类别索引), ...]

    所以我们可以在内存里替换 unknown，
    而不用破坏磁盘上的 dataset_51。
    """

    transform = transforms.Compose(
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

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=transform,
    )

    # --------------------------------------------------------
    # 类别映射安全检查
    # --------------------------------------------------------
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
            f"类别数应为 {NUM_CLASSES}，"
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

    # --------------------------------------------------------
    # 保留 50 个 known 类样本
    # --------------------------------------------------------
    known_samples = [
        (
            path,
            target,
        )
        for path, target
        in train_dataset.samples
        if target != UNKNOWN_INDEX
    ]

    old_unknown_count = (
        len(
            train_dataset.samples
        )
        - len(
            known_samples
        )
    )

    # 旧 dataset_51/train/unknown 应该是 200 张。
    if (
        old_unknown_count
        != 200
    ):
        raise ValueError(
            "原 train unknown 应有 200 张，"
            f"实际为 {old_unknown_count}。"
        )

    # --------------------------------------------------------
    # 读取 v3 的 200 张 unknown
    # --------------------------------------------------------
    if not UNKNOWN_V3_DIR.exists():
        raise FileNotFoundError(
            f"找不到 v3 unknown：{UNKNOWN_V3_DIR}"
        )

    v3_paths = sorted(
        path
        for path
        in UNKNOWN_V3_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if (
        len(v3_paths)
        != EXPECTED_V3_IMAGES
    ):
        raise ValueError(
            "v3 unknown 应为 200 张，"
            f"实际为 {len(v3_paths)}。"
        )

    v3_unknown_samples = [
        (
            str(path),
            UNKNOWN_INDEX,
        )
        for path in v3_paths
    ]

    # --------------------------------------------------------
    # 用 v3 替换旧 unknown
    # --------------------------------------------------------
    train_dataset.samples = (
        known_samples
        + v3_unknown_samples
    )

    # ImageFolder 中 imgs 通常对应 samples；
    # targets 保存所有标签。
    # 为保持内部状态一致，一并更新。
    train_dataset.imgs = (
        train_dataset.samples
    )

    train_dataset.targets = [
        target
        for _, target
        in train_dataset.samples
    ]

    return (
        train_dataset,
        val_dataset,
        old_unknown_count,
        len(
            v3_unknown_samples
        ),
    )


# ============================================================
# 7. DataLoader
# ============================================================

def build_dataloaders(
    train_dataset,
    val_dataset,
):
    """构建训练和验证 DataLoader。"""

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# 8. 加载 Baseline 并解冻 layer4 + fc
# ============================================================

def build_model(
    device: torch.device,
):
    """
    与实验 4A 完全一致：

        baseline_best.pth
                ↓
        冻结所有参数
                ↓
        解冻 layer4
                ↓
        解冻 fc
    """

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "找不到 Baseline checkpoint："
            f"{BASELINE_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
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

    # 先冻结全部参数。
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 解冻 layer4。
    for parameter in (
        model.layer4.parameters()
    ):
        parameter.requires_grad = True

    # 解冻 fc。
    for parameter in (
        model.fc.parameters()
    ):
        parameter.requires_grad = True

    return (
        model.to(
            device
        ),
        checkpoint,
    )


# ============================================================
# 9. Fine-tuning 的 train/eval 状态
# ============================================================

def set_finetune_train_mode(
    model,
) -> None:
    """
    与实验 4A 保持一致。

    整体：
        model.eval()

    只让：
        layer4
        fc
    进入 train。

    layer4 里的 BatchNorm 继续保持 eval，
    减少小数据集下 BN 统计量漂移。
    """

    model.eval()

    model.layer4.train()
    model.fc.train()

    for module in (
        model.layer4.modules()
    ):
        if isinstance(
            module,
            nn.BatchNorm2d,
        ):
            module.eval()


# ============================================================
# 10. Weighted CrossEntropyLoss
# ============================================================

def build_weighted_loss(
    device: torch.device,
):
    """
    与实验 4A 完全一致：

        known 类 0~49：weight = 1.0
        unknown 50：weight = 2.0

    所以实验 4A 和 4E 的 Loss 条件相同。
    """

    class_weights = torch.ones(
        NUM_CLASSES,
        dtype=torch.float32,
        device=device,
    )

    class_weights[
        UNKNOWN_INDEX
    ] = UNKNOWN_WEIGHT

    return nn.CrossEntropyLoss(
        weight=class_weights
    )


# ============================================================
# 11. 训练一个 Epoch
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """完成一个 Epoch 的训练。"""

    set_finetune_train_mode(
        model
    )

    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        # ------------------------------
        # 训练最核心五步
        # ------------------------------

        # 1. 清空上一个 Batch 的梯度。
        optimizer.zero_grad()

        # 2. Forward。
        outputs = model(
            images
        )

        # 3. 计算加权交叉熵 Loss。
        loss = criterion(
            outputs,
            labels,
        )

        # 4. Backward。
        loss.backward()

        # 5. 更新 layer4 + fc。
        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions
            == labels
        ).sum().item()

    train_loss = (
        running_loss
        / total_samples
    )

    train_accuracy = (
        correct
        / total_samples
    )

    return (
        train_loss,
        train_accuracy,
    )


# ============================================================
# 12. 验证
# ============================================================

def evaluate(
    model,
    val_loader,
    criterion,
    device,
):
    """
    在完全不变的 val 上计算：

        Val Loss
        Val Accuracy
        51 类 Macro-F1
        50 known 类 Macro-F1
        Unknown Precision
        Unknown Recall
        Unknown F1
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.inference_mode():

        for images, labels in val_loader:
            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            y_true.extend(
                labels
                .cpu()
                .tolist()
            )

            y_pred.extend(
                predictions
                .cpu()
                .tolist()
            )

    val_loss = (
        running_loss
        / total_samples
    )

    val_accuracy = accuracy_score(
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
        "val_loss": float(
            val_loss
        ),

        "val_accuracy": float(
            val_accuracy
        ),

        "val_macro_f1": float(
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
# 13. 保存训练日志
# ============================================================

def save_history(
    history,
) -> None:
    """保存每个 Epoch 的指标。"""

    if not history:
        return

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[
                0
            ].keys(),
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# 14. 保存 checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    class_to_idx,
    epoch,
    metrics,
    selection_metric,
):
    """
    保存 checkpoint。

    分别保存：
        1. Macro-F1 最佳模型
        2. Unknown F1 最佳模型
    """

    checkpoint = {
        "model_name": (
            "resnet18"
        ),

        "experiment": (
            "unknown_hard_v3"
        ),

        "selection_metric": (
            selection_metric
        ),

        "epoch": epoch,

        "num_classes": (
            NUM_CLASSES
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "class_to_idx": (
            class_to_idx
        ),

        "metrics": (
            metrics
        ),

        "config": {
            "random_seed": (
                RANDOM_SEED
            ),

            "image_size": (
                IMAGE_SIZE
            ),

            "batch_size": (
                BATCH_SIZE
            ),

            "epochs": (
                EPOCHS
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "trainable_modules": [
                "layer4",
                "fc",
            ],

            "unknown_weight": (
                UNKNOWN_WEIGHT
            ),

            "unknown_train_version": (
                "v3_hard"
            ),

            "unknown_train_sources": (
                EXPECTED_V3_SOURCES
            ),

            "unknown_train_images": (
                EXPECTED_V3_IMAGES
            ),

            "hard_negative_sources": (
                40
            ),

            "hard_negative_images": (
                160
            ),

            "regular_unknown_sources": (
                20
            ),

            "regular_unknown_images": (
                40
            ),

            "data_augmentation": (
                False
            ),

            "starting_checkpoint": str(
                BASELINE_CHECKPOINT
            ),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# 15. 主程序
# ============================================================

def main():
    print("=" * 84)

    print(
        "阶段四 · Unknown实验4E："
        "Hard Negative v3 + weight=2"
    )

    print("=" * 84)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(
        RANDOM_SEED
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
        f"Epochs：{EPOCHS}"
    )

    print(
        "Learning Rate："
        f"{LEARNING_RATE}"
    )

    print(
        "Unknown Weight："
        f"{UNKNOWN_WEIGHT}"
    )

    # --------------------------------------------------------
    # 数据隔离检查
    # --------------------------------------------------------

    (
        train_source_ids,
        val_source_ids,
        overlap,
    ) = verify_unknown_source_isolation()

    print(
        "\nv3 unknown train 来源数："
        f"{len(train_source_ids)}"
    )

    print(
        "unknown val 来源数："
        f"{len(val_source_ids)}"
    )

    print(
        "来源交集："
        f"{overlap}"
    )

    # --------------------------------------------------------
    # 构建替换了 v3 unknown 的训练集
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        old_unknown_count,
        new_unknown_count,
    ) = build_datasets()

    (
        train_loader,
        val_loader,
    ) = build_dataloaders(
        train_dataset,
        val_dataset,
    )

    print(
        "\n旧 unknown 图片数："
        f"{old_unknown_count}"
    )

    print(
        "v3 unknown 图片数："
        f"{new_unknown_count}"
    )

    print(
        "最终训练集："
        f"{len(train_dataset)} 张"
    )

    print(
        "验证集："
        f"{len(val_dataset)} 张"
    )

    print(
        "\n控制变量确认："
    )

    print(
        "- unknown 总图片仍为 200 张"
    )

    print(
        "- layer4 + fc 不变"
    )

    print(
        "- lr=1e-4 不变"
    )

    print(
        "- unknown weight=2 不变"
    )

    print(
        "- 5 Epoch 不变"
    )

    print(
        "- 主要变化：普通 unknown -> Hard Negative v3"
    )

    # --------------------------------------------------------
    # 模型、Loss、Optimizer
    # --------------------------------------------------------

    (
        model,
        baseline_checkpoint,
    ) = build_model(
        device
    )

    print(
        "\n起始 Baseline Macro-F1："
        f"{baseline_checkpoint['val_macro_f1']:.4f}"
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "可训练参数："
        f"{trainable_parameters:,} / "
        f"{total_parameters:,}"
    )

    criterion = (
        build_weighted_loss(
            device
        )
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: (
                parameter.requires_grad
            ),
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_macro_epoch = None

    best_unknown_f1 = -1.0
    best_unknown_epoch = None

    history = []

    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    print(
        "\n开始训练..."
    )

    print(
        "-" * 84
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        start_time = (
            time.perf_counter()
        )

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result = {
            "epoch": (
                epoch
            ),

            "train_loss": (
                train_loss
            ),

            "train_accuracy": (
                train_accuracy
            ),

            **metrics,

            "epoch_seconds": (
                elapsed
            ),
        }

        history.append(
            result
        )

        save_history(
            history
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Loss {metrics['val_loss']:.4f} | "
            f"Val Acc {metrics['val_accuracy']:.4f} | "
            f"Macro-F1 {metrics['val_macro_f1']:.4f} | "
            f"Known-F1 {metrics['known_macro_f1']:.4f} | "
            f"Unknown P {metrics['unknown_precision']:.4f} | "
            f"Unknown R {metrics['unknown_recall']:.4f} | "
            f"Unknown F1 {metrics['unknown_f1']:.4f} | "
            f"Time {elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # 保存总体 Macro-F1 最佳模型
        # ----------------------------------------------------

        if (
            metrics[
                "val_macro_f1"
            ]
            > best_macro_f1
        ):

            best_macro_f1 = (
                metrics[
                    "val_macro_f1"
                ]
            )

            best_macro_epoch = (
                epoch
            )

            save_checkpoint(
                path=BEST_MACRO_PATH,
                model=model,
                optimizer=optimizer,
                class_to_idx=(
                    train_dataset
                    .class_to_idx
                ),
                epoch=epoch,
                metrics=result,
                selection_metric=(
                    "val_macro_f1"
                ),
            )

            print(
                "  -> 新的最佳 Macro-F1 模型，已保存。"
            )

        # ----------------------------------------------------
        # 保存 Unknown F1 最佳模型
        # ----------------------------------------------------

        if (
            metrics[
                "unknown_f1"
            ]
            > best_unknown_f1
        ):

            best_unknown_f1 = (
                metrics[
                    "unknown_f1"
                ]
            )

            best_unknown_epoch = (
                epoch
            )

            save_checkpoint(
                path=BEST_UNKNOWN_PATH,
                model=model,
                optimizer=optimizer,
                class_to_idx=(
                    train_dataset
                    .class_to_idx
                ),
                epoch=epoch,
                metrics=result,
                selection_metric=(
                    "unknown_f1"
                ),
            )

            print(
                "  -> 新的最佳 Unknown F1 模型，已保存。"
            )

    # --------------------------------------------------------
    # 最终实验总结
    # --------------------------------------------------------

    print(
        "-" * 84
    )

    print(
        "Unknown 实验 4E 完成。"
    )

    print(
        "\n实验4A参考："
    )

    print(
        "旧 unknown："
        "20来源 × 10张 = 200张"
    )

    print(
        "Macro-F1 = 0.8417"
    )

    print(
        "Unknown F1 = 0.2056"
    )

    print(
        "\n实验4C参考："
    )

    print(
        "随机多样化 unknown："
        "100来源 × 2张 = 200张"
    )

    print(
        "Macro-F1 = 0.8191"
    )

    print(
        "Unknown F1 = 0.1124"
    )

    print(
        "\n实验4E："
    )

    print(
        "Hard v3："
        "40 Hard×4 + 20普通×2 = 200张"
    )

    print(
        "最佳 Macro-F1："
        f"{best_macro_f1:.4f} "
        f"(Epoch {best_macro_epoch})"
    )

    print(
        "最佳 Unknown F1："
        f"{best_unknown_f1:.4f} "
        f"(Epoch {best_unknown_epoch})"
    )

    print(
        f"\n最佳总体模型：{BEST_MACRO_PATH}"
    )

    print(
        f"最佳Unknown模型：{BEST_UNKNOWN_PATH}"
    )

    print(
        f"实验日志：{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()

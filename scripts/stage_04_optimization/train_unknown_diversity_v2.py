"""
阶段四 · Unknown 专项实验 4C：提高 unknown 来源车型多样性
文件名：train_unknown_diversity_v2.py

实验背景
========
旧 unknown 训练方案（v1）：
    20 个来源车型 × 每类 10 张 = 200 张

新 unknown 训练方案（v2）：
    100 个来源车型 × 每类 2 张 = 200 张

也就是说：
    unknown 的总训练图片数量仍然保持 200 张，
    主要改变的是“来源车型多样性”：
        20 种 -> 100 种

这样做是为了回答一个明确问题：
    Unknown F1 偏低，是否主要因为模型见过的“非目标车型种类”太少？

为了做控制变量，本实验其余训练条件都与实验 4A 保持一致：
    - 从 baseline_best.pth 开始
    - 只训练 layer4 + fc
    - Learning Rate = 1e-4
    - Epoch = 5
    - 不使用随机数据增强
    - unknown 的 CrossEntropyLoss 权重仍为 2.0

唯一核心变化：
    dataset_51/train/unknown 中旧的 200 张
                  ↓
    unknown_train_v2/unknown 中新的 200 张

重要：
    本脚本不会修改 dataset_51。
    它只是在程序内：
        1. 从 dataset_51/train 读取 50 个 known 类；
        2. 排除原来的 unknown 图片；
        3. 加入 unknown_train_v2/unknown 的 200 张新图片。

运行：
    python train_unknown_diversity_v2.py

输出：
    outputs/unknown_diversity_v2/
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

# 原 51 类数据集：
TRAIN_DIR = PROJECT_ROOT / "dataset_51" / "train"
VAL_DIR = PROJECT_ROOT / "dataset_51" / "val"

# 新版 unknown 图片：
UNKNOWN_V2_DIR = (
    PROJECT_ROOT
    / "unknown_train_v2"
    / "unknown"
)

# 用于再次检查 train / val unknown 来源没有泄漏：
UNKNOWN_TRAIN_V2_SOURCES = (
    PROJECT_ROOT
    / "unknown_train_sources_v2.txt"
)

UNKNOWN_VAL_SOURCES = (
    PROJECT_ROOT
    / "unknown_val_sources.txt"
)

# 与实验 4A 一样，从同一个 Baseline 开始。
BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_resnet18"
    / "baseline_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unknown_diversity_v2"
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

LEARNING_RATE = 1e-4
NUM_WORKERS = 0

NUM_CLASSES = 51
UNKNOWN_INDEX = 50

# 与实验 4A 完全一致：
UNKNOWN_WEIGHT = 2.0

EXPECTED_UNKNOWN_V2_IMAGES = 200
EXPECTED_UNKNOWN_V2_SOURCES = 100

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def set_random_seed(seed: int) -> None:
    """固定随机种子，使实验尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_source_ids(path: Path):
    """
    读取 unknown 来源车型 ID。

    这里的 txt 每行一个 4 位车型 ID。
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

    return list(
        dict.fromkeys(ids)
    )


def build_datasets():
    """
    构建实验 4C 的 train / val Dataset。

    这里有一个很重要的工程技巧：

    先用 ImageFolder 正常读取：
        dataset_51/train

    它原本包含：
        50 个 known 类
        + 旧 unknown 的 200 张

    然后我们直接修改 train_dataset.samples：

        先删掉 target == 50 的旧 unknown 样本
        再加入 unknown_train_v2/unknown 的新 200 张

    这样：
        - 不需要复制 50 个 known 类
        - 不修改磁盘上的 dataset_51
        - class_to_idx 仍保持原来的 0~50 映射

    torchvision 的 ImageFolder 本质上就是：
        图片路径 -> class index
    的 Dataset，因此可以用这种方式替换样本列表。
    """

    transform = transforms.Compose(
        [
            transforms.Resize(256),
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

    # --------------------------------------------------------
    # 先正常读取原训练集
    # --------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform,
    )

    # 验证集保持完全不变。
    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=transform,
    )

    # --------------------------------------------------------
    # 基础安全检查
    # --------------------------------------------------------
    if (
        train_dataset.class_to_idx
        != val_dataset.class_to_idx
    ):
        raise ValueError(
            "train 和 val 的类别映射不一致。"
        )

    if len(
        train_dataset.classes
    ) != NUM_CLASSES:
        raise ValueError(
            f"类别数应为 {NUM_CLASSES}，"
            f"实际为 {len(train_dataset.classes)}。"
        )

    actual_unknown_index = (
        train_dataset
        .class_to_idx
        .get("unknown")
    )

    if (
        actual_unknown_index
        != UNKNOWN_INDEX
    ):
        raise ValueError(
            "unknown 索引不是 50。"
        )

    # --------------------------------------------------------
    # 找出 50 个 known 类原有样本
    # --------------------------------------------------------
    known_samples = [
        (path, target)
        for path, target
        in train_dataset.samples
        if target != UNKNOWN_INDEX
    ]

    old_unknown_count = (
        len(train_dataset.samples)
        - len(known_samples)
    )

    if (
        old_unknown_count
        != EXPECTED_UNKNOWN_V2_IMAGES
    ):
        raise ValueError(
            "原 dataset_51/train/unknown "
            f"预计应有 200 张，实际为 {old_unknown_count} 张。"
        )

    # --------------------------------------------------------
    # 读取 v2 unknown 的 200 张图片
    # --------------------------------------------------------
    if not UNKNOWN_V2_DIR.exists():
        raise FileNotFoundError(
            f"找不到 v2 unknown 目录：{UNKNOWN_V2_DIR}"
        )

    unknown_v2_paths = sorted(
        path
        for path in UNKNOWN_V2_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if (
        len(unknown_v2_paths)
        != EXPECTED_UNKNOWN_V2_IMAGES
    ):
        raise ValueError(
            "v2 unknown 图片应为 200 张，"
            f"实际为 {len(unknown_v2_paths)} 张。"
        )

    # 给每一张新的 unknown 图片统一赋标签 index=50。
    unknown_v2_samples = [
        (
            str(path),
            UNKNOWN_INDEX,
        )
        for path in unknown_v2_paths
    ]

    # --------------------------------------------------------
    # 替换训练 Dataset 的 samples
    # --------------------------------------------------------
    train_dataset.samples = (
        known_samples
        + unknown_v2_samples
    )

    # ImageFolder 内部 imgs 通常是 samples 的别名。
    # 为了保持属性一致，这里同步更新。
    train_dataset.imgs = (
        train_dataset.samples
    )

    # targets 也同步更新，
    # 方便后续如果代码需要查看训练标签。
    train_dataset.targets = [
        target
        for _, target
        in train_dataset.samples
    ]

    return (
        train_dataset,
        val_dataset,
        old_unknown_count,
        len(unknown_v2_samples),
    )


def verify_unknown_source_isolation():
    """
    再次验证：
        unknown train v2 来源
        与
        unknown val 来源
    没有任何交集。

    这一步属于防止数据泄漏的工程检查。
    """

    train_source_ids = read_source_ids(
        UNKNOWN_TRAIN_V2_SOURCES
    )

    val_source_ids = read_source_ids(
        UNKNOWN_VAL_SOURCES
    )

    if (
        len(train_source_ids)
        != EXPECTED_UNKNOWN_V2_SOURCES
    ):
        raise ValueError(
            "unknown_train_sources_v2.txt "
            f"应有 100 个来源，实际为 {len(train_source_ids)}。"
        )

    overlap = (
        set(train_source_ids)
        & set(val_source_ids)
    )

    if overlap:
        raise ValueError(
            "发现 unknown train / val 来源泄漏："
            f"{sorted(overlap)}"
        )

    return (
        len(train_source_ids),
        len(val_source_ids),
        overlap,
    )


def build_dataloaders(
    train_dataset,
    val_dataset,
):
    """构建 DataLoader。"""

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


def build_model(
    device: torch.device,
):
    """
    与实验 4A 完全相同：

        baseline_best.pth
            ↓
        冻结全部参数
            ↓
        解冻 layer4 + fc
    """

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"找不到 Baseline checkpoint：{BASELINE_CHECKPOINT}"
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

    # 先冻结全部。
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
        model.to(device),
        checkpoint,
    )


def set_finetune_train_mode(
    model,
) -> None:
    """
    与实验 4A 相同：
        layer4 + fc train
        其余保持 eval
        layer4 的 BatchNorm 保持 eval
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


def build_weighted_loss(
    device: torch.device,
):
    """
    与实验 4A 保持完全相同：

        known 类权重 = 1
        unknown 权重 = 2
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


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """训练一个 Epoch。"""

    set_finetune_train_mode(
        model
    )

    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 核心训练五步：
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

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


def evaluate(
    model,
    val_loader,
    criterion,
    device,
):
    """
    验证指标：
        Accuracy
        51 类 Macro-F1
        50 个 known 类 Macro-F1
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
                labels.cpu().tolist()
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
            range(NUM_CLASSES)
        ),
        average="macro",
        zero_division=0,
    )

    known_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(
            range(UNKNOWN_INDEX)
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
        "val_loss": (
            val_loss
        ),
        "val_accuracy": (
            val_accuracy
        ),
        "val_macro_f1": (
            macro_f1
        ),
        "known_macro_f1": (
            known_macro_f1
        ),
        "unknown_precision": (
            float(
                precision[0]
            )
        ),
        "unknown_recall": (
            float(
                recall[0]
            )
        ),
        "unknown_f1": (
            float(
                f1[0]
            )
        ),
        "unknown_support": (
            int(
                support[0]
            )
        ),
    }


def save_history(
    history,
) -> None:
    """保存训练日志。"""

    if not history:
        return

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                history[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            history
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    class_to_idx,
    epoch,
    metrics,
    selection_metric,
):
    """保存最佳 checkpoint。"""

    checkpoint = {
        "model_name": (
            "resnet18"
        ),

        "experiment": (
            "unknown_diversity_v2"
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

        "metrics": metrics,

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
                "v2"
            ),

            "unknown_train_sources": (
                EXPECTED_UNKNOWN_V2_SOURCES
            ),

            "unknown_train_images": (
                EXPECTED_UNKNOWN_V2_IMAGES
            ),

            "data_augmentation": (
                False
            ),

            "starting_checkpoint": (
                str(
                    BASELINE_CHECKPOINT
                )
            ),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def main():
    """阶段四 Unknown 实验 4C 主流程。"""

    print("=" * 82)
    print(
        "阶段四 · Unknown实验4C："
        "v2来源多样性 + weight=2"
    )
    print("=" * 82)

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
    # 检查 train / val unknown 来源隔离
    # --------------------------------------------------------
    (
        train_source_count,
        val_source_count,
        overlap,
    ) = verify_unknown_source_isolation()

    print(
        "\nunknown train v2 来源数："
        f"{train_source_count}"
    )

    print(
        "unknown val 来源数："
        f"{val_source_count}"
    )

    print(
        "来源交集："
        f"{overlap}"
    )

    # --------------------------------------------------------
    # 构建替换过 unknown 的 Dataset
    # --------------------------------------------------------
    (
        train_dataset,
        val_dataset,
        old_unknown_count,
        new_unknown_count,
    ) = build_datasets()

    train_loader, val_loader = (
        build_dataloaders(
            train_dataset,
            val_dataset,
        )
    )

    print(
        "\n旧 unknown 图片数："
        f"{old_unknown_count}"
    )

    print(
        "新 unknown 图片数："
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

    # 因为 200 张替换 200 张，
    # 最终 train 应仍然是原来的 5902 张。
    print(
        "\n控制变量确认："
        "unknown总图片数量没有变化，"
        "主要变化是来源车型从20种增加到100种。"
    )

    # --------------------------------------------------------
    # 模型 / Loss / Optimizer
    # --------------------------------------------------------
    (
        model,
        baseline_checkpoint,
    ) = build_model(
        device
    )

    print(
        "起始 Baseline Macro-F1："
        f"{baseline_checkpoint['val_macro_f1']:.4f}"
    )

    criterion = (
        build_weighted_loss(
            device
        )
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_macro_epoch = None

    best_unknown_f1 = -1.0
    best_unknown_epoch = None

    history = []

    print(
        "\n开始训练..."
    )

    print(
        "-" * 82
    )

    # --------------------------------------------------------
    # 训练 5 个 Epoch
    # --------------------------------------------------------
    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        start = (
            time.perf_counter()
        )

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        seconds = (
            time.perf_counter()
            - start
        )

        result = {
            "epoch": epoch,

            "train_loss": (
                train_loss
            ),

            "train_accuracy": (
                train_accuracy
            ),

            **metrics,

            "epoch_seconds": (
                seconds
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
            f"Time {seconds:.1f}s"
        )

        # 保存总体 Macro-F1 最佳模型
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

        # 保存 Unknown F1 最佳模型
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
    # 实验总结
    # --------------------------------------------------------
    print(
        "-" * 82
    )

    print(
        "Unknown 实验 4C 完成。"
    )

    print(
        "\n实验4A参考："
    )

    print(
        "旧 unknown："
        "20来源 × 10张"
    )

    print(
        "Macro-F1 = 0.8417"
    )

    print(
        "Unknown F1 = 0.2056"
    )

    print(
        "\n实验4C："
    )

    print(
        "新 unknown："
        "100来源 × 2张"
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

"""
阶段三：Baseline 视觉分类模型训练
文件名：train_baseline.py

项目背景：
    使用已经构建好的 dataset_51 数据集训练一个 51 类车型分类 Baseline：
    50 个目标车型 + 1 个 unknown。

Baseline 方案：
    1. 使用 torchvision 提供的 ImageNet 预训练 ResNet18。
    2. 冻结 ResNet18 的预训练特征提取部分。
    3. 将原来的 1000 类分类层替换为 51 类分类层。
    4. 第一版只训练新的全连接分类层（fc）。
    5. 每个 Epoch 在验证集上计算 Accuracy 和 Macro-F1。
    6. 按验证集 Macro-F1 保存最佳模型。

运行命令：
    python train_baseline.py

主要输出：
    outputs/baseline_resnet18/
        ├── baseline_best.pth
        ├── training_history.csv
        └── class_mapping.json
"""

from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset_51"
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
CLASS_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "stage_01_02_data"
    / "candidate_classes_v1.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_resnet18"
BEST_MODEL_PATH = OUTPUT_DIR / "baseline_best.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
CLASS_MAPPING_PATH = OUTPUT_DIR / "class_mapping.json"


# ============================================================
# 2. Baseline 超参数
# ============================================================

RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
NUM_CLASSES = 51

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_target_ids(path: Path) -> list[str]:
    """
    从 candidate_classes_v1.txt 读取 50 个目标车型 ID，
    并检查数量、重复项和排序。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到目标类别文件：{path}")

    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(ids) != 50:
        raise ValueError(f"目标车型必须恰好 50 类，现在读取到 {len(ids)} 类。")

    if len(set(ids)) != 50:
        raise ValueError("目标车型 ID 中存在重复项。")

    if ids != sorted(ids):
        raise ValueError("目标车型 ID 必须按升序排列。")

    return ids


def build_transforms():
    """
    创建 Baseline 图片预处理。

    当前 Baseline 暂时不加入随机数据增强，
    这样后续阶段四加入增强后可以做公平对比。
    """
    baseline_transform = transforms.Compose(
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

    return baseline_transform, baseline_transform


def build_datasets():
    """
    使用 ImageFolder 读取 train / val。

    ImageFolder 会把每个子文件夹名称当成类别名称。
    """
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"找不到训练集目录：{TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"找不到验证集目录：{VAL_DIR}")

    train_transform, val_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform,
    )

    return train_dataset, val_dataset


def validate_class_mapping(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    target_ids: list[str],
) -> None:
    """
    检查类别索引是否满足：
        0~49 = 50 个目标车型
        50   = unknown
    """
    expected_classes = target_ids + ["unknown"]

    if train_dataset.classes != expected_classes:
        raise ValueError(
            "训练集类别顺序与预期不一致。\n"
            f"预期：{expected_classes}\n"
            f"实际：{train_dataset.classes}"
        )

    if val_dataset.classes != expected_classes:
        raise ValueError(
            "验证集类别顺序与预期不一致。\n"
            f"预期：{expected_classes}\n"
            f"实际：{val_dataset.classes}"
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train 和 val 的类别到索引映射不一致。")


def build_dataloaders(train_dataset, val_dataset):
    """
    Dataset 负责定义“有什么数据、怎么预处理”。
    DataLoader 负责定义“每次取多少张、是否打乱、怎么组成 Batch”。
    """
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader


def build_model(device: torch.device):
    """
    创建 ImageNet 预训练 ResNet18，
    冻结预训练参数，并将 1000 分类层替换成 51 分类层。
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # 冻结预训练特征提取部分。
    for parameter in model.parameters():
        parameter.requires_grad = False

    # ResNet18 原 fc 的输入特征维度为 512。
    in_features = model.fc.in_features

    # 替换为 51 类输出。
    model.fc = nn.Linear(
        in_features=in_features,
        out_features=NUM_CLASSES,
    )

    return model.to(device)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """
    完成一个 Epoch 的训练。

    核心顺序：
        forward
        -> loss
        -> zero_grad
        -> backward
        -> optimizer.step
    """
    # 当前 Baseline 只训练 fc。
    # 用 eval 保持冻结的 BatchNorm 统计量不变。
    model.eval()
    model.fc.train()

    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def evaluate(
    model,
    val_loader,
    criterion,
    device,
):
    """
    在验证集上计算：
        val_loss
        val_accuracy
        val_macro_f1
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    val_loss = running_loss / total_samples
    val_accuracy = correct_predictions / total_samples

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return val_loss, val_accuracy, val_macro_f1


def save_class_mapping(train_dataset) -> None:
    """保存类别名称与模型索引之间的映射。"""
    mapping = {
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": {
            str(index): class_name
            for class_name, index in train_dataset.class_to_idx.items()
        },
    }

    CLASS_MAPPING_PATH.write_text(
        json.dumps(
            mapping,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_history(history: list[dict]) -> None:
    """保存每个 Epoch 的训练和验证指标。"""
    if not history:
        return

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    model,
    optimizer,
    train_dataset,
    epoch,
    val_accuracy,
    val_macro_f1,
) -> None:
    """
    保存当前最佳模型，以及恢复模型时需要的重要信息。
    """
    checkpoint = {
        "model_name": "resnet18",
        "epoch": epoch,
        "num_classes": NUM_CLASSES,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "config": {
            "random_seed": RANDOM_SEED,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "num_workers": NUM_WORKERS,
            "training_strategy": "freeze_backbone_train_fc_only",
            "pretrained_weights": "ResNet18_Weights.DEFAULT",
        },
    }

    torch.save(checkpoint, BEST_MODEL_PATH)


def main() -> None:
    """Baseline 训练主流程。"""

    print("=" * 70)
    print("阶段三：ResNet18 Baseline 训练")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"计算设备：{device}")
    print(f"数据集目录：{DATASET_ROOT}")
    print(f"Batch Size：{BATCH_SIZE}")
    print(f"Epochs：{EPOCHS}")
    print(f"Learning Rate：{LEARNING_RATE}")

    # 1. 读取目标车型 ID。
    target_ids = read_target_ids(CLASS_FILE)

    # 2. 构建 Dataset。
    train_dataset, val_dataset = build_datasets()

    # 3. 检查类别映射。
    validate_class_mapping(
        train_dataset,
        val_dataset,
        target_ids,
    )

    print("\n类别映射检查通过。")
    print(f"训练集图片数：{len(train_dataset)}")
    print(f"验证集图片数：{len(val_dataset)}")
    print(f"类别数：{len(train_dataset.classes)}")
    print(f"unknown 索引：{train_dataset.class_to_idx['unknown']}")

    save_class_mapping(train_dataset)

    # 4. 构建 DataLoader。
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
    )

    # 5. 创建迁移学习模型。
    print("\n正在加载 ImageNet 预训练 ResNet18...")
    model = build_model(device)

    # 6. 多分类损失函数。
    criterion = nn.CrossEntropyLoss()

    # 7. 第一版只优化新的 fc 分类层。
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    history = []

    print("\n开始训练...")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy, val_macro_f1 = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_seconds = time.perf_counter() - epoch_start_time

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": epoch_seconds,
        }

        history.append(epoch_result)

        # 每个 Epoch 都写一次，避免意外中断后日志完全丢失。
        save_history(history)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Time: {epoch_seconds:.1f}s"
        )

        # 老师最终按 Macro-F1 评分，所以也按 Macro-F1 选最佳模型。
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                epoch=epoch,
                val_accuracy=val_accuracy,
                val_macro_f1=val_macro_f1,
            )

            print(
                "  -> 新的最佳模型，已保存："
                f"{BEST_MODEL_PATH.name}"
            )

    print("-" * 70)
    print("Baseline 训练完成。")
    print(f"最佳验证 Macro-F1：{best_macro_f1:.4f}")
    print(f"最佳模型：{BEST_MODEL_PATH}")
    print(f"训练日志：{HISTORY_PATH}")
    print(f"类别映射：{CLASS_MAPPING_PATH}")


if __name__ == "__main__":
    main()

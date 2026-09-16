"""
阶段四 · 实验 2：layer4 Fine-tuning + 数据增强
文件名：train_finetune_aug.py

实验目的：
    实验 1 已经证明，解冻 ResNet18 的 layer4 + fc 能显著提升性能。
    但 Train Acc 很快接近 100%，已经出现明显过拟合趋势。

本实验只改变训练集的数据增强：
    1. RandomResizedCrop
    2. RandomHorizontalFlip
    3. 轻微 ColorJitter

为了做控制变量实验：
    - 仍然从 baseline_best.pth 开始
    - 仍然只训练 layer4 + fc
    - 学习率仍然是 1e-4
    - 仍然训练 5 个 Epoch
    - 验证集预处理保持不变

运行：
    python train_finetune_aug.py
"""

from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "dataset_51" / "train"
VAL_DIR = PROJECT_ROOT / "dataset_51" / "val"

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_resnet18"
    / "baseline_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_layer4_aug"
BEST_MODEL_PATH = OUTPUT_DIR / "finetune_aug_best.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


# ============================================================
# 2. 超参数
# ============================================================

RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_random_seed(seed: int) -> None:
    """固定随机种子，使实验尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_datasets():
    """
    构建训练集和验证集。

    train：
        加入随机数据增强，提高泛化能力。

    val：
        保持固定预处理，保证不同实验之间可以公平比较。
    """

    # 训练集数据增强。
    train_transform = transforms.Compose(
        [
            # 随机裁剪并缩放到 224x224。
            # scale 不设置得太激进，避免把汽车主体裁掉太多。
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),

            # 50% 概率水平翻转。
            # 汽车朝左或朝右一般不会改变车型类别。
            transforms.RandomHorizontalFlip(p=0.5),

            # 轻微改变亮度、对比度、饱和度、色调，
            # 模拟不同光照和拍摄环境。
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    # 验证集不加随机增强。
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

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform,
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train 和 val 的类别映射不一致。")

    if len(train_dataset.classes) != 51:
        raise ValueError(
            f"类别数应为 51，当前为 {len(train_dataset.classes)}。"
        )

    if train_dataset.class_to_idx.get("unknown") != 50:
        raise ValueError("unknown 索引不是 50，请检查数据集。")

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset):
    """将 Dataset 按 Batch 组织成 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

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

    return train_loader, val_loader


def load_baseline_and_unfreeze_layer4(device: torch.device):
    """
    从 Baseline checkpoint 恢复模型。

    与实验 1 一样：
        - 前面层冻结
        - layer4 解冻
        - fc 解冻

    这样实验 1 和实验 2 的主要变量就是：
        是否加入数据增强。
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

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        checkpoint["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 先冻结全部参数。
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 解冻 layer4。
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # 解冻分类头。
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model.to(device), checkpoint


def set_finetune_train_mode(model) -> None:
    """
    设置 Fine-tuning 训练状态。

    冻结部分保持 eval；
    layer4 + fc 参与训练；
    layer4 中 BatchNorm 保持 eval，避免小数据集改变其统计量。
    """

    model.eval()
    model.layer4.train()
    model.fc.train()

    for module in model.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """训练一个 Epoch。"""

    set_finetune_train_mode(model)

    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = running_loss / total_samples
    train_accuracy = correct / total_samples

    return train_loss, train_accuracy


def evaluate(
    model,
    val_loader,
    criterion,
    device,
    unknown_index: int,
):
    """
    在验证集评估：
        Val Loss
        Val Accuracy
        Macro-F1
        Unknown Precision / Recall / F1
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    val_loss = running_loss / total_samples

    val_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[unknown_index],
        average=None,
        zero_division=0,
    )

    return (
        val_loss,
        val_accuracy,
        macro_f1,
        float(precision[0]),
        float(recall[0]),
        float(f1[0]),
    )


def save_history(history) -> None:
    """保存每个 Epoch 的实验指标。"""

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
    class_to_idx,
    epoch,
    metrics,
) -> None:
    """保存当前 Macro-F1 最好的模型。"""

    checkpoint = {
        "model_name": "resnet18",
        "experiment": "finetune_layer4_augmentation",
        "epoch": epoch,
        "num_classes": 51,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "metrics": metrics,
        "config": {
            "random_seed": RANDOM_SEED,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "trainable_modules": ["layer4", "fc"],
            "data_augmentation": True,
            "augmentation": {
                "random_resized_crop_scale": [0.8, 1.0],
                "random_resized_crop_ratio": [0.9, 1.1],
                "horizontal_flip_p": 0.5,
                "color_jitter": {
                    "brightness": 0.15,
                    "contrast": 0.15,
                    "saturation": 0.15,
                    "hue": 0.02,
                },
            },
            "starting_checkpoint": str(
                BASELINE_CHECKPOINT
            ),
        },
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )


def main():
    """阶段四实验 2 主流程。"""

    print("=" * 74)
    print("阶段四 · 实验2：layer4 Fine-tuning + 数据增强")
    print("=" * 74)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"设备：{device}")
    print(f"Epochs：{EPOCHS}")
    print(f"Learning Rate：{LEARNING_RATE}")

    print("\n训练集数据增强：")
    print("- RandomResizedCrop(scale=0.8~1.0)")
    print("- RandomHorizontalFlip(p=0.5)")
    print("- 轻微 ColorJitter")

    train_dataset, val_dataset = build_datasets()

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
    )

    unknown_index = train_dataset.class_to_idx["unknown"]

    print(f"\n训练集：{len(train_dataset)} 张")
    print(f"验证集：{len(val_dataset)} 张")
    print(f"unknown index：{unknown_index}")

    model, baseline_checkpoint = (
        load_baseline_and_unfreeze_layer4(
            device
        )
    )

    baseline_macro_f1 = baseline_checkpoint[
        "val_macro_f1"
    ]

    print(
        "起始 Baseline Macro-F1："
        f"{baseline_macro_f1:.4f}"
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "可训练参数："
        f"{trainable_parameters:,} / "
        f"{total_parameters:,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_epoch = None
    history = []

    print("\n开始训练...")
    print("-" * 74)

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            val_loss,
            val_accuracy,
            macro_f1,
            unknown_precision,
            unknown_recall,
            unknown_f1,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            unknown_index,
        )

        seconds = time.perf_counter() - start

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": macro_f1,
            "unknown_precision": unknown_precision,
            "unknown_recall": unknown_recall,
            "unknown_f1": unknown_f1,
            "epoch_seconds": seconds,
        }

        history.append(result)
        save_history(history)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"Macro-F1 {macro_f1:.4f} | "
            f"Unknown F1 {unknown_f1:.4f} | "
            f"Time {seconds:.1f}s"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                class_to_idx=train_dataset.class_to_idx,
                epoch=epoch,
                metrics=result,
            )

            print(
                "  -> 本实验新的最佳模型，已保存。"
            )

    print("-" * 74)
    print("数据增强实验完成。")
    print(
        f"Baseline Macro-F1：{baseline_macro_f1:.4f}"
    )
    print(
        f"本实验最佳 Macro-F1：{best_macro_f1:.4f}"
    )
    print(f"本实验最佳 Epoch：{best_epoch}")
    print(f"最佳模型：{BEST_MODEL_PATH}")
    print(f"实验日志：{HISTORY_PATH}")
    print(
        "\n请将本实验结果与实验1的 Macro-F1=0.8239 对比。"
    )


if __name__ == "__main__":
    main()

"""
阶段四 · 实验 1：解冻 ResNet18 的 layer4 进行 Fine-tuning
文件名：train_finetune_layer4.py

实验目的：
    Baseline 只训练最后的 fc 分类头，验证集 Macro-F1 = 0.6017，
    但 unknown F1 仅约 0.0241，而且多个细粒度车型类别表现较弱。

    本实验只做一个主要改变：
        在 Baseline 最佳模型的基础上，解冻 ResNet18 最后一个残差阶段 layer4，
        让模型的高层视觉特征也能够针对车型数据进行调整。

为了让实验可解释，本实验暂时“不加入数据增强”，保持验证预处理、
Batch Size 等核心数据流程与 Baseline 一致。

运行：
    python train_finetune_layer4.py

输入：
    outputs/baseline_resnet18/baseline_best.pth
    dataset_51/train/
    dataset_51/val/

输出：
    outputs/finetune_layer4/
        ├── finetune_best.pth
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_layer4"
BEST_MODEL_PATH = OUTPUT_DIR / "finetune_best.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


# ============================================================
# 2. 实验参数
# ============================================================

RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32

# 从 Baseline 最佳模型继续微调 5 个 Epoch。
EPOCHS = 5

# Fine-tuning 使用比 Baseline 分类头训练更小的学习率。
# 原因：layer4 已经包含 ImageNet 学到的有效特征，
# 我们只希望“微调”，而不是大幅破坏已有参数。
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
    构建 train / val Dataset。

    注意：
        本实验暂时不加入随机数据增强。
        这是为了尽量只比较“解冻 layer4”本身带来的变化。

    下一轮实验再单独加入数据增强。
    """
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

    if len(train_dataset.classes) != 51:
        raise ValueError(
            f"类别数应为 51，现在是 {len(train_dataset.classes)}。"
        )

    if train_dataset.class_to_idx.get("unknown") != 50:
        raise ValueError(
            "unknown 的类别索引不是 50，请停止实验检查数据。"
        )

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


def load_baseline_and_unfreeze_layer4(
    device: torch.device,
):
    """
    加载 Baseline 最佳模型，然后只解冻：
        - layer4
        - fc

    ResNet18 可以粗略理解为：
        前面层 -> layer1 -> layer2 -> layer3 -> layer4 -> fc

    layer4 靠近输出端，负责较高级的视觉特征。
    解冻它比“整个网络全部解冻”更保守，也更适合当前小数据集。
    """
    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"找不到 Baseline 模型：{BASELINE_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    num_classes = checkpoint["num_classes"]

    # 先建立与 Baseline 一致的网络结构。
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    # 恢复 Baseline 已经训练好的全部参数。
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 第一步：全部冻结。
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 第二步：只解冻最后的 layer4。
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # 第三步：分类头继续训练。
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    return model, checkpoint


def set_finetune_train_mode(model) -> None:
    """
    设置 Fine-tuning 训练状态。

    先 model.eval()：
        保持前面被冻结的网络处于推理模式。

    再 layer4.train() / fc.train()：
        指定真正需要训练的模块。

    对小数据集，为了让训练更稳定，再把 BatchNorm 层保持 eval 模式，
    避免较小 Batch 改写预训练时积累的运行均值和方差。
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
    """完成一个 Fine-tuning Epoch。"""

    set_finetune_train_mode(model)

    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清除上一 Batch 的梯度。
        optimizer.zero_grad()

        # Forward。
        outputs = model(images)

        # Loss。
        loss = criterion(
            outputs,
            labels,
        )

        # Backward。
        loss.backward()

        # 根据梯度更新 layer4 + fc。
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )
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
    验证模型，并同时计算：
        Accuracy
        Macro-F1
        unknown Precision
        unknown Recall
        unknown F1
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

            running_loss += (
                loss.item() * batch_size
            )
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)

            y_true.extend(
                labels.cpu().tolist()
            )
            y_pred.extend(
                predictions.cpu().tolist()
            )

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

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[unknown_index],
            average=None,
            zero_division=0,
        )
    )

    unknown_precision = float(precision[0])
    unknown_recall = float(recall[0])
    unknown_f1 = float(f1[0])

    return (
        val_loss,
        val_accuracy,
        macro_f1,
        unknown_precision,
        unknown_recall,
        unknown_f1,
    )


def save_history(history):
    """保存每个 Epoch 的实验指标。"""
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
    """保存当前 Macro-F1 最优的 Fine-tuning 模型。"""
    checkpoint = {
        "model_name": "resnet18",
        "experiment": "finetune_layer4",
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
            "trainable_modules": [
                "layer4",
                "fc",
            ],
            "data_augmentation": False,
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
    """阶段四实验 1 主流程。"""

    print("=" * 72)
    print("阶段四 · 实验1：ResNet18 layer4 Fine-tuning")
    print("=" * 72)

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

    print(f"设备：{device}")
    print(f"Epochs：{EPOCHS}")
    print(f"Learning Rate：{LEARNING_RATE}")

    train_dataset, val_dataset = (
        build_datasets()
    )

    train_loader, val_loader = (
        build_dataloaders(
            train_dataset,
            val_dataset,
        )
    )

    unknown_index = (
        train_dataset.class_to_idx["unknown"]
    )

    print(
        f"训练集：{len(train_dataset)} 张"
    )
    print(
        f"验证集：{len(val_dataset)} 张"
    )
    print(
        f"unknown index：{unknown_index}"
    )

    model, baseline_checkpoint = (
        load_baseline_and_unfreeze_layer4(
            device
        )
    )

    print(
        "起始 Baseline Macro-F1："
        f"{baseline_checkpoint['val_macro_f1']:.4f}"
    )

    # 查看有多少参数真正参与训练。
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

    # optimizer 只接收 requires_grad=True 的参数：
    # 即 layer4 + fc。
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = baseline_checkpoint[
        "val_macro_f1"
    ]

    history = []

    print("\n开始 Fine-tuning...")
    print("-" * 72)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        start = time.perf_counter()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
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

        seconds = (
            time.perf_counter() - start
        )

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

        # 只有真正超过 Baseline 才保存为新的最佳 Fine-tuning 模型。
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                class_to_idx=(
                    train_dataset.class_to_idx
                ),
                epoch=epoch,
                metrics=result,
            )

            print(
                "  -> 超过 Baseline，"
                "新的最佳 Fine-tuning 模型已保存。"
            )

    print("-" * 72)
    print("Fine-tuning 实验完成。")
    print(
        "Baseline Macro-F1："
        f"{baseline_checkpoint['val_macro_f1']:.4f}"
    )
    print(
        "本实验最佳 Macro-F1："
        f"{best_macro_f1:.4f}"
    )

    if BEST_MODEL_PATH.exists():
        print(
            f"最佳模型：{BEST_MODEL_PATH}"
        )
    else:
        print(
            "本实验没有超过 Baseline，"
            "因此没有生成新的最佳模型文件。"
        )

    print(
        f"实验日志：{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()

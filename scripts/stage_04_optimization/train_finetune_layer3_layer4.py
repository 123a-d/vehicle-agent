"""
阶段四 · 实验 3：解冻 layer3 + layer4 + fc，并使用分层学习率
文件名：train_finetune_layer3_layer4.py

实验目的：
    实验 1（layer4 + fc）已经把 Macro-F1 从 0.6017 提升到 0.8239。
    本实验进一步解冻 layer3，观察更深一层的高阶视觉特征是否还能提升车型识别。

为了保持实验可解释性，本实验：
    1. 仍然从同一个 Baseline checkpoint 开始；
    2. 不加入数据增强；
    3. Batch Size、Epoch 数、验证集预处理保持不变；
    4. 主要改变：
       - 解冻范围：layer4 + fc -> layer3 + layer4 + fc
       - 使用分层学习率（越靠前的层学习率越小）

分层学习率：
    layer3 = 1e-5
    layer4 = 5e-5
    fc     = 1e-4

运行：
    python train_finetune_layer3_layer4.py
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
# 1. 路径配置
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "finetune_layer3_layer4"
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "finetune_layer3_layer4_best.pth"
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

# 分层学习率
LR_LAYER3 = 1e-5
LR_LAYER4 = 5e-5
LR_FC = 1e-4

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

    本实验不加入随机数据增强，
    目的是尽量把“是否解冻 layer3”作为主要实验变量。
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

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train 和 val 的类别映射不一致。")

    if len(train_dataset.classes) != 51:
        raise ValueError(
            f"类别数应为 51，当前为 {len(train_dataset.classes)}。"
        )

    if train_dataset.class_to_idx.get("unknown") != 50:
        raise ValueError(
            "unknown 索引不是 50，请检查 dataset_51。"
        )

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset):
    """把 Dataset 按 Batch 组织成 DataLoader。"""

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


def load_baseline_and_unfreeze(device: torch.device):
    """
    加载 Baseline checkpoint，并设置可训练模块：

        conv1 / layer1 / layer2 -> 冻结
        layer3                 -> 解冻
        layer4                 -> 解冻
        fc                     -> 解冻
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

    # 解冻 layer3。
    for parameter in model.layer3.parameters():
        parameter.requires_grad = True

    # 解冻 layer4。
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # 解冻 fc。
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model.to(device), checkpoint


def set_finetune_train_mode(model) -> None:
    """
    设置 Fine-tuning 训练状态。

    整体先进入 eval，使被冻结部分保持稳定；
    再让 layer3、layer4、fc 进入训练状态。

    layer3 / layer4 内部的 BatchNorm 继续保持 eval，
    避免小数据集和小 Batch 改写其运行统计量。
    """

    model.eval()

    model.layer3.train()
    model.layer4.train()
    model.fc.train()

    for module in model.layer3.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    for module in model.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_optimizer(model):
    """
    创建使用分层学习率的 Adam 优化器。

    layer3：1e-5
    layer4：5e-5
    fc：1e-4

    越靠前的层越通用，所以学习率更小。
    """

    return torch.optim.Adam(
        [
            {
                "params": model.layer3.parameters(),
                "lr": LR_LAYER3,
            },
            {
                "params": model.layer4.parameters(),
                "lr": LR_LAYER4,
            },
            {
                "params": model.fc.parameters(),
                "lr": LR_FC,
            },
        ]
    )


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """完成一个训练 Epoch。"""

    set_finetune_train_mode(model)

    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. 清梯度
        optimizer.zero_grad()

        # 2. Forward
        outputs = model(images)

        # 3. Loss
        loss = criterion(outputs, labels)

        # 4. Backward
        loss.backward()

        # 5. 根据不同参数组的学习率更新
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

            loss = criterion(outputs, labels)

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
    """保存每个 Epoch 的训练 / 验证指标。"""

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
    """保存本实验 Macro-F1 最优 checkpoint。"""

    checkpoint = {
        "model_name": "resnet18",
        "experiment": "finetune_layer3_layer4",
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
            "learning_rates": {
                "layer3": LR_LAYER3,
                "layer4": LR_LAYER4,
                "fc": LR_FC,
            },
            "trainable_modules": [
                "layer3",
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
    """阶段四 · 实验 3 主流程。"""

    print("=" * 76)
    print(
        "阶段四 · 实验3："
        "layer3 + layer4 + fc Fine-tuning"
    )
    print("=" * 76)

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

    print("\n分层学习率：")
    print(f"- layer3：{LR_LAYER3}")
    print(f"- layer4：{LR_LAYER4}")
    print(f"- fc：{LR_FC}")

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
        load_baseline_and_unfreeze(device)
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

    optimizer = build_optimizer(model)

    best_macro_f1 = -1.0
    best_epoch = None

    history = []

    print("\n开始训练...")
    print("-" * 76)

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

    print("-" * 76)
    print("实验 3 完成。")
    print(
        f"Baseline Macro-F1：{baseline_macro_f1:.4f}"
    )
    print("实验1参考 Macro-F1：0.8239")
    print(
        f"实验3最佳 Macro-F1：{best_macro_f1:.4f}"
    )
    print(
        f"实验3最佳 Epoch：{best_epoch}"
    )
    print(
        f"最佳模型：{BEST_MODEL_PATH}"
    )
    print(
        f"实验日志：{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()

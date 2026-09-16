"""
阶段四 · Unknown 专项实验 4A：提高 unknown 类的损失权重
文件名：train_unknown_weighted.py

当前最佳实验1：
    Macro-F1 = 0.8239
    Unknown F1 = 0.1458

本实验保持实验1其余条件不变，只改变损失函数：
    普通类别权重 = 1.0
    unknown 权重 = 2.0

目的：
    当真实 unknown 被预测错时，给予更大的损失惩罚，
    观察是否能提高 unknown 的 Precision / Recall / F1。

注意：
    unknown 训练样本并不比单个目标车型少，
    所以这里不是“样本不平衡补偿”，
    而是人为提高 unknown 错分代价的 cost-sensitive training。

控制变量：
    - 从 baseline_best.pth 开始
    - 解冻 layer4 + fc
    - lr = 1e-4
    - 5 Epoch
    - 不加数据增强

运行：
    python train_unknown_weighted.py
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "dataset_51" / "train"
VAL_DIR = PROJECT_ROOT / "dataset_51" / "val"

BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "baseline_resnet18" / "baseline_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unknown_weighted"
BEST_MACRO_PATH = OUTPUT_DIR / "best_macro_f1.pth"
BEST_UNKNOWN_PATH = OUTPUT_DIR / "best_unknown_f1.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

NUM_CLASSES = 51
UNKNOWN_INDEX = 50
UNKNOWN_WEIGHT = 2.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_random_seed(seed: int) -> None:
    """固定随机种子，使实验尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_datasets():
    """
    与实验1保持一致：
    不加入随机数据增强。
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

    if len(train_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"类别数应为 {NUM_CLASSES}，当前为 {len(train_dataset.classes)}。"
        )

    if train_dataset.class_to_idx.get("unknown") != UNKNOWN_INDEX:
        raise ValueError("unknown 索引不是 50，请检查数据集。")

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset):
    """把 Dataset 组织成 DataLoader。"""

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


def build_model(device: torch.device):
    """
    与实验1保持一致：
        从 baseline_best.pth 恢复
        冻结全部
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

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        checkpoint["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model.to(device), checkpoint


def set_finetune_train_mode(model) -> None:
    """
    与实验1相同：
    layer4 + fc 参与训练；
    layer4 里的 BatchNorm 保持 eval。
    """

    model.eval()
    model.layer4.train()
    model.fc.train()

    for module in model.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_weighted_loss(device: torch.device):
    """
    本实验的核心变化。

    class_weights:
        0~49 类权重 = 1.0
        unknown(50) = 2.0
    """

    class_weights = torch.ones(
        NUM_CLASSES,
        dtype=torch.float32,
        device=device,
    )

    class_weights[UNKNOWN_INDEX] = UNKNOWN_WEIGHT

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

    set_finetune_train_mode(model)

    running_loss = 0.0
    total_samples = 0
    correct = 0

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

        correct += (
            predictions == labels
        ).sum().item()

    return (
        running_loss / total_samples,
        correct / total_samples,
    )


def evaluate(
    model,
    val_loader,
    criterion,
    device,
):
    """
    验证指标：
        Val Accuracy
        51类 Macro-F1
        50个 known 类 Macro-F1
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

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

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
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": macro_f1,
        "known_macro_f1": known_macro_f1,
        "unknown_precision": float(precision[0]),
        "unknown_recall": float(recall[0]),
        "unknown_f1": float(f1[0]),
        "unknown_support": int(support[0]),
    }


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
    path,
    model,
    optimizer,
    class_to_idx,
    epoch,
    metrics,
    selection_metric,
):
    """保存模型 checkpoint。"""

    checkpoint = {
        "model_name": "resnet18",
        "experiment": "unknown_weighted",
        "selection_metric": selection_metric,
        "epoch": epoch,
        "num_classes": NUM_CLASSES,
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
            "unknown_index": UNKNOWN_INDEX,
            "unknown_weight": UNKNOWN_WEIGHT,
            "data_augmentation": False,
            "starting_checkpoint": str(BASELINE_CHECKPOINT),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def main():
    """Unknown 专项实验 4A。"""

    print("=" * 80)
    print(
        "阶段四 · Unknown专项实验4A："
        "Weighted CrossEntropyLoss"
    )
    print("=" * 80)

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
    print(f"Unknown Weight：{UNKNOWN_WEIGHT}")

    train_dataset, val_dataset = build_datasets()

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
    )

    print(f"\n训练集：{len(train_dataset)} 张")
    print(f"验证集：{len(val_dataset)} 张")
    print(f"unknown index：{UNKNOWN_INDEX}")

    model, baseline_checkpoint = build_model(device)

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

    # 核心变化：Weighted CrossEntropyLoss
    criterion = build_weighted_loss(device)

    # 与实验1相同。
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

    print("\n开始训练...")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
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

        seconds = time.perf_counter() - start

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            **metrics,
            "epoch_seconds": seconds,
        }

        history.append(result)
        save_history(history)

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

        if metrics["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["val_macro_f1"]
            best_macro_epoch = epoch

            save_checkpoint(
                path=BEST_MACRO_PATH,
                model=model,
                optimizer=optimizer,
                class_to_idx=train_dataset.class_to_idx,
                epoch=epoch,
                metrics=result,
                selection_metric="val_macro_f1",
            )

            print(
                "  -> 新的最佳 Macro-F1 模型，已保存。"
            )

        if metrics["unknown_f1"] > best_unknown_f1:
            best_unknown_f1 = metrics["unknown_f1"]
            best_unknown_epoch = epoch

            save_checkpoint(
                path=BEST_UNKNOWN_PATH,
                model=model,
                optimizer=optimizer,
                class_to_idx=train_dataset.class_to_idx,
                epoch=epoch,
                metrics=result,
                selection_metric="unknown_f1",
            )

            print(
                "  -> 新的最佳 Unknown F1 模型，已保存。"
            )

    print("-" * 80)
    print("Unknown 专项实验 4A 完成。")

    print(
        "\n实验1参考："
        "Macro-F1=0.8239，Unknown F1=0.1458"
    )

    print(
        "本实验最佳 Macro-F1："
        f"{best_macro_f1:.4f} "
        f"(Epoch {best_macro_epoch})"
    )

    print(
        "本实验最佳 Unknown F1："
        f"{best_unknown_f1:.4f} "
        f"(Epoch {best_unknown_epoch})"
    )

    print(f"\n最佳总体模型：{BEST_MACRO_PATH}")
    print(f"最佳Unknown模型：{BEST_UNKNOWN_PATH}")
    print(f"实验日志：{HISTORY_PATH}")


if __name__ == "__main__":
    main()

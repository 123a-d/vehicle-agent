"""
阶段四 · Unknown 专项实验 4G
文件名：train_known_unknown_gate.py

目标
====
前面的实验已经说明：

1. 把 unknown 直接当普通“第51类”：
   有一定作用，但 Unknown F1 仍然很低。

2. Softmax confidence threshold：
   有小幅改善，但大量 unknown 会被高置信度误认成 known。

3. Prototype cosine similarity：
   known 与 unknown 的 similarity 分布高度重叠，基本无效。

因此本实验改变问题形式：

    不再要求一个 51 类分类器同时承担：
        “这是什么车型？”
        “这是不是我认识的车型？”

而是拆成两级：

    第一级：Known/Unknown Gate（二分类）
        只回答：
            这张图像不像属于目标 50 类？

    第二级：车型分类
        只有 Gate 判断为 known 时，
        才让原来的车型分类器在 50 个目标类中选一个。

最终推理：
    image
      ↓
    ResNet18 backbone -> 512维 feature
      ↓
    Known/Unknown Gate
      ├── unknown -> 输出 unknown
      └── known   -> 原分类器在 0~49 中选车型

为什么这比“第51类”更合理
========================
unknown 不是一个统一的视觉类别，它的语义是：
    “不属于这 50 个目标车型”。

所以用一个单独的二分类 Gate 来学习“目标域 / 非目标域”的边界，
在工程上更符合实际含义。

训练 Gate 的数据
=================
Known：
    使用 50 个目标车型在 D:\\cardate\\train 中的训练图片。

Unknown：
    使用原始训练集里所有非目标车型来源，
    严格排除官方 unknown val 的 12 个来源车型。

为了同时保证：
    - 来源多样性
    - 每个来源有一定类内变化
    - 总图片数不要太夸张

默认每个非目标来源最多抽 7 张。

大约会得到：
    50 个 known 类的全部训练图
    +
    700+ 个 non-target 来源 × 最多 7 张

数量大致接近平衡。

计算方式
========
为了节省 CPU：

1. 先用当前最佳实验4A模型只提取一次 512维特征；
2. 保存 feature cache；
3. Gate 只在 512维特征上训练一个很小的 Linear 层；
4. 后续 Gate 训练非常快。

Gate 结构非常简单：
    512维 feature -> Linear(512, 1) -> sigmoid

标签：
    known   = 1
    unknown = 0

数据隔离
========
- 官方 unknown val 的 12 个来源绝不用于 Gate 训练。
- Gate 内部还会额外留出一批 non-target source IDs 做内部验证，
  用来观察 Gate 是否能泛化到“没参与 Gate 训练”的 unknown 来源。

输出
====
outputs/known_unknown_gate/
    ├── feature_cache.pt
    ├── best_gate.pth
    ├── gate_training_history.csv
    ├── official_val_per_image.csv
    ├── threshold_scan.csv
    └── analysis_summary.txt

运行
====
    python train_known_unknown_gate.py

注意
====
本实验不会修改 dataset_51，也不会修改原模型参数。
"""

from pathlib import Path
from collections import defaultdict
import csv
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_TRAIN_DIR = (
    Path(os.environ.get("VEHICLE_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))
    / "train"
)

OFFICIAL_VAL_DIR = (
    PROJECT_ROOT
    / "dataset_51"
    / "val"
)

TARGET_CLASSES_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "stage_01_02_data"
    / "candidate_classes_v1.txt"
)

UNKNOWN_VAL_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_val_sources.txt"
)

# 当前表现最好的实验 4A 模型。
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unknown_weighted"
    / "best_macro_f1.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "known_unknown_gate"
)

FEATURE_CACHE = (
    OUTPUT_DIR
    / "feature_cache.pt"
)

BEST_GATE_PATH = (
    OUTPUT_DIR
    / "best_gate.pth"
)

TRAIN_HISTORY_CSV = (
    OUTPUT_DIR
    / "gate_training_history.csv"
)

OFFICIAL_VAL_CSV = (
    OUTPUT_DIR
    / "official_val_per_image.csv"
)

THRESHOLD_SCAN_CSV = (
    OUTPUT_DIR
    / "threshold_scan.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "analysis_summary.txt"
)


# ============================================================
# 2. 参数
# ============================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

NUM_CLASSES = 51
UNKNOWN_INDEX = 50
FEATURE_DIM = 512

# 每个 non-target 来源最多抽 7 张。
UNKNOWN_IMAGES_PER_SOURCE = 7

# Gate 内部验证：
# 留出 15% 的 non-target 来源，
# 这些来源完全不参与 Gate 训练。
UNKNOWN_INTERNAL_VAL_RATIO = 0.15

# Known 类不能按“来源车型”整体留出，
# 否则 Gate 会从没见过某些目标车型。
# 因此对每个 target 类随机留 10% 图片做内部验证。
KNOWN_INTERNAL_VAL_RATIO = 0.10

# Gate 很小，在缓存特征上训练很快。
GATE_EPOCHS = 30
GATE_LR = 1e-3
GATE_WEIGHT_DECAY = 1e-4

# 最终扫描 gate threshold。
GATE_THRESHOLDS = [
    round(float(x), 2)
    for x in np.arange(
        0.05,
        1.00,
        0.05,
    )
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

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


# ============================================================
# 3. 基础函数
# ============================================================

def set_random_seed(seed: int) -> None:
    """固定随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_ids(path: Path):
    """从 txt 读取 4 位车型 ID。"""

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    ids = []

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        match = re.search(
            r"(?<!\d)(\d{4})(?!\d)",
            line,
        )

        if match:
            ids.append(
                match.group(1)
            )

    return list(
        dict.fromkeys(ids)
    )


def list_images(class_dir: Path):
    """列出车型目录中的有效图片。"""

    if not class_dir.exists():
        return []

    return sorted(
        path
        for path in class_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def build_transform():
    """保持与原模型一致的预处理。"""

    return transforms.Compose(
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


# ============================================================
# 4. 加载当前最佳视觉模型
# ============================================================

def load_backbone_model(
    device: torch.device,
):
    """
    加载实验 4A 最佳 ResNet18。

    这个模型不再训练。
    它只负责：
        - 提取 512维 feature
        - 提供原车型分类 logits
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
        NUM_CLASSES,
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

    return model, checkpoint


def extract_features(
    model,
    images,
):
    """
    提取 ResNet18 fc 前面的 512维 feature。
    """

    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)

    features = torch.flatten(
        x,
        1,
    )

    # Gate 使用归一化 feature，
    # 避免向量长度差异过度影响结果。
    features = F.normalize(
        features,
        p=2,
        dim=1,
    )

    return features


# ============================================================
# 5. 构建 Gate 的训练图片列表
# ============================================================

def build_gate_items():
    """
    构建 Gate 数据。

    known：
        50 个目标车型的全部原始 train 图片。

    unknown：
        所有非目标来源，
        排除官方 unknown val 来源，
        每来源稳定随机抽最多 7 张。

    每条 item：
        image_path
        binary_label: known=1 / unknown=0
        source_id
        source_type
    """

    target_ids = read_ids(
        TARGET_CLASSES_FILE
    )

    official_unknown_val_ids = set(
        read_ids(
            UNKNOWN_VAL_SOURCES_FILE
        )
    )

    target_set = set(
        target_ids
    )

    if len(
        target_set
    ) != 50:
        raise ValueError(
            "目标车型数量不是 50。"
        )

    items = []

    # --------------------------------------------------------
    # Known：50个目标类全部训练图片
    # --------------------------------------------------------
    for source_id in target_ids:

        images = list_images(
            ORIGINAL_TRAIN_DIR
            / source_id
        )

        if not images:
            raise ValueError(
                f"目标类 {source_id} 没有图片。"
            )

        for path in images:
            items.append(
                {
                    "image_path": str(
                        path
                    ),
                    "binary_label": 1,
                    "source_id": (
                        source_id
                    ),
                    "source_type": (
                        "known"
                    ),
                }
            )

    # --------------------------------------------------------
    # Unknown：所有其他来源，排除 official unknown val
    # --------------------------------------------------------
    unknown_source_ids = []

    for class_dir in sorted(
        ORIGINAL_TRAIN_DIR.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        source_id = class_dir.name

        if not re.fullmatch(
            r"\d{4}",
            source_id,
        ):
            continue

        if source_id in target_set:
            continue

        if (
            source_id
            in official_unknown_val_ids
        ):
            continue

        images = list_images(
            class_dir
        )

        if not images:
            continue

        source_rng = random.Random(
            RANDOM_SEED
            + int(source_id)
        )

        selected_count = min(
            UNKNOWN_IMAGES_PER_SOURCE,
            len(images),
        )

        selected = source_rng.sample(
            images,
            selected_count,
        )

        unknown_source_ids.append(
            source_id
        )

        for path in selected:
            items.append(
                {
                    "image_path": str(
                        path
                    ),
                    "binary_label": 0,
                    "source_id": (
                        source_id
                    ),
                    "source_type": (
                        "unknown"
                    ),
                }
            )

    return (
        items,
        target_ids,
        unknown_source_ids,
        official_unknown_val_ids,
    )


# ============================================================
# 6. 图片 Dataset
# ============================================================

class GateImageDataset(Dataset):
    """Gate 特征提取阶段使用的图片 Dataset。"""

    def __init__(
        self,
        items,
        transform,
    ):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(
            self.items
        )

    def __getitem__(
        self,
        index,
    ):
        item = self.items[
            index
        ]

        path = item[
            "image_path"
        ]

        with Image.open(
            path
        ) as image:

            image = image.convert(
                "RGB"
            )

            tensor = self.transform(
                image
            )

        return (
            tensor,
            int(
                item[
                    "binary_label"
                ]
            ),
            item[
                "source_id"
            ],
            path,
        )


# ============================================================
# 7. 提取并缓存 Gate 特征
# ============================================================

def extract_gate_feature_cache(
    model,
    items,
    device,
):
    """
    将所有 Gate 图片一次性转换为 512维 feature。

    CPU 最耗时的是这里。
    一旦缓存完成，后面 Gate 训练非常快。
    """

    dataset = GateImageDataset(
        items=items,
        transform=build_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_features = []
    all_labels = []
    all_source_ids = []
    all_paths = []

    with torch.inference_mode():

        for (
            images,
            labels,
            source_ids,
            paths,
        ) in loader:

            images = images.to(
                device
            )

            features = extract_features(
                model,
                images,
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.float()
            )

            all_source_ids.extend(
                list(
                    source_ids
                )
            )

            all_paths.extend(
                list(
                    paths
                )
            )

    feature_tensor = torch.cat(
        all_features,
        dim=0,
    )

    label_tensor = torch.cat(
        all_labels,
        dim=0,
    )

    cache = {
        "features": (
            feature_tensor
        ),
        "labels": (
            label_tensor
        ),
        "source_ids": (
            all_source_ids
        ),
        "image_paths": (
            all_paths
        ),
        "known_label": 1,
        "unknown_label": 0,
        "source_checkpoint": str(
            CHECKPOINT_PATH
        ),
    }

    torch.save(
        cache,
        FEATURE_CACHE,
    )

    return cache


# ============================================================
# 8. Gate 内部 train / val 切分
# ============================================================

def build_internal_split(
    cache,
):
    """
    内部切分原则：

    Unknown：
        按来源车型 ID 切。
        15% unknown source IDs 完全留作内部验证。

    Known：
        每个目标车型内部随机留 10% 图片做验证。

    这样内部验证更接近：
        “Gate 是否能识别没参与 Gate 训练的 unknown 来源？”
    """

    labels = cache[
        "labels"
    ]

    source_ids = cache[
        "source_ids"
    ]

    source_to_indices = defaultdict(
        list
    )

    for index, source_id in enumerate(
        source_ids
    ):
        source_to_indices[
            source_id
        ].append(
            index
        )

    known_sources = sorted(
        {
            source_ids[i]
            for i in range(
                len(source_ids)
            )
            if (
                int(
                    labels[
                        i
                    ].item()
                )
                == 1
            )
        }
    )

    unknown_sources = sorted(
        {
            source_ids[i]
            for i in range(
                len(source_ids)
            )
            if (
                int(
                    labels[
                        i
                    ].item()
                )
                == 0
            )
        }
    )

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Unknown：整类来源留出
    # --------------------------------------------------------
    shuffled_unknown_sources = (
        unknown_sources.copy()
    )

    rng.shuffle(
        shuffled_unknown_sources
    )

    unknown_val_source_count = max(
        1,
        int(
            len(
                shuffled_unknown_sources
            )
            * UNKNOWN_INTERNAL_VAL_RATIO
        ),
    )

    unknown_val_sources = set(
        shuffled_unknown_sources[
            :unknown_val_source_count
        ]
    )

    # --------------------------------------------------------
    # Known：每个 target 内部随机留 10%
    # --------------------------------------------------------
    train_indices = []
    val_indices = []

    for source_id in known_sources:

        indices = (
            source_to_indices[
                source_id
            ].copy()
        )

        source_rng = random.Random(
            RANDOM_SEED
            + int(source_id)
        )

        source_rng.shuffle(
            indices
        )

        val_count = max(
            1,
            int(
                len(indices)
                * KNOWN_INTERNAL_VAL_RATIO
            ),
        )

        val_indices.extend(
            indices[
                :val_count
            ]
        )

        train_indices.extend(
            indices[
                val_count:
            ]
        )

    # --------------------------------------------------------
    # Unknown 按来源整体放 train / val
    # --------------------------------------------------------
    for source_id in unknown_sources:

        indices = (
            source_to_indices[
                source_id
            ]
        )

        if (
            source_id
            in unknown_val_sources
        ):
            val_indices.extend(
                indices
            )
        else:
            train_indices.extend(
                indices
            )

    return (
        sorted(
            train_indices
        ),
        sorted(
            val_indices
        ),
        unknown_val_sources,
    )


# ============================================================
# 9. Gate 模型
# ============================================================

class KnownUnknownGate(nn.Module):
    """
    非常小的二分类 Gate：

        512维 feature
            ↓
        Linear(512, 1)
            ↓
        sigmoid
            ↓
        P(known)

    输出 logits，
    训练时使用 BCEWithLogitsLoss。
    """

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            FEATURE_DIM,
            1,
        )

    def forward(
        self,
        features,
    ):
        return self.linear(
            features
        ).squeeze(
            1
        )


# ============================================================
# 10. Gate 二分类指标
# ============================================================

def binary_metrics(
    logits,
    labels,
    threshold=0.5,
):
    """计算 Gate 自己的 known/unknown 二分类指标。"""

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >= threshold
    ).long()

    labels_int = (
        labels.long()
    )

    accuracy = (
        predictions
        == labels_int
    ).float().mean().item()

    # 这里用 macro-F1，
    # 保证 known 和 unknown 两个二分类类别同等重要。
    f1 = f1_score(
        labels_int.cpu().numpy(),
        predictions.cpu().numpy(),
        average="macro",
        zero_division=0,
    )

    return (
        float(
            accuracy
        ),
        float(
            f1
        ),
    )


# ============================================================
# 11. 训练 Gate
# ============================================================

def train_gate(
    cache,
    train_indices,
    val_indices,
    device,
):
    """
    在缓存 feature 上训练 Gate。

    由于 feature 已经提取完成，
    这里不再跑 ResNet18，
    所以 30 个 Epoch 会非常快。
    """

    features = cache[
        "features"
    ]

    labels = cache[
        "labels"
    ]

    train_x = features[
        train_indices
    ].to(
        device
    )

    train_y = labels[
        train_indices
    ].to(
        device
    )

    val_x = features[
        val_indices
    ].to(
        device
    )

    val_y = labels[
        val_indices
    ].to(
        device
    )

    gate = KnownUnknownGate().to(
        device
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=GATE_LR,
        weight_decay=(
            GATE_WEIGHT_DECAY
        ),
    )

    best_val_f1 = -1.0
    best_epoch = None

    history = []

    for epoch in range(
        1,
        GATE_EPOCHS + 1,
    ):

        gate.train()

        optimizer.zero_grad()

        train_logits = gate(
            train_x
        )

        loss = criterion(
            train_logits,
            train_y
        )

        loss.backward()

        optimizer.step()

        gate.eval()

        with torch.inference_mode():

            train_logits_eval = gate(
                train_x
            )

            val_logits = gate(
                val_x
            )

            (
                train_acc,
                train_f1,
            ) = binary_metrics(
                train_logits_eval,
                train_y,
            )

            (
                val_acc,
                val_f1,
            ) = binary_metrics(
                val_logits,
                val_y,
            )

        row = {
            "epoch": epoch,
            "train_loss": float(
                loss.item()
            ),
            "train_accuracy": (
                train_acc
            ),
            "train_macro_f1": (
                train_f1
            ),
            "val_accuracy": (
                val_acc
            ),
            "val_macro_f1": (
                val_f1
            ),
        }

        history.append(
            row
        )

        if (
            val_f1
            > best_val_f1
        ):

            best_val_f1 = (
                val_f1
            )

            best_epoch = (
                epoch
            )

            torch.save(
                {
                    "gate_state_dict": (
                        gate.state_dict()
                    ),
                    "feature_dim": (
                        FEATURE_DIM
                    ),
                    "known_label": 1,
                    "unknown_label": 0,
                    "epoch": epoch,
                    "internal_val_macro_f1": (
                        val_f1
                    ),
                    "source_checkpoint": str(
                        CHECKPOINT_PATH
                    ),
                    "config": {
                        "unknown_images_per_source": (
                            UNKNOWN_IMAGES_PER_SOURCE
                        ),
                        "unknown_internal_val_ratio": (
                            UNKNOWN_INTERNAL_VAL_RATIO
                        ),
                        "known_internal_val_ratio": (
                            KNOWN_INTERNAL_VAL_RATIO
                        ),
                        "gate_lr": (
                            GATE_LR
                        ),
                        "gate_weight_decay": (
                            GATE_WEIGHT_DECAY
                        ),
                    },
                },
                BEST_GATE_PATH,
            )

    save_csv(
        TRAIN_HISTORY_CSV,
        history,
    )

    return (
        best_val_f1,
        best_epoch,
    )


# ============================================================
# 12. 加载最佳 Gate
# ============================================================

def load_best_gate(
    device,
):
    """加载内部验证 F1 最好的 Gate。"""

    checkpoint = torch.load(
        BEST_GATE_PATH,
        map_location=device,
        weights_only=False,
    )

    gate = KnownUnknownGate().to(
        device
    )

    gate.load_state_dict(
        checkpoint[
            "gate_state_dict"
        ]
    )

    gate.eval()

    return (
        gate,
        checkpoint,
    )


# ============================================================
# 13. 在官方 val 上同时跑 Gate + 车型分类器
# ============================================================

def collect_official_val_results(
    model,
    gate,
    device,
):
    """
    对官方 dataset_51/val 做最终级联分析。

    每张图记录：

        true label

        gate_known_probability
            Gate 认为“属于目标50类”的概率

        classifier_known_pred
            原分类器只在 0~49 中选出的车型

        classifier_known_confidence
            对这个 known 车型的 softmax 概率

    最终 threshold 扫描时：

        if gate_known_probability < threshold:
            final = unknown
        else:
            final = classifier_known_pred
    """

    dataset = datasets.ImageFolder(
        root=OFFICIAL_VAL_DIR,
        transform=build_transform(),
    )

    if (
        dataset.class_to_idx.get(
            "unknown"
        )
        != UNKNOWN_INDEX
    ):
        raise ValueError(
            "官方 val unknown 索引不是50。"
        )

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

            images = images.to(
                device
            )

            # 提取 feature。
            features = extract_features(
                model,
                images,
            )

            # Gate：P(known)
            gate_logits = gate(
                features
            )

            known_probabilities = (
                torch.sigmoid(
                    gate_logits
                )
            )

            # 原车型分类器：
            # 只在目标 50 类中选择，
            # 因为 unknown 决策已经交给 Gate。
            #
            # 注意 model.fc 训练时接收的是未归一化的原始 feature，
            # 而 extract_features 当前返回了归一化 feature。
            # 因此这里重新提取一次原始 feature 会比较浪费。
            #
            # 为保持正确性，我们利用：
            # normalized feature 方向与原 feature 相同，
            # 但 fc 的 bias 会受尺度影响，所以不能直接拿归一化 feature。
            #
            # 因此这里直接跑完整 model 得到正确 logits。
            full_logits = model(
                images
            )

            known_logits = (
                full_logits[
                    :, :UNKNOWN_INDEX
                ]
            )

            known_softmax = torch.softmax(
                known_logits,
                dim=1,
            )

            (
                known_confidences,
                known_pred_indices,
            ) = known_softmax.max(
                dim=1
            )

            for i in range(
                labels.size(0)
            ):

                true_index = int(
                    labels[
                        i
                    ].item()
                )

                pred_index = int(
                    known_pred_indices[
                        i
                    ].item()
                )

                image_path = (
                    dataset.samples[
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
                            dataset.classes[
                                true_index
                            ]
                        ),
                        "gate_known_probability": float(
                            known_probabilities[
                                i
                            ].item()
                        ),
                        "known_pred_index": (
                            pred_index
                        ),
                        "known_pred_label": (
                            dataset.classes[
                                pred_index
                            ]
                        ),
                        "known_confidence": float(
                            known_confidences[
                                i
                            ].item()
                        ),
                        "is_true_unknown": int(
                            true_index
                            == UNKNOWN_INDEX
                        ),
                    }
                )

                global_index += 1

    return rows


# ============================================================
# 14. 51类最终指标
# ============================================================

def calculate_final_metrics(
    y_true,
    y_pred,
):
    """计算最终 51 类指标。"""

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
# 15. 扫描 Gate threshold
# ============================================================

def scan_gate_thresholds(
    rows,
):
    """
    Gate threshold 的含义：

        gate_known_probability >= threshold
            -> 接受为 known
            -> 使用车型分类结果

        gate_known_probability < threshold
            -> 输出 unknown

    threshold 越高：
        Gate 越谨慎，
        越容易输出 unknown。
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
        GATE_THRESHOLDS
    ):

        y_pred = []

        for row in rows:

            if (
                row[
                    "gate_known_probability"
                ]
                < threshold
            ):

                final_pred = (
                    UNKNOWN_INDEX
                )

            else:

                final_pred = (
                    row[
                        "known_pred_index"
                    ]
                )

            y_pred.append(
                final_pred
            )

        metrics = (
            calculate_final_metrics(
                y_true,
                y_pred,
            )
        )

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
                "gate_threshold": (
                    threshold
                ),
                **metrics,
                "known_rejected_count": (
                    known_rejected_count
                ),
                "known_rejection_rate": (
                    known_rejection_rate
                ),
            }
        )

    return scan_rows


# ============================================================
# 16. CSV 保存
# ============================================================

def save_csv(
    path,
    rows,
):
    """保存 CSV。"""

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
# 17. 总结
# ============================================================

def write_summary(
    gate_checkpoint,
    rows,
    threshold_rows,
    gate_feature_count,
    internal_unknown_val_sources,
):
    """自动找出最值得看的 Gate operating points。"""

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

    # 工程上限制 known rejection <= 5%
    feasible_5 = [
        row
        for row in threshold_rows
        if (
            row[
                "known_rejection_rate"
            ]
            <= 0.05
        )
    ]

    best_under_5 = (
        max(
            feasible_5,
            key=lambda row: (
                row[
                    "unknown_f1"
                ]
            ),
        )
        if feasible_5
        else None
    )

    # 再给一个稍宽松的 <=10%
    feasible_10 = [
        row
        for row in threshold_rows
        if (
            row[
                "known_rejection_rate"
            ]
            <= 0.10
        )
    ]

    best_under_10 = (
        max(
            feasible_10,
            key=lambda row: (
                row[
                    "unknown_f1"
                ]
            ),
        )
        if feasible_10
        else None
    )

    true_unknown_gate_probs = [
        row[
            "gate_known_probability"
        ]
        for row in rows
        if (
            row[
                "is_true_unknown"
            ]
            == 1
        )
    ]

    true_known_gate_probs = [
        row[
            "gate_known_probability"
        ]
        for row in rows
        if (
            row[
                "is_true_unknown"
            ]
            == 0
        )
    ]

    known_mean = float(
        np.mean(
            true_known_gate_probs
        )
    )

    known_median = float(
        np.median(
            true_known_gate_probs
        )
    )

    unknown_mean = float(
        np.mean(
            true_unknown_gate_probs
        )
    )

    unknown_median = float(
        np.median(
            true_unknown_gate_probs
        )
    )

    lines = [
        "阶段四 · Known/Unknown Gate 实验",
        "=" * 70,
        "",
        f"Backbone: {CHECKPOINT_PATH}",
        (
            "Gate best internal val Macro-F1: "
            f"{gate_checkpoint['internal_val_macro_f1']:.4f}"
        ),
        (
            "Gate best epoch: "
            f"{gate_checkpoint['epoch']}"
        ),
        (
            "Gate feature count: "
            f"{gate_feature_count}"
        ),
        (
            "Internal held-out unknown sources: "
            f"{len(internal_unknown_val_sources)}"
        ),
        "",
        "官方 val Gate known-probability 分布:",
        (
            "True known: "
            f"mean={known_mean:.4f}, "
            f"median={known_median:.4f}"
        ),
        (
            "True unknown: "
            f"mean={unknown_mean:.4f}, "
            f"median={unknown_median:.4f}"
        ),
        "",
        "Macro-F1 最佳:",
        str(
            best_macro
        ),
        "",
        "Unknown F1 最佳:",
        str(
            best_unknown
        ),
        "",
        "Known rejection <=5% 最佳:",
        str(
            best_under_5
        ),
        "",
        "Known rejection <=10% 最佳:",
        str(
            best_under_10
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return (
        best_macro,
        best_unknown,
        best_under_5,
        best_under_10,
        known_mean,
        known_median,
        unknown_mean,
        unknown_median,
    )


# ============================================================
# 18. 主程序
# ============================================================

def main():

    print("=" * 88)

    print(
        "阶段四 · Unknown实验4G："
        "Two-stage Known/Unknown Gate"
    )

    print("=" * 88)

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

    # --------------------------------------------------------
    # 加载 backbone
    # --------------------------------------------------------

    model, model_checkpoint = (
        load_backbone_model(
            device
        )
    )

    # --------------------------------------------------------
    # 构建 Gate 数据
    # --------------------------------------------------------

    (
        items,
        target_ids,
        unknown_source_ids,
        official_unknown_val_ids,
    ) = build_gate_items()

    known_count = sum(
        1
        for item in items
        if (
            item[
                "binary_label"
            ]
            == 1
        )
    )

    unknown_count = (
        len(items)
        - known_count
    )

    print(
        f"\nGate known 图片：{known_count}"
    )

    print(
        f"Gate unknown 图片：{unknown_count}"
    )

    print(
        "Gate unknown 来源数："
        f"{len(unknown_source_ids)}"
    )

    print(
        "官方 unknown val 来源已排除："
        f"{len(official_unknown_val_ids)}"
    )

    overlap = (
        set(
            unknown_source_ids
        )
        & set(
            official_unknown_val_ids
        )
    )

    print(
        f"来源交集：{overlap}"
    )

    if overlap:
        raise ValueError(
            "Gate unknown train 与官方 val 来源泄漏。"
        )

    # --------------------------------------------------------
    # 提取 feature cache
    # --------------------------------------------------------

    print(
        "\n正在提取 Gate 训练特征..."
    )

    start = time.perf_counter()

    cache = extract_gate_feature_cache(
        model=model,
        items=items,
        device=device,
    )

    print(
        "Feature cache shape："
        f"{tuple(cache['features'].shape)}"
    )

    print(
        "特征提取耗时："
        f"{time.perf_counter() - start:.1f}s"
    )

    # --------------------------------------------------------
    # 内部 train / val split
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
        internal_unknown_val_sources,
    ) = build_internal_split(
        cache
    )

    print(
        "\nGate内部训练样本："
        f"{len(train_indices)}"
    )

    print(
        "Gate内部验证样本："
        f"{len(val_indices)}"
    )

    print(
        "Gate内部留出的 unknown 来源："
        f"{len(internal_unknown_val_sources)}"
    )

    # --------------------------------------------------------
    # 训练 Gate
    # --------------------------------------------------------

    print(
        "\n开始训练二分类 Gate..."
    )

    (
        best_internal_f1,
        best_gate_epoch,
    ) = train_gate(
        cache=cache,
        train_indices=train_indices,
        val_indices=val_indices,
        device=device,
    )

    print(
        "Gate内部最佳 Macro-F1："
        f"{best_internal_f1:.4f}"
    )

    print(
        f"最佳 Epoch：{best_gate_epoch}"
    )

    # --------------------------------------------------------
    # 官方 val 级联评估
    # --------------------------------------------------------

    gate, gate_checkpoint = (
        load_best_gate(
            device
        )
    )

    print(
        "\n正在评估官方 val..."
    )

    official_rows = (
        collect_official_val_results(
            model=model,
            gate=gate,
            device=device,
        )
    )

    save_csv(
        OFFICIAL_VAL_CSV,
        official_rows,
    )

    threshold_rows = (
        scan_gate_thresholds(
            official_rows
        )
    )

    save_csv(
        THRESHOLD_SCAN_CSV,
        threshold_rows,
    )

    (
        best_macro,
        best_unknown,
        best_under_5,
        best_under_10,
        known_mean,
        known_median,
        unknown_mean,
        unknown_median,
    ) = write_summary(
        gate_checkpoint=gate_checkpoint,
        rows=official_rows,
        threshold_rows=threshold_rows,
        gate_feature_count=len(
            items
        ),
        internal_unknown_val_sources=(
            internal_unknown_val_sources
        ),
    )

    # --------------------------------------------------------
    # 终端重点输出
    # --------------------------------------------------------

    print(
        "\n" + "-" * 88
    )

    print(
        "官方 val Gate known-probability："
    )

    print(
        "True known："
        f"mean={known_mean:.4f}, "
        f"median={known_median:.4f}"
    )

    print(
        "True unknown："
        f"mean={unknown_mean:.4f}, "
        f"median={unknown_median:.4f}"
    )

    print(
        "\nMacro-F1 最佳 Gate threshold："
    )

    print(
        f"threshold = {best_macro['gate_threshold']:.2f}"
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
        "\nUnknown F1 最佳 Gate threshold："
    )

    print(
        f"threshold = {best_unknown['gate_threshold']:.2f}"
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
        best_under_5
        is not None
    ):

        print(
            "\nKnown rejection <=5% 最佳："
        )

        print(
            f"threshold = {best_under_5['gate_threshold']:.2f}"
        )

        print(
            f"Macro-F1 = {best_under_5['macro_f1']:.4f}"
        )

        print(
            f"Known-F1 = {best_under_5['known_macro_f1']:.4f}"
        )

        print(
            f"Unknown F1 = {best_under_5['unknown_f1']:.4f}"
        )

        print(
            "Unknown P/R = "
            f"{best_under_5['unknown_precision']:.4f} / "
            f"{best_under_5['unknown_recall']:.4f}"
        )

        print(
            "Known rejection rate = "
            f"{best_under_5['known_rejection_rate']:.4f}"
        )

    if (
        best_under_10
        is not None
    ):

        print(
            "\nKnown rejection <=10% 最佳："
        )

        print(
            f"threshold = {best_under_10['gate_threshold']:.2f}"
        )

        print(
            f"Macro-F1 = {best_under_10['macro_f1']:.4f}"
        )

        print(
            f"Known-F1 = {best_under_10['known_macro_f1']:.4f}"
        )

        print(
            f"Unknown F1 = {best_under_10['unknown_f1']:.4f}"
        )

        print(
            "Unknown P/R = "
            f"{best_under_10['unknown_precision']:.4f} / "
            f"{best_under_10['unknown_recall']:.4f}"
        )

        print(
            "Known rejection rate = "
            f"{best_under_10['known_rejection_rate']:.4f}"
        )

    print(
        "\n实验4A参考："
    )

    print(
        "Macro-F1 = 0.8417"
    )

    print(
        "Unknown F1 = 0.2056"
    )

    print(
        "\n输出："
    )

    print(
        f"- {FEATURE_CACHE}"
    )

    print(
        f"- {BEST_GATE_PATH}"
    )

    print(
        f"- {TRAIN_HISTORY_CSV}"
    )

    print(
        f"- {OFFICIAL_VAL_CSV}"
    )

    print(
        f"- {THRESHOLD_SCAN_CSV}"
    )

    print(
        f"- {SUMMARY_TXT}"
    )


if __name__ == "__main__":
    main()

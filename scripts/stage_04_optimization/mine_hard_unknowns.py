"""
阶段四 · Unknown 专项实验 4D：Hard Negative Mining
文件名：mine_hard_unknowns.py

目标
====
使用当前表现最好的实验 4A 模型：

    outputs/unknown_weighted/best_macro_f1.pth

去扫描原始训练集中的大量“非目标车型”，找出：

    明明不属于目标 50 类，
    但模型却高置信度地把它认成某个 known 类

的来源车型。

这种样本就是 Hard Negative（困难负样本）。

为什么做这一步
============
实验 4C 证明：

    随机把 unknown 来源从 20 类扩到 100 类，
    但每类只有 2 张，并没有改善 unknown，
    反而使 Unknown F1 从 0.2056 降到 0.1124。

因此下一步不再“随机增加 unknown”，而是有针对性地寻找
最容易欺骗当前模型的非目标车型。

本脚本只做分析，不训练，也不复制图片。

扫描策略
========
1. 从 D:\\cardate\\train 中寻找非目标车型；
2. 排除：
       - 50 个目标车型
       - unknown val 的来源车型
       - 实验 4A 已经训练过的 20 个旧 unknown 来源
3. 每个候选车型固定随机抽 3 张图片；
4. 使用实验 4A 最佳模型推理；
5. 对每张图片计算：
       - 模型原始 Top-1 类别
       - Top-1 confidence
       - 50 个 known 类中的最大 confidence
       - 是否被模型预测成 unknown
6. 按“平均最大 known confidence”从高到低对来源车型排序。

重点理解
========
这里的 hard_score 不是新的神经网络指标。

hard_score = 一个非目标车型抽样图片的
             mean(max known confidence)

例如：
    某非目标车型 3 张图的最大 known confidence：
        0.95, 0.91, 0.88

    hard_score = 0.9133

说明模型非常容易把这种车自信地认成目标 50 类，
它就特别适合作为后续 unknown 训练的困难负样本。

运行
====
    python mine_hard_unknowns.py

输出
====
outputs/hard_negative_mining/
    ├── hard_negative_per_image.csv
    ├── hard_negative_source_ranking.csv
    ├── top_80_hard_source_ids.txt
    └── mining_summary.txt
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_TRAIN_DIR = (
    Path(os.environ.get("VEHICLE_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))
    / "train"
)

TARGET_CLASSES_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "stage_01_02_data"
    / "candidate_classes_v1.txt"
)

OLD_UNKNOWN_TRAIN_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_train_sources.txt"
)

UNKNOWN_VAL_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_val_sources.txt"
)

# 当前阶段四表现最好的模型：
# 实验 4A，unknown weight=2。
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unknown_weighted"
    / "best_macro_f1.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hard_negative_mining"
)

PER_IMAGE_CSV = (
    OUTPUT_DIR
    / "hard_negative_per_image.csv"
)

SOURCE_RANKING_CSV = (
    OUTPUT_DIR
    / "hard_negative_source_ranking.csv"
)

TOP_SOURCE_IDS_FILE = (
    OUTPUT_DIR
    / "top_80_hard_source_ids.txt"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "mining_summary.txt"
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

# 每个非目标来源车型只随机抽 3 张。
# 约 700 多个来源时，总推理图片约 2000 多张，
# CPU 也可以接受。
SAMPLES_PER_SOURCE = 3

# 先输出排名最靠前的 80 个来源 ID，
# 方便后面构建 Hard Negative v3 数据集。
TOP_K_SOURCES = 80

# 为了帮助理解结果，额外统计几个高置信度比例。
CONFIDENCE_LEVELS = [
    0.50,
    0.70,
    0.90,
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
# 3. 基础工具函数
# ============================================================

def set_random_seed(seed: int) -> None:
    """固定随机种子，使抽样结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_ids(path: Path):
    """
    从 txt 中读取 4 位车型 ID。

    兼容：
        0009
        0009  其他文字
    """

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    result = []

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        match = re.search(
            r"(?<!\d)(\d{4})(?!\d)",
            line,
        )

        if match:
            result.append(
                match.group(1)
            )

    return list(
        dict.fromkeys(result)
    )


def list_images(class_dir: Path):
    """列出一个车型目录中的有效图片文件。"""

    return sorted(
        path
        for path in class_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ============================================================
# 4. 构造要扫描的图片列表
# ============================================================

def build_scan_items():
    """
    构造 Hard Negative Mining 的候选图片列表。

    排除：
        1. 50 个 target 类
        2. unknown val 的来源
        3. 实验 4A 已经训练过的 20 个 old unknown 来源

    为什么排除旧 20 类？
        当前扫描模型本身已经把这 20 类作为 unknown 训练过，
        它们不再代表“模型从未作为 unknown 见过的困难来源”。

    返回：
        scan_items:
            [
                {
                    "image_path": ...,
                    "source_id": ...
                },
                ...
            ]

        candidate_source_ids:
            实际参与扫描的来源车型 ID
    """

    if not ORIGINAL_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"找不到原始训练集：{ORIGINAL_TRAIN_DIR}"
        )

    target_ids = set(
        read_ids(
            TARGET_CLASSES_FILE
        )
    )

    old_unknown_ids = set(
        read_ids(
            OLD_UNKNOWN_TRAIN_SOURCES_FILE
        )
    )

    val_unknown_ids = set(
        read_ids(
            UNKNOWN_VAL_SOURCES_FILE
        )
    )

    excluded_ids = (
        target_ids
        | old_unknown_ids
        | val_unknown_ids
    )

    candidate_source_ids = []
    scan_items = []

    # 为了让每个车型的随机抽样彼此稳定，
    # 每个 source_id 使用独立的随机数生成器。
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

        if source_id in excluded_ids:
            continue

        images = list_images(
            class_dir
        )

        if len(images) == 0:
            continue

        sample_count = min(
            SAMPLES_PER_SOURCE,
            len(images),
        )

        # source_id 被加入 seed，
        # 使每个来源的抽样稳定，同时又不是总取文件夹前 3 张。
        source_rng = random.Random(
            RANDOM_SEED
            + int(source_id)
        )

        selected = source_rng.sample(
            images,
            sample_count,
        )

        candidate_source_ids.append(
            source_id
        )

        for path in selected:
            scan_items.append(
                {
                    "image_path": str(
                        path
                    ),
                    "source_id": (
                        source_id
                    ),
                }
            )

    return (
        scan_items,
        candidate_source_ids,
        target_ids,
        old_unknown_ids,
        val_unknown_ids,
    )


# ============================================================
# 5. 自定义 Dataset
# ============================================================

class MiningDataset(Dataset):
    """
    这是一个非常简单的自定义 Dataset。

    每次 __getitem__：
        读取一张图片
        -> RGB
        -> transform
        -> 返回 tensor、source_id、image_path

    和训练 Dataset 不同：
        这里不需要真实的 0~50 分类 label，
        因为我们已经知道这些图片全部来自“非目标车型”。
    """

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

        image_path = item[
            "image_path"
        ]

        with Image.open(
            image_path
        ) as image:
            image = image.convert(
                "RGB"
            )

            image_tensor = (
                self.transform(
                    image
                )
            )

        return (
            image_tensor,
            item["source_id"],
            image_path,
        )


# ============================================================
# 6. 加载当前最佳模型
# ============================================================

def load_model(
    device: torch.device,
):
    """
    加载实验 4A 当前最佳模型。

    这里只做推理：
        model.eval()
        torch.inference_mode()
    不会修改任何模型参数。
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

    if (
        checkpoint.get(
            "num_classes"
        )
        != NUM_CLASSES
    ):
        raise ValueError(
            "checkpoint 类别数不是 51。"
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

    # checkpoint 中保存了训练时的类别映射。
    class_to_idx = checkpoint.get(
        "class_to_idx"
    )

    if not class_to_idx:
        raise ValueError(
            "checkpoint 中没有 class_to_idx。"
        )

    if (
        class_to_idx.get(
            "unknown"
        )
        != UNKNOWN_INDEX
    ):
        raise ValueError(
            "checkpoint 中 unknown 索引不是 50。"
        )

    idx_to_class = {
        index: class_name
        for class_name, index
        in class_to_idx.items()
    }

    return (
        model,
        checkpoint,
        idx_to_class,
    )


# ============================================================
# 7. 扫描非目标车型
# ============================================================

def run_mining(
    model,
    dataset,
    idx_to_class,
    device,
):
    """
    对候选图片做推理。

    这里最关键的两个 confidence：

    1. top1_confidence
       在全部 51 类中，模型最高的 softmax 概率。

    2. max_known_confidence
       只在 known 0~49 中找最大概率。

    为什么 Hard Negative 排名使用 max_known_confidence？
        因为我们真正关心的是：

        “这张真实非目标图片，
         模型有多自信地想把它塞进某个 known 类？”

    即使模型最终 Top-1 已经是 unknown，
    max_known_confidence 仍能告诉我们 known 类对它有多强的吸引力。
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    per_image_rows = []

    with torch.inference_mode():
        for (
            images,
            source_ids,
            image_paths,
        ) in loader:

            images = images.to(
                device
            )

            logits = model(
                images
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            # ------------------------------------------------
            # 全 51 类 Top-1
            # ------------------------------------------------
            top1_confidences, top1_indices = (
                probabilities.max(
                    dim=1
                )
            )

            # ------------------------------------------------
            # 只看 50 个 known 类
            # ------------------------------------------------
            known_probabilities = (
                probabilities[
                    :, :UNKNOWN_INDEX
                ]
            )

            (
                max_known_confidences,
                max_known_indices,
            ) = known_probabilities.max(
                dim=1
            )

            unknown_confidences = (
                probabilities[
                    :, UNKNOWN_INDEX
                ]
            )

            batch_size = (
                images.size(0)
            )

            for i in range(
                batch_size
            ):
                top1_index = int(
                    top1_indices[
                        i
                    ].item()
                )

                max_known_index = int(
                    max_known_indices[
                        i
                    ].item()
                )

                top1_conf = float(
                    top1_confidences[
                        i
                    ].item()
                )

                max_known_conf = float(
                    max_known_confidences[
                        i
                    ].item()
                )

                unknown_conf = float(
                    unknown_confidences[
                        i
                    ].item()
                )

                per_image_rows.append(
                    {
                        "source_id": (
                            source_ids[i]
                        ),

                        "image_path": (
                            image_paths[i]
                        ),

                        "top1_index": (
                            top1_index
                        ),

                        "top1_label": (
                            idx_to_class[
                                top1_index
                            ]
                        ),

                        "top1_confidence": (
                            top1_conf
                        ),

                        "predicted_unknown": int(
                            top1_index
                            == UNKNOWN_INDEX
                        ),

                        "max_known_index": (
                            max_known_index
                        ),

                        "max_known_label": (
                            idx_to_class[
                                max_known_index
                            ]
                        ),

                        "max_known_confidence": (
                            max_known_conf
                        ),

                        "unknown_confidence": (
                            unknown_conf
                        ),
                    }
                )

    return per_image_rows


# ============================================================
# 8. 按来源车型聚合并排名
# ============================================================

def build_source_ranking(
    per_image_rows,
):
    """
    把逐图片结果聚合成“每个非目标车型”的统计。

    hard_score：
        mean_max_known_confidence

    分数越高：
        这个非目标车型越容易被模型高置信度当成 known，
        越值得作为 Hard Negative。
    """

    grouped = defaultdict(
        list
    )

    for row in per_image_rows:
        grouped[
            row["source_id"]
        ].append(
            row
        )

    ranking_rows = []

    for (
        source_id,
        rows,
    ) in grouped.items():

        known_confidences = np.asarray(
            [
                row[
                    "max_known_confidence"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        top1_known_rows = [
            row
            for row in rows
            if (
                row[
                    "predicted_unknown"
                ]
                == 0
            )
        ]

        unknown_pred_rate = (
            sum(
                row[
                    "predicted_unknown"
                ]
                for row in rows
            )
            / len(rows)
        )

        # 看它最常被误认为哪个目标车型。
        known_pred_counter = Counter(
            row[
                "max_known_label"
            ]
            for row in rows
        )

        (
            most_common_known_label,
            most_common_count,
        ) = known_pred_counter.most_common(
            1
        )[0]

        result = {
            "source_id": source_id,

            "scanned_images": (
                len(rows)
            ),

            # Hard Negative 排名的主指标：
            "hard_score": float(
                np.mean(
                    known_confidences
                )
            ),

            "mean_max_known_confidence": float(
                np.mean(
                    known_confidences
                )
            ),

            "median_max_known_confidence": float(
                np.median(
                    known_confidences
                )
            ),

            "max_known_confidence": float(
                np.max(
                    known_confidences
                )
            ),

            "min_max_known_confidence": float(
                np.min(
                    known_confidences
                )
            ),

            "unknown_pred_rate": float(
                unknown_pred_rate
            ),

            "top1_predicted_known_rate": float(
                len(
                    top1_known_rows
                )
                / len(rows)
            ),

            "most_confused_known_label": (
                most_common_known_label
            ),

            "most_confused_count": (
                most_common_count
            ),
        }

        # 额外统计：
        # 这个车型抽到的图片中，有多少比例让模型对 known
        # 产生 >=0.5 / 0.7 / 0.9 的置信度。
        for level in CONFIDENCE_LEVELS:
            key = (
                "known_conf_"
                f"ge_{level:.2f}_rate"
            )

            result[key] = float(
                np.mean(
                    known_confidences
                    >= level
                )
            )

        ranking_rows.append(
            result
        )

    # hard_score 从高到低。
    ranking_rows.sort(
        key=lambda row: (
            row["hard_score"],
            row[
                "max_known_confidence"
            ],
        ),
        reverse=True,
    )

    # 给每个来源增加排名。
    for rank, row in enumerate(
        ranking_rows,
        start=1,
    ):
        row["rank"] = rank

    # 把 rank 放到第一列只是为了 CSV 更好看。
    ranking_rows = [
        {
            "rank": row["rank"],
            **{
                key: value
                for key, value
                in row.items()
                if key != "rank"
            },
        }
        for row in ranking_rows
    ]

    return ranking_rows


# ============================================================
# 9. 保存结果
# ============================================================

def save_csv(
    path: Path,
    rows,
):
    """保存 CSV。"""

    if not rows:
        raise ValueError(
            f"没有内容可保存：{path}"
        )

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


def save_top_source_ids(
    ranking_rows,
):
    """保存 Hard Negative 排名前 80 的来源车型 ID。"""

    top_rows = ranking_rows[
        :TOP_K_SOURCES
    ]

    TOP_SOURCE_IDS_FILE.write_text(
        "\n".join(
            row["source_id"]
            for row in top_rows
        )
        + "\n",
        encoding="utf-8",
    )

    return top_rows


def build_summary(
    checkpoint,
    candidate_source_count,
    scan_image_count,
    ranking_rows,
    top_rows,
):
    """生成终端和 txt 都能阅读的总结。"""

    # 高风险来源：平均 known confidence >= 0.7
    source_mean_ge_07 = sum(
        1
        for row in ranking_rows
        if (
            row[
                "mean_max_known_confidence"
            ]
            >= 0.70
        )
    )

    # 极高风险来源：平均 known confidence >= 0.9
    source_mean_ge_09 = sum(
        1
        for row in ranking_rows
        if (
            row[
                "mean_max_known_confidence"
            ]
            >= 0.90
        )
    )

    top10 = ranking_rows[
        :10
    ]

    lines = [
        "阶段四 Hard Negative Mining 总结",
        "=" * 68,
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        (
            "Checkpoint Epoch: "
            f"{checkpoint.get('epoch')}"
        ),
        "",
        (
            "候选非目标来源车型数: "
            f"{candidate_source_count}"
        ),
        (
            "实际扫描图片数: "
            f"{scan_image_count}"
        ),
        (
            "每来源最多抽样: "
            f"{SAMPLES_PER_SOURCE} 张"
        ),
        "",
        (
            "平均 max-known confidence >= 0.70 的来源数: "
            f"{source_mean_ge_07}"
        ),
        (
            "平均 max-known confidence >= 0.90 的来源数: "
            f"{source_mean_ge_09}"
        ),
        "",
        "Top 10 Hard Negative 来源:",
    ]

    for row in top10:
        lines.append(
            (
                f"#{row['rank']:02d} "
                f"source={row['source_id']} | "
                f"hard_score={row['hard_score']:.4f} | "
                f"max={row['max_known_confidence']:.4f} | "
                "most_confused="
                f"{row['most_confused_known_label']} | "
                "unknown_pred_rate="
                f"{row['unknown_pred_rate']:.2f}"
            )
        )

    lines.extend(
        [
            "",
            (
                f"已输出 Top {len(top_rows)} "
                "Hard Negative 来源 ID。"
            ),
            "",
            "注意：",
            "本脚本只负责挖掘和排名，尚未把这些来源加入训练。",
            "下一步应先查看排名结果，再构建 hard-negative unknown v3。",
        ]
    )

    summary_text = "\n".join(
        lines
    )

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8",
    )

    return (
        summary_text,
        source_mean_ge_07,
        source_mean_ge_09,
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():
    print("=" * 82)
    print(
        "阶段四 · Unknown实验4D："
        "Hard Negative Mining"
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
        f"当前模型：{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # 构造待扫描数据
    # --------------------------------------------------------
    (
        scan_items,
        candidate_source_ids,
        target_ids,
        old_unknown_ids,
        val_unknown_ids,
    ) = build_scan_items()

    print(
        f"\n目标车型排除数：{len(target_ids)}"
    )

    print(
        "旧 unknown train 排除数："
        f"{len(old_unknown_ids)}"
    )

    print(
        "unknown val 排除数："
        f"{len(val_unknown_ids)}"
    )

    print(
        "实际候选非目标来源："
        f"{len(candidate_source_ids)}"
    )

    print(
        "实际待扫描图片："
        f"{len(scan_items)}"
    )

    # --------------------------------------------------------
    # 推理预处理
    # --------------------------------------------------------
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

    dataset = MiningDataset(
        items=scan_items,
        transform=transform,
    )

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------
    (
        model,
        checkpoint,
        idx_to_class,
    ) = load_model(
        device
    )

    print(
        "\n开始扫描 Hard Negatives..."
    )

    # --------------------------------------------------------
    # 扫描
    # --------------------------------------------------------
    per_image_rows = run_mining(
        model=model,
        dataset=dataset,
        idx_to_class=idx_to_class,
        device=device,
    )

    # --------------------------------------------------------
    # 聚合排名
    # --------------------------------------------------------
    ranking_rows = (
        build_source_ranking(
            per_image_rows
        )
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    save_csv(
        PER_IMAGE_CSV,
        per_image_rows,
    )

    save_csv(
        SOURCE_RANKING_CSV,
        ranking_rows,
    )

    top_rows = (
        save_top_source_ids(
            ranking_rows
        )
    )

    (
        summary_text,
        source_mean_ge_07,
        source_mean_ge_09,
    ) = build_summary(
        checkpoint=checkpoint,
        candidate_source_count=len(
            candidate_source_ids
        ),
        scan_image_count=len(
            scan_items
        ),
        ranking_rows=ranking_rows,
        top_rows=top_rows,
    )

    # --------------------------------------------------------
    # 终端重点输出
    # --------------------------------------------------------
    print(
        "\n" + "-" * 82
    )

    print(
        "Hard Negative Mining 完成。"
    )

    print(
        "\n平均 known confidence >= 0.70 的来源："
        f"{source_mean_ge_07}"
    )

    print(
        "平均 known confidence >= 0.90 的来源："
        f"{source_mean_ge_09}"
    )

    print(
        "\nTop 10 最困难非目标车型："
    )

    for row in ranking_rows[
        :10
    ]:
        print(
            f"#{row['rank']:02d} | "
            f"source {row['source_id']} | "
            f"hard_score {row['hard_score']:.4f} | "
            f"max {row['max_known_confidence']:.4f} | "
            "最常误认 "
            f"{row['most_confused_known_label']} | "
            "unknown预测率 "
            f"{row['unknown_pred_rate']:.2f}"
        )

    print(
        "\n输出文件："
    )

    print(
        f"- {PER_IMAGE_CSV}"
    )

    print(
        f"- {SOURCE_RANKING_CSV}"
    )

    print(
        f"- {TOP_SOURCE_IDS_FILE}"
    )

    print(
        f"- {SUMMARY_FILE}"
    )

    print(
        "\n下一步：先分析 Top Hard Negative，"
        "再构建 targeted unknown v3，暂时不要直接训练。"
    )


if __name__ == "__main__":
    main()

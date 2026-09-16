"""
阶段四 · Unknown 专项实验 4E：构建 Hard Negative v3 数据集
文件名：build_unknown_train_v3_hard.py

实验目标
========
基于实验 4D 的 Hard Negative Mining 结果，
构建一个更有针对性的 unknown 训练集 v3。

v3 总图片数量仍然保持 200 张，
以便和实验 4A 做控制变量比较。

组成：
    1. Top 40 Hard Negative 来源 × 每类 4 张 = 160 张
    2. 旧 v1 的 20 个普通 unknown 来源 × 每类 2 张 = 40 张

总计：
    160 + 40 = 200 张

为什么这样设计
==============
实验 4C 说明：

    100 个来源 × 每类 2 张

虽然来源更多，但每种车型图片太少，
Unknown F1 反而下降。

所以 v3 不再追求“来源越多越好”，而是：

    有针对性地选择最容易骗过模型的 Hard Negative
    +
    每个 Hard Negative 保留更多图片
    +
    保留一部分普通 unknown，避免数据过度集中在极端难例

输入
====
1. Hard Negative 排名：
    outputs/hard_negative_mining/hard_negative_source_ranking.csv

2. 旧 unknown 来源：
    unknown_train_sources.txt

3. unknown val 来源：
    unknown_val_sources.txt

4. 50 个目标车型：
    candidate_classes_v1.txt

5. 原始训练集：
    D:\\cardate\\train

输出
====
unknown_train_v3_hard/unknown/
    共 200 张

unknown_train_sources_v3_hard.txt
unknown_train_v3_hard_manifest.csv
unknown_train_v3_hard_plan.csv

注意
====
本脚本只创建新数据，
不会修改 dataset_51，
也不会删除旧 unknown_train_v2。
"""

from pathlib import Path
import csv
import os
import random
import re
import shutil


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

OLD_UNKNOWN_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_train_sources.txt"
)

UNKNOWN_VAL_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_val_sources.txt"
)

HARD_RANKING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "hard_negative_mining"
    / "hard_negative_source_ranking.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "unknown_train_v3_hard"
)

OUTPUT_UNKNOWN_DIR = (
    OUTPUT_ROOT
    / "unknown"
)

OUTPUT_SOURCES_FILE = (
    PROJECT_ROOT
    / "unknown_train_sources_v3_hard.txt"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "unknown_train_v3_hard_manifest.csv"
)

OUTPUT_PLAN = (
    PROJECT_ROOT
    / "unknown_train_v3_hard_plan.csv"
)


# ============================================================
# 2. 实验参数
# ============================================================

RANDOM_SEED = 42

# v3 核心组成：
TOP_HARD_SOURCE_COUNT = 40
HARD_IMAGES_PER_SOURCE = 4

OLD_SOURCE_COUNT = 20
OLD_IMAGES_PER_SOURCE = 2

EXPECTED_TOTAL_IMAGES = (
    TOP_HARD_SOURCE_COUNT
    * HARD_IMAGES_PER_SOURCE
    +
    OLD_SOURCE_COUNT
    * OLD_IMAGES_PER_SOURCE
)

EXPECTED_TOTAL_SOURCES = (
    TOP_HARD_SOURCE_COUNT
    +
    OLD_SOURCE_COUNT
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 3. 工具函数
# ============================================================

def read_ids(path: Path):
    """
    从 txt 中读取 4 位车型 ID。

    支持每行：
        0009

    也容忍：
        0009  其他说明
    """

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

        if line.startswith("#"):
            continue

        match = re.search(
            r"(?<!\d)(\d{4})(?!\d)",
            line,
        )

        if match:
            ids.append(
                match.group(1)
            )

    # 去重并保持原顺序。
    return list(
        dict.fromkeys(ids)
    )


def list_images(class_dir: Path):
    """列出一个原始车型目录中的所有有效图片。"""

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


def read_top_hard_sources():
    """
    从 hard_negative_source_ranking.csv 中读取排名最靠前的 Hard Sources。

    文件是实验 4D 自动生成的。

    我们读取前 TOP_HARD_SOURCE_COUNT 个有效来源。
    """

    if not HARD_RANKING_FILE.exists():
        raise FileNotFoundError(
            "找不到 Hard Negative 排名："
            f"{HARD_RANKING_FILE}"
        )

    rows = []

    with HARD_RANKING_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(
            file
        )

        required_columns = {
            "rank",
            "source_id",
            "hard_score",
            "most_confused_known_label",
        }

        missing_columns = (
            required_columns
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing_columns:
            raise ValueError(
                "Hard Negative CSV 缺少字段："
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            source_id = (
                row[
                    "source_id"
                ].strip()
            )

            if not re.fullmatch(
                r"\d{4}",
                source_id,
            ):
                continue

            rows.append(
                row
            )

    if (
        len(rows)
        < TOP_HARD_SOURCE_COUNT
    ):
        raise ValueError(
            "Hard Negative 来源数量不足 "
            f"{TOP_HARD_SOURCE_COUNT} 个。"
        )

    return rows[
        :TOP_HARD_SOURCE_COUNT
    ]


def prepare_output_directory():
    """
    重建 v3 输出目录。

    只操作：
        unknown_train_v3_hard

    不会触碰 dataset_51 或旧 v1/v2。
    """

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_UNKNOWN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def stable_sample_images(
    source_id: str,
    sample_count: int,
):
    """
    从某个来源车型中稳定随机抽取图片。

    每个 source_id 使用独立 seed，
    保证重复运行时抽到相同图片。

    这样实验具有可复现性。
    """

    class_dir = (
        ORIGINAL_TRAIN_DIR
        / source_id
    )

    images = list_images(
        class_dir
    )

    if (
        len(images)
        < sample_count
    ):
        raise ValueError(
            f"来源 {source_id} 只有 {len(images)} 张图片，"
            f"不足 {sample_count} 张。"
        )

    source_rng = random.Random(
        RANDOM_SEED
        + int(source_id)
    )

    return (
        source_rng.sample(
            images,
            sample_count,
        ),
        len(images),
    )


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 82)
    print(
        "阶段四 · Unknown v3："
        "Hard Negative 定向数据集构建"
    )
    print("=" * 82)

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # 1. 读取所有关键 ID
    # --------------------------------------------------------

    target_ids = read_ids(
        TARGET_CLASSES_FILE
    )

    old_unknown_ids = read_ids(
        OLD_UNKNOWN_SOURCES_FILE
    )

    val_unknown_ids = read_ids(
        UNKNOWN_VAL_SOURCES_FILE
    )

    hard_rows = (
        read_top_hard_sources()
    )

    hard_ids = [
        row["source_id"].strip()
        for row in hard_rows
    ]

    print(
        f"目标车型数量：{len(target_ids)}"
    )

    print(
        "旧 unknown 来源数量："
        f"{len(old_unknown_ids)}"
    )

    print(
        "unknown val 来源数量："
        f"{len(val_unknown_ids)}"
    )

    print(
        "选取 Hard Negative 来源数量："
        f"{len(hard_ids)}"
    )

    # --------------------------------------------------------
    # 2. 数据泄漏 / 重叠检查
    # --------------------------------------------------------

    target_set = set(
        target_ids
    )

    old_unknown_set = set(
        old_unknown_ids
    )

    val_unknown_set = set(
        val_unknown_ids
    )

    hard_set = set(
        hard_ids
    )

    if (
        len(target_ids)
        != 50
    ):
        raise ValueError(
            "目标车型应为 50 个。"
        )

    if (
        len(old_unknown_ids)
        != OLD_SOURCE_COUNT
    ):
        raise ValueError(
            "旧 unknown 来源应为 20 个，"
            f"实际为 {len(old_unknown_ids)}。"
        )

    if (
        len(hard_ids)
        != TOP_HARD_SOURCE_COUNT
    ):
        raise ValueError(
            "Hard Negative 来源数应为 40 个。"
        )

    # Hard Sources 绝不能和 target 重叠。
    overlap = (
        hard_set
        & target_set
    )

    if overlap:
        raise ValueError(
            "Hard Negative 中混入目标车型："
            f"{sorted(overlap)}"
        )

    # Hard Sources 绝不能和 val unknown 重叠。
    overlap = (
        hard_set
        & val_unknown_set
    )

    if overlap:
        raise ValueError(
            "Hard Negative 与 unknown val 来源重叠："
            f"{sorted(overlap)}"
        )

    # 实验 4D 本来就排除了旧 20 类，
    # 这里再次验证。
    overlap = (
        hard_set
        & old_unknown_set
    )

    if overlap:
        raise ValueError(
            "Hard Negative 与旧 unknown 来源发生重叠："
            f"{sorted(overlap)}"
        )

    # old unknown 也必须与 val unknown 隔离。
    overlap = (
        old_unknown_set
        & val_unknown_set
    )

    if overlap:
        raise ValueError(
            "旧 unknown train 与 val 来源重叠："
            f"{sorted(overlap)}"
        )

    final_source_ids = (
        hard_ids
        + old_unknown_ids
    )

    final_source_set = set(
        final_source_ids
    )

    if (
        len(final_source_set)
        != EXPECTED_TOTAL_SOURCES
    ):
        raise ValueError(
            "v3 最终来源数不等于 60。"
        )

    if (
        final_source_set
        & val_unknown_set
    ):
        raise ValueError(
            "v3 train 与 unknown val 存在来源重叠。"
        )

    # --------------------------------------------------------
    # 3. 创建新输出目录
    # --------------------------------------------------------

    prepare_output_directory()

    manifest_rows = []
    plan_rows = []

    copied_count = 0

    # --------------------------------------------------------
    # 4. 先加入 Top 40 Hard Sources
    # --------------------------------------------------------

    hard_row_by_id = {
        row[
            "source_id"
        ].strip(): row
        for row in hard_rows
    }

    for source_id in hard_ids:

        (
            selected_images,
            available_count,
        ) = stable_sample_images(
            source_id,
            HARD_IMAGES_PER_SOURCE,
        )

        hard_info = (
            hard_row_by_id[
                source_id
            ]
        )

        plan_rows.append(
            {
                "source_id": (
                    source_id
                ),
                "source_kind": (
                    "hard_negative"
                ),
                "hard_rank": (
                    hard_info[
                        "rank"
                    ]
                ),
                "hard_score": (
                    hard_info[
                        "hard_score"
                    ]
                ),
                "most_confused_known_label": (
                    hard_info[
                        "most_confused_known_label"
                    ]
                ),
                "available_images": (
                    available_count
                ),
                "selected_images": (
                    HARD_IMAGES_PER_SOURCE
                ),
            }
        )

        for local_index, src_path in enumerate(
            selected_images,
            start=1,
        ):

            dest_name = (
                f"hard_"
                f"{source_id}_"
                f"{local_index:02d}_"
                f"{src_path.name}"
            )

            dest_path = (
                OUTPUT_UNKNOWN_DIR
                / dest_name
            )

            shutil.copy2(
                src_path,
                dest_path,
            )

            manifest_rows.append(
                {
                    "label": "unknown",
                    "label_index": 50,
                    "source_id": source_id,
                    "source_kind": (
                        "hard_negative"
                    ),
                    "hard_rank": (
                        hard_info[
                            "rank"
                        ]
                    ),
                    "hard_score": (
                        hard_info[
                            "hard_score"
                        ]
                    ),
                    "source_path": str(
                        src_path
                    ),
                    "destination_path": str(
                        dest_path
                    ),
                }
            )

            copied_count += 1

    # --------------------------------------------------------
    # 5. 再加入旧 20 个普通 unknown 来源
    # --------------------------------------------------------

    for source_id in old_unknown_ids:

        (
            selected_images,
            available_count,
        ) = stable_sample_images(
            source_id,
            OLD_IMAGES_PER_SOURCE,
        )

        plan_rows.append(
            {
                "source_id": (
                    source_id
                ),
                "source_kind": (
                    "old_regular_unknown"
                ),
                "hard_rank": "",
                "hard_score": "",
                "most_confused_known_label": "",
                "available_images": (
                    available_count
                ),
                "selected_images": (
                    OLD_IMAGES_PER_SOURCE
                ),
            }
        )

        for local_index, src_path in enumerate(
            selected_images,
            start=1,
        ):

            dest_name = (
                f"regular_"
                f"{source_id}_"
                f"{local_index:02d}_"
                f"{src_path.name}"
            )

            dest_path = (
                OUTPUT_UNKNOWN_DIR
                / dest_name
            )

            shutil.copy2(
                src_path,
                dest_path,
            )

            manifest_rows.append(
                {
                    "label": "unknown",
                    "label_index": 50,
                    "source_id": source_id,
                    "source_kind": (
                        "old_regular_unknown"
                    ),
                    "hard_rank": "",
                    "hard_score": "",
                    "source_path": str(
                        src_path
                    ),
                    "destination_path": str(
                        dest_path
                    ),
                }
            )

            copied_count += 1

    # --------------------------------------------------------
    # 6. 保存来源列表 / Manifest / Plan
    # --------------------------------------------------------

    OUTPUT_SOURCES_FILE.write_text(
        "\n".join(
            final_source_ids
        )
        + "\n",
        encoding="utf-8",
    )

    with OUTPUT_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=manifest_rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    with OUTPUT_PLAN.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=plan_rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            plan_rows
        )

    # --------------------------------------------------------
    # 7. 最终验证
    # --------------------------------------------------------

    actual_output_images = [
        path
        for path in OUTPUT_UNKNOWN_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    manifest_source_ids = {
        row[
            "source_id"
        ]
        for row in manifest_rows
    }

    hard_image_count = sum(
        1
        for row in manifest_rows
        if (
            row[
                "source_kind"
            ]
            == "hard_negative"
        )
    )

    regular_image_count = sum(
        1
        for row in manifest_rows
        if (
            row[
                "source_kind"
            ]
            == "old_regular_unknown"
        )
    )

    if (
        copied_count
        != EXPECTED_TOTAL_IMAGES
    ):
        raise ValueError(
            "复制图片总数不等于 200。"
        )

    if (
        len(actual_output_images)
        != EXPECTED_TOTAL_IMAGES
    ):
        raise ValueError(
            "v3 输出目录实际图片数不等于 200。"
        )

    if (
        len(manifest_source_ids)
        != EXPECTED_TOTAL_SOURCES
    ):
        raise ValueError(
            "v3 实际来源车型数不等于 60。"
        )

    if (
        hard_image_count
        != 160
    ):
        raise ValueError(
            "Hard Negative 图片数不是 160。"
        )

    if (
        regular_image_count
        != 40
    ):
        raise ValueError(
            "普通 unknown 图片数不是 40。"
        )

    # --------------------------------------------------------
    # 8. 终端总结
    # --------------------------------------------------------

    print("\n" + "-" * 82)
    print(
        "Unknown v3 Hard 数据集构建完成。"
    )

    print(
        "\n组成："
    )

    print(
        "Hard Negative："
        "40 个来源 × 4 张 = "
        f"{hard_image_count} 张"
    )

    print(
        "旧普通 unknown："
        "20 个来源 × 2 张 = "
        f"{regular_image_count} 张"
    )

    print(
        "最终 unknown 图片数："
        f"{copied_count}"
    )

    print(
        "最终来源车型数："
        f"{len(manifest_source_ids)}"
    )

    print(
        "与 unknown val 来源交集："
        f"{manifest_source_ids & val_unknown_set}"
    )

    print(
        "\nTop 10 Hard 来源预览："
    )

    for row in hard_rows[
        :10
    ]:
        print(
            f"rank {row['rank']} | "
            f"source {row['source_id']} | "
            f"hard_score {float(row['hard_score']):.4f} | "
            "最常误认 "
            f"{row['most_confused_known_label']}"
        )

    print(
        "\n输出："
    )

    print(
        f"- 图片目录：{OUTPUT_UNKNOWN_DIR}"
    )

    print(
        f"- 来源列表：{OUTPUT_SOURCES_FILE}"
    )

    print(
        f"- Manifest：{OUTPUT_MANIFEST}"
    )

    print(
        f"- Plan：{OUTPUT_PLAN}"
    )

    print(
        "\n下一步："
        "用这 200 张 v3 unknown 替换旧 unknown，"
        "保持实验4A的 layer4+fc、lr=1e-4、weight=2、5 Epoch 不变，"
        "进行实验4E训练。"
    )


if __name__ == "__main__":
    main()

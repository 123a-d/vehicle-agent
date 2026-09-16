"""
阶段四 · Unknown 专项实验 4C（第一步）
文件名：build_unknown_train_v2.py

目标：
    不增加 unknown 的总图片数量，而是显著增加 unknown 来源车型的“多样性”。

旧方案 v1：
    20 个非目标车型 × 每类 10 张 = 200 张 unknown 训练图

新方案 v2：
    100 个非目标车型 × 每类 2 张 = 200 张 unknown 训练图

这样做的好处：
    unknown 总图片数量仍然是 200 张，
    主要变量变成“unknown 来源车型数量”：
        20 类 -> 100 类

    因此后面如果 Unknown F1 明显提升，
    我们就更有理由认为：
        “unknown 来源多样性”比“反复看少数 unknown 车型”
        更有价值。

v2 的 100 个来源：
    - 保留旧 v1 的 20 个 unknown 来源车型
    - 再从原始训练集随机加入 80 个新的非目标车型
    - 严格排除：
        1. 50 个目标车型
        2. unknown val 的 12 个来源车型

这仍然保证：
    unknown_train_source_ids ∩ unknown_val_source_ids = 空集

输入文件：
    D:\\car_project\\candidate_classes_v1.txt
    D:\\car_project\\unknown_train_sources.txt
    D:\\car_project\\unknown_val_sources.txt

原始图片：
    D:\\cardate\\train\\<车型ID>\\...

输出：
    D:\\car_project\\unknown_train_v2\\unknown\\
        共 200 张图片

    D:\\car_project\\unknown_train_sources_v2.txt
    D:\\car_project\\unknown_train_v2_manifest.csv
    D:\\car_project\\unknown_train_v2_plan.csv

注意：
    这个脚本只创建新的 unknown_train_v2，
    不会修改原来的 dataset_51，也不会删除旧 unknown 数据。
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

# 原始 827 类训练集
ORIGINAL_TRAIN_DIR = (
    Path(os.environ.get("VEHICLE_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))
    / "train"
)

TARGET_CLASSES_FILE = (
    PROJECT_ROOT / "scripts" / "stage_01_02_data" / "candidate_classes_v1.txt"
)

OLD_UNKNOWN_TRAIN_SOURCES_FILE = (
    PROJECT_ROOT / "unknown_train_sources.txt"
)

UNKNOWN_VAL_SOURCES_FILE = (
    PROJECT_ROOT / "unknown_val_sources.txt"
)

# 新版 unknown 图片只单独存一份，不复制 50 个 known 类。
OUTPUT_ROOT = (
    PROJECT_ROOT / "unknown_train_v2"
)

OUTPUT_UNKNOWN_DIR = (
    OUTPUT_ROOT / "unknown"
)

OUTPUT_SOURCES_FILE = (
    PROJECT_ROOT / "unknown_train_sources_v2.txt"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT / "unknown_train_v2_manifest.csv"
)

OUTPUT_PLAN = (
    PROJECT_ROOT / "unknown_train_v2_plan.csv"
)


# ============================================================
# 2. 实验参数
# ============================================================

RANDOM_SEED = 42

# 保留旧 unknown 的 20 个来源，并新增 80 个来源：
# 总共 100 个来源车型。
EXTRA_SOURCE_COUNT = 80

# 每个来源只取 2 张：
# 100 × 2 = 200 张。
IMAGES_PER_SOURCE = 2

EXPECTED_OLD_SOURCE_COUNT = 20
EXPECTED_TOTAL_SOURCE_COUNT = 100
EXPECTED_TOTAL_IMAGES = 200

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def read_ids(path: Path):
    """
    从 txt 中读取 4 位车型 ID。

    支持：
        0009

    也容忍：
        0009  其他说明

    只取每行中找到的第一个 4 位数字。
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

    # 去重，同时尽量保持原顺序。
    return list(
        dict.fromkeys(ids)
    )


def list_images(class_dir: Path):
    """列出一个原始车型目录中的图片。"""

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


def collect_available_source_ids(
    excluded_ids,
):
    """
    找到可以作为新增 unknown 来源的原始车型 ID。

    条件：
        1. 目录名是 4 位数字
        2. 不属于 excluded_ids
        3. 至少有 IMAGES_PER_SOURCE 张图片
    """

    if not ORIGINAL_TRAIN_DIR.exists():
        raise FileNotFoundError(
            "找不到原始训练集目录："
            f"{ORIGINAL_TRAIN_DIR}"
        )

    available = []

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

        if (
            len(images)
            >= IMAGES_PER_SOURCE
        ):
            available.append(
                source_id
            )

    return available


def prepare_output_directory():
    """
    清空并重新建立：
        unknown_train_v2/unknown/

    只会操作新目录 unknown_train_v2，
    不会碰 dataset_51。
    """

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_UNKNOWN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def main():
    print("=" * 76)
    print(
        "阶段四 · Unknown v2："
        "提高 unknown 来源车型多样性"
    )
    print("=" * 76)

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # 1. 读取已有实验配置
    # --------------------------------------------------------

    target_ids = read_ids(
        TARGET_CLASSES_FILE
    )

    old_train_unknown_ids = read_ids(
        OLD_UNKNOWN_TRAIN_SOURCES_FILE
    )

    val_unknown_ids = read_ids(
        UNKNOWN_VAL_SOURCES_FILE
    )

    print(
        f"目标车型数量：{len(target_ids)}"
    )

    print(
        "旧 unknown train 来源数量："
        f"{len(old_train_unknown_ids)}"
    )

    print(
        "unknown val 来源数量："
        f"{len(val_unknown_ids)}"
    )

    if len(target_ids) != 50:
        raise ValueError(
            "candidate_classes_v1.txt "
            "应包含 50 个目标车型。"
        )

    if (
        len(old_train_unknown_ids)
        != EXPECTED_OLD_SOURCE_COUNT
    ):
        raise ValueError(
            "旧 unknown_train_sources.txt "
            f"预计包含 {EXPECTED_OLD_SOURCE_COUNT} 个来源，"
            f"实际为 {len(old_train_unknown_ids)}。"
        )

    # --------------------------------------------------------
    # 2. 做严格的数据泄漏检查
    # --------------------------------------------------------

    target_set = set(
        target_ids
    )

    old_train_set = set(
        old_train_unknown_ids
    )

    val_unknown_set = set(
        val_unknown_ids
    )

    if (
        target_set
        & old_train_set
    ):
        raise ValueError(
            "旧 unknown train 中混入了目标车型。"
        )

    if (
        target_set
        & val_unknown_set
    ):
        raise ValueError(
            "unknown val 中混入了目标车型。"
        )

    if (
        old_train_set
        & val_unknown_set
    ):
        raise ValueError(
            "旧 unknown train 与 unknown val "
            "存在来源车型重叠。"
        )

    # --------------------------------------------------------
    # 3. 选择额外 80 个 unknown 来源
    # --------------------------------------------------------

    # 新来源不能是：
    #   - 目标 50 类
    #   - 旧 unknown train 的 20 类
    #   - unknown val 的来源类
    excluded_for_extra = (
        target_set
        | old_train_set
        | val_unknown_set
    )

    available_extra_ids = (
        collect_available_source_ids(
            excluded_for_extra
        )
    )

    print(
        "可供新增的非目标来源车型："
        f"{len(available_extra_ids)}"
    )

    if (
        len(available_extra_ids)
        < EXTRA_SOURCE_COUNT
    ):
        raise ValueError(
            "可用的额外 unknown 来源不足。"
        )

    extra_ids = random.sample(
        available_extra_ids,
        EXTRA_SOURCE_COUNT,
    )

    # v2 = 原来的 20 类 + 新增 80 类
    v2_source_ids = (
        old_train_unknown_ids
        + sorted(extra_ids)
    )

    if (
        len(v2_source_ids)
        != EXPECTED_TOTAL_SOURCE_COUNT
    ):
        raise ValueError(
            "v2 unknown 来源总数不等于 100。"
        )

    # 再次确保与 val 完全隔离。
    if (
        set(v2_source_ids)
        & val_unknown_set
    ):
        raise ValueError(
            "v2 unknown train 与 unknown val "
            "出现来源 ID 重叠。"
        )

    # --------------------------------------------------------
    # 4. 建立新目录并抽取图片
    # --------------------------------------------------------

    prepare_output_directory()

    manifest_rows = []
    plan_rows = []

    copied_count = 0

    for source_id in v2_source_ids:

        class_dir = (
            ORIGINAL_TRAIN_DIR
            / source_id
        )

        images = list_images(
            class_dir
        )

        if (
            len(images)
            < IMAGES_PER_SOURCE
        ):
            raise ValueError(
                f"{source_id} 图片不足 "
                f"{IMAGES_PER_SOURCE} 张。"
            )

        # 每个来源车型独立随机抽 2 张。
        selected_images = random.sample(
            images,
            IMAGES_PER_SOURCE,
        )

        source_kind = (
            "old_v1_source"
            if source_id
            in old_train_set
            else "new_v2_source"
        )

        plan_rows.append(
            {
                "source_id": source_id,
                "source_kind": source_kind,
                "available_images": len(
                    images
                ),
                "selected_images": (
                    IMAGES_PER_SOURCE
                ),
            }
        )

        for local_index, src_path in enumerate(
            selected_images,
            start=1,
        ):

            # 文件名前加 source_id，
            # 防止不同原始目录里出现同名图片。
            dest_name = (
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
                    "source_kind": source_kind,
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
    # 5. 保存可追溯文件
    # --------------------------------------------------------

    OUTPUT_SOURCES_FILE.write_text(
        "\n".join(v2_source_ids)
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
    # 6. 最终验证
    # --------------------------------------------------------

    actual_files = list(
        OUTPUT_UNKNOWN_DIR.iterdir()
    )

    actual_source_ids = {
        row["source_id"]
        for row in manifest_rows
    }

    if (
        copied_count
        != EXPECTED_TOTAL_IMAGES
    ):
        raise ValueError(
            "最终复制图片数不是 200。"
        )

    if (
        len(actual_files)
        != EXPECTED_TOTAL_IMAGES
    ):
        raise ValueError(
            "输出目录中的实际图片数不是 200。"
        )

    if (
        len(actual_source_ids)
        != EXPECTED_TOTAL_SOURCE_COUNT
    ):
        raise ValueError(
            "实际来源车型数不是 100。"
        )

    print("\n" + "-" * 76)
    print("Unknown v2 构建完成。")

    print(
        "旧方案：20 个来源 × 10 张 "
        "= 200 张"
    )

    print(
        "新方案：100 个来源 × 2 张 "
        "= 200 张"
    )

    print(
        f"其中旧来源：{len(old_train_unknown_ids)} 个"
    )

    print(
        f"新增来源：{len(extra_ids)} 个"
    )

    print(
        f"最终图片数：{copied_count}"
    )

    print(
        "与 unknown val 来源交集："
        f"{set(v2_source_ids) & val_unknown_set}"
    )

    print(
        f"\n图片目录：{OUTPUT_UNKNOWN_DIR}"
    )

    print(
        f"来源列表：{OUTPUT_SOURCES_FILE}"
    )

    print(
        f"Manifest：{OUTPUT_MANIFEST}"
    )

    print(
        f"Plan：{OUTPUT_PLAN}"
    )

    print(
        "\n下一步："
        "使用这 200 张 v2 unknown "
        "替换训练时的旧 unknown，"
        "其余训练条件保持与实验4A一致。"
    )


if __name__ == "__main__":
    main()

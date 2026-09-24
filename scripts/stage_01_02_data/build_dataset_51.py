from pathlib import Path
import csv
import os
import random
import shutil

# =========================
# 1. 需要你确认/修改的路径
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("VEHICLE_DATA_ROOT", PROJECT_ROOT / "data" / "raw")
)
CLASS_FILE = SCRIPT_DIR / "candidate_classes_v1.txt"
UNKNOWN_TRAIN_FILE = PROJECT_ROOT / "unknown_train_sources.txt"
UNKNOWN_VAL_FILE = PROJECT_ROOT / "unknown_val_sources.txt"
OUTPUT_ROOT = PROJECT_ROOT / "dataset_51"
MANIFEST_FILE = PROJECT_ROOT / "dataset_51_manifest.csv"

# 为了保证每次运行抽到相同图片，固定随机种子。
RANDOM_SEED = 42

# 每个 unknown 来源车型抽多少张图。
UNKNOWN_TRAIN_PER_SOURCE = 10
UNKNOWN_VAL_PER_SOURCE = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_ids(path: Path) -> list[str]:
    """读取 txt 中的四位类别 ID，自动忽略空行。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_images(folder: Path) -> list[Path]:
    """列出某个类别文件夹中的全部图片。"""
    if not folder.exists():
        raise FileNotFoundError(f"找不到类别目录：{folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_one(source: Path, destination: Path) -> None:
    """复制一张图片，同时自动创建目标文件夹。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    # -------------------------
    # 2. 读取三组类别 ID
    # -------------------------
    target_ids = read_ids(CLASS_FILE)
    unknown_train_ids = read_ids(UNKNOWN_TRAIN_FILE)
    unknown_val_ids = read_ids(UNKNOWN_VAL_FILE)

    # -------------------------
    # 3. 先做安全检查
    # -------------------------
    if len(target_ids) != 50:
        raise ValueError(f"目标类别必须恰好 50 类，现在是 {len(target_ids)} 类。")

    if len(set(target_ids)) != 50:
        raise ValueError("candidate_classes_v1.txt 中存在重复 ID。")

    overlap_target_train = set(target_ids) & set(unknown_train_ids)
    overlap_target_val = set(target_ids) & set(unknown_val_ids)
    overlap_unknown = set(unknown_train_ids) & set(unknown_val_ids)

    if overlap_target_train:
        raise ValueError(f"unknown_train 与目标 50 类发生冲突：{sorted(overlap_target_train)}")
    if overlap_target_val:
        raise ValueError(f"unknown_val 与目标 50 类发生冲突：{sorted(overlap_target_val)}")
    if overlap_unknown:
        raise ValueError(
            "unknown 的训练来源和验证来源必须使用不同车型 ID，"
            f"当前重复：{sorted(overlap_unknown)}"
        )

    # 避免重复运行后旧文件混入新数据。
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"{OUTPUT_ROOT} 已存在。\n"
            "如果你确认要重新构建，请先手动删除这个 dataset_51 文件夹再运行。"
        )

    rng = random.Random(RANDOM_SEED)
    manifest_rows = []

    # -------------------------
    # 4. 复制 50 个目标类别
    # -------------------------
    print("开始复制 50 个目标车型...")

    for split in ("train", "val"):
        for index, class_id in enumerate(target_ids, start=1):
            source_folder = DATA_ROOT / split / class_id
            destination_folder = OUTPUT_ROOT / split / class_id
            files = list_images(source_folder)

            for source in files:
                destination = destination_folder / source.name
                copy_one(source, destination)

                manifest_rows.append({
                    "split": split,
                    "label": class_id,
                    "source_class_id": class_id,
                    "source_file": str(source),
                    "destination_file": str(destination),
                })

            print(
                f"[{split}] {index:02d}/50  "
                f"{class_id}  {len(files)} 张"
            )

    # -------------------------
    # 5. 构造 unknown 训练集
    # -------------------------
    print("\n开始构造 unknown 训练集...")

    unknown_train_folder = OUTPUT_ROOT / "train" / "unknown"

    for class_id in unknown_train_ids:
        files = list_images(DATA_ROOT / "train" / class_id)

        if len(files) < UNKNOWN_TRAIN_PER_SOURCE:
            raise ValueError(
                f"unknown_train 来源 {class_id} 只有 {len(files)} 张图，"
                f"不足 {UNKNOWN_TRAIN_PER_SOURCE} 张。"
            )

        selected = rng.sample(files, UNKNOWN_TRAIN_PER_SOURCE)

        for source in selected:
            # 加来源 ID，防止不同文件夹里存在同名图片。
            destination = unknown_train_folder / f"{class_id}__{source.name}"
            copy_one(source, destination)

            manifest_rows.append({
                "split": "train",
                "label": "unknown",
                "source_class_id": class_id,
                "source_file": str(source),
                "destination_file": str(destination),
            })

        print(f"[unknown/train] {class_id}  抽取 {len(selected)} 张")

    # -------------------------
    # 6. 构造 unknown 验证集
    # -------------------------
    print("\n开始构造 unknown 验证集...")

    unknown_val_folder = OUTPUT_ROOT / "val" / "unknown"

    for class_id in unknown_val_ids:
        files = list_images(DATA_ROOT / "val" / class_id)

        if len(files) < UNKNOWN_VAL_PER_SOURCE:
            raise ValueError(
                f"unknown_val 来源 {class_id} 只有 {len(files)} 张图，"
                f"不足 {UNKNOWN_VAL_PER_SOURCE} 张。"
            )

        selected = rng.sample(files, UNKNOWN_VAL_PER_SOURCE)

        for source in selected:
            destination = unknown_val_folder / f"{class_id}__{source.name}"
            copy_one(source, destination)

            manifest_rows.append({
                "split": "val",
                "label": "unknown",
                "source_class_id": class_id,
                "source_file": str(source),
                "destination_file": str(destination),
            })

        print(f"[unknown/val] {class_id}  抽取 {len(selected)} 张")

    # -------------------------
    # 7. 保存数据来源清单
    # -------------------------
    with MANIFEST_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "label",
                "source_class_id",
                "source_file",
                "destination_file",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # -------------------------
    # 8. 最后做一次数量核验
    # -------------------------
    train_class_dirs = [
        p for p in (OUTPUT_ROOT / "train").iterdir()
        if p.is_dir()
    ]
    val_class_dirs = [
        p for p in (OUTPUT_ROOT / "val").iterdir()
        if p.is_dir()
    ]

    unknown_train_count = len(list_images(unknown_train_folder))
    unknown_val_count = len(list_images(unknown_val_folder))

    print("\n==============================")
    print("51 类数据集构建完成")
    print("==============================")
    print(f"train 类别目录数：{len(train_class_dirs)}")
    print(f"val 类别目录数：{len(val_class_dirs)}")
    print(f"unknown train：{unknown_train_count} 张")
    print(f"unknown val：{unknown_val_count} 张")
    print(f"Manifest：{MANIFEST_FILE}")
    print(f"数据集目录：{OUTPUT_ROOT}")

    if len(train_class_dirs) != 51 or len(val_class_dirs) != 51:
        raise RuntimeError("最终 train/val 的类别目录数不是 51，请检查。")


if __name__ == "__main__":
    main()

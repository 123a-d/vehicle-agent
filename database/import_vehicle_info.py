"""
Stage 7.2
把 50 个目标车型的信息导入 SQLite。

数据来源：

1. candidate_classes_v1.txt
   → 告诉程序哪 50 个 vehicle_id 是我们的目标车型

2. class_info.json
   → 保存所有车型的名称和类型信息

最终写入：

vehicle_agent.db
└── vehicle_info
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


# ============================================================
# 1. 项目路径
# ============================================================

# 当前文件：
#
# D:\car_project\database\import_vehicle_info.py
#
# parents[1] 就是：
#
# D:\car_project
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# 原始车型信息
CLASS_INFO_PATH = (
    PROJECT_ROOT
    / "class_info.json"
)


# Stage 1 选出的 50 个目标车型 ID
CANDIDATE_CLASSES_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "stage_01_02_data"
    / "candidate_classes_v1.txt"
)


# SQLite 数据库
DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "vehicle_agent.db"
)


# ============================================================
# 2. 提取比较可读的字符串
# ============================================================

def extract_readable_text(
    value: str | None,
) -> str:
    """
    class_info.json 中的一些字段形式类似：

        中文内容==Volvo_Volvo S90

    PowerShell 中中文有时可能乱码，
    但 == 后面的英文信息比较稳定。

    所以：

        A==B

    我们优先取：

        B
    """

    if not value:
        return ""

    value = str(value).strip()

    if "==" in value:
        return (
            value
            .split("==")[-1]
            .strip()
        )

    return value


# ============================================================
# 3. 读取50个目标车型ID
# ============================================================

def load_target_ids() -> list[str]:
    """
    从 candidate_classes_v1.txt 读取：

        0009
        0080
        0084
        ...

    返回：

        [
            "0009",
            "0080",
            ...
        ]
    """

    if not CANDIDATE_CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"找不到目标类别文件："
            f"{CANDIDATE_CLASSES_PATH}"
        )

    lines = (
        CANDIDATE_CLASSES_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    target_ids = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    return target_ids


# ============================================================
# 4. 读取 class_info.json
# ============================================================

def load_class_info() -> dict[str, dict]:
    """
    当前 class_info.json 的顶层是 list：

    [
        {
            "id": "0000",
            "name_from_new": "...",
            "name_from_old": "...",
            "type_info": "..."
        },
        ...
    ]

    这里转换成：

    {
        "0000": {...},
        "0001": {...},
        ...
    }

    这样以后可以直接：

        class_info["0009"]

    查询车型。
    """

    if not CLASS_INFO_PATH.exists():
        raise FileNotFoundError(
            f"找不到车型信息文件："
            f"{CLASS_INFO_PATH}"
        )

    with open(
        CLASS_INFO_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        raw_data = json.load(f)

    # ----------------------------------------
    # 情况1：
    # 顶层本来就是 list
    # ----------------------------------------

    if isinstance(raw_data, list):

        result = {}

        for item in raw_data:

            vehicle_id = str(
                item.get(
                    "id",
                    ""
                )
            ).strip()

            if vehicle_id:
                result[vehicle_id] = item

        return result

    # ----------------------------------------
    # 情况2：
    # 如果以后 class_info.json 改成 dict，
    # 也能够兼容。
    # ----------------------------------------

    if isinstance(raw_data, dict):
        return raw_data

    raise ValueError(
        "class_info.json 格式无法识别"
    )


# ============================================================
# 5. 解析车型类型
# ============================================================

def parse_vehicle_type(
    type_info: str | None,
) -> tuple[str, str]:
    """
    例如：

        Sedan####Upper Mid-size Sedan

    拆成：

        vehicle_type
            = Sedan

        vehicle_subtype
            = Upper Mid-size Sedan
    """

    readable = extract_readable_text(
        type_info
    )

    if "####" in readable:

        parts = readable.split(
            "####",
            maxsplit=1,
        )

        return (
            parts[0].strip(),
            parts[1].strip(),
        )

    return (
        readable,
        "",
    )


# ============================================================
# 6. 导入数据库
# ============================================================

def import_vehicle_info() -> int:
    """
    把50个目标车型写入 vehicle_info 表。

    返回：
        成功导入的数量
    """

    target_ids = load_target_ids()

    class_info = load_class_info()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    imported_count = 0

    try:

        for vehicle_id in target_ids:

            info = class_info.get(
                vehicle_id
            )

            if info is None:

                print(
                    "[Warning] "
                    f"找不到 vehicle_id="
                    f"{vehicle_id}"
                )

                continue

            # --------------------------------
            # 车型名称
            #
            # 优先使用 name_from_old 中
            # == 后面的可读内容。
            # --------------------------------

            vehicle_name = (
                extract_readable_text(
                    info.get(
                        "name_from_old"
                    )
                )
            )

            # 如果 name_from_old 没有内容，
            # 再尝试 name_from_new。
            if not vehicle_name:

                vehicle_name = (
                    extract_readable_text(
                        info.get(
                            "name_from_new"
                        )
                    )
                )

            # --------------------------------
            # 类型信息
            # --------------------------------

            (
                vehicle_type,
                vehicle_subtype,
            ) = parse_vehicle_type(
                info.get(
                    "type_info"
                )
            )

            # --------------------------------
            # INSERT OR REPLACE
            #
            # vehicle_id 是 PRIMARY KEY。
            #
            # 如果该ID已经存在：
            #     更新它
            #
            # 如果不存在：
            #     新增它
            # --------------------------------

            sql = """
            INSERT OR REPLACE INTO vehicle_info (
                vehicle_id,
                vehicle_name,
                vehicle_type,
                vehicle_subtype
            )
            VALUES (?, ?, ?, ?)
            """

            connection.execute(
                sql,
                (
                    vehicle_id,
                    vehicle_name,
                    vehicle_type,
                    vehicle_subtype,
                ),
            )

            imported_count += 1

        # 正式保存修改
        connection.commit()

    finally:

        connection.close()

    return imported_count


# ============================================================
# 7. 查看数据库中有多少车型
# ============================================================

def count_vehicle_info() -> int:

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        cursor = connection.execute(
            """
            SELECT COUNT(*)
            FROM vehicle_info
            """
        )

        row = cursor.fetchone()

        return int(
            row[0]
        )

    finally:

        connection.close()


# ============================================================
# 8. 命令行入口
# ============================================================

def main():

    imported_count = (
        import_vehicle_info()
    )

    total_count = (
        count_vehicle_info()
    )

    print(
        "车型信息导入完成。"
    )

    print(
        f"本次处理：{imported_count} 条"
    )

    print(
        f"数据库当前共有："
        f"{total_count} 条车型信息"
    )

    print(
        f"数据库：{DATABASE_PATH}"
    )


if __name__ == "__main__":
    main()
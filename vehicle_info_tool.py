"""
Stage 7.2

数据库车型查询 Tool：

    query_vehicle_info()

作用：

输入：
    vehicle_id

例如：
    0009

输出：
    Volvo S90
    Sedan
    Upper Mid-size Sedan

后面 DeepSeek Agent 会直接调用这个 Tool。
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


# ============================================================
# 1. 数据库位置
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "vehicle_agent.db"
)


# ============================================================
# 2. 查询车型 Tool
# ============================================================

def query_vehicle_info(
    vehicle_id: str,
) -> dict:
    """
    根据 vehicle_id 查询车型详细信息。

    例如：

        query_vehicle_info("0009")

    返回：

        {
            "tool_name": "query_vehicle_info",
            "success": True,
            "vehicle_id": "0009",
            "vehicle_name": "Volvo_Volvo S90",
            ...
        }
    """

    # ----------------------------------------
    # 基础输入检查
    # ----------------------------------------

    if not vehicle_id:

        return {
            "tool_name":
                "query_vehicle_info",

            "success":
                False,

            "error_type":
                "InvalidVehicleId",

            "error":
                "vehicle_id 不能为空",
        }

    vehicle_id = (
        str(vehicle_id)
        .strip()
    )

    # ----------------------------------------
    # 数据库是否存在
    # ----------------------------------------

    if not DATABASE_PATH.exists():

        return {
            "tool_name":
                "query_vehicle_info",

            "success":
                False,

            "error_type":
                "DatabaseNotFound",

            "error":
                f"数据库不存在："
                f"{DATABASE_PATH}",
        }

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        # ------------------------------------
        # ? 是 SQL 参数占位符。
        #
        # 不要这样：
        #
        # "... WHERE vehicle_id = "
        # + vehicle_id
        #
        # 使用参数化 SQL 更安全。
        # ------------------------------------

        sql = """
        SELECT
            vehicle_id,
            vehicle_name,
            vehicle_type,
            vehicle_subtype
        FROM vehicle_info
        WHERE vehicle_id = ?
        """

        cursor = connection.execute(
            sql,
            (
                vehicle_id,
            ),
        )

        # fetchone()
        #
        # = 取查询结果的第一行
        row = cursor.fetchone()

    finally:

        connection.close()

    # ----------------------------------------
    # 没查到
    # ----------------------------------------

    if row is None:

        return {
            "tool_name":
                "query_vehicle_info",

            "success":
                False,

            "error_type":
                "VehicleNotFound",

            "error":
                f"数据库中没有车型："
                f"{vehicle_id}",
        }

    # ----------------------------------------
    # 查询成功
    # ----------------------------------------

    return {
        "tool_name":
            "query_vehicle_info",

        "success":
            True,

        "vehicle_id":
            row[0],

        "vehicle_name":
            row[1],

        "vehicle_type":
            row[2],

        "vehicle_subtype":
            row[3],
    }


# ============================================================
# 3. 命令行测试
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "查询SQLite中的车型信息"
        )
    )

    parser.add_argument(
        "vehicle_id",
        help=(
            "车型ID，例如 0009"
        ),
    )

    args = parser.parse_args()

    result = query_vehicle_info(
        args.vehicle_id
    )

    print(result)


if __name__ == "__main__":
    main()
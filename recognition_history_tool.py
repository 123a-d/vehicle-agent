"""
Stage 7.4
识别历史查询 Tool

核心函数：

    query_recognition_history()

作用：

根据 session_id，
从 SQLite 中查询当前会话的车辆识别历史。

数据库：

    vehicle_agent.db

表：

    recognition_records
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


# ============================================================
# 1. 数据库路径
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
# 2. 查询识别历史 Tool
# ============================================================

def query_recognition_history(
    session_id: str,
    limit: int = 10,
) -> dict:
    """
    查询某个 Session 的识别历史。

    参数
    ----
    session_id:
        当前会话 ID。

        例如：
            session_001

    limit:
        最多返回多少条记录。

        默认：
            10

    返回
    ----
    {
        "tool_name": "query_recognition_history",
        "success": True,
        "session_id": "session_001",
        "count": 2,
        "records": [...]
    }
    """


    # ========================================================
    # 3. 参数检查
    # ========================================================

    if not session_id:

        return {
            "tool_name":
                "query_recognition_history",

            "success":
                False,

            "error_type":
                "InvalidSessionId",

            "error":
                "session_id 不能为空",
        }


    session_id = (
        str(session_id)
        .strip()
    )


    # limit 应该是整数
    try:

        limit = int(
            limit
        )

    except (
        TypeError,
        ValueError,
    ):

        return {
            "tool_name":
                "query_recognition_history",

            "success":
                False,

            "error_type":
                "InvalidLimit",

            "error":
                "limit 必须是整数",
        }


    # 防止 limit 太小或太大
    if limit < 1:
        limit = 1

    if limit > 100:
        limit = 100


    # ========================================================
    # 4. 检查数据库
    # ========================================================

    if not DATABASE_PATH.exists():

        return {
            "tool_name":
                "query_recognition_history",

            "success":
                False,

            "error_type":
                "DatabaseNotFound",

            "error":
                f"数据库不存在："
                f"{DATABASE_PATH}",
        }


    # ========================================================
    # 5. 连接 SQLite
    # ========================================================

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    try:

        # ====================================================
        # 6. SELECT 查询
        # ====================================================
        #
        # WHERE session_id = ?
        #
        # 表示：
        #     只查询当前 Session 的数据。
        #
        #
        # ORDER BY id DESC
        #
        # 表示：
        #     按记录 ID 从大到小排列。
        #
        # 也就是：
        #     最新记录排在前面。
        #
        #
        # LIMIT ?
        #
        # 表示：
        #     最多返回多少条。
        #
        # ====================================================

        sql = """
        SELECT
            id,
            session_id,
            image_path,
            vehicle_id,
            vehicle_name,
            confidence,
            status,
            created_at
        FROM recognition_records
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """


        cursor = connection.execute(
            sql,
            (
                session_id,
                limit,
            ),
        )


        # ====================================================
        # 7. fetchall()
        # ====================================================
        #
        # 7.2 中我们用：
        #
        #     fetchone()
        #
        # 因为车型 ID 理论上只对应一条记录。
        #
        # 这里一个 Session 可能有很多历史记录，
        # 所以使用：
        #
        #     fetchall()
        #
        # 一次取出所有查询结果。
        #
        rows = cursor.fetchall()


    except sqlite3.Error as exc:

        return {
            "tool_name":
                "query_recognition_history",

            "success":
                False,

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),
        }


    finally:

        connection.close()


    # ========================================================
    # 8. 把数据库 tuple 转成 dict
    # ========================================================

    records = []


    for row in rows:

        record = {
            "record_id":
                row[0],

            "session_id":
                row[1],

            "image_path":
                row[2],

            "vehicle_id":
                row[3],

            "vehicle_name":
                row[4],

            "confidence":
                row[5],

            "status":
                row[6],

            "created_at":
                row[7],
        }

        records.append(
            record
        )


    # ========================================================
    # 9. 返回结构化结果
    # ========================================================

    return {
        "tool_name":
            "query_recognition_history",

        "success":
            True,

        "session_id":
            session_id,

        "count":
            len(records),

        "records":
            records,
    }


# ============================================================
# 10. 命令行测试
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "查询某个 Session 的车辆识别历史"
        )
    )


    parser.add_argument(
        "session_id",
        help=(
            "Session ID，例如 session_001"
        ),
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "最多返回多少条，默认10"
        ),
    )


    args = parser.parse_args()


    result = query_recognition_history(
        session_id=args.session_id,
        limit=args.limit,
    )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
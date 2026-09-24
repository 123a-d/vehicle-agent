"""
Stage 7.3
识别记录保存 Tool

核心函数：

    save_recognition_record()

作用：

把一次车辆识别结果永久保存到 SQLite：

    vehicle_agent.db
        ↓
    recognition_records 表

后续 Agent 可以形成：

    recognize_vehicle()
        ↓
    query_vehicle_info()
        ↓
    save_recognition_record()
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


# ============================================================
# 1. 数据库路径
# ============================================================

# 当前文件位于：
#
# D:\car_project\recognition_record_tool.py
#
# 所以当前文件所在目录就是项目根目录。
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
# 2. 保存识别记录 Tool
# ============================================================

def save_recognition_record(
    session_id: str,
    image_path: str,
    vehicle_id: str,
    vehicle_name: str,
    confidence: float,
    status: str,
) -> dict:
    """
    保存一次车辆识别结果。

    参数：

    session_id
        当前会话ID。

        例如：
            session_001

    image_path
        用户上传或指定的车辆图片路径。

    vehicle_id
        模型预测车型ID。

        例如：
            0009

    vehicle_name
        对应车型名称。

        例如：
            Volvo_Volvo S90

    confidence
        模型置信度。

        例如：
            0.993418

    status
        推理层给出的业务状态。

        例如：
            recognized
            uncertain
            unknown

    返回：

        {
            "tool_name": "save_recognition_record",
            "success": True,
            "record_id": 1,
            ...
        }
    """

    # ========================================================
    # 3. 基础参数检查
    # ========================================================

    # session_id 用于以后区分不同对话。
    if not session_id:

        return {
            "tool_name":
                "save_recognition_record",

            "success":
                False,

            "error_type":
                "InvalidSessionId",

            "error":
                "session_id 不能为空",
        }

    if not image_path:

        return {
            "tool_name":
                "save_recognition_record",

            "success":
                False,

            "error_type":
                "InvalidImagePath",

            "error":
                "image_path 不能为空",
        }

    if not vehicle_id:

        return {
            "tool_name":
                "save_recognition_record",

            "success":
                False,

            "error_type":
                "InvalidVehicleId",

            "error":
                "vehicle_id 不能为空",
        }


    # ========================================================
    # 4. 检查数据库是否存在
    # ========================================================

    if not DATABASE_PATH.exists():

        return {
            "tool_name":
                "save_recognition_record",

            "success":
                False,

            "error_type":
                "DatabaseNotFound",

            "error":
                f"数据库不存在："
                f"{DATABASE_PATH}",
        }


    # ========================================================
    # 5. 类型标准化
    # ========================================================

    session_id = (
        str(session_id)
        .strip()
    )

    image_path = (
        str(image_path)
        .strip()
    )

    vehicle_id = (
        str(vehicle_id)
        .strip()
    )

    vehicle_name = (
        str(vehicle_name)
        .strip()
    )

    status = (
        str(status)
        .strip()
    )

    try:

        confidence = float(
            confidence
        )

    except (
        TypeError,
        ValueError,
    ):

        return {
            "tool_name":
                "save_recognition_record",

            "success":
                False,

            "error_type":
                "InvalidConfidence",

            "error":
                "confidence 必须是数字",
        }


    # ========================================================
    # 6. 连接 SQLite
    # ========================================================

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    try:

        # ====================================================
        # 7. INSERT SQL
        # ====================================================
        #
        # 这次与 7.2 最大的区别：
        #
        # 7.2：
        #     SELECT
        #     从数据库读取
        #
        # 7.3：
        #     INSERT
        #     向数据库写入
        #
        # created_at 不需要手动传，
        # 因为建表时已经设置：
        #
        # DEFAULT CURRENT_TIMESTAMP
        # ====================================================

        sql = """
        INSERT INTO recognition_records (
            session_id,
            image_path,
            vehicle_id,
            vehicle_name,
            confidence,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """


        # ====================================================
        # 8. 执行 INSERT
        # ====================================================

        cursor = connection.execute(
            sql,
            (
                session_id,
                image_path,
                vehicle_id,
                vehicle_name,
                confidence,
                status,
            ),
        )


        # ====================================================
        # 9. commit
        # ====================================================
        #
        # INSERT 会修改数据库，
        # 所以必须 commit。
        #
        # 可以暂时理解为：
        #
        #     “确认保存这次数据库修改”
        #
        connection.commit()


        # ====================================================
        # 10. 获取刚刚新增记录的 ID
        # ====================================================
        #
        # recognition_records 的 id 是：
        #
        # INTEGER PRIMARY KEY AUTOINCREMENT
        #
        # 所以每插入一条记录：
        #
        #     1
        #     2
        #     3
        #     ...
        #
        # cursor.lastrowid
        #
        # 可以获得刚刚那条记录的 id。
        #
        record_id = cursor.lastrowid


    except sqlite3.Error as exc:

        # 如果数据库本身执行失败，
        # rollback 撤销本次尚未提交的修改。
        connection.rollback()

        return {
            "tool_name":
                "save_recognition_record",

            "success":
                False,

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),
        }


    finally:

        # 不管成功或失败，
        # 最终都关闭连接。
        connection.close()


    # ========================================================
    # 11. 返回结构化 Tool Result
    # ========================================================

    return {
        "tool_name":
            "save_recognition_record",

        "success":
            True,

        "record_id":
            record_id,

        "session_id":
            session_id,

        "image_path":
            image_path,

        "vehicle_id":
            vehicle_id,

        "vehicle_name":
            vehicle_name,

        "confidence":
            confidence,

        "status":
            status,
    }


# ============================================================
# 12. 命令行测试入口
# ============================================================

def main():
    """
    允许我们暂时不经过 Agent，
    直接在 PowerShell 测试 Tool。

    示例：

    python recognition_record_tool.py ^
        session_001 ^
        "D:\\car_project\\0009_0030.jpg" ^
        0009 ^
        "Volvo_Volvo S90" ^
        0.993418 ^
        recognized
    """

    parser = argparse.ArgumentParser(
        description=(
            "保存车辆识别记录到 SQLite"
        )
    )


    parser.add_argument(
        "session_id"
    )

    parser.add_argument(
        "image_path"
    )

    parser.add_argument(
        "vehicle_id"
    )

    parser.add_argument(
        "vehicle_name"
    )

    parser.add_argument(
        "confidence",
        type=float,
    )

    parser.add_argument(
        "status"
    )


    args = parser.parse_args()


    result = save_recognition_record(
        session_id=args.session_id,
        image_path=args.image_path,
        vehicle_id=args.vehicle_id,
        vehicle_name=args.vehicle_name,
        confidence=args.confidence,
        status=args.status,
    )


    # 使用 JSON 格式输出，
    # 比直接 print(dict) 更方便阅读。
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
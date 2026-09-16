"""
Stage 7.1
SQLite 数据库初始化

作用：
1. 创建 vehicle_agent.db
2. 创建 vehicle_info 表
3. 创建 recognition_records 表

注意：
本文件只负责“建数据库结构”。

现在还不负责：
- Agent
- Tool Calling
- 保存识别记录
- 查询车型

那些功能后面逐步实现。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# ============================================================
# 1. 数据库文件路径
# ============================================================

# 当前 Python 文件所在目录：
#
# D:\car_project\database
#
DATABASE_DIR = Path(__file__).resolve().parent


# 最终数据库文件：
#
# D:\car_project\database\vehicle_agent.db
#
DATABASE_PATH = (
    DATABASE_DIR
    / "vehicle_agent.db"
)


# ============================================================
# 2. 创建数据库连接
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    创建并返回一个 SQLite 数据库连接。

    sqlite3.connect() 的特点：

    如果数据库文件已经存在：
        → 直接连接

    如果数据库文件不存在：
        → 自动创建一个新的数据库文件
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection


# ============================================================
# 3. 创建 vehicle_info 表
# ============================================================

def create_vehicle_info_table(
    connection: sqlite3.Connection,
) -> None:
    """
    创建车型信息表。

    以后 query_vehicle_info() Tool
    会从这里查询车型信息。
    """

    sql = """
    CREATE TABLE IF NOT EXISTS vehicle_info (
        vehicle_id TEXT PRIMARY KEY,
        vehicle_name TEXT NOT NULL,
        vehicle_type TEXT,
        vehicle_subtype TEXT
    )
    """

    connection.execute(sql)


# ============================================================
# 4. 创建 recognition_records 表
# ============================================================

def create_recognition_records_table(
    connection: sqlite3.Connection,
) -> None:
    """
    创建识别历史记录表。

    每进行一次车辆识别，
    后续都可以把结果写入这里。
    """

    sql = """
    CREATE TABLE IF NOT EXISTS recognition_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        session_id TEXT NOT NULL,

        image_path TEXT NOT NULL,

        vehicle_id TEXT,

        vehicle_name TEXT,

        confidence REAL,

        status TEXT,

        created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
    )
    """

    connection.execute(sql)


# ============================================================
# 5. 初始化数据库
# ============================================================

def init_database() -> None:
    """
    初始化整个数据库。

    流程：

        connect
          ↓
        create vehicle_info
          ↓
        create recognition_records
          ↓
        commit
          ↓
        close
    """

    connection = get_connection()

    try:

        create_vehicle_info_table(
            connection
        )

        create_recognition_records_table(
            connection
        )

        # commit = 正式保存数据库修改。
        #
        # 如果不 commit，
        # 某些修改可能不会真正写入数据库。
        connection.commit()

    finally:

        # 无论成功还是异常，
        # 最终都关闭数据库连接。
        connection.close()


# ============================================================
# 6. 命令行入口
# ============================================================

def main():

    init_database()

    print(
        "SQLite 数据库初始化完成。"
    )

    print(
        f"数据库路径：{DATABASE_PATH}"
    )


if __name__ == "__main__":
    main()
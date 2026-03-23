"""数据库初始化与连接管理"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine

import config

# 确保数据目录存在
os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{config.DATABASE_PATH}", echo=False)


def _migrate_db() -> None:
    """检查并执行数据库迁移（为旧表补充新列）"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    try:
        # 检查 inventory_events 表是否缺少 category 列
        cursor.execute("PRAGMA table_info(inventory_events)")
        columns = {row[1] for row in cursor.fetchall()}
        if columns and "category" not in columns:
            cursor.execute('ALTER TABLE inventory_events ADD COLUMN category TEXT DEFAULT ""')
            conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """创建所有表（如果不存在），并执行迁移"""
    SQLModel.metadata.create_all(engine)
    _migrate_db()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """获取数据库会话，支持 with 语句自动关闭"""
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()

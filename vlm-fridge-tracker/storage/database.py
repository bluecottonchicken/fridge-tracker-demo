"""数据库初始化与连接管理"""

import os

from sqlmodel import SQLModel, Session, create_engine

import config

# 确保数据目录存在
os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{config.DATABASE_PATH}", echo=False)


def init_db() -> None:
    """创建所有表（如果不存在）"""
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """获取数据库会话"""
    return Session(engine)

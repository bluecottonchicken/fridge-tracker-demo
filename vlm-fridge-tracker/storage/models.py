"""ORM 模型：用户、分析会话、库存事件"""

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    key_code: str = Field(unique=True, index=True)
    name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisSession(SQLModel, table=True):
    __tablename__ = "analysis_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    video_path: str
    segment_index: int = 0
    keyframe_dir: str = ""
    raw_response: str = ""
    analyzed_at: datetime = Field(default_factory=datetime.now)


class InventoryEvent(SQLModel, table=True):
    __tablename__ = "inventory_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="analysis_sessions.id")
    user_id: int = Field(foreign_key="users.id")
    action: str  # put_in / take_out
    item: str
    original_item: str = ""  # VLM 原始识别结果
    is_corrected: bool = False  # 是否经过用户修正
    quantity: int = 1
    confidence: float = 0.0
    description: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class ItemKnowledge(SQLModel, table=True):
    """物品知识库：为 RAG 积累数据"""
    __tablename__ = "item_knowledge"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    item_name: str  # 最终确认的物品名称
    original_name: str = ""  # VLM 原始识别名称
    description: str = ""  # 外观特征描述
    source: str = "vlm_accepted"  # vlm_accepted / user_corrected
    created_at: datetime = Field(default_factory=datetime.now)

from __future__ import annotations

import json
from enum import Enum
from pydantic import BaseModel, Field


class Action(str, Enum):
    PUT_IN = "put_in"
    TAKE_OUT = "take_out"


class HandObservation(BaseModel):
    """单帧中的手部观察"""
    frame_number: int
    holding_item: str = ""
    direction: str = ""  # into_fridge / out_of_fridge / unclear


class FridgeEvent(BaseModel):
    """单个物品进出事件"""
    action: Action
    item: str
    quantity: int = 1
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""


class AnalysisResult(BaseModel):
    """VLM 单次分析的完整结果"""
    hand_observations: list[HandObservation] = []
    events: list[FridgeEvent] = []
    fridge_state_after: list[str] = []
    notes: str = ""

    @classmethod
    def from_json(cls, raw: str) -> AnalysisResult:
        """从 VLM 原始 JSON 字符串解析，容错处理"""
        # 去除 markdown 代码块包裹
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return cls.model_validate(data)

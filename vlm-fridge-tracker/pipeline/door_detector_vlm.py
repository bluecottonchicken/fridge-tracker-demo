"""使用 Gemini Flash 判断每帧冰箱门状态（开/关）"""

import json

import numpy as np
from google import genai
from google.genai import types

import config
from pipeline.door_detector import DoorState
from pipeline.utils import frame_to_pil, motion
from vlm.gemini_client import extract_text

DOOR_PROMPT = """你是一个冰箱门状态检测器。判断每张图片中冰箱门是否打开。

判断标准：
- 能看到冰箱内部货架或食物 → open
- 只看到冰箱门外表面或没有冰箱 → closed

输出 JSON 数组，每个元素对应一张图片，不要输出任何其他内容：
["open", "closed", "open", ...]"""


def detect(frames: list[np.ndarray], debug: bool = False) -> tuple[list[DoorState], list[float]]:
    """使用 VLM 判断门状态，返回与原始帧等长的状态列表和运动量列表

    流程：
    1. 按 DOOR_VLM_SAMPLE_FPS 从 frames 中采样
    2. 批量发给 Gemini Flash 判断开/关
    3. 将采样结果插值回原始帧长度
    4. 运动量照常逐帧计算（供关键帧加权使用）
    """
    if not frames:
        return [], []

    # 计算运动量（与 cv 版一致）
    motions: list[float] = [0.0]
    for i in range(1, len(frames)):
        motions.append(motion(frames[i - 1], frames[i]))

    # 按配置帧率采样
    sample_interval = max(1, int(config.VIDEO_FPS_SAMPLE_RATE / config.DOOR_VLM_SAMPLE_FPS))
    sample_indices = list(range(0, len(frames), sample_interval))
    sampled_frames = [frames[i] for i in sample_indices]

    if debug:
        print(f"  [VLM门检测] 从 {len(frames)} 帧中采样 {len(sampled_frames)} 帧 (间隔{sample_interval})")

    # 调用 Gemini Flash
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    contents: list = []
    for i, frame in enumerate(sampled_frames):
        contents.append(f"图{i + 1}:")
        contents.append(frame_to_pil(frame))
    contents.append(DOOR_PROMPT)

    try:
        response = client.models.generate_content(
            model=config.DOOR_VLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw_states = json.loads(extract_text(response))
    except Exception as e:
        print(f"  [VLM门检测] 调用失败: {e}，回退为全部 CLOSED")
        return [DoorState.CLOSED] * len(frames), motions

    if debug:
        print(f"  [VLM门检测] 结果: {raw_states}")

    # 解析采样点的状态
    sampled_states = []
    for s in raw_states:
        if isinstance(s, str) and s.lower() == "open":
            sampled_states.append(DoorState.OPEN)
        else:
            sampled_states.append(DoorState.CLOSED)

    # 插值回原始帧长度：每个采样点的状态覆盖到下一个采样点之前的所有帧
    states: list[DoorState] = []
    for frame_idx in range(len(frames)):
        # 找到最近的采样点
        sample_pos = min(frame_idx // sample_interval, len(sampled_states) - 1)
        states.append(sampled_states[sample_pos])

    return states, motions

"""根据门状态序列，从帧序列中切出"开门→关门"片段"""

import numpy as np

from pipeline.door_detector import DoorState


def extract(
    frames: list[np.ndarray],
    states: list[DoorState],
) -> list[list[np.ndarray]]:
    """返回所有开门片段，每个片段是一组连续帧"""
    segments: list[list[np.ndarray]] = []
    current_segment: list[np.ndarray] = []
    in_segment = False

    for frame, state in zip(frames, states):
        if state == DoorState.OPEN:
            if not in_segment:
                in_segment = True
            current_segment.append(frame)
        else:
            if in_segment:
                # 关门，保存片段
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
                in_segment = False

    # 视频结束时门仍开着，也保存
    if current_segment:
        segments.append(current_segment)

    return segments

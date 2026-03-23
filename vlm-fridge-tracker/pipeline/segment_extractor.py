"""根据门状态序列，从帧序列中切出"开门→关门"片段"""

from typing import Optional

import numpy as np

import config
from pipeline.door_detector import DoorState


def extract(
    frames: list[np.ndarray],
    states: list[DoorState],
    motions: Optional[list[float]] = None,
) -> list[tuple[list[np.ndarray], list[float]]]:
    """返回所有开门片段，每个片段是 (帧列表, 运动量列表) 的元组。
    短间隔的相邻片段会自动合并（CLOSED 间隔 ≤ MERGE_GAP_FRAMES）。
    """
    if motions is None:
        motions = [0.0] * len(frames)

    # 第一轮：按状态切割原始片段
    raw_segments: list[dict] = []  # {"frames": [...], "motions": [...], "gap_after": int}
    current_frames: list[np.ndarray] = []
    current_motions: list[float] = []
    in_segment = False
    gap_count = 0

    for frame, state, motion in zip(frames, states, motions):
        if state == DoorState.OPEN:
            if not in_segment:
                in_segment = True
                if raw_segments:
                    raw_segments[-1]["gap_after"] = gap_count
                gap_count = 0
            current_frames.append(frame)
            current_motions.append(motion)
        else:
            if in_segment:
                if current_frames:
                    raw_segments.append({"frames": current_frames, "motions": current_motions, "gap_after": -1})
                current_frames = []
                current_motions = []
                in_segment = False
                gap_count = 0
            gap_count += 1

    if current_frames:
        raw_segments.append({"frames": current_frames, "motions": current_motions, "gap_after": -1})

    if not raw_segments:
        return []

    # 第二轮：合并间隔过短的相邻片段
    merged: list[tuple[list[np.ndarray], list[float]]] = [
        (raw_segments[0]["frames"], raw_segments[0]["motions"])
    ]
    for i in range(1, len(raw_segments)):
        prev_gap = raw_segments[i - 1]["gap_after"]
        if 0 <= prev_gap <= config.MERGE_GAP_FRAMES:
            merged[-1][0].extend(raw_segments[i]["frames"])
            merged[-1][1].extend(raw_segments[i]["motions"])
        else:
            merged.append((raw_segments[i]["frames"], raw_segments[i]["motions"]))

    return merged

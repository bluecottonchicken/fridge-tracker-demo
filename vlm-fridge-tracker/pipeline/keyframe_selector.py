"""从一个片段中选取关键帧：首帧 + 加权采样帧 + 尾帧，数量根据片段时长动态计算"""

from typing import Optional

import numpy as np

import config


def select(
    segment: list[np.ndarray],
    motion_scores: Optional[list] = None,
) -> list[np.ndarray]:
    """根据片段时长动态决定关键帧数量，按运动量加权采样（有运动量时）或均匀采样"""
    n = len(segment)

    if n <= config.MIN_KEYFRAMES:
        return list(segment)

    # 根据片段时长动态计算关键帧数
    duration_seconds = n / config.VIDEO_FPS_SAMPLE_RATE
    total = int(duration_seconds * config.KEYFRAMES_PER_SECOND)
    total = max(config.MIN_KEYFRAMES, min(total, config.MAX_KEYFRAMES, n))

    if total >= n:
        return list(segment)

    # 首帧和尾帧固定，中间采样 total-2 帧
    middle_count = total - 2
    middle_range = list(range(1, n - 1))

    if motion_scores and len(motion_scores) == n and middle_count < len(middle_range):
        # 运动量加权采样：运动量大的帧被选中概率更高
        weights = np.array([motion_scores[i] for i in middle_range], dtype=np.float64)
        weights += 1.0  # 避免全零（确保静止帧也有微小概率）
        weights /= weights.sum()
        chosen = np.random.choice(middle_range, size=middle_count, replace=False, p=weights)
        indices = sorted([0] + chosen.tolist() + [n - 1])
    else:
        # 无运动量数据时回退到均匀采样
        indices = [0]
        for i in range(middle_count):
            idx = 1 + int(i * (n - 2) / middle_count)
            indices.append(idx)
        indices.append(n - 1)
        indices = sorted(set(indices))

    return [segment[i] for i in indices]

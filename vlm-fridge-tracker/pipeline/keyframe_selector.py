"""从一个片段中选取关键帧：首帧 + 均匀间隔帧 + 尾帧，数量根据片段时长动态计算"""

import numpy as np

import config


def select(segment: list[np.ndarray]) -> list[np.ndarray]:
    """根据片段时长动态决定关键帧数量，均匀采样"""
    n = len(segment)

    if n <= config.MIN_KEYFRAMES:
        return list(segment)

    # 根据片段时长动态计算关键帧数
    duration_seconds = n / config.VIDEO_FPS_SAMPLE_RATE
    total = int(duration_seconds * config.KEYFRAMES_PER_SECOND)
    total = max(config.MIN_KEYFRAMES, min(total, config.MAX_KEYFRAMES, n))

    if total >= n:
        return list(segment)

    # 首帧和尾帧固定，中间均匀采样
    middle_count = total - 2
    indices = [0]

    for i in range(middle_count):
        idx = 1 + int(i * (n - 2) / middle_count)
        indices.append(idx)

    indices.append(n - 1)

    # 去重并排序
    indices = sorted(set(indices))

    return [segment[i] for i in indices]

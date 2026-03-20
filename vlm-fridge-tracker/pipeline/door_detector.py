"""判断每一帧的冰箱门状态（开/关）"""

from enum import Enum

import cv2
import numpy as np

import config


class DoorState(Enum):
    OPEN = "open"
    CLOSED = "closed"


def _brightness(frame: np.ndarray) -> float:
    """计算帧的平均亮度"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _motion(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """计算两帧之间的运动量（像素差异）"""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.sum(diff > 25))


def detect(frames: list[np.ndarray]) -> list[DoorState]:
    """对帧序列逐帧判断门状态，返回与帧等长的状态列表"""
    if not frames:
        return []

    baseline_brightness = _brightness(frames[0])
    states: list[DoorState] = [DoorState.CLOSED]
    stable_count = 0

    for i in range(1, len(frames)):
        current_brightness = _brightness(frames[i])
        brightness_delta = current_brightness - baseline_brightness
        motion = _motion(frames[i - 1], frames[i])

        prev_state = states[-1]

        if prev_state == DoorState.CLOSED:
            # 亮度突升 + 有运动 → 开门
            if (brightness_delta > config.BRIGHTNESS_OPEN_THRESHOLD
                    and motion > config.MOTION_THRESHOLD):
                states.append(DoorState.OPEN)
                stable_count = 0
            else:
                states.append(DoorState.CLOSED)
        else:
            # 已开门状态：亮度回落 + 持续无运动 → 关门
            if motion < config.MOTION_THRESHOLD:
                stable_count += 1
            else:
                stable_count = 0

            if (brightness_delta < config.BRIGHTNESS_CLOSE_THRESHOLD
                    and stable_count >= config.STABLE_FRAMES_TO_CLOSE):
                states.append(DoorState.CLOSED)
                baseline_brightness = _brightness(frames[i])
            else:
                states.append(DoorState.OPEN)

    return states

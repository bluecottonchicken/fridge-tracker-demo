"""判断每一帧的冰箱门状态（开/关）"""

from enum import Enum

import cv2
import numpy as np

import config
from pipeline.utils import motion


class DoorState(Enum):
    OPEN = "open"
    CLOSED = "closed"


def _brightness(frame: np.ndarray) -> float:
    """计算帧的平均亮度"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def detect(frames: list[np.ndarray], debug: bool = False) -> tuple[list[DoorState], list[float]]:
    """对帧序列逐帧判断门状态，返回与帧等长的状态列表和运动量列表

    Args:
        frames: 视频帧列表
        debug: 是否打印每帧的亮度、运动量、状态判定详情

    Returns:
        (states, motions): 门状态列表 和 每帧运动量列表（首帧运动量为 0）
    """
    if not frames:
        return [], []

    baseline_brightness = _brightness(frames[0])
    states: list[DoorState] = [DoorState.CLOSED]
    motions: list[float] = [0.0]
    stable_count = 0
    open_count = 0  # 开门后已持续的帧数
    peak_brightness = baseline_brightness  # 开门期间的亮度峰值

    if debug:
        print(f"  [DEBUG] 帧0: 亮度={baseline_brightness:.1f} (基准) → CLOSED")

    for i in range(1, len(frames)):
        current_brightness = _brightness(frames[i])
        brightness_delta = current_brightness - baseline_brightness
        motion = motion(frames[i - 1], frames[i])
        motions.append(motion)

        prev_state = states[-1]

        if prev_state == DoorState.CLOSED:
            # 亮度突升 + 有运动 → 开门
            if (brightness_delta > config.BRIGHTNESS_OPEN_THRESHOLD
                    and motion > config.MOTION_THRESHOLD):
                states.append(DoorState.OPEN)
                stable_count = 0
                open_count = 1
                peak_brightness = current_brightness
                if debug:
                    print(f"  [DEBUG] 帧{i}: 亮度={current_brightness:.1f} Δ={brightness_delta:+.1f} 运动={motion:.0f} → ★ OPEN（开门触发）")
            else:
                states.append(DoorState.CLOSED)
                if debug:
                    b_ok = "✓" if brightness_delta > config.BRIGHTNESS_OPEN_THRESHOLD else "✗"
                    m_ok = "✓" if motion > config.MOTION_THRESHOLD else "✗"
                    print(f"  [DEBUG] 帧{i}: 亮度={current_brightness:.1f} Δ={brightness_delta:+.1f}{b_ok} 运动={motion:.0f}{m_ok} → CLOSED")
        else:
            open_count += 1

            # 持续追踪开门期间的亮度峰值
            if current_brightness > peak_brightness:
                peak_brightness = current_brightness

            # 最短开门保护：未满 MIN_OPEN_FRAMES 帧时强制保持 OPEN
            if open_count < config.MIN_OPEN_FRAMES:
                states.append(DoorState.OPEN)
                if motion < config.MOTION_THRESHOLD:
                    stable_count += 1
                else:
                    stable_count = 0
                if debug:
                    print(f"  [DEBUG] 帧{i}: 亮度={current_brightness:.1f} Δ={brightness_delta:+.1f} 运动={motion:.0f} → OPEN（保护期{open_count}/{config.MIN_OPEN_FRAMES}）")
                continue

            # 已开门状态：亮度回落 + 持续无运动 → 关门
            # 关门判定：亮度须从峰值回落至少 BRIGHTNESS_CLOSE_RATIO 的涨幅
            if motion < config.MOTION_THRESHOLD:
                stable_count += 1
            else:
                stable_count = 0

            brightness_rise = peak_brightness - baseline_brightness
            close_threshold = peak_brightness - brightness_rise * config.BRIGHTNESS_CLOSE_RATIO

            if (current_brightness < close_threshold
                    and stable_count >= config.STABLE_FRAMES_TO_CLOSE):
                states.append(DoorState.CLOSED)
                baseline_brightness = _brightness(frames[i])
                open_count = 0
                if debug:
                    print(f"  [DEBUG] 帧{i}: 亮度={current_brightness:.1f} 峰值={peak_brightness:.1f} 关门阈值={close_threshold:.1f} 稳定={stable_count} → ★ CLOSED（关门触发）")
            else:
                states.append(DoorState.OPEN)
                if debug and i % 5 == 0:  # OPEN 状态每5帧打印一次，避免刷屏
                    print(f"  [DEBUG] 帧{i}: 亮度={current_brightness:.1f} 峰值={peak_brightness:.1f} 关门阈值={close_threshold:.1f} 运动={motion:.0f} 稳定={stable_count}/{config.STABLE_FRAMES_TO_CLOSE} → OPEN")

    return states, motions

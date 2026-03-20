"""读取视频文件，按采样率输出帧序列"""

import cv2
import numpy as np

import config


def load(video_path: str) -> list[np.ndarray]:
    """读取视频文件，按 VIDEO_FPS_SAMPLE_RATE 采样返回帧列表"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / config.VIDEO_FPS_SAMPLE_RATE))

    frames: list[np.ndarray] = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            frames.append(frame)
        frame_index += 1

    cap.release()

    if not frames:
        raise ValueError(f"视频文件无有效帧: {video_path}")

    return frames

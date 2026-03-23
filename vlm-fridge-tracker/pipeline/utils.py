"""pipeline 公共工具函数"""

import cv2
import numpy as np
from PIL import Image


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV BGR帧 转 PIL Image"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def motion(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """计算两帧之间的运动量（像素差异）"""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.sum(diff > 25))

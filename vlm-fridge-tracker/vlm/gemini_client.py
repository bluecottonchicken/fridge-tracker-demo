"""调用 Gemini API，发送关键帧+Prompt，返回原始响应"""

import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image

import config
from vlm.prompt_template import SYSTEM_PROMPT, build_user_prompt


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV BGR帧 转 PIL Image"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def analyze(keyframes: list[np.ndarray]) -> str:
    """发送关键帧到 Gemini，返回原始 JSON 字符串"""
    genai.configure(api_key=config.GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # 构建多模态内容：图片 + 文本
    contents: list = []
    for i, frame in enumerate(keyframes):
        contents.append(f"图{i + 1}:")
        contents.append(_frame_to_pil(frame))

    contents.append(build_user_prompt(len(keyframes)))

    response = model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
        ),
    )

    return response.text

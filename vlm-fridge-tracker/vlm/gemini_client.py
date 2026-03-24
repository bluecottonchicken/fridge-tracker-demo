"""调用 Gemini API，发送关键帧+Prompt，返回原始响应"""

import time

import numpy as np
from google import genai
from google.genai import types

import config
from pipeline.utils import frame_to_pil

if config.PROMPT_TEMPLATE == "v4":
    from vlm.prompt_template_v4 import SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT, build_user_prompt, build_refine_prompt, build_redescribe_prompt
elif config.PROMPT_TEMPLATE == "v3":
    from vlm.prompt_template_v3 import SYSTEM_PROMPT, build_user_prompt, build_refine_prompt, build_redescribe_prompt
    REFINE_SYSTEM_PROMPT = SYSTEM_PROMPT
elif config.PROMPT_TEMPLATE == "v2":
    from vlm.prompt_template_v2 import SYSTEM_PROMPT, build_user_prompt, build_refine_prompt, build_redescribe_prompt
    REFINE_SYSTEM_PROMPT = SYSTEM_PROMPT
else:
    from vlm.prompt_template import SYSTEM_PROMPT, build_user_prompt, build_refine_prompt, build_redescribe_prompt
    REFINE_SYSTEM_PROMPT = SYSTEM_PROMPT

# Token 用量累计器
_usage_stats = {
    "prompt_tokens": 0,
    "candidates_tokens": 0,
    "total_tokens": 0,
    "call_count": 0,
}


def extract_text(response) -> str:
    """从 Gemini 响应中手动提取文本部分，跳过 thought_signature 等非文本 part，避免 SDK warning"""
    return "".join(
        getattr(part, 'text', '') or ''
        for part in response.candidates[0].content.parts
    )


def get_usage_stats() -> dict:
    """获取本次运行的累计 token 用量"""
    return dict(_usage_stats)


def reset_usage_stats() -> None:
    """重置 token 用量统计"""
    _usage_stats["prompt_tokens"] = 0
    _usage_stats["candidates_tokens"] = 0
    _usage_stats["total_tokens"] = 0
    _usage_stats["call_count"] = 0


def _call_with_retry(contents: list, system_instruction: str) -> str:
    """带指数退避重试的 generate_content 包装，使用 google.genai 客户端并自动记录 token 用量"""
    last_error = None
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    for attempt in range(1 + config.VLM_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                )
            )
            # 记录 token 用量
            meta = getattr(response, "usage_metadata", None)
            if meta:
                _usage_stats["prompt_tokens"] += getattr(meta, "prompt_token_count", 0)
                _usage_stats["candidates_tokens"] += getattr(meta, "candidates_token_count", 0)
                _usage_stats["total_tokens"] += getattr(meta, "total_token_count", 0)
            _usage_stats["call_count"] += 1
            return extract_text(response)
        except Exception as e:
            last_error = e
            if attempt < config.VLM_MAX_RETRIES:
                wait = 2 ** attempt
                print(f"      Gemini 调用失败（第{attempt + 1}次），{wait}秒后重试: {e}")
                time.sleep(wait)
    raise last_error


def analyze(keyframes: list[np.ndarray], rag_context: str = "") -> str:
    """发送关键帧到 Gemini，返回原始 JSON 字符串"""
    # 构建多模态内容：图片 + 文本
    contents: list = []
    for i, frame in enumerate(keyframes):
        contents.append(f"图{i + 1}:")
        contents.append(frame_to_pil(frame))

    contents.append(build_user_prompt(len(keyframes), rag_context))

    return _call_with_retry(contents, SYSTEM_PROMPT)


def refine_item(keyframes: list[np.ndarray], item_description: str, rag_context: str = "") -> str:
    """二次聚焦分析：针对低置信度物品重新识别，返回原始 JSON 字符串"""
    contents: list = []
    for i, frame in enumerate(keyframes):
        contents.append(f"图{i + 1}:")
        contents.append(frame_to_pil(frame))

    contents.append(build_refine_prompt(item_description, rag_context))

    return _call_with_retry(contents, REFINE_SYSTEM_PROMPT)


def redescribe_item(
    keyframes: list[np.ndarray], correct_name: str, wrong_name: str
) -> str:
    """用户修正后重新描述：让 Gemini 重新观察并描述正确物品的外观，返回 JSON 字符串"""
    contents: list = []
    for i, frame in enumerate(keyframes):
        contents.append(f"图{i + 1}:")
        contents.append(frame_to_pil(frame))

    contents.append(build_redescribe_prompt(correct_name, wrong_name))

    return _call_with_retry(contents, REFINE_SYSTEM_PROMPT)

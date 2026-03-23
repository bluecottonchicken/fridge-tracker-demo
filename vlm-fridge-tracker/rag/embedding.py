"""Embedding 生成与余弦相似度计算"""

import json
from typing import Optional

import numpy as np
import google.generativeai as genai

import config

EMBEDDING_MODEL = "models/text-embedding-004"


def get_embedding(text: str) -> list[float]:
    """调用 Gemini embedding API 生成单个文本的向量"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
    )
    return result["embedding"]


def get_embeddings_batch(texts: list) -> list[Optional[list]]:
    """批量生成多个文本的 embedding，一次 API 调用

    Returns:
        与 texts 等长的列表，失败的位置为 None
    """
    if not texts:
        return []
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
    )
    return result["embedding"]


def cosine_similarity(vec_a, vec_b) -> float:
    """计算两个向量的余弦相似度（numpy 加速）"""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query_vec, matrix) -> np.ndarray:
    """一个查询向量与多个候选向量批量计算余弦相似度

    Args:
        query_vec: 查询向量 (D,)
        matrix: 候选矩阵 (N, D)

    Returns:
        相似度数组 (N,)
    """
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(m.shape[0])
    m_norms = np.linalg.norm(m, axis=1)
    m_norms = np.where(m_norms == 0, 1.0, m_norms)  # 避免除零
    return np.dot(m, q) / (m_norms * q_norm)


def serialize_embedding(embedding: list[float]) -> str:
    """将 embedding 序列化为 JSON 字符串，用于存入 SQLite"""
    return json.dumps(embedding)


def deserialize_embedding(data: str) -> list[float]:
    """从 JSON 字符串反序列化 embedding"""
    return json.loads(data)

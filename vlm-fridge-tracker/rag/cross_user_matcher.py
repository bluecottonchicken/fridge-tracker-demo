"""跨用户物品匹配：基于品类库输出，而非具体物品名"""

import json
from typing import Optional

from sqlmodel import Session, select

from storage.models import ItemCategory
from rag.embedding import get_embedding, batch_cosine_similarity, deserialize_embedding


def get_known_categories(
    db: Session,
    query_items: Optional[list] = None,
    top_n: int = 5,
    fallback_n: int = 30,
) -> list[dict]:
    """获取品类库中与用户已知物品最相关的品类，用于 Prompt 注入

    Args:
        query_items: 用户已知物品名列表（冰箱状态+高频物品），用于 embedding 相似度排序
        top_n: 有 query 时返回的品类数量
        fallback_n: 无 query 或 embedding 失败时返回的品类数量上限

    Returns:
        [{"category": "牛奶", "description": "白色纸盒/利乐砖"}, ...]
    """
    categories = db.exec(select(ItemCategory)).all()
    if not categories:
        return []

    # 有 query 且品类数超过 top_n 时，用 embedding 筛选最相关品类
    if query_items and len(categories) > top_n:
        valid_cats = [(cat, deserialize_embedding(cat.embedding)) for cat in categories if cat.embedding]
        if valid_cats:
            try:
                query_text = " ".join(query_items)
                query_emb = get_embedding(query_text)
                cat_list, emb_list = zip(*valid_cats)
                sims = batch_cosine_similarity(query_emb, list(emb_list))
                ranked_indices = sims.argsort()[::-1][:top_n]
                return [
                    {"category": cat_list[i].category, "description": cat_list[i].description}
                    for i in ranked_indices
                ]
            except Exception:
                pass  # embedding 失败，回退全量返回

    return [
        {"category": cat.category, "description": cat.description}
        for cat in categories[:fallback_n]
    ]

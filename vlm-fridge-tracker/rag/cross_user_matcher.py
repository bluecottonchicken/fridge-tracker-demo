"""跨用户物品匹配：基于品类库输出，而非具体物品名"""

from typing import Optional

from sqlmodel import Session, select

from storage.models import ItemCategory
from rag.embedding import (
    get_embeddings_batch,
    batch_cosine_similarity,
    deserialize_embedding,
)


def get_known_categories(
    db: Session,
    query_items: Optional[list] = None,
    per_item_top_k: int = 3,
    fallback_n: int = 30,
) -> list[dict]:
    """获取品类库中与用户已知物品最相关的品类，用于 Prompt 注入

    逐个 query item 独立检索 top-k 品类，合并去重后返回。
    避免将所有 query items 拼接为一个字符串做 centroid embedding，
    防止语义稀释导致离群品类（如用户同时有肉蛋奶和蔬菜时丢失蔬菜品类）。

    Args:
        query_items: 用户已知物品名列表（冰箱状态+高频物品）
        per_item_top_k: 每个 query item 检索的品类数
        fallback_n: 无 query 或 embedding 失败时返回的品类数量上限

    Returns:
        [{"category": "牛奶", "description": "白色纸盒/利乐砖"}, ...]
    """
    categories = db.exec(select(ItemCategory)).all()
    if not categories:
        return []

    if query_items and len(categories) > per_item_top_k:
        valid_cats = [
            (cat, deserialize_embedding(cat.embedding))
            for cat in categories if cat.embedding
        ]
        if valid_cats:
            try:
                cat_list, emb_list = zip(*valid_cats)
                emb_matrix = list(emb_list)

                # 批量获取所有 query items 的 embedding（一次 API 调用）
                query_embs = get_embeddings_batch(query_items)

                # 逐个 item 独立检索 top-k，合并去重
                selected_indices = set()
                for query_emb in query_embs:
                    if query_emb is None:
                        continue
                    sims = batch_cosine_similarity(query_emb, emb_matrix)
                    top_k_idx = sims.argsort()[::-1][:per_item_top_k]
                    top_k_idx = [idx for idx in top_k_idx if sims[idx] >= 0.5]
                    selected_indices.update(top_k_idx)

                if selected_indices:
                    return [
                        {"category": cat_list[i].category, "description": cat_list[i].description}
                        for i in sorted(selected_indices)
                    ]
            except Exception as e:
                print(f"  ⚠ 品类检索 embedding 失败: {e}，回退全量返回")

    return [
        {"category": cat.category, "description": cat.description}
        for cat in categories[:fallback_n]
    ]

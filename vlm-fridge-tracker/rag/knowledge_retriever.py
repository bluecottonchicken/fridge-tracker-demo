"""基于向量的私有知识库检索"""

from sqlmodel import Session, select

from storage.models import ItemKnowledge
from rag.embedding import (
    get_embedding,
    deserialize_embedding,
    batch_cosine_similarity,
)

def retrieve_relevant_knowledge(
    db: Session,
    user_id: int,
    query_text: str,
    top_k: int = 3,
    threshold: float = 0.5,
    precomputed_embedding: list = None,
) -> list[dict]:
    """
    根据 VLM 盲猜的外观和名称，从数据库中召回最相似的历史物品经验。
    支持对召回结果按照物品名称去重，确保多样性。
    """
    query_emb = precomputed_embedding
    if query_emb is None:
        try:
            query_emb = get_embedding(query_text)
        except Exception as e:
            print(f"  ⚠ 知识库检索 embedding 失败: {e}")
            return []

    knowledge_records = db.exec(
        select(ItemKnowledge).where(ItemKnowledge.user_id == user_id)
    ).all()

    valid_records = [(k, deserialize_embedding(k.embedding)) for k in knowledge_records if k.embedding]
    if not valid_records:
        return []

    records, emb_list = zip(*valid_records)
    sims = batch_cosine_similarity(query_emb, list(emb_list))
    
    # 提取超过 Top-K 的索引以便去重
    top_k_idx = sims.argsort()[::-1][:top_k * 3]
    
    results = []
    seen_items = set()
    for idx in top_k_idx:
        if sims[idx] >= threshold:
            rec = records[idx]
            if rec.item_name in seen_items:
                continue
            seen_items.add(rec.item_name)
            results.append({
                "item_name": rec.item_name,
                "original_name": getattr(rec, "original_name", ""),
                "description": rec.description,
                "source": getattr(rec, "source", ""),
                "similarity": round(float(sims[idx]), 3)
            })
            if len(results) >= top_k:
                break
            
    return results
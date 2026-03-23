"""新物品归类：embedding 匹配现有品类 → 匹配不到则调 Gemini 归类"""

import json
from typing import Optional

import google.generativeai as genai
from sqlmodel import Session, select

import config
from storage.models import ItemCategory
from rag.embedding import (
    get_embedding,
    serialize_embedding,
    batch_cosine_similarity,
    deserialize_embedding,
)

CATEGORIZE_PROMPT_TEMPLATE = """你是一个食物分类专家。请判断以下物品属于哪个食物品类。

物品名称：{item_name}
物品描述：{description}

现有品类列表：
{category_list}

要求：
1. 从现有品类中选择最匹配的一个，直接返回品类名。
2. 品类粒度要保持一致：例如"橙子"包含脐橙和血橙，"牛奶"包含纯牛奶和脱脂牛奶，除非必要不要新增品类。
3. 只有当现有品类中确实没有匹配项时，才返回一个新品类名（保持与现有品类相同的粒度）。

严格输出以下 JSON，不要输出任何其他内容：
{{"category": "品类名", "is_new": true/false}}"""


def match_category(
    db: Session,
    item_name: str,
    description: str = "",
    threshold: float = config.CATEGORY_MATCH_THRESHOLD,
    precomputed_embedding: Optional[list] = None,
) -> dict:
    """将物品匹配到品类

    流程：
    1. embedding 相似度匹配现有品类
    2. 匹配不到则调 Gemini 归类
    3. Gemini 返回的品类若已存在则直接用，否则新建

    Args:
        precomputed_embedding: 预计算的 embedding，避免重复调用 API

    Returns:
        {"category": "牛奶", "matched_by": "embedding|gemini|new", "similarity": 0.92}
    """
    categories = db.exec(select(ItemCategory)).all()
    if not categories:
        return {"category": item_name, "matched_by": "none", "similarity": 0.0}

    # Step 1: embedding 匹配（优先使用预计算的 embedding）
    query_emb = precomputed_embedding
    if query_emb is None:
        query_text = f"{item_name} {description}".strip()
        try:
            query_emb = get_embedding(query_text)
        except Exception:
            # embedding 失败，直接走 Gemini
            return _gemini_categorize(db, item_name, description, categories)

    # 批量计算相似度：将所有有 embedding 的品类组成矩阵一次算完
    valid_cats = [(cat, deserialize_embedding(cat.embedding)) for cat in categories if cat.embedding]
    best_match = None
    best_sim = 0.0
    if valid_cats:
        cat_list, emb_list = zip(*valid_cats)
        sims = batch_cosine_similarity(query_emb, list(emb_list))
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        best_match = cat_list[best_idx]

    if best_match and best_sim >= threshold:
        # 命中：更新品类的 user_names 和 vlm_aliases
        _update_category_names(db, best_match, item_name, description)
        return {
            "category": best_match.category,
            "matched_by": "embedding",
            "similarity": round(best_sim, 3),
        }

    # Step 2: embedding 未命中，调 Gemini 归类
    return _gemini_categorize(db, item_name, description, categories)


def _gemini_categorize(
    db: Session,
    item_name: str,
    description: str,
    categories: list[ItemCategory],
) -> dict:
    """调用 Gemini 判断物品品类"""
    model = genai.GenerativeModel(model_name=config.GEMINI_MODEL)

    cat_list = "\n".join(f"  - {c.category}" for c in categories)
    prompt = CATEGORIZE_PROMPT_TEMPLATE.format(
        item_name=item_name,
        description=description,
        category_list=cat_list,
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text)
    except Exception:
        return {"category": item_name, "matched_by": "fallback", "similarity": 0.0}

    category_name = result.get("category", item_name)
    is_new = result.get("is_new", False)

    # 检查 Gemini 返回的品类是否已存在
    existing = db.exec(
        select(ItemCategory).where(ItemCategory.category == category_name)
    ).first()

    if existing:
        _update_category_names(db, existing, item_name, description)
        return {
            "category": category_name,
            "matched_by": "gemini",
            "similarity": 1.0,
        }

    if is_new:
        # 新建品类
        try:
            emb = serialize_embedding(
                get_embedding(f"{category_name} {description}")
            )
        except Exception:
            emb = ""

        new_cat = ItemCategory(
            category=category_name,
            description=description,
            vlm_aliases=json.dumps([item_name], ensure_ascii=False),
            user_names="[]",
            embedding=emb,
        )
        db.add(new_cat)
        db.commit()
        return {
            "category": category_name,
            "matched_by": "new",
            "similarity": 1.0,
        }

    return {"category": category_name, "matched_by": "gemini", "similarity": 1.0}


def _update_category_names(
    db: Session,
    category: ItemCategory,
    item_name: str,
    description: str,
) -> None:
    """更新品类的 user_names 列表（去重追加）"""
    try:
        user_names = json.loads(category.user_names)
    except (json.JSONDecodeError, TypeError):
        user_names = []

    if item_name not in user_names and item_name != category.category:
        user_names.append(item_name)
        category.user_names = json.dumps(user_names, ensure_ascii=False)
        db.add(category)
        db.commit()

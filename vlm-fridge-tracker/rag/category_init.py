"""调用 Gemini 生成初始食物品类库，写入 item_categories 表"""

import json

import google.generativeai as genai
from sqlmodel import Session, select

import config
from storage.models import ItemCategory
from rag.embedding import get_embeddings_batch, serialize_embedding


CATEGORY_GENERATION_PROMPT = """你是一个食物分类专家。请列出冰箱中常见的食物品类。

要求：
1. 品类粒度统一且适中。例如用"牛奶"而非"伊利牛奶"或"乳制品"，用"橙子"而非"血橙"或"水果"。
2. 涵盖以下大类中的常见品种：乳制品、蛋类、肉类、海鲜、水果、蔬菜、饮料、调味品、熟食/预制食品、主食、零食。
3. 每个品类给出：类别名、常见外观描述（颜色、包装、形状）。
4. 除非品类间差异极大（如外观、存储方式完全不同），否则不要拆分为子品类。
5. 列出约 50-80 个品类即可，不必穷举。

严格输出以下 JSON 数组，不要输出任何其他内容：
[
  {
    "category": "品类名",
    "description": "常见外观描述"
  }
]"""


def generate_categories() -> list[dict]:
    """调用 Gemini 生成食物品类列表"""
    model = genai.GenerativeModel(model_name=config.GEMINI_MODEL)

    response = model.generate_content(
        CATEGORY_GENERATION_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)


def init_categories(db: Session) -> int:
    """初始化品类库：如果表为空则调用 Gemini 生成并写入

    Returns:
        写入的品类数量，如果已有数据则返回 0
    """
    existing = db.exec(select(ItemCategory)).first()
    if existing:
        return 0

    print("  初始化食物品类库（首次运行，调用 Gemini 生成）...")
    categories = generate_categories()

    # 过滤有效品类
    valid_cats = [(c.get("category", ""), c.get("description", "")) for c in categories if c.get("category")]
    if not valid_cats:
        return 0

    # 批量生成 embedding（一次 API 调用）
    texts = [f"{name} {desc}".strip() for name, desc in valid_cats]
    try:
        embeddings = get_embeddings_batch(texts)
    except Exception:
        embeddings = [None] * len(texts)

    count = 0
    for (name, desc), emb_vec in zip(valid_cats, embeddings):
        emb = serialize_embedding(emb_vec) if emb_vec else ""
        item_cat = ItemCategory(
            category=name,
            description=desc,
            embedding=emb,
        )
        db.add(item_cat)
        count += 1

    db.commit()
    print(f"  已生成 {count} 个食物品类")
    return count


def list_categories(db: Session) -> list[ItemCategory]:
    """获取所有品类"""
    return list(db.exec(select(ItemCategory)).all())

"""查询用户高频物品列表和历史修正映射"""

from sqlmodel import Session, select

from storage.models import ItemKnowledge


def get_frequent_items(db: Session, user_id: int, top_n: int = 10) -> list[dict]:
    """返回用户最常见的物品（按出现次数降序）

    返回格式: [{"name": "伊利纯牛奶", "description": "白色利乐砖", "count": 5}, ...]
    """
    records = db.exec(
        select(ItemKnowledge)
        .where(ItemKnowledge.user_id == user_id)
    ).all()

    # 按 item_name 聚合计数
    counter: dict[str, dict] = {}
    for r in records:
        key = r.item_name
        if key not in counter:
            counter[key] = {"name": key, "description": r.description, "count": 0}
        counter[key]["count"] += 1
        # 优先用纠错后的描述（source=user_corrected），否则用最新非空描述
        if r.description:
            if r.source == "user_corrected" or not counter[key]["description"]:
                counter[key]["description"] = r.description

    ranked = sorted(counter.values(), key=lambda x: x["count"], reverse=True)
    return ranked[:top_n]


def get_alias_mappings(db: Session, user_id: int, top_n: int = 10) -> list[dict]:
    """返回用户的修正记录，按正确物品名聚合

    返回格式: [{
        "corrected": "培根",
        "wrong_names": ["无糖可乐", "猪肉"],
        "correction_count": 3,
        "description": "扁平长方形透明塑料盒，红白相间肉片",
    }, ...]

    按修正次数降序排列，修正越多的物品越靠前。
    """
    records = db.exec(
        select(ItemKnowledge)
        .where(ItemKnowledge.user_id == user_id)
        .where(ItemKnowledge.source == "user_corrected")
        .where(ItemKnowledge.original_name != "")
    ).all()

    # 按正确物品名聚合
    aggregated: dict[str, dict] = {}
    for r in records:
        if r.original_name == r.item_name:
            continue
        key = r.item_name
        if key not in aggregated:
            aggregated[key] = {
                "corrected": key,
                "wrong_names": [],
                "correction_count": 0,
                "description": "",
            }
        aggregated[key]["correction_count"] += 1
        if r.original_name not in aggregated[key]["wrong_names"]:
            aggregated[key]["wrong_names"].append(r.original_name)
        # 优先用最新的非空描述
        if r.description:
            aggregated[key]["description"] = r.description

    # 按修正次数降序
    ranked = sorted(aggregated.values(), key=lambda x: x["correction_count"], reverse=True)
    return ranked[:top_n]

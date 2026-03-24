"""将 RAG 检索结果组装为上下文文本，注入 VLM Prompt"""


def build_context(
    fridge_state: dict[str, int],
    frequent_items: list[dict],
    alias_mappings: list[dict],
    categories: list[dict],
) -> str:
    """将各模块的检索结果拼装为 Prompt 上下文段落

    Args:
        fridge_state: 上次关门后冰箱内物品 {名称: 数量}
        frequent_items: 该用户高频物品 [{"name", "description", "count"}, ...]
        alias_mappings: 修正记录 [{"original", "corrected"}, ...]
        categories: 品类库 [{"category", "description"}, ...]

    Returns:
        格式化的上下文字符串，为空则返回空字符串
    """
    sections = []

    # 1. 冰箱上次状态（带数量）
    if fridge_state:
        items_str = "、".join(
            f"{name} x{qty}" if qty > 1 else name
            for name, qty in fridge_state.items()
        )
        sections.append(f"上次关门后冰箱内有：{items_str}")

    # 2. 用户常见物品
    if frequent_items:
        lines = []
        for item in frequent_items:
            desc = f"（{item['description']}）" if item.get("description") else ""
            lines.append(f"  - {item['name']}{desc}")
        sections.append("该用户常见物品：\n" + "\n".join(lines))

    # 3. 历史修正记录（按正确物品名聚合，多次修正的物品加权警告）
    if alias_mappings:
        lines = []
        for m in alias_mappings:
            count = m.get("correction_count", 1)
            wrong = "、".join(f"\"{w}\"" for w in m.get("wrong_names", []))
            desc = m.get("description", "")
            if count >= 2:
                # 多次修正：加重警告
                line = f"  - 「{m['corrected']}」已被修正{count}次，请务必注意！曾被错误识别为：{wrong}"
                if desc:
                    line += f"\n    正确外观特征：{desc}"
            else:
                line = f"  - 曾将「{m['corrected']}」错误识别为 {wrong}"
                if desc:
                    line += f"（{desc}）"
            lines.append(line)
        sections.append(
            "历史修正记录（以下物品曾被错误识别，请格外注意避免重复犯错）：\n"
            + "\n".join(lines)
        )

    # 4. 已知食物品类（替代原来的"其他用户常见物品"）
    if categories:
        lines = []
        for cat in categories:
            desc = f"（{cat['description']}）" if cat.get("description") else ""
            lines.append(f"  - {cat['category']}{desc}")
        sections.append(
            "已知食物类别：\n"
            + "\n".join(lines)
        )

    if not sections:
        return ""

    return "【用户上下文】\n" + "\n\n".join(sections)

"""查询用户当前冰箱状态：基于 InventoryEvent SQL 聚合计算"""

from sqlalchemy import func, case, literal_column
from sqlmodel import Session, select

from storage.models import InventoryEvent


def get_last_fridge_state(db: Session, user_id: int) -> dict[str, int]:
    """基于所有历史事件 SQL 聚合计算当前冰箱库存，按品类归一化，返回 {品类名: 数量}"""
    # 用 category 归一化（category 为空时回退到 item）
    key_expr = func.coalesce(
        func.nullif(InventoryEvent.category, ""),
        InventoryEvent.item,
    ).label("key")

    qty_expr = func.sum(
        case(
            (InventoryEvent.action == "put_in", InventoryEvent.quantity),
            else_=-InventoryEvent.quantity,
        )
    ).label("net_qty")

    stmt = (
        select(key_expr, qty_expr)
        .where(InventoryEvent.user_id == user_id)
        .group_by(key_expr)
        .having(qty_expr > 0)
    )

    return {row.key: int(row.net_qty) for row in db.exec(stmt).all()}

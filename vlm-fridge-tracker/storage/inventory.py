"""库存查询逻辑：写入事件、查询当前库存"""

from typing import Optional

from sqlmodel import Session, select, col

from storage.models import User, AnalysisSession, InventoryEvent, ItemKnowledge


def get_user(session: Session, key_code: str) -> Optional[User]:
    """根据 key_code 查找用户，不存在返回 None"""
    return session.exec(select(User).where(User.key_code == key_code)).first()


def create_user(session: Session, key_code: str) -> User:
    """创建新用户"""
    user = User(key_code=key_code)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def list_all_users(session: Session) -> list[User]:
    """列出所有用户"""
    return list(session.exec(select(User).order_by(col(User.created_at))).all())


def save_session_and_events(
    db: Session,
    user: User,
    video_path: str,
    segment_index: int,
    keyframe_dir: str,
    raw_response: str,
    events: list[dict],
) -> AnalysisSession:
    """保存一次分析会话及其所有事件"""
    analysis = AnalysisSession(
        user_id=user.id,
        video_path=video_path,
        segment_index=segment_index,
        keyframe_dir=keyframe_dir,
        raw_response=raw_response,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    for evt in events:
        item_name = evt["item"]
        inv_event = InventoryEvent(
            session_id=analysis.id,
            user_id=user.id,
            action=evt["action"],
            item=item_name,
            original_item=item_name,
            quantity=evt.get("quantity", 1),
            confidence=evt.get("confidence", 0.0),
            description=evt.get("description", ""),
        )
        db.add(inv_event)

    db.commit()
    return analysis


def get_current_inventory(db: Session, user_id: int) -> dict[str, int]:
    """计算用户当前库存：聚合所有 put_in 和 take_out 事件，负数钳制为 0"""
    events = db.exec(
        select(InventoryEvent)
        .where(InventoryEvent.user_id == user_id)
        .order_by(col(InventoryEvent.timestamp))
    ).all()

    inventory: dict[str, int] = {}
    for evt in events:
        if evt.action == "put_in":
            inventory[evt.item] = inventory.get(evt.item, 0) + evt.quantity
        elif evt.action == "take_out":
            inventory[evt.item] = inventory.get(evt.item, 0) - evt.quantity
            if inventory[evt.item] < 0:
                inventory[evt.item] = 0

    # 移除数量为 0 的物品
    inventory = {k: v for k, v in inventory.items() if v > 0}

    return inventory


def get_session_events(db: Session, session_id: int) -> list[InventoryEvent]:
    """获取某次分析会话的所有事件"""
    return list(db.exec(
        select(InventoryEvent)
        .where(InventoryEvent.session_id == session_id)
        .order_by(col(InventoryEvent.id))
    ).all())


def correct_event(
    db: Session,
    event: InventoryEvent,
    new_item_name: str = "",
    new_quantity: Optional[int] = None,
    new_action: str = "",
) -> None:
    """修正某个事件的物品名称、数量或操作方向"""
    if new_item_name:
        event.item = new_item_name
    if new_quantity is not None:
        event.quantity = new_quantity
    if new_action in ("put_in", "take_out"):
        event.action = new_action
    event.is_corrected = True
    db.add(event)
    db.commit()


def add_manual_event(
    db: Session,
    session_id: int,
    user_id: int,
    action: str,
    item: str,
    quantity: int = 1,
) -> InventoryEvent:
    """手动补录遗漏的物品事件"""
    event = InventoryEvent(
        session_id=session_id,
        user_id=user_id,
        action=action,
        item=item,
        original_item="",
        is_corrected=True,
        quantity=quantity,
        confidence=1.0,
        description="用户手动补录",
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


MAX_KNOWLEDGE_PER_ITEM = 5  # 同一物品最多保留的知识记录数


def save_to_knowledge(
    db: Session, user_id: int, event: InventoryEvent, embedding: str = ""
) -> None:
    """将事件信息存入物品知识库供 RAG 使用。

    同一 user_id + item_name 最多保留 MAX_KNOWLEDGE_PER_ITEM 条记录，
    超出时淘汰最旧的 vlm_accepted 记录；若全是 user_corrected 则淘汰最旧的。
    """
    source = "user_corrected" if event.is_corrected else "vlm_accepted"

    # 检查是否有完全重复的描述（同用户、同物品、同描述），有则跳过
    if event.description:
        dup = db.exec(
            select(ItemKnowledge)
            .where(ItemKnowledge.user_id == user_id)
            .where(ItemKnowledge.item_name == event.item)
            .where(ItemKnowledge.description == event.description)
        ).first()
        if dup:
            # 描述完全相同，只升级 source 优先级
            if source == "user_corrected" and dup.source != "user_corrected":
                dup.source = source
                if embedding:
                    dup.embedding = embedding
                db.add(dup)
                db.commit()
            return

    # 写入新记录
    knowledge = ItemKnowledge(
        user_id=user_id,
        item_name=event.item,
        original_name=event.original_item,
        description=event.description,
        source=source,
        embedding=embedding,
    )
    db.add(knowledge)
    db.commit()

    # 检查是否超出上限，超出则淘汰
    existing = db.exec(
        select(ItemKnowledge)
        .where(ItemKnowledge.user_id == user_id)
        .where(ItemKnowledge.item_name == event.item)
        .order_by(col(ItemKnowledge.created_at))
    ).all()

    if len(existing) > MAX_KNOWLEDGE_PER_ITEM:
        # 优先淘汰最旧的 vlm_accepted
        to_remove = len(existing) - MAX_KNOWLEDGE_PER_ITEM
        candidates = [r for r in existing if r.source == "vlm_accepted"]
        if len(candidates) < to_remove:
            # vlm_accepted 不够淘汰，再从最旧的记录中补
            candidates = list(existing)
        for r in candidates[:to_remove]:
            db.delete(r)
        db.commit()

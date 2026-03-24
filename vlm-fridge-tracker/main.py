"""主入口：串联所有模块，读取视频 → 提帧 → 调VLM → 存入数据库 → 打印结果"""

import argparse
import os
import sys
import concurrent.futures
from datetime import datetime

import cv2

from pipeline import video_loader, door_detector, door_detector_vlm, segment_extractor, keyframe_selector
from pipeline.direction_inferrer import infer_directions
from pipeline.output import capture_stdout
from vlm import gemini_client
from models.schemas import AnalysisResult
from storage.database import init_db, get_session
from sqlmodel import select
from storage.models import AnalysisSession, ItemCategory
from storage.inventory import (
    get_user, create_user, list_all_users,
    save_session_and_events, get_session_events,
    get_current_inventory, get_current_inventory_by_category,
    correct_event, add_manual_event, save_to_knowledge,
)
from rag.state_tracker import get_last_fridge_state
from rag.user_history import get_frequent_items, get_alias_mappings
from rag.cross_user_matcher import get_known_categories
from rag.prompt_enhancer import build_context
from rag.embedding import get_embedding, get_embeddings_batch, serialize_embedding, batch_cosine_similarity, deserialize_embedding
from rag.category_init import init_categories
from rag.category_matcher import match_category
from rag.knowledge_retriever import retrieve_relevant_knowledge

import config


# ── 显示辅助 ──────────────────────────────────────────────────


def _print_rag_context(
    fridge_state: dict[str, int],
    frequent_items: list[dict],
    alias_mappings: list[dict],
    categories: list[dict],
) -> None:
    """可视化打印 RAG 上下文详情"""
    print("\n  ┌─ RAG 上下文详情 ─────────────────────────")

    if fridge_state:
        items_str = ", ".join(
            f"{name} x{qty}" if qty > 1 else name
            for name, qty in fridge_state.items()
        )
        print(f"  │ 冰箱上次状态: {items_str}")
    else:
        print("  │ 冰箱上次状态: （无记录）")

    if frequent_items:
        items_str = ", ".join(f"{i['name']}(x{i['count']})" for i in frequent_items[:5])
        print(f"  │ 用户高频物品: {items_str}")
    else:
        print("  │ 用户高频物品: （无记录）")

    if alias_mappings:
        parts = []
        for m in alias_mappings[:5]:
            count = m.get("correction_count", 1)
            wrong = ",".join(m.get("wrong_names", []))
            tag = f"x{count}" if count >= 2 else ""
            parts.append(f"{wrong}→{m['corrected']}{tag}")
        print(f"  │ 历史修正记录: {', '.join(parts)}")
    else:
        print("  │ 历史修正记录: （无）")

    if categories:
        cat_str = ", ".join(c["category"] for c in categories[:10])
        remaining = len(categories) - 10
        if remaining > 0:
            cat_str += f" ...等{len(categories)}个品类"
        print(f"  │ 已知品类库:   {cat_str}")

    print("  └───────────────────────────────────────────")


def _print_rag_match_result(event_item: str, match_result: dict) -> None:
    """可视化打印单个物品的 RAG 品类匹配结果"""
    cat = match_result["category"]
    method = match_result["matched_by"]
    sim = match_result.get("similarity", 0)

    method_label = {
        "embedding": f"embedding匹配(相似度:{sim})",
        "gemini": "Gemini归类",
        "new": "新建品类",
        "fallback": "匹配失败,使用原名",
        "none": "品类库为空",
    }.get(method, method)

    if cat != event_item:
        print(f"      RAG: \"{event_item}\" → 品类「{cat}」 [{method_label}]")
    else:
        print(f"      RAG: \"{event_item}\" → 品类「{cat}」 [{method_label}]")


def _find_rag_match(
    item_name: str,
    fridge_state: dict[str, int],
    frequent_items: list[dict],
    alias_mappings: list[dict],
) -> str:
    """检查 VLM 识别结果是否精确匹配到 RAG 中的已知信息（仅用于显示标注）

    使用精确匹配避免子串误匹配（如"奶"匹配"牛奶"和"酸奶"）。
    此函数不影响识别结果，仅用于终端输出的辅助标注。
    """
    item_lower = item_name.lower().strip()

    # 1. 精确匹配历史修正记录
    for m in alias_mappings:
        if m["corrected"].lower().strip() == item_lower:
            wrong = "、".join(m.get("wrong_names", []))
            count = m.get("correction_count", 1)
            return f"修正记录「{m['corrected']}」(曾误识别为{wrong}，修正{count}次)"

    # 2. 精确匹配用户高频物品
    for fi in frequent_items:
        if fi["name"].lower().strip() == item_lower:
            return f"高频物品「{fi['name']}」(出现{fi['count']}次)"

    # 3. 精确匹配冰箱当前状态
    for fs_item, qty in fridge_state.items():
        if fs_item.lower().strip() == item_lower:
            qty_str = f" x{qty}" if qty > 1 else ""
            return f"冰箱中已有「{fs_item}」{qty_str}"

    return ""



# ── 帧选择 ────────────────────────────────────────────────────


def _pick_item_frames(
    keyframes: list,
    raw_response: str,
    item_name: str,
    item_desc: str = "",
    max_frames: int = 3,
) -> list:
    """从关键帧中挑出包含指定物品的帧（基于 hand_observations 的关联）

    fallback：匹配不到具体物品时，返回所有【手中持有任意物品】的帧；再匹配不到则返回首中尾帧
    """
    import json as _json

    try:
        data = _json.loads(raw_response)
        observations = data.get("hand_observations", []) if isinstance(data, dict) else []
        
        target_indices = []
        all_holding_indices = []
        
        for obs in observations:
            if not isinstance(obs, dict): continue
            holding_item = obs.get("holding_item", "")
            if not holding_item or holding_item in ("无", "none", "空", "None"): continue
            
            idx = obs.get("frame_number", 0) - 1
            if not (0 <= idx < len(keyframes)): continue
            
            all_holding_indices.append(idx)
            
            # 严谨的子串包含匹配
            if item_name in holding_item or holding_item in item_name:
                if idx not in target_indices:
                    target_indices.append(idx)
                    
        # 1. 优先返回精确匹配到的目标帧
        if target_indices:
            return [keyframes[i] for i in target_indices[:max_frames]]
            
        # 2. 严谨的 Fallback: 名字没对上，但画面里确实有手拿东西的动作
        # 把"手拿任何东西"的帧发给二次确认模型，缩小它的注意力范围，坚决不发首尾背景帧
        if all_holding_indices:
            selected = sorted(list(set(all_holding_indices)))
            if len(selected) > max_frames:
                step = len(selected) / max_frames
                selected = [selected[int(i * step)] for i in range(max_frames)]
            return [keyframes[i] for i in selected]
            
    except Exception as e:
        print(f"      帧匹配异常: {e}")

    # 3. 最差的兜底（没检测到手持物，极少发生）
    if len(keyframes) <= max_frames:
        return keyframes
    mid = len(keyframes) // 2
    return [keyframes[0], keyframes[mid], keyframes[-1]]



# ── 修正流程 ──────────────────────────────────────────────────


def _correction_flow(db, user, session_id, keyframes):
    """单个片段的修正流程：用户检查识别结果、修正、补录、品类匹配"""
    events = get_session_events(db, session_id)
    if not events:
        return

    print(f"\n  识别结果:")
    for idx, evt in enumerate(events):
        direction = "放入" if evt.action == "put_in" else "取出"
        print(f"    {idx + 1}. [{direction}] {evt.item} x{evt.quantity}  置信度:{evt.confidence:.0%}  {evt.description}")

    answer = input("\n  是否需要修正？(y/n，回车默认n): ").strip().lower()
    if answer == "y":
        for idx, evt in enumerate(events):
            direction = "放入" if evt.action == "put_in" else "取出"
            print(f"    {idx + 1}. [{direction}] {evt.item} x{evt.quantity}")
            changes = []

            # 修正物品名称
            new_name = input(f"       物品名称 → 修正为（回车跳过）: ").strip()

            # 修正数量
            qty_input = input(f"       数量 {evt.quantity} → 修正为（回车跳过）: ").strip()
            new_qty = int(qty_input) if qty_input.isdigit() and int(qty_input) > 0 else None

            # 修正操作方向
            opposite = "取出" if evt.action == "put_in" else "放入"
            flip = input(f"       操作方向 [{direction}] → 改为{opposite}？(y/n，回车跳过): ").strip().lower()
            new_action = ("take_out" if evt.action == "put_in" else "put_in") if flip == "y" else ""

            # 有任何修正则执行
            if new_name or new_qty is not None or new_action:
                correct_event(db, evt, new_item_name=new_name, new_quantity=new_qty, new_action=new_action)
                if new_name:
                    changes.append(f"{evt.original_item} → {new_name}")
                if new_qty is not None:
                    changes.append(f"数量 → {new_qty}")
                if new_action:
                    changes.append(f"{direction} → {opposite}")
                print(f"       已修正: {', '.join(changes)}")

                # 名称修正后，让 Gemini 重新描述正确物品的外观
                if new_name and keyframes:
                    print(f"       重新描述中...")
                    try:
                        import json as _json
                        # 从 hand_observations 精准定位包含该物品的帧
                        session_obj = db.get(AnalysisSession, session_id)
                        item_frames = _pick_item_frames(
                            keyframes,
                            session_obj.raw_response if session_obj else "",
                            evt.original_item,
                            evt.description
                        )
                        redesc_raw = gemini_client.redescribe_item(
                            item_frames, new_name, evt.original_item
                        )
                        redesc_data = _json.loads(redesc_raw)
                        new_desc = redesc_data.get("description", "")
                        if new_desc:
                            evt.description = new_desc
                            db.add(evt)
                            db.commit()
                            print(f"       新描述: {new_desc}")
                    except Exception as e:
                        print(f"       重新描述失败: {e}")

    # 补录遗漏物品
    miss_answer = input("\n  是否有遗漏的物品？(y/n，回车默认n): ").strip().lower()
    while miss_answer == "y":
        action_input = input("    操作类型 (1=放入, 2=取出): ").strip()
        if action_input == "1":
            action = "put_in"
        elif action_input == "2":
            action = "take_out"
        else:
            print("    无效输入，跳过。")
            miss_answer = input("\n  还有遗漏吗？(y/n，回车默认n): ").strip().lower()
            continue

        item_name = input("    物品名称: ").strip()
        if not item_name:
            print("    名称为空，跳过。")
            miss_answer = input("\n  还有遗漏吗？(y/n，回车默认n): ").strip().lower()
            continue

        qty_input = input("    数量（回车默认1）: ").strip()
        quantity = int(qty_input) if qty_input.isdigit() else 1

        evt = add_manual_event(db, session_id, user.id, action, item_name, quantity)
        direction = "放入" if action == "put_in" else "取出"
        print(f"    已补录: [{direction}] {item_name} x{quantity}")

        # 补录的也加到 events 列表，后面统一写入知识库
        events.append(evt)

        miss_answer = input("\n  还有遗漏吗？(y/n，回车默认n): ").strip().lower()

    # 写入知识库 + 品类匹配（双轨向量生成）
    rag_texts = []
    cat_texts = []
    for evt in events:
        # 1. 知识库向量（包含易错线索，用于高精度召回）
        rag_text = f"正品名称: {evt.item}。外观特征: {evt.description}"
        original = getattr(evt, "original_item", None)
        if evt.is_corrected and original and original != evt.item:
            rag_text += f"。(注：该物品在视觉上极易被误识别为 '{original}')"
        rag_texts.append(rag_text.strip())
        
        # 2. 品类归类向量（纯净版，避免语义被易错线索污染）
        cat_texts.append(f"{evt.item} {evt.description}".strip())

    try:
        # 合并请求一次性获取所有 embedding，节省网络时间
        all_vectors = get_embeddings_batch(rag_texts + cat_texts)
        half = len(events)
        rag_vectors, cat_vectors = all_vectors[:half], all_vectors[half:]
    except Exception as e:
        print(f"  ⚠ embedding 批量获取失败: {e}，跳过向量写入")
        rag_vectors = cat_vectors = [None] * len(events)

    print("\n  ┌─ RAG 品类匹配 ─────────────────────────────")
    for idx, (evt, rag_vec, cat_vec) in enumerate(zip(events, rag_vectors, cat_vectors)):
        emb = serialize_embedding(rag_vec) if rag_vec else ""

        # 知识库写入门控：防止不可靠的识别结果污染知识库
        stage1_stage2_disagree = evt.original_item and evt.original_item != evt.item and not evt.is_corrected
        should_save = evt.is_corrected or (evt.confidence >= config.KNOWLEDGE_SAVE_THRESHOLD and not stage1_stage2_disagree)

        if should_save:
            save_to_knowledge(db, user.id, evt, embedding=emb)
        elif stage1_stage2_disagree:
            print(f"  │ ⊘ \"{evt.item}\" 未写入知识库（Stage1「{evt.original_item}」与Stage2「{evt.item}」不一致，未经用户确认）")
        else:
            reason = f"置信度 {evt.confidence:.0%} < {config.KNOWLEDGE_SAVE_THRESHOLD:.0%}"
            print(f"  │ ⊘ \"{evt.item}\" 未写入知识库（{reason}，未经用户修正）")

        # 品类匹配：必须传入纯净版 embedding(cat_vec)，避免归类混乱
        match_result = match_category(
            db, evt.item, evt.description, precomputed_embedding=cat_vec
        )
        evt.category = match_result["category"]
        db.add(evt)
        _print_rag_match_result(evt.item, match_result)
    db.commit()
    print("  └───────────────────────────────────────────────")



# ── 主管线 ────────────────────────────────────────────────────


def process_video(video_path: str, user_key: str, is_new_user: bool = False, debug: bool = False) -> None:
    """处理单个视频文件的完整流程"""
    config.init_genai()
    init_db()

    with capture_stdout() as logger, get_session() as db:
        user = get_user(db, user_key)
        if user is None:
            if is_new_user:
                user = create_user(db, user_key)
                print(f"已创建新用户: {user.key_code}")
            else:
                all_users = list_all_users(db)
                print(f"错误: 用户 '{user_key}' 不存在。")
                if all_users:
                    names = ", ".join(u.key_code for u in all_users)
                    print(f"已有用户: {names}")
                    print("如需创建新用户，请加 --new-user 参数。")
                else:
                    print("当前无任何用户，请加 --new-user 参数创建首个用户。")
                sys.exit(1)

        print(f"用户: {user.key_code} (id={user.id})")
        print(f"视频: {os.path.abspath(video_path)}")
        print(f"配置: 模型={config.GEMINI_MODEL}, Prompt={config.PROMPT_TEMPLATE}, 门检测={config.DOOR_DETECTION_METHOD}")

        # 0. 初始化品类库（首次运行时自动生成）
        init_categories(db)

        # 1. 读取视频
        print(f"\n[1/5] 读取视频: {video_path}")
        frames = video_loader.load(video_path)
        print(f"      采样得到 {len(frames)} 帧")

        # 2. 检测门状态
        print(f"[2/5] 检测门状态 (方式: {config.DOOR_DETECTION_METHOD})...")
        if config.DOOR_DETECTION_METHOD == "vlm":
            states, motions = door_detector_vlm.detect(frames, debug=debug)
        else:
            states, motions = door_detector.detect(frames, debug=debug)
        open_count = sum(1 for s in states if s == door_detector.DoorState.OPEN)
        print(f"      开门帧: {open_count}, 关门帧: {len(states) - open_count}")

        # 3. 切割片段
        print("[3/5] 切割开关门片段...")
        segments = segment_extractor.extract(frames, states, motions)
        print(f"      检测到 {len(segments)} 次开关门操作")

        if not segments:
            print("未检测到开关门操作，退出。")
            return

        # 4. 逐片段分析（每个片段分析后立即修正+品类匹配，确保下一片段的 RAG 上下文准确）
        saved_session_ids = []
        failed_segments = []

        for i, (segment, seg_motions) in enumerate(segments):
            print(f"\n[4/5] 分析第 {i + 1}/{len(segments)} 次操作 ({len(segment)} 帧)...")

            try:
                # 构建/更新 RAG 上下文（每个片段前重新查询 DB，前一片段的事件已 commit）
                fridge_state = get_last_fridge_state(db, user.id)
                frequent_items = get_frequent_items(db, user.id)
                alias_mappings = get_alias_mappings(db, user.id)
                query_items = list(fridge_state.keys()) + [fi["name"] for fi in frequent_items]
                categories = get_known_categories(db, query_items=query_items or None)

                # 预加载全量品类库 + embedding，供 Stage 2 按物品相似度筛选
                all_cat_records = db.exec(select(ItemCategory)).all()
                all_cat_with_emb = [
                    ({"category": c.category, "description": c.description},
                     deserialize_embedding(c.embedding))
                    for c in all_cat_records if c.embedding
                ]

                # 仅作终端打印显示
                _print_rag_context(fridge_state, frequent_items, alias_mappings, categories)

                # 提取关键帧（运动量加权采样）
                keyframes = keyframe_selector.select(segment, seg_motions)
                print(f"      提取 {len(keyframes)} 张关键帧")

                # 保存关键帧图片
                keyframe_dir = os.path.join("output", "keyframes", f"segment_{i + 1}")
                os.makedirs(keyframe_dir, exist_ok=True)
                for j, kf in enumerate(keyframes):
                    path = os.path.join(keyframe_dir, f"frame_{j + 1}.jpg")
                    cv2.imwrite(path, kf)
                print(f"      关键帧已保存至 {keyframe_dir}/")

                # Stage 1: 盲猜（不注入 RAG，避免上下文污染）
                print(f"      [Stage 1] 调用 {config.GEMINI_MODEL} 盲猜分析中 (无 RAG)...")
                raw_response = gemini_client.analyze(keyframes, rag_context=None)

                # 解析结果
                try:
                    result = AnalysisResult.from_json(raw_response)
                except Exception as e:
                    print(f"      解析失败: {e}")
                    print(f"      原始响应: {raw_response}")
                    failed_segments.append(i + 1)
                    continue

                # v4 方向推导：根据 hand_position 序列修正事件方向
                if config.PROMPT_TEMPLATE == "v4":
                    result = infer_directions(result)

                # 打印分析结果
                print(f"\n[5/5] 第 {i + 1} 次操作分析结果:")
                print("-" * 50)

                if result.hand_observations:
                    print("\n  手部观察:")
                    for obs in result.hand_observations:
                        dir_label = {"into_fridge": "→冰箱", "out_of_fridge": "冰箱→", "unclear": "不明"}.get(obs.direction, obs.direction)
                        print(f"    帧{obs.frame_number}: 手持[{obs.holding_item}] 方向:{dir_label}")

                if not result.events:
                    print("  未检测到物品进出。")
                else:
                    for event in result.events:
                        direction = "放入 ←" if event.action.value == "put_in" else "取出 →"
                        print(
                            f"  {direction} {event.item} x{event.quantity}"
                            f"  (置信度: {event.confidence:.0%})"
                        )
                        if event.description:
                            print(f"         {event.description}")
                        # 标注 RAG 匹配情况
                        rag_tag = _find_rag_match(event.item, fridge_state, frequent_items, alias_mappings)
                        if rag_tag:
                            print(f"         [RAG] {rag_tag}")

                # Stage 2: 动态 RAG 二次确认
                stage1_items = [event.item for event in result.events]
                if result.events:
                    import json as _json
                    print(f"\n  [Stage 2] 结合动态 RAG 进行二次确认...")

                    # 批量预算所有物品的 embedding（一次 API 调用），复用于知识检索 + 品类筛选
                    item_descs = [f"{event.item} {event.description}".strip() for event in result.events]
                    try:
                        item_embeddings = get_embeddings_batch(item_descs)
                    except Exception as e:
                        print(f"    ⚠ 物品 embedding 批量计算失败: {e}")
                        item_embeddings = [None] * len(result.events)

                    # 在主线程预先为每个物品查询向量库，避免 SQLite 跨线程并发查询报错
                    refine_payloads = []
                    for event, item_emb in zip(result.events, item_embeddings):
                        desc = f"{event.item} {event.description}".strip()
                        knowledge = retrieve_relevant_knowledge(
                            db, user.id, desc, top_k=3, precomputed_embedding=item_emb
                        )

                        # 精准筛选相关的历史修正记录（两层匹配）
                        # 第1层：Stage 1 猜测名直接命中 alias 的 wrong_names（精确字符串匹配）
                        # 第2层：知识检索结果的物品名命中 alias 的 corrected（跨语言/同义词桥接）
                        retrieved_names = {k["item_name"] for k in knowledge}
                        relevant_aliases = [
                            m for m in alias_mappings
                            if event.item in m.get("wrong_names", [])
                            or m["corrected"] in retrieved_names
                        ]

                        # 用 alias 的正确名做补充检索，打破 Stage 1 错误锚定
                        if relevant_aliases:
                            seen = {k["item_name"] for k in knowledge}
                            for alias in relevant_aliases:
                                extra = retrieve_relevant_knowledge(db, user.id, alias["corrected"], top_k=1)
                                for k in extra:
                                    if k["item_name"] not in seen:
                                        knowledge.append(k)
                                        seen.add(k["item_name"])
                            alias_names = ", ".join(a["corrected"] for a in relevant_aliases)
                            print(f"    ⚠ \"{event.item}\" 命中历史修正记录，可能实际是: {alias_names}")

                        if knowledge:
                            names = ", ".join(f"{k['item_name']}({k['similarity']})" for k in knowledge)
                            print(f"    🔍 为 \"{event.item}\" 召回经验: {names}")
                        refine_payloads.append((event, desc, knowledge, relevant_aliases, item_emb))

                    # 保存 Stage 2 每个物品的专属关键帧
                    stage2_kf_dir = os.path.join("output", "keyframes_stage2", f"segment_{i + 1}")
                    os.makedirs(stage2_kf_dir, exist_ok=True)

                    def _refine_task(payload):
                        evt, desc, knowledge, relevant_aliases, item_emb = payload
                        try:
                            item_frames = _pick_item_frames(keyframes, raw_response, evt.item)

                            # 保存该物品的 Stage 2 关键帧
                            safe_name = evt.item.replace("/", "_").replace(" ", "_")
                            for fi, frame in enumerate(item_frames):
                                save_path = os.path.join(stage2_kf_dir, f"{safe_name}_{fi + 1}.jpg")
                                cv2.imwrite(save_path, frame)

                            knowledge_text = ""
                            if knowledge:
                                knowledge_text = "\n【用户历史视觉与纠错经验 (高度相关)】\n"
                                for k in knowledge:
                                    sim = k['similarity']
                                    knowledge_text += f" - 物品: {k['item_name']}, 外观特征: {k['description']} (匹配度:{sim})\n"
                                    if k.get('source') == 'user_corrected' and k['original_name'] and k['original_name'] != k['item_name']:
                                        knowledge_text += f"   (注: 该物品在视觉上曾被大模型误认为 '{k['original_name']}')\n"

                            # 动态构建专属 Context
                            # 放入时不需要参考冰箱内的现有状态；取出时仅参考现有状态
                            stage2_fridge_state = fridge_state if evt.action.value == "take_out" else {}

                            # 从全量品类库中筛选与当前物品最相关的 top 10（embedding 相似度）
                            if item_emb is not None and all_cat_with_emb:
                                cat_dicts, cat_embs = zip(*all_cat_with_emb)
                                sims = batch_cosine_similarity(item_emb, list(cat_embs))
                                top_indices = sims.argsort()[::-1][:10]
                                filtered_categories = [cat_dicts[i] for i in top_indices]
                            else:
                                filtered_categories = categories[:10]

                            # 精准注入与当前物品相关的历史修正记录（仅当 Stage 1 猜测名命中 alias 时）
                            # 未命中则传空列表，避免无关 alias 干扰
                            base_context = build_context(stage2_fridge_state, [], relevant_aliases, filtered_categories)
                            focused_context = f"{base_context}{knowledge_text}"
                            
                            refine_raw = gemini_client.refine_item(item_frames, evt.description, focused_context)
                            return evt, _json.loads(refine_raw)
                        except Exception as e:
                            print(f"    ✗ \"{evt.item}\" 二次识别失败: {e}")
                            return evt, None

                    # 使用多线程并行执行当前片段中所有物品的二次确认（限制 max_workers=3 防止触发并发限制）
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_event = {executor.submit(_refine_task, p): p[0] for p in refine_payloads}
                        for future in concurrent.futures.as_completed(future_to_event):
                            event = future_to_event[future]
                            evt_result, refine_data = future.result()
                            if refine_data:
                                new_item = refine_data.get("item", event.item)
                                new_conf = refine_data.get("confidence", event.confidence)
                                new_desc = refine_data.get("description", event.description)

                                name_changed = new_item != event.item
                                if name_changed and new_conf < event.confidence:
                                    # Stage 2 给了不同名字但自身置信度更低，不采信
                                    print(f"    ✗ \"{event.item}\" → \"{new_item}\" 被拒绝（置信度 {new_conf:.0%} < 原 {event.confidence:.0%}），保留原结果")
                                elif name_changed or new_conf > event.confidence:
                                    print(f"    ✓ \"{event.item}\" → \"{new_item}\" (置信度: {event.confidence:.0%} → {new_conf:.0%}) [二次确认]")
                                    print(f"      {new_desc}")
                                    event.item = new_item
                                    event.confidence = new_conf
                                    event.description = new_desc
                                else:
                                    print(f"    - \"{event.item}\" 二次确认无误，保留原结果")

                if result.fridge_state_after:
                    print(f"\n  冰箱内物品: {', '.join(result.fridge_state_after)}")

                if result.notes:
                    print(f"  备注: {result.notes}")

                print("-" * 50)

                # 存入数据库（original_item 保存 Stage 1 原始名，供 alias 系统追踪错误模式）
                events_data = [
                    {
                        "action": event.action.value,
                        "item": event.item,
                        "original_item": stage1_items[idx] if idx < len(stage1_items) else event.item,
                        "quantity": event.quantity,
                        "confidence": event.confidence,
                        "description": event.description,
                    }
                    for idx, event in enumerate(result.events)
                ]

                analysis = save_session_and_events(
                    db=db,
                    user=user,
                    video_path=video_path,
                    segment_index=i,
                    keyframe_dir=keyframe_dir,
                    raw_response=raw_response,
                    events=events_data,
                )
                saved_session_ids.append(analysis.id)

                # 每个片段分析后立即修正和品类匹配，确保下一片段的 RAG 上下文准确
                print("\n" + "=" * 60)
                print(f"  第 {i + 1} 次操作分析完成 — 请检查识别结果")
                print("=" * 60)
                _correction_flow(db, user, analysis.id, keyframes)

            except Exception as e:
                print(f"      ✗ 第 {i + 1} 次操作分析失败: {e}")
                failed_segments.append(i + 1)
                continue

        if failed_segments:
            print(f"\n  ⚠ 共 {len(failed_segments)} 个片段分析失败: {failed_segments}")

        # 打印最终入库记录
        print("\n" + "─" * 60)
        print("  本次录入记录（最终）:")
        for sid in saved_session_ids:
            events = get_session_events(db, sid)
            if not events:
                continue
            for evt in events:
                direction = "放入" if evt.action == "put_in" else "取出"
                corrected_tag = " [已修正]" if evt.is_corrected else ""
                print(f"    [{direction}] {evt.item} x{evt.quantity}{corrected_tag}")

        # 打印用户当前总库存（按品类聚合）
        inventory = get_current_inventory_by_category(db, user.id)
        print(f"\n{'─' * 60}")
        print(f"  用户 [{user.key_code}] 当前冰箱库存:")
        if inventory:
            for cat, qty in inventory.items():
                print(f"    • {cat} x{qty}")
        else:
            print("    （空）")
        print("─" * 60)

        # 打印 Gemini API 用量统计
        usage = gemini_client.get_usage_stats()
        if usage["call_count"] > 0:
            print(f"\n{'─' * 60}")
            print(f"  Gemini API 用量统计:")
            print(f"    调用次数:     {usage['call_count']}")
            print(f"    输入 tokens:  {usage['prompt_tokens']:,}")
            print(f"    输出 tokens:  {usage['candidates_tokens']:,}")
            print(f"    合计 tokens:  {usage['total_tokens']:,}")
            print("─" * 60)

        # 保存运行记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("output", "reports", f"run_{user_key}_{timestamp}.md")

    # with 块结束后 stdout 已恢复、db 已关闭
    logger.save(report_path)
    print(f"\n运行记录已保存至: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="冰箱物品追踪系统")
    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--user", "-u", default="default", help="用户 key code（默认: default）")
    parser.add_argument("--new-user", action="store_true", help="创建新用户（首次使用时需要）")
    parser.add_argument("--list-users", action="store_true", help="列出所有已注册用户")
    parser.add_argument("--debug", action="store_true", help="打印门状态检测的逐帧调试信息")
    args = parser.parse_args()

    # 列出用户
    if args.list_users:
        init_db()
        with get_session() as db:
            users = list_all_users(db)
            if users:
                print("已注册用户:")
                for u in users:
                    print(f"  • {u.key_code}  (创建于 {u.created_at.strftime('%Y-%m-%d %H:%M')})")
            else:
                print("当前无任何用户。使用 --new-user 创建。")
        return

    # 分析视频
    if not args.video:
        parser.error("请提供视频文件路径（或使用 --list-users 查看用户）")

    process_video(args.video, args.user, args.new_user, debug=args.debug)


if __name__ == "__main__":
    main()

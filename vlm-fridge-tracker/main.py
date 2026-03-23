"""主入口：串联所有模块，读取视频 → 提帧 → 调VLM → 存入数据库 → 打印结果"""

import argparse
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import cv2

from pipeline import video_loader, door_detector, door_detector_vlm, segment_extractor, keyframe_selector
from pipeline.direction_inferrer import infer_directions
from vlm import gemini_client
from models.schemas import AnalysisResult
from storage.database import init_db, get_session
from storage.models import AnalysisSession
from storage.inventory import (
    get_user, create_user, list_all_users,
    save_session_and_events, get_session_events, get_current_inventory,
    correct_event, add_manual_event, save_to_knowledge,
)
from rag.state_tracker import get_last_fridge_state
from rag.user_history import get_frequent_items, get_alias_mappings
from rag.cross_user_matcher import get_known_categories
from rag.prompt_enhancer import build_context
from rag.embedding import get_embeddings_batch, serialize_embedding
from rag.category_init import init_categories
from rag.category_matcher import match_category

import config


class OutputLogger:
    """双通道输出：同时写终端和缓冲区，运行结束后保存为 md 文件"""

    def __init__(self, original_stdout):
        self.original = original_stdout
        self.buffer: list[str] = []

    def write(self, text: str) -> None:
        self.original.write(text)
        self.buffer.append(text)

    def flush(self) -> None:
        self.original.flush()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = "".join(self.buffer)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 冰箱物品追踪 — 运行记录\n\n")
            f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("```\n")
            f.write(content)
            f.write("```\n")


@contextmanager
def _capture_stdout() -> Generator[OutputLogger, None, None]:
    """安全地捕获 stdout，异常时也能恢复"""
    logger = OutputLogger(sys.stdout)
    sys.stdout = logger
    try:
        yield logger
    finally:
        sys.stdout = logger.original


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
    """检查 VLM 识别结果是否匹配到 RAG 中的已知信息，返回匹配说明"""
    item_lower = item_name.lower()

    # 1. 检查是否匹配到历史修正记录（纠错后的正确名称）
    for m in alias_mappings:
        if m["corrected"].lower() in item_lower or item_lower in m["corrected"].lower():
            wrong = "、".join(m.get("wrong_names", []))
            count = m.get("correction_count", 1)
            desc = m.get("description", "")
            tag = f"匹配到RAG修正记录「{m['corrected']}」(曾被误识别为{wrong}，已修正{count}次)"
            if desc:
                tag += f"，外观：{desc[:30]}..."
            return tag

    # 2. 检查是否匹配到用户高频物品
    for fi in frequent_items:
        if fi["name"].lower() in item_lower or item_lower in fi["name"].lower():
            desc = fi.get("description", "")
            tag = f"匹配到RAG高频物品「{fi['name']}」(出现{fi['count']}次)"
            if desc:
                tag += f"，描述：{desc[:30]}..."
            return tag

    # 3. 检查是否在冰箱当前状态中
    for fs_item, qty in fridge_state.items():
        if fs_item.lower() in item_lower or item_lower in fs_item.lower():
            qty_str = f" x{qty}" if qty > 1 else ""
            return f"匹配到RAG冰箱状态：「{fs_item}」{qty_str}已在冰箱中"

    return ""


def _pick_item_frames(
    keyframes: list,
    raw_response: str,
    item_name: str,
    max_frames: int = 3,
) -> list:
    """从关键帧中挑出包含指定物品的帧（基于 hand_observations）

    fallback：匹配不到则返回首帧+中间帧+尾帧
    """
    import json as _json

    # 尝试从 hand_observations 中找到包含该物品的帧
    try:
        data = _json.loads(raw_response)
        observations = data.get("hand_observations", [])
        frame_indices = []
        for obs in observations:
            if item_name in obs.get("holding_item", ""):
                idx = obs.get("frame_number", 0) - 1  # frame_number 从 1 开始
                if 0 <= idx < len(keyframes) and idx not in frame_indices:
                    frame_indices.append(idx)
        if frame_indices:
            return [keyframes[i] for i in frame_indices[:max_frames]]
    except Exception as e:
        print(f"      帧匹配失败，使用默认帧: {e}")

    # fallback：首帧 + 中间帧 + 尾帧
    if len(keyframes) <= max_frames:
        return keyframes
    mid = len(keyframes) // 2
    return [keyframes[0], keyframes[mid], keyframes[-1]]


def _correction_flow(db, user, saved_session_ids, session_keyframes):
    """修正流程：用户检查识别结果、修正、补录、品类匹配"""
    print("\n" + "=" * 60)
    print("  分析完成 — 请检查识别结果")
    print("=" * 60)

    for sid in saved_session_ids:
        events = get_session_events(db, sid)
        if not events:
            continue

        print(f"\n  会话 #{sid} 识别结果:")
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
                    if new_name and sid in session_keyframes:
                        print(f"       重新描述中...")
                        try:
                            import json as _json
                            # 从 hand_observations 精准定位包含该物品的帧
                            session_obj = db.get(AnalysisSession, sid)
                            item_frames = _pick_item_frames(
                                session_keyframes[sid],
                                session_obj.raw_response if session_obj else "",
                                evt.original_item,
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

            evt = add_manual_event(db, sid, user.id, action, item_name, quantity)
            direction = "放入" if action == "put_in" else "取出"
            print(f"    已补录: [{direction}] {item_name} x{quantity}")

            # 补录的也加到 events 列表，后面统一写入知识库
            events.append(evt)

            miss_answer = input("\n  还有遗漏吗？(y/n，回车默认n): ").strip().lower()

        # 写入知识库 + 品类匹配（含 RAG 可视化）
        emb_texts = [f"{evt.item} {evt.description}".strip() for evt in events]
        try:
            emb_vectors = get_embeddings_batch(emb_texts)
        except Exception as e:
            print(f"  ⚠ embedding 获取失败: {e}，跳过知识库向量写入")
            emb_vectors = [None] * len(events)

        print("\n  ┌─ RAG 品类匹配 ─────────────────────────────")
        for evt, emb_vector in zip(events, emb_vectors):
            emb = serialize_embedding(emb_vector) if emb_vector else ""
            save_to_knowledge(db, user.id, evt, embedding=emb)

            # 品类匹配：传入预计算的 embedding，避免重复 API 调用
            match_result = match_category(
                db, evt.item, evt.description, precomputed_embedding=emb_vector
            )
            evt.category = match_result["category"]
            db.add(evt)
            _print_rag_match_result(evt.item, match_result)
        db.commit()
        print("  └───────────────────────────────────────────────")


def process_video(video_path: str, user_key: str, is_new_user: bool = False, debug: bool = False) -> None:
    """处理单个视频文件的完整流程"""
    config.init_genai()
    init_db()

    with _capture_stdout() as logger, get_session() as db:
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
        print(f"\n[1/6] 读取视频: {video_path}")
        frames = video_loader.load(video_path)
        print(f"      采样得到 {len(frames)} 帧")

        # 2. 检测门状态
        print(f"[2/6] 检测门状态 (方式: {config.DOOR_DETECTION_METHOD})...")
        if config.DOOR_DETECTION_METHOD == "vlm":
            states, motions = door_detector_vlm.detect(frames, debug=debug)
        else:
            states, motions = door_detector.detect(frames, debug=debug)
        open_count = sum(1 for s in states if s == door_detector.DoorState.OPEN)
        print(f"      开门帧: {open_count}, 关门帧: {len(states) - open_count}")

        # 3. 切割片段
        print("[3/6] 切割开关门片段...")
        segments = segment_extractor.extract(frames, states, motions)
        print(f"      检测到 {len(segments)} 次开关门操作")

        if not segments:
            print("未检测到开关门操作，退出。")
            return

        # 4. 构建 RAG 上下文
        print("\n[4/6] 构建 RAG 上下文...")
        fridge_state = get_last_fridge_state(db, user.id)
        frequent_items = get_frequent_items(db, user.id)
        alias_mappings = get_alias_mappings(db, user.id)
        query_items = list(fridge_state.keys()) + [i["name"] for i in frequent_items]
        categories = get_known_categories(db, query_items=query_items or None)
        rag_context = build_context(fridge_state, frequent_items, alias_mappings, categories)

        if rag_context:
            _print_rag_context(fridge_state, frequent_items, alias_mappings, categories)
        else:
            print("      无历史数据，跳过")

        # 5. 逐片段分析
        saved_session_ids = []
        session_keyframes = {}

        failed_segments = []

        for i, (segment, seg_motions) in enumerate(segments):
            print(f"\n[5/6] 分析第 {i + 1}/{len(segments)} 次操作 ({len(segment)} 帧)...")

            try:
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

                # 调用 VLM（带 RAG 上下文）
                print(f"      调用 {config.GEMINI_MODEL} 分析中...")
                raw_response = gemini_client.analyze(keyframes, rag_context)

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
                print(f"\n[6/6] 第 {i + 1} 次操作分析结果:")
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

                # 二次聚焦分析：低置信度物品（精选包含该物品的帧，而非全部关键帧）
                low_conf_events = [
                    e for e in result.events
                    if e.confidence < config.REFINE_CONFIDENCE_THRESHOLD
                ]
                if low_conf_events:
                    import json as _json
                    print(f"\n  ⚠ 发现 {len(low_conf_events)} 个低置信度物品，触发二次识别...")
                    for event in low_conf_events:
                        desc = f"{event.item} {event.description}".strip()
                        item_frames = _pick_item_frames(keyframes, raw_response, event.item)
                        print(f"    二次识别: \"{event.item}\" (置信度:{event.confidence:.0%}, 精选{len(item_frames)}帧)...")
                        try:
                            refine_raw = gemini_client.refine_item(item_frames, desc, rag_context)
                            refine_data = _json.loads(refine_raw)
                            new_item = refine_data.get("item", event.item)
                            new_conf = refine_data.get("confidence", event.confidence)
                            new_desc = refine_data.get("description", event.description)

                            if new_conf > event.confidence:
                                print(
                                    f"    ✓ \"{event.item}\" → \"{new_item}\""
                                    f"  (置信度: {event.confidence:.0%} → {new_conf:.0%})"
                                    f"  [二次识别]"
                                )
                                print(f"      {new_desc}")
                                event.item = new_item
                                event.confidence = new_conf
                                event.description = new_desc
                            else:
                                print(f"    - 二次识别未提升置信度，保留原结果")
                        except Exception as e:
                            print(f"    ✗ 二次识别失败: {e}")

                if result.fridge_state_after:
                    print(f"\n  冰箱内物品: {', '.join(result.fridge_state_after)}")

                if result.notes:
                    print(f"  备注: {result.notes}")

                print("-" * 50)

                # 存入数据库
                events_data = [
                    {
                        "action": event.action.value,
                        "item": event.item,
                        "quantity": event.quantity,
                        "confidence": event.confidence,
                        "description": event.description,
                    }
                    for event in result.events
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
                session_keyframes[analysis.id] = keyframes

            except Exception as e:
                print(f"      ✗ 第 {i + 1} 次操作分析失败: {e}")
                failed_segments.append(i + 1)
                continue

        if failed_segments:
            print(f"\n  ⚠ 共 {len(failed_segments)} 个片段分析失败: {failed_segments}")

        # 修正流程
        _correction_flow(db, user, saved_session_ids, session_keyframes)

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

        # 打印用户当前总库存
        inventory = get_current_inventory(db, user.id)
        print(f"\n{'─' * 60}")
        print(f"  用户 [{user.key_code}] 当前冰箱库存:")
        if inventory:
            for item, qty in inventory.items():
                print(f"    • {item} x{qty}")
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

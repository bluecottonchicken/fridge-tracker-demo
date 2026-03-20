"""主入口：串联所有模块，读取视频 → 提帧 → 调VLM → 存入数据库 → 打印结果"""

import argparse
import os
import sys

import cv2

from pipeline import video_loader, door_detector, segment_extractor, keyframe_selector
from vlm import gemini_client
from models.schemas import AnalysisResult
from storage.database import init_db, get_session
from storage.inventory import (
    get_user, create_user, list_all_users,
    save_session_and_events, get_session_events, get_current_inventory,
    correct_event, add_manual_event, save_to_knowledge,
)

import config


def process_video(video_path: str, user_key: str, is_new_user: bool = False) -> None:
    """处理单个视频文件的完整流程"""
    # 初始化数据库
    init_db()
    db = get_session()

    user = get_user(db, user_key)
    if user is None:
        if is_new_user:
            user = create_user(db, user_key)
            print(f"已创建新用户: {user.key_code}")
        else:
            # 用户不存在，提示已有用户列表
            all_users = list_all_users(db)
            print(f"错误: 用户 '{user_key}' 不存在。")
            if all_users:
                names = ", ".join(u.key_code for u in all_users)
                print(f"已有用户: {names}")
                print("如需创建新用户，请加 --new-user 参数。")
            else:
                print("当前无任何用户，请加 --new-user 参数创建首个用户。")
            db.close()
            sys.exit(1)

    print(f"用户: {user.key_code} (id={user.id})")

    # 1. 读取视频
    print(f"\n[1/5] 读取视频: {video_path}")
    frames = video_loader.load(video_path)
    print(f"      采样得到 {len(frames)} 帧")

    # 2. 检测门状态
    print("[2/5] 检测门状态...")
    states = door_detector.detect(frames)
    open_count = sum(1 for s in states if s == door_detector.DoorState.OPEN)
    print(f"      开门帧: {open_count}, 关门帧: {len(states) - open_count}")

    # 3. 切割片段
    print("[3/5] 切割开关门片段...")
    segments = segment_extractor.extract(frames, states)
    print(f"      检测到 {len(segments)} 次开关门操作")

    if not segments:
        print("未检测到开关门操作，退出。")
        db.close()
        return

    # 4. 逐片段分析
    saved_session_ids = []

    for i, segment in enumerate(segments):
        print(f"\n[4/5] 分析第 {i + 1}/{len(segments)} 次操作 ({len(segment)} 帧)...")

        # 提取关键帧
        keyframes = keyframe_selector.select(segment)
        print(f"      提取 {len(keyframes)} 张关键帧")

        # 保存关键帧图片
        keyframe_dir = os.path.join("output", "keyframes", f"segment_{i + 1}")
        os.makedirs(keyframe_dir, exist_ok=True)
        for j, kf in enumerate(keyframes):
            path = os.path.join(keyframe_dir, f"frame_{j + 1}.jpg")
            cv2.imwrite(path, kf)
        print(f"      关键帧已保存至 {keyframe_dir}/")

        # 调用 VLM
        print(f"      调用 {config.GEMINI_MODEL} 分析中...")
        raw_response = gemini_client.analyze(keyframes)

        # 解析结果
        try:
            result = AnalysisResult.from_json(raw_response)
        except Exception as e:
            print(f"      解析失败: {e}")
            print(f"      原始响应: {raw_response}")
            continue

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

    # ==================== 修正流程 ====================
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
                new_name = input(f"    {idx + 1}. [{direction}] {evt.item} → 修正为（回车跳过）: ").strip()
                if new_name:
                    correct_event(db, evt, new_name)
                    print(f"       已修正: {evt.original_item} → {new_name}")

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

        # 无论是否修正，都写入知识库
        for evt in events:
            save_to_knowledge(db, user.id, evt)

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

    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="冰箱物品追踪系统")
    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--user", "-u", default="default", help="用户 key code（默认: default）")
    parser.add_argument("--new-user", action="store_true", help="创建新用户（首次使用时需要）")
    parser.add_argument("--list-users", action="store_true", help="列出所有已注册用户")
    args = parser.parse_args()

    # 列出用户
    if args.list_users:
        init_db()
        db = get_session()
        users = list_all_users(db)
        if users:
            print("已注册用户:")
            for u in users:
                print(f"  • {u.key_code}  (创建于 {u.created_at.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("当前无任何用户。使用 --new-user 创建。")
        db.close()
        return

    # 分析视频
    if not args.video:
        parser.error("请提供视频文件路径（或使用 --list-users 查看用户）")

    process_video(args.video, args.user, args.new_user)


if __name__ == "__main__":
    main()

"""根据 hand_observations 的位置序列推导物品进出方向（v4 prompt 专用）"""

from models.schemas import AnalysisResult, Action

# 位置到数值的映射：越大越深入冰箱
_POS_RANK = {
    "outside_fridge": 0,
    "at_entrance": 1,
    "inside_fridge": 2,
}


def infer_directions(result: AnalysisResult) -> AnalysisResult:
    """根据 hand_observations 中的 hand_position 序列推导每个 event 的方向

    逻辑：
    - 追踪每个物品在 hand_observations 中的位置变化轨迹
    - 手持物品从外向内移动（rank 递增）→ put_in
    - 手持物品从内向外移动（rank 递减）→ take_out
    - 轨迹不明确 → 保留 VLM 原始判断
    """
    if not result.hand_observations or not result.events:
        return result

    # 按物品名聚合其位置轨迹
    # key: 物品名（小写），value: [(frame_number, position_rank)]
    item_trajectories: dict[str, list[tuple[int, int]]] = {}
    for obs in result.hand_observations:
        item = obs.holding_item.strip()
        if not item:
            continue
        rank = _POS_RANK.get(obs.hand_position, -1)
        if rank < 0:
            continue
        key = item.lower()
        if key not in item_trajectories:
            item_trajectories[key] = []
        item_trajectories[key].append((obs.frame_number, rank))

    # 对每条轨迹排序并判断方向
    item_directions: dict[str, str] = {}  # item_lower -> "put_in" / "take_out"
    for item_key, trajectory in item_trajectories.items():
        trajectory.sort(key=lambda x: x[0])  # 按帧号排序
        if len(trajectory) < 2:
            continue
        first_rank = trajectory[0][1]
        last_rank = trajectory[-1][1]
        if last_rank > first_rank:
            # 从外向内：put_in
            item_directions[item_key] = "put_in"
        elif last_rank < first_rank:
            # 从内向外：take_out
            item_directions[item_key] = "take_out"

    # 用推导结果修正 events 中的 action
    corrected_count = 0
    for event in result.events:
        event_key = event.item.strip().lower()
        inferred = item_directions.get(event_key)
        if inferred and inferred != event.action.value:
            event.action = Action(inferred)
            corrected_count += 1

    if corrected_count > 0:
        print(f"      [方向推导] 修正了 {corrected_count} 个事件的方向")

    return result

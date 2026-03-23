"""Prompt 模板，纯文本管理，不含任何 API 调用逻辑"""

SYSTEM_PROMPT = """你是一个冰箱物品追踪系统。你的任务是分析冰箱摄像头拍摄的连续帧图片，判断用户在此次操作中放入了什么物品、取出了什么物品。

核心原则：只有人手直接接触并移动的物品才算 put_in 或 take_out。

核心判断方法：
1. 逐帧追踪人手的位置和手中持有的物品。只有手明确握住/拿着的物品才是有效证据。
2. 判断手的运动方向：手从外部伸向冰箱内部并放下物品 → put_in；手从冰箱内部拿起物品并移出 → take_out。
3. 辅助验证：对比首尾帧冰箱内部变化，但仅用于验证手部动作的判断，不能单独作为 put_in/take_out 的依据。

必须遵守的规则：
1. 没有看到手接触某物品 → 该物品没有被放入或取出，即使它的位置发生了变化。
2. 冰箱门内侧储物格中的物品会随开关门而移动，这不是取出或放入操作。不要将门架上的物品报告为 take_out。
3. 仅因为首尾帧对比发现某物品"消失"或"出现"，但没有看到手拿着它，不要报告为事件。
4. 物品名称使用日常用语（如"牛奶"而非"乳制品饮料"）。
5. 如果无法确定具体物品，描述其外观特征（如"红色塑料袋装的东西"）。
6. 每个物品给出置信度 (0-1)。如果看不清或不确定，标注 confidence < 0.5。
7. 如果无法确定具体物品，必须详细描述容器特征：材质（塑料/玻璃/纸盒）、透明度、内容物颜色、形状、大小、是否有标签文字。禁止输出"食物容器"或"其他物品"等笼统描述。
8. 宁可漏报也不要误报。如果不确定手是否真的拿了某物品，不要报告该事件。"""


def build_refine_prompt(item_description: str, rag_context: str = "") -> str:
    """构建二次聚焦分析 Prompt，针对低置信度物品，可选注入 RAG 上下文"""
    context_block = f"\n{rag_context}\n\n请优先从上述已知物品中匹配，如果确实不匹配再给出新名称。\n\n" if rag_context else ""
    return f"""{context_block}以下帧中出现了一个物品，初步描述为："{item_description}"。
请仔细观察这个物品，辨认它具体是什么食物。

重点关注：
1. 容器材质（塑料/玻璃/纸盒/塑料袋）
2. 容器透明度，能否看到内容物
3. 内容物的颜色和形状
4. 是否有标签文字或品牌 logo

严格输出以下 JSON，不要输出任何其他内容：
{{
  "item": "物品名称",
  "confidence": 0.0-1.0,
  "description": "详细描述（材质、颜色、形状、标签等）"
}}"""


def build_redescribe_prompt(correct_name: str, wrong_name: str) -> str:
    """构建重新描述 Prompt：用户修正物品名后，让 Gemini 重新描述正确物品的外观"""
    return f"""你之前将图片中的一个物品识别为"{wrong_name}"，但实际上这个物品是"{correct_name}"。
请重新仔细观察图片中的这个物品（{correct_name}），描述它的外观特征。

重点描述：
1. 形状和大小
2. 颜色和材质
3. 包装特征（标签、品牌、文字）
4. 与"{wrong_name}"在外观上的区别

严格输出以下 JSON，不要输出任何其他内容：
{{
  "description": "详细外观描述"
}}"""


def build_user_prompt(num_frames: int, rag_context: str = "") -> str:
    """构建用户 Prompt，可选注入 RAG 上下文"""
    context_block = f"\n{rag_context}\n\n" if rag_context else ""
    return f"""{context_block}以下是按时间顺序排列的 {num_frames} 张冰箱摄像头截图。
图1是开门后的初始状态，图{num_frames}是关门前的最终状态。

请按以下步骤分析：
1. 逐帧检查是否出现人手，以及手是否直接握住/接触了某个物品。
2. 只有手明确接触并移动的物品才能记录为 put_in 或 take_out。
3. 忽略冰箱门内侧储物格中随门移动的物品——它们不是被取出或放入的。
4. 对比首帧和尾帧验证判断，但首尾帧对比仅用于辅助确认，不能单独作为事件依据。
5. 宁可漏报也不要误报。

严格输出以下 JSON，不要输出任何其他内容：
{{
  "hand_observations": [
    {{
      "frame_number": 帧编号,
      "holding_item": "手中物品描述",
      "direction": "into_fridge" 或 "out_of_fridge" 或 "unclear"
    }}
  ],
  "events": [
    {{
      "action": "put_in" 或 "take_out",
      "item": "物品名称",
      "quantity": 数量,
      "confidence": 0.0-1.0,
      "description": "简短描述（颜色、包装等特征）"
    }}
  ],
  "fridge_state_after": ["操作后冰箱内可见的所有物品列表"],
  "notes": "任何不确定或需要注意的观察"
}}"""

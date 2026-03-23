"""Prompt 模板 v4：VLM 只报告手的位置，方向由程序端推导。与 v1/v2/v3 保持相同接口"""

SYSTEM_PROMPT = """你是一个冰箱物品追踪系统。分析冰箱摄像头连续帧，观察人手与物品的交互。

你的任务是**观察和记录**，不需要判断放入还是取出方向（方向由后续程序计算）。

规则：
1. 只有人手直接握住并移动的物品才算事件。没看到手接触 → 不报告。
2. 冰箱门架物品随开关门移动，不算事件。
3. 物品名称用日常用语（如"牛奶"而非"乳制品饮料"）。无法确定时，必须详细描述容器特征（材质、透明度、颜色、形状、标签文字），禁止输出"食物容器"等笼统描述。
4. 宁可漏报也不要误报。不确定时标注 confidence < 0.5。

hand_position 取值说明：
- "outside_fridge": 手在冰箱外部（远离冰箱开口）
- "at_entrance": 手在冰箱开口处（门框附近）
- "inside_fridge": 手伸入冰箱内部"""


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
    return f"""{context_block}以下是按时间顺序排列的 {num_frames} 张冰箱摄像头截图（图1=开门初始状态，图{num_frames}=关门前最终状态）。

请逐帧观察并记录：
1. 每帧中手的位置（outside_fridge / at_entrance / inside_fridge）
2. 手中是否握着物品，握着什么
3. 手接触的每个物品的名称、数量、外观描述

输出以下 JSON：
{{
  "hand_observations": [
    {{
      "frame_number": 帧编号,
      "holding_item": "手中物品描述（没有则为空字符串）",
      "hand_position": "outside_fridge" 或 "at_entrance" 或 "inside_fridge"
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

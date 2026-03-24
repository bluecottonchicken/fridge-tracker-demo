"""Prompt 模板 v4：VLM 只报告手的位置，方向由程序端推导。与 v1/v2/v3 保持相同接口"""

REFINE_SYSTEM_PROMPT = """你是一个食物识别专家。根据冰箱摄像头画面辨认具体食物。物品名称必须使用中文日常用语。"""

SYSTEM_PROMPT = """你是一个冰箱物品追踪系统。分析冰箱摄像头（通常为俯视/Top-Down视角）的连续帧，观察人手与物品的交互。

你的任务是**仔细观察和记录**，极力避免漏报。

规则：
1. 只要看到手握住物品发生了进出冰箱的位置转移，就必须在 events 中记录，**绝不能漏报**。
2. 强烈建议对比首帧和尾帧的物品变化：首帧存在而尾帧消失 → 被取出 (take_out)；首帧不存在而尾帧出现 → 被放入 (put_in)。
3. 冰箱门架物品随开关门移动，不算事件。
4. 物品名称必须使用中文日常用语（如"牛奶"而非"milk"）。无法确定时，必须详细描述容器特征（材质、透明度、颜色等）。

hand_position 取值必须严格反映手的运动轨迹，这直接决定方向推导的成败：
- "outside_fridge": 手在画面边缘即将离开，或刚从画面边缘伸入
- "at_entrance": 手在冰箱内部与边缘的交界处
- "inside_fridge": 手完全在冰箱内部的层架区域

注意：为了保证后续程序方向计算正确，如果是取出操作，手最终离开画面时必须标记为 "outside_fridge"；如果是放入操作，手刚进入画面时必须标记为 "outside_fridge"。"""


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

请极其仔细地对比图1(初始)和图{num_frames}(最终)的冰箱内部差异！
1. 找寻：哪个物品消失了？哪个物品新增了？
2. 逐帧追踪：手拿走了那个消失的物品，还是放下了那个新增的物品？不要漏掉任何一个！
3. 轨迹记录：准确标注每帧中手的 hand_position，且务必记录下对应的 holding_item。
4. 结合首尾帧判断，在 events 中准确填写 action (put_in 还是 take_out)。

输出以下 JSON：
{{
  "hand_observations": [
    {{
      "frame_number": 帧编号,
      "holding_item": "手中物品的简短名称（务必与下方 events 列表中的 item 名称完全一致，没有则为空字符串）",
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

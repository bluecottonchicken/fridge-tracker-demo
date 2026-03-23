# 工作进度记录 — 2026-03-20

## 本次改动总览

在 Phase 3 RAG 基础上，完善了用户修正流程、性能优化、修正后重新描述机制，以及运行记录保存功能。

---

## 1. 用户修正流程扩展

**问题**：原修正流程只支持修改物品名称，无法修正数量和操作方向。

**改动**：
- `storage/inventory.py` — `correct_event()` 新增 `new_quantity` 和 `new_action` 可选参数
- `main.py` — 修正交互流程扩展为三项：物品名称、数量（如1盒猕猴桃=6个）、操作方向（put_in ↔ take_out）

**场景**：
- 数量修正：Gemini 输出"猕猴桃 x1"，实际是一盒6个
- 方向修正：Gemini 误判放入为取出，导致库存反向计算

---

## 2. 性能优化

**问题**：RAG 品类匹配阶段每个物品触发 2-3 次 API 调用，3个物品需要十几秒。

**改动**：

| 优化项 | 文件 | 效果 |
|---|---|---|
| 消除重复 embedding | `rag/category_matcher.py` — 新增 `precomputed_embedding` 参数 | 每物品省 1 次 API 往返（~2-3秒） |
| 批量 embedding | `rag/embedding.py` — 新增 `get_embeddings_batch()` | 多物品 embedding 从 N 次调用降为 1 次 |
| Gemini 全局初始化 | `config.py` — 新增 `init_genai()`，移除各模块散落的 `genai.configure()` | 代码更干净 |
| 品类初始化批量化 | `rag/category_init.py` — 改用 `get_embeddings_batch()` | 首次初始化 50-80 品类从 N 次调用降为 1 次 |

**注意**：Python 3.9 不支持 `int | None` 语法，需使用 `Optional[int]`（踩了两次坑）。

---

## 3. 修正后重新描述（Redescribe）

**问题**：用户修正物品名后（如"奶酪"→"蛋白棒"），`evt.description` 仍是 VLM 原始描述（描述的是奶酪外观），导致 embedding 品类匹配被错误描述误导。

**解决方案**：
1. 名称修正后，自动将关键帧回传 Gemini，Prompt 告知"你之前识别为X，实际是Y，重新描述外观"
2. Gemini 返回正确描述后更新 `evt.description`
3. 后续 embedding 和品类匹配使用正确描述

**改动**：
- `vlm/prompt_template.py` — 新增 `build_redescribe_prompt(correct_name, wrong_name)`
- `vlm/gemini_client.py` — 新增 `redescribe_item()`
- `main.py` — 暂存每个 session 的关键帧（`session_keyframes`）；修正后调用 redescribe

**精准帧选取优化**：
- `main.py` — 新增 `_pick_item_frames()` 函数
- 从 `raw_response` 的 `hand_observations` 中匹配 `holding_item`，定位包含该物品的 2-3 帧
- Fallback：匹配不到则取首帧+中间帧+尾帧
- 效果：redescribe 从发送 10-50 张图片降为 2-3 张，延迟降低 60-70%

---

## 4. 运行记录保存

**需求**：每次运行后将终端输出保存为 md 文件。

**改动**：
- `main.py` — 新增 `OutputLogger` 类，双通道输出（终端 + 缓冲区）
- 运行结束后自动保存至 `output/reports/run_{用户名}_{时间戳}.md`
- `input()` 提示文字也会被记录，方便回溯修正过程

---

## 改动文件汇总

| 文件 | 改动内容 |
|---|---|
| `config.py` | 新增 `init_genai()` 全局初始化 |
| `storage/inventory.py` | `correct_event()` 支持修正数量和操作方向 |
| `rag/embedding.py` | 新增 `get_embeddings_batch()` 批量接口 |
| `rag/category_init.py` | 改用批量 embedding |
| `rag/category_matcher.py` | 支持 `precomputed_embedding` 参数 |
| `vlm/prompt_template.py` | 新增 `build_redescribe_prompt()` |
| `vlm/gemini_client.py` | 新增 `redescribe_item()`；移除重复 `genai.configure()` |
| `main.py` | 修正流程扩展 + 重新描述 + 精准帧选取 + 批量 embedding + OutputLogger |
| `CLAUDE-activeContext.md` | 同步更新所有新增机制文档 |

---

## 5. 门状态检测优化

**问题**：一次开关门被误判为两次（黑衣手臂遮光导致亮度短暂回落）。

**改动**：
- `config.py` — 新增 `MIN_OPEN_FRAMES=10`（开门保护期）、`MERGE_GAP_FRAMES=5`（片段合并间隔）、`BRIGHTNESS_CLOSE_RATIO=0.7`（关门判定比例阈值）；移除固定的 `BRIGHTNESS_CLOSE_THRESHOLD`
- `pipeline/door_detector.py` — 三重改进：
  1. **最短开门保护**：开门后 10 帧内强制维持 OPEN
  2. **峰值比例关门判定**：记录开门期间亮度峰值，关门需亮度回落至少峰值涨幅的 70%（手臂遮光通常只遮 30-50%，不会误触发）
  3. **调试模式**：新增 `debug` 参数，打印每帧亮度/运动量/判定详情
- `pipeline/segment_extractor.py` — 两轮处理：先切割再合并间隔 ≤5 帧的相邻片段
- `main.py` — 新增 `--debug` 命令行参数

---

## 6. RAG 修正记录聚合加权

**问题**：历史修正记录只有简单的"原始名→修正名"映射，描述信息浪费；多次误判同一物品无法加权警告。

**改动**：
- `rag/user_history.py` — `get_alias_mappings()` 按正确物品名聚合：收集所有错误名、修正次数、最新描述；`get_frequent_items()` 优先取纠错后描述
- `rag/prompt_enhancer.py` — 修正 ≥2 次加重警告 + 附带外观描述；修正 1 次简短提示

---

## 7. VLM Prompt 强化手部判定

**问题**：VLM 识别了不存在的动作，误将冰箱门储物格中随门移动的物品报告为 take_out。

**改动**：
- `vlm/prompt_template.py` — SYSTEM_PROMPT 和 build_user_prompt 强化：
  - 核心原则："只有人手直接接触并移动的物品才算 put_in/take_out"
  - 明确禁止将门架储物格物品报告为 take_out
  - 新增"宁可漏报不要误报"兜底规则

---

## 8. RAG 匹配标注

**需求**：识别结果中标注是否匹配到 RAG 已知信息。

**改动**：
- `main.py` — 新增 `_find_rag_match()` 函数，按优先级匹配：修正记录 > 高频物品 > 冰箱状态
- 输出示例：`[RAG] 匹配到RAG修正记录「培根」(曾被误识别为无糖可乐、猪肉，已修正3次)`

---

## 9. 冰箱状态带数量

**问题**：RAG 注入的冰箱状态只有物品名列表，VLM 不知道每种物品有几个。

**改动**：
- `rag/state_tracker.py` — 返回值从 `list[str]` 改为 `dict[str, int]`
- `rag/prompt_enhancer.py` — 注入格式变为 `牛奶 x2、鸡蛋`
- `main.py` — `_print_rag_context` 和 `_find_rag_match` 适配 dict

---

## 10. ItemKnowledge 知识库去重 + 上限封顶

**问题**：同一物品每次运行都追加新记录，长期膨胀。

**改动**：
- `storage/inventory.py` — `save_to_knowledge()` 改进：
  - 描述完全相同 → 跳过（可升级 source 优先级）
  - 描述不同 → 正常写入（保留多角度描述）
  - 同一物品上限 5 条，超出淘汰最旧的 vlm_accepted

---

## 本轮新增/更新文件汇总

| 文件 | 改动内容 |
|---|---|
| `config.py` | 新增门检测参数，移除 BRIGHTNESS_CLOSE_THRESHOLD |
| `pipeline/door_detector.py` | 开门保护 + 峰值比例关门 + debug 模式 |
| `pipeline/segment_extractor.py` | 短间隔片段自动合并 |
| `rag/user_history.py` | 修正记录聚合加权 + 高频物品优先取纠错描述 |
| `rag/prompt_enhancer.py` | 分级警告注入 + 冰箱状态带数量 |
| `rag/state_tracker.py` | 返回 dict[str, int] |
| `vlm/prompt_template.py` | 强化手部判定规则 |
| `storage/inventory.py` | 知识库去重 + 上限封顶 |
| `main.py` | --debug 参数 + RAG 匹配标注 + 适配新数据格式 |
| `docs/rag-flow.md` | 更新检索模块、上下文示例、新增知识库去重章节 |
| `docs/future-improvements.md` | 新建，记录暂不实施的优化项 |

# 设计审查：改进建议（按重要性排序）

> 审查日期：2026-03-22

---

## P0 — 必须修复（影响正确性）

### 1. 品类匹配 embedding 阈值过低（0.65）

`category_matcher.py` 用余弦相似度 0.65 做匹配。食品名称语义接近（"纯牛奶" vs "酸奶" 都含奶），极易**错误合并不同品类**，导致库存数据混乱。

**建议**：
- 阈值提高到 0.75-0.80
- embedding 匹配后加一轮关键词校验（名称中的核心词是否一致）

**状态**：✅ 已修复 — 阈值调整为 0.78，提取为 `config.CATEGORY_MATCH_THRESHOLD` 配置项

### ~~2. 冰箱库存可能出现负数~~（勘误：降级为 P2）

原代码已有负数保护（`<=0` 时移除物品），不存在正确性问题。但全量加载到 Python 层聚合存在性能隐患。

**状态**：✅ 已优化 — 改为 SQL 层 `SUM(CASE ...)` + `HAVING > 0` 聚合

### ~~3. 修正记录跨用户数据污染~~（勘误：不存在此问题）

`user_history.py` 中 `get_alias_mappings()` 已按 `user_id` 过滤，不存在跨用户数据污染

---

## P1 — 强烈建议（影响效果与性价比）

### 4. refine（二次聚焦）策略性价比低

`gemini_client.py` 对低置信度物品（< 0.5）用**同样的图片**再调用一次 Gemini。图片信息量没有增加，模型大概率给出同样的不确定答案。白白消耗 token。

**建议**：
- 改为**裁剪/放大感兴趣区域**后再发送
- 或直接跳过 refine，让用户在修正阶段处理
- 省下的 token 预算可在主分析中发送更多关键帧

**状态**：✅ 已优化 — refine 改为通过 `_pick_item_frames()` 精选包含目标物品的帧（最多3帧），而非发送全部关键帧。减少 token 开销，提升信息密度。

### 5. 关键帧提取策略过于机械

`keyframe_selector.py` 采用"首帧 + 尾帧 + 均匀采样"。关键动作帧（手伸入/取出）可能集中在片段中间 1-2 秒，均匀采样会浪费大量配额在无用帧上。

**建议**：
- 加入**运动量加权采样**——运动量大的帧附近密集采样，运动量小的区域稀疏采样
- 门检测阶段已计算过每帧运动量，直接复用即可，零额外开销

**状态**：✅ 已优化 — `door_detector.detect()` 现返回运动量列表，经 `segment_extractor` 传递至 `keyframe_selector.select()`，中间帧改为运动量加权随机采样

### 6. Prompt 过于冗长

`prompt_template.py` 的 system prompt 规则描述过多。Gemini 的指令遵循能力有限，过长 prompt 会导致模型"选择性遵守"，核心规则反而被忽略。

**建议**：
- 核心规则精简到 5 条以内
- 用 few-shot example 替代冗长的文字描述
- 规则按优先级分层，最重要的放最前面

**状态**：⏸ 持续迭代中 — 已建立多版本 prompt 体系（v1~v4），通过 `config.PROMPT_TEMPLATE` 切换：
- `v1`：原版完整约束
- `v2`：精简去重版（system prompt 8条→5条，user prompt 去除重复规则）
- `v3`：v2 + 方向判断三级优先级约束（首尾帧 > 手势轨迹 > 标低置信度）
- `v4`：VLM 只报告手的位置（outside/at_entrance/inside），方向由程序端 `direction_inferrer.py` 推导

### 7. 缺少错误恢复机制

处理到第 N 个片段时若 Gemini API 报错或网络断开，整个运行结果全部丢失，无 checkpoint 机制。

**建议**：
- 每处理完一个片段立即持久化结果
- 支持从断点恢复（记录已完成的 segment index）

**状态**：✅ 已修复 — 每个片段的处理逻辑包裹在 try/except 中，单个片段失败（API 报错/网络断开/解析失败）不影响其他片段继续分析和修正流程。已成功的片段结果照常入库。运行结束时汇报失败片段列表

---

## P2 — 值得优化（影响性能与扩展性）

### 8. 品类库全量注入不可扩展

`cross_user_matcher.py` 直接取全部品类（上限 30）注入 prompt。品类增长到 100+ 时，要么截断丢失信息，要么 prompt 爆 token。

**建议**：用 embedding 相似度检索 Top-5 最相关品类注入，而非全量倒入

**状态**：✅ 已优化 — `get_known_categories()` 接受 `query_items`（用户冰箱状态+高频物品），通过 embedding 相似度排序返回 Top-5 最相关品类；无 query 时回退全量返回

### 9. 多片段 VLM 分析为串行处理

`main.py` 中逐 segment 循环调用 Gemini，完全串行。Gemini API 支持并发。

**建议**：用 `asyncio` 或 `concurrent.futures` 并行处理多片段的 VLM 分析

### 10. 数据库缺少索引

`models.py` 的 `inventory_events` 表无 `(user_id, item)` 联合索引，`item_knowledge` 表无 `(user_id, item_name)` 索引。数据增长后查询变慢。

**建议**：为高频查询字段添加联合索引

**状态**：✅ 已修复 — `inventory_events` 表为 `user_id`、`item` 添加索引；`item_knowledge` 表为 `user_id`、`item_name` 添加索引

### 11. 片段合并阈值过小

`segment_extractor.py` 相邻段间隔 ≤ 5 帧才合并（按 5fps 仅 1 秒）。人"关门看一眼再开门"（2-3 秒间隔）会导致一次操作被切成多段，增加 VLM 调用次数且丢失上下文连续性。

**建议**：合并阈值提高到 10-15 帧（2-3 秒），或做成可配置项

**状态**：✅ 已修复 — `MERGE_GAP_FRAMES` 从 5 调整为 15（按 5fps ≈ 3 秒）

---

## P3 — 锦上添花（影响泛化能力与准确表述）

### 12. 门状态检测依赖硬编码亮度阈值

`door_detector.py` 用绝对亮度阈值 `BRIGHTNESS_OPEN_THRESHOLD = 30` 判定开门。冰箱灯泡故障、白天强环境光、透明门冰箱等场景必然失效。

**建议**：改为**相对亮度变化率**，或基于前 N 帧亮度基线的自适应阈值

**状态**：✅ 已优化 — 新增 `door_detector_vlm.py`（Gemini Flash 视觉判断门状态），通过 `config.DOOR_DETECTION_METHOD` 切换 `"cv"` / `"vlm"`。VLM 模式对光线环境免疫，CV 模式保留供离线使用

### 13. "RAG" 命名不够精确

当前设计本质是"历史上下文增强"——每次固定注入冰箱状态、高频物品、修正历史、品类库，并非根据当前视频帧内容动态检索。不影响功能，但在文档和技术交流中可能造成误解。

**建议**：可保留 RAG 命名（业界已泛化使用），但在文档中明确说明是 context injection 模式而非 query-based retrieval 模式

**状态**：✅ 已修复 — `rag-flow.md` 概述和对比表中明确标注为 Context Injection 模式

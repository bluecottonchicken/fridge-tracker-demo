# 工作进度记录 — 2026-03-22

## 概述

对 `vlm-fridge-tracker` 项目进行全面设计审查，识别 13 个改进点（P0~P3），完成其中 12 项的实现或优化。

---

## 一、设计审查

对整个 VLM 冰箱追踪管线进行审查，产出 `docs/design-review_20260322_by_claude.md`，按重要性分为 P0（正确性）、P1（效果与性价比）、P2（性能与扩展性）、P3（泛化能力）四级。

---

## 二、已完成工作

### P0 — 正确性修复

| # | 问题 | 修改 | 涉及文件 |
|---|------|------|----------|
| 1 | embedding 阈值 0.65 过低 | 阈值提高为 0.78→0.8，提取为 `config.CATEGORY_MATCH_THRESHOLD` | `config.py`, `rag/category_matcher.py` |
| 2 | 库存负数（勘误：降级 P2） | 改为 SQL 层 `SUM(CASE ...)` + `HAVING > 0` 聚合 | `rag/state_tracker.py` |
| 3 | 跨用户污染（勘误：不存在） | 确认代码已按 `user_id` 过滤，无需修改 | — |

### P1 — 效果与性价比

| # | 问题 | 修改 | 涉及文件 |
|---|------|------|----------|
| 4 | refine 二次聚焦白耗 token | 改为 `_pick_item_frames()` 精选含目标物品的帧（最多3帧） | `vlm/gemini_client.py` |
| 5 | 关键帧均匀采样浪费配额 | 复用 door_detector 的运动量数据，运动量加权随机采样 | `pipeline/door_detector.py`, `pipeline/segment_extractor.py`, `pipeline/keyframe_selector.py` |
| 6 | Prompt 过长致选择性遵守 | 建立 v1~v4 多版本 prompt 体系，config 切换 | `vlm/prompt_template_v2.py` (新), `vlm/prompt_template_v3.py` (新), `vlm/prompt_template_v4.py` (新), `vlm/gemini_client.py`, `config.py` |
| 7 | 无错误恢复机制 | 每片段 try/except，失败不影响其他片段，结束时汇报失败列表 | `main.py` |

### P2 — 性能与扩展性

| # | 问题 | 修改 | 涉及文件 |
|---|------|------|----------|
| 8 | 品类库全量注入不可扩展 | `get_known_categories()` 接受 `query_items`，embedding Top-5 返回 | `rag/cross_user_matcher.py` |
| 9 | 多片段 VLM 串行 | 未实现（优先级低，待需要时再加并行） | — |
| 10 | 数据库缺索引 | `inventory_events` 和 `item_knowledge` 表添加索引 | `storage/models.py` |
| 11 | 片段合并阈值过小 | `MERGE_GAP_FRAMES` 从 5 调整为 15（≈3秒） | `config.py` |

### P3 — 泛化能力

| # | 问题 | 修改 | 涉及文件 |
|---|------|------|----------|
| 12 | 门检测依赖硬编码亮度阈值 | 新增 `door_detector_vlm.py`（Gemini Flash 视觉判断），config 切换 cv/vlm | `pipeline/door_detector_vlm.py` (新), `main.py`, `config.py` |
| 13 | "RAG" 命名不精确 | 文档中明确标注为 Context Injection 模式 | `docs/rag-flow.md` |

---

## 三、Prompt 演进历程

### 背景

VLM 对物品进出方向的判断存在误判。经过多轮讨论和迭代，最终形成了"VLM 只观测位置、程序推导方向"的架构。

### 版本演进

- **v1**（原版）：完整约束规则，Gemini 输出 `direction`（into_fridge / out_of_fridge / unclear）
- **v2**（精简）：8条规则→5条，去除 user prompt 中与 system prompt 的重复内容
- **v3**（方向优先级）：v2 基础上增加三级方向判断优先级约束（首尾帧比较 > 手势轨迹 > 标低置信度），新增 `direction_basis` 字段
- **v4**（位置+推导）：VLM 只报告 `hand_position`（outside_fridge / at_entrance / inside_fridge），方向由 `direction_inferrer.py` 程序端推导

### v4 核心设计思想

将认知任务拆分为两部分：
1. **空间观察**（VLM 擅长）：识别手在哪里、手里拿着什么
2. **时序推理**（程序擅长）：根据位置轨迹变化推导进出方向

这避免了 VLM 在时序推理上的不稳定性。

---

## 四、VLM 门检测方案

### 问题
CV 方案依赖亮度差检测开门，不同环境光线差异大，硬编码阈值泛化能力差。

### 方案
新增 `door_detector_vlm.py`，使用 Gemini Flash 对采样帧进行视觉判断（"能看到冰箱内部货架或食物 → open"），对光线环境免疫。

### 配置
`config.DOOR_DETECTION_METHOD`：`"cv"`（离线快速）或 `"vlm"`（准确但需网络）

---

## 五、Bug 修复记录

| Bug | 原因 | 修复 |
|-----|------|------|
| `TypeError: unsupported operand type(s) for \|` | Python 3.9 不支持 `X \| None` 语法 | 改为 `Optional[X]` + `from typing import Optional` |
| `404 models/gemini-3.1-flash-preview` | 模型名称错误 | 改为 `gemini-3-flash-preview` |
| `ValidationError: AnalysisResult` 解析失败 | Gemini 返回 JSON 数组 `[{...}]` 而非对象 | `from_json()` 添加数组检测，取第一个元素 |

---

## 六、待办事项

- [ ] #9：多片段 VLM 并行处理（`asyncio` / `concurrent.futures`）
- [ ] Prompt 持续迭代：根据实际运行效果对比 v1~v4 表现
- [ ] 方向推导准确率的量化评估

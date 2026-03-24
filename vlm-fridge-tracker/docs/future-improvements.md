# 未来可优化项

记录当前已知但暂不实施的优化点，供后续迭代参考。

---

## 1. 品类 user_names 反哺 Prompt

**现状**：`ItemCategory.user_names` 积累了用户对物品的叫法（如"伊利纯牛奶"），但 `get_known_categories()` 只返回品类名和描述，没有将这些别名注入 VLM prompt。

**潜在价值**：如果注入 user_names，VLM 可以在不确定时返回用户习惯的名称，提升识别一致性。

**暂不实施的原因**：当前策略是让 VLM 返回通用品类名（如"牛奶"而非"伊利纯牛奶"），除非 100% 确认品牌。这个策略更稳健，避免了错误的品牌归属。

**风险提醒**：如果用户冰箱里同时存在两种同品类但不同品牌/种类的物品（如伊利纯牛奶和蒙牛酸奶），都返回"牛奶"会导致库存合并，取出时无法区分。目前这种情况少见，可通过用户修正环节兜底。如果未来用户反馈此问题频繁出现，可考虑启用 user_names 注入。

---

## 第二档：影响可维护性和开发效率

### 2. Prompt 模板 v1-v4 四文件并存

**现状**：`prompt_template.py`（v1）、`v2`、`v3`、`v4` 四个文件同时存在。其中 `build_refine_prompt()` 和 `build_redescribe_prompt()` 在四个文件中完全相同，改一处要改四处。`gemini_client.py` 用模块级 `if/elif` 在导入时选择版本，运行时无法切换。当前生产用 v4，v1-v3 是死代码。

**建议**：确认 v4 为生产版后删除或归档 v1-v3；将公共函数（refine/redescribe）提取到共享模块；版本选择改为工厂模式或注册表。

### 3. 全部用 print() 而非 logging

**现状**：整个项目无一处使用 Python `logging` 模块。调试信息通过 `if debug: print(...)` 手工控制。无法按级别过滤、无法输出到文件、无法在生产环境静默调试信息。

**建议**：引入 `logging`，用 `logger.debug/info/warning` 替换 `print`，配合 `--debug` 参数设置日志级别。

### 4. main.py 过于庞大（600+ 行）

**现状**：`process_video()` 一个函数包揽视频加载、门检测、片段切割、RAG 上下文构建、VLM 分析、二次聚焦、结果打印、数据库存储、用户修正交互、库存展示、API 用量统计、报告保存。

**建议**：拆分为独立步骤模块或至少拆分为多个函数，如 `_analyze_segment()`、`_print_results()`、`_save_results()` 等。

### 5. OutputLogger 耦合在 main.py 中

**现状**：`main.py:34-67` 通过替换 `sys.stdout` 实现双通道输出（终端+缓冲区），多线程不安全，且与业务逻辑混在一起。

**建议**：提取到独立模块（如 `utils/logger.py`），或改用 `logging` 的 `FileHandler` + `StreamHandler` 实现。

### 6. 数据库迁移无版本管理

**现状**：`database.py:18-30` 手写 `PRAGMA table_info` + `ALTER TABLE` 的迁移方式，只能加列不能改列，无回滚机制，无版本号。

**建议**：引入 Alembic 等迁移框架，或至少维护一个版本号表记录已执行的迁移。

---

## 第三档：锦上添花

### 7. 硬编码路径未收进 config

**现状**：`main.py` 中多处硬编码 `"output/keyframes"` 和 `"output/reports"` 路径。

**建议**：在 `config.py` 中添加 `KEYFRAME_OUTPUT_DIR` 和 `REPORT_OUTPUT_DIR` 常量。

### 8. 模型定义分散

**现状**：`models/schemas.py`（Pydantic，VLM 响应解析）与 `storage/models.py`（SQLModel，ORM）职责不同但目录命名容易混淆。

**建议**：统一到 `models/api.py` + `models/db.py`，或在各模块开头注释说明职责划分。

### 9. config.py 中 DOOR_VLM_MODEL 和 GEMINI_MODEL 冗余

**现状**：`config.py:21` 和 `config.py:36` 都指向 `"gemini-3-flash-preview"`。语义不同（门检测 vs 物品分析），但当前值相同。

**建议**：如果确定门检测和物品分析始终用同一模型，可合并为一个变量；如果未来可能分开，保留两个但加注释说明。

### 10. 关键帧采样无 seed

**现状**：`keyframe_selector.py:37` 的 `np.random.choice()` 无固定 seed，同一视频每次运行结果不同，不利于调试复现。

**建议**：添加可配置的 seed 参数，调试时固定、生产时随机。

### 11. 零测试代码

**现状**：项目无任何测试文件、无 pytest 配置。

**建议**：优先为以下模块添加单元测试：
- `models/schemas.py` — JSON 解析容错
- `storage/inventory.py` — 库存计算逻辑
- `rag/category_matcher.py` — 品类匹配逻辑
- `rag/embedding.py` — embedding 序列化/反序列化

### 12. 多用户场景下的数据库与查询优化

**现状**：当前所有查询（库存计算、高频物品、修正记录）都是全量加载到 Python 内存后聚合，适合单用户/少量数据的原型阶段。随着用户增多和历史数据积累，性能会逐渐下降。

**瓶颈分析**：

| 查询 | 当前实现 | 问题 |
|------|---------|------|
| `get_current_inventory` | 加载用户全部 InventoryEvent，Python 逐条聚合 | O(n) 内存，n=该用户全部事件数 |
| `get_current_inventory_by_category` | 同上 | 同上 |
| `get_frequent_items` | 加载用户全部 ItemKnowledge，Python 聚合 | 同上 |
| `get_alias_mappings` | 加载用户全部修正记录，Python 过滤+聚合 | 同上 |
| `get_known_categories` | `.all()` 加载全部品类 | 品类数有限（<200），暂可接受 |

**阶段一：SQL 聚合替代 Python 聚合（用户 ~100，无需改表结构）**

将内存聚合下推到 SQL 层，减少数据传输量和内存占用：

```python
# get_current_inventory — 改为 SQL 聚合
# 现在：加载所有事件 → Python 循环累加
# 改为：
SELECT item,
       SUM(CASE WHEN action='put_in' THEN quantity ELSE -quantity END) AS qty
FROM inventory_events
WHERE user_id = ?
GROUP BY item
HAVING qty > 0

# get_frequent_items — 改为 SQL 聚合
# 现在：加载所有 ItemKnowledge → Python Counter
# 改为：
SELECT item_name, COUNT(*) as cnt, description
FROM item_knowledge
WHERE user_id = ?
GROUP BY item_name
ORDER BY cnt DESC
LIMIT ?
```

注意：`get_frequent_items` 中有「优先取 user_corrected 的描述」逻辑，可用 `ORDER BY CASE WHEN source='user_corrected' THEN 0 ELSE 1 END` 配合窗口函数或子查询实现，但会增加 SQL 复杂度。如果描述选择逻辑不是性能瓶颈，可以只对计数部分做 SQL 聚合，描述部分保留 Python 处理。

**阶段二：物化视图/快照表（用户 ~1000+，事件量百万级）**

当事件表增长到百万级，每次 `SUM` 聚合也会变慢。引入物化快照：

```
新增表：user_inventory_snapshot
  - user_id (索引)
  - item
  - category
  - quantity
  - updated_at

维护策略：
  - 每次 process_video 结束后，重算该用户的快照（事件量小，秒级完成）
  - 查询库存时直接读快照表，O(1) 级别
  - 用户修正事件后触发快照更新
```

这样 `get_current_inventory` 变为一次简单的 `SELECT * FROM user_inventory_snapshot WHERE user_id = ?`，不再需要聚合全部历史事件。

**阶段三：索引优化（贯穿所有阶段）**

当前已有索引：`InventoryEvent.user_id`、`InventoryEvent.item`、`ItemCategory.category`。

建议补充：
```sql
-- 库存查询核心路径：按用户聚合事件
CREATE INDEX ix_events_user_action ON inventory_events(user_id, action);

-- 知识库查询：按用户+物品名聚合
CREATE INDEX ix_knowledge_user_item ON item_knowledge(user_id, item_name);

-- 修正记录查询：按用户+来源过滤
CREATE INDEX ix_knowledge_user_source ON item_knowledge(user_id, source);
```

**阶段四：超大规模（用户 10000+，如果走到这一步）**

- SQLite → PostgreSQL（并发写入、连接池、更好的查询优化器）
- embedding 向量检索从逐条 cosine → pgvector 扩展或独立向量数据库（如 Qdrant）
- 品类库改为全局共享 + 用户级别名映射，避免每个用户重复存储相同品类

### 13. door_detector_vlm.py 的 Gemini 调用无重试且未计入 token 统计

**现状**：`door_detector_vlm.py:68-78` 直接调用 `model.generate_content`，不像 `gemini_client.py` 有 `_call_with_retry` 包装和 token 用量统计。

**建议**：复用 `gemini_client._call_with_retry`，或将重试和统计逻辑提取为公共工具。

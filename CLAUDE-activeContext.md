# 冰箱物品追踪系统

## 项目概述

基于 VLM（Gemini/Qwen）识别用户在冰箱中放入和取出物品的系统。通过分析预录视频片段，输出结构化的物品进出记录。

## 开发环境

- Mac 本地开发
- 使用预录视频文件，不使用实时摄像头
- 后续部署至树莓派

## 架构流程

```
用户验证 → 视频文件 → 逐帧读取 → 门状态检测 → 片段切割 → 关键帧提取 → VLM分析 → 存入数据库 → 用户修正/补录 → 写入知识库 → 结构化输出
```

## 目录结构

```
vlm-fridge-tracker/
├── main.py                    # 入口：串联流程，用户验证 → 读取视频 → 提帧 → 调VLM → 存库 → 打印
├── config.py                  # 配置：所有常量、阈值、API Key、数据库路径
│
├── pipeline/
│   ├── video_loader.py        # 读取视频文件，逐帧输出
│   ├── door_detector.py       # 判断每一帧的门状态（开/关）
│   ├── segment_extractor.py   # 根据门状态切割出"开门→关门"片段
│   └── keyframe_selector.py   # 从片段中动态选取关键帧（按时长计算数量）
│
├── vlm/
│   ├── prompt_template.py     # Prompt模板（手部检测为核心判断依据）
│   ├── gemini_client.py       # 调用Gemini API，发送图片+Prompt，返回原始响应
│   └── qwen_client.py         # 调用Qwen API（备选）
│
├── models/
│   └── schemas.py             # Pydantic模型：定义事件、物品、VLM响应的数据结构
│
├── storage/                   # 【Phase 2 新增】
│   ├── __init__.py
│   ├── database.py            # SQLite 初始化与连接管理
│   ├── models.py              # SQLModel ORM 模型
│   └── inventory.py           # 用户管理、事件写入、库存查询
│
├── data/                      # 【运行时自动创建】
│   └── fridge.db              # SQLite 数据库
│
├── .env                       # API Key（不入库）
├── .gitignore
└── requirements.txt
```

## 各文件职责

| 文件 | 输入 | 输出 | 唯一职责 |
|------|------|------|---------|
| `main.py` | 视频路径 + user key | 控制台 + 数据库 | 串联所有模块，用户验证，执行主流程 |
| `config.py` | `.env` + 硬编码默认值 | 配置对象 | 集中管理所有可调参数 |
| `video_loader.py` | 视频文件路径 | 帧序列(numpy数组) | 读取视频，按帧率输出帧 |
| `door_detector.py` | 单帧图片 | `OPEN` / `CLOSED` | 通过亮度+运动判断门状态 |
| `segment_extractor.py` | 帧序列 + 门状态序列 | 片段列表(帧组) | 切出每次开关门之间的帧 |
| `keyframe_selector.py` | 一个片段(帧组) | 动态数量关键帧 | 按时长均匀采样关键帧 |
| `prompt_template.py` | 无 | Prompt字符串 | 存放和格式化Prompt模板 |
| `gemini_client.py` | 关键帧 + Prompt | JSON字符串 | 调Gemini API，返回原始响应 |
| `schemas.py` | JSON字符串 | Python对象 | 解析验证VLM输出 |
| `storage/database.py` | config | engine + session | SQLite 初始化与连接管理 |
| `storage/models.py` | 无 | ORM 类 | 定义 User、AnalysisSession、InventoryEvent、ItemKnowledge |
| `storage/inventory.py` | db session | 查询结果 | 用户CRUD、事件写入、修正、补录、知识库写入、库存聚合 |

## 主流程（main.py）

```
0. 验证用户 key_code（不存在则报错，--new-user 则创建）
1. video_loader.load("video.mp4")           → 全部帧
2. door_detector.detect(帧)                  → 每帧的门状态
3. segment_extractor.extract(帧, 门状态)      → 片段列表
4. 对每个片段:
   a. keyframe_selector.select(片段)          → 动态数量关键帧
   b. prompt_template.build(帧数量)           → Prompt
   c. gemini_client.analyze(关键帧, Prompt)   → JSON字符串
   d. schemas.parse(JSON字符串)               → 结构化结果
   e. 存入数据库（analysis_session + inventory_events）
   f. 打印分析结果
5. 用户交互修正流程（逐会话）:
   a. 展示识别结果，询问是否修正物品名称
   b. 询问是否有遗漏物品，支持手动补录（放入/取出）
   c. 所有记录（修正的、补录的、原样接受的）写入 item_knowledge 知识库
6. 打印最终入库记录 + 用户当前冰箱总库存
```

## 门状态检测策略

- 亮度检测：冰箱开门后内部灯亮 → 帧平均亮度突升 > 阈值 → OPEN
- 运动检测：帧差法 → 像素变化量 > 阈值 → 确认有操作
- 关门判断：亮度回落 + 持续N秒无运动 → CLOSED → 触发分析
- 两者 AND 逻辑，减少误触发

## 关键帧提取策略

从片段中提取 3-5 帧：
- 第1帧：开门后稳定帧（背景/初始状态）
- 中间帧：运动最大帧（操作高峰）
- 最后帧：关门前稳定帧（操作后状态）

## Prompt 设计

### 结构

```
System Prompt（角色 + 规则）→ Task Prompt（任务描述）→ Output Format（JSON格式）
```

### Prompt 内容

```
【System】
你是一个冰箱物品追踪系统。你的任务是分析冰箱摄像头拍摄的
连续帧图片，判断用户在此次操作中放入了什么物品、取出了什么物品。

【规则】
1. 对比第一帧（操作前状态）和最后一帧（操作后状态），
   结合中间帧的动作判断物品的进出。
2. 物品名称使用日常用语（如"牛奶"而非"乳制品饮料"）。
3. 如果无法确定具体物品，描述其外观特征
   （如"红色塑料袋装的东西"）。
4. 每个物品给出置信度 (0-1)。
5. 如果看不清或不确定操作类型，标注 confidence < 0.5。

【输入】
以下是按时间顺序排列的 N 张冰箱摄像头截图。
图1是开门后的初始状态，图N是关门前的最终状态。

【输出格式】
严格输出以下 JSON，不要输出任何其他内容：
{
  "events": [
    {
      "action": "put_in" | "take_out",
      "item": "物品名称",
      "quantity": 数量,
      "confidence": 0.0-1.0,
      "description": "简短描述（颜色、包装等特征）"
    }
  ],
  "fridge_state_after": ["操作后冰箱内可见的所有物品列表"],
  "notes": "任何不确定或需要注意的观察"
}
```

### Prompt 设计要点

- **`fridge_state_after`**：记录操作后冰箱全貌，为后期 RAG 状态校正提供数据
- **`description`**：物品外观特征，为后期 RAG 提供物品指纹，区分同类不同个体
- **`confidence`**：低置信度事件标记为"待确认"，不直接更新库存

## 技术栈

| 用途 | 技术 | 理由 |
|------|------|------|
| 视频处理 | `opencv-python` | 成熟稳定 |
| 门检测 | OpenCV帧差 + numpy | 无需GPU |
| VLM | `google-generativeai` | 多帧支持好 |
| 数据校验 | `pydantic` | 强制结构化输出 |
| 配置管理 | `python-dotenv` | API Key管理 |
| 数据库 | SQLite | 轻量，零配置 |
| ORM | `sqlmodel` | Pydantic + SQLAlchemy 融合 |

---

## Phase 2 - 结构化存储 + 库存管理 【已完成 - 数据层】

### 目标

将 Phase 1 的控制台输出持久化，维护冰箱实时库存状态，支持多用户。

### 新增模块

```
├── storage/
│   ├── __init__.py
│   ├── database.py            # SQLite 初始化与连接管理
│   ├── models.py              # SQLModel ORM 模型（User, AnalysisSession, InventoryEvent）
│   └── inventory.py           # 用户管理、事件写入、库存查询
│
├── data/
│   └── fridge.db              # SQLite 数据库文件（运行时自动创建）
```

### 数据库设计（SQLite，4张表）

| 表 | 字段 | 用途 |
|---|---|---|
| **users** | `id`, `key_code`(唯一), `name`, `created_at` | 用户表 |
| **analysis_sessions** | `id`, `user_id`, `video_path`, `segment_index`, `keyframe_dir`, `raw_response`, `analyzed_at` | 分析记录 |
| **inventory_events** | `id`, `session_id`, `user_id`, `action`, `item`, `original_item`, `is_corrected`, `quantity`, `confidence`, `description`, `timestamp` | 库存事件（含修正追溯） |
| **item_knowledge** | `id`, `user_id`, `item_name`, `original_name`, `description`, `source`, `created_at` | 物品知识库（为 RAG 积累数据） |

- 一次开关门 = 一个 `analysis_session`，其中所有物品事件共享同一个 `session_id`
- 当前库存通过聚合 `inventory_events`（put_in 加，take_out 减）动态计算，不单独存表
- `item_knowledge` 的 `source` 字段区分 `vlm_accepted`（VLM 识别，用户确认）和 `user_corrected`（用户修正/手动补录）

### 用户管理机制

- 每个用户由唯一 `key_code` 标识（类似用户名）
- 首次使用需 `--new-user` 显式创建，防止输错 key 误创建
- 输错 key 会报错并列出已有用户
- 支持 `--list-users` 查看所有注册用户

### 命令行用法

```bash
# 首次使用 — 创建新用户并分析视频
python main.py --user xiaoming --new-user videos/IMG_7512.mov

# 后续使用 — 用已有 key 分析新视频（库存累积更新）
python main.py --user xiaoming videos/IMG_7514.MOV

# 忘了 key — 查看所有已注册用户
python main.py --list-users

# 输错 key — 报错提示
python main.py --user wrong_key videos/video.mov
# → 错误: 用户 'wrong_key' 不存在。
# → 已有用户: xiaoming
# → 如需创建新用户，请加 --new-user 参数。

# 不传 --user 则默认 key 为 "default"
python main.py --new-user videos/video.mov
```

### 用户修正与补录系统

分析完成后进入交互式修正流程：

**第一步：修正物品名称**
```
  会话 #1 识别结果:
    1. [放入] 白色盒子 x1  置信度:60%  白色利乐砖
    2. [放入] 红色袋子 x1  置信度:50%  红色塑料袋

  是否需要修正？(y/n，回车默认n): y
    1. [放入] 白色盒子 → 修正为（回车跳过）: 伊利纯牛奶
    2. [放入] 红色袋子 → 修正为（回车跳过）:
```

**第二步：补录遗漏物品**
```
  是否有遗漏的物品？(y/n，回车默认n): y
    操作类型 (1=放入, 2=取出): 1
    物品名称: 鸡蛋
    数量（回车默认1）: 6
    已补录: [放入] 鸡蛋 x6

  还有遗漏吗？(y/n，回车默认n): n
```

**数据去向：**

| 数据 | 存储位置 | 用途 |
|---|---|---|
| 修正后的物品名 | `inventory_events.item` | 库存计算 |
| VLM 原始识别 | `inventory_events.original_item` | 追溯对比 |
| 是否修正 | `inventory_events.is_corrected` | 标记 |
| 所有记录 | `item_knowledge` 表 | Phase 3 RAG 检索（标注 `vlm_accepted` 或 `user_corrected`） |

如果 VLM 全部识别正确，两步都直接回车跳过即可。

### 运行后输出

分析完成后：
1. **交互式修正** — 修正名称 + 补录遗漏
2. **本次录入记录（最终）** — 含 [已修正] 标记
3. **用户当前冰箱总库存** — 聚合该用户历史所有操作后的库存状态

### 主流程变化

```
原来: 视频 → 分析 → 打印结果
现在: 视频 + user_key → 分析 → 存库 → 用户修正/补录 → 写入知识库 → 打印最终结果
```

### 技术选型

| 用途 | 技术 |
|---|---|
| 数据库 | SQLite（轻量，存储于 `data/fridge.db`） |
| ORM | SQLModel（Pydantic + SQLAlchemy 融合） |

### 已修改文件清单

| 文件 | 操作 |
|---|---|
| `storage/__init__.py` | 新建 |
| `storage/models.py` | 新建 — User、AnalysisSession、InventoryEvent、ItemKnowledge ORM 模型 |
| `storage/database.py` | 新建 — SQLite 初始化与连接管理 |
| `storage/inventory.py` | 新建 — 用户管理、事件写入、修正、补录、知识库写入、库存查询 |
| `main.py` | 改造 — 接入数据库、用户验证、交互式修正/补录流程、argparse 增加 --new-user/--list-users |
| `config.py` | 新增 DATABASE_PATH 配置 |
| `requirements.txt` | 新增 sqlmodel 依赖 |

### 待开发

- API 服务（FastAPI）——等数据层稳定后再加

### 难点

- 库存状态漂移：VLM 漏检或误判 → 已通过用户修正/补录系统缓解
- 物品去重：同一物品不同表述（"牛奶" vs "伊利纯牛奶"）→ 留给 Phase 3 RAG

---

## Phase 3 - RAG 增强 【待开发】

### 目标

利用 Phase 2 积累的 `item_knowledge` 和 `inventory_events` 数据，通过文本 RAG 增强 VLM Prompt，提高识别准确率。

### 核心策略（已确认）

**不做图片检索 RAG**——原因：摄像头俯视角度 vs 商品正面照差异大、食物在画面中占比小、手部遮挡严重，图片匹配效果差。

**做文本 RAG**——增强 VLM 的 Prompt，而非替代 VLM 的视觉能力：

| 优先级 | 策略 | 数据来源 | 效果 |
|---|---|---|---|
| **第一** | 用户历史记录 | `item_knowledge` + `inventory_events`（Phase 2 已有） | 高 |
| **第二** | fridge_state_after 对比校正 | 每次 VLM 输出的冰箱全貌 vs 数据库库存 | 中高 |
| **第三** | 商品文本知识库（可选） | 手动/校正积累的商品特征描述 | 中 |

**第一优先级示例**：
```
VLM 看到"手里拿着一个白色盒子"，不确定是什么
RAG 检索 item_knowledge：该用户 3 次放入"伊利纯牛奶"（白色盒装）
注入 Prompt："该用户常见物品：伊利纯牛奶（白色盒装）、蒙牛酸奶（蓝色盒装）..."
VLM 结合视觉 + 历史上下文 → 置信度从 0.4 提到 0.8
```

**第二优先级示例**：
```
上次关门后 fridge_state_after = [牛奶, 鸡蛋, 苹果]
这次开门首帧看到 = [牛奶, 鸡蛋]
→ 苹果不见了？提醒 VLM 注意这个差异
```

### 数据基础

`item_knowledge` 表（Phase 2 已建好，持续积累中）：
- `item_name`：最终确认名称
- `original_name`：VLM 原始识别
- `description`：外观特征
- `source`：`vlm_accepted` / `user_corrected`

用户修正数据权重更高（source=user_corrected），VLM 确认数据作为补充。

### 新增模块

```
├── rag/
│   ├── knowledge_base.py      # 从 item_knowledge 表检索用户常见物品
│   ├── history_retriever.py   # 检索近期操作历史 + fridge_state_after 对比
│   └── enhanced_prompt.py     # 将检索结果注入 VLM Prompt
```

### 技术选型

| 用途 | 技术 |
|------|------|
| 向量数据库 | ChromaDB（本地轻量） |
| 嵌入模型 | text-embedding-3-small 或 本地模型 |
| RAG框架 | LlamaIndex |

### 难点

- 知识库冷启动：初期数据少 → 但用户修正系统已在持续积累
- 检索质量：错误检索反而误导 VLM → user_corrected 数据优先
- Prompt 长度管理：注入过多上下文会增加成本和延迟

---

## Phase 4 - 异常校正 + 用户界面 【未开发】

### 目标

提供用户可交互的界面，支持手动校正、推送通知、数据可视化。

### 新增模块

```
├── web/
│   ├── app.py                 # Web UI 入口
│   ├── dashboard.py           # 库存仪表盘页面
│   └── correction.py          # 人工校正界面
│
├── notification/
│   └── notifier.py            # 推送通知（过期提醒等）
```

### 核心功能

- **Web 仪表盘**：
  - 实时库存展示（物品列表 + 放入时间）
  - 历史操作时间线
  - 每次操作的关键帧回放

- **人工校正**：
  - 用户可修正 VLM 误判（"这不是牛奶，是豆浆"）
  - 校正数据回流 RAG 知识库，提升后续识别准确率

- **智能提醒**：
  - 物品过期预警（基于放入时间 + 品类平均保质期）
  - 库存不足提醒（"牛奶已经3天没有了"）
  - 异常检测（物品放入时间过长未取出）

### 技术选型

| 用途 | 技术 |
|------|------|
| Web框架 | Streamlit（快速原型）或 Next.js（正式版） |
| 通知推送 | 微信/Telegram Bot 或 邮件 |
| 部署 | 树莓派本地 + 可选云端同步 |

### 难点

- 过期时间估算：不同品类保质期差异大，需要品类知识库
- 用户体验：校正流程要足够简单，否则用户不会用
- 实时性：树莓派上运行 Web UI 的性能限制

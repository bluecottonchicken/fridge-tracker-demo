# 工作进度 2026-03-25

## 已完成的代码改动

### 1. Stage 2 决策逻辑重构（main.py）

**问题**：Stage 1 识别正确时，Stage 2 有时反而改错。根本原因是决策逻辑依赖 VLM 自评的 confidence 分数做仲裁，而 VLM confidence 不可信。

**改动**：去掉 confidence 对比，改为 RAG 佐证检查。

- Stage 2 给出不同物品名时，只有该名称在 RAG 知识库中有历史记录（精确匹配 `item_name`）才采信
- 无 RAG 佐证则保留 Stage 1 结果
- Stage 2 名称一致时，若描述更丰富则更新描述

**改动文件**：`main.py` 第 582-616 行

### 2. 库存显示改为具体物品名（main.py）

**问题**：终端输出的"当前冰箱库存"显示的是品类名（如"牛奶 x2"），用户希望看到具体物品名。

**改动**：将 `get_current_inventory_by_category` 替换为 `get_current_inventory`。

**改动文件**：`main.py` 第 670 行

---

## 已完成的设计方案

### Multimodal RAG 方案（docs/multimodal-rag-plan.md）

**背景**：当前文字描述 RAG 随物品增多区分度下降（"白色利乐砖"匹配多种牛奶），且无法支持跨用户知识共享。

**方案核心**：
- 用 bbox 裁剪物品区域，生成 image embedding 做检索，跳过"文字描述"的有损压缩环节
- 全用户共享一个图像 RAG 库（跨用户相似度较低但排序仍正确）
- text embedding 作为 fallback 保留

**技术选型**：`gemini-embedding-2-preview`（2026-03-10 发布）
- 原生支持图像输入，与现有 `google-genai` SDK 兼容，零新依赖
- 文本和图像映射到同一向量空间，支持跨模态检索
- 现有 `gemini-embedding-001` 仅支持文本，需统一升级

**详细方案**：见 `docs/multimodal-rag-plan.md`

---

## 讨论过但未采用的方案

### Stage 2 prompt 改为"验证模式"

将 Stage 2 从"重新识别"改为"验证 Stage 1 的结果"，默认信任 Stage 1。

**未采用原因**：告诉 VLM Stage 1 的答案会产生锚定效应，Stage 2 几乎总是 confirm，丧失纠错能力。而 LLM 无法真正"忽略已知信息做独立判断"，双输出（独立判断 + 验证意见）的自洽性检查也是伪命题——同一次生成中 VLM 自己写的前后几乎一定自洽。

**最终选择**：不改 prompt，只改决策逻辑（RAG 佐证检查）。

### CLIP/SigLIP 本地模型做图像 embedding

**未采用原因**：发现 `gemini-embedding-2-preview` 已原生支持图像，与现有 SDK 兼容，无需引入新依赖。且 Gemini embedding-2 的文本和图像在同一向量空间，天然支持跨模态 fallback。

---

## 待实施

- Multimodal RAG 的 4 个 Phase 实施（详见 `docs/multimodal-rag-plan.md` 第 5 节）
  - Phase 1：基础设施（config、embedding 函数、数据库字段）
  - Phase 2：bbox 获取与裁剪（Stage 1 prompt、schema、裁剪工具）
  - Phase 3：写入与检索（main.py 集成、检索逻辑改造）
  - Phase 4：验证与调优

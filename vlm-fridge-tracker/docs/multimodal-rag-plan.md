# Multimodal RAG 方案设计

## 1. 为什么要做图像 RAG

### 现状问题

当前 RAG 检索链路：`物品外观 → VLM 生成文字描述 → text embedding → cosine 相似度匹配`

文字描述是对视觉信息的**有损压缩**。两个只差一个 logo 颜色的牛奶品牌，VLM 描述出来可能都是"白色利乐砖包装"，text embedding 几乎重合。随着知识库中物品种类增多，这种碰撞越来越严重——检索返回大量相似但错误的结果，成为 Stage 2 的噪音源。

这不是调参（阈值、top_k）能解决的结构性问题。

### 为什么图像 embedding 能解决

图像 embedding 跳过了"文字描述"这个有损环节：`物品外观 → 裁剪物品区域 → image embedding → cosine 相似度匹配`

蓝色标签和绿色标签在文字空间里几乎一样（都是"白色纸盒包装"），但在图像 embedding 空间里有明确距离。图像保留了文字丢失的视觉细节：颜色渐变、logo 形状、包装纹理、标签文字等。

---

## 2. 核心挑战与应对

### 挑战 A：物品在画面中占比小

冰箱摄像头拍到的全帧中，物品可能只占 10-20%，其余是冰箱层架、手臂、其他物品。如果对全帧做 image embedding，背景噪音会淹没物品信号。

**应对：bbox 裁剪**

在 Stage 1 的 VLM 输出中增加 `item_bbox` 字段，让模型输出物品在帧中的大致位置坐标。裁剪后 resize 到标准尺寸，物品成为画面主体。

- Gemini 支持 bounding box 输出（归一化坐标 0-1000）
- 即使 bbox 有 ±20% 误差，裁剪后物品仍是主体内容，embedding 仍能有效区分不同物品
- 对 bbox 做合理性校验（面积不能太小 <5% 或太大 >50%），不合格时 fallback 到 text embedding

### 挑战 B：跨用户匹配——不同摄像头角度和光照

不同用户的冰箱摄像头位置、角度、光照不同。即使做了 bbox 裁剪，同一物品从不同角度拍摄的图像仍有差异。

**应对：依赖 embedding 模型的泛化能力 + 分级置信**

`gemini-embedding-2-preview` 在海量多模态数据上训练，具备对角度、光照变化的鲁棒性。bbox 裁剪后物品是主体，品牌 logo、包装颜色等特征在不同角度下仍可识别。

跨用户匹配的相似度分数会比同用户低（如 0.75 vs 0.90），但**排序通常仍正确**——正确物品仍排在最前面。利用 user_id 信息区分同用户命中和跨用户命中，分级解读置信度。

### 挑战 C：手部遮挡

手持物品时手会遮住物品一部分。

**应对：恒定噪声因子，不影响相对排序**

每次拍到手持物品时，手都在画面中。这是一个对所有物品一视同仁的噪声因子，不影响不同物品之间的 embedding 距离排序。

### 挑战 D：VLM bbox 可能完全错误

极端情况下 VLM 输出的 bbox 可能框到手上或旁边的架子上，产生垃圾 embedding。

**应对：校验 + fallback**

- bbox 面积校验（太小/太大/超出画面边界 → 拒绝）
- 图像 embedding 作为增强手段，text embedding 作为兜底始终保留
- 错误的图像 embedding 不会破坏系统——最差情况等同于没有图像 embedding，回退到现有的 text 检索

---

## 3. 好处与风险总结

### 好处

| 好处 | 说明 |
|---|---|
| 区分度大幅提升 | 视觉细节（logo、标签颜色、包装纹理）在图像空间中有明确距离，text 空间中丢失 |
| 跨用户知识共享 | 统一的图像 RAG 库，新用户首次遇到已知物品时即可受益 |
| 随数据增长不退化 | 视觉差异不会因物品数量增多而模糊（不同于文字描述的碰撞问题） |
| 与现有架构兼容 | image embedding 和 text embedding 存储、检索机制完全相同（都是向量 + cosine 相似度），无需改数据库架构 |

### 风险

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| bbox 质量不稳定 | 中 | 合理性校验 + fallback 到 text embedding |
| embedding-2 处于 preview 阶段 | 中 | API 接口稳定（与 embedding-001 同 SDK），模型 GA 后平滑切换 |
| API 成本增加 | 低 | 图像 embedding 调用量与现有 text embedding 相当，无额外数量级增长 |
| 跨用户角度差异大时精度下降 | 中 | 分级置信解读；text embedding 作为同空间补充信号 |
| 历史 embedding 迁移 | 一次性 | 升级到 embedding-2 后需重新生成所有历史向量 |
| Stage 1 prompt 复杂度增加 | 低 | bbox 设为可选字段，缺失时不影响主流程 |

---

## 4. 技术选型

### Embedding 模型：gemini-embedding-2-preview（已确认可用）

**选定方案：`gemini-embedding-2-preview`**（2026-03-10 发布，public preview）

| 特性 | 说明 |
|---|---|
| 模态支持 | 文本、图像、视频、音频、PDF 映射到**同一个向量空间** |
| 向量维度 | 3072（可选降至 768 或 1536，推荐 768 以节省存储和计算） |
| 图像限制 | 每次请求最多 6 张，支持 PNG/JPEG |
| SDK | 使用现有 `google-genai` SDK，调用方式与当前 text embedding 一致 |
| 跨模态检索 | 图像 embedding 和文字 embedding 在同一空间，天然支持跨模态 cosine 相似度比较 |

**为什么不用 CLIP/SigLIP：**
- Gemini Embedding 2 与项目现有 SDK（`google-genai`）完全兼容，零新依赖
- 统一向量空间意味着 image query 可以匹配 text records（反之亦然），实现跨模态 fallback
- 无需本地 GPU，API 调用即可

**调用示例（图像 embedding）：**
```python
from google.genai import types

result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
    config=types.EmbedContentConfig(output_dimensionality=768),
)
embedding = result.embeddings[0].values  # list[float], 768 维
```

**注意：模型升级影响**
- 现有 text embedding 使用 `gemini-embedding-001`（向量维度不同），升级到 `gemini-embedding-2-preview` 后需要**重新生成所有历史 text embedding**
- 或者分开存储：旧 text embedding 用 001，新 image embedding 用 embedding-2。但两个模型的向量空间不兼容，不能直接做 cosine 比较
- **推荐：统一升级到 embedding-2**，一次性迁移，后续维护简单

### bbox 获取

使用 Stage 1 VLM 输出。在 `hand_observations` 中增加 `item_bbox` 字段（归一化坐标）。不额外引入目标检测模型，避免增加依赖。

### 存储

在 `ItemKnowledge` 表中新增 `image_embedding` 字段（与现有 `embedding` 字段并列）。同一条记录同时持有 text embedding 和 image embedding。两种 embedding 均使用 `gemini-embedding-2-preview` 生成，处于同一向量空间。

### 检索策略

```
查询物品 → bbox 裁剪 → 生成 image embedding
                          ↓
              image embedding 检索（全用户共享库）
                          ↓
                   命中？──→ 是：返回结果（标注同用户/跨用户）
                    │
                    └─ 否：fallback 到 text embedding 检索（同一向量空间，跨模态可比）
```

---

## 5. 实施计划

### Phase 1：基础设施

**改动文件：**

1. **config.py** — 统一升级 embedding 模型 + 新增 bbox 配置
   - `EMBEDDING_MODEL` 从 `"models/gemini-embedding-001"` 改为 `"gemini-embedding-2-preview"`
   - `EMBEDDING_DIM = 768`：向量维度（embedding-2 支持 768/1536/3072，推荐 768）
   - `BBOX_MIN_AREA_RATIO = 0.05`：bbox 最小面积比（校验用）
   - `BBOX_MAX_AREA_RATIO = 0.50`：bbox 最大面积比（校验用）

2. **rag/embedding.py** — 新增图像 embedding 函数，升级现有函数
   - `get_image_embedding(image_bytes: bytes, mime_type: str) -> list[float]`：单张图片 embedding
   - `get_image_embeddings_batch(images: list[tuple[bytes, str]]) -> list[list[float]]`：批量
   - 现有 `get_embedding` / `get_embeddings_batch` 升级为使用 embedding-2 + output_dimensionality 参数

3. **storage/models.py** — ItemKnowledge 新增字段
   - `image_embedding: str = ""`：JSON 序列化的图像 embedding（768 维，与 text embedding 同空间）

### Phase 2：bbox 获取与裁剪

**改动文件：**

4. **vlm/prompt_template_v4.py** — Stage 1 prompt 增加 bbox 输出
   - `hand_observations` 中增加 `item_bbox: [y1, x1, y2, x2]`（归一化 0-1000）
   - 标注为可选字段，描述清楚坐标含义

5. **models/schemas.py** — HandObservation 增加 bbox 字段
   - `item_bbox: list[int] = []`

6. **pipeline/utils.py** 或新建 **rag/image_utils.py** — bbox 裁剪工具函数
   - `crop_item_from_frame(frame, bbox, target_size=(224, 224)) -> PIL.Image`
   - 包含 bbox 合理性校验逻辑

### Phase 3：写入与检索

**改动文件：**

7. **main.py** — Stage 1 解析后提取 bbox，裁剪生成图像 embedding
   - `_pick_item_frames` 改造：同时返回裁剪后的物品图
   - 写入知识库时同时保存 image_embedding
   - Stage 2 检索时优先用 image embedding

8. **rag/knowledge_retriever.py** — 新增图像检索函数
   - `retrieve_by_image(db, image_embedding, user_id=None, top_k, threshold) -> list[dict]`
   - `user_id=None` 时搜索全用户共享库
   - 返回结果标注 `is_same_user: bool`

9. **main.py** — Stage 2 检索逻辑改造
   - 先调 `retrieve_by_image`（图像检索，全用户）
   - 无命中时 fallback 到现有的 `retrieve_relevant_knowledge`（文字检索）
   - 合并结果，去重后注入 Stage 2 prompt

### Phase 4：验证与调优

- 对比同一物品在不同帧的 image embedding 相似度（同用户基线）
- 对比不同物品之间的 image embedding 距离（确认区分度）
- 调整 bbox 校验阈值和检索相似度阈值
- 测试跨用户场景：模拟不同摄像头角度

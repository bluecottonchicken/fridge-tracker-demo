import os
from dotenv import load_dotenv

load_dotenv()


# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def init_genai() -> None:
    """全局初始化 Gemini API，程序启动时调用一次"""
    # google.genai 推荐在使用时实例化 Client，不需要全局配置
    pass

# --- 摄像头/视频 ---
VIDEO_FPS_SAMPLE_RATE = 5  # 每秒采样帧数

# --- 门状态检测 ---
DOOR_DETECTION_METHOD = "vlm"  # 可选: "cv"(亮度+运动，离线快速) / "vlm"(Gemini Flash，准确但需网络)
DOOR_VLM_MODEL = "gemini-3-flash-preview"  # 门状态检测用的模型（仅 vlm 模式生效）
DOOR_VLM_SAMPLE_FPS = 0.5  # vlm 模式的采样帧率（每秒帧数，0.5=每2秒1帧）
BRIGHTNESS_OPEN_THRESHOLD = 30    # 亮度变化超过此值判定为开门（仅 cv 模式生效）
BRIGHTNESS_CLOSE_RATIO = 0.7     # 关门判定：亮度须回落至少开门涨幅的70%
MOTION_THRESHOLD = 5000           # 帧差像素变化量阈值
STABLE_FRAMES_TO_CLOSE = 5       # 连续N帧无运动判定为关门
MIN_OPEN_FRAMES = 10             # 开门后至少维持N帧（防抖动误判关门）
MERGE_GAP_FRAMES = 15            # 两片段间CLOSED帧数≤此值则合并为一个片段（按5fps≈3秒）

# --- 关键帧 ---
KEYFRAMES_PER_SECOND = 2   # 每秒提取关键帧数
MIN_KEYFRAMES = 4           # 最少关键帧数
MAX_KEYFRAMES = 300          # 上限封顶

# --- VLM ---
GEMINI_MODEL = "gemini-3-flash-preview"  # 可选: "gemini-3.1-pro-preview"(更准但慢) / "gemini-3-flash-preview"(快且便宜)
PROMPT_TEMPLATE = "v4"  # "v1"(原版) / "v2"(精简) / "v3"(方向优先级) / "v4"(VLM报位置+程序算方向)
VLM_MAX_RETRIES = 2
CATEGORY_MATCH_THRESHOLD = 0.75   # 品类 embedding 匹配阈值（低于此值走 Gemini 归类）
KNOWLEDGE_SAVE_THRESHOLD = 0.7    # 未经用户修正的事件，置信度低于此不写入知识库
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- 数据库 ---
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "data", "fridge.db")

import os
from dotenv import load_dotenv

load_dotenv()


# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- 摄像头/视频 ---
VIDEO_FPS_SAMPLE_RATE = 5  # 每秒采样帧数

# --- 门状态检测 ---
BRIGHTNESS_OPEN_THRESHOLD = 30    # 亮度变化超过此值判定为开门
BRIGHTNESS_CLOSE_THRESHOLD = 15   # 亮度回落低于此值判定为关门
MOTION_THRESHOLD = 5000           # 帧差像素变化量阈值
STABLE_FRAMES_TO_CLOSE = 5       # 连续N帧无运动判定为关门

# --- 关键帧 ---
KEYFRAMES_PER_SECOND = 2   # 每秒提取关键帧数
MIN_KEYFRAMES = 4           # 最少关键帧数
MAX_KEYFRAMES = 50          # 上限封顶

# --- VLM ---
GEMINI_MODEL = "gemini-2.5-flash"
VLM_MAX_RETRIES = 2

# --- 数据库 ---
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "data", "fridge.db")

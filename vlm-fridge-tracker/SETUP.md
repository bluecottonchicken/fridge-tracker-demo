# vlm-fridge-tracker 环境配置与运行指南

## 配置虚拟环境

```bash
# 进入项目目录
cd "/Users/qiuyiyin-mbp-m5/Documents/Personal Projects/VLM Demo/vlm-fridge-tracker"

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

核心依赖包括：`opencv-python`, `google-generativeai`, `pydantic`, `sqlmodel`, `python-dotenv`, `Pillow`, `numpy`。

## 配置环境变量

项目需要 Gemini API Key，在项目根目录创建 `.env` 文件：

```bash
echo "GEMINI_API_KEY=你的API密钥" > .env
```

## 运行程序

```bash
# 首次使用：创建新用户并分析视频
python main.py --user xiaoming --new-user videos/video.mov

# 后续使用：已有用户分析新视频
python main.py --user xiaoming videos/video.mov

# 使用默认用户
python main.py --new-user videos/video.mov

# 查看所有已注册用户
python main.py --list-users
```

## 注意事项

- 若 `.venv` 已存在，直接 `source .venv/bin/activate` 激活即可，无需重建
- 运行前确保已配置 `.env` 文件中的 `GEMINI_API_KEY`
- 视频文件路径作为命令行参数传入

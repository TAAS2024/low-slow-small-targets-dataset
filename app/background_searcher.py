"""
background_searcher.py — 背景图搜索模块
========================================
根据 background_spec（scene_type, time_of_day, weather, camera_position）
从 curated_backgrounds 图库中搜索匹配的背景图。

图库路径: ../1-background-pool/curated_backgrounds/
目录结构:
  airfield/  coastal/  desert/  forest/  industrial/
  mountain/  pure_sky/  rural/  urban/

对于 missing 的场景类型，自动 fallback：
  night_city → urban
  residential → rural
  snow → mountain

使用方式:
  from background_searcher import search_background
  result = search_background(background_spec)
  # → {"path": "/images/bg_xxx.jpg", "scene_type": "urban", ...}
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional

# 图库根目录
POOL_ROOT = Path(__file__).resolve().parent.parent / "1-background-pool" / "curated_backgrounds"

# 输出目录（web 可访问）
OUTPUT_DIR = Path(__file__).resolve().parent / "output_images"

# scene_type → 目录名映射 + fallback
SCENE_DIR_MAP = {
    "puresky":     "pure_sky",
    "urban":       "urban",
    "rural":       "rural",
    "mountain":    "mountain",
    "coastal":     "coastal",
    "desert":      "desert",
    "forest":      "forest",
    "industrial":  "industrial",
    "airfield":    "airfield",
    # fallbacks
    "night_city":  "urban",       # 无 night_city 目录 → 用 urban
    "residential": "rural",       # 无 residential 目录 → 用 rural
    "snow":        "mountain",    # 无 snow 目录 → 用 mountain
}

# 有效目录列表
VALID_DIRS = set(SCENE_DIR_MAP.values())


def search_background(background_spec: dict, session_id: int = 0) -> dict:
    """
    根据 background_spec 搜索背景图。

    Args:
        background_spec: {"scene_type": "urban", "time_of_day": "afternoon",
                          "weather": "overcast", "camera_position": "bottom"}
        session_id: 会话 ID，用于输出文件命名

    Returns:
        {
            "ok": True,
            "image_url": "/images/session_0_bg.jpg",
            "source_path": "/abs/path/to/original.jpg",
            "scene_type": "urban",
            "dir_used": "urban",
        }
    """
    scene_type = background_spec.get("scene_type", "puresky")
    dir_name = SCENE_DIR_MAP.get(scene_type, "pure_sky")
    search_dir = POOL_ROOT / dir_name

    if not search_dir.exists() or not search_dir.is_dir():
        return {"ok": False, "error": f"目录不存在: {search_dir}"}

    # 列出所有 jpg 文件
    images = list(search_dir.glob("*.jpg")) + list(search_dir.glob("*.jpeg")) + list(search_dir.glob("*.png"))
    if not images:
        return {"ok": False, "error": f"目录无图片: {search_dir}"}

    # 随机选一张
    chosen = random.choice(images)

    # 复制到 output_images
    OUTPUT_DIR.mkdir(exist_ok=True)
    ext = chosen.suffix
    dest_name = f"session_{session_id}_bg{ext}"
    dest_path = OUTPUT_DIR / dest_name
    shutil.copy2(str(chosen), str(dest_path))

    return {
        "ok": True,
        "image_url": f"/images/{dest_name}",
        "source_path": str(chosen),
        "scene_type": scene_type,
        "dir_used": dir_name,
    }


def get_placeholder_image(session_id: int, label: str, color: str = "#334155") -> dict:
    """
    生成占位图（纯色 PNG），用于还没有真实生成结果的 slot。
    使用 PIL 生成简单的纯色 + 文字标签图片。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # 无 PIL 时返回 None
        return {"ok": False, "error": "PIL not available"}

    OUTPUT_DIR.mkdir(exist_ok=True)
    dest_name = f"session_{session_id}_placeholder_{label}.png"
    dest_path = OUTPUT_DIR / dest_name

    img = Image.new("RGB", (400, 400), color)
    draw = ImageDraw.Draw(img)

    # 尝试用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # 居中文字
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (400 - tw) // 2
    y = (400 - th) // 2
    draw.text((x, y), label, fill="#e2e8f0", font=font)

    img.save(str(dest_path))

    return {
        "ok": True,
        "image_url": f"/images/{dest_name}",
        "placeholder": True,
    }


if __name__ == "__main__":
    # 测试
    spec = {"scene_type": "urban", "time_of_day": "afternoon", "weather": "overcast", "camera_position": "bottom"}
    result = search_background(spec, session_id=999)
    print(result)

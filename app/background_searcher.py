"""
background_searcher.py — 背景图搜索模块
========================================
根据 background_spec（scene_type, time_of_day, weather, camera_position）
从 curated_backgrounds 图库中搜索匹配的背景图。

图库路径: ../1-background-pool/curated_backgrounds/
目录结构:
  airfield/  coastal/  desert/  forest/  industrial/
  mountain/  pure_sky/  rural/  urban/

每个场景目录内除图片外还有 `_metadata.jsonl`，逐图记录：
  {"filename", "url", "source", "photographer", "category",
   "time_of_day", "weather"}
可用标签（实测全库统计）：
  time_of_day ∈ {dawn, morning, afternoon, dusk, night}
  weather     ∈ {clear, overcast, rainy, foggy, dusty, backlight}

v7.2 修复
---------
原实现只按 scene_type 选目录，然后在目录里 `random.choice()`，
**完全忽略 time_of_day / weather**——与模块设计意图（"搜索匹配的背景图"）
不符，导致"雪天黄昏的住宅区"可能拿到任意一张山景。
现在改为**标签感知检索**，分四级降级回退，永不失败：

  ① time_of_day 且 weather 都命中
  ② 仅 time_of_day 命中
  ③ 仅 weather 命中
  ④ 任意（等价于旧行为）

命中等级会写进返回值 `match_level`，便于审计与复现。
紧急回退开关：环境变量 `BG_MATCH=random` → 恢复纯随机（复现旧结果用）。

v7.5 修复（2026-10-11）
---------------------
新增**滑动排除队列**：模块级环形队列 `_RECENT_BG` 记录最近 `BG_EXCLUDE_WINDOW`
（默认 16）张被选中的背景文件名，检索时先过滤掉它们；仅当过滤后候选数低于
`BG_EXCLUDE_KEEP_MIN`（默认 1）时才回退到全量，保证小目录不会因排除而取不到图。
—— 该机制此前只写在论文 §3.3.1 与本文档的说明里，代码未落地，本轮补齐。

**保证强度（勿过度表述）**：队列长度为 16，因此语义是「同一背景不会在连续 16 次
检索内重复出现」。只有当某层的候选数 ≤ 16 时，该层才会被「无重复地遍历至耗尽」；
候选数更大时，被挤出窗口的历史背景可以被重新抽中。实测见
`experiments/bcd/_reports/exclude_queue_probe_2026-10-11.json`：
mountain/dusk+clear（该层 14 张）连抽 30 次 → 14 张全不同、第 15 次才首次重复；
urban/noon+overcast（该层 70 张）→ 首次重复出现在第 24 次。

对于 missing 的场景类型，自动 fallback：
  night_city → urban
  residential → rural
  snow → mountain

使用方式:
  from background_searcher import search_background
  result = search_background(background_spec)
  # → {"ok": True, "image_url": "/images/session_0_bg.jpg",
  #    "source_path": "...", "scene_type": "urban", "dir_used": "urban",
  #    "match_level": "tod+weather", ...}
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # → 0-workspace
from lss_paths import POOL_ROOT, OUTPUT_DIR  # noqa: E402  (single source of truth)

# ── 滑动排除队列（v7.5, 2026-10-11）─────────────────────────────
# 原实现每次独立 random.choice，连续合成可能反复拿到同一张背景；
# 论文 §3.3.1 所述 "sliding exclusion queue" 此前只写在文档里、代码未落地。
# 现在落为模块级环形队列：最近 EXCLUDE_WINDOW 张用过的背景名被排除，
# 只有当过滤后候选数低于 EXCLUDE_KEEP_MIN 时才回退到全量。
from collections import deque as _deque

EXCLUDE_WINDOW = int(os.environ.get("BG_EXCLUDE_WINDOW", "16"))
EXCLUDE_KEEP_MIN = int(os.environ.get("BG_EXCLUDE_KEEP_MIN", "1"))
_RECENT_BG: "_deque[str]" = _deque(maxlen=max(1, EXCLUDE_WINDOW))

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

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")

# ── 标签归一化 ───────────────────────────────────────────────
# 左侧是 LLM / 调用方可能给出的写法，右侧是图库元数据里实际存在的标签
# （按优先级排序，元组中第一个是首选匹配，其余为可接受的次选）。
TOD_ALIASES: dict[str, tuple] = {
    "dawn": ("dawn",), "sunrise": ("dawn",), "early_morning": ("dawn",),
    "morning": ("morning",),
    "noon": ("afternoon",), "midday": ("afternoon",), "afternoon": ("afternoon",),
    "day": ("afternoon",), "daytime": ("afternoon",),
    "dusk": ("dusk",), "sunset": ("dusk",), "evening": ("dusk",), "twilight": ("dusk",),
    "night": ("night",), "midnight": ("night",), "nighttime": ("night",),
}

WX_ALIASES: dict[str, tuple] = {
    "clear": ("clear",), "sunny": ("clear",), "fair": ("clear",),
    "sunshine": ("clear",),
    "cloudy": ("overcast",), "overcast": ("overcast",), "clouds": ("overcast",),
    "gloomy": ("overcast",),
    "rain": ("rainy",), "rainy": ("rainy",), "raining": ("rainy",),
    "storm": ("rainy",), "stormy": ("rainy",), "wet": ("rainy",),
    "fog": ("foggy",), "foggy": ("foggy",), "mist": ("foggy",), "misty": ("foggy",),
    "haze": ("foggy", "dusty"), "hazy": ("foggy", "dusty"),
    "dust": ("dusty",), "dusty": ("dusty",), "sandstorm": ("dusty",),
    "backlight": ("backlight",), "backlit": ("backlight",), "glare": ("backlight",),
}

# 检索模式：labels = 标签感知（默认）｜random = 纯随机（旧行为）
MATCH_MODE = os.environ.get("BG_MATCH", "labels").strip().lower()

_meta_cache: dict[str, dict] = {}


def _load_meta(scene_dir: Path) -> dict:
    """读取并缓存某个场景目录的 `_metadata.jsonl`：filename → {tod, weather}。"""
    key = str(scene_dir)
    if key in _meta_cache:
        return _meta_cache[key]

    meta: dict[str, dict] = {}
    mf = scene_dir / "_metadata.jsonl"
    if mf.exists():
        try:
            with mf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fn = o.get("filename")
                    if fn:
                        meta[fn] = {
                            "time_of_day": o.get("time_of_day"),
                            "weather": o.get("weather"),
                        }
        except OSError:
            pass

    _meta_cache[key] = meta
    return meta


def _normalize(value, table: dict) -> Optional[str]:
    """把调用方给的值归一到图库标签；无法识别返回 None。"""
    if not value:
        return None
    v = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    hit = table.get(v)
    return hit[0] if hit else None


def _normalize_multi(value, table: dict) -> tuple:
    """同 _normalize，但返回全部可接受标签（用于 haze 这类一对多）。"""
    if not value:
        return ()
    v = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return table.get(v, ())


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
            "match_level": "tod+weather" | "tod" | "weather" | "any",
            "requested": {"time_of_day": "...", "weather": "..."},
            "matched_labels": {"filename": ..., "time_of_day": ..., "weather": ...},
            "pool_size": 3750,
            "candidates": 412,
        }
    """
    scene_type = background_spec.get("scene_type", "puresky")
    dir_name = SCENE_DIR_MAP.get(scene_type, "pure_sky")
    search_dir = POOL_ROOT / dir_name

    if not search_dir.exists() or not search_dir.is_dir():
        return {"ok": False, "error": f"目录不存在: {search_dir}"}

    # 列出所有图片（显式限定扩展名，避免把 _metadata.jsonl 当图）
    images = []
    for pat in IMAGE_EXTS:
        images += list(search_dir.glob(pat))
    if not images:
        return {"ok": False, "error": f"目录无图片: {search_dir}"}

    req_tod = background_spec.get("time_of_day")
    req_wx = background_spec.get("weather")

    meta = _load_meta(search_dir)
    chosen = None
    match_level = "any"
    n_cand = len(images)

    if MATCH_MODE != "random" and meta:
        tod_ok = set(_normalize_multi(req_tod, TOD_ALIASES))
        wx_ok = set(_normalize_multi(req_wx, WX_ALIASES))

        def _tags(p: Path):
            m = meta.get(p.name) or {}
            return m.get("time_of_day"), m.get("weather")

        tiers = []
        if tod_ok or wx_ok:
            tiers = [
                ("tod+weather",
                 [p for p in images if _tags(p)[0] in tod_ok and _tags(p)[1] in wx_ok]),
                ("tod",
                 [p for p in images if _tags(p)[0] in tod_ok]),
                ("weather",
                 [p for p in images if _tags(p)[1] in wx_ok]),
            ]

        for level, pool in tiers:
            # 滑动排除队列：优先排除最近几轮用过的背景，避免连续合成拿到同一张
            fresh = [p for p in pool if p.name not in _RECENT_BG]
            if len(fresh) < min(EXCLUDE_KEEP_MIN, len(pool)):
                fresh = pool
            if fresh:
                chosen = random.choice(fresh)
                match_level = level
                n_cand = len(fresh)
                break

    if chosen is None:
        fresh = [p for p in images if p.name not in _RECENT_BG]
        if len(fresh) < min(EXCLUDE_KEEP_MIN, len(images)):
            fresh = images
        chosen = random.choice(fresh)
        match_level = "any"
        n_cand = len(fresh)

    # 记入滑动排除队列（放不下时才淘汰最旧的）
    _RECENT_BG.append(chosen.name)

    # 复制到 output_images
    OUTPUT_DIR.mkdir(exist_ok=True)
    ext = chosen.suffix
    dest_name = f"session_{session_id}_bg{ext}"
    dest_path = OUTPUT_DIR / dest_name
    shutil.copy2(str(chosen), str(dest_path))

    tags = meta.get(chosen.name, {})
    print(f"[background_searcher] scene={dir_name} req=({req_tod},{req_wx}) "
          f"→ {chosen.name} match={match_level} cand={n_cand}/{len(images)} "
          f"labels=({tags.get('time_of_day')},{tags.get('weather')})", flush=True)

    return {
        "ok": True,
        "image_url": f"/images/{dest_name}",
        "source_path": str(chosen),
        "scene_type": scene_type,
        "dir_used": dir_name,
        "match_level": match_level,
        "requested": {"time_of_day": req_tod, "weather": req_wx},
        "matched_labels": {"filename": chosen.name, **tags},
        "pool_size": len(images),
        "candidates": n_cand,
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
    # 自测：九类场景 × 五种时段 × 六种天气，检查命中等级分布
    import collections

    scenes = ["pure_sky", "coastal", "urban", "rural", "forest",
              "mountain", "desert", "industrial", "airfield"]
    tods = ["dawn", "morning", "afternoon", "dusk", "night"]
    wxs = ["clear", "cloudy", "rain", "fog", "haze"]

    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "selftest":
        stats = collections.Counter()
        for s in scenes:
            for t in tods:
                for w in wxs:
                    r = search_background(
                        {"scene_type": s, "time_of_day": t, "weather": w},
                        session_id=999)
                    stats[(s, r.get("match_level"))] += 1
        for k in sorted(stats):
            print(k, stats[k])
        print("TOTAL:", dict(collections.Counter(
            {lvl: sum(v for (s, l), v in stats.items() if l == lvl)
             for lvl in ("tod+weather", "tod", "weather", "any")})))
    else:
        spec = {"scene_type": "urban", "time_of_day": "afternoon",
                "weather": "overcast", "camera_position": "bottom"}
        print(search_background(spec, session_id=999))

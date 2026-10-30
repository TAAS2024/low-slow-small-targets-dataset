"""
condition_generator.py — 条件图生成器 (v6.0)
============================================
从背景图 + full_scene_spec 生成全部 5 类中间产物：
  bg_depth, bg_seg, drone_depth, drone_seg, final_rgb

管线: BG → MidasDepth → HeuristicSeg → Trajectory Gaussian Blob → 输出

使用方式:
  from condition_generator import generate_all_conditions
  result = generate_all_conditions(bg_path, full_scene_spec, session_id)
  # → {"ok": True, "images": {"bg_depth": [...], "bg_seg": [...], ...}}
"""

import os
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ── 路径配置 ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "output_images"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (512, 512)
H, W = TARGET_SIZE

# 语义分割颜色表 (6 超类 + 无人机)
SEG_COLORS = {
    0: (0, 0, 0),         # background / ignore
    1: (135, 206, 235),   # sky
    2: (128, 128, 128),   # building
    3: (34, 139, 34),     # vegetation
    4: (210, 180, 140),   # ground
    5: (30, 144, 255),    # water
    6: (112, 128, 144),   # mountain
    7: (255, 50, 50),     # drone (injected)
}

# ── 单例 MidasDetector ───────────────────────────────────
_detector = None


def _get_detector():
    """懒加载 MidasDetector 单例（首次加载 ~10-15s）。"""
    global _detector
    if _detector is None:
        from controlnet_aux import MidasDetector
        # v7.3: 深度估计器不应硬依赖网络。原实现无条件联网（HEAD 查 revision），
        #   在 proxy 未启动 / hf-mirror 抖动时会先空转 5 次重试（~23s）再抛
        #   ConnectionError，直接崩掉整条生成管线（实测 2026-10-05：/api/parse
        #   只回一张背景图，十槽位空 9 个，前端表现为「只有背景」）。
        #   改为 local_files_only 优先：命中缓存时零网络、启动更快、模型版本可复现
        #   （研究产物应固定权重而非跟随上游更新）；未命中再联网（仅首次部署）。
        try:
            _detector = MidasDetector.from_pretrained(
                "lllyasviel/Annotators", local_files_only=True)
        except Exception as _e:
            print(f"  \u26a0 MidasDetector 本地缓存未命中"
                  f"（{type(_e).__name__}），联网下载 ...", flush=True)
            _detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return _detector


# ============================================================
# 1. 深度提取
# ============================================================

def extract_depth(image: Image.Image) -> np.ndarray:
    """从 RGB 图像提取单目深度图 (0=近, 1=远)。"""
    detector = _get_detector()
    depth_img = detector(image)
    depth = np.array(depth_img).astype(np.float32)[:, :, 0] / 255.0
    return depth


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """深度图 → RGB 可视化 (近=亮, 远=暗)。"""
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    d_vis = ((1.0 - d_norm) * 255).astype(np.uint8)
    return np.stack([d_vis] * 3, axis=-1)


# ============================================================
# 2. 启发式语义分割
# ============================================================

def heuristic_segment(bg_arr: np.ndarray) -> np.ndarray:
    """基于颜色阈值 + 空间位置的启发式 6 类分割。"""
    r = bg_arr[:, :, 0].astype(int)
    g = bg_arr[:, :, 1].astype(int)
    b = bg_arr[:, :, 2].astype(int)
    y_idx = np.arange(H)[:, None].repeat(W, axis=1)

    seg = np.zeros((H, W), dtype=np.uint8)

    seg[(r > 150) & (g > 155) & (b > 160) & (y_idx < H * 0.55)] = 1   # sky
    seg[(g > r + 15) & (g > b + 10) & (seg == 0)] = 3                  # vegetation
    seg[(b > r + 20) & (b > g + 10) & (seg == 0)] = 5                  # water
    seg[(y_idx > H * 0.6) & (seg == 0)] = 4                            # ground
    seg[seg == 0] = 2                                                   # building (default)

    # 双重平滑消除噪点
    seg = cv2.medianBlur(seg.astype(np.uint8), 11)
    seg = cv2.medianBlur(seg.astype(np.uint8), 5)
    return seg


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """分割标签 → RGB 可视化。"""
    s_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, col in SEG_COLORS.items():
        s_rgb[seg == cid] = col
    return s_rgb


# ============================================================
# 3. 轨迹帧参数计算
# ============================================================

def trajectory_to_frames(full_scene_spec: dict) -> list[dict]:
    """
    将 full_scene_spec.trajectory 转换为逐帧渲染参数。

    Input: [{"t":0, "action":"hover", "distance":75, "norm_u":0.5, "norm_v":0.3}, ...]
    Output: [{"nu":0.5, "nv":0.3, "size":28, "dval":0.38, "action":"hover"}, ...]
    """
    traj = full_scene_spec.get("trajectory", [])
    if not traj:
        # 默认单帧：中间位置，75m
        traj = [{"t": 0, "action": "hover", "distance": 75, "norm_u": 0.5, "norm_v": 0.3}]

    frames = []
    for pt in traj:
        distance = float(pt.get("distance", 75))
        nu = float(pt.get("norm_u", 0.5))
        nv = float(pt.get("norm_v", 0.3))
        action = pt.get("action", "hover")

        # 距离 → 像素尺寸映射
        size = int(np.clip(3000 / max(distance, 25), 10, 60))

        # 距离 → 深度值映射 (0=near, 1=far)
        dval = float(np.clip(distance / 200, 0.15, 0.7))

        frames.append({
            "nu": nu, "nv": nv,
            "size": size, "dval": dval,
            "action": action, "distance": distance,
        })

    return frames


# ============================================================
# 4. Gaussian Blob 注入
# ============================================================

def inject_gaussian_blob(depth: np.ndarray, seg: np.ndarray,
                         nu: float, nv: float,
                         size: int, dval: float) -> tuple[np.ndarray, np.ndarray]:
    """
    在 depth 和 seg 图上注入一个 Gaussian 无人机 blob。

    Args:
        depth: 原始深度图 [H, W]
        seg: 原始分割图 [H, W]
        nu, nv: 归一化中心坐标 (0-1)
        size: 无人机像素大小
        dval: 无人机深度值

    Returns:
        (modified_depth, modified_seg)
    """
    cx = int(np.clip(nu * W, size // 2, W - size // 2))
    cy = int(np.clip(nv * H, size // 2, H - size // 2))
    sigma = size / 4.5

    y, x = np.ogrid[:H, :W]
    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    mask = gauss > 0.25

    # 深度混合 (alpha blend)
    d_mod = depth.copy()
    d_mod[mask] = dval * gauss[mask] + depth[mask] * (1 - gauss[mask])

    # 分割注入
    s_mod = seg.copy()
    s_mod[mask] = 7  # drone class

    return d_mod, s_mod


# ============================================================
# 5. 主入口：生成全部条件图
# ============================================================

def generate_all_conditions(bg_path: str,
                            full_scene_spec: dict,
                            session_id: int = 0) -> dict:
    """
    从背景图生成全部 5 类条件图。

    Args:
        bg_path: 背景图文件路径 (绝对路径)
        full_scene_spec: LLM 输出的全场景 JSON
        session_id: 会话 ID（用于文件名前缀）

    Returns:
        {
            "ok": True,
            "images": {
                "bg_depth": ["/images/session_0_bg_depth.png"],
                "bg_seg":   ["/images/session_0_bg_seg.png"],
                "drone_depth": ["/images/session_0_drone_depth_0.png", ...],
                "drone_seg":   ["/images/session_0_drone_seg_0.png", ...],
                "final_rgb":   ["/images/session_0_final_rgb_0.png", ...],
            }
        }
    """
    sid = session_id

    # ── Step 1: 加载背景图 ──
    original = Image.open(bg_path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    bg_arr = np.array(original)

    # ── Step 2: 深度提取 ──
    depth = extract_depth(original)

    # 保存背景深度图
    d_rgb = depth_to_rgb(depth)
    bg_depth_name = f"session_{sid}_bg_depth.png"
    bg_depth_path = OUTPUT_DIR / bg_depth_name
    Image.fromarray(d_rgb).save(str(bg_depth_path))

    # ── Step 3: 语义分割 ──
    seg = heuristic_segment(bg_arr)

    # 保存背景分割图
    s_rgb = seg_to_rgb(seg)
    bg_seg_name = f"session_{sid}_bg_seg.png"
    bg_seg_path = OUTPUT_DIR / bg_seg_name
    Image.fromarray(s_rgb).save(str(bg_seg_path))

    # ── Step 4: 轨迹帧生成 ──
    frames = trajectory_to_frames(full_scene_spec)

    drone_depth_urls = []
    drone_seg_urls = []
    final_rgb_urls = []

    for fi, frame in enumerate(frames):
        # 注入 Gaussian blob
        d_mod, s_mod = inject_gaussian_blob(
            depth, seg,
            frame["nu"], frame["nv"],
            frame["size"], frame["dval"],
        )

        # 无人机深度图
        dd_rgb = depth_to_rgb(d_mod)
        dd_name = f"session_{sid}_drone_depth_{fi}.png"
        dd_path = OUTPUT_DIR / dd_name
        Image.fromarray(dd_rgb).save(str(dd_path))
        drone_depth_urls.append(f"/images/{dd_name}")

        # 无人机分割图
        ds_rgb = seg_to_rgb(s_mod)
        ds_name = f"session_{sid}_drone_seg_{fi}.png"
        ds_path = OUTPUT_DIR / ds_name
        Image.fromarray(ds_rgb).save(str(ds_path))
        drone_seg_urls.append(f"/images/{ds_name}")

        # 最终叠加图 (seg 半透明覆盖原图)
        overlay = (bg_arr * 0.55 + ds_rgb * 0.45).astype(np.uint8)
        ov_name = f"session_{sid}_final_rgb_{fi}.png"
        ov_path = OUTPUT_DIR / ov_name
        Image.fromarray(overlay).save(str(ov_path))
        final_rgb_urls.append(f"/images/{ov_name}")

    return {
        "ok": True,
        "images": {
            "bg_depth": [f"/images/{bg_depth_name}"],
            "bg_seg": [f"/images/{bg_seg_name}"],
            "drone_depth": drone_depth_urls,
            "drone_seg": drone_seg_urls,
            "final_rgb": final_rgb_urls,
        },
        "frame_count": len(frames),
        "trajectory_summary": [
            {"t": fi, "action": f["action"], "distance": f["distance"],
             "size_px": f["size"], "depth_val": round(f["dval"], 2)}
            for fi, f in enumerate(frames)
        ],
    }


# ============================================================
# 6. 快速预览（不走完整管线，仅覆盖层图）
# ============================================================

def generate_overlay_only(bg_path: str,
                          full_scene_spec: dict,
                          session_id: int = 0) -> dict:
    """
    仅生成 seg overlay 图（跳过 depth/seg 保存，适合快速预览）。
    其他 slot 返回占位。
    """
    sid = session_id
    original = Image.open(bg_path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    bg_arr = np.array(original)
    depth = extract_depth(original)
    seg = heuristic_segment(bg_arr)
    frames = trajectory_to_frames(full_scene_spec)

    final_rgb_urls = []
    for fi, frame in enumerate(frames):
        _, s_mod = inject_gaussian_blob(
            depth, seg, frame["nu"], frame["nv"],
            frame["size"], frame["dval"],
        )
        ds_rgb = seg_to_rgb(s_mod)
        overlay = (bg_arr * 0.55 + ds_rgb * 0.45).astype(np.uint8)
        ov_name = f"session_{sid}_final_rgb_{fi}.png"
        ov_path = OUTPUT_DIR / ov_name
        Image.fromarray(overlay).save(str(ov_path))
        final_rgb_urls.append(f"/images/{ov_name}")

    return {
        "ok": True,
        "images": {"final_rgb": final_rgb_urls},
        "frame_count": len(frames),
    }


# ─────────────────────────────────────────────────────────────
# v7.2 并发保护
#   Flask app.run() 默认 threaded=True：两个请求会同时进入同一个 GPU
#   管线（首次调用还会并发触发模型加载），轻则显存翻倍 OOM，重则
#   CUDA 上下文报错。这里把重资产入口串行化；用 RLock 保证内部
#   互相调用（focus_reposition → generate_with_lora）不会自锁死。
# ─────────────────────────────────────────────────────────────
import functools as _functools
import threading as _threading

_PIPE_LOCK = _threading.RLock()


def _serialized(_fn):
    @_functools.wraps(_fn)
    def _wrapper(*a, **kw):
        with _PIPE_LOCK:
            return _fn(*a, **kw)
    return _wrapper


for _name in ("generate_all_conditions", "generate_overlay_only"):
    if callable(globals().get(_name)):
        globals()[_name] = _serialized(globals()[_name])
del _name

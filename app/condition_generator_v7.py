"""
condition_generator_v7.py — ControlNet(depth+seg) 条件驱动全管线
=================================================================
v7.1 升级：标记注入 → ControlNet 驱动的无人机局部重绘（替代简单 img2img）

管线:
  BG → MidasDepth(→resize到BG分辨率) → HeuristicSeg
     → LoRA 生成无人机 (512² 白底) → rembg 抠图
     → drone silhouette 注入 bg_depth/bg_seg → drone_depth, drone_seg
     → ControlNet(depth=drone_depth, seg=drone_seg) 重绘 → final_rgb
     → final_rgb → IR (白热红外) → bbox (Seg→框可视化)
     → drone_raw, drone_clean 作为备选保留

产物:
  bg, bg_depth, bg_seg, drone_depth, drone_seg, final_rgb, ir, bbox, drone_clean, drone_raw

使用方式:
  python condition_generator_v7.py          # 命令行 demo
  from condition_generator_v7 import generate_with_lora  # Web 集成
"""

import os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from PIL import Image
import cv2
import torch

# ── 路径配置 ──────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output_images"
OUTPUT_DIR.mkdir(exist_ok=True)

PROJECT_ROOT = HERE.parent

# ── 加载 IR 转换器（归档结构：generator/）───────────────────
sys.path.insert(0, str(PROJECT_ROOT / "generator"))
from rgb2ir_converter import rgb_to_whitehot

# 语义分割颜色表（与 ControlNet seg 兼容）
SEG_COLORS = {
    0: (0, 0, 0),        # 未分类/忽略
    1: (135, 206, 235),  # 天空
    2: (128, 128, 128),  # 建筑/硬表面
    3: (34, 139, 34),    # 植被
    4: (210, 180, 140),  # 地面
    5: (30, 144, 255),   # 水域
    6: (112, 128, 144),  # 道路
    7: (255, 50, 50),    # 无人机 🚁
}


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

_detector = None

def _get_midas():
    global _detector
    if _detector is None:
        from controlnet_aux import MidasDetector
        _detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return _detector


def extract_depth(image: Image.Image) -> np.ndarray:
    """Midas 深度提取，输出 (H,W) float32 [0,1]"""
    d = _get_midas()(image)
    return np.array(d).astype(np.float32)[:, :, 0] / 255.0


def extract_depth_matched(image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    提取深度并强制 resize 到 target_size（匹配背景分辨率）。
    Midas 默认输出 768 长边，但可能偏离，用 PIL 强制对齐。
    
    Args:
        image: PIL Image (RGB)
        target_size: (W, H) PIL 格式的目标尺寸
    Returns:
        (H, W) float32 depth [0, 1]
    """
    depth_raw = extract_depth(image)
    tw, th = target_size  # (W, H)
    h, w = depth_raw.shape
    if h != th or w != tw:
        # 用 PIL 保证精确尺寸，避免 cv2.resize 浮点偏差
        depth_pil = Image.fromarray((depth_raw * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((tw, th), Image.BILINEAR)
        depth_raw = np.array(depth_pil).astype(np.float32) / 255.0
    return depth_raw


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """归一化深度图 → (H,W,3) uint8 RGB（远=白, 近=黑）"""
    dn = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    dv = ((1.0 - dn) * 255).astype(np.uint8)
    return np.stack([dv] * 3, axis=-1)


def heuristic_segment(bg_arr: np.ndarray) -> np.ndarray:
    """启发式 6 类语义分割（天空、建筑、植被、地面、水域、道路）"""
    H_img, W_img = bg_arr.shape[:2]
    r = bg_arr[:, :, 0].astype(int)
    g = bg_arr[:, :, 1].astype(int)
    b = bg_arr[:, :, 2].astype(int)
    yi = np.arange(H_img)[:, None].repeat(W_img, axis=1)

    seg = np.zeros((H_img, W_img), dtype=np.uint8)

    # 1: 天空 (亮、偏蓝/白、上半部分)
    seg[(r > 150) & (g > 155) & (b > 160) & (yi < H_img * 0.55)] = 1

    # 3: 植被 (绿通道突出)
    seg[(g > r + 15) & (g > b + 10) & (seg == 0)] = 3

    # 5: 水域 (蓝通道突出)
    seg[(b > r + 20) & (b > g + 10) & (seg == 0)] = 5

    # 4: 地面 (下半部分)
    seg[(yi > H_img * 0.6) & (seg == 0)] = 4

    # 2: 建筑/硬表面 (default)
    seg[seg == 0] = 2

    seg = cv2.medianBlur(seg.astype(np.uint8), 11)
    seg = cv2.medianBlur(seg.astype(np.uint8), 5)
    return seg


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """语义类别 → RGB 可视化"""
    H_img, W_img = seg.shape[:2]
    sr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    for cid, col in SEG_COLORS.items():
        sr[seg == cid] = col
    return sr


def trajectory_to_frames(spec: dict) -> list:
    """场景 JSON → 逐帧参数列表"""
    traj = spec.get("trajectory", [
        {"t": 0, "action": "hover", "distance": 75,
         "norm_u": 0.5, "norm_v": 0.3}
    ])
    frames = []
    for pt in traj:
        d = float(pt.get("distance", 75))
        frames.append({
            "nu": float(pt.get("norm_u", 0.5)),
            "nv": float(pt.get("norm_v", 0.3)),
            "size": int(np.clip(3600 / max(d, 25), 12, 96)),  # +20%
            "dval": float(np.clip(d / 200, 0.15, 0.7)),
            "action": pt.get("action", "hover"),
            "distance": d,
        })
    return frames


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

_inpainter = None

def _get_inpainter():
    global _inpainter
    if _inpainter is None:
        import transformers.utils
        if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
            transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

        sys.path.insert(0, str(PROJECT_ROOT / "generator"))
        from lora_inpainter_v7 import LoraInpainterV7
        _inpainter = LoraInpainterV7()
        _inpainter.load()
    return _inpainter


def generate_with_lora(bg_path: str, full_scene_spec: dict,
                       session_id: int = 0,
                       gen_steps: int = 20,
                       blend_steps: int = 15,
                       blend_strength: float = 0.65,
                       cnet_scales: tuple = (0.85, 0.85),
                       use_lora: bool = True,
                       skip_cnet: bool = False) -> dict:
    """
    从背景图 + 场景 JSON 生成完整管线产物。

    JSON 驱动参数（每架无人机由 trajectory 数组指定）:
      - norm_u/norm_v: 归一化位置 (0~1)
      - distance: 距离(m)，决定 size_px 和 dval
      - action: hover/...

    ControlNet 条件:
      - drone_depth: bg_depth + drone silhouette 写入 dval
      - drone_seg: bg_seg + drone silhouette 写入 class_id=7

    Returns:
        {"ok": True, "images": {...}, "frame_count": N}
    """
    inpainter = _get_inpainter()
    sid = session_id

    # ── 加载背景 ──
    original = Image.open(bg_path).convert("RGB")
    bg_arr = np.array(original)
    bg_bgr = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2BGR)

    # ── 背景深度/分割（匹配原始分辨率！） ──
    depth = extract_depth_matched(Image.fromarray(bg_arr),
                                  (bg_arr.shape[1], bg_arr.shape[0]))
    seg = heuristic_segment(bg_arr)

    # 保存 bg（始终覆盖，防止 session_id 复用时显示旧图）
    bg_name = f"session_{sid}_bg.png"
    Image.fromarray(bg_arr).save(str(OUTPUT_DIR / bg_name))

    # 保存 bg_depth
    bg_depth_name = f"session_{sid}_bg_depth.png"
    Image.fromarray(depth_to_rgb(depth)).save(str(OUTPUT_DIR / bg_depth_name))

    # 保存 bg_seg
    bg_seg_name = f"session_{sid}_bg_seg.png"
    Image.fromarray(seg_to_rgb(seg)).save(str(OUTPUT_DIR / bg_seg_name))

    # ── 逐帧生成 ──
    frames = trajectory_to_frames(full_scene_spec)
    drone_depth_urls, drone_seg_urls, final_rgb_urls = [], [], []
    drone_clean_urls, drone_raw_urls = [], []
    ir_urls, bbox_urls = [], []

    for fi, frame in enumerate(frames):
        t0 = time.time()
        gen_seed = int(time.time_ns() / 1000 + fi * 777) % (2 ** 31)
        blend_seed = gen_seed + 100_000

        print(f"  [{fi + 1}/{len(frames)}] 无人机@({frame['nu']:.2f},{frame['nv']:.2f}) "
              f"size={frame['size']}px depth={frame['dval']:.3f} seed={gen_seed} ...")

        # 调用新管线：bg + depth + seg → 标记注入 → ControlNet 重绘
        out = inpainter.inpaint(
            background_bgr=bg_bgr,
            bg_depth=depth,
            bg_seg=seg,
            nu=frame["nu"],
            nv=frame["nv"],
            target_px=frame["size"],
            dval=frame["dval"],
            class_id=7,
            gen_seed=gen_seed,
            gen_steps=gen_steps,
            blend_seed=blend_seed,
            blend_steps=blend_steps,
            blend_strength=blend_strength,
            cnet_scales=cnet_scales,
            time_of_day=full_scene_spec.get("time_of_day", "afternoon"),
            weather=full_scene_spec.get("weather", "clear"),
            use_lora=use_lora,
            skip_cnet=skip_cnet,
        )

        # ── 保存 final_rgb ──
        final_name = f"session_{sid}_final_rgb_{fi}.png"
        final_bgr = out["result"]
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(final_rgb).save(str(OUTPUT_DIR / final_name))
        final_rgb_urls.append(f"/images/{final_name}")

        # ── 保存 IR (白热红外) ──
        ir_name = f"session_{sid}_ir_{fi}.png"
        ir_rgb = rgb_to_whitehot(final_bgr)  # BGR in → RGB out
        Image.fromarray(ir_rgb).save(str(OUTPUT_DIR / ir_name))
        ir_urls.append(f"/images/{ir_name}")

        # ── 保存 drone_depth（注入后的深度图——现在它是生成条件！） ──
        if "drone_depth" in out:
            dd_name = f"session_{sid}_drone_depth_{fi}.png"
            Image.fromarray(depth_to_rgb(out["drone_depth"])).save(
                str(OUTPUT_DIR / dd_name))
            drone_depth_urls.append(f"/images/{dd_name}")

            ds_name = f"session_{sid}_drone_seg_{fi}.png"
            Image.fromarray(seg_to_rgb(out["drone_seg"])).save(
                str(OUTPUT_DIR / ds_name))
            drone_seg_urls.append(f"/images/{ds_name}")

            # ── 提取 bbox + 可视化 ──
            ds = out["drone_seg"]
            ys, xs = np.where(ds == 7)
            if len(ys) > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bbox_viz = np.array(final_rgb).copy()  # 在 final_rgb 上画框
                cv2.rectangle(bbox_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bbox_viz, f"({x1},{y1})-({x2},{y2})", (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                bbox_name = f"session_{sid}_bbox_{fi}.png"
                Image.fromarray(bbox_viz).save(str(OUTPUT_DIR / bbox_name))
                bbox_urls.append(f"/images/{bbox_name}")
            else:
                bbox_urls.append("")

            # drone_clean: 抠图后的干净无人机
            if "drone_clean_arr" in out:
                dc_name = f"session_{sid}_drone_clean_{fi}.png"
                dc_arr = out["drone_clean_arr"]
                # 白底以便查看
                dc_bg = np.full_like(dc_arr, 255)
                mask = dc_arr.max(axis=2) > 10
                dc_bg[mask] = dc_arr[mask]
                Image.fromarray(dc_bg).save(str(OUTPUT_DIR / dc_name))
                drone_clean_urls.append(f"/images/{dc_name}")
            else:
                drone_clean_urls.append("")

            # drone_raw: LoRA 原始生成
            dr_name = f"session_{sid}_drone_raw_{fi}.png"
            out["drone_full"].save(str(OUTPUT_DIR / dr_name))
            drone_raw_urls.append(f"/images/{dr_name}")

            # drone_rgba: 保存 RGBA 供 focus reposition 复用
            if "drone_rgba" in out:
                drgba_name = f"session_{sid}_drone_rgba_{fi}.png"
                out["drone_rgba"].save(str(OUTPUT_DIR / drgba_name))
        else:
            drone_depth_urls.append("")
            drone_seg_urls.append("")
            drone_clean_urls.append("")
            drone_raw_urls.append("")
            bbox_urls.append("")

        elapsed = time.time() - t0
        print(f"       ✅ {elapsed:.1f}s")

    return {
        "ok": True,
        "images": {
            "bg": [f"/images/{bg_name}"],
            "bg_depth": [f"/images/{bg_depth_name}"],
            "bg_seg": [f"/images/{bg_seg_name}"],
            "drone_depth": drone_depth_urls,
            "drone_seg": drone_seg_urls,
            "final_rgb": final_rgb_urls,
            "ir": ir_urls,
            "bbox": bbox_urls,
            "drone_clean": drone_clean_urls,
            "drone_raw": drone_raw_urls,
        },
        "frame_count": len(frames),
        "trajectory_summary": [
            {"t": fi, "action": f["action"], "distance": f["distance"],
             "size_px": f["size"], "depth_val": round(f["dval"], 3)}
            for fi, f in enumerate(frames)
        ],
    }


def unload_inpainter():
    global _inpainter
    if _inpainter:
        _inpainter.unload()
        _inpainter = None


def focus_reposition(bg_path: str, drone_rgba_path: str,
                     full_scene_spec: dict,
                     drone_pos_x: float, drone_pos_y: float,
                     drone_scale: float,
                     session_id: int = 0,
                     frame_index: int = 0,
                     blend_steps: int = 15,
                     blend_strength: float = 0.65,
                     cnet_scales: tuple = (0.85, 0.85),
                     use_lora: bool = True,
                     skip_cnet: bool = False) -> dict:
    """
    快速重定位：跳过 LoRA 生成和 rembg，仅重新合成 + 注入 + 重绘。
    不重新生成 bg_depth/bg_seg、不重新跑 LoRA 生成/抠图。

    Args:
        bg_path: 背景图路径
        drone_rgba_path: 已抠图无人机 RGBA（session_X_drone_rgba_Y.png）
        full_scene_spec: 场景 spec（用于 time_of_day/weather）
        drone_pos_x: 前端红点 X 百分比 (0-100)
        drone_pos_y: 前端红点 Y 百分比 (0-100)
        drone_scale: 前端比例尺 (0.03-0.5)

    Returns:
        {"ok": True, "images": {"drone_depth": [...], "drone_seg": [...],
         "final_rgb": [...], "ir": [...], "bbox": [...]}}
    """
    inpainter = _get_inpainter()
    sid = session_id

    # ── 加载背景 + 条件图 ──
    original = Image.open(bg_path).convert("RGB")
    bg_arr = np.array(original)
    bg_bgr = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2BGR)

    depth = extract_depth_matched(Image.fromarray(bg_arr),
                                  (bg_arr.shape[1], bg_arr.shape[0]))
    seg = heuristic_segment(bg_arr)

    # ── 加载 drone_rgba ──
    drone_rgba = Image.open(drone_rgba_path).convert("RGBA")

    # ── 坐标转换：前端百分比(0-100) → nu/nv(0-1) ──
    nu = drone_pos_x / 100.0
    nv = drone_pos_y / 100.0

    # ── 比例尺 → target_px / dval ──
    # droneScale 0.03-0.5, 映射到 12-96 px (与原 trajectory_to_frames 对齐)
    target_px = int(np.clip(drone_scale * 192, 12, 96))
    dval = float(np.clip(drone_scale * 1.4, 0.15, 0.7))

    gen_seed = int(time.time_ns() / 1000 + frame_index * 777) % (2 ** 31)
    blend_seed = gen_seed + 100_000

    print(f"  [Focus] 无人机@({nu:.2f},{nv:.2f}) size={target_px}px "
          f"depth={dval:.3f} scale={drone_scale:.2f}")

    # ── 调用 inpainter.reposition() ──
    out = inpainter.reposition(
        background_bgr=bg_bgr,
        bg_depth=depth,
        bg_seg=seg,
        drone_rgba=drone_rgba,
        nu=nu, nv=nv,
        target_px=target_px,
        dval=dval,
        class_id=7,
        blend_seed=blend_seed,
        blend_steps=blend_steps,
        blend_strength=blend_strength,
        cnet_scales=cnet_scales,
        time_of_day=full_scene_spec.get("time_of_day", "afternoon"),
        weather=full_scene_spec.get("weather", "clear"),
        use_lora=use_lora,
        skip_cnet=skip_cnet,
    )

    # ── 保存产物 ──
    fi = frame_index
    images = {}

    # final_rgb
    final_name = f"session_{sid}_final_rgb_{fi}.png"
    final_bgr = out["result"]
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(final_rgb).save(str(OUTPUT_DIR / final_name))
    images["final_rgb"] = [f"/images/{final_name}"]

    # IR
    ir_name = f"session_{sid}_ir_{fi}.png"
    ir_rgb = rgb_to_whitehot(final_bgr)
    Image.fromarray(ir_rgb).save(str(OUTPUT_DIR / ir_name))
    images["ir"] = [f"/images/{ir_name}"]

    # drone_depth
    dd_name = f"session_{sid}_drone_depth_{fi}.png"
    Image.fromarray(depth_to_rgb(out["drone_depth"])).save(str(OUTPUT_DIR / dd_name))
    images["drone_depth"] = [f"/images/{dd_name}"]

    # drone_seg
    ds_name = f"session_{sid}_drone_seg_{fi}.png"
    Image.fromarray(seg_to_rgb(out["drone_seg"])).save(str(OUTPUT_DIR / ds_name))
    images["drone_seg"] = [f"/images/{ds_name}"]

    # bbox
    ds = out["drone_seg"]
    ys, xs = np.where(ds == 7)
    if len(ys) > 0:
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox_viz = np.array(final_rgb).copy()
        cv2.rectangle(bbox_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bbox_viz, f"({x1},{y1})-({x2},{y2})",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        bbox_name = f"session_{sid}_bbox_{fi}.png"
        Image.fromarray(bbox_viz).save(str(OUTPUT_DIR / bbox_name))
        images["bbox"] = [f"/images/{bbox_name}"]
    else:
        images["bbox"] = [""]

    return {
        "ok": True,
        "images": images,
        "nu": nu, "nv": nv,
        "target_px": target_px, "dval": dval,
    }


# ═══════════════════════════════════════════════════════════
# 离线 Demo
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    BG_PATH = str(PROJECT_ROOT /
                  "1-background-pool/curated_backgrounds/pure_sky/"
                  "bg_00070a0c1b49.jpg")

    SCENES = {
        "scene1_sky_hover": {
            "trajectory": [{"t": 0, "action": "hover", "distance": 60,
                            "norm_u": 0.5, "norm_v": 0.4}],
            "scene_type": "puresky", "time_of_day": "afternoon",
            "weather": "clear",
            "camera": {"position": "bottom"},
            "meta": {"raw_input": "蓝天正前，无人机悬停60m"}},
    }

    print("=" * 60)
    print("  ControlNet(depth+seg) 条件驱动 v7.1 Demo")
    print("=" * 60)

    for name, spec in SCENES.items():
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        result = generate_with_lora(BG_PATH, spec,
                                    session_id=hash(name) % 10000)
        print(f"  产物: {len(result['images']['final_rgb'])} 帧")
        for k, v in result["images"].items():
            print(f"    {k}: {v}")

    unload_inpainter()
    print(f"\n✅ 完成，产物在 {OUTPUT_DIR}/")

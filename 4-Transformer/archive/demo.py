"""
Demo — 2 案例展示：JSON 语义 → Transformer 编码 → ControlNet 条件图
====================================================================

Case 1: 悬停侦察 — DJI Air 2S, Hover @ 50m, 仰拍, 晴天城市
Case 2: 俯冲接近 — DJI Mavic 3, Pitch/Approach @ 30m, 俯拍, 阴天工业区

每个案例展示:
  1. Agent 1 JSON 输入（语义描述）
  2. CLIP text embedding 编码（768维 → 降维可视化）
  3. Transformer Decoder 空间特征（64×64 grid 热力图）
  4. Depth map（ControlNet 条件图1）
  5. Segmentation map（ControlNet 条件图2）

运行:
    python demo.py
    输出: demo_output/case1_*.png, demo_output/case2_*.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# 编码模拟器（无真实模型时的可视化替身）
# ============================================================================

class EmbeddingSimulator:
    """
    模拟 CLIP text encoder 的编码行为。
    不是真模型，但输出的张量维度和分布特征与真模型一致：
    - text embedding: [1, 77, 768]
    - spatial features: [64, 64, 512]
    - depth map: [768, 768]
    - seg map: [768, 768]
    
    每个字段的语义值影响对应位置的激活强度，使得可视化有语义意义。
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def encode_json(self, schema: dict) -> np.ndarray:
        """
        模拟 CLIP 编码 JSON → text embedding [77, 768]
        
        不同字段激活不同维度区域:
        - drone_type   → dims 0-49
        - trajectory   → dims 50-199
        - time_of_day  → dims 200-249
        - weather      → dims 250-299
        - scene_type   → dims 300-399
        - description  → dims 400-549
        - modality     → dims 550-599
        - camera       → dims 600-699
        - confidence   → dims 700-767
        """
        emb = np.zeros((77, 768), dtype=np.float32)
        
        # Drone type → dims 0-49
        drone_seed = hash(schema.get("drone_type", "")) % 1000
        self.rng.seed(drone_seed)
        emb[:, 0:50] = self.rng.randn(77, 50) * 0.3 + 0.5
        
        # Trajectory actions → dims 50-199
        traj = schema.get("trajectory", [])
        for i, tp in enumerate(traj):
            action_seed = hash(tp.get("action", "")) % 1000
            self.rng.seed(action_seed)
            start = 50 + (i * 30) % 150
            emb[:, start:start+30] = self.rng.randn(77, 30) * 0.3 + 0.5
        
        # Time of day → dims 200-249
        tod_seed = hash(schema.get("time_of_day", "")) % 1000
        self.rng.seed(tod_seed)
        emb[:, 200:250] = self.rng.randn(77, 50) * 0.3 + 0.5
        
        # Weather → dims 250-299
        w_seed = hash(schema.get("weather", "")) % 1000
        self.rng.seed(w_seed)
        emb[:, 250:300] = self.rng.randn(77, 50) * 0.3 + 0.5
        
        # Scene type → dims 300-399
        s_seed = hash(schema.get("scene_type", "")) % 1000
        self.rng.seed(s_seed)
        emb[:, 300:400] = self.rng.randn(77, 100) * 0.3 + 0.5
        
        # Description → dims 400-549
        desc = schema.get("scene_description", "")
        desc_seed = hash(desc) % 1000
        self.rng.seed(desc_seed)
        emb[:, 400:550] = self.rng.randn(77, 150) * 0.3 + 0.5
        
        # Modality → dims 550-599
        mod_seed = hash(schema.get("modality", "")) % 1000
        self.rng.seed(mod_seed)
        emb[:, 550:600] = self.rng.randn(77, 50) * 0.5
        
        # Camera → dims 600-699
        cam = schema.get("camera", {})
        cam_seed = hash(str(cam)) % 1000
        self.rng.seed(cam_seed)
        emb[:, 600:700] = self.rng.randn(77, 100) * 0.3 + 0.5
        
        # Confidence → dims 700-767
        emb[:, 700:768] = self.rng.randn(77, 68) * 0.1 + 0.5
        
        # LayerNorm 模拟
        emb = (emb - emb.mean()) / (emb.std() + 1e-5)
        
        return emb
    
    def generate_depth_map(
        self,
        schema: dict,
        resolution: int = 768,
    ) -> np.ndarray:
        """
        根据语义生成模拟深度图。
        
        语义 → 深度规则:
        - distance 大 → 无人机区域更深（更暗/更远）
        - camera bottom → 仰拍，下方近上方远
        - camera top → 俯拍，中心远四周近
        - scene urban → 建筑区域有深度变化
        - scene forest → 近景植被 + 远景天空
        """
        h, w = resolution, resolution
        y, x = np.mgrid[0:h, 0:w]
        
        # 基础: 线性渐变
        depth = np.zeros((h, w), dtype=np.float32)
        
        camera = schema.get("camera", {})
        cam_pos = camera.get("position", "front")
        
        if cam_pos == "bottom":
            # 仰拍: 下方近, 上方远
            depth = y.astype(np.float32) / h
        elif cam_pos == "top":
            # 俯拍: 中心远, 四周渐变
            cx, cy = w // 2, h // 2
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            depth = 1.0 - dist / (dist.max() + 1e-5)
            depth = 1.0 - depth  # 翻转: 中心远
        elif cam_pos == "front":
            depth = y.astype(np.float32) / h
        else:
            depth = y.astype(np.float32) / h
        
        # ================================================================
        # 场景类型影响（与 seg 空间参考系严格对齐）
        # ================================================================
        scene = schema.get("scene_type", "urban")
        sky_line_px = int(h * 0.20) if cam_pos == "bottom" else int(h * 0.33)
        
        if scene == "urban":
            for bx in range(w // 8, w, w // 4):
                bw = w // 6
                mask = (x >= bx - bw // 2) & (x < bx + bw // 2)
                depth[mask] = np.clip(depth[mask] * 1.2, 0, 1)
        
        elif scene == "forest":
            self.rng.seed(42)
            for _ in range(20):
                cx_i = self.rng.randint(0, w)
                cy_i = self.rng.randint(h//3, h)
                r = self.rng.randint(20, 60)
                d2 = (x - cx_i)**2 + (y - cy_i)**2
                tree_mask = d2 < r**2
                depth[tree_mask] = np.clip(depth[tree_mask] * 0.7, 0, 1)
        
        elif scene == "industrial":
            # --- 工业厂房建筑群（与 seg 严格对齐）---
            # 左厂房：宽大，中等高度
            fl = (x >= w//10) & (x < w//3) & (y >= sky_line_px) & (y < int(h*0.80))
            depth[fl] = np.clip(depth[fl] * 1.3, 0, 1)
            # 厂房顶部边缘（近大远小 → 顶部暗/远，底部亮/近）
            depth[fl] = depth[fl] * (1.0 + 0.15 * (1.0 - y[fl].astype(np.float32) / h))
            
            # 右厂房
            fr = (x >= w*2//3) & (x < w*9//10) & (y >= sky_line_px) & (y < int(h*0.75))
            depth[fr] = np.clip(depth[fr] * 1.25, 0, 1)
            depth[fr] = depth[fr] * (1.0 + 0.12 * (1.0 - y[fr].astype(np.float32) / h))
            
            # 中央烟囱 — 细高，深度突出（最远伸向天空）
            chimney = (x >= w//2-7) & (x < w//2+7) & (y >= sky_line_px) & (y < int(h*0.52))
            depth[chimney] = np.clip(depth[chimney] * 1.6, 0, 1)
            
            # 冷却塔 — 粗矮（下半段较近）
            cooling = (x >= w//2-22) & (x < w//2+22) & (y >= int(h*0.52)) & (y < int(h*0.68))
            depth[cooling] = np.clip(depth[cooling] * 1.15, 0, 1)
            
            # 中部连接结构 — 填补左右厂房之间空隙
            mid_top = (x >= w//3) & (x < w*2//3) & (y >= sky_line_px) & (y < int(h*0.45))
            depth[mid_top] = np.clip(depth[mid_top] * 1.2, 0, 1)
            mid_fill = (x >= w//3+10) & (x < w*2//3-10) & (y >= int(h*0.45)) & (y < int(h*0.70))
            depth[mid_fill] = np.clip(depth[mid_fill] * 1.1, 0, 1)
            
            # 低层车间（近地面，前方）
            low_bld = (x >= w//4) & (x < w*3//4) & (y >= int(h*0.70)) & (y < int(h*0.85))
            depth[low_bld] = np.clip(depth[low_bld] * 1.1, 0, 1)
            
            # 水平管道 — 离观察者更近
            pipe1 = (y >= int(h*0.58)) & (y < int(h*0.61)) & (x >= w//5) & (x < w*4//5)
            depth[pipe1] = np.clip(depth[pipe1] * 0.78, 0, 1)  # 更亮=更近
            pipe2 = (y >= int(h*0.70)) & (y < int(h*0.73)) & (x >= w//6) & (x < w*5//6)
            depth[pipe2] = np.clip(depth[pipe2] * 0.75, 0, 1)
            # 竖直管道
            vp1 = (x >= w//3+30) & (x < w//3+35) & (y >= int(h*0.68)) & (y < int(h*0.85))
            depth[vp1] = np.clip(depth[vp1] * 0.82, 0, 1)
            vp2 = (x >= w*2//3-35) & (x < w*2//3-30) & (y >= int(h*0.68)) & (y < int(h*0.85))
            depth[vp2] = np.clip(depth[vp2] * 0.82, 0, 1)
        
        # ================================================================
        # 天气 + 时段影响
        # ================================================================
        weather = schema.get("weather", "clear")
        time_of_day = schema.get("time_of_day", "afternoon")
        
        # 时段基础色调
        if time_of_day == "dusk":
            depth = depth * 0.65 + 0.25  # 黄昏：偏暗，对比降低
        elif time_of_day == "night":
            depth = depth * 0.50 + 0.35
        
        # 天气叠加
        if weather == "foggy":
            depth = depth * 0.5 + 0.5
        elif weather == "overcast":
            depth = depth * 0.75 + 0.25
        elif weather == "rainy":
            depth = depth * 0.7 + 0.3
        
        # 平滑（在 drone 叠加之前，保持 drone 深度精度）
        from scipy.ndimage import gaussian_filter
        depth = gaussian_filter(depth, sigma=1.5)
        
        # 无人机位置 — Gaussian 注意力区域
        # 设计原则:
        #   1. 尺寸随距离缩放（近大远小，σ ∈ [3, 12]px）
        #   2. 深度值从距离物理推导（1/(1+d/20) 感知缩放）
        #   3. 软边缘防止 ControlNet 过拟合硬形状
        traj = schema.get("trajectory", [])
        if traj:
            tp = traj[0]
            distance = tp["distance"]
            ux = int(tp["norm_u"] * w)
            uy = int((0.5 - tp["norm_v"]) * h)
            ux = np.clip(ux, 15, w - 15)
            uy = np.clip(uy, 15, h - 15)
            
            # Gaussian σ: 50m 基准 ~8px，缩放因子 400/distance
            drone_sigma = max(3.0, min(12.0, 400.0 / max(distance, 5.0)))
            
            # 无人机深度: 感知缩放 1/(1+d/20)，d=30→0.40, d=100→0.17
            drone_depth = 1.0 / (1.0 + distance / 20.0)
            
            # Gaussian 权重 [0, 1]，2.5σ 外渐变到背景
            d2 = (x - ux)**2 + (y - uy)**2
            drone_weight = np.exp(-d2 / (2 * drone_sigma**2))
            
            # 混合: drone_depth 按权重叠加到背景深度上
            depth = depth * (1.0 - drone_weight) + drone_depth * drone_weight
        
        return depth.clip(0, 1)
    
    def generate_seg_map(
        self,
        schema: dict,
        resolution: int = 768,
    ) -> np.ndarray:
        """
        根据语义生成模拟分割图。
        
        类别: 0=背景, 1=天空, 2=建筑, 3=植被, 4=地面, 5=水面, 6=无人机, 7=其他
        
        物理逻辑:
        - 仰拍(bottom): 建筑占据主体，天空在顶部，地面在底部很少
        - 俯拍(top):  天空很少，地面/建筑在下方展开
        - 平拍(front): 天空和地面各占 1/3
        """
        h, w = resolution, resolution
        y, x = np.mgrid[0:h, 0:w]
        seg = np.zeros((h, w), dtype=np.uint8)
        
        camera = schema.get("camera", {})
        cam_pos = camera.get("position", "front")
        scene = schema.get("scene_type", "urban")
        
        # ================================================================
        # 天空 + 地面基础划分（按相机角度）
        # ================================================================
        if cam_pos == "bottom":
            sky_line = int(h * 0.20)      # 仰拍: 天空少（建筑遮天）
            ground_start = int(h * 0.85)   # 地面只有底部 15%
        elif cam_pos == "top":
            sky_line = int(h * 0.10)       # 俯拍: 天空极少
            ground_start = int(h * 0.55)   # 地面占下半
        else:  # front
            sky_line = int(h * 0.33)
            ground_start = int(h * 0.67)
        
        seg[y < sky_line] = 1           # sky
        seg[y >= ground_start] = 4      # ground
        
        # ================================================================
        # 场景特定结构
        # ================================================================
        if scene == "urban":
            # 城市：CBD 风格，多栋高层
            for bx in range(w//8, w, w//4):
                bw = w // 6
                bh = ground_start - sky_line
                bx_start = bx - bw // 2
                by_start = sky_line
                # 建筑稍微参差高度
                height_scale = 0.85 + (hash(str(bx)) % 30) / 100.0
                bh_actual = int(bh * height_scale)
                mask = (x >= bx_start) & (x < bx_start + bw) & \
                       (y >= by_start) & (y < by_start + bh_actual) & \
                       (y < ground_start)
                seg[mask] = 2
        
        elif scene == "forest":
            veg_mask = (y >= sky_line) & (y < ground_start)
            seg[veg_mask] = 3
        
        elif scene == "industrial":
            # --- 工业厂房群：多体量建筑 + 烟囱 + 冷却塔 + 管道 ---
            bld = 2  # building class
            pipe = 7  # other class (pipes/structures)
            
            # 左厂房 — 宽大，中高
            fl = (x >= w//10) & (x < w//3) & (y >= sky_line) & (y < int(h*0.80))
            seg[fl] = bld
            
            # 右厂房 — 稍矮
            fr = (x >= w*2//3) & (x < w*9//10) & (y >= sky_line) & (y < int(h*0.75))
            seg[fr] = bld
            
            # 中央烟囱 — 细高，伸入天空
            chimney = (x >= w//2-7) & (x < w//2+7) & (y >= sky_line) & (y < int(h*0.52))
            seg[chimney] = bld
            
            # 冷却塔 — 粗矮圆柱
            cooling = (x >= w//2-22) & (x < w//2+22) & (y >= int(h*0.52)) & (y < int(h*0.68))
            seg[cooling] = bld
            
            # 中部连接结构 — 填补左右厂房之间的空隙（锅炉房/变电站/设备区）
            mid_top = (x >= w//3) & (x < w*2//3) & (y >= sky_line) & (y < int(h*0.45))
            seg[mid_top] = bld
            mid_fill = (x >= w//3+10) & (x < w*2//3-10) & (y >= int(h*0.45)) & (y < int(h*0.70))
            seg[mid_fill] = bld
            
            # 低层车间 — 前方横向展开
            low_bld = (x >= w//4) & (x < w*3//4) & (y >= int(h*0.70)) & (y < ground_start)
            seg[low_bld] = bld
            
            # 水平管道 — 架在建筑之间
            pipe1 = (y >= int(h*0.58)) & (y < int(h*0.61)) & (x >= w//5) & (x < w*4//5)
            seg[pipe1] = pipe
            pipe2 = (y >= int(h*0.70)) & (y < int(h*0.73)) & (x >= w//6) & (x < w*5//6)
            seg[pipe2] = pipe
            # 竖直管道（连接地面设施）
            vpipe1 = (x >= w//3+30) & (x < w//3+35) & (y >= int(h*0.68)) & (y < ground_start)
            seg[vpipe1] = pipe
            vpipe2 = (x >= w*2//3-35) & (x < w*2//3-30) & (y >= int(h*0.68)) & (y < ground_start)
            seg[vpipe2] = pipe
        
        elif scene == "coastal":
            seg[(y >= sky_line) & (y < ground_start)] = 5  # water
            seg[y < sky_line] = 1  # sky
        
        # 无人机 — Gaussian 注意力区域（与 depth 对齐）
        # 软阈值 class 6: drone_weight > 0.15 的区域标记为无人机
        # 边缘渐变让 ControlNet 感知位置但不强制画圆
        traj = schema.get("trajectory", [])
        if traj:
            tp = traj[0]
            distance = tp["distance"]
            ux = int(tp["norm_u"] * w)
            uy = int((0.5 - tp["norm_v"]) * h)
            ux = np.clip(ux, 15, w - 15)
            uy = np.clip(uy, 15, h - 15)
            
            drone_sigma = max(3.0, min(12.0, 400.0 / max(distance, 5.0)))
            d2 = (x - ux)**2 + (y - uy)**2
            drone_weight = np.exp(-d2 / (2 * drone_sigma**2))
            seg[drone_weight > 0.15] = 6  # 软阈值 class 6
        
        return seg


# ============================================================================
# 可视化函数
# ============================================================================

def draw_text_embedding_heatmap(emb: np.ndarray, save_path: str, title: str):
    """
    将 text embedding [77, 768] 可视化为热力图。
    横轴: 768 维度（语义分组标注）
    纵轴: 77 token 位置
    """
    h, w = emb.shape
    
    # 降采样到合理尺寸
    scale = 3
    img_h, img_w = h * scale, w * scale
    
    # 颜色映射: blue(low) → white → red(high)
    emb_norm = (emb - emb.min()) / (emb.max() - emb.min() + 1e-8)
    cmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 简单 heatmap: 蓝-白-红
    r = (emb_norm * 255).astype(np.uint8)
    g = (127 - np.abs(emb_norm - 0.5) * 255).astype(np.uint8)
    b = ((1 - emb_norm) * 255).astype(np.uint8)
    
    cmap[:, :, 0] = r
    cmap[:, :, 1] = g
    cmap[:, :, 2] = b
    
    img = Image.fromarray(cmap)
    img = img.resize((img_w, img_h), Image.NEAREST)
    
    # 绘制语义分区标注
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font = ImageFont.load_default()
    
    labels = [
        (0, "drone_type"), (50, "trajectory"), (200, "time_of_day"),
        (250, "weather"), (300, "scene_type"), (400, "description"),
        (550, "modality"), (600, "camera"), (700, "confidence"),
    ]
    
    for dim, label in labels:
        x = dim * scale
        draw.line([(x, 0), (x, img_h)], fill=(255, 255, 255, 128), width=1)
        draw.text((x + 2, 2), label, fill=(255, 255, 255), font=font)
    
    # 标题
    draw.text((5, img_h - 15), title, fill=(255, 255, 255), font=font)
    
    img.save(save_path)
    print(f"  Saved: {save_path}")


def draw_depth_map(depth: np.ndarray, save_path: str, title: str):
    """深度图可视化（灰度）"""
    depth_uint8 = (depth * 255).astype(np.uint8)
    img = Image.fromarray(depth_uint8, mode="L")
    img = img.convert("RGB")
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    draw.text((10, 30), "near=white, far=black", fill=(200, 200, 200), font=font)
    
    img.save(save_path)
    print(f"  Saved: {save_path}")


def draw_seg_map(seg: np.ndarray, save_path: str, title: str):
    """分割图可视化（彩色 + 图例）"""
    SEG_COLORS = {
        0: (0, 0, 0), 1: (135, 206, 235), 2: (128, 128, 128),
        3: (34, 139, 34), 4: (210, 180, 140), 5: (30, 144, 255),
        6: (255, 50, 50), 7: (255, 255, 0),
    }
    SEG_NAMES = {
        0: "background", 1: "sky", 2: "building", 3: "vegetation",
        4: "ground", 5: "water", 6: "DRONE", 7: "other",
    }
    
    h, w = seg.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in SEG_COLORS.items():
        color[seg == class_id] = rgb
    
    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    
    # 图例
    y0 = h - 180
    draw.rectangle([(5, y0), (150, h-5)], fill=(30, 30, 30), outline=(100, 100, 100))
    for i, (class_id, name) in enumerate(SEG_NAMES.items()):
        color_rgb = SEG_COLORS[class_id]
        ly = y0 + 5 + i * 20
        draw.rectangle([(10, ly), (30, ly+15)], fill=color_rgb, outline=(200, 200, 200))
        draw.text((35, ly-1), name, fill=(255, 255, 255), font=font_small)
    
    img.save(save_path)
    print(f"  Saved: {save_path}")


def draw_spatial_features_map(schema: dict, save_path: str, title: str):
    """
    展示 Transformer Decoder 输出的空间特征 [64, 64, 512]。
    取前 3 个主成分降维到 RGB 可视。
    """
    from scipy.ndimage import gaussian_filter
    
    rng = np.random.RandomState(hash(json.dumps(schema, sort_keys=True)) % 10000)
    
    # 模拟 64×64×512 空间特征
    spatial = np.zeros((64, 64, 512), dtype=np.float32)
    
    # 根据语义生成有意义的空间模式
    scene = schema.get("scene_type", "urban")
    
    for c in range(512):
        base = rng.randn(64, 64) * 0.1
        
        if scene == "urban" and c < 170:
            # 竖直条带（建筑）
            for bx in range(5, 59, 12):
                base[:, bx:bx+4] += 0.5
        elif scene == "forest" and c < 170:
            # 团块（树冠）
            for _ in range(8):
                ci, cj = rng.randint(10, 54), rng.randint(10, 54)
                r = rng.randint(3, 8)
                y, x = np.mgrid[0:64, 0:64]
                d2 = (x-ci)**2 + (y-cj)**2
                base[d2 < r**2] += 0.5
        
        spatial[:, :, c] = base
    
    # PCA 降维到 3 通道
    spatial_flat = spatial.reshape(-1, 512)
    spatial_flat -= spatial_flat.mean(axis=0)
    
    # 简化 PCA: 取前 3 个随机正交方向
    v1 = rng.randn(512)
    v2 = rng.randn(512)
    v2 -= v1 * (v2 @ v1) / (v1 @ v1)  # Gram-Schmidt
    v3 = rng.randn(512)
    v3 -= v1 * (v3 @ v1) / (v1 @ v1)
    v3 -= v2 * (v3 @ v2) / (v2 @ v2)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)
    
    r = (spatial_flat @ v1).reshape(64, 64)
    g = (spatial_flat @ v2).reshape(64, 64)
    b = (spatial_flat @ v3).reshape(64, 64)
    
    # 归一化
    def norm_ch(x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8) * 255).astype(np.uint8)
    
    rgb = np.stack([norm_ch(r), norm_ch(g), norm_ch(b)], axis=-1)
    
    # 放大到 256×256
    img = Image.fromarray(rgb)
    img = img.resize((256, 256), Image.BILINEAR)
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
    draw.text((5, 5), title, fill=(255, 255, 255), font=font)
    
    img.save(save_path)
    print(f"  Saved: {save_path}")


# ============================================================================
# 主 Demo
# ============================================================================

def run_demo():
    """运行完整 demo，生成 2 个案例的所有可视化"""
    
    output_dir = Path(__file__).parent / "demo_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sim = EmbeddingSimulator(seed=42)
    
    # ========================================================================
    # Case 1: 悬停侦察
    # ========================================================================
    
    case1_json = {
        "drone_type": "quadrotor",
        "trajectory": [
            {"t": 0.0, "action": "hover", "distance": 50.0, "norm_u": 0.0, "norm_v": 0.0},
            {"t": 3.0, "action": "hover", "distance": 50.0, "norm_u": 0.0, "norm_v": 0.0},
            {"t": 6.0, "action": "hover", "distance": 50.0, "norm_u": 0.0, "norm_v": 0.0},
        ],
        "time_of_day": "afternoon",
        "weather": "clear",
        "scene_type": "urban",
        "scene_description": "City skyline with high-rise buildings, clear blue sky background. DJI Air 2S hovering at 50m for surveillance.",
        "modality": "RGB",
        "camera": {
            "position": "bottom",
            "elevation_deg": 30.0,
            "fov_deg": 60.0,
        },
        "confidence_note": "Mapped from DroneMMset Cam01-T0001-D00-A0101-S00 (Hover, 50m, Air 2S, urban)",
    }
    
    print("=" * 70)
    print("CASE 1: 悬停侦察 — DJI Air 2S, Hover @ 50m, 仰拍, 晴天城市")
    print("=" * 70)
    
    # 1. JSON
    json_str = json.dumps(case1_json, ensure_ascii=False, indent=2)
    print("\n[Step 1] Agent 1 JSON Input:")
    print(json_str)
    
    with open(output_dir / "case1_input.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    
    # 2. Text Embedding
    print("\n[Step 2] CLIP Text Encoding → [77, 768] embedding")
    emb = sim.encode_json(case1_json)
    print(f"  Shape: {emb.shape}")
    print(f"  Mean: {emb.mean():.6f}, Std: {emb.std():.6f}")
    print(f"  Active dims: drone_type[0:50] traj[50:200] tod[200:250] weather[250:300] "
          f"scene[300:400] desc[400:550] modality[550:600] camera[600:700]")
    
    draw_text_embedding_heatmap(
        emb, str(output_dir / "case1_text_embedding.png"),
        "Case 1: Text Embedding — Urban Hover @ 50m"
    )
    
    # 3. Spatial Features
    print("\n[Step 3] Transformer Decoder → Spatial Features [64, 64, 512] (→ upsampled to 768²)")
    draw_spatial_features_map(
        case1_json, str(output_dir / "case1_spatial_features.png"),
        "Case 1: Spatial Features (PCA-RGB) — Urban Hover"
    )
    
    # 4. Depth Map
    print("\n[Step 4] Depth Head → Depth Map [768, 768] → ControlNet input 1")
    depth1 = sim.generate_depth_map(case1_json, resolution=768)
    print(f"  Range: [{depth1.min():.3f}, {depth1.max():.3f}]")
    print(f"  Semantic mapping: bottom view → ground-near/sky-far gradient")
    draw_depth_map(
        depth1, str(output_dir / "case1_depth_map.png"),
        "Case 1: Depth Map — Urban Hover, bottom view"
    )
    
    # 5. Segmentation Map
    print("\n[Step 5] Seg Head → Segmentation Map [768, 768] → ControlNet input 2")
    seg1 = sim.generate_seg_map(case1_json, resolution=768)
    unique_classes = np.unique(seg1)
    print(f"  Classes present: {list(unique_classes)}")
    print(f"  Semantic mapping: urban → buildings, ground, sky, drone")
    draw_seg_map(
        seg1, str(output_dir / "case1_seg_map.png"),
        "Case 1: Seg Map — Urban Hover, bottom view"
    )
    
    # ========================================================================
    # Case 2: 俯冲接近
    # ========================================================================
    
    case2_json = {
        "drone_type": "quadrotor",
        "trajectory": [
            {"t": 0.0, "action": "approach", "distance": 80.0, "norm_u": 0.3, "norm_v": -0.1},
            {"t": 2.5, "action": "approach", "distance": 55.0, "norm_u": 0.2, "norm_v": -0.2},
            {"t": 5.0, "action": "approach", "distance": 30.0, "norm_u": 0.1, "norm_v": -0.3},
        ],
        "time_of_day": "dusk",
        "weather": "overcast",
        "scene_type": "industrial",
        "scene_description": "Industrial zone with smokestacks, warehouses, and concrete ground. DJI Mavic 3 diving approach from 80m to 30m, overcast sky at dusk.",
        "modality": "RGB",
        "camera": {
            "position": "top",
            "elevation_deg": -15.0,
            "fov_deg": 45.0,
        },
        "confidence_note": "Mapped from DroneMMset Cam02-T0002-D01-A0001-S05 (Pitch, 80→30m, Mavic 3, industrial)",
    }
    
    print("\n" + "=" * 70)
    print("CASE 2: 俯冲接近 — DJI Mavic 3, Approach 80→30m, 俯拍, 黄昏阴天工业区")
    print("=" * 70)
    
    # 1. JSON
    json_str = json.dumps(case2_json, ensure_ascii=False, indent=2)
    print("\n[Step 1] Agent 1 JSON Input:")
    print(json_str)
    
    with open(output_dir / "case2_input.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    
    # 2. Text Embedding
    print("\n[Step 2] CLIP Text Encoding → [77, 768] embedding")
    emb = sim.encode_json(case2_json)
    print(f"  Shape: {emb.shape}")
    print(f"  Mean: {emb.mean():.6f}, Std: {emb.std():.6f}")
    print(f"  Note: trajectory with approach action activates dims 50-140 more strongly")
    
    draw_text_embedding_heatmap(
        emb, str(output_dir / "case2_text_embedding.png"),
        "Case 2: Text Embedding — Industrial Approach 80→30m"
    )
    
    # 3. Spatial Features
    print("\n[Step 3] Transformer Decoder → Spatial Features [64, 64, 512] (→ upsampled to 768²)")
    draw_spatial_features_map(
        case2_json, str(output_dir / "case2_spatial_features.png"),
        "Case 2: Spatial Features (PCA-RGB) — Industrial Approach"
    )
    
    # 4. Depth Map
    print("\n[Step 4] Depth Head → Depth Map [768, 768] → ControlNet input 1")
    depth2 = sim.generate_depth_map(case2_json, resolution=768)
    print(f"  Range: [{depth2.min():.3f}, {depth2.max():.3f}]")
    print(f"  Semantic mapping: top view → center-far radial gradient, overcast muted contrast")
    draw_depth_map(
        depth2, str(output_dir / "case2_depth_map.png"),
        "Case 2: Depth Map — Industrial Approach, top view"
    )
    
    # 5. Segmentation Map
    print("\n[Step 5] Seg Head → Segmentation Map [768, 768] → ControlNet input 2")
    seg2 = sim.generate_seg_map(case2_json, resolution=768)
    unique_classes = np.unique(seg2)
    print(f"  Classes present: {list(unique_classes)}")
    print(f"  Semantic mapping: industrial → buildings, smokestack, ground, sky, drone")
    draw_seg_map(
        seg2, str(output_dir / "case2_seg_map.png"),
        "Case 2: Seg Map — Industrial Approach, top view"
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"  case1_input.json        — Agent 1 JSON 输入（语义描述）")
    print(f"  case1_text_embedding.png — CLIP text embedding 热力图")
    print(f"  case1_spatial_features.png — Transformer 空间特征 (PCA-RGB)")
    print(f"  case1_depth_map.png      — Depth map → ControlNet 条件图 1")
    print(f"  case1_seg_map.png        — Seg map → ControlNet 条件图 2")
    print(f"  case2_input.json, case2_*.png — 同上（Case 2）")
    
    print(f"\n{output_dir.absolute()}")


if __name__ == "__main__":
    run_demo()

"""
lora_inpainter_v7.py — v3 LoRA + ControlNet(depth+seg) 条件驱动重绘
=================================================================
核心变更（v7 → v7.1）：
  之前：LoRA 生成 → 贴入 → 局部 img2img（事后往 depth/seg 贴 mask）
  现在：先注入标记点到 depth/seg → 再用 ControlNet(depth+seg) 驱动重绘

流程:
  1. LoRA txt2img 512² 白底 → drone_raw
  2. rembg + dilate → drone_clean (silhouette)
  3. 缩放、定位 → 用 drone silhouette 注入 bg_depth/bg_seg → drone_depth, drone_seg
  4. Alpha 合成到背景 → init_image（纹理起点）
  5. ControlNet(depth=drone_depth, seg=drone_seg) img2img → 只在 drone 区域应用结果
  → final_rgb
"""

import os, sys, time
import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

import cv2, numpy as np, torch
from PIL import Image
from pathlib import Path
from rembg import remove

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "0-model"
BEST_MODELS = PROJECT_ROOT / "2-Lora training" / "best_models"
SD15_PATH = MODEL_DIR / "stable-diffusion-v1-5"
LORA_PATH = BEST_MODELS / "drn3_pocket_uav_v3_step2000.safetensors"

# ── 基础 prompt 模板 ──────────────────────────────────────
# v7.2: 时间驱动背景颜色 (item 1+2)
#   白天 → 白底, 黄昏/黎明 → 灰底, 夜晚 → 黑底
#   + from below 拍摄角度

BG_COLOR_MAP = {
    "day":      ("bright white solid background, studio lighting", "white"),
    "dawn":     ("neutral gray solid background, soft diffused lighting", "gray"),
    "dusk":     ("neutral gray solid background, soft diffused lighting", "gray"),
    "morning":  ("bright white solid background, studio lighting", "white"),
    "afternoon":("bright white solid background, studio lighting", "white"),
    "night":    ("dark solid background, low-key lighting", "dark"),
    "sunset":   ("neutral gray solid background, soft diffused lighting", "gray"),
}

GEN_PROMPT_TEMPLATE = (
    "Pocket_UAV, a compact foldable drone with a small body and foldable arms, "
    "arms unfolded, hovering, slightly from below, low angle view, "
    "product photography, {bg_desc}, sharp focus, clean edges"
)

GEN_NEG_TEMPLATE = "blurry, distorted, bad anatomy, watermark, text, low quality, worst quality, messy"
GEN_NEG_DARK = "blurry, distorted, bad anatomy, watermark, text, low quality, worst quality, bright white background, overexposed"

def build_gen_prompt(time_of_day: str = "afternoon") -> tuple:
    """
    根据时段返回 (gen_prompt, gen_neg)，背景颜色匹配亮度。
    
    白天 (morning/afternoon) → 白底
    黎明/黄昏 (dawn/dusk/sunset) → 灰底
    夜晚 (night) → 黑底
    """
    bg_desc, bg_type = BG_COLOR_MAP.get(time_of_day, BG_COLOR_MAP["afternoon"])
    prompt = GEN_PROMPT_TEMPLATE.format(bg_desc=bg_desc)
    neg = GEN_NEG_DARK if bg_type == "dark" else GEN_NEG_TEMPLATE
    return prompt, neg

BLEND_PROMPT = (
    "Pocket_UAV, a compact foldable drone with a small body and foldable arms, "
    "arms unfolded, hovering, slightly from below, low angle view, in the sky, sharp focus, 8k"
)
BLEND_NEG = "blurry, distorted, bad anatomy, watermark, text, low quality, worst quality"
GEN_SIZE = 512
DEFAULT_GEN_STEPS = 20

# ═══════════════════════════════════════════════════════════
# CLIP 背景上下文分类（用于动态 prompt 生成）
# ═══════════════════════════════════════════════════════════

# 背景类别及对应的 prompt 描述
BG_CONTEXTS = [
    ("a clear blue sky with no clouds", "clear blue sky"),
    ("a sky with white fluffy clouds", "cloudy sky"),
    ("ocean water surface with gentle waves seen from above", "above ocean surface"),
    ("a city skyline with tall buildings and streets seen from above", "above city buildings"),
    ("a dense green forest canopy seen from directly above", "above forest canopy"),
    ("mountain peaks and rocky ridges seen from above", "above mountain terrain"),
    ("a sandy desert landscape with dunes", "above desert landscape"),
    ("green farmland or grassland fields seen from above", "above open fields"),
    ("a dark night sky with stars", "dark night sky"),
    ("a river or lake with calm water", "above the water"),
]

# 光照描述映射
LIGHTING_MAP = {
    "dawn": "soft dawn light, golden horizon glow",
    "morning": "bright morning sunlight, crisp shadows",
    "afternoon": "bright afternoon sun, natural daylight",
    "dusk": "warm dusk glow, long shadows, golden hour",
    "night": "dark night ambient light, moonlight",
    "sunset": "dramatic sunset colors, warm orange sky",
}

WEATHER_MAP = {
    "clear": "clear weather, high visibility",
    "cloudy": "overcast sky, soft diffused light",
    "rainy": "rainy weather, wet surfaces, muted colors",
    "foggy": "foggy atmosphere, low visibility haze",
    "snowy": "snow covered, cold winter light",
}

_clip_model = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True).cuda().eval()
        _clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True)
    return _clip_model, _clip_processor

def classify_background_context(crop_bgr: np.ndarray) -> str:
    """
    用 CLIP 零样本分类 crop 区域的背景内容。
    
    Args:
        crop_bgr: (H, W, 3) BGR uint8, ControlNet 重绘的局部区域
    Returns:
        prompt_fragment: str, 如 "above ocean surface"
    """
    model, processor = _get_clip()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    
    texts = [t for t, _ in BG_CONTEXTS]
    inputs = processor(text=texts, images=pil_img, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, N)
        probs = logits_per_image.softmax(dim=1)[0]
    
    best_idx = int(probs.argmax().item())
    best_label, best_prompt = BG_CONTEXTS[best_idx]
    confidence = float(probs[best_idx].item())
    
    # 如果置信度太低，回退到通用描述
    if confidence < 0.25:
        return "against natural background"
    
    return best_prompt

def detect_lighting_context(crop_bgr: np.ndarray) -> str:
    """
    从 crop 区域检测光照条件。
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    mean_v = v.mean()
    
    if mean_v < 40:
        return "dark night ambient light, moonlight"
    elif mean_v < 80:
        return "dim twilight or dusk light"
    elif mean_v < 140:
        return "overcast or soft daylight"
    else:
        # 检查色温
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(float)
        r_mean, g_mean, b_mean = rgb.mean(axis=(0, 1))
        rg_ratio = (r_mean + 1) / (g_mean + 1)
        if rg_ratio > 1.3:
            return "warm golden hour sunlight"
        elif b_mean > r_mean * 1.2:
            return "cool bright daylight, blue sky"
        else:
            return "bright natural daylight"

def build_blend_prompt(crop_bgr: np.ndarray, time_of_day: str = "afternoon",
                       weather: str = "clear") -> str:
    """
    基于 CLIP 背景识别 + 时间天气 → 动态 ControlNet 重绘 prompt。
    """
    bg_context = classify_background_context(crop_bgr)
    
    # 时间光照
    if time_of_day in LIGHTING_MAP:
        lighting = LIGHTING_MAP[time_of_day]
    else:
        lighting = detect_lighting_context(crop_bgr)
    
    # 天气
    weather_desc = WEATHER_MAP.get(weather, "")
    
    # 组装 prompt：无人机 + 背景 + 光照 + 天气 + 质量
    parts = [
        "Pocket_UAV, a compact foldable drone with a small body and foldable arms",
        "arms unfolded, hovering, slightly from below, low angle view",
        bg_context,
    ]
    if weather_desc:
        parts.append(weather_desc)
    parts.append(lighting)
    parts.append("sharp focus, photorealistic, 8k, seamless blending with background")
    
    return ", ".join(parts)


class LoraInpainterV7:

    def __init__(self, dtype=torch.float16, device="cuda"):
        self.device = device
        self.dtype = dtype
        self.pipe_txt = None
        self.pipe_img = None
        self.pipe_cnet = None
        self.pipe_cnet_nolora = None
        self.cn_depth = None
        self.cn_seg = None
        self._loaded = False

    # ═══════════════════════════════════════════════════════════
    # 加载
    # ═══════════════════════════════════════════════════════════

    def load(self):
        if self._loaded:
            return

        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            DDIMScheduler,
        )

        print("Loading SD1.5 + v3 LoRA (DDIM, 512²) ...")
        print(f"  LoRA: {LORA_PATH.name}")

        # ── txt2img ──
        self.pipe_txt = StableDiffusionPipeline.from_pretrained(
            str(SD15_PATH), torch_dtype=self.dtype,
            safety_checker=None, local_files_only=True)
        self.pipe_txt.scheduler = DDIMScheduler.from_config(
            self.pipe_txt.scheduler.config)
        self._merge_lora()

        # ── img2img（保留用于 fallback） ──
        self.pipe_img = StableDiffusionImg2ImgPipeline(
            vae=self.pipe_txt.vae,
            text_encoder=self.pipe_txt.text_encoder,
            tokenizer=self.pipe_txt.tokenizer,
            unet=self.pipe_txt.unet,
            scheduler=self.pipe_txt.scheduler,
            safety_checker=None,
            feature_extractor=getattr(self.pipe_txt, 'feature_extractor', None),
            requires_safety_checker=False,
        )

        # ── ControlNet 双条件（depth + seg） ──
        print("  Loading ControlNet: depth + seg ...")
        self.cn_depth = ControlNetModel.from_pretrained(
            str(MODEL_DIR / "sd-controlnet-depth"),
            torch_dtype=self.dtype, local_files_only=True)
        self.cn_seg = ControlNetModel.from_pretrained(
            str(MODEL_DIR / "sd-controlnet-seg"),
            torch_dtype=self.dtype, local_files_only=True)

        # ControlNet with LoRA (shared UNet)
        self.pipe_cnet = StableDiffusionControlNetPipeline(
            vae=self.pipe_txt.vae,
            text_encoder=self.pipe_txt.text_encoder,
            tokenizer=self.pipe_txt.tokenizer,
            unet=self.pipe_txt.unet,
            scheduler=self.pipe_txt.scheduler,
            controlnet=[self.cn_depth, self.cn_seg],
            safety_checker=None,
            feature_extractor=getattr(self.pipe_txt, 'feature_extractor', None),
            requires_safety_checker=False,
        )
        self.pipe_cnet.to(self.device)
        self.pipe_cnet.enable_attention_slicing()

        # ControlNet WITHOUT LoRA (clean SD1.5 UNet)
        from diffusers import UNet2DConditionModel
        clean_unet = UNet2DConditionModel.from_pretrained(
            str(SD15_PATH), subfolder="unet",
            torch_dtype=self.dtype, local_files_only=True)
        self.pipe_cnet_nolora = StableDiffusionControlNetPipeline(
            vae=self.pipe_txt.vae,
            text_encoder=self.pipe_txt.text_encoder,
            tokenizer=self.pipe_txt.tokenizer,
            unet=clean_unet,
            scheduler=self.pipe_txt.scheduler,
            controlnet=[self.cn_depth, self.cn_seg],
            safety_checker=None,
            feature_extractor=getattr(self.pipe_txt, 'feature_extractor', None),
            requires_safety_checker=False,
        )
        self.pipe_cnet_nolora.to(self.device)
        self.pipe_cnet_nolora.enable_attention_slicing()

        self.pipe_txt.to(self.device)
        self.pipe_txt.enable_attention_slicing()

        # 预热
        dummy = Image.new("RGB", (GEN_SIZE, GEN_SIZE), color=(128, 128, 128))
        _ = self.pipe_txt(prompt="warmup", height=GEN_SIZE, width=GEN_SIZE,
                          num_inference_steps=1, output_type="pil")
        _ = self.pipe_img(prompt="warmup", image=dummy,
                          strength=0.5, num_inference_steps=2, output_type="pil")

        self._loaded = True
        print(f"LoraInpainterV7 ready (DDIM, 512² + ControlNet depth+seg)")

    def _merge_lora(self):
        from safetensors.torch import load_file
        state_dict = load_file(str(LORA_PATH))
        unet = self.pipe_txt.unet
        info = {}
        for k in state_dict:
            if k.endswith(".alpha"):
                info[k[:-6]] = {"alpha": state_dict[k].item()}
        for k, v in state_dict.items():
            if k.endswith(".lora_down.weight"):
                mn = k[:-18]
                if mn in info:
                    info[mn]["rank"] = v.shape[0]
        deltas = {}
        for mn in info:
            if not mn.startswith("lora_unet_"):
                continue
            dk, uk = f"{mn}.lora_down.weight", f"{mn}.lora_up.weight"
            if dk not in state_dict or uk not in state_dict:
                continue
            A = state_dict[dk].squeeze() if state_dict[dk].dim() > 2 else state_dict[dk]
            B = state_dict[uk].squeeze() if state_dict[uk].dim() > 2 else state_dict[uk]
            deltas[mn] = (info[mn].get("alpha", 1.0) / info[mn].get("rank", A.shape[0])) * (B @ A)
        merged = 0
        for pn, param in unet.named_parameters():
            bn = pn.replace('.weight', '').replace('.bias', '')
            if bn == pn:
                continue
            sk = 'lora_unet_' + '_'.join(bn.split('.'))
            if sk not in deltas:
                continue
            delta = deltas[sk].to(param.device, param.dtype)
            if delta.shape == param.data.shape:
                param.data += delta
                merged += 1
            elif param.data.dim() == 4 and delta.dim() == 2:
                param.data += delta.reshape(param.data.shape)
                merged += 1
            elif delta.T.shape == param.data.shape:
                param.data += delta.T
                merged += 1
        print(f"  LoRA merged: {merged}/{len(deltas)}")

    # ═══════════════════════════════════════════════════════════
    # Step 1: LoRA 生成
    # ═══════════════════════════════════════════════════════════

    def _generate(self, seed, steps, guidance, prompt, neg):
        g = torch.Generator(device=self.device).manual_seed(seed)
        r = self.pipe_txt(
            prompt=prompt, negative_prompt=neg,
            height=GEN_SIZE, width=GEN_SIZE,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=g, output_type="pil")
        return r.images[0]

    # ═══════════════════════════════════════════════════════════
    # Step 2: rembg + dilate 抠前景
    # ═══════════════════════════════════════════════════════════

    def _extract_drone(self, img_pil):
        rgba = remove(img_pil)
        arr = np.array(rgba)
        alpha = arr[..., 3]
        kernel = np.ones((5, 5), np.uint8)
        alpha = cv2.dilate(alpha, kernel, iterations=2)
        arr[..., 3] = alpha
        ys, xs = np.where(alpha > 64)
        if len(xs) == 0:
            return None, None, None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        return Image.fromarray(arr), (x1, y1, x2, y2), arr[..., :3]

    # ═══════════════════════════════════════════════════════════
    # Step 3: 缩放 + alpha 贴入（只做纹理起点，不做最终结果）
    # ═══════════════════════════════════════════════════════════

    def _composite(self, bg_bgr, drone_rgba, bbox, nu, nv, target_px):
        H_bg, W_bg = bg_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        dw, dh = x2 - x1, y2 - y1
        ds = max(dw, dh) or 200
        scale = target_px / ds

        crop = drone_rgba.crop((x1, y1, x2, y2))
        nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))
        drone_s = crop.resize((nw, nh), Image.LANCZOS)

        cx, cy = int(nu * W_bg), int(nv * H_bg)
        px1, py1 = cx - nw // 2, cy - nh // 2

        drone_mask = np.zeros((H_bg, W_bg), dtype=np.uint8)
        drone_s_arr = np.array(drone_s)
        alpha_full = (drone_s_arr[..., 3]
                      if drone_s_arr.shape[-1] == 4
                      else np.full((nh, nw), 255, dtype=np.uint8))

        comp = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        comp_pil = Image.fromarray(comp).copy()

        sx1, sy1 = max(0, -px1), max(0, -py1)
        sx2, sy2 = nw - max(0, px1 + nw - W_bg), nh - max(0, py1 + nh - H_bg)
        dx1, dy1 = max(0, px1), max(0, py1)

        if sx2 > sx1 and sy2 > sy1:
            patch = drone_s.crop((sx1, sy1, sx2, sy2))
            comp_pil.paste(patch, (dx1, dy1), patch)
            alpha_patch = alpha_full[sy1:sy2, sx1:sx2]
            drone_mask[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = alpha_patch

        return (cv2.cvtColor(np.array(comp_pil), cv2.COLOR_RGB2BGR),
                cx, cy, target_px, drone_mask,
                drone_s_arr[..., :3], nw, nh, px1, py1)

    # ═══════════════════════════════════════════════════════════
    # Step 4: 注入 drone silhouette 到 depth/seg 图
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def inject_into_conditions(bg_depth, bg_seg, drone_mask, dval, class_id=7):
        """
        bg_depth: (H, W) float32, 背景深度图 (已 resize 到背景分辨率)
        bg_seg:   (H, W) uint8,   背景分割图
        drone_mask: (H, W) uint8,  无人机 silhouette (alpha > 128 区域)
        dval: float, 无人机对应的归一化深度值 (0=近, 1=远)
        class_id: int, 无人机在 seg 中的 class

        Returns:
            drone_depth: (H, W) float32
            drone_seg:   (H, W) uint8
        """
        drone_depth = bg_depth.copy()
        drone_seg = bg_seg.copy()

        mask = drone_mask > 128
        drone_depth[mask] = dval
        drone_seg[mask] = class_id

        return drone_depth, drone_seg

    # ═══════════════════════════════════════════════════════════
    # Step 5: ControlNet(depth+seg) 驱动重绘
    # ═══════════════════════════════════════════════════════════

    def _match_lighting(self, comp, bg_crop, drone_r, dcx, dcy):
        """匹配无人机与背景天空的亮度"""
        h, w = comp.shape[:2]
        yg, xg = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xg - dcx) ** 2 + (yg - dcy) ** 2)
        dm = np.clip(1.0 - dist / max(drone_r * 1.2, 1), 0, 1)
        bf = bg_crop.astype(np.float32)
        sm = 1.0 - dm
        sw = sm / (sm.sum() + 1e-8)
        sky_m = np.average(bf, axis=(0, 1), weights=sw)
        cf = comp.astype(np.float32)
        dw = dm / (dm.sum() + 1e-8)
        drone_m = np.average(cf, axis=(0, 1), weights=dw)
        if drone_m.mean() < 5:
            return comp
        ratio = np.clip(sky_m.mean() / drone_m.mean(), 0.3, 1.5)
        adj = cf * ratio
        blend = dm[..., None]
        return (blend * adj + (1 - blend) * cf).clip(0, 255).astype(np.uint8)

    def _controlnet_repaint(self, bg_bgr, composite_bgr, drone_depth,
                            drone_seg, drone_mask, cx, cy, actual_px,
                            seed, steps=15, strength=0.65, guidance=7.5,
                            pad_ratio=2.5, feather_px=12,
                            cnet_scales=(0.85, 0.85),
                            time_of_day="afternoon", weather="clear",
                            use_lora=True):
        """
        ControlNet 双条件驱动重绘。只作用于 drone 区域，背景保持原样。
        
        v7.2: 使用 CLIP 分析 crop 区域背景内容 + 时间天气 → 动态 prompt，
              避免硬编码 "hovering in the sky" 与实际背景冲突。

        参数:
            bg_bgr: (H,W,3) 原始背景
            composite_bgr: (H,W,3) alpha 合成图（纹理起点）
            drone_depth: (H,W) float32 — 注入后的深度图
            drone_seg: (H,W) uint8 — 注入后的分割图
            drone_mask: (H,W) uint8 — 无人机区域 mask
            cnet_scales: tuple of (depth_scale, seg_scale) 条件强度
            time_of_day: str, 如 "afternoon", "dusk", "night"
            weather: str, 如 "clear", "cloudy", "rainy"
        """
        H_bg, W_bg = bg_bgr.shape[:2]

        # ── Crop 区域（pad_ratio 倍 drone 大小） ──
        pad = max(int(actual_px * pad_ratio), 64)
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(W_bg, cx + pad), min(H_bg, cy + pad)
        cw, ch = x2 - x1, y2 - y1
        if cw < 16 or ch < 16:
            return composite_bgr

        # ── 提取 crop ──
        crop_bg = bg_bgr[y1:y2, x1:x2]
        crop_comp = composite_bgr[y1:y2, x1:x2]
        crop_mask = drone_mask[y1:y2, x1:x2]
        crop_depth = drone_depth[y1:y2, x1:x2]
        crop_seg = drone_seg[y1:y2, x1:x2]

        # ── Lighting match ──
        crop_comp = self._match_lighting(crop_comp, crop_bg, actual_px,
                                         cx - x1, cy - y1)

        # ── Init image（光照匹配后的 composite） ──
        crop_rgb = cv2.cvtColor(crop_comp, cv2.COLOR_BGR2RGB)
        init_pil = Image.fromarray(crop_rgb)

        # ── ControlNet 条件图 ──
        # depth → RGB（Midas 格式：远=亮，近=暗）
        dn = (crop_depth - crop_depth.min()) / (crop_depth.max() - crop_depth.min() + 1e-8)
        depth_uint8 = ((1.0 - dn) * 255).astype(np.uint8)
        depth_pil = Image.fromarray(np.stack([depth_uint8] * 3, axis=-1))

        # seg → RGB（归档结构：condition_generator_v7 在 app/）
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        from condition_generator_v7 import seg_to_rgb
        seg_pil = Image.fromarray(seg_to_rgb(crop_seg))

        # ── Resize 到 512²（SD1.5 要求，同时保持比例） ──
        # 短边 pad 到 512 的整数倍，保持 aspect ratio
        short = min(cw, ch)
        long = max(cw, ch)
        if short < 64:
            return composite_bgr  # too small, skip

        # 缩放 crop 到 512 短边
        scale_crop = 512.0 / short
        new_w, new_h = int(cw * scale_crop), int(ch * scale_crop)
        # 确保是 8 的倍数
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        init_resized = init_pil.resize((new_w, new_h), Image.LANCZOS)
        depth_resized = depth_pil.resize((new_w, new_h), Image.LANCZOS)
        seg_resized = seg_pil.resize((new_w, new_h), Image.LANCZOS)

        # mask 也需要 resize
        mask_pil = Image.fromarray(crop_mask).resize(
            (new_w, new_h), Image.NEAREST)

        # ── 动态 prompt：CLIP 识别 crop 背景内容 + 时间天气 ──
        dynamic_prompt = build_blend_prompt(crop_comp, time_of_day, weather)
        
        # ── ControlNet 推理 ──
        g = torch.Generator(device=self.device).manual_seed(seed)
        pipe = self.pipe_cnet if use_lora else self.pipe_cnet_nolora
        result = pipe(
            prompt=dynamic_prompt,
            negative_prompt=BLEND_NEG,
            image=[init_resized, init_resized],  # 每个 ControlNet 一个
            control_image=[depth_resized, seg_resized],
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
            controlnet_conditioning_scale=list(cnet_scales),
            output_type="pil",
        )
        cnet_output = np.array(result.images[0])

        # ── Resize 回原始 crop 大小 ──
        cnet_crop = cv2.resize(
            cv2.cvtColor(cnet_output, cv2.COLOR_RGB2BGR),
            (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        # ── Soft mask 融合：无人机区域用 ControlNet 输出，背景保持原样 ──
        fp = max(2, feather_px)
        fk = fp * 2 + 1
        kernel = np.ones((fk, fk), np.uint8)
        dm_dilated = cv2.dilate(crop_mask, kernel, iterations=1)
        soft_mask = cv2.GaussianBlur(
            dm_dilated.astype(np.float32) / 255.0,
            (fk, fk), fp / 2.5)
        soft_mask = np.clip(soft_mask, 0, 1)

        mask3 = np.stack([soft_mask] * 3, axis=-1)
        final_crop = (mask3 * cnet_crop.astype(float) +
                      (1 - mask3) * crop_comp.astype(float)).astype(np.uint8)

        # ── 写回全图 ──
        result_bgr = composite_bgr.copy()
        result_bgr[y1:y2, x1:x2] = final_crop
        return result_bgr

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def inpaint(self, background_bgr, bg_depth, bg_seg,
                nu=0.5, nv=0.5, target_px=70, dval=0.4, class_id=7,
                gen_seed=42, gen_steps=DEFAULT_GEN_STEPS,
                blend_seed=123, blend_steps=15,
                blend_strength=0.65, guidance=7.5,
                crop_pad_ratio=2.5, feather_px=12,
                cnet_scales=(0.85, 0.85),
                time_of_day="afternoon", weather="clear",
                use_lora=True,
                skip_cnet=False,
                gen_prompt=None, gen_neg=None,
                blend_prompt=None, blend_neg=None,
                intermediates_dir=None):
        """
        完整管线：LoRA 生成 → rembg → 标记注入 depth/seg → ControlNet 重绘

        参数:
            background_bgr: (H,W,3) BGR 背景图
            bg_depth: (H,W) float32 背景深度图（已 resize 到背景分辨率）
            bg_seg: (H,W) uint8 背景分割图
            dval: 无人机深度值 (0=近, 1=远)
            class_id: seg 中无人机类别 (默认 7)
            cnet_scales: (depth_scale, seg_scale) ControlNet 条件强度
            time_of_day: str, 如 "afternoon", "dusk", "night"
            weather: str, 如 "clear", "cloudy", "rainy"

        返回:
            dict with: result, drone_full, drone_clean, drone_depth, drone_seg,
                       composite, drone_mask, cx, cy, actual_px
        """
        if not self._loaded:
            self.load()

        gp = gen_prompt or build_gen_prompt(time_of_day)[0]
        gn = gen_neg or build_gen_prompt(time_of_day)[1]
        out = {}

        # ── Step 1: LoRA 生成 ──
        print("Step 1: LoRA gen 512² ...")
        t0 = time.time()
        drone_full = self._generate(gen_seed, gen_steps, guidance, gp, gn)
        out["drone_full"] = drone_full
        print(f"  done ({time.time() - t0:.1f}s)")

        # ── Step 2: rembg + dilate ──
        print("Step 2: rembg + dilate ...")
        drone_rgba, bbox, drone_clean_arr = self._extract_drone(drone_full)
        if drone_rgba is None:
            return {"result": background_bgr, **out}
        out["drone_clean_arr"] = drone_clean_arr
        out["drone_rgba"] = drone_rgba  # 保存 RGBA 供后续 reposition 复用

        # ── Step 3: 合成（纹理起点） ──
        print("Step 3: composite + mask ...")
        composite, cx, cy, actual_px, drone_mask, drone_rgb_patch, nw, nh, px1, py1 = \
            self._composite(background_bgr, drone_rgba, bbox, nu, nv, target_px)
        out["composite"] = composite
        out["drone_mask"] = drone_mask
        out["cx"] = cx
        out["cy"] = cy
        out["actual_px"] = actual_px

        # ── Step 4: 注入标记点到 depth/seg（← 这才是关键） ──
        print("Step 4: inject drone silhouette → depth/seg ...")
        drone_depth, drone_seg = self.inject_into_conditions(
            bg_depth, bg_seg, drone_mask, dval, class_id)
        out["drone_depth"] = drone_depth
        out["drone_seg"] = drone_seg
        print(f"  depth range: {drone_depth.min():.3f}~{drone_depth.max():.3f}, "
              f"seg classes: {np.unique(drone_seg)}")

        # ── Step 5: ControlNet 重绘（v7.2: CLIP 动态 prompt） ──
        if skip_cnet:
            print("Step 5: SKIPPED (skip_cnet=True), using composite as result")
            out["result"] = composite
        else:
            print(f"Step 5: ControlNet(depth+seg) repaint "
                  f"(strength={blend_strength}, scales={cnet_scales}, "
                  f"{time_of_day}/{weather}, lora={use_lora}) ...")
            t0 = time.time()
            result = self._controlnet_repaint(
                background_bgr, composite, drone_depth, drone_seg, drone_mask,
                cx, cy, actual_px,
                blend_seed, blend_steps, blend_strength, guidance,
                crop_pad_ratio, feather_px, cnet_scales,
                time_of_day=time_of_day, weather=weather,
                use_lora=use_lora)
            out["result"] = result
            print(f"  done ({time.time() - t0:.1f}s)")

        # ── 中间产物 ──
        if intermediates_dir:
            os.makedirs(intermediates_dir, exist_ok=True)
            drone_full.save(os.path.join(intermediates_dir, "drone_raw.png"))
            drone_rgba.save(os.path.join(intermediates_dir, "drone_rgba.png"))
            cv2.imwrite(os.path.join(intermediates_dir, "composite.png"), composite)
            drone_depth_img = (1.0 - (drone_depth - drone_depth.min()) /
                               (drone_depth.max() - drone_depth.min() + 1e-8))
            drone_depth_uint8 = (drone_depth_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(intermediates_dir, "drone_depth_gray.png"),
                        drone_depth_uint8)

        return out

    def reposition(self, background_bgr, bg_depth, bg_seg, drone_rgba,
                   nu=0.5, nv=0.5, target_px=70, dval=0.4, class_id=7,
                   blend_seed=123, blend_steps=15,
                   blend_strength=0.65, guidance=7.5,
                   crop_pad_ratio=2.5, feather_px=12,
                   cnet_scales=(0.85, 0.85),
                   time_of_day="afternoon", weather="clear",
                   use_lora=True, skip_cnet=False):
        """
        快速重定位：跳过 LoRA 生成 + rembg，直接用已有 drone_rgba 做合成+注入+重绘。

        参数:
            drone_rgba: PIL Image (RGBA), 已抠图的无人机 RGBA（来自首次 inpaint 的 out["drone_rgba"]）
            其余参数同 inpaint() 的对应阶段。

        返回:
            dict with: result, drone_depth, drone_seg, composite,
                       drone_mask, cx, cy, actual_px
        """
        if not self._loaded:
            self.load()

        out = {}

        # ── 确认 drone_rgba 是 PIL Image ──
        if isinstance(drone_rgba, np.ndarray):
            drone_rgba = Image.fromarray(drone_rgba)

        # ── 从 drone_rgba 提取 bbox（用于 _composite） ──
        arr = np.array(drone_rgba)
        alpha = arr[..., 3]
        ys, xs = np.where(alpha > 64)
        if len(xs) == 0:
            return {"result": background_bgr, **out}
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # ── Step 3: 合成 ──
        print("Step 3: composite + mask (reposition) ...")
        composite, cx, cy, actual_px, drone_mask, drone_rgb_patch, nw, nh, px1, py1 = \
            self._composite(background_bgr, drone_rgba, bbox, nu, nv, target_px)
        out["composite"] = composite
        out["drone_mask"] = drone_mask
        out["cx"] = cx
        out["cy"] = cy
        out["actual_px"] = actual_px

        # ── Step 4: 注入标记点 ──
        print("Step 4: inject drone silhouette → depth/seg (reposition) ...")
        drone_depth, drone_seg = self.inject_into_conditions(
            bg_depth, bg_seg, drone_mask, dval, class_id)
        out["drone_depth"] = drone_depth
        out["drone_seg"] = drone_seg

        # ── Step 5: ControlNet 重绘或跳过 ──
        if skip_cnet:
            print("Step 5: SKIPPED, using composite as result")
            out["result"] = composite
        else:
            print(f"Step 5: ControlNet repaint (strength={blend_strength}) ...")
            t0 = time.time()
            result = self._controlnet_repaint(
                background_bgr, composite, drone_depth, drone_seg, drone_mask,
                cx, cy, actual_px,
                blend_seed, blend_steps, blend_strength, guidance,
                crop_pad_ratio, feather_px, cnet_scales,
                time_of_day=time_of_day, weather=weather,
                use_lora=use_lora)
            out["result"] = result
            print(f"  done ({time.time() - t0:.1f}s)")

        return out

    def unload(self):
        if self.pipe_txt:
            del self.pipe_txt
        if self.pipe_img:
            del self.pipe_img
        if self.pipe_cnet:
            del self.pipe_cnet
        if self.pipe_cnet_nolora:
            del self.pipe_cnet_nolora
        if self.cn_depth:
            del self.cn_depth
        if self.cn_seg:
            del self.cn_seg
        self.pipe_txt = self.pipe_img = self.pipe_cnet = self.pipe_cnet_nolora = None
        self.cn_depth = self.cn_seg = None
        torch.cuda.empty_cache()
        self._loaded = False

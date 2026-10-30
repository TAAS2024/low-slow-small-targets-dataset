"""
Agent 5 v2: 三阶段 LoRA 局部重绘管线
=====================================
正确方案：LoRA 大图生成 → 缩放插入 → 局部 img2img 融合

问题: 当 drone 区域太小 (~40px) 时，LoRA 无法在 ControlNet txt2img 中正常激活。
解决:
  1. 先用 LoRA 在 512×512 干净画布上生成无人机（LoRA 需要足够大的画布才能触发）
  2. 绿幕抠图 → 等比例缩放 → 贴入背景
  3. 在插入位置附近裁剪 → img2img 少量步数重绘 → 羽化贴回保证融合度
"""

import os
import sys
import time

# ── Monkey-patch: diffusers 0.32.x + transformers 4.57+ 兼容 ──
import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ── 项目根 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "0-model"
LORA_DIR = PROJECT_ROOT / "2-Lora training" / "drone_target_v2"

SD15_PATH = MODEL_DIR / "stable-diffusion-v1-5"
LORA_PATH = LORA_DIR / "drn3_uav_lora_v2.safetensors"

# ── 提示词 ──────────────────────────────────────────────
DRONE_GEN_PROMPT = (
    "drn3_uav, a commercial quadcopter drone, solid green background, "
    "front view, sharp focus, professional aerial photography, 8k, highly detailed"
)
DRONE_GEN_NEGATIVE = (
    "blurry, distorted, bad anatomy, watermark, text, people, birds, "
    "airplane, building, cloud, sky, low quality, worst quality, shadow"
)

BLEND_PROMPT = (
    "drn3_uav, a commercial quadcopter drone flying in clear blue sky, "
    "front view, daytime, sharp focus, professional aerial photography, 8k, highly detailed"
)
BLEND_NEGATIVE = (
    "blurry, distorted, bad anatomy, watermark, text, "
    "people, birds, airplane, low quality, worst quality"
)


class LoraInpainterV2:
    """三阶段 LoRA 局部重绘器"""

    def __init__(self, dtype=torch.float16, device="cuda"):
        self.device = device
        self.dtype = dtype
        self.pipe_txt = None       # txt2img (用于生成无人机)
        self.pipe_img = None       # img2img (用于局部融合)
        self._loaded = False

    # ══════════════════════════════════════════════════════
    # 加载模型
    # ══════════════════════════════════════════════════════

    def load(self):
        if self._loaded:
            return

        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        print("⏳ 加载 SD1.5 ...")
        self.pipe_txt = StableDiffusionPipeline.from_pretrained(
            str(SD15_PATH),
            torch_dtype=self.dtype,
            safety_checker=None,
            local_files_only=True,
        )

        # 合并 LoRA 到 UNet
        self._merge_lora()

        # 基于共享组件创建 img2img pipeline
        print("⏳ 创建 img2img pipeline (共享 UNet) ...")
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

        self.pipe_txt.to(self.device)
        self.pipe_txt.enable_attention_slicing()

        # 预热
        print("⏳ 预热推理 ...")
        dummy = Image.new("RGB", (512, 512), color=(128, 128, 128))
        _ = self.pipe_txt(prompt="warmup", num_inference_steps=1, output_type="pil")
        _ = self.pipe_img(prompt="warmup", image=dummy, strength=0.5,
                          num_inference_steps=2, output_type="pil")

        self._loaded = True
        print("✅ LoraInpainterV2 就绪")

    def _merge_lora(self):
        """手动将 sd-scripts LoRA 合并到 UNet。"""
        from safetensors.torch import load_file

        print(f"⏳ 合并 LoRA: {LORA_PATH.name} ...")
        state_dict = load_file(str(LORA_PATH))
        unet = self.pipe_txt.unet

        # 提取 rank + alpha
        lora_info = {}
        for k in state_dict:
            if k.endswith(".alpha"):
                module_name = k[:-6]
                lora_info[module_name] = {"alpha": state_dict[k].item()}

        for k, v in state_dict.items():
            if k.endswith(".lora_down.weight"):
                module_name = k[:-18]
                if module_name in lora_info:
                    lora_info[module_name]["rank"] = v.shape[0]

        # 计算每个模块的 delta
        module_deltas = {}
        for module_name in lora_info:
            if not module_name.startswith("lora_unet_"):
                continue
            down_key = f"{module_name}.lora_down.weight"
            up_key = f"{module_name}.lora_up.weight"
            if down_key not in state_dict or up_key not in state_dict:
                continue

            A = state_dict[down_key]
            B = state_dict[up_key]
            alpha = lora_info[module_name].get("alpha", 1.0)
            rank = lora_info[module_name].get("rank", A.shape[0])
            scale = alpha / rank

            A_2d = A.squeeze() if A.dim() > 2 else A
            B_2d = B.squeeze() if B.dim() > 2 else B
            delta_2d = scale * (B_2d @ A_2d)

            module_deltas[module_name] = delta_2d

        # 合并到 UNet 参数
        merged = 0
        for param_name, param in unet.named_parameters():
            base_name = param_name.replace('.weight', '').replace('.bias', '')
            if base_name == param_name:
                continue
            sd_key = 'lora_unet_' + '_'.join(base_name.split('.'))
            if sd_key not in module_deltas:
                continue

            delta = module_deltas[sd_key].to(param.device, param.dtype)
            if delta.shape == param.data.shape:
                param.data += delta
                merged += 1
            elif param.data.dim() == 4 and delta.dim() == 2:
                param.data += delta.reshape(param.data.shape)
                merged += 1
            elif delta.T.shape == param.data.shape:
                param.data += delta.T
                merged += 1

        print(f"   ✅ 合并 {merged}/{len(module_deltas)} UNet LoRA 权重")
        if merged == 0:
            print("   ⚠️ 未成功合并任何权重！")

    # ══════════════════════════════════════════════════════
    # 核心管线
    # ══════════════════════════════════════════════════════

    def inpaint(
        self,
        background: np.ndarray,      # (H,W,3) BGR
        nu: float = 0.5,             # 归一化 x 位置
        nv: float = 0.5,             # 归一化 y 位置
        target_px: int = 40,         # 目标无人机像素大小
        gen_prompt: str = DRONE_GEN_PROMPT,
        blend_prompt: str = BLEND_PROMPT,
        gen_seed: int = 42,
        blend_seed: int = 123,
        gen_steps: int = 30,
        blend_steps: int = 12,
        blend_strength: float = 0.45,
        guidance_scale: float = 7.5,
        intermediates_dir: str = None,  # 保存中间产物的目录
    ) -> dict:
        """
        完整三阶段管线。

        Returns:
            dict: {"result": BGR ndarray, "drone_full": PIL, "drone_rgba": PIL,
                   "composite": BGR ndarray, "cx": int, "cy": int}
        """
        if not self._loaded:
            self.load()

        intermediates = {}

        # ── Step 1: LoRA 大图生成 ──
        print("🎨 Step 1: LoRA 大图生成无人机 ...")
        t0 = time.time()
        drone_full = self._generate_drone(gen_prompt, gen_seed, gen_steps, guidance_scale)
        intermediates["drone_full"] = drone_full
        print(f"   ✅ 生成完成 ({time.time() - t0:.1f}s)")

        # ── Step 2: 绿幕抠图 + 缩放 + 贴入 ──
        print("✂️  Step 2: 绿幕抠图 + 缩放插入 ...")
        drone_rgba, drone_bbox = self._extract_drone(drone_full)
        intermediates["drone_rgba"] = drone_rgba
        intermediates["drone_bbox"] = drone_bbox

        if drone_rgba is None:
            print("   ⚠️ 抠图失败，使用完整生成图")
            return {"result": background, **intermediates}

        composite, cx, cy, actual_px = self._scale_and_paste(
            drone_rgba, drone_bbox, background, nu, nv, target_px
        )
        intermediates["composite"] = composite
        intermediates["cx"] = cx
        intermediates["cy"] = cy
        print(f"   ✅ 缩放: {drone_bbox[2]-drone_bbox[0]}px → {actual_px}px, 中心=({cx},{cy})")

        # ── Step 3: 局部 img2img 融合 ──
        print("🖌️  Step 3: 局部 img2img 融合 ...")
        t0 = time.time()
        result = self._local_blend(
            background, composite, cx, cy, actual_px,
            blend_prompt, blend_seed, blend_steps, blend_strength, guidance_scale,
        )
        intermediates["result"] = result
        print(f"   ✅ 融合完成 ({time.time() - t0:.1f}s)")

        # 保存中间产物
        if intermediates_dir:
            import os
            os.makedirs(intermediates_dir, exist_ok=True)
            if drone_full is not None:
                drone_full.save(os.path.join(intermediates_dir, "drone_full_generated.png"))
            if drone_rgba is not None:
                drone_rgba.save(os.path.join(intermediates_dir, "drone_extracted_rgba.png"))
            if composite is not None:
                cv2.imwrite(os.path.join(intermediates_dir, "composite_before_blend.png"), composite)

        return intermediates

    # ── Step 1 实现 ──────────────────────────────────────

    def _generate_drone(self, prompt, seed, steps, guidance):
        """在 512×512 纯色背景上生成无人机。"""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe_txt(
            prompt=prompt,
            negative_prompt=DRONE_GEN_NEGATIVE,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        )
        return result.images[0]

    # ── Step 2 实现 ──────────────────────────────────────

    def _extract_drone(self, img_pil: Image.Image):
        """
        绿幕抠图：在 HSV 空间提取绿色背景，反转得到无人机 mask。

        Returns:
            (drone_rgba: PIL Image, bbox: (x1,y1,x2,y2)) 或 (None, None)
        """
        img = np.array(img_pil)                    # RGB
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 绿色范围 (OpenCV HSV: H∈[35,85])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 反转 → drone mask
        drone_mask = cv2.bitwise_not(green_mask)

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        drone_mask = cv2.morphologyEx(drone_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        drone_mask = cv2.morphologyEx(drone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找到最大连通区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(drone_mask, connectivity=8)
        if num_labels <= 1:
            return None, None

        # 排除背景(0)，找最大面积
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        drone_mask_clean = (labels == largest_idx).astype(np.uint8) * 255

        # 轻微羽化
        drone_mask_clean = cv2.GaussianBlur(drone_mask_clean, (3, 3), 1)

        # 获取 bbox
        ys, xs = np.where(drone_mask_clean > 128)
        if len(xs) == 0:
            return None, None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        # 构建 RGBA
        rgba = np.dstack([img, drone_mask_clean])
        return Image.fromarray(rgba, "RGBA"), (x1, y1, x2, y2)

    def _scale_and_paste(
        self,
        drone_rgba: Image.Image,
        drone_bbox: tuple,
        background: np.ndarray,
        nu: float, nv: float,
        target_px: int,
    ):
        """缩放无人机并 alpha 合成到背景上。"""
        bg_h, bg_w = background.shape[:2]
        x1, y1, x2, y2 = drone_bbox
        drone_w, drone_h = x2 - x1, y2 - y1
        drone_size = max(drone_w, drone_h)

        # 计算缩放因子
        if drone_size <= 0:
            drone_size = 200
        scale = target_px / drone_size

        # 裁剪 drone bbox + 缩放
        drone_cropped = drone_rgba.crop((x1, y1, x2, y2))
        new_w = max(1, int(drone_w * scale))
        new_h = max(1, int(drone_h * scale))
        drone_scaled = drone_cropped.resize((new_w, new_h), Image.LANCZOS)

        # 粘贴中心
        cx = int(nu * bg_w)
        cy = int(nv * bg_h)
        px1 = cx - new_w // 2
        py1 = cy - new_h // 2
        px2 = px1 + new_w
        py2 = py1 + new_h

        # Alpha 合成
        bg_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        composite = bg_pil.copy()

        # 裁剪到画布内
        src_x1 = max(0, -px1)
        src_y1 = max(0, -py1)
        src_x2 = new_w - max(0, px2 - bg_w)
        src_y2 = new_h - max(0, py2 - bg_h)
        dst_x1 = max(0, px1)
        dst_y1 = max(0, py1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            drone_patch = drone_scaled.crop((src_x1, src_y1, src_x2, src_y2))
            composite.paste(drone_patch, (dst_x1, dst_y1), drone_patch)

        composite_bgr = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
        return composite_bgr, cx, cy, target_px

    # ── Step 3 实现 ──────────────────────────────────────

    def _local_blend(
        self,
        background: np.ndarray,
        composite: np.ndarray,
        cx: int, cy: int,
        drone_px: int,
        prompt: str,
        seed: int,
        steps: int,
        strength: float,
        guidance: float,
    ):
        """
        局部 img2img 融合：裁剪 drone 周围区域 → img2img 低强度重绘 → 羽化贴回。
        """
        H, W = background.shape[:2]

        # 裁剪区域：drone 大小的 3 倍 padding
        pad = max(int(drone_px * 2.0), 64)
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(W, cx + pad)
        y2 = min(H, cy + pad)

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 32 or crop_h < 32:
            # 区域太小，直接返回 composite
            return composite

        # 提取 crop → resize 到 512×512
        crop = composite[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb).resize((512, 512), Image.LANCZOS)

        # Img2Img 重绘
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe_img(
            prompt=prompt,
            negative_prompt=BLEND_NEGATIVE,
            image=crop_pil,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        )

        blended_full = result.images[0]  # 512×512
        blended_crop = np.array(blended_full.resize((crop_w, crop_h), Image.LANCZOS))
        blended_crop_bgr = cv2.cvtColor(blended_crop, cv2.COLOR_RGB2BGR)

        # 羽化 mask：中心区域权重 1，边缘渐变到 0
        mask = np.ones((crop_h, crop_w), dtype=np.float32)
        feather = pad // 3
        if feather > 4:
            # 内缩 feather px 区域渐变
            mask[:feather, :] = np.linspace(0, 1, feather)[:, np.newaxis]
            mask[-feather:, :] = np.linspace(1, 0, feather)[:, np.newaxis]
            mask[:, :feather] *= np.linspace(0, 1, feather)[np.newaxis, :]
            mask[:, -feather:] *= np.linspace(1, 0, feather)[np.newaxis, :]

        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended_final = (mask_3ch * blended_crop_bgr.astype(float) +
                         (1 - mask_3ch) * crop.astype(float)).astype(np.uint8)

        # 贴回
        result_img = composite.copy()
        result_img[y1:y2, x1:x2] = blended_final
        return result_img

    # ══════════════════════════════════════════════════════
    # 卸载
    # ══════════════════════════════════════════════════════

    def unload(self):
        if self.pipe_txt is not None:
            del self.pipe_txt
            self.pipe_txt = None
        if self.pipe_img is not None:
            del self.pipe_img
            self.pipe_img = None
        torch.cuda.empty_cache()
        self._loaded = False

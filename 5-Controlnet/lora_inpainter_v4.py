"""
Agent 5 v4: 直方图匹配 + 极低 blend
====================================
v3 的问题: LoRA 训练数据是摄影棚光照，prompt 强行改光照无效。
v4 方案:
  1. LoRA 只管画对角度和机型 (front/bottom view)
  2. 绿幕抠图后，直方图匹配到背景天空色调
  3. img2img blend 降到极低 (0.10-0.15)，只做边缘融合
"""

import os, sys, time
import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

import cv2, numpy as np, torch
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "0-model"
LORA_DIR = PROJECT_ROOT / "2-Lora training" / "drone_target_v2"
SD15_PATH = MODEL_DIR / "stable-diffusion-v1-5"
LORA_PATH = LORA_DIR / "drn3_uav_lora_v2.safetensors"

DRONE_GEN_PROMPT = (
    "drn3_uav, a commercial quadcopter drone, solid green background, "
    "front view, sharp focus, 8k, highly detailed"
)
DRONE_GEN_NEGATIVE = (
    "blurry, distorted, bad anatomy, watermark, text, people, birds, "
    "airplane, building, cloud, sky, low quality, worst quality, shadow"
)


class LoraInpainterV4:
    def __init__(self, dtype=torch.float16, device="cuda"):
        self.device = device
        self.dtype = dtype
        self.pipe_txt = None
        self.pipe_img = None
        self._loaded = False

    def load(self):
        if self._loaded: return
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        print("⏳ 加载 SD1.5 + LoRA ...")
        self.pipe_txt = StableDiffusionPipeline.from_pretrained(
            str(SD15_PATH), torch_dtype=self.dtype,
            safety_checker=None, local_files_only=True)
        self._merge_lora()
        self.pipe_img = StableDiffusionImg2ImgPipeline(
            vae=self.pipe_txt.vae, text_encoder=self.pipe_txt.text_encoder,
            tokenizer=self.pipe_txt.tokenizer, unet=self.pipe_txt.unet,
            scheduler=self.pipe_txt.scheduler, safety_checker=None,
            feature_extractor=getattr(self.pipe_txt, 'feature_extractor', None),
            requires_safety_checker=False)
        self.pipe_txt.to(self.device)
        self.pipe_txt.enable_attention_slicing()
        dummy = Image.new("RGB", (512, 512), color=(128, 128, 128))
        _ = self.pipe_txt(prompt="warmup", num_inference_steps=1, output_type="pil")
        _ = self.pipe_img(prompt="warmup", image=dummy, strength=0.5,
                          num_inference_steps=2, output_type="pil")
        self._loaded = True
        print("✅ LoraInpainterV4 就绪")

    def _merge_lora(self):
        from safetensors.torch import load_file
        print(f"⏳ 合并 LoRA: {LORA_PATH.name} ...")
        state_dict = load_file(str(LORA_PATH))
        unet = self.pipe_txt.unet
        lora_info = {}
        for k in state_dict:
            if k.endswith(".alpha"):
                lora_info[k[:-6]] = {"alpha": state_dict[k].item()}
        for k, v in state_dict.items():
            if k.endswith(".lora_down.weight"):
                mn = k[:-18]
                if mn in lora_info:
                    lora_info[mn]["rank"] = v.shape[0]
        module_deltas = {}
        for module_name in lora_info:
            if not module_name.startswith("lora_unet_"): continue
            dk = f"{module_name}.lora_down.weight"
            uk = f"{module_name}.lora_up.weight"
            if dk not in state_dict or uk not in state_dict: continue
            A, B = state_dict[dk], state_dict[uk]
            alpha = lora_info[module_name].get("alpha", 1.0)
            rank = lora_info[module_name].get("rank", A.shape[0])
            A2, B2 = A.squeeze() if A.dim()>2 else A, B.squeeze() if B.dim()>2 else B
            module_deltas[module_name] = (alpha/rank) * (B2 @ A2)
        merged = 0
        for pn, param in unet.named_parameters():
            bn = pn.replace('.weight','').replace('.bias','')
            if bn == pn: continue
            sk = 'lora_unet_' + '_'.join(bn.split('.'))
            if sk not in module_deltas: continue
            delta = module_deltas[sk].to(param.device, param.dtype)
            if delta.shape == param.data.shape:
                param.data += delta; merged += 1
            elif param.data.dim()==4 and delta.dim()==2:
                param.data += delta.reshape(param.data.shape); merged += 1
            elif delta.T.shape == param.data.shape:
                param.data += delta.T; merged += 1
        print(f"   ✅ 合并 {merged}/{len(module_deltas)} LoRA 权重")

    # ═══════════════════ 主线 ═══════════════════
    def inpaint(self, background, nu=0.5, nv=0.5, target_px=60,
                gen_seed=42, blend_seed=123, gen_steps=30,
                blend_steps=8, blend_strength=0.12, guidance=7.5,
                crop_pad_ratio=2.0, feather_ratio=0.5,
                drone_prompt=None, drone_neg=None,
                blend_prompt=None, blend_neg=None,
                intermediates_dir=None):
        if not self._loaded: self.load()
        im = {}

        # Step 1: LoRA 生成
        print("🎨 Step 1: LoRA 生成 ...")
        t0 = time.time()
        drone_full = self._generate_drone(gen_seed, gen_steps, guidance,
                                          drone_prompt, drone_neg)
        im["drone_full"] = drone_full
        print(f"   ✅ ({time.time()-t0:.1f}s)")

        # Step 2: 抠图 + 贴入
        print("✂️  Step 2: 抠图 + 贴入 ...")
        drone_rgba, drone_bbox = self._extract_drone(drone_full)
        if drone_rgba is None:
            return {"result": background, **im}
        composite, cx, cy, actual_px = self._scale_and_paste(
            drone_rgba, drone_bbox, background, nu, nv, target_px)
        im["composite"] = composite
        print(f"   ✅ {drone_bbox[2]-drone_bbox[0]}px → {actual_px}px, ({cx},{cy})")

        # Step 3: 直方图匹配 + 极低 blend
        print(f"🖌️  Step 3: 直方图匹配 + blend (str={blend_strength}) ...")
        t0 = time.time()
        result = self._histo_blend(
            background, composite, cx, cy, actual_px,
            blend_seed, blend_steps, blend_strength, guidance,
            crop_pad_ratio, feather_ratio,
            blend_prompt, blend_neg)
        im["result"] = result
        print(f"   ✅ ({time.time()-t0:.1f}s)")

        if intermediates_dir:
            os.makedirs(intermediates_dir, exist_ok=True)
            drone_full.save(os.path.join(intermediates_dir, "drone_full.png"))
            drone_rgba.save(os.path.join(intermediates_dir, "drone_rgba.png"))
            cv2.imwrite(os.path.join(intermediates_dir, "composite.png"), composite)
            cv2.imwrite(os.path.join(intermediates_dir, "result.png"), result)
        return im

    # ═══════════════════ 子步骤 ═══════════════════
    def _generate_drone(self, seed, steps, guidance, prompt, neg):
        g = torch.Generator(device=self.device).manual_seed(seed)
        r = self.pipe_txt(
            prompt=prompt or DRONE_GEN_PROMPT,
            negative_prompt=neg or DRONE_GEN_NEGATIVE,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=g, output_type="pil")
        return r.images[0]

    def _extract_drone(self, img_pil):
        """v4 改进: 更宽 HSV 范围 + 更强形态学"""
        img = np.array(img_pil)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 更宽的绿色范围（容忍不均匀绿幕）
        green = cv2.inRange(hsv, (30, 35, 35), (90, 255, 255))
        drone = cv2.bitwise_not(green)

        # 更强形态学：开运算去噪 → 闭运算填洞
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        drone = cv2.morphologyEx(drone, cv2.MORPH_OPEN, k5, iterations=1)
        drone = cv2.morphologyEx(drone, cv2.MORPH_CLOSE, k7, iterations=2)

        # 取最大连通域
        n, labels, stats, _ = cv2.connectedComponentsWithStats(drone, 8)
        if n <= 1: return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        drone_clean = (labels == (np.argmax(areas) + 1)).astype(np.uint8) * 255

        # 边缘羽化（更宽的 blur 半径）
        drone_clean = cv2.GaussianBlur(drone_clean, (7, 7), 3)

        ys, xs = np.where(drone_clean > 64)
        if len(xs) == 0: return None, None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        return Image.fromarray(np.dstack([img, drone_clean]), "RGBA"), (x1, y1, x2, y2)

    def _scale_and_paste(self, drone_rgba, bbox, bg, nu, nv, target_px):
        H, W = bg.shape[:2]
        x1, y1, x2, y2 = bbox
        dw, dh = x2 - x1, y2 - y1
        ds = max(dw, dh) or 200
        scale = target_px / ds
        drone_crop = drone_rgba.crop((x1, y1, x2, y2))
        nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))
        drone_s = drone_crop.resize((nw, nh), Image.LANCZOS)
        cx, cy = int(nu * W), int(nv * H)
        px1, py1 = cx - nw // 2, cy - nh // 2
        px2, py2 = px1 + nw, py1 + nh
        comp = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).copy()
        sx1, sy1 = max(0, -px1), max(0, -py1)
        sx2, sy2 = nw - max(0, px2 - W), nh - max(0, py2 - H)
        dx1, dy1 = max(0, px1), max(0, py1)
        if sx2 > sx1 and sy2 > sy1:
            patch = drone_s.crop((sx1, sy1, sx2, sy2))
            comp.paste(patch, (dx1, dy1), patch)
        return cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2BGR), cx, cy, target_px

    def _histo_match(self, src, ref, mask=None):
        """
        将 src 的直方图匹配到 ref。
        如果提供 mask，只对 mask 区域做匹配。
        返回匹配后的图像。
        """
        result = src.copy().astype(np.float32)
        for c in range(3):
            s = src[:, :, c]
            r = ref[:, :, c]
            if mask is not None:
                s_vals = s[mask > 0].ravel()
                r_vals = r.ravel()
            else:
                s_vals = s.ravel()
                r_vals = r.ravel()

            # 计算 CDF
            s_hist, s_bins = np.histogram(s_vals, bins=256, range=(0, 256))
            r_hist, r_bins = np.histogram(r_vals, bins=256, range=(0, 256))
            s_cdf = np.cumsum(s_hist).astype(np.float64) / s_hist.sum()
            r_cdf = np.cumsum(r_hist).astype(np.float64) / r_hist.sum()

            # 映射表
            lookup = np.interp(s_cdf, r_cdf, np.arange(256))
            result[:, :, c] = lookup[np.clip(s.astype(int), 0, 255)]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _histo_blend(self, background, composite, cx, cy, drone_px,
                     seed, steps, strength, guidance,
                     pad_ratio=2.0, feather_ratio=0.5,
                     blend_prompt=None, blend_neg=None):
        """
        直方图匹配 + 极低 blend 方案。

        1. Crop 包含无人机的区域
        2. 直方图匹配：将 crop 中无人机区域的颜色拉到背景
        3. img2img (极低 strength) 只做边缘融合
        4. 羽化 mask 贴回
        """
        H, W = background.shape[:2]
        pad = max(int(drone_px * pad_ratio), 32)
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(W, cx + pad), min(H, cy + pad)
        cw, ch = x2 - x1, y2 - y1
        if cw < 16 or ch < 16:
            return composite

        crop_bg = background[y1:y2, x1:x2]       # 原始背景 crop
        crop_comp = composite[y1:y2, x1:x2]      # 贴了无人机的 crop

        # ── 构建无人机 mask ──
        # 无人机 = composite 和 background 不同的像素
        diff = np.abs(crop_comp.astype(float) - crop_bg.astype(float)).max(axis=2)
        drone_mask = (diff > 8).astype(np.uint8) * 255
        # 膨胀一点确保覆盖边缘
        drone_mask = cv2.dilate(drone_mask, np.ones((3, 3), np.uint8), iterations=1)
        drone_mask_f = drone_mask.astype(np.float32) / 255.0

        # ── 直方图匹配：无人机 → 背景色调 ──
        drone_pixels_mask = drone_mask_f > 0.3
        if drone_pixels_mask.sum() > 50:
            crop_matched = self._histo_match(crop_comp, crop_bg, mask=drone_pixels_mask)
        else:
            crop_matched = crop_comp

        # ── 如果需要 img2img（边缘融合） ──
        if strength > 0.01 and blend_prompt:
            crop_rgb = cv2.cvtColor(crop_matched, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb).resize((512, 512), Image.LANCZOS)
            g = torch.Generator(device=self.device).manual_seed(seed)
            r = self.pipe_img(
                prompt=blend_prompt, negative_prompt=blend_neg or "",
                image=crop_pil, strength=strength,
                num_inference_steps=steps, guidance_scale=guidance,
                generator=g, output_type="pil")
            blended_crop = np.array(r.images[0].resize((cw, ch), Image.LANCZOS))
            blended_bgr = cv2.cvtColor(blended_crop, cv2.COLOR_RGB2BGR)
        else:
            blended_bgr = crop_matched

        # ── 羽化 mask ──
        mask = np.ones((ch, cw), dtype=np.float32)
        feather_px = max(2, int(min(cw, ch) * feather_ratio))
        ramp = np.linspace(0, 1, feather_px)
        mask[:feather_px, :] *= ramp[:, np.newaxis]
        mask[-feather_px:, :] *= ramp[::-1, np.newaxis]
        mask[:, :feather_px] *= ramp[np.newaxis, :]
        mask[:, -feather_px:] *= ramp[::-1][np.newaxis, :]
        mask_3ch = np.stack([mask] * 3, axis=-1)

        blended_final = (mask_3ch * blended_bgr.astype(float) +
                         (1 - mask_3ch) * crop_bg.astype(float)).astype(np.uint8)

        # ── 用原始 background 重建结果（不是 composite） ──
        result = background.copy()
        result[y1:y2, x1:x2] = blended_final
        return result

    def unload(self):
        if self.pipe_txt: del self.pipe_txt
        if self.pipe_img: del self.pipe_img
        self.pipe_txt = self.pipe_img = None
        torch.cuda.empty_cache()
        self._loaded = False

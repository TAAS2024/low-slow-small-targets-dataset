"""
Agent 5 v6: v3 LoRA (512²) + 白底 prompt + rembg 抠图 + Crop-Scale-Blend
======================================================================
与 v5 代码逻辑完全一致，唯一改动：
  - LoRA 路径 → v3 Pocket_UAV (step2000, rank=8, 768² 训练)
  - prompt 加 "clean white background" 引导白底生成
  - 生成仍是 512×512（管线后续缩放 + img2img，768² 没必要）

流程:
  1. LoRA 生成无人机 (512×512, 白底 prompt → rembg 抠前景)
  2. 缩放到 target_px → alpha 合成到背景
  3. crop drone 区域 → resize 512 → img2img → blend 回贴
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


class LoraInpainterV6:

    def __init__(self, dtype=torch.float16, device="cuda"):
        self.device = device
        self.dtype = dtype
        self.pipe_txt = None
        self.pipe_img = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        print(f"⏳ 加载 SD1.5 + v3 LoRA (step2000) ...")
        print(f"   LoRA: {LORA_PATH.name}")
        self.pipe_txt = StableDiffusionPipeline.from_pretrained(
            str(SD15_PATH), torch_dtype=self.dtype,
            safety_checker=None, local_files_only=True,
        )
        self._merge_lora()
        self.pipe_img = StableDiffusionImg2ImgPipeline(
            vae=self.pipe_txt.vae, text_encoder=self.pipe_txt.text_encoder,
            tokenizer=self.pipe_txt.tokenizer, unet=self.pipe_txt.unet,
            scheduler=self.pipe_txt.scheduler, safety_checker=None,
            feature_extractor=getattr(self.pipe_txt, 'feature_extractor', None),
            requires_safety_checker=False,
        )
        self.pipe_txt.to(self.device)
        self.pipe_txt.enable_attention_slicing()
        # warmup
        dummy = Image.new("RGB", (512, 512), color=(128, 128, 128))
        _ = self.pipe_txt(prompt="warmup", num_inference_steps=1, output_type="pil")
        _ = self.pipe_img(prompt="warmup", image=dummy, strength=0.5,
                          num_inference_steps=2, output_type="pil")
        self._loaded = True
        print(f"✅ LoraInpainterV6 就绪")

    def _merge_lora(self):
        from safetensors.torch import load_file
        print(f"⏳ 合并 LoRA 权重 ...")
        state_dict = load_file(str(LORA_PATH))
        unet = self.pipe_txt.unet
        # Parse LoRA metadata
        lora_info = {}
        for k in state_dict:
            if k.endswith(".alpha"):
                lora_info[k[:-6]] = {"alpha": state_dict[k].item()}
        for k, v in state_dict.items():
            if k.endswith(".lora_down.weight"):
                mn = k[:-18]
                if mn in lora_info:
                    lora_info[mn]["rank"] = v.shape[0]
        # Compute and merge
        module_deltas = {}
        for module_name in lora_info:
            if not module_name.startswith("lora_unet_"):
                continue
            dk = f"{module_name}.lora_down.weight"
            uk = f"{module_name}.lora_up.weight"
            if dk not in state_dict or uk not in state_dict:
                continue
            A, B = state_dict[dk], state_dict[uk]
            alpha = lora_info[module_name].get("alpha", 1.0)
            rank = lora_info[module_name].get("rank", A.shape[0])
            A2 = A.squeeze() if A.dim() > 2 else A
            B2 = B.squeeze() if B.dim() > 2 else B
            module_deltas[module_name] = (alpha / rank) * (B2 @ A2)
        merged = 0
        for pn, param in unet.named_parameters():
            bn = pn.replace('.weight', '').replace('.bias', '')
            if bn == pn:
                continue
            sk = 'lora_unet_' + '_'.join(bn.split('.'))
            if sk not in module_deltas:
                continue
            delta = module_deltas[sk].to(param.device, param.dtype)
            if delta.shape == param.data.shape:
                param.data += delta
                merged += 1
            elif param.data.dim() == 4 and delta.dim() == 2:
                param.data += delta.reshape(param.data.shape)
                merged += 1
            elif delta.T.shape == param.data.shape:
                param.data += delta.T
                merged += 1
        print(f"   ✅ 合并 {merged}/{len(module_deltas)} LoRA 权重")

    # ══════════════════════════════════════════════════
    # 抠图：rembg
    # ══════════════════════════════════════════════════
    def _extract_drone_rembg(self, img_pil):
        """rembg 前景分割 → RGBA + bbox"""
        rgba = remove(img_pil)
        arr = np.array(rgba)
        alpha = arr[..., 3]
        ys, xs = np.where(alpha > 64)
        if len(xs) == 0:
            return None, None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        return rgba, (x1, y1, x2, y2)

    # ══════════════════════════════════════════════════
    # 颜色匹配
    # ══════════════════════════════════════════════════
    def _match_lighting(self, composite_crop, bg_crop, drone_radius, drone_cx, drone_cy):
        h, w = composite_crop.shape[:2]
        ygrid, xgrid = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xgrid - drone_cx)**2 + (ygrid - drone_cy)**2)
        drone_mask = np.clip(1.0 - dist / max(drone_radius * 1.2, 1), 0, 1)
        bg_float = bg_crop.astype(np.float32)
        sky_mask = 1.0 - drone_mask
        sky_weights = sky_mask / (sky_mask.sum() + 1e-8)
        sky_mean = np.average(bg_float, axis=(0, 1), weights=sky_weights)
        comp_float = composite_crop.astype(np.float32)
        drone_weights = drone_mask / (drone_mask.sum() + 1e-8)
        drone_mean = np.average(comp_float, axis=(0, 1), weights=drone_weights)
        sky_lum = sky_mean.mean()
        drone_lum = drone_mean.mean()
        if drone_lum < 5:
            return composite_crop
        ratio = np.clip(sky_lum / drone_lum, 0.3, 1.5)
        adjusted = comp_float * ratio
        blend = drone_mask[..., np.newaxis]
        return (blend * adjusted + (1 - blend) * comp_float).clip(0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════════
    def inpaint(self, background, nu=0.5, nv=0.5, target_px=70,
                gen_seed=42, gen_steps=30, blend_seed=123, blend_steps=10,
                blend_strength=0.25, guidance=7.5,
                crop_pad_ratio=2.0, feather_ratio=0.5,
                gen_prompt=None, gen_neg=None,
                blend_prompt=None, blend_neg=None,
                intermediates_dir=None):
        if not self._loaded:
            self.load()

        out = {}

        # Step 1: 白底 prompt 生成 (512×512)
        print("🎨 Step 1: LoRA 生成 (512×512) ...")
        t0 = time.time()
        drone_full = self._generate(gen_seed, gen_steps, guidance, gen_prompt, gen_neg)
        out["drone_full"] = drone_full
        print(f"   ✅ ({time.time()-t0:.1f}s)")

        # Step 2: rembg 抠图 + 贴入
        print("✂️  Step 2: rembg 抠图 + 贴入 ...")
        drone_rgba, bbox = self._extract_drone_rembg(drone_full)
        if drone_rgba is None:
            return {"result": background, **out}
        composite, cx, cy, actual_px = self._scale_and_paste(
            drone_rgba, bbox, background, nu, nv, target_px)
        out["composite"] = composite
        dw = bbox[2] - bbox[0]
        print(f"   ✅ {dw}px → {actual_px}px, center ({cx},{cy})")

        # Step 3: crop-scale-blend
        print(f"🖌️  Step 3: crop-scale-blend (pad={crop_pad_ratio}×, strength={blend_strength}) ...")
        t0 = time.time()
        result = self._local_blend(
            background, composite, cx, cy, actual_px,
            blend_seed, blend_steps, blend_strength, guidance,
            crop_pad_ratio, feather_ratio,
            blend_prompt, blend_neg)
        out["result"] = result
        print(f"   ✅ ({time.time()-t0:.1f}s)")

        if intermediates_dir:
            os.makedirs(intermediates_dir, exist_ok=True)
            drone_full.save(os.path.join(intermediates_dir, "drone_full.png"))
            drone_rgba.save(os.path.join(intermediates_dir, "drone_rgba.png"))
            cv2.imwrite(os.path.join(intermediates_dir, "composite.png"), composite)

        return out

    def _generate(self, seed, steps, guidance, prompt, neg):
        g = torch.Generator(device=self.device).manual_seed(seed)
        r = self.pipe_txt(
            prompt=prompt, negative_prompt=neg,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=g, output_type="pil",
        )
        return r.images[0]

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

    def _local_blend(self, background, composite, cx, cy, drone_px,
                     seed, steps, strength, guidance, pad_ratio, feather_ratio,
                     blend_prompt, blend_neg):
        H, W = background.shape[:2]
        pad = max(int(drone_px * pad_ratio), 32)
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(W, cx + pad), min(H, cy + pad)
        cw, ch = x2 - x1, y2 - y1
        if cw < 16 or ch < 16:
            return composite

        crop = composite[y1:y2, x1:x2]
        crop = self._match_lighting(crop, background[y1:y2, x1:x2],
                                    drone_px, cx - x1, cy - y1)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb).resize((512, 512), Image.LANCZOS)

        g = torch.Generator(device=self.device).manual_seed(seed)
        r = self.pipe_img(
            prompt=blend_prompt, negative_prompt=blend_neg,
            image=crop_pil, strength=strength,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=g, output_type="pil",
        )
        blended_crop = np.array(r.images[0].resize((cw, ch), Image.LANCZOS))
        blended_bgr = cv2.cvtColor(blended_crop, cv2.COLOR_RGB2BGR)

        # Feather mask
        mask = np.ones((ch, cw), dtype=np.float32)
        feather_px = max(2, int(min(cw, ch) * feather_ratio))
        ramp = np.linspace(0, 1, feather_px)
        mask[:feather_px, :] *= ramp[:, np.newaxis]
        mask[-feather_px:, :] *= ramp[::-1, np.newaxis]
        mask[:, :feather_px] *= ramp[np.newaxis, :]
        mask[:, -feather_px:] *= ramp[::-1][np.newaxis, :]

        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended_final = (mask_3ch * blended_bgr.astype(float) +
                         (1 - mask_3ch) * crop.astype(float)).astype(np.uint8)

        result = composite.copy()
        result[y1:y2, x1:x2] = blended_final
        return result

    def unload(self):
        if self.pipe_txt:
            del self.pipe_txt
        if self.pipe_img:
            del self.pipe_img
        self.pipe_txt = self.pipe_img = None
        torch.cuda.empty_cache()
        self._loaded = False

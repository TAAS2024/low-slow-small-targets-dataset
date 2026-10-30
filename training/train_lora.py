#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a kohya-compatible LoRA adapter on the SD 1.5 architecture.

This is the diffusers + PEFT counterpart of the recipe that produced the
checkpoint used in the paper. It trains low-rank adapters over the UNet
attention projections (and optionally the CLIP text-encoder attention
projections), then exports the result **in kohya key format** so that
``generator/lora_loader.py`` can merge it into a pipeline without a conversion
step.

    python training/train_lora.py --config training/configs/lora_sd15_drn3.yaml
    python training/train_lora.py --config training/configs/lora_sd15_drn3.yaml \
        --set data.resolution=768 --set train.max_train_steps=2000

LICENCE WARNING — read before you publish anything trained here
---------------------------------------------------------------
The base model is Stable Diffusion v1.5, distributed under **CreativeML
OpenRAIL-M**. An adapter trained on it is a derivative work: it inherits the
OpenRAIL-M terms, including the Attachment A use-based restrictions, and those
restrictions must be shipped with the weights. The MIT licence of this
repository covers this script, not the adapter it produces. If you substitute a
non-commercial backbone, the adapter becomes non-commercial as well.
See THIRD_PARTY_LICENSES.md.

Tested with Python 3.10, torch 2.6.0 + cu124, diffusers 0.39.0, peft 0.13.
A rank-8 adapter on this recipe fits in 8 GB of VRAM at 512x512 with gradient
checkpointing enabled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lss_paths  # noqa: E402  (path resolution lives in one place)

LICENCE_BANNER = """
────────────────────────────────────────────────────────────────────────────
Base model: {model}
Licence:    CreativeML OpenRAIL-M (SD 1.5 family) — see THIRD_PARTY_LICENSES.md
The adapter produced by this run is a DERIVATIVE of that model and inherits its
licence and use-based restrictions. It is not MIT-licensed.
────────────────────────────────────────────────────────────────────────────
"""


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "base_model": "",
    "base_model_id": "sd-legacy/stable-diffusion-v1-5",
    "data": {
        "image_dir": "",
        "resolution": 512,
        "num_repeats": 10,
        "flip_aug": False,
        "caption_extension": ".txt",
        "class_token": "drn3_uav",
    },
    "lora": {
        "rank": 8,
        "alpha": 4,
        "dropout": 0.0,
        "train_text_encoder": False,
        "text_encoder_rank": 8,
        "text_encoder_alpha": 4,
    },
    "train": {
        "max_train_steps": 3000,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine_with_restarts",
        "lr_num_cycles": 3,
        "lr_warmup_steps": 0,
        "adam_weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_xformers": True,
        "min_snr_gamma": 5.0,
        "seed": 42,
    },
    "output": {
        "output_dir": "runs/lora_sd15_drn3",
        "save_every_n_steps": 500,
        "export_format": "kohya",
        "emit_metadata": True,
    },
    "provenance": {
        "adapter_name": "drn3_pocket_uav",
        "notes": "CDFF target adapter, SD 1.5 architecture, CreativeML OpenRAIL-M derivative",
    },
}


def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(text: str):
    """Turn a `--set key=value` string into a Python scalar."""
    low = text.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def load_config(path: str | None, overrides: list[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path:
        cfg = _deep_update(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def resolve_base_model(cfg: dict) -> str:
    """Prefer a local diffusers snapshot, fall back to a hub id."""
    explicit = (cfg.get("base_model") or "").strip()
    if explicit:
        if not Path(explicit).is_dir():
            raise SystemExit(f"base_model path does not exist: {explicit}")
        return explicit
    if lss_paths.SD15_PATH.is_dir():
        return str(lss_paths.SD15_PATH)
    return cfg["base_model_id"]


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class CaptionedImageDataset(Dataset):
    """Images + sidecar ``.txt`` captions, the layout kohya expects.

    The directory may also be a kohya ``N_name`` folder; the numeric repeat
    prefix is read from the directory name when ``num_repeats`` is not set
    explicitly.
    """

    def __init__(self, image_dir: Path, tokenizer, resolution: int,
                 flip_aug: bool, repeats: int, caption_ext: str,
                 class_token: str | None):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise SystemExit(f"data.image_dir does not exist: {self.image_dir}")
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.flip_aug = flip_aug
        self.class_token = class_token

        pairs = []
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            cap = p.with_suffix(caption_ext)
            if not cap.exists():
                continue
            text = cap.read_text(encoding="utf-8").strip()
            if class_token and class_token not in text:
                text = f"{class_token}, {text}" if text else class_token
            pairs.append((p, text))
        if not pairs:
            raise SystemExit(
                f"no image + {caption_ext} caption pairs found in {self.image_dir}")
        self.pairs = pairs * max(1, repeats)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print(f"dataset: {len(pairs)} image/caption pairs × {repeats} repeats "
              f"= {len(self.pairs)} samples @ {resolution}px")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        path, caption = self.pairs[idx]
        img = Image.open(path).convert("RGB")
        if self.flip_aug and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        tokens = self.tokenizer(
            caption, max_length=77, padding="max_length", truncation=True,
            return_tensors="pt")
        return {"pixel_values": self.transform(img), "input_ids": tokens.input_ids[0]}


# ══════════════════════════════════════════════════════════════════════════
# LoRA export: PEFT state dict → kohya key format
# ══════════════════════════════════════════════════════════════════════════

def peft_to_kohya(state: dict, prefix: str, alpha: float) -> dict:
    """Translate PEFT names into the ``lora_unet_*`` / ``lora_te_*`` layout.

    PEFT:  base_model.model.down_blocks.1.attentions.0...to_q.lora_A.default.weight
    kohya: lora_unet_down_blocks_1_attentions_0_..._to_q.lora_down.weight  (+ .alpha)

    The scaling in both formats is ``alpha / rank`` applied to ``B @ A``, so the
    export is weight-exact rather than approximate.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if ".lora_A." in key:
            module, kind = key.split(".lora_A.")[0], "down"
            value = value
        elif ".lora_B." in key:
            module, kind = key.split(".lora_B.")[0], "up"
        else:
            continue
        module = module.replace("base_model.model.", "")
        name = prefix + module.replace(".", "_")
        out[f"{name}.lora_{kind}.weight"] = value.detach().to(torch.float16).cpu()
        out[f"{name}.alpha"] = torch.tensor(float(alpha), dtype=torch.float16)
    return out


def _metadata(cfg: dict, steps: int, n_images: int) -> dict:
    t = cfg["train"]
    return {
        "modelspec.architecture": "stable-diffusion-v1/lora",
        "modelspec.implementation": "diffusers+peft",
        "modelspec.title": str(cfg["provenance"]["adapter_name"]),
        "modelspec.resolution": f'{cfg["data"]["resolution"]}x{cfg["data"]["resolution"]}',
        "modelspec.date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modelspec.license": "CreativeML OpenRAIL-M",
        "ss_base_model_version": "sd_v1",
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(cfg["lora"]["rank"]),
        "ss_network_alpha": str(cfg["lora"]["alpha"]),
        "ss_learning_rate": str(t["learning_rate"]),
        "ss_lr_scheduler": str(t["lr_scheduler"]),
        "ss_max_train_steps": str(t["max_train_steps"]),
        "ss_min_snr_gamma": str(t["min_snr_gamma"]),
        "ss_mixed_precision": str(t["mixed_precision"]),
        "ss_gradient_checkpointing": str(bool(t["gradient_checkpointing"])).lower(),
        "ss_seed": str(t["seed"]),
        "ss_num_train_images": str(n_images),
        "ss_trained_steps": str(steps),
        "ss_note": "derivative of Stable Diffusion v1.5 — OpenRAIL-M",
    }


# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

def build_lr_lambda(total_steps: int, num_cycles: int, warmup: int, kind: str):
    if kind == "constant":
        return lambda step: 1.0
    cycle = max(1, total_steps // max(1, num_cycles))

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, step / max(1, warmup))
        progress = (step % cycle) / cycle
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cosine, 0.01)   # floor at 1 %, as in the reference recipe

    return lr_lambda


def min_snr_weight(alphas_cumprod: torch.Tensor, timesteps: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    a = alphas_cumprod.to(timesteps.device)[timesteps].float()
    snr = a / (1.0 - a)
    return torch.clamp(snr, max=gamma) / snr


def train(cfg: dict) -> Path:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from tqdm import tqdm

    base = resolve_base_model(cfg)
    print(LICENCE_BANNER.format(model=base))

    t = cfg["train"]
    seed = int(t["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    out_dir = Path(cfg["output"]["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    accelerator = Accelerator(
        mixed_precision=t["mixed_precision"],
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        project_config=ProjectConfiguration(project_dir=str(out_dir),
                                            logging_dir=str(out_dir / "logs")),
    )
    print(f"device={accelerator.device}  mixed_precision={t['mixed_precision']}")

    print("loading SD 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base,
        torch_dtype=torch.float16 if t["mixed_precision"] == "fp16" else torch.float32,
        safety_checker=None,
        local_files_only=Path(base).is_dir(),
    )
    tokenizer, text_encoder, unet, vae = (
        pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae)
    noise_scheduler: DDPMScheduler = pipe.scheduler
    del pipe

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if t["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    if t["use_xformers"]:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("xformers: enabled")
        except Exception as exc:                     # noqa: BLE001
            print(f"xformers unavailable ({exc}); continuing without it")

    lora = cfg["lora"]
    unet = get_peft_model(unet, LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        target_modules=r".*(to_q|to_k|to_v|to_out\.0).*",
        lora_dropout=float(lora["dropout"]),
        bias="none",
    ))
    if lora["train_text_encoder"]:
        text_encoder = get_peft_model(text_encoder, LoraConfig(
            r=int(lora["text_encoder_rank"]),
            lora_alpha=int(lora["text_encoder_alpha"]),
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=float(lora["dropout"]),
            bias="none",
        ))
        text_encoder.requires_grad_(True)
    unet.print_trainable_parameters()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    if lora["train_text_encoder"]:
        trainable += [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(t["learning_rate"]),
                                  weight_decay=float(t["adam_weight_decay"]))

    dataset = CaptionedImageDataset(
        Path(cfg["data"]["image_dir"]).expanduser(),
        tokenizer,
        resolution=int(cfg["data"]["resolution"]),
        flip_aug=bool(cfg["data"]["flip_aug"]),
        repeats=int(cfg["data"]["num_repeats"]),
        caption_ext=str(cfg["data"]["caption_extension"]),
        class_token=cfg["data"]["class_token"],
    )
    dataloader = DataLoader(dataset, batch_size=int(t["batch_size"]),
                            shuffle=True, num_workers=0)

    total_steps = int(t["max_train_steps"])
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    text_encoder = accelerator.prepare(text_encoder)
    vae = accelerator.prepare(vae)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(total_steps, int(t["lr_num_cycles"]),
                                   int(t["lr_warmup_steps"]), t["lr_scheduler"]))
    scheduler = accelerator.prepare(scheduler)

    alphas_cumprod = noise_scheduler.alphas_cumprod
    unet.train()
    text_encoder.eval() if not lora["train_text_encoder"] else text_encoder.train()
    vae.eval()

    save_every = int(cfg["output"]["save_every_n_steps"])
    export_format = str(cfg["output"]["export_format"])
    n_images = len(dataset.pairs) // max(1, int(cfg["data"]["num_repeats"]))

    def export(step: int, target: Path) -> None:
        unwrapped = accelerator.unwrap_model(unet)
        state = {k: v for k, v in unwrapped.state_dict().items() if "lora_" in k}
        if export_format == "kohya":
            payload = peft_to_kohya(state, "lora_unet_", float(lora["alpha"]))
            if lora["train_text_encoder"]:
                te_state = {k: v for k, v in
                            accelerator.unwrap_model(text_encoder).state_dict().items()
                            if "lora_" in k}
                payload.update(peft_to_kohya(te_state, "lora_te_",
                                             float(lora["text_encoder_alpha"])))
        else:
            payload = {k: v.to(torch.float16).cpu() for k, v in state.items()}
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = _metadata(cfg, step, n_images) if cfg["output"]["emit_metadata"] else None
        save_file(payload, str(target), metadata=meta)
        print(f"\n  exported {len(payload)} tensors → {target}")

    global_step = 0
    progress = tqdm(total=total_steps, desc="train")
    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                ids = batch["input_ids"].to(accelerator.device)
                if lora["train_text_encoder"]:
                    encoder_hidden = text_encoder(ids)[0]
                else:
                    with torch.no_grad():
                        encoder_hidden = text_encoder(ids)[0]

                pred = unet(noisy, timesteps,
                            encoder_hidden_states=encoder_hidden).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="none")
                loss = loss.mean(dim=[1, 2, 3])
                if t["min_snr_gamma"]:
                    loss = loss * min_snr_weight(alphas_cumprod, timesteps,
                                                 float(t["min_snr_gamma"]))
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad],
                        float(t["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                 snr=str(bool(t["min_snr_gamma"])))
            if global_step % save_every == 0:
                export(global_step,
                       out_dir / "checkpoints" / f"step{global_step}.safetensors")
    progress.close()

    final = out_dir / "final" / "lora_weights.safetensors"
    export(global_step, final)

    summary = {
        "base_model": base,
        "base_model_license": "CreativeML OpenRAIL-M",
        "derivative_license": "CreativeML OpenRAIL-M (inherited; not MIT)",
        "steps": global_step,
        "images": n_images,
        "export": str(final),
        "config": cfg,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndone → {final}")
    print("The adapter is a derivative of SD 1.5 and is licensed CreativeML "
          "OpenRAIL-M; ship THIRD_PARTY_LICENSES.md with it.")
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent /
                                            "configs/lora_sd15_drn3.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key, e.g. --set train.max_train_steps=2000")
    ap.add_argument("--print-config", action="store_true",
                    help="resolve and print the configuration, then exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return 0
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Low-Slow-Small Object Synthetic Data Generation Pipeline

> **Dual-Modal (RGB + TIR) Synthetic Data Generation for Low-Altitude, Slow-Speed, Small-Target Detection**
>
> UAV · Kite · Balloon · Airship — Scene Composition via ControlNet + Diffusion Models

---

## Overview

This project builds a dual-modal synthetic data generation pipeline for **low-slow-small (LSS) objects** — drones, kites, balloons, and airships. The core idea is to generate realistic background scenes using a ControlNet-conditioned diffusion model, then composite LSS targets onto those scenes in both **RGB** and **thermal infrared (TIR)** modalities, validated by a three-stage agent verification chain with closed-loop feedback.

### Why Synthetic Data?

- LSS objects are rare, small, and hard to annotate in real-world imagery
- Thermal infrared data is especially scarce and expensive to collect
- Synthetic generation enables controlled diversity (weather, time-of-day, target placement) and unlimited scale

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Dual-Branch Generation Pipeline                         │
│                                                                          │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐  │
│  │  COCO-Stuff Segmentation    │    │  Anti-UAV410 TIR Distribution   │  │
│  │  (183-class semantic maps)  │    │  (spatial bbox distribution)    │  │
│  └─────────────┬───────────────┘    └───────────────┬─────────────────┘  │
│                │                                    │                    │
│                ▼                                    ▼                    │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐  │
│  │  ControlNet-Seg (scale=0.75)│    │  Target Placement Sampler       │  │
│  │  + SD 1.5 (512×512, 25 steps)│   │  (size, position, count)        │  │
│  └─────────────┬───────────────┘    └───────────────┬─────────────────┘  │
│                │                                    │                    │
│                ▼                                    ▼                    │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐  │
│  │  Background Scene Pool      │    │  DroneMMset RGB Targets         │  │
│  │  (~10K aerial views)        │    │  Anti-UAV410 TIR Targets        │  │
│  └─────────────┬───────────────┘    └───────────────┬─────────────────┘  │
│                │                                    │                    │
│                └────────────┬───────────────────────┘                    │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                Three-Stage Agent Verification Chain                  │  │
│  │  Stage 1 (CLIP Semantic) → Stage 2 (YOLO Detection) → Stage 3 (IQA) │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Closed-Loop Feedback → Environment Branch Agent                    │  │
│  │  (24 conditions: 6 weather × 4 time-of-day)                        │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|:--|:--|
| **SD 1.5 over SDXL** | SDXL requires >12GB VRAM; SD 1.5 runs on 8GB RTX 4060 Laptop with 2.9GB peak |
| **ControlNet-Seg over Transformer layout prediction** | COCO-Stuff bbox only exists for "things" — sky/tree/mountain/water have zero bbox → Transformer fails (see Failure Analysis) |
| **Segmentation maps directly into ControlNet** | Pre-trained ADE20k ControlNet already encodes layout→appearance mapping; no retraining needed |
| **6 super-classes (not 183)** | Merging COCO-Stuff's 183 classes into sky/tree/building/mountain/water/ground matches ADE20k ControlNet's training distribution |

---

## Dataset

Four public datasets are used. **Datasets are too large to include in this repository** — download them separately:

| Dataset | Size | Modality | Role | Download |
|:--|:--|:--|:--|:--|
| **COCO-Stuff** | 629 MB (56,817 images) | RGB semantic segmentation | Scene layout source (15 natural background classes) | [cocodataset.org](https://cocodataset.org/#download) → 2017 Stuffthingmaps |
| **Anti-UAV410** | ~12 GB (438K frames) | Pure thermal infrared (TIR) | TIR UAV spatial distribution + YOLO fine-tuning | [GitHub](https://github.com/ucas-vg/Anti-UAV410) |
| **SIDBench** | 35 MB (123 Python files) | RGB synthesis detection toolkit | Stage 3 artifact detection reference | [GitHub](https://github.com/megvii-research/SIDBench) |
| **DroneMMset** | ~12 GB (7,752 frames) | RGB + IR fully paired | Three-stage validation chain calibration baseline | [GitHub](https://github.com/DroneMMset/DroneMMset) |

### COCO-Stuff → 6 Super-Class Mapping

```
COCO-Stuff 183 classes → 6 super-classes → ADE20k-style RGB colors

Super-Class    COCO-Stuff Classes                                        RGB Color
────────────   ────────────────────────────────────────────────────     ──────────
sky            sky(157), clouds(106), fog(120)                          (128,192,255)
tree           tree(169), plant-other(94), grass(129), bush(97), etc.   (0,128,0)
building       building-other(96), house(128), skyscraper(158), etc.    (128,128,128)
mountain       mountain(135), hill(127), rock(150)                      (139,90,43)
water          water-other(178), sea(148), river(155)                   (30,144,255)
ground         road(98), sidewalk(95), sand(163), dirt(165), etc.       (200,180,140)
```

> **Why 6 super-classes?** ADE20k (on which ControlNet-Seg was trained) uses a similar class granularity. Direct 183→RGB mapping produces colors ControlNet has never seen during training, degrading layout fidelity.

---

## Models

**Models are too large to include in this repository** (~30 GB total). Download from Hugging Face:

| Model | Size | Purpose | Hugging Face |
|:--|:--|:--|:--|
| `stable-diffusion-v1-5` | 21 GB | Main diffusion generator (VAE + UNet + Text Encoder) | [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| `sd-controlnet-seg` | 2.7 GB | ControlNet segmentation conditioning | [`lllyasviel/sd-controlnet-seg`](https://huggingface.co/lllyasviel/sd-controlnet-seg) |
| `clip-vit-large-patch14` | 6.4 GB | CLIP ViT-L/14 (reserve, for Stage 1 verification) | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) |
| `yolov8x.pt` | 131 MB | YOLOv8x (for Stage 2 detection verification) | [`ultralytics/yolov8`](https://github.com/ultralytics/ultralytics) |

Place downloaded models under `0-model/`:

```
0-model/
├── stable-diffusion-v1-5/
│   ├── vae/
│   ├── unet/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── scheduler/
│   └── ...
├── sd-controlnet-seg/
├── clip-vit-large-patch14/
└── yolov8x.pt
```

### SD 1.5 Setup Notes

1. Download **all components** (VAE, UNet, Text Encoder) — some partial downloads are UNet-only
2. If using Chinese mirrors (hf-mirror.com): `export HF_ENDPOINT=https://hf-mirror.com`
3. If UNet loads only `diffusion_pytorch_model.safetensors` but `diffusers` expects `*.fp16.safetensors`, create a symlink:
   ```bash
   cd 0-model/stable-diffusion-v1-5/unet
   ln -s diffusion_pytorch_model.safetensors diffusion_pytorch_model.fp16.safetensors
   ```

---

## Code Structure

```
low-slow-small-dataset-generation/
│
├── 2-ControlNet/                      # Step 2: Scene Generation (current active module)
│   ├── coco_seg_converter.py          # 183-class grayscale PNG → 6-superclass RGB conversion
│   ├── generate_scenes.py             # Batch scene generation with ControlNet + SD 1.5
│   ├── compare_seg2scene.py           # Three-mode comparison visualization tool
│   ├── controlnet_renderer.py         # ControlNet rendering core (pipeline load / inference / postprocess)
│   ├── scene_parser.py                # Segmentation map statistical analysis
│   ├── enrich_fast.py                 # Fast pixel-ratio enrichment from COCO-Stuff PNGs
│   └── outputs/
│       └── comparisons/               # Generated comparison figures
│
├── 0-model/                           # Downloaded models (not in repo — see above)
├── 0-database/                        # Downloaded datasets (not in repo — see above)
```

---

## VRAM Requirements

| Configuration | Peak VRAM | Minimum GPU |
|:--|:--|:--|
| SD 1.5 text-to-image (no ControlNet) | ~2.2 GB | 4 GB |
| SD 1.5 + ControlNet-Seg | ~2.9 GB | 6 GB |
| Dual pipeline (ablation mode) | ~5.1 GB | 8 GB |
| SDXL (1024×1024) | >12 GB | 16 GB (not tested) |

This project is developed on an **RTX 4060 Laptop (8GB VRAM)** and all experiments run within this constraint.



## License

This project is released under the MIT License. Individual datasets and models retain their original licenses — refer to each source for details.

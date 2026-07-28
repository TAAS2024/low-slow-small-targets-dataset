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
└── 1-布局生成器/                      # Step 1: Transformer layout prediction (DEPRECATED)
```

---

## Pipeline Quick Start

### Prerequisites

```bash
pip install diffusers transformers accelerate pillow numpy torch torchvision
```

### Step 1: Convert COCO-Stuff PNGs to ControlNet-compatible RGB

```bash
cd 2-ControlNet

python coco_seg_converter.py \
    --png-dirs ../0-database/coco-stuff/train2017 \
    --output-dir outputs/seg_rgb \
    --max-images 10000
```

This creates 6-superclass RGB segmentation maps in ADE20k color space.

### Step 2: Generate Background Scenes

```bash
python generate_scenes.py \
    --image-id 59906 \
    --seed 42 \
    --steps 25 \
    --control-scale 0.75 \
    --guidance-scale 7.5
```

### Step 3: Create Comparison Figures

Three modes are available:

```bash
# Single comparison: segmentation map vs generated scene
python compare_seg2scene.py --mode single --image-id 59906 --control-scale 0.75 --seed 42

# Multi-scale: same seg map across different ControlNet strengths
python compare_seg2scene.py --mode multiscale --image-id 59906 \
    --scales 0.0 0.3 0.5 0.75 1.0 --seed 42

# Ablation: Pure SD1.5 vs SD1.5+ControlNet (same seed, same prompt)
python compare_seg2scene.py --mode ablation --image-id 59906 356253 144486 \
    --control-scale 0.75 --seed 42
```

---

## Results

### Ablation Experiment: Pure SD 1.5 vs. SD 1.5 + ControlNet

Three representative COCO-Stuff scenes were selected for controlled ablation:

| Scene ID | Composition | Characteristic |
|:--|:--|:--|
| **59906** | building 35.9%, sky 26.0%, tree 16.5%, ground 13.7% | Building-dominant — strongest ControlNet effect |
| **356253** | ground 52.8%, sky 28.3%, tree 10.7% | Ground-dominant — simpler structure |
| **144486** | ground 58.7%, sky 21.5%, tree 10.1% | Ground+sky — semi-open scene |

**Ablation Setup** (fair comparison):
- Both pipelines use **identical seed=42** → same initial noise
- Both pipelines use **identical prompt**: `"photorealistic aerial view of a city with buildings, sky, and trees"`
- Both pipelines use **identical SD 1.5 weights and denoising steps=25**
- Pure SD 1.5: `StableDiffusionPipeline` (no ControlNet)
- SD 1.5 + CN: `StableDiffusionControlNetPipeline` (conditioning_scale=0.75)
- Output: 3-column figures [Seg Map | Pure SD1.5 | SD1.5+CN] (no text annotations)

**Key Finding**: For building-dominant scenes (59906), ControlNet dramatically constrains spatial layout — buildings appear at the correct locations. For ground-dominant scenes, the effect is subtler since the prompt alone can produce reasonable terrain.

### Why conditioning_scale = 0.75?

Tested across [0.0, 0.3, 0.5, 0.75, 1.0]:

| Scale | Effect |
|:--|:--|
| 0.0 | ControlNet disabled → equivalent to pure SD 1.5, layout random |
| 0.3 | Weak constraint — layout slightly influenced but unreliable |
| 0.5 | Moderate constraint — most regions align |
| **0.75** | **Strong constraint with natural texture** ← selected |
| 1.0 | Over-constrained — perfect layout but rigid, less natural appearance |

0.75 is the sweet spot between layout fidelity and visual naturalness.

### Performance (RTX 4060 Laptop, 8GB VRAM)

| Mode | Peak VRAM | Speed |
|:--|:--|:--|
| Single pipeline (SD1.5 + ControlNet) | ~2.9 GB | ~12s / image @ 25 steps |
| Dual pipeline (ablation: 2× UNet in memory) | ~5.1 GB | ~25s / pair |
| GPU utilization | 85–95% | — |

---

## Failure Analysis: Step 1 Transformer Layout Prediction (Deprecated)

### Original Idea

Train a Transformer to predict bounding box layouts from COCO-Stuff category labels, then use the predicted bboxes to compose background scenes for the diffusion model.

### Root Cause of Failure

COCO-Stuff annotations are split into two groups:
- **Things** (80 classes): have instance annotations → have bounding boxes
- **Stuff** (91 classes): only panoptic annotations → **no bounding boxes**

Our 15 background categories (sky, tree, mountain, water, etc.) are predominantly **stuff** — their bbox annotations are entirely zero. The Transformer had no spatial signal to learn from.

### Two Failed Training Attempts

| Version | Parameters | Matching | Problem | IoU |
|:--|:--|:--|:--|:--|
| V1 | 7.5M | MSE Loss (fixed order) | Mode collapse: all bboxes converge to cx=0.5, unable to distinguish classes | ~0.053 |
| V2 | 44.5M | Hungarian matching + larger model | Confidence collapse: model learns to output near-zero confidence to minimize loss | ~0.024 |

- **V1**: With fixed-order MSE matching, the model must predict bboxes at fixed output slots. Since training data has no bboxes for stuff classes, the model converges to a "universal center" (cx=0.5) regardless of input.
- **V2**: Hungarian matching lets the model freely choose which bboxes to predict. The model discovers that predicting nothing (confidence→0) achieves lower loss than making wrong predictions — complete prediction collapse.

### Why ControlNet Works Better

The COCO-Stuff segmentation maps **already exist** — no need to predict them. ControlNet-Seg, pre-trained on ADE20k, already knows "blue region → sky, gray region → building." The segmentation map is an inherently superior layout representation: every pixel encodes its class membership, far more precise than bboxes.

---

## Novelty & Related Work

Key related work: DetDiffusion (CVPR 2024), AeroGen, AnySynth, CFHA, EarthBridge, DiffusionAgent.

Our **combined contribution** that differs from prior work:

1. **Three-stage Agent Verification Chain** (CLIP → YOLO → IQA) with structured failure codes — no prior work applies multi-stage agent verification to synthetic data quality
2. **Closed-loop Feedback**: verification failures automatically trigger environment re-generation (weather/time-of-day variation) — agent→agent automated loop
3. **Dual-branch Decoupling**: RGB spatial layout branch and TIR modality branch separated, enabling uni-modal datasets (RGB-only backgrounds + TIR-only targets) to jointly produce dual-modal outputs

---

## VRAM Requirements

| Configuration | Peak VRAM | Minimum GPU |
|:--|:--|:--|
| SD 1.5 text-to-image (no ControlNet) | ~2.2 GB | 4 GB |
| SD 1.5 + ControlNet-Seg | ~2.9 GB | 6 GB |
| Dual pipeline (ablation mode) | ~5.1 GB | 8 GB |
| SDXL (1024×1024) | >12 GB | 16 GB (not tested) |

This project is developed on an **RTX 4060 Laptop (8GB VRAM)** and all experiments run within this constraint.

---

## Next Steps

1. **Batch generation**: 5,000–10,000 background scenes for a sufficient scene pool
2. **Multi-seed experiments**: validate stability across seeds (7, 13, 42, 99)
3. **Drone compositing**: background + UAV placement + blending pipeline (Step 3)
4. **Diversity analysis**: compare generated 6-class distribution against original COCO-Stuff distribution
5. **Edge case testing**: pure-ground / pure-sky / pure-water scenes for robustness

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{zhang2023controlnet,
  title     = {Adding Conditional Control to Text-to-Image Diffusion Models},
  author    = {Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{rombach2022high,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{caesar2018cocostuff,
  title     = {{COCO-Stuff}: Thing and Stuff Classes in Context},
  author    = {Caesar, Holger and Uijlings, Jasper and Ferrari, Vittorio},
  booktitle = {ECCV},
  year      = {2018}
}
```

---

## License

This project is released under the MIT License. Individual datasets and models retain their original licenses — refer to each source for details.

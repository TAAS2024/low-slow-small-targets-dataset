# Low-Slow-Small Object Synthetic Data Generation Pipeline

> **Dual-Modal (RGB + TIR) Synthetic Data Generation for Low-Altitude, Slow-Speed, Small-Target Detection**
>
> UAV · Kite · Balloon · Airship — LLM→Transformer→ControlNet + LoRA + IR Conversion + Agent Verification

---

## Overview

A dual-modal synthetic data generation pipeline for **low-slow-small (LSS) objects** — drones, kites, balloons, and airships. The core architecture:

> **LLM → Transformer → ControlNet (spatial skeleton) + Drone LoRA (appearance) → RGB→IR Conversion → 4-Stage Agent Verification → Closed-Loop Feedback**

The user describes a scene in natural language (e.g., "阴天傍晚，四旋翼从远处飞近，仰拍"), and the system automatically generates paired RGB + thermal infrared (TIR) training images with precise spatial control.

### Key Innovation

- **Agent 1 (LLM)** parses natural language → structured 9-field JSON
- **Agent 2 (Transformer)** encodes JSON → per-frame ControlNet conditioning vectors
- **Agent 3 (ControlNet)** generates spatial skeleton (depth/seg maps) — determines **WHERE**
- **Agent 4 (Drone LoRA)** fills appearance on the skeleton — determines **WHAT**
- **Agent 5 (IR Converter)** converts RGB → pseudo-color thermal IR
- **Agent 6 (4-Stage Verification)** filters outputs: background realism → semantic → detection → quality
- **Agent 7 (Closed-Loop)** routes failure codes → adjusts parameters → regenerates

### Why This Matters

- LSS objects are rare, small, and hard to annotate in real-world imagery
- Thermal infrared data is especially scarce and expensive to collect
- **No public dataset exists** for non-UAV LSS categories (kite/balloon/airship)
- Most public UAV datasets are drone-mounted (looking down), not ground-based (looking up at sky)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    📥  Natural Language Input                              │
│  "阴天下午，四旋翼从远处飞近，在工业厂房区域仰拍观察"                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 1: LLM Semantic Parsing                                    │     │
│  │  NL → 9-Field JSON (drone_type, trajectory[], weather, ...)      │     │
│  │  DeepSeek / OpenAI / Claude / Dry-Run                            │     │
│  └──────────────────────────────┬──────────────────────────────────┘     │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 2: Transformer Spatiotemporal Encoding                     │     │
│  │  5 Encoders: Position → Depth → Pose → Weather-Time → Camera     │     │
│  │  SpatialQueryGenerator: 256 query → 16×16 → ConvTranspose →     │     │
│  │  64×64 per-frame conditioning vectors                          │     │
│  └──────────────────────────────┬──────────────────────────────────┘     │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 3: ControlNet Spatial Control (WHERE)                      │     │
│  │  Conditioning Vectors → Depth/Seg Maps → Spatial Skeleton        │     │
│  │  SD1.5 + ControlNet-Seg, conditioning_scale=0.75                 │     │
│  └──────────────────────────────┬──────────────────────────────────┘     │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 4: Drone LoRA Rendering (WHAT)                            │     │
│  │  ControlNet Spatial Skeleton + Drone LoRA → Complete RGB Scene   │     │
│  │  LoRA rank=16, alpha=8, trained on 98 commercial drone images    │     │
│  └──────────────────────────────┬──────────────────────────────────┘     │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 5: IR Domain Conversion                                    │     │
│  │  RGB → White-Hot Pseudo-Color Thermal IR                         │     │
│  │  Post-processing approach (avoids VAE domain mismatch on IR)     │     │
│  └──────────────────────────────┬──────────────────────────────────┘     │
│                                 │                                         │
│                    ┌────────────┴────────────┐                            │
│                    ▼                         ▼                            │
│              RGB Output                 IR Output                         │
│                    │                         │                            │
│                    └────────────┬────────────┘                            │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 6: 4-Stage Verification Chain                              │     │
│  │  S0: Background Realism (ResNet binary classifier)               │     │
│  │  S1: CLIP Semantic (viewpoint/weather alignment)                  │     │
│  │  S2: YOLO Detection (target detectability)                       │     │
│  │  S3: IQA Quality (blur/artifact/LoRA fusion quality)             │     │
│  └──────────────────────────────┬──────────────────────────────────┘     │
│                                 │                                         │
│                                 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Agent 7: Closed-Loop Feedback (CDFF)                             │     │
│  │  Failure Code → Component Router → CN/LoRA/IR Buffers            │     │
│  │  → Incremental Fine-tuning → Weight Replacement                  │     │
│  │  Three paradigms: A (pass→train) / B (fail→fix→contrast) /       │     │
│  │  C (rank→align)                                                  │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  6 Weather × 5 Time-of-Day = 30 environmental conditions                 │
│  4 Target Classes: UAV / Kite / Balloon / Airship                        │
│  2 Modalities: RGB + Thermal Infrared (TIR)                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why 7 Agents?

A generation pipeline of this complexity is naturally agentic. Each module has an independent interface — testable, replaceable, and independently observable.

| Monolithic Script | Agent Architecture |
|:--|:--|
| One change breaks everything | Each Agent has independent interface |
| Failure → restart from scratch | Failure codes → retry only the failed step |
| Black box | Structured logs at every step |
| Sequential only | Independent Agents can run in parallel |
| No human intervention | Human-in-the-loop calibration at failure |

### Why Single LoRA + IR Conversion (not Multi-LoRA)?

Initial design used 3 LoRAs (RGB background + IR background + drone target) fused in latent space. IR background LoRA failed because SD 1.5 VAE was trained on natural RGB images — IR grayscale (three identical channels) falls outside the VAE training manifold, producing pure noise.

**Solution**: Single drone LoRA generates RGB → `rgb2ir_converter.py` post-processes to white-hot pseudo-color IR.

### Why SD 1.5 (not SDXL)?

SDXL requires >12GB VRAM. SD 1.5 runs comfortably on 8GB RTX 4060 Laptop, with ControlNet adding only ~0.7GB overhead.

### Why ControlNet-Seg (not Depth, not Canny)?

COCO-Stuff provides semantic segmentation annotations — each pixel labeled as sky/building/tree/etc. ControlNet-Seg pre-trained on ADE20k directly maps "blue region → sky" without retraining. This is the most natural fit.

### Why Structured Failure Codes?

Binary pass/fail is insufficient for closed-loop improvement. Codes like `S2_POSITION_OFFSET` or `S3_LORA_INTERFERENCE` enable precise routing to the responsible component. See `7-持续学习循环设计.md` for the full routing table.

---

## Dataset Foundation

Three public, ground-based upward-looking datasets form the data foundation:

| Dataset | Size | Modality | Use |
|:--|:--|:--|:--|
| **DroneMMset** | 320 video pairs (14.6 GB) | RGB + IR fully paired | Primary: drone LoRA training + background pool |
| **Anti-UAV-RGBT** | 318 sequences (6.3 GB) | RGB + IR paired | Background pool expansion + validation |
| **Anti-UAV410** | 410 sequences (12 GB, 438K bboxes) | Pure TIR | Downstream YOLO mAP validation |

> **Critical lesson**: Most public UAV datasets are drone-mounted (looking down at ground). We exclusively use ground-based upward-looking datasets where the drone is the **observed target**, not the **collection platform**.

### Background Pool Construction Pipeline

```
Merged Pool (RGB 18,615 / IR 18,648 frames)
    │
    ▼
pHash Deduplication → K-means Clustering (n=20) → Uniform Sampling
    │
    ▼
Manual QC (remove blur/overexposure) → BLIP Captioning
    │
    ▼
Final Training Set: IR 576 / RGB 590 frames (Kohya format)
```

---

## Models

Download from Hugging Face, place under `0-model/`:

| Model | Size | Purpose |
|:--|:--|:--|
| `stable-diffusion-v1-5` | 21 GB | Base diffusion model (VAE + UNet + Text Encoder) |
| `sd-controlnet-seg` | 2.7 GB | ControlNet segmentation conditioning |
| `clip-vit-large-patch14` | 6.4 GB | CLIP ViT-L/14 (Stage 1 verification) |
| `yolov8x.pt` | 131 MB | YOLOv8x (Stage 2 detection verification) |
| `blip-image-captioning-base` | 990 MB | BLIP captioning for training data |

```
0-model/
├── stable-diffusion-v1-5/
│   ├── vae/  unet/  text_encoder/  tokenizer/  scheduler/
├── sd-controlnet-seg/
├── clip-vit-large-patch14/
├── blip-image-captioning-base/
└── yolov8x.pt
```

---

## Code Structure

```
root/
│
├── 3-LLM starter/                     # Agent 1: LLM Semantic Parsing
│   ├── llm_parser.py                  # NL→JSON core, 4 backends
│   ├── web_app.py                     # Flask web UI (3-panel layout)
│   ├── .env                           # API keys
│   ├── templates/index.html           # Frontend
│   └── sessions/sample_session.json   # Schema reference
│
├── 4-Transformer/                     # Agent 2: Spatiotemporal Encoding
│   ├── transformer_b.py               # SpatialQueryGenerator + 5 Encoders + Fusion
│   ├── json_schema.py                 # Agent 1 Schema mirror (DroneAction, SceneSpec)
│   ├── config.py                      # Model config
│   ├── gt_generator.py                # Auto GT (Depth Anything v2 + SAM 2)
│   ├── training_pipeline.py           # 3-layer training scheduler
│   └── demo.py                        # End-to-end verification
│
├── 2-Lora training/                   # LoRA Training
│   ├── drone_target_v2/
│   │   └── drn3_uav_lora_v2.safetensors  # Drone LoRA weights
│   ├── rgb2ir_converter.py            # Agent 5: RGB→IR conversion
│   ├── train_drn3_lora.py             # LoRA training script
│   └── dataset/                       # Kohya-format training sets
│
├── 1-background-pool/                 # Preprocessed Data
│   ├── RGB_raw_frames/                # 3,771 DroneMMset frames
│   ├── IR_raw_frames/                 # 3,804 DroneMMset frames (grayscale)
│   ├── RGB_raw_frames_antiuav/        # 14,844 Anti-UAV-RGBT frames
│   ├── IR_raw_frames_antiuav/         # 14,844 Anti-UAV-RGBT frames (grayscale)
│   ├── convert_ir_to_grayscale.py     # IR color unification
│   └── drone_patches/                 # Drone bbox crops
│
├── 0-model/                           # Downloaded models (~30 GB)
├── 0-database/                        # Raw datasets
    ├── dronemmset/                    # 320 RGB+IR video pairs
    ├── Anti-UAV-RGBT/                 # 318 RGB+IR sequences
    └── Anti-UAV410/                   # 410 TIR sequences

```

---

## Agent Details

### Agent 1: LLM Semantic Parser

Converts natural language to structured 9-field JSON:

| Field | Type | Example |
|:--|:--|:--|
| `drone_type` | enum | `quadrotor` |
| `trajectory[]` | array | `[{t:0, action:"approach", distance:100, norm_u:0.5, norm_v:0.8}]` |
| `time_of_day` | enum | `dawn / morning / afternoon / dusk / night` |
| `weather` | enum | `clear / overcast / rainy / foggy / dusty / backlight` |
| `scene_type` | enum | `urban / rural / mountain / coastal / desert / forest / industrial / airfield` |
| `scene_description` | string | ControlNet-prompt-style English description |
| `modality` | enum | `RGB / IR` |
| `camera` | object | `{position, elevation_deg, fov_deg}` |
| `confidence_note` | string | Dry-run flag or LLM metadata |

Supports 4 backends: DeepSeek (default), OpenAI, Claude, Dry-Run (keyword matching). Includes Flask web UI at `127.0.0.1:5000` with session management and JSON persistence.

```bash
cd "3-LLM starter"
python3 llm_parser.py "阴天下午，四旋翼在城市上空悬停，正面拍摄"
python3 llm_parser.py --dry "晴天上午城市高楼四旋翼飞近仰拍"
python3 web_app.py  # → http://127.0.0.1:5000
```

### Agent 2: Transformer Spatiotemporal Encoder

Bridges semantic JSON to pixel-level ControlNet conditioning. Five encoder modules:

| Encoder | Input | Output |
|:--|:--|:--|
| **Position** | trajectory[].norm_u, norm_v | SpatialQueryGenerator: 256 query → 16×16 → ConvTranspose → 64×64 feature map |
| **Depth** | trajectory[].distance | Scale factor (Air 2S@50m/60°FOV/512px = 2.7px physical basis) |
| **Pose** | trajectory[].action (8 types) | Keypoint offset vectors |
| **Weather-Time** | weather (6) + time_of_day (5) | Learnable environment embedding |
| **Camera** | fov + elevation + position | Projection matrix |

**SpatialQueryGenerator** is the core innovation: 256 learnable queries are reshaped to 16×16 grid and upsampled via ConvTranspose to 64×64, preserving per-query spatial semantics. This replaces the naive mean-pooling approach that destroyed all position information.

Ground truth is auto-generated via Depth Anything v2 (depth maps) + SAM 2 (segmentation masks), enabling zero-manual-annotation training.

Training follows a 3-layer progressive strategy: Layer1 visual pretraining (3rd Anti-UAV, 58K images) → Layer2 semantic fine-tuning (DroneMMset, 7.7K pairs) → Layer3 CDFF continual evolution.

### Agent 3: ControlNet Spatial Control

Generates spatial skeleton maps (depth + segmentation) from Transformer conditioning vectors. SD1.5 + ControlNet-Seg, conditioning_scale=0.75 (optimal balance between layout fidelity and visual naturalness).

COCO-Stuff 183 classes are aggregated into 6 superclasses for ControlNet-Seg conditioning:

| Superclass | COCO-Stuff Categories | RGB Color |
|:--|:--|:--|
| sky | sky, clouds, fog | (128, 192, 255) |
| tree | tree, plant, grass, bush, flower, moss | (0, 128, 0) |
| building | building, house, skyscraper, wall, window | (128, 128, 128) |
| mountain | mountain, hill, rock | (139, 90, 43) |
| water | water, sea, river | (30, 144, 255) |
| ground | road, sidewalk, sand, dirt | (200, 180, 140) |

### Agent 4: Drone LoRA Rendering

A single LoRA (rank=16, alpha=8) trained on 98 commercial drone images at 512×512. The LoRA fills drone appearance onto the ControlNet spatial skeleton — ControlNet controls WHERE, LoRA controls WHAT.

Key training lessons:
- **Flip augmentation must be disabled** for asymmetric objects like quadcopters (rotor positions reversed on flip)
- **800 steps sufficient** for small datasets (<100 images); 6000 steps caused severe overfitting
- **Rank=16** accommodates quadcopter's 4-rotor structure better than rank=8

### Agent 5: IR Domain Conversion

Post-processing approach: `rgb2ir_converter.py` maps RGB → white-hot pseudo-color thermal IR with subtle blue tint. Chosen over direct IR generation because SD 1.5 VAE cannot encode IR grayscale images (three identical channels fall outside the VAE training manifold).

### Agent 6: 4-Stage Verification Chain

Filters generated outputs from coarse to fine:

```
S0: Background Realism → ResNet/EfficientNet binary classifier
    Trained on 576 real IR backgrounds vs generated IR backgrounds
    Failure: S0_BG_UNREALISTIC → regenerate background

S1: CLIP Semantic → CLIP ViT-L/14 viewpoint/weather alignment
    Generated image vs Agent 1 scene_description
    Failure: S1_SEMANTIC_MISMATCH → adjust LLM prompt

S2: YOLO Detection → YOLOv8 fine-tuned on Anti-UAV410
    Target detectability + position/scale accuracy
    Failure: S2_UNDETECTABLE / S2_POSITION_OFFSET → adjust LoRA/Transformer

S3: IQA Quality → BRISQUE + NIQE + artifact detection
    Blur, LoRA fusion artifacts, ControlNet boundary artifacts
    Failure: S3_BLUR / S3_ARTIFACT → adjust generation parameters
```

### Agent 7: Closed-Loop Feedback (CDFF)

Structured failure codes route to the responsible component. Three self-training paradigms:

| Paradigm | Mechanism | Used For |
|:--|:--|:--|
| **A: Pass→Train** | Verified samples → training set → standard fine-tune | LoRA, YOLO |
| **B: Fail→Fix→Contrast** | Failed sample + fixed sample → contrastive pair → CN-LoRA | ControlNet |
| **C: Rank→Align** | Batch scoring → top-50% → incremental fine-tune | IR CNN |

Each trainable component has an independent FailureBuffer. When a buffer reaches threshold (30-50 samples), an asynchronous learning loop triggers incremental fine-tuning. Generation and learning are decoupled — GPU cannot simultaneously infer and train.

See `7-持续学习循环设计.md` for the full CDFF v2.0 specification.

---

## VRAM Requirements

| Configuration | Peak VRAM |
|:--|:--|
| SD 1.5 text-to-image | ~2.2 GB |
| SD 1.5 + ControlNet-Seg | ~2.9 GB |
| LoRA training (rank=16, batch=1, FP16) | ~3.9 GB |
| Dual pipeline (ablation mode) | ~5.1 GB |
| SDXL | >12 GB |

Developed on RTX 4060 Laptop (8GB VRAM). All experiments run within this constraint.

---

## Citation

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

@inproceedings{hu2021lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {ICLR},
  year      = {2022}
}
```

---

## License

MIT License. Individual datasets and models retain their original licenses.

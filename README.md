# 0-workspace — Clean Code Archive of the LSS Generation Architecture

> This directory is the **clean code archive** of the **CDFF (Closed-loop Data Flywheel Framework)** low-slow-small (LSS) target synthetic-data generation architecture.
>
> It retains only the **final active version** of the architecture code (generation side + validation side + loop mechanism + frontend/backend),
> and **excludes**: datasets, model weights, historical versions (`_v1`–`_v6`, etc.), intermediate artifacts, and demo/test scripts.
>
> Original project root: `D:\learning\ObsidianVault\Paper-低慢小数据集生成架构`. Nothing in the original project has been deleted.

[中文版](README_CN.md)

---

## 1. Architecture Overview

CDFF is a **dual-agent adversarial + closed-loop feedback** architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENERATOR (generation side)               │
│  M1 Semantic parsing → M2 Background matching → M3 Scene extraction → M4 LoRA generation  │
│  → M5 ControlNet repainting → M6 RGB→IR conversion              │
└──────────────────────────────┬──────────────────────────────────┘
                               │  pixel-aligned RGB–IR pair + bbox
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VALIDATOR (validation side)               │
│  V1–V5  input / in-generation interception (22 rule checks)     │
│  V6–V9  output review (BRISQUE → consistency → trajectory physics → YOLO)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  structured failure codes (S6_BLUR / S8_POSITION_JUMP …)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP (loop mechanism)                  │
│  failure code → Component Router → FailureBuffer → incremental fine-tuning → weight replacement  │
│  three paradigms: A(Pass→Train) / B(Fail→Fix→Contrast) / C(Rank→Align)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
0-workspace/
├── README.md                          # this document (English)
├── README_CN.md                       # Chinese version
├── app/                               # frontend + backend (Flask service + generation orchestration)
│   ├── web_app.py                     #   Flask entry point (the only frontend)
│   ├── llm_parser.py                  #   M1 semantic parsing
│   ├── background_searcher.py         #   M2 background matching
│   ├── condition_generator_v7.py      #   M3 scene extraction + per-frame orchestration
│   ├── .env.example                   #   environment-variable template (sanitised)
│   └── templates/index.html           #   frontend page
├── generator/                         # generation-side core engine
│   ├── lora_inpainter_v7.py           #   M4 LoRA generation + M5 ControlNet repainting
│   └── rgb2ir_converter.py            #   M6 RGB→white-hot IR
├── validator/                         # validation side (V1–V9)
│   ├── v1_json_validator.py           #   V1
│   ├── v2_transformer_validator.py    #   V2 (target module deprecated, see §5)
│   ├── v3_controlnet_validator.py     #   V3
│   ├── v4_lora_validator.py           #   V4
│   ├── v5_ir_validator.py             #   V5
│   ├── v6_quality/                    #   V6 (4 files, flat import within the directory)
│   │   ├── quality_validator.py       #     V6 entry point
│   │   ├── rgb_quality.py             #     EfficientNet binary classification
│   │   ├── ir_sanity.py               #     IR signal-level ×3 checks
│   │   └── calibrate.py               #     BRISQUE threshold calibration
│   ├── v7_consistency/                #   V7 (5 files)
│   │   ├── consistency_validator.py   #     V7 entry point
│   │   ├── size_consistency.py        #     size consistency
│   │   ├── lighting_consistency.py    #     lighting consistency
│   │   ├── ir_bbox_check.py           #     IR bbox alignment
│   │   └── cross_modal_alignment.py   #     cross-modal alignment
│   ├── v8_trajectory_validator.py     #   V8 trajectory physics
│   └── v9_detection_validator.py      #   V9 YOLO detection
└── loop/                              # loop mechanism (closed-loop feedback)
    ├── validator_pipeline.py          #   S6→S9 short-circuit chaining + failure write to Buffer
    ├── failure_buffer.py              #   FailureBuffer (failure buffer pool)
    ├── trainable_classifier.py        #   paradigm A: Pass→Train trainable classifier
    └── 7-持续学习循环设计.md           #   closed-loop mechanism design spec (CDFF v2.0)
```

---

## 3. Paper Modules ↔ Code Mapping

### Generation Side (Generator, M1–M6)

| Paper module | File | Core function / class | Lines |
|---|---|---|---|
| M1 Semantic parsing | `app/llm_parser.py` | `parse()` / `parse_to_dual_json()` / `SceneSpec` | 695 |
| M2 Background matching | `app/background_searcher.py` | `search_background()` | 149 |
| M3 Scene extraction | `app/condition_generator_v7.py` | `extract_depth_matched()` / `heuristic_segment()` / `trajectory_to_frames()` | 536 |
| M4 LoRA generation | `generator/lora_inpainter_v7.py` | `LoraInpainterV7` (LoRA txt2img part) | 807 |
| M5 ControlNet repainting | `generator/lora_inpainter_v7.py` | `LoraInpainterV7.inpaint()` (depth+seg dual conditioning) | 807 |
| M6 RGB→IR | `generator/rgb2ir_converter.py` | `rgb_to_whitehot()` | 136 |

### Validation Side (Validator, V1–V9)

| Paper stage | File | Core class | # checks | Failure-code prefix |
|---|---|---|---|---|
| V1 | `validator/v1_json_validator.py` | `S1JSONValidator` | 6 | `A1_*` |
| V2 | `validator/v2_transformer_validator.py` | `S2TransformerValidator` | 6 | `A2_*` |
| V3 | `validator/v3_controlnet_validator.py` | `S3ControlNetValidator` | 4 | `A3_*` |
| V4 | `validator/v4_lora_validator.py` | `S4LoRAValidator` | 3 | `A4_*` |
| V5 | `validator/v5_ir_validator.py` | `S5IRValidator` | 3 | `A5_*` |
| V6 | `validator/v6_quality/quality_validator.py` | `QualityValidator` | 3 | `S6_*` |
| V7 | `validator/v7_consistency/consistency_validator.py` | `ConsistencyValidator` | 4 | `S7_*` |
| V8 | `validator/v8_trajectory_validator.py` | `TrajectoryValidator` | 4 | `S8_*` |
| V9 | `validator/v9_detection_validator.py` | `DetectionValidator` | 3 | `S9_*` |

### Loop Mechanism (Closed-Loop)

| File | Purpose | Corresponding README paradigm |
|---|---|---|
| `loop/failure_buffer.py` | `FailureBuffer`: failed-sample buffer + threshold triggering | shared base for the three paradigms |
| `loop/validator_pipeline.py` | `ValidatorPipeline`: S6→S9 short-circuit evaluation, failures written to Buffer | failure-code routing |
| `loop/trainable_classifier.py` | trainable EfficientNet classifier (`train`/`infer`) | paradigm A (Pass→Train) |
| `loop/7-持续学习循环设计.md` | full closed-loop mechanism spec (§6 / §9.1 / §13 / §16) | CDFF v2.0 design document |

---

## 4. Key Dependency Chain (internal call relationships)

```
web_app.py (Flask entry point)
  ├─ import llm_parser.py            → parse_to_dual_json()
  ├─ import background_searcher.py   → search_background()
  └─ import condition_generator_v7.py (imported inside the function)
        ├─ import generator/lora_inpainter_v7.py  → LoraInpainterV7.inpaint()
        │     └─ reads 0-model/stable-diffusion-v1-5 + sd-controlnet-*
        │        reads 2-Lora training/best_models/drn3_pocket_uav_v3_step2000.safetensors
        └─ import generator/rgb2ir_converter.py   → rgb_to_whitehot()

validator_pipeline.py (S6→S9 chaining)
  ├─ import v6_quality/quality_validator.py
  ├─ import v7_consistency/consistency_validator.py
  ├─ import v8_trajectory_validator.py
  ├─ import v9_detection_validator.py
  └─ import failure_buffer.py        → FailureBuffer
```

> ⚠️ **Hard-coded paths note**: the archived code retains the original project's relative-path assumptions
> (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`, `MODEL_DIR = .../0-model`,
> `POOL_ROOT = .../1-background-pool/curated_backgrounds`, `BEST_MODELS = .../2-Lora training/best_models`).
> Since this archive excludes datasets and models, **this code must be run inside the original project directory**; this directory is
> positioned as an **architecture index and code copy** for understanding, reference, and migration, rather than an independently executable environment.

---

## 5. Deduplication Notes (what this archive excludes)

| Excluded content | Original location | Reason |
|---|---|---|
| `lora_inpainter.py` ~ `_v6.py` (6 files) | `5-Controlnet/` | historical versions; the final is `_v7` |
| `demo_v5/v6/v6_batch_A.py` | `5-Controlnet/` | old demos |
| `condition_generator.py` (no suffix) | `3-LLM starter/` | old version (Gaussian blob), superseded by v7 |
| the entire `4-Transformer/` module | project root | Agent 2 spatio-temporal encoder removed from the CDFF six-module architecture |
| all demo/test scripts except `V4-trainable/` | `6-Validator/` | demo and test scripts |
| three `archive/` directories | several locations | historical archives |
| model weights / datasets / background pool / intermediate artifacts | `0-model/ 0-database/ 1-background-pool/ …` | non-code |

**Retained exception**: `loop/trainable_classifier.py` is named after V4, but it is the trainable component
(`train`/`infer` interface) of closed-loop paradigm A and belongs to the loop-mechanism design, so it is retained.

---

## 6. Running (inside the original project)

```bash
# frontend + backend (Flask)
cd "3-LLM starter"
cp .env.example .env        # fill in the API key
python web_app.py           # → http://127.0.0.1:5000

# validation-side chained pipeline (S6→S9)
cd "6-Validator/V5-pipeline/code"
python validator_pipeline.py

```

---

## 7. Original Path ↔ Archived Path Mapping

| Archived path | Original project path |
|---|---|
| `app/web_app.py` | `3-LLM starter/web_app.py` |
| `app/llm_parser.py` | `3-LLM starter/llm_parser.py` |
| `app/background_searcher.py` | `3-LLM starter/background_searcher.py` |
| `app/condition_generator_v7.py` | `3-LLM starter/condition_generator_v7.py` |
| `app/templates/index.html` | `3-LLM starter/templates/index.html` |
| `generator/lora_inpainter_v7.py` | `5-Controlnet/lora_inpainter_v7.py` |
| `generator/rgb2ir_converter.py` | `2-Lora training/rgb2ir_converter.py` |
| `validator/v1_json_validator.py` | `6-Validator/S1 Agent 1 JSON校验/code/s1_json_validator.py` |
| `validator/v2_transformer_validator.py` | `6-Validator/S2 Agent 2 Transformer校验/code/s2_transformer_validator.py` |
| `validator/v3_controlnet_validator.py` | `6-Validator/S3 Agent 3 ControlNet校验/code/s3_controlnet_validator.py` |
| `validator/v4_lora_validator.py` | `6-Validator/S4 Agent 4 LoRA校验/code/s4_lora_validator.py` |
| `validator/v5_ir_validator.py` | `6-Validator/S5 Agent 5 IR校验/code/s5_ir_validator.py` |
| `validator/v6_quality/*` | `6-Validator/S6 RGB 图像质量检查/code/*` |
| `validator/v7_consistency/*` | `6-Validator/S7 无人机与场景一致性检查/code/*` |
| `validator/v8_trajectory_validator.py` | `6-Validator/S8 轨迹物理合理性检查/code/trajectory_validator.py` |
| `validator/v9_detection_validator.py` | `6-Validator/S9 YOLO结果检查/code/detection_validator.py` |
| `loop/validator_pipeline.py` | `6-Validator/V5-pipeline/code/validator_pipeline.py` |
| `loop/failure_buffer.py` | `6-Validator/V5-pipeline/code/failure_buffer.py` |
| `loop/trainable_classifier.py` | `6-Validator/V4-trainable/code/trainable_classifier.py` |
| `loop/7-持续学习循环设计.md` | `·重点节点总结笔记/7-持续学习循环设计.md` |

---

## 8. Runnability Verification & Fix Log

After archiving, all 27 `.py` files were tested for runnability, and 4 breakages caused by
"directory reorganisation + file renaming" (`sys.path` / import breakage) were fixed. The following fixes exist
only in the archived copy and **do not modify the original project**.

### Fix list

| # | File | Original issue | Fix |
|---|---|---|---|
| 1 | `app/web_app.py` | top-level `from condition_generator import ...` (old, deprecated) crashed on startup; duplicate `/api/generate-conditions` old endpoint | removed the old import and the entire old endpoint (its function is taken over by `/api/generate-lora`) |
| 2 | `app/condition_generator_v7.py` | `sys.path` pointed at `2-Lora training/`, `5-Controlnet/` (original project paths) | repointed to the archive's `generator/` (rgb2ir + lora_inpainter) |
| 3 | `generator/lora_inpainter_v7.py` | runtime `from condition_generator_v7 import seg_to_rgb` could not find `app/` | inject `sys.path → app/` before the import |
| 4 | `loop/validator_pipeline.py` | `sys.path` pointed at `S6/S7/S8/S9/code` (original project paths) + old names `trajectory_validator` / `detection_validator` | repointed to `validator/v6_quality`, `validator/v7_consistency`, `validator/`, and switched to the new `v8_*` / `v9_*` names |

The docstring usage examples in `v1`–`v5` were also updated from the old module names (e.g. `s1_json_validator`) to the new names.

### Verification results (all passed, 2026-08-31)

| Test | Result |
|---|---|
| Syntax check (`py_compile`, 27 files) | ✓ all passed |
| Module import (18 modules, incl. cross-directory dependencies) | ✓ all passed |
| Key symbol existence (27 classes / functions) | ✓ all present |
| Pure-function smoke: `seg_to_rgb` | ✓ output shape / colour correct |
| Pure-function smoke: `rgb_to_whitehot` | ✓ white-hot IR conversion OK |
| Pure-function smoke: `trajectory_to_frames` | ✓ trajectory parsing OK |
| Flask route list (after endpoint removal) | ✓ no stale references, 11 clean routes |

> Note: the verification above covers "code structure is complete, modules load, and core pure functions run".
> **End-to-end generation (LoRA + ControlNet inference) still requires the original project's model weights
> (`0-model/`, `best_models/`) and the nine-category background pool (`1-background-pool/`), which this archive does not
> include** — so full inference cannot run independently inside the archive directory.

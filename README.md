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
├── LICENSE                            # MIT
├── requirements.txt                   # pinned dependency set
├── .gitignore
├── lss_paths.py                       # **single source of truth** for every asset path
├── experiments/                       # reproduction scripts + summary JSON (§10)
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

The tree above is the **final, consolidated pipeline** — it is what `0-workspace/` contained in
the research project, promoted to the repository root.

Nothing else is tracked. The research tree this archive was extracted from
(`0-database/`, `2-Lora training/`, `3-LLM starter/`, `4-Transformer/`,
`5-Controlnet/`, `6-Validator/`) is **not** part of the repository: it holds the
intermediate iterations, demo scripts and per-agent prototypes that the above
supersedes. This repository carries the final version only — the original
project tree on disk keeps everything.

### What is deliberately absent

| Excluded | Size in the research tree | Why |
|---|---|---|
| `0-model/` (SD1.5, 3× ControlNet, CLIP-ViT-L, YOLOv8x) | 22 GB | public weights, re-downloadable |
| `1-background-pool/` (9 scenes, 37,852 files) | 11 GB | the dataset itself |
| `0-database/` | 304 MB | image corpora, YOLO labels, tfevents; the configs it held are superseded |
| `2-Lora training/` | 344 MB | LoRA training harness; the adapter it produced is a weight, not code |
| `3-LLM starter/`, `5-Controlnet/` | 9 MB | M1–M5 prototypes (`lora_inpainter` v1→v6, every demo); superseded by `app/` + `generator/` |
| `4-Transformer/` | 1.8 MB | the Depth-Anything-V2 trajectory branch, abandoned in v20 |
| `6-Validator/` | 3.8 MB | the nine stand-alone agent trees before they were flattened into `validator/` |
| `experiments/**/{manifest.json, samples/, crops/, _scratch/}` | 63 GB | per-sample dumps, regenerable |
| `·-1豆包案例图/`, `5-Controlnet/demo_output*/` | 82 MB | reference and generated imagery |
| `z-文件格式参考/` | 7 MB | third-party papers, not ours to redistribute |
| `·重点节点总结笔记/`, `参考资料/` | 21 MB | research notes, task lists, 12 manuscript drafts, tutor-feedback audits |

Everything a reader needs to *rebuild* the pipeline is here; everything that is either
redistributable from upstream, regenerable from the tracked scripts, or superseded by a
later version of the same module is not.

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

## 5. Deduplication Notes (one file per module, no versioned siblings)

Within the tracked tree the audit found no duplicate function bodies and no versioned
siblings: every module appears exactly once, in its final form. The iterations that led there
are **not** carried — they remain in the original project tree only:

| Not carried | Where it still lives | Why it was dropped |
|---|---|---|
| `lora_inpainter.py` ~ `_v6.py` (6 files) | `5-Controlnet/` | superseded by `generator/lora_inpainter_v7.py` |
| `demo_v5/v6/v6_batch_A.py` | `5-Controlnet/` | old demos; `v7` is the shipped path |
| `condition_generator.py` (no suffix) | `3-LLM starter/` | Gaussian-blob predecessor of v7 |
| the whole Transformer branch | `4-Transformer/` | Agent 2 spatio-temporal encoder removed in v20 |
| per-agent `tests/` and demo scripts | `6-Validator/S*/` | flattened into `validator/v1…v9` |
| the `archive/` directories | `5-Controlnet/`, `4-Transformer/` | exploratory lines |
| `kohya_dataset/` configs, LoRA training harness | `0-database/`, `2-Lora training/` | setup, not pipeline code |

Two consequences worth knowing when reading the code:

- **`app/llm_parser.py` is the single parser.** There is no second copy to drift against; the
  evaluation scripts under `experiments/llm_eval/` import this one.
- **`validator/v6_quality/` computes BRISQUE through `piq`**, which ships its own weights. The
  115 KB SVM head under `6-Validator/S6 …/models/` was only used by the earlier stand-alone
  agent and is not needed at runtime.

**Retained exception**: `loop/trainable_classifier.py` is named after V4, but it is the trainable
component (`train`/`infer` interface) of closed-loop paradigm A and belongs to the loop-mechanism
design, so it is retained.

---

## 6. Running

> **An auditable archive, not a turnkey pipeline.** Every number in the paper is reproducible from a bare
> clone (41/41, no weights, no GPU, no network). Generating a new sample is a different promise: it needs
> the assets below, and even fully provisioned a fresh sample still fails S7 on the size check (§9).
> Keep "recompute the tables" and "regenerate the dataset" apart in your own writing.

All asset paths are resolved by `lss_paths.py`, the single source of truth:

1. `LSS_PROJECT_ROOT` environment variable, if set;
2. otherwise the target of the `0-model` / `1-background-pool` / `2-Lora training` symlinks
   (this is how the archive works in place inside the original project tree);
3. otherwise the archive directory itself (standalone checkout).

```bash
# check the resolution first — prints every path with an OK/MISSING marker
python lss_paths.py

# frontend + backend (Flask)
cd app
cp .env.example .env        # fill in the API key
python web_app.py           # → http://127.0.0.1:5000

# validation side — each validator is a CLI of its own
python validator/v6_quality/quality_validator.py  IMG.png --ir IR.png
python validator/v7_consistency/consistency_validator.py IMG.png --ir IR.png --gt gt.json
python validator/v8_trajectory_validator.py traj.json
python validator/v9_detection_validator.py IMG.png --gt bboxes.json --model /path/to/yolov8x.pt

# reproduce every number in the manuscript
python experiments/bcd/_reports/scripts/audit_tables_v20.py
```

`loop/validator_pipeline.py` is a **library**, not a script: it has no `__main__`, so running it
directly does nothing. Use `ValidatorPipeline(s0, s1, s2, s3).validate(sample)` — `sample` is
`{"id", "rgb", "ir", "rgb_bbox", "ir_bbox", "gt_bboxes", "drone_category", "distance_m"}`, images as
`np.ndarray`. Note the two bbox conventions: `rgb_bbox` / `ir_bbox` are **normalised `[cx, cy, w, h]`**
(pixel `[x1, y1, x2, y2]` is rejected as a size mismatch), while `gt_bboxes` is **pixel `[x1, y1, x2, y2]`**.
`v7 --gt` therefore wants `{"rgb_bbox": [cx, cy, w, h], "drone_category": ..., "distance_m": ...}` and
`v9 --gt` wants a bare list of boxes `[[x1, y1, x2, y2]]`.

**Standalone checkout** (no symlinks, weights stored elsewhere):

```bash
export LSS_PROJECT_ROOT=/path/to/project   # must contain 0-model/, 1-background-pool/, 2-Lora training/
python lss_paths.py                        # verify before running anything
```

Full inference needs more than this repository carries — ~12 GB, none of it tracked here:

| Asset | Location (`lss_paths.py` resolves it) | How to get it | Size |
|---|---|---|---|
| SD 1.5 base | `0-model/stable-diffusion-v1-5` | `hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir <path>` | 6.1 G |
| ControlNet depth | `0-model/sd-controlnet-depth` | `hf download lllyasviel/sd-controlnet-depth --local-dir <path>` | 2.7 G |
| ControlNet seg | `0-model/sd-controlnet-seg` | `hf download lllyasviel/sd-controlnet-seg --local-dir <path>` | 2.7 G |
| LoRA | `2-Lora training/best_models/drn3_pocket_uav_v3_step2000.safetensors` | train your own | 67 M |
| background pool | `1-background-pool/curated_backgrounds/`, nine classes | collect your own | 11 G |
| yolov8x (V9 only) | `0-model/yolov8x.pt` | `python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"` | 131 M |

The first three must be **diffusers directory snapshots**, not single `.safetensors` files — the code calls
`from_pretrained(str(path))` on each. Behind the GFW, `export HF_ENDPOINT=https://hf-mirror.com` makes all
of the above work unchanged; on a legacy CLI swap `hf download` for `huggingface-cli download`.

**Two checkpoints that must be in the HF cache, or the run dies silently** — they are loaded by repo id
with `local_files_only=True`, so pointing a local directory at them does nothing:

```bash
hf download lllyasviel/Annotators --include "dpt_hybrid-midas-501f0c75.pt"   # ~470 MB
hf download openai/clip-vit-base-patch32                                     # ~605 MB
```

Without the first, generation dies at depth extraction (before ControlNet); without the second, at the
repaint step. The error is `OSError: ... is not a local folder and is not a valid model identifier`, which
reads like a code bug and is not. `0-model/clip-vit-large-patch14` in the project tree (6.4 G) is a
*different* checkpoint, does not satisfy the second, and can be deleted.

**The LoRA is an extension point, not an asset.** The paper's claim is that *any* LoRA trained on this
dataset can be dropped in, which is why none ships here. The contract:

- kohya / `sd-scripts` safetensors, keys `lora_unet_*.lora_down.weight` / `.lora_up.weight` / `.alpha`;
- `generator/lora_inpainter_v7.py::_merge_lora` merges the **UNet only** and reads `lora_unet_`-prefixed
  keys; anything aimed at the text encoder (`lora_te_*`) is **skipped silently** — such a LoRA loads fine
  and changes almost nothing;
- base must be **SD 1.5**; SDXL / SD3 LoRAs do not match the keys;
- the path is hard-coded to one filename. To use your own: overwrite that file, or set
  `LSS_PROJECT_ROOT=<another tree>`, or edit the single line `lss_paths.LORA_PATH`;
- debug order: run with `use_lora=False` until the base model chain works, then suspect the LoRA.
  Training scripts and the training set are not in this repository.

**Where the background pool comes from**: there is no public dataset for it, and this repository
ships no collector — `app/background_searcher.py` only **searches**, it does not download. Build your own:
one directory per class (`airfield` / `coastal` / `desert` / `forest` / `industrial` / `mountain` /
`pure_sky` / `rural` / `urban`), 16:9 photographs inside, plus a `_metadata.jsonl` per directory, one
record per line:

```json
{"filename": "bg_00070a0c1b49.jpg", "url": "https://…", "source": "unsplash",
 "photographer": "…", "category": "pure_sky", "time_of_day": "dusk", "weather": "clear"}
```

Valid values are `time_of_day ∈ {dawn, morning, afternoon, dusk, night}` and
`weather ∈ {clear, overcast, rainy, foggy, dusty, backlight}`. The searcher scores time-of-day × weather
with a four-level fallback and reports which level hit in `match_level`; `BG_MATCH=random` restores pure
random selection. The paper's pool is 37,852 images — a scale reference, not a requirement; a few dozen
will exercise the pipeline, just with less variety.

**How the LoRA was trained**: the training script and dataset are not in this repository, but the
released weights carry the full recipe in their `__metadata__` (native kohya `sd-scripts` fields).
Reproducing it:

| Item | Value |
|---|---|
| Base | SD 1.5 (`ss_sd_model_name = 451f4fe1…`), `--v1` |
| Data | 49 `pocket_uav` images, `num_repeats = 10` → 490 per epoch |
| Resolution | 512×512, `enable_bucket = false` |
| Network | `networks.lora`, `network_dim = 8`, `network_alpha = 4` |
| Optimiser | AdamW8bit, lr `1e-4`, `cosine_with_restarts`, warmup 0, grad ckpt, fp16, bs 1 |
| Steps | capped at 3000; the shipped checkpoint is step 2000 (epoch 4) |
| Other | `min_snr_gamma = 5.0`, `multires_noise_discount = 0.3`, seed 42 |

```bash
git clone https://github.com/kohya-ss/sd-scripts && cd sd-scripts
accelerate launch --num_cpu_threads_per_process 8 train_network.py \
  --pretrained_model_name_or_path <SD1.5 dir> \
  --train_data_dir <dataset root> \
  --network_module networks.lora --network_dim 8 --network_alpha 4 \
  --resolution 512,512 --enable_bucket=false \
  --learning_rate 1e-4 --optimizer_type AdamW8bit \
  --lr_scheduler cosine_with_restarts --lr_warmup_steps 0 \
  --min_snr_gamma 5 --multires_noise_discount 0.3 \
  --max_train_steps 3000 --save_every_n_epochs 1 \
  --mixed_precision fp16 --gradient_checkpointing --seed 42 \
  --output_dir <out dir> --output_name drn3_pocket_uav
```

Dataset layout: images plus same-named `.txt` captions inside `<dataset root>/10_Pocket_UAV/` — **the
`10_` prefix is the `num_repeats`**. Captions are a trigger word plus structured attributes; the actual
token distribution in the shipped file is `pocket_uav, a compact foldable drone with a small body and
foldable arms, arms unfolded, hovering, front view`. Output is 792 keys (576 `lora_unet_*` + 216
`lora_te_*`); the 216 text-encoder ones are not merged by this repository's loader — see the contract above.

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

The right-hand column names a directory that is **not** in this repository (the exclusion table in §2); it is where
the file came from in the original project tree, which is unchanged on disk. The mapping is
documentation of provenance, not a set of links you can follow here.

Two files are renamed rather than moved: `validator/v8_trajectory_validator.py` and
`validator/v9_detection_validator.py` drop the `s8_`/`s9_` prefix their originals carried.

---

## 8. Runnability Verification & Fix Log

After archiving, all 29 `.py` files of the final pipeline were tested for runnability, and 6 breakages caused by
"directory reorganisation + file renaming + the archive rewrite" (`sys.path` / import breakage) were fixed. The following fixes exist
only in the archived copy and **do not modify the original project**.

### Fix list

| # | File | Original issue | Fix |
|---|---|---|---|
| 1 | `app/web_app.py` | top-level `from condition_generator import ...` (old, deprecated) crashed on startup; duplicate `/api/generate-conditions` old endpoint | removed the old import and the entire old endpoint (its function is taken over by `/api/generate-lora`) |
| 2 | `app/condition_generator_v7.py` | `sys.path` pointed at `2-Lora training/`, `5-Controlnet/` (original project paths) | repointed to the archive's `generator/` (rgb2ir + lora_inpainter) |
| 3 | `generator/lora_inpainter_v7.py` | runtime `from condition_generator_v7 import seg_to_rgb` could not find `app/` | inject `sys.path → app/` before the import |
| 4 | `loop/validator_pipeline.py` | `sys.path` pointed at `S6/S7/S8/S9/code` (original project paths) + old names `trajectory_validator` / `detection_validator` | repointed to `validator/v6_quality`, `validator/v7_consistency`, `validator/`, and switched to the new `v8_*` / `v9_*` names |

| 5 | `lss_paths.py` (**new**) + 5 call sites | asset paths (`../1-background-pool`, `../0-model`, `../2-Lora training`) were re-derived in every module from its own `__file__`, so the tree only worked through three symlinks (`0-model`, `1-background-pool`, `2-Lora training`) pre-created inside the archive | added `lss_paths.py` as the **single source of truth**; `background_searcher.py`, `condition_generator_v7.py`, `web_app.py`, `lora_inpainter_v7.py` and `validator_pipeline.py` now import from it. Resolution order: `LSS_PROJECT_ROOT` env var → symlink target of the three asset directories → the archive directory itself |

| 6 | `app/condition_generator_v7.py` | the `sys.path` rewrite in fix #2 **dropped** `from rgb2ir_converter import rgb_to_whitehot` — the module was added to `sys.path` but its symbol never imported, so every run reached the IR step (after ~4 min of ControlNet) and died with `NameError: name 'rgb_to_whitehot' is not defined`. Found 2026-10-30 by actually generating a sample, not by importing modules | re-added the import after the path search |

The docstring usage examples in `v1`–`v5` were also updated from the old module names (e.g. `s1_json_validator`) to the new names.

### Verification results (all passed, 2026-08-31)

| Test | Result |
|---|---|
| Syntax check (`py_compile`, 29 files) | ✓ all passed |
| Module import (18 modules, incl. cross-directory dependencies) | ✓ all passed |
| Key symbol existence (27 classes / functions) | ✓ all present |
| Pure-function smoke: `seg_to_rgb` | ✓ output shape / colour correct |
| Pure-function smoke: `rgb_to_whitehot` | ✓ white-hot IR conversion OK |
| Pure-function smoke: `trajectory_to_frames` | ✓ trajectory parsing OK |
| Flask route list (after endpoint removal) | ✓ no stale references, 11 clean routes |

The 2026-08-31 pass above only proved "modules load and pure functions run". A second pass on 2026-10-30
went through the whole pipeline on a real GPU (RTX 4060 Laptop, torch 2.6.0+cu124), inside the original
project tree where the weights live — this is what turned up fix #6:

| Test | Result |
|---|---|
| `python lss_paths.py` from `0-workspace/` | ✓ 8/8 assets resolve (POOL_ROOT, SD15, 2× ControlNet, LoRA) |
| `generate_with_lora` on one `pure_sky` frame, 60 m, 1 drone | ✓ `ok=True`, 11 artifacts (bg, bg_depth, bg_seg, drone_raw, drone_clean, drone_rgba, drone_depth, drone_seg, final_rgb, ir, bbox) |
| Wall clock, single frame | 144 s cold (model load included), ≈30 s warm |
| `app/smoke_endpoints.py` (33 assertions, every route stubbed) | ✓ 33/33 |
| `v6 quality_validator` on the generated frame | ✓ PASS |
| `v7 consistency_validator` | ✓ runs — lighting PASS, size FAIL (see below) |
| `v8 trajectory_validator` on a 3-frame sequence | ✓ PASS, score 1.000 |
| `v9 detection_validator` with `yolov8x.pt` | ✓ runs, 8 predictions, gt box missed (COCO has no UAV class) |
| `audit_tables_v20.py` from a bare clone | ✓ 41/41 tables match the manuscript |

> **One open calibration gap, recorded rather than fixed.** The generator sizes the drone by an empirical
> rule (`size_px = clip(3600 / max(distance, 25), 12, 96)` → 60 px at 60 m), whereas `v7`'s size check
> derives the expected width from physics (`real_w / (2·d·tan(fov/2))` → 4.6 px for a 0.25 m quadcopter
> at 60 m). The two differ by ~13×, so a freshly generated sample fails `S7` on size unless
> `drone_category` names an airframe on the order of 3 m. The results in the manuscript are unaffected —
> they are recomputed from committed per-sample JSON — but **regenerating data end-to-end will hit this**.
> It is a modelling decision, not a code defect, so it is left as is.

---

## 9. Paper ↔ Code Claim Audit (manuscript v20)

Every number in the manuscript was recomputed from the raw per-sample JSON with
`experiments/bcd/_reports/scripts/audit_tables_v20.py` (43 assertions, `Decimal` +
`ROUND_HALF_UP`):

| Table | Content | Result |
|---|---|---|
| Table 3 | 300-pair benchmark: 1.226±0.448 px, scale 0.028±0.021, IoU 0.706±0.126, 94.0% @IoU≥0.5, BRISQUE 20.55, 88.3% pass rate | ✓ 43/43 |
| Table 4 | E1 boundary: LCR 1.338 vs 1.655 (p=4.9e-05), RLJ / BGE null | ✓ |
| Table 5 | Ablation E2 / E3 (merged vs disabled) | ✓ |
| Table 6 | Closed loop A1 / A2 / A3 | ✓ |
| Table 7 | Downstream proxy, 8 arms | ✓ |
| Table 8 | LLM vs template parsing, 7 metrics | ✓ |
| Table 2 | Throughput 14.58 s mean / 14.57 s median | ✓ |
| §3.1 | Background pool 37,852 images over 9 categories | ✓ recounted from disk |

Implementation parameters quoted in the manuscript were checked against the code defaults:

| Claim (manuscript) | Code default | Status |
|---|---|---|
| `N = 15`, `s = 0.65` | `steps=15, strength=0.65` (`lora_inpainter_v7.py`) | ✓ |
| both conditions at scale `0.85` | `cnet_scales=(0.85, 0.85)` | ✓ |
| guidance `7.5` | `guidance=7.5` | ✓ |
| BRISQUE acceptance threshold `45` | `brisque_thr=45.0` (`v6_quality/quality_validator.py`) | ✓ |
| V9: IoU `0.5` at detector confidence `0.25` | `iou_thr=0.5, conf_thr=0.25` (`v9_detection_validator.py`) | ✓ |
| V1–V9, nine stage-wise checks | `validator/v1…v9` | ✓ |
| sliding exclusion queue | `app/probe_exclude_queue.py` + exclusion queue in `background_searcher.py` | ✓ |
| three failure buffers routed V4/V6, V3/V7, V1/V8 at 50 / 30 / 20 | `loop/failure_buffer.py` implements **one** parameterised buffer (`threshold=50`); the pipeline instantiates a single instance | ⚠ partial — routing layer not implemented |
| Eq. (12) acceptance-rate rollback below 10 % | not implemented anywhere in the project (`grep -ri rollback/acceptance_rate` = 0 hits) | ⚠ not implemented (design constraint) |

The last two rows are stated in Section 3.3 as part of the method. They are **design
constraints, not shipped code**: the archive implements the buffer mechanism itself, the
deterministic V3/V5/V8 rules and the stage-wise short-circuit, but neither the three-way
buffer routing nor the acceptance-rate rollback. The manuscript already frames the five
anchors as "a design argument rather than a demonstrated guarantee"; the same caveat applies
to these two mechanisms.

---

## 10. Experiments (reproduction artefacts)

`experiments/` contains the 50 scripts that produced the manuscript and the summary
JSON/Markdown of each run (4.5 MB). Per-sample `manifest.json`, YOLO training dumps and the
intermediate `_smoke*/`, `_v8_prework/`, `_v9_prehalofix/`, `*_buggy_*`, `*_stale_*`
directories are excluded.

```
experiments/
├── bcd/            # main benchmark: out_B0, out_B_A1..A3, out_D, out_E1..E6,
│                   # out_downstream, out_fusion_e1e4, _reports/scripts/audit_tables_v20.py
├── dataset_v9/     # v9 dataset export + annotations
├── llm_eval/       # LLM-vs-template parsing evaluation (Table 8)
├── p1/ step_sweep/ # parameter sweeps
└── new test/
```

To recompute every table from the raw artefacts:

```bash
python experiments/bcd/_reports/scripts/audit_tables_v20.py
```

`audit_tables_v20.py` reads only the JSON already in the repository, so it runs from a bare
checkout. The scripts that *generated* those JSONs (`p1/`, `step_sweep/`, `bcd/run_*`) are a
record of how the numbers were obtained: they resolve paths against the original project root
and need `0-model/`, `1-background-pool/` and the stage trees to be present. Run them there,
not here.

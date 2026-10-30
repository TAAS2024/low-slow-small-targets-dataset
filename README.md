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

### 2.1 Research-stage trees (kept for provenance)

The pipeline did not appear in one go. Each numbered directory below is the stage tree it was
actually built in, kept verbatim so every design decision in the paper can be traced back to the
code that produced it. These trees are *history*, not part of the runtime path — the final modules
in `app/`, `generator/`, `validator/`, `loop/` supersede them.

```
0-database/              # dataset assembly (image corpora + kohya configs)
│                        #   only the configs are tracked — the image corpora are not
├── README.md
├── kohya_dataset/training_config.toml
├── kohya_dataset/11_drn3_pocket_uav/dataset_config.toml
└── kohya_dataset/sample_prompts.txt
2-Lora training/         # M0: drone LoRA training (the adapter the generator loads)
├── train_drn3_lora.py           #   kohya/sd-scripts wrapper, v3-768 recipe
├── training_config_v3_768.toml  #   rank / lr / steps actually used
├── collect_drone_images.py      #   corpus harvesting
└── rgb2ir_converter.py          #   prototype of M6
3-LLM starter/           # M1–M3 prototypes (pre-refactor, v7 lives here first)
├── web_app.py, llm_parser.py, background_searcher.py, condition_generator_v7.py
├── condition_generator.py       #   v6 predecessor, kept to show the refactor delta
├── e2e_webcheck.py              #   end-to-end HTTP smoke test + _e2e_report.json
├── templates/index.html
└── sessions/sample_session.json
4-Transformer/           # M2.5: Depth-Anything-V2 trajectory branch (deprecated in v20)
├── smoke_test.py, README.md
└── archive/                     #   transformer_b.py, training_pipeline.py, json_schema.py,
                                 #   gen_jsons_*.py, demo_*.py — the full exploratory line
5-Controlnet/            # M4/M5 engine, every iteration v1→v7
├── lora_inpainter.py … lora_inpainter_v7.py    # generator evolution, one file per version
├── demo_v5.py, demo_v6.py, demo_v6_batch_A.py, demo_v7.py
├── diagnose_pipeline.py                        #   NaN/black-frame triage tool
├── test_blend_v2.py, test_blend_strength_comparison.py
└── archive/                                    #   dual_controlnet_pipeline.py, full_pipeline.py…
6-Validator/             # V1–V9, built as nine independent agents before being flattened
├── S1 Agent 1 JSON校验/ … S9 YOLO结果检查/    # each: code/ config/ tests/ demos/
├── V4-trainable/, V5-pipeline/                 #   paradigm-A and chaining prototypes
├── Demo S1~5/, Demo S6~9/                      #   cross-agent demos + failure logs
└── S6 …/models/brisque_svm_weights.pt          #   115 KB SVM head, tracked (V6 needs it)
```

The research notes, task lists, manuscript drafts and tutor-feedback audits that sat next to these
trees (`·重点节点总结笔记/`, `参考资料/`) are deliberately **not** in the repository — they are
working documents, not architecture.

### 2.2 What is deliberately absent

| Excluded | Size in the research tree | Why |
|---|---|---|
| `0-model/` (SD1.5, 3× ControlNet, CLIP-ViT-L, YOLOv8x) | 22 GB | public weights, re-downloadable |
| `1-background-pool/` (9 scenes, 37,852 files) | 11 GB | the dataset itself |
| `0-database/{11_drone_clean, Drone_pic, 官方数据集, lora_logs_v3_768}` | 304 MB | image corpora + YOLO labels + tfevents |
| `2-Lora training/{best_models, demo_output*, drn3_pocket_uav}` | 344 MB | trained LoRA adapters (`.safetensors`) |
| `experiments/**/{manifest.json, samples/, crops/, _scratch/}` | 63 GB | per-sample dumps, regenerable |
| `·-1豆包案例图/`, `5-Controlnet/demo_output*/` | 82 MB | reference and generated imagery |
| `z-文件格式参考/` | 7 MB | third-party papers, not ours to redistribute |
| `·重点节点总结笔记/`, `参考资料/` | 21 MB | research notes, task lists, 12 manuscript drafts, tutor-feedback audits — working documents |

Everything a reader needs to *rebuild* the pipeline is here; everything that is either
redistributable from upstream or regenerable from the tracked scripts is not.

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

## 5. Deduplication Notes (why the final tree is lean)

The repository carries the pipeline **twice**, on purpose:

- **§2 final tree** (`app/`, `generator/`, `validator/`, `loop/`) — one file per module, no
  duplicated implementations. This is the only tree that runs.
- **§2.1 stage trees** (`5-Controlnet/`, `3-LLM starter/`, `6-Validator/`, …) — the iterations
  each module went through, kept verbatim. Nothing here is imported at runtime.

So the duplication is deliberate and partitioned. Within the *final* tree the audit found no
duplicate function bodies and no versioned siblings:

| Not carried into the final tree | Where it still lives verbatim | Why it was dropped from the final tree |
|---|---|---|
| `lora_inpainter.py` ~ `_v6.py` (6 files) | `5-Controlnet/` | superseded by `generator/lora_inpainter_v7.py` |
| `demo_v5/v6/v6_batch_A.py` | `5-Controlnet/` | old demos; `v7` is the shipped path |
| `condition_generator.py` (no suffix) | `3-LLM starter/` | Gaussian-blob predecessor of v7 |
| the whole Transformer branch | `4-Transformer/` | Agent 2 spatio-temporal encoder removed in v20 |
| per-agent `tests/` and demo scripts | `6-Validator/S*/` | flattened into `validator/v1…v9` |
| the three `archive/` directories | `5-Controlnet/`, `4-Transformer/` | exploratory lines |

**Retained exception**: `loop/trainable_classifier.py` is named after V4, but it is the trainable
component (`train`/`infer` interface) of closed-loop paradigm A and belongs to the loop-mechanism
design, so it is retained.

---

## 6. Running

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

# validation-side chained pipeline (V6→V9)
cd loop
python validator_pipeline.py

# reproduce every number in the manuscript
python experiments/bcd/_reports/scripts/audit_tables_v20.py
```

**Standalone checkout** (no symlinks, weights stored elsewhere):

```bash
export LSS_PROJECT_ROOT=/path/to/project   # must contain 0-model/, 1-background-pool/, 2-Lora training/
python lss_paths.py                        # verify before running anything
```

> Full inference needs the model weights (`0-model/stable-diffusion-v1-5`, the two ControlNet
> checkpoints and `2-Lora training/best_models/drn3_pocket_uav_v3_step2000.safetensors`) plus the
> nine-category background pool (37,852 images). None of them are in this repository.
>
> **loop design document**: `loop/7-持续学习循环设计.md`

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

Every "original project path" in this table that lies under a tracked stage tree (§2.1) can be
cross-checked directly in the repository. The design note `7-持续学习循环设计.md` is the one
exception: its research-tree original is a working document and lives outside git, so only the
curated copy in `loop/` is versioned.

---

## 8. Runnability Verification & Fix Log

After archiving, all 30 `.py` files were tested for runnability, and 5 breakages caused by
"directory reorganisation + file renaming" (`sys.path` / import breakage) were fixed. The following fixes exist
only in the archived copy and **do not modify the original project**.

### Fix list

| # | File | Original issue | Fix |
|---|---|---|---|
| 1 | `app/web_app.py` | top-level `from condition_generator import ...` (old, deprecated) crashed on startup; duplicate `/api/generate-conditions` old endpoint | removed the old import and the entire old endpoint (its function is taken over by `/api/generate-lora`) |
| 2 | `app/condition_generator_v7.py` | `sys.path` pointed at `2-Lora training/`, `5-Controlnet/` (original project paths) | repointed to the archive's `generator/` (rgb2ir + lora_inpainter) |
| 3 | `generator/lora_inpainter_v7.py` | runtime `from condition_generator_v7 import seg_to_rgb` could not find `app/` | inject `sys.path → app/` before the import |
| 4 | `loop/validator_pipeline.py` | `sys.path` pointed at `S6/S7/S8/S9/code` (original project paths) + old names `trajectory_validator` / `detection_validator` | repointed to `validator/v6_quality`, `validator/v7_consistency`, `validator/`, and switched to the new `v8_*` / `v9_*` names |

| 5 | `lss_paths.py` (**new**) + 5 call sites | asset paths (`../1-background-pool`, `../0-model`, `../2-Lora training`) were re-derived in every module from its own `__file__`, so the tree only worked through three symlinks (`0-model`, `1-background-pool`, `2-Lora training`) pre-created inside the archive | added `lss_paths.py` as the **single source of truth**; `background_searcher.py`, `condition_generator_v7.py`, `web_app.py`, `lora_inpainter_v7.py` and `validator_pipeline.py` now import from it. Resolution order: `LSS_PROJECT_ROOT` env var → symlink target of the three asset directories → the archive directory itself |

The docstring usage examples in `v1`–`v5` were also updated from the old module names (e.g. `s1_json_validator`) to the new names.

### Verification results (all passed, 2026-08-31)

| Test | Result |
|---|---|
| Syntax check (`py_compile`, 30 files) | ✓ all passed |
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

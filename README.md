# CDFF — Closed-loop Data Flywheel Framework for LSS Target Dataset Generation

English | [中文](README_CN.md)

A synthetic data generation framework for low-slow-small (LSS) aerial targets, built on two adversarial agents and a closed feedback loop. Every sample has its target position, scale and bounding box fixed before rendering, and verified afterwards by an independent validator.

Paper: *A Spatially Controlled Closed-Loop Framework for Synthetic Low-Slow-Small Target Dataset Generation* (under review, Neurocomputing).

---

## 1. The problem

Low-slow-small targets — small multi-rotors, tethered kites, aerostats, balloons — occupy 15×15 to 30×30 pixels at 200–500 m stand-off ranges. Minimal radar cross-section, weak acoustic emission and faint thermal signature make real data expensive to collect. Generating training data with diffusion models looks like a shortcut, until two obstacles appear.

**Geometry cannot be controlled unattended.** Layout-conditioned models need a hand-prepared condition map per sample; prompt conditioning gives no trustworthy coordinates or scale. The generator does not know where the target should land or how large it should be, so there is no box to record.

**There is no criterion for which samples to keep.** Perceptual metrics (BRISQUE, ImageReward) are computed on the finished composite and remain decoupled from the geometric consistency a detector needs: an image can be photorealistic and still violate perspective or scale.

CDFF treats these as one problem: turn an ordinary background photograph into a sample whose geometry is declared up front and can be checked afterwards.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        GENERATOR                              │
│  M1 parse → M2 retrieve → M3 depth/affordance → M4 markers    │
│  → M5 LoRA synthesis → M6 ControlNet repaint → M7 pseudo-IR   │
└───────────────────────────────┬──────────────────────────────┘
                                │ pixel-aligned RGB–IR pair + box
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                        VALIDATOR                              │
│  V1–V5 input and in-generation gates   V6–V9 output review    │
└───────────────────────────────┬──────────────────────────────┘
                                │ structured failure codes
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                      CLOSED LOOP                              │
│  code → component router → FailureBuffer → update → swap      │
└──────────────────────────────────────────────────────────────┘
```

The generator derives spatial conditioning from the background itself. Monocular relative depth assigns insertion depth, a rule-based affordance map supplies the semantic condition, and marker injection writes both into the ControlNet condition maps. The target footprint is fixed analytically from the declared stand-off range rather than read from the depth estimate — precisely because it is analytic, the realised box can be scored against it later.

The validator is decoupled so that no module scores its own output. Nine stages run in order, short-circuiting on success and returning a code naming the stage that failed.

The loop routes failures back to the module that produced them, under three paradigms: Pass→Train, Fail→Fix→Contrast, Rank→Align.

---

## 3. Repository layout

```
.
├── lss_paths.py                    # asset path resolution (single source)
├── app/
│   ├── web_app.py                  # Flask entry point
│   ├── llm_parser.py               # M1 scene description → schema
│   ├── background_searcher.py      # M2 retrieval, four-tier fallback
│   └── condition_generator_v7.py   # M3 priors + per-frame orchestration
├── generator/
│   ├── lora_inpainter_v7.py        # M4/M5 LoRA synthesis + repaint
│   └── rgb2ir_converter.py         # M6 RGB → pseudo-IR
├── validator/                      # V1–V9
│   ├── v1_json_validator.py
│   ├── v2_transformer_validator.py
│   ├── v3_controlnet_validator.py
│   ├── v4_lora_validator.py
│   ├── v5_ir_validator.py
│   ├── v6_quality/                 # perceptual quality + IR signal
│   ├── v7_consistency/             # size / lighting / IR box / cross-modal
│   ├── v8_trajectory_validator.py
│   └── v9_detection_validator.py
├── loop/
│   ├── validator_pipeline.py       # S6→S9 short-circuit chain
│   ├── failure_buffer.py           # buffer and threshold trigger
│   └── trainable_classifier.py     # paradigm A classifier
└── experiments/                    # reproduction scripts and summaries
```

---

## 4. Paper ↔ code

| Paper module | File | Entry point |
|---|---|---|
| M1 scene parsing | `app/llm_parser.py` | `parse_to_dual_json()` |
| M2 background retrieval | `app/background_searcher.py` | `search_background()` |
| M3 depth / affordance | `app/condition_generator_v7.py` | `extract_depth_matched()` / `heuristic_segment()` |
| M4 marker injection + LoRA | `generator/lora_inpainter_v7.py` | `LoraInpainterV7._generate()` |
| M5 ControlNet repaint | `generator/lora_inpainter_v7.py` | `LoraInpainterV7.inpaint()` |
| M6 RGB → pseudo-IR | `generator/rgb2ir_converter.py` | `rgb_to_whitehot()` |

| Stage | File | Checks | Failure code |
|---|---|---|---|
| V1 JSON structure | `validator/v1_json_validator.py` | 6 | `A1_*` |
| V2 Transformer | `validator/v2_transformer_validator.py` | 6 | `A2_*` |
| V3 ControlNet condition | `validator/v3_controlnet_validator.py` | 4 | `A3_*` |
| V4 LoRA sampling | `validator/v4_lora_validator.py` | 3 | `A4_*` |
| V5 infrared channel | `validator/v5_ir_validator.py` | 3 | `A5_*` |
| V6 perceptual quality | `validator/v6_quality/quality_validator.py` | 3 | `S6_*` |
| V7 consistency | `validator/v7_consistency/consistency_validator.py` | 4 | `S7_*` |
| V8 trajectory physics | `validator/v8_trajectory_validator.py` | 4 | `S8_*` |
| V9 detection | `validator/v9_detection_validator.py` | 3 | `S9_*` |

Loop components: `failure_buffer.py` (accumulation and threshold), `validator_pipeline.py` (short-circuit chain), `trainable_classifier.py` (paradigm A).

---

## 5. Installation

```bash
git clone https://github.com/TAAS2024/low-slow-small-targets-dataset.git
cd low-slow-small-targets-dataset
pip install -r requirements.txt
```

Requires PyTorch (CUDA optional), diffusers, transformers, ControlNet, `piq`, ultralytics and Flask. The per-pair timing reported in the paper was measured on a single 8 GB GPU.

---

## 6. Models and assets

Weights and images are not distributed with this repository. All paths resolve through `lss_paths.py`; point `LSS_PROJECT_ROOT` at the directory holding them.

| Asset | Default path | How to obtain |
|---|---|---|
| SD 1.5 base | `0-model/stable-diffusion-v1-5` | `hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir <path>` |
| ControlNet depth | `0-model/sd-controlnet-depth` | `hf download lllyasviel/sd-controlnet-depth --local-dir <path>` |
| ControlNet seg | `0-model/sd-controlnet-seg` | `hf download lllyasviel/sd-controlnet-seg --local-dir <path>` |
| MiDaS weights | HF cache | `hf download lllyasviel/Annotators --include "dpt_hybrid-midas-501f0c75.pt"` |
| CLIP ViT-B/32 | HF cache | `hf download openai/clip-vit-base-patch32` |
| YOLOv8x (V9 only) | `0-model/yolov8x.pt` | `python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"` |

The first three must be diffusers directory snapshots, since the code calls `from_pretrained` on them; a single `.safetensors` file will not work. MiDaS and CLIP are loaded by repository id, so they have to land in the HuggingFace cache — pointing at a local directory is ignored. Set `HF_ENDPOINT=https://hf-mirror.com` where applicable.

**Background pool.** The paper uses 37,852 photographs across nine categories (`airfield` / `coastal` / `desert` / `forest` / `industrial` / `mountain` / `pure_sky` / `rural` / `urban`), none of which ship here. To supply your own, create one directory per category with 16:9 photographs and a `_metadata.jsonl` inside each describing diurnal phase and weather; the retriever uses those fields for its four-tier fallback. There is no minimum count — a few dozen images exercise the pipeline.

**Target LoRA.** Deliberately not bundled: the paper's claim is that any target LoRA can be substituted, so this is an extension point rather than a fixed asset. Provide a kohya-format LoRA on the SD 1.5 architecture (`lora_unet_*` keys) and edit `lss_paths.LORA_PATH`. Note the loader merges the UNet branch only; any text-encoder component is skipped.

---

## 7. Usage

```bash
# verify assets resolve
python lss_paths.py

# web interface
cd app && cp .env.example .env   # add your API key
python web_app.py                # → http://127.0.0.1:5000

# individual validators
python validator/v6_quality/quality_validator.py IMG.png --ir IR.png
python validator/v7_consistency/consistency_validator.py IMG.png --ir IR.png --gt gt.json
python validator/v8_trajectory_validator.py traj.json
python validator/v9_detection_validator.py IMG.png --gt bboxes.json --model yolov8x.pt

# recompute every table in the paper
python experiments/bcd/_reports/scripts/audit_tables_v20.py
```

`loop/validator_pipeline.py` is a library, not a script:

```python
from loop.validator_pipeline import ValidatorPipeline

pipe = ValidatorPipeline(s0, s1, s2, s3)
result = pipe.validate({
    "id": "s001",
    "rgb": rgb_ndarray, "ir": ir_ndarray,
    "rgb_bbox": [cx, cy, w, h],      # normalised
    "ir_bbox":  [cx, cy, w, h],      # normalised
    "gt_bboxes": [[x1, y1, x2, y2]], # pixels
})
```

Two box conventions coexist: `rgb_bbox` and `ir_bbox` are normalised `[cx, cy, w, h]`, while `gt_bboxes` are pixel `[x1, y1, x2, y2]`.

---

## 8. Results

The benchmark is 300 visible and pseudo-infrared pairs, stratified over nine scene categories, five diurnal phases and five weather classes, with all 300 conditioning tuples unique:

| Metric | Value |
|---|---|
| Position error E_pos | 1.226 ± 0.448 px |
| Scale error E_scale | 0.028 ± 0.021 |
| IoU | 0.706 ± 0.126 |
| Share at IoU ≥ 0.5 | 94.0% |
| BRISQUE | 20.55 |
| Acceptance rate | 88.3% |
| Generation time per pair | 14.58 s |

On the value of the chain over a scalar criterion: across 302 composites, the BRISQUE threshold alone accepts 267 (88.4%), while the V6→V7→V9 chain accepts 161 (53.3%), agreeing on 158. The paper additionally reports ablations (E1–E3), the three closed-loop paradigms, a downstream proxy and a parser comparison; every number is recomputable from the per-sample JSON in this repository via `audit_tables_v20.py`.

A boundary on the evidence: the experiments characterise the **generation process** — whether the target lands at the requested position and scale, and whether defective samples are caught — not the utility of the product. No detector was trained on CDFF output and evaluated on real imagery, so nothing here supports a claim of improved detection accuracy.

---

## 9. Known limitations

- **Two geometries for scale remain unreconciled.** The generator fixes the footprint analytically from the declared range, while the V7 size check falls back to physical projection (real airframe dimensions and a 60° FOV) when no upstream expected value is supplied. The two differ by roughly an order of magnitude, so small targets fail that check in a directly executed pipeline. The scale error reported in the paper is computed from the declared value and is unaffected.
- The three-way failure-buffer routing and the "threshold rollback when batch acceptance drops below 10%" rule are design constraints. The buffer itself and stage-wise short-circuiting are implemented; the routing scheduler and rollback are not.
- V9 uses a COCO-pretrained YOLOv8x, whose class list contains no aerial target. It serves as a downstream saliency proxy, not as evidence of detection accuracy.

## 10. Citation

```bibtex
@article{li_cdff,
  title  = {A Spatially Controlled Closed-Loop Framework for Synthetic
            Low-Slow-Small Target Dataset Generation},
  author = {Li, Qingming and Zhang, Yanxin and Zhu, Zhongpan and He, Bin},
  note   = {Manuscript under review, Neurocomputing}
}
```

## 11. License

MIT

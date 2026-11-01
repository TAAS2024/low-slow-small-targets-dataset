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
│   ├── lora_loader.py              # LoRA load / inspect / merge (kohya format)
│   └── rgb2ir_converter.py         # M6 RGB → pseudo-IR
├── validator/                      # V1–V9
│   ├── v1_json_validator.py
│   ├── v2_transformer_validator.py
│   ├── v3_controlnet_validator.py
│   ├── v4_lora_validator.py
│   ├── v5_ir_validator.py
│   ├── v6_quality/                 # perceptual quality + IR signal
│   ├── v7_consistency/             # size / lighting / IR box / cross-modal
│   │   └── test_s7_tristate.py     # regression: PASS/FAIL/SKIP must not blur
│   ├── v8_trajectory_validator.py
│   └── v9_detection_validator.py
├── loop/
│   ├── validator_pipeline.py       # S6→S9 short-circuit chain
│   ├── failure_buffer.py           # buffer and threshold trigger
│   └── trainable_classifier.py     # paradigm A classifier
├── training/                       # target LoRA: recipe + trainer + notes
│   ├── train_lora.py               # diffusers+PEFT trainer, exports kohya keys
│   ├── README.md                   # recipe as recorded in the released adapter
│   └── configs/
│       ├── lora_sd15_drn3.yaml     # config for train_lora.py
│       └── kohya_sd15_drn3.toml    # the sd-scripts recipe (original path)
├── preprocessing/                  # dataset construction, CPU only
│   ├── prepare_background_pool.py  # photos → nine-category pool + JSONL
│   └── prepare_lora_dataset.py     # images → kohya training directory
├── notebooks/
│   ├── 01_quickstart.ipynb         # generated by build_quickstart.py
│   ├── build_quickstart.py
│   └── smoke_run.py                # executes every cell without weights
├── tests/
│   └── test_lora_loader.py         # key translation + merge numerics
├── experiments/                    # reproduction scripts and summaries
├── THIRD_PARTY_LICENSES.md         # every third-party licence and its effect
└── NOTICE                          # attribution, incl. the base-model licence
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
| target adapter (training) | `training/train_lora.py` | `python training/train_lora.py --config training/configs/lora_sd15_drn3.yaml` |
| adapter merge | `generator/lora_loader.py` | `merge_lora_into_unet()` |
| dataset construction | `preprocessing/*.py` | see `preprocessing/README.md` |

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

V7 reports each dimension in one of three states — `PASS`, `FAIL`, `SKIP` — and the last one is not a pass: a dimension with no usable input is recorded as `SKIP` and excluded from the verdict, and a sample with no evaluated dimension at all is `SKIP` rather than accepted. `python validator/v7_consistency/test_s7_tristate.py` checks this without any model weights.

---

## 5. Installation

```bash
git clone https://github.com/TAAS2024/low-slow-small-targets-dataset.git
cd low-slow-small-targets-dataset
pip install -r requirements.txt          # PyTorch: use the +cu124 wheel for GPU
```

CPU-only is enough for the validators, the auditors, the preprocessing scripts and
`tests/`. Generation needs a GPU: the per-pair timing reported in the paper was
measured on a single 8 GB card. `ultralytics` is **not** installed by default —
V9 is optional and its licence is AGPL-3.0 (see §11).

---

## 6. Models and assets

Weights and images are not distributed with this repository. All paths resolve through `lss_paths.py`; point `LSS_PROJECT_ROOT` at the directory holding them.

| Asset | Default path | How to obtain | Licence |
|---|---|---|---|
| SD 1.5 base | `0-model/stable-diffusion-v1-5` | `hf download sd-legacy/stable-diffusion-v1-5 --local-dir <path>` | CreativeML OpenRAIL-M |
| ControlNet depth | `0-model/sd-controlnet-depth` | `hf download lllyasviel/sd-controlnet-depth --local-dir <path>` | OpenRAIL |
| ControlNet seg | `0-model/sd-controlnet-seg` | `hf download lllyasviel/sd-controlnet-seg --local-dir <path>` | OpenRAIL |
| MiDaS weights | HF cache | `hf download lllyasviel/Annotators --include "dpt_hybrid-midas-501f0c75.pt"` | MIT (MiDaS) |
| CLIP ViT-B/32 | HF cache | `hf download openai/clip-vit-base-patch32` | MIT |
| YOLOv8x (V9 only) | `0-model/yolov8x.pt` | `python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"` | **AGPL-3.0** |
| Target LoRA | `2-Lora training/best_models/*.safetensors` | train it — `training/README.md` | inherits OpenRAIL-M |

The first three must be diffusers directory snapshots, since the code calls `from_pretrained` on them; a single `.safetensors` file will not work. MiDaS and CLIP are loaded by repository id, so they have to land in the HuggingFace cache — pointing at a local directory is ignored. Set `HF_ENDPOINT=https://hf-mirror.com` where applicable.

**Background pool.** The paper uses 37,852 photographs across nine categories (`airfield` / `coastal` / `desert` / `forest` / `industrial` / `mountain` / `pure_sky` / `rural` / `urban`), none of which ship here. To supply your own, run `preprocessing/prepare_background_pool.py` — it writes one directory per category with 16:9 photographs and the `_metadata.jsonl` the retriever reads for its four-tier fallback. There is no minimum count; a few dozen images exercise the pipeline. See `preprocessing/README.md`.

**Target LoRA.** Deliberately not bundled: the paper's claim is that any target LoRA can be substituted, so this is an extension point rather than a fixed asset. Provide a kohya-format adapter on the SD 1.5 architecture (`lora_unet_*` keys) and edit `lss_paths.LORA_PATH`; `generator/lora_loader.py` merges the UNet branch only, and `python generator/lora_loader.py <adapter>` tells you what an adapter contains before you run the pipeline. To train one, see `training/README.md`.


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
python validator/v7_consistency/consistency_validator.py IMG.png --expected-px 60 --gt gt.json
python validator/v7_consistency/consistency_validator.py IMG.png --gt gt.json --physical-fallback
python validator/v8_trajectory_validator.py traj.json
python validator/v9_detection_validator.py IMG.png --gt bboxes.json --model yolov8x.pt

# recompute every table in the paper from the per-sample records (51 assertions, 51 pass in a fresh
# clone; 53 with the background pool on disk — the pool itself is not redistributed)
python experiments/bcd/_reports/scripts/audit_tables_v20.py

# LoRA: inspect an adapter, then train one
python generator/lora_loader.py path/to/adapter.safetensors
python generator/lora_loader.py path/to/adapter.safetensors --json
python training/train_lora.py --print-config                 # resolve the YAML, no GPU
python training/train_lora.py --config training/configs/lora_sd15_drn3.yaml \
    --set data.image_dir="$LSS_PROJECT_ROOT/0-database/kohya_dataset/10_drn3_uav"

# data preparation (CPU only)
python preprocessing/prepare_background_pool.py --src ~/photos --labels labels.jsonl
python preprocessing/prepare_lora_dataset.py --src ~/drones \
    --out "$LSS_PROJECT_ROOT/0-database/kohya_dataset" --class-name drn3_uav

# self-checks that need no weights
python tests/test_lora_loader.py
python validator/v7_consistency/test_s7_tristate.py
python notebooks/smoke_run.py                                 # executes every notebook cell
```

`loop/validator_pipeline.py` is a library, not a script:

```python
from loop.validator_pipeline import ValidatorPipeline, build_sample_from_geometry

pipe = ValidatorPipeline(s0, s1, s2, s3)
result = pipe.validate({
    "id": "s001",
    "rgb": rgb_ndarray, "ir": ir_ndarray,
    "rgb_bbox": [cx, cy, w, h],      # normalised
    "ir_bbox":  [cx, cy, w, h],      # normalised
    "expected_px": 60,               # declared target width → enables E_scale
    "gt_bboxes": [[x1, y1, x2, y2]], # pixels
})
```

`build_sample_from_geometry()` closes the loop from the generator side: both entry
points of `app/condition_generator_v7.py` return a `geometry` list (one record per
frame, carrying `bbox_px`, `bbox_norm`, `expected_px`, `realised_px`, `scale_err`),
so the realised box and the declared footprint reach the validator instead of dying
in the generator's local scope:

```python
out = generate_with_lora(bg_path, spec)      # or focus_reposition(...)
for geo in out["geometry"]:
    sample = build_sample_from_geometry(geo, rgb=frame_rgb, ir=ir_rgb)
    result = pipe.validate(sample)
    print(geo["scale_err"], result.s1["scale_err"])   # identical by construction
```

Two box conventions coexist: `rgb_bbox` and `ir_bbox` are normalised `[cx, cy, w, h]`, while `gt_bboxes` are pixel `[x1, y1, x2, y2]`. Width from a pixel pair is `x2 - x1 + 1` in `build_sample_from_geometry` and `x2 - x1` in `experiments/bcd/adapters.py` — a 1 px difference kept deliberately, because the latter is the path that produced the reported figures.

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

On the value of the chain over a scalar criterion: across 302 composites, the BRISQUE threshold alone accepts 267 (88.4%), while the V6→V7→V9 chain accepts 161 (53.3%), agreeing on 158. That pool is `out_E2/metrics.json`: the 300 main-benchmark samples plus the same hard case (a0149, target 20 px at 180 m) from the repaint and paste arms — merged because it passes the scalar threshold while the chain rejects it, which moves the Jaccard index *down* from 0.590 to 0.585, so the pooling is not a selection effect. Of the 109 composites the scalar threshold admits and the chain rejects, 108 carry the undetectable-target code and one fails the visible-quality stage, so the 40.8% gap measures detector saliency on targets a few dozen pixels wide rather than gross image defects. `audit_tables_v20.py` recomputes every table from those per-sample records and asserts each cell; it currently passes 51/51 in a fresh clone and 53/53 where the background pool is on disk, the two extra assertions being the pool census. The background pool itself (37,852 images) is not in this repository, so for Table 2 the scene rows stay unasserted, and `audit_v20.py` is an author-side tool that needs the paper's .docx source and is not a repository-internal entry point.
**Ablations E1–E3.** E1 compares marker injection against hard paste on 150 paired targets and is a null result on geometry and no-reference quality (position error 1.241 against 1.241 px, IoU 0.706 against 0.706, BRISQUE 20.67 against 20.64); the one resolved difference is the local contrast ratio (1.338 against 1.655, paired p = 4.9e-05). E2 asks whether both geometric conditions are required, regenerating 100 paired targets under dual, depth-only, segmentation-only and two intermediate segmentation scales. Alignment and global quality are identical across those arms by construction — coordinate, scale and marker geometry are fixed upstream of the generative model — and the arms separate only in boundary fusion: dual is worse than depth-only (boundary-gradient excess 2.286 against 2.114, paired p = 1.1e-05) and better than segmentation-only (2.316, p = 0.006, uncorrected). With the depth branch held at 0.85, lowering the segmentation scale to 0.60 and 0.40 moves the excess to 2.276 and 2.249 (p = 0.525 and 0.149), which is monotonic but shallow: the segmentation weight accounts for 0.037 of the 0.172 gap to depth alone, so over-constraint is not the source. E3 disables the adapter and leaves placement untouched (1.252 against 1.249 px, p = 0.53), while box IoU moves from 0.706 to 0.744 (p = 9.0e-04, rank-biserial r = 0.33) and tracks the silhouette the adapter renders, since the reference box is fixed upstream by marker injection. Records: `out_E1_fusion`, `out_E4_both`, `out_E4_depth`, `out_E4_seg`, `out_E4_d85s60`, `out_E4_d85s40`, `out_E4sweep_fusion`.

Section 4.5 of the paper reports sixteen paired tests, two-sided and uncorrected. Under a Bonferroni threshold of 0.05/16 = 3.1e-03 three results remain resolved — the repaint boundary effect (p = 4.9e-05), dual against depth-only (p = 1.1e-05) and the adapter's shift in box IoU (p = 9.0e-04). The remainder are exploratory and are read as such in the paper.

**Closed-loop regimes.** Three regimes ran for four cycles of 60 target points (`out_B_A1`, `out_B_A2`, `out_B_A3`). A1 leaves the loop open and holds 0.567–0.583 per-attempt acceptance, 138 of 240 points (57.5%). A2 reallocates the sampling budget after every cycle: the 40 strata are the cells of the sampling grid over slant distance and weather, each weight is multiplied by the ratio of that stratum's smoothed pass rate to the global pass rate, smoothed as (passes + 2) / (attempts + 4), floored at 30% of a uniform share and renormalised. It completes 172 of 240 points (71.7%) at 720 attempts; A3 adds curriculum hardening and completes the same 172 at 704 attempts. Over four cycles the weights move from 40 equal strata (max/min = 1.00) to a 6.15-fold spread, and the cost per accepted sample rises from 30.9 s (A1) to 74.7 s (A2) and 84.9 s (A3). The loop therefore acts on which strata fill the quota, not on how well a sample renders: the parameter-update paths are gated by buffer capacities of 50, 30 and 20 that four cycles of 60 points rarely reach, and BRISQUE and IoU do not improve monotonically across the cycles. A composition caveat applies — the final validation stage requires a detector to localise every ground-truth box and 71% of A1 rejections carry the undetectable-target code, so loop-accepted samples are a detector-selected subset, and evaluating a detector of the same family on it is circular.


A boundary on the evidence: the experiments characterise the **generation process** — whether the target lands at the requested position and scale, and whether defective samples are caught — not the utility of the product. No detector was trained on CDFF output and evaluated on real imagery, so nothing here supports a claim of improved detection accuracy.

---

## 9. Known limitations

- **Scale is checked against the declared value, and the physical projection is off by default.** The generator fixes the footprint analytically from the declared range (Eq. 3), and V7 compares the realised box against that same declared value (`check_alignment`, `expected_px`). The alternative physical projection — real airframe dimensions and a 60° FOV — is still in the file as an opt-in only (`physical_fallback=True`, `--physical-fallback`); it differs from the 1/d placement geometry by roughly an order of magnitude, so it is not usable as a gate. A dimension with no input is recorded as `SKIP`, never as a pass. The `alignment_tolerance` default of 0.50 is the runtime configuration that produced the reported acceptance figures; tightening it changes those numbers.
- The three-way failure-buffer routing and the "threshold rollback when batch acceptance drops below 10%" rule are design constraints. The buffer itself and stage-wise short-circuiting are implemented; the routing scheduler and rollback are not.
- V9 uses a COCO-pretrained YOLOv8x, whose class list contains no aerial target. It serves as a downstream saliency proxy, not as evidence of detection accuracy.
- The closed loop adapts the sampling budget, not the generator. Over four cycles the parameter-update paths, gated by buffer capacities of 50, 30 and 20, were never reached, and BRISQUE and IoU do not improve monotonically; the measured effect is quota completion from 57.5% to 71.7% at 2.4 times the cost per accepted sample.
- Three of the sixteen paired comparisons in §4.5 survive a Bonferroni threshold of 0.05/16 = 3.1e-03. The others are uncorrected single-comparison values and are read as exploratory.
- The E2 arms are identical in alignment and global quality by construction, so that table cannot show a conditioning effect on placement; the comparison is only informative through the boundary-fusion metrics.

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

**The source code in this repository is MIT. The project as a whole is not.**
Model weights, model derivatives and one optional dependency carry other
licences, and no part of those terms is overridden by the MIT text in `LICENSE`.

| Component | Licence | Redistributed here? |
|---|---|---|
| our source code (`app/`, `generator/`, `validator/`, `loop/`, `training/`, `preprocessing/`, `notebooks/`, `tests/`, `experiments/`) | **MIT** | yes |
| Stable Diffusion v1.5 base model | **CreativeML OpenRAIL-M** | no — downloaded |
| ControlNet depth / seg | **OpenRAIL** | no — downloaded |
| any LoRA adapter trained here, and any checkpoint merged from it | **CreativeML OpenRAIL-M (inherited)** | no |
| MiDaS, CLIP | MIT | no — downloaded |
| Ultralytics YOLOv8 (optional, V9) | **AGPL-3.0** | no — not installed by default |
| background pool, drone source images, benchmark images | original sources' terms | no |

### Inheriting the base model's licence

- A LoRA adapter is a **derivative** of the base model. Ours is trained on SD 1.5,
  so the adapter — and any checkpoint merged with it — stays under CreativeML
  OpenRAIL-M, including its Attachment A use-based restrictions. Ship that
  licence and those restrictions with the weights.
- CreativeML OpenRAIL-M permits commercial use *subject to those restrictions*; it
  is not a public-domain or MIT-style grant. Do not describe the adapter as MIT
  or Apache-2.0, and do not describe the project as "fully MIT".
- If you substitute a **non-commercial** backbone (research-only, CC-BY-NC, …),
  then every derived weight — adapter included — is likewise non-commercial. The
  MIT code licence cannot change a weight's licence.
- The ControlNet branches are OpenRAIL as well, so a pipeline combining SD 1.5 +
  ControlNet inherits the stricter of the two, and both are OpenRAIL-M-family.

### The AGPL boundary

V9 (`validator/v9_detection_validator.py`) and the downstream evaluator import
`ultralytics` lazily, only when a detector path is supplied, and the package is
commented out of `requirements.txt`. Using this repository's source does not
by itself put you under AGPL — but a *distributed product* that bundles
`ultralytics` does, and AGPL §13 counts network use as distribution. Pass any
object with a `.predict()` method, e.g. an Apache-2.0 DETR/RT-DETR checkpoint, to
stay clear of it.

Full inventory, per-library terms and a decision table: `THIRD_PARTY_LICENSES.md`.
Attribution and the base-model statement: `NOTICE`.

### Citation of the base model

> Runway ML / Stability AI, *Stable Diffusion v1.5*, CreativeML OpenRAIL-M.
> Zhang, Rao & Agrawala, *Adding Conditional Control to Text-to-Image Diffusion
> Models* (ControlNet), for the depth and segmentation branches.


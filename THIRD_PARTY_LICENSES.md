# Third-party components, licences and obligations

This repository ships **code only**. Model weights are not redistributed: every
weight file is downloaded by the user at run time (see `README.md` §6). That does
not make the project licence-free — the code calls third-party libraries and
loads third-party weights, and each of those carries its own terms.

The rule that governs this repository:

> **The MIT licence in `LICENSE` covers the original source code in this
> repository and nothing else.** It does not relicense any base model, any
> ControlNet branch, any adapter weight, any detector, any library or any
> background photograph. Every artefact below stays under its own licence, and any
> weight derived from a base model stays under **that base model's** licence.

Last verified: 2026-10-30.

---

## 1. Base generative model and conditioning branches

These are the components that determine the licence of any LoRA adapter or
fine-tuned weight produced with this pipeline.

| Component | Where it comes from | Licence | How this repository uses it |
|---|---|---|---|
| **Stable Diffusion v1.5** | `sd-legacy/stable-diffusion-v1-5` (successor of `runwayml/stable-diffusion-v1-5`) | **CreativeML OpenRAIL-M** | Base UNet / VAE / text encoder. Loaded at run time by `generator/`; weights are never committed (`.gitignore` blocks `*.safetensors`, `*.ckpt`, `*.bin`). |
| **ControlNet depth** | `lllyasviel/sd-controlnet-depth` | **OpenRAIL** (OpenRAIL-M family, as declared on the model card) | Depth-conditioned repaint branch (`generator/lora_inpainter_v7.py`). |
| **ControlNet segmentation** | `lllyasviel/sd-controlnet-seg` | **OpenRAIL** (OpenRAIL-M family) | Segmentation-conditioned repaint branch, same file. |
| **Target LoRA adapter** | produced by this project (see `training/`) | **Inherits CreativeML OpenRAIL-M** — it is a derivative of SD 1.5 | Not bundled. The loader (`generator/lora_loader.py`) expects a kohya-format adapter on the SD 1.5 architecture. |

### What CreativeML OpenRAIL-M requires

OpenRAIL-M is *not* a non-commercial licence, and it is *not* MIT/Apache either.
It grants use and redistribution, including commercial use, **subject to
use-based restrictions** (Attachment A of the licence: no unlawful
discrimination, no disinformation, no exploitation of minors, no medical or
legal advice without qualification, etc.), and it **propagates**:

- A LoRA adapter, a merged checkpoint or any other weight derived from SD 1.5 is
  a *derivative* and must be distributed under the **same** CreativeML
  OpenRAIL-M terms — the use-based restrictions must travel with the weights.
- The licence text and its restrictions must accompany any redistribution of
  those derived weights.
- A pipeline that combines SD 1.5 + ControlNet inherits the stricter of the two:
  OpenRAIL-M for both branches, so the combined model output remains subject to
  Attachment A.

**Consequence for this repository:** nobody may describe the trained adapter, or
a merged model built from it, as "MIT" or "Apache-2.0", and nobody may state that
the *project as a whole* is MIT. `README.md` §11 and `NOTICE` say this
explicitly.

> If you substitute the backbone with a **non-commercial** base model (for
> example a research-only or CC-BY-NC checkpoint), then the entire derivative
> weight set — adapter included — becomes non-commercial, and the MIT code
> licence cannot change that. Code licence and weight licence are independent
> axes.

---

## 2. Perception and tooling models

| Component | Source | Licence | Role |
|---|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | **MIT** (`openai/CLIP`) | Text/image embedding used by the V2 and V6/V7 checks. |
| MiDaS DPT-Hybrid | `intel-isl/MiDaS` | **MIT** | Monocular depth prior (M3). Downloaded through the `lllyasviel/Annotators` hub repo, whose model card declares `license: other`; the mirror is a download channel only, the weights themselves are MiDaS under MIT. |
| U-2-Net segmentation (via `rembg`) | `danielgatis/rembg` / `xuebinqin/U-2-Net` | **MIT** (rembg code) / **Apache-2.0** (U-2-Net weights) | Alpha matte extraction of the rendered target. |
| YOLOv8x | `ultralytics` | **AGPL-3.0** | **Optional and never imported at package level.** V9 (`validator/v9_detection_validator.py`) and `experiments/bcd/eval_downstream.py` import `ultralytics` lazily, only when the caller supplies a detector path. `requirements.txt` keeps it commented out for exactly this reason. |

### What AGPL-3.0 means here

The AGPL obligation attaches when **you** combine and distribute a work that
includes ultralytics code. Keeping the import lazy and the package optional means
this repository's own source can be used, read and modified without triggering
AGPL — but:

- If you build a distributable product around V9 / the downstream evaluator with
  `ultralytics` installed and bundled, that product must be released under
  AGPL-3.0 (network use counts as distribution under AGPL §13).
- If you need a permissive detector instead, substitute any object exposing
  `.predict()` — `experiments/bcd/adapters.py` shows the interface — e.g. an
  Apache-2.0 DETR/RT-DETR checkpoint.
- The COCO-pretrained YOLOv8x weights are covered by the same AGPL terms.

---

## 3. Libraries

Installed by `requirements.txt`; none are vendored into this repository.

| Library | Licence | Library | Licence |
|---|---|---|---|
| torch | BSD-3-Clause | Flask | BSD-3-Clause |
| torchvision | BSD-3-Clause | python-dotenv | BSD-3-Clause |
| diffusers | Apache-2.0 | python-docx | MIT |
| transformers | Apache-2.0 | openai (SDK) | Apache-2.0 |
| peft | Apache-2.0 | anthropic (SDK) | MIT |
| accelerate | Apache-2.0 | piq | Apache-2.0 |
| safetensors | Apache-2.0 | tqdm | MPL-2.0 / MIT |
| opencv-python | Apache-2.0 | PyYAML | MIT |
| numpy | BSD-3-Clause | SciPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) | matplotlib | PSF-based (BSD-compatible) |
| einops | MIT | rembg | MIT |

`python-docx` is used only by the author-side auditing tools under
`experiments/bcd/_reports/scripts/`; the repository-internal audit
(`audit_tables_v20.py`) runs without it.

---

## 4. Data

- **Background pool (37,852 photographs).** Not distributed. The paper's pool was
  assembled from public image sources under their own terms (the per-image
  `_metadata.jsonl` records `source`, `url` and `photographer` so provenance stays
  auditable). `preprocessing/prepare_background_pool.py` builds a pool in the same
  format from images **you** have the right to use — check the source licence
  before adding a photograph, and keep the attribution fields populated.
- **Drone imagery used for the adapter.** Not distributed. Bring your own source
  images and their captions; `preprocessing/prepare_lora_dataset.py` converts them
  into a kohya-format directory.
- **The 300-pair benchmark.** The per-sample records (`experiments/bcd/out_*/`)
  ship; the images themselves do not.

---

## 5. Cheat sheet

| If you want to… | Then… |
|---|---|
| reuse this repository's source code | MIT — keep the copyright notice. |
| redistribute a LoRA adapter trained on SD 1.5 | ship it under CreativeML OpenRAIL-M, with Attachment A restrictions attached. Never label it MIT. |
| merge the adapter into an SD 1.5 checkpoint and publish the result | same as above; the merged file is OpenRAIL-M. |
| publish generated images | the OpenRAIL-M use restrictions apply to your use of the model; the licence does not claim copyright over outputs, but downstream use restrictions still bind the model's use. |
| ship a product bundling `ultralytics` | AGPL-3.0 for that product, or swap the detector. |
| describe the project in a paper or a README | "code MIT; base model and derivatives under CreativeML OpenRAIL-M; optional detector under AGPL-3.0" — not "MIT project". |

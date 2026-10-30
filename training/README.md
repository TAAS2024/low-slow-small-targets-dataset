# Training the target LoRA

The generator does not ship with a target adapter: any kohya-format LoRA on the
SD 1.5 architecture can be dropped in (`lss_paths.LORA_PATH`), which is the point
of the design. This directory contains the recipe that produced the adapter used
for the paper's experiments, so the pipeline can be retrained end to end, and a
diffusers + PEFT reimplementation of the same recipe.

Two files, two purposes:

| File | What it is | Use it when |
|---|---|---|
| `configs/kohya_sd15_drn3.toml` | the `kohya-ss/sd-scripts` recipe **recorded in the metadata of the released adapter** (`ss_optimizer = bitsandbytes.optim.adamw.AdamW8bit`, `modelspec.implementation = sd-scripts`) | you want to reproduce the released checkpoint as closely as possible |
| `train_lora.py` + `configs/lora_sd15_drn3.yaml` | a self-contained diffusers + PEFT trainer that reads the same hyper-parameters and exports **kohya-format keys** | you want to train inside this repository without installing sd-scripts |

Both write `.safetensors` files that `generator/lora_loader.py` can merge; the
export step translates PEFT names (`…to_q.lora_A.default.weight`) into kohya
names (`lora_unet_…to_q.lora_down.weight` + `.alpha`), so no conversion tool is
needed between training and inference.

---

## 1. Recipe as recorded in the released adapter

Read from the safetensors header of `drn3_pocket_uav_v3_step2000.safetensors`
(80 metadata keys, `ss_*` / `modelspec.*`). These are measurements of the
artefact, not a reconstruction.

| Item | Value |
|---|---|
| base model | Stable Diffusion 1.5 (`ss_base_model_version = sd_v1`) |
| adapter type | `networks.lora`, dim **8**, alpha **4** |
| adapter scope | 576 `lora_unet_*` modules **and** 216 `lora_te_*` modules |
| dataset | kohya subset `10_Pocket_UAV`, **49 images × 10 repeats = 490** samples / epoch |
| resolution | 512 × 512, no bucketing (`enable_bucket = false`) |
| augmentation | `flip_aug = false`, `color_aug = false`, `random_crop = false`, `shuffle_caption = false`, `keep_tokens = 1` |
| optimizer | `bitsandbytes.optim.adamw.AdamW8bit`, lr **1e-4** (one lr; text-encoder lr unset) |
| scheduler | `cosine_with_restarts`, warmup 0 steps, floor 1 % of base lr |
| steps | `max_train_steps = 3000`; the released file is the **step-2000** checkpoint (`ss_epoch = 4`) |
| loss shaping | `min_snr_gamma = 5.0`, `noise_offset` off |
| precision | `fp16`, `cache_latents = true`, `gradient_checkpointing = true` |
| seed | `42` |

The paper's §4.1 reports the same backbone, the same two ControlNet branches
(conditioning scale 0.85), a rank-8 / alpha-4 adapter, 3,000 steps at lr 1e-4 and
the step-2000 checkpoint.

### Provenance note (recorded, not reconciled)

The paper's §4.1 states the adapter was trained over **98** drone images. The
released adapter's own metadata records **49** images in the subset that was
actually used (`ss_dataset_dirs = {"10_Pocket_UAV": {"n_repeats": 10, "img_count": 49}}`),
and the training directory in the research tree holds **98** image/caption pairs.
Both numbers exist as artefacts; this file reports them as found rather than
asserting that one equals the other. If you are reproducing the run, use
`configs/kohya_sd15_drn3.toml` with whatever pool you have — the adapter's
behaviour is not sensitive to the count in a way this repository can claim.

---

## 2. Build the training set

```bash
python preprocessing/prepare_lora_dataset.py \
    --src /data/drone_photos \
    --out "$LSS_PROJECT_ROOT/0-database/kohya_dataset" \
    --class-name drn3_uav --resolution 512 --repeats 10
```

The script white-flattens alpha, scales the longest side to the target
resolution, pads to a square and writes one templated `.txt` caption per image —
the same geometry the generator produces before compositing. Folder name
`10_drn3_uav` encodes repeats = 10, per kohya convention.

Check that you hold the rights to the source photographs before training, and
remember that the result is a derivative weight (see §5).

---

## 3. Train

### kohya-ss / sd-scripts (the original path)

```bash
git clone https://github.com/kohya-ss/sd-scripts && cd sd-scripts
pip install -r requirements.txt
accelerate launch --num_cpu_threads_per_process 2 train_network.py \
    --config_file /path/to/low-slow-small-targets-dataset/training/configs/kohya_sd15_drn3.toml
```

Edit the three paths at the top of the TOML (`pretrained_model_name_or_path`,
`image_dir`, `sample_prompts`) first.

### This repository's trainer

```bash
# inspect the resolved configuration without touching a GPU
python training/train_lora.py --print-config

# 512², 3,000 steps, rank-8 — the paper's recipe
python training/train_lora.py --config training/configs/lora_sd15_drn3.yaml \
    --set data.image_dir="$LSS_PROJECT_ROOT/0-database/kohya_dataset/10_drn3_uav"

# a 768² variant, fewer steps
python training/train_lora.py --set data.resolution=768 --set train.max_train_steps=2000
```

Output layout:

```
runs/lora_sd15_drn3/
├── resolved_config.yaml          # what actually ran, after --set overrides
├── checkpoints/step500.safetensors …
├── final/lora_weights.safetensors
└── run_summary.json              # plus the licence fields
```

Every export carries `ss_*` / `modelspec.*` metadata, including
`modelspec.license = CreativeML OpenRAIL-M`, so a downloaded adapter can be
identified without its README.

### Deviations from the kohya recipe — stated, not hidden

| Aspect | kohya (released) | `train_lora.py` |
|---|---|---|
| optimizer | AdamW8bit (bitsandbytes) | `torch.optim.AdamW` (fp32 states) |
| LoRA scope | UNet + text encoder | UNet only by default; `lora.train_text_encoder: true` adds CLIP attention adapters |
| scheduler | kohya `cosine_with_restarts` | same shape, floor 1 %, implemented locally |
| augmentation, resolution, steps, rank/alpha, seed | as in §1 | identical |

Neither path promises byte-identical weights to the released checkpoint: the
training data is not distributed, and a fp16 run is not bit-reproducible across
GPU models. What is reproducible is the configuration and the export contract.

---

## 4. Verify the adapter before using it

```bash
python generator/lora_loader.py runs/lora_sd15_drn3/final/lora_weights.safetensors
python tests/test_lora_loader.py        # key translation + numerics, no weights needed
```

`inspect_lora` reports the rank / alpha pair, how many modules each half covers
and the licence string from the metadata. A merge that silently adapts zero
modules is the failure mode this catches: the loader prints
`LoRA merged: n/m` and a `0/m` means the key naming did not match.

Set the path once:

```python
import lss_paths
lss_paths.LORA_PATH          # ← what generator/lora_inpainter_v7.py merges
```

or export `LSS_PROJECT_ROOT` and place the file where `lss_paths` expects it.

---

## 5. Licence

The base model is Stable Diffusion v1.5, **CreativeML OpenRAIL-M**. Anything
trained here is a derivative of it:

- the adapter inherits OpenRAIL-M, including the Attachment A use restrictions;
- the restrictions must be shipped with the weights when you redistribute them;
- it is not MIT, and it must not be described as MIT — the MIT licence of this
  repository covers the scripts, not the weights they produce;
- if you swap in a non-commercial backbone, the adapter becomes non-commercial
  too.

`THIRD_PARTY_LICENSES.md` has the full inventory and a decision table.

---

## 6. Hardware actually used

One WSL2 workstation: i9-13900HX, RTX 4060 Laptop with 8,188 MiB of VRAM, driver
560.94, PyTorch 2.6.0 + CUDA 12.4, Diffusers 0.39.0, fp16. A rank-8 adapter at
512² with gradient checkpointing fits in that budget; 768² leaves far less
headroom at the same batch size, which is why the paper's runs stay at 512².

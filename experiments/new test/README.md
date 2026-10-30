# Chapter 4 Experiment Executor

A self-contained runner for the Chapter 4 experiment analysis of the CDFF
pipeline.  Drop your generated RGB-IR pairs into `input/`, describe them in
`input/manifest.json`, run `python evaluator.py`, and the executor produces the
full per-sample analysis, aggregate statistics, a British-English report and
the publication figures.

The evaluator is **independent of the generation pipeline** — it does not load
SD1.5, ControlNet or the LoRA weights.  It only needs NumPy, OpenCV, PyTorch,
`piq`, Pillow and Matplotlib.

---

## Directory layout

```
new test/
├── evaluator.py          # entry point (run this)
├── eval_core.py          # metric library (alignment / BRISQUE / diversity)
├── report.py             # British-English report writer
├── figures.py            # Figure 6 and Figure 7 renderer
├── README.md
├── input/
│   ├── manifest.json     # describe your 20 samples here
│   ├── samples/
│   │   └── s000/         # one folder per sample
│   │       ├── rgb.png
│   │       ├── ir.png
│   │       └── drone_mask.png   (optional)
│   └── ablation/         # optional, for Figure 7 / Table 3
│       ├── manifest.json
│       ├── inject/
│       └── paste/
└── output/               # created on run
    ├── results.json
    ├── summary.json
    ├── report.md
    └── figures/
        ├── fig6_alignment.png
        └── fig7_ablation.png
```

---

## How to place your images

1. Create one folder per sample under `input/samples/` — e.g. `s000`, `s001`,
   … up to `s019`.
2. Put the generated files inside each folder:

   | File | Required | Purpose |
   |---|---|---|
   | `rgb.png` | yes | generated RGB frame |
   | `ir.png`  | yes | generated IR (white-hot) frame |
   | `drone_mask.png` | optional | binary drone mask used to measure the realised box |
   | `composite.png`   | optional | hard-paste composite (kept for reference) |

3. Fill in `input/manifest.json` for every sample.  The relative paths in the
   manifest are resolved against the `input/` directory.

---

## Manifest fields

Each entry in `manifest.json` → `samples` accepts:

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | str | unique identifier (e.g. `s000`) |
| `scene` | str | one of the nine categories (`pure_sky`, `coastal`, `urban`, `rural`, `forest`, `mountain`, `desert`, `industrial`, `airfield`) |
| `time_of_day` | str | `afternoon` / `dusk` / `night` |
| `weather` | str | free text (e.g. `clear`, `cloudy`) |
| `rgb` / `ir` | str | relative path to the frame |
| `drone_mask` | str or null | relative path to the drone mask (used to measure the realised box) |
| `ir_drone_mask` | str or null | relative path to an IR-side mask (optional) |
| `composite` | str or null | relative path to the hard-paste composite |
| `nu` | float | requested horizontal position in `[0,1]` (centre = `nu × W`) |
| `nv` | float | requested vertical position in `[0,1]` (centre = `nv × H`) |
| `target_px` | float | requested target extent in pixels |
| `bbox` | list or null | explicit realised box `[x1,y1,x2,y2]` — overrides `drone_mask` |
| `ir_bbox` | list or null | explicit IR box `[x1,y1,x2,y2]` |

**Ground-truth box priority**: the explicit `bbox` field is used first; if it
is absent the `drone_mask` is thresholded to derive the box; if neither is
available the alignment metrics are reported as `null` (skipped) and only
BRISQUE / diversity are computed.

**Cross-modal box**: if `ir_bbox` / `ir_drone_mask` are absent, the IR box is
assumed to coincide with the RGB box (`assumed: true`, IoU 1.0), which is
expected by construction because the white-hot conversion is pixel-wise.

---

## How to run

```bash
cd "experiments/new test"

# simplest form
python evaluator.py

# with explicit paths and parameters
python evaluator.py \
    --input input \
    --output output \
    --brisque-thr 45 \
    --cross-modal-thr 0.95 \
    --device auto          # auto | cpu | cuda
```

Command-line options:

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `input` | input directory containing `manifest.json` |
| `--output` | `output` | output directory |
| `--brisque-thr` | `45` | BRISQUE acceptance threshold |
| `--cross-modal-thr` | `0.95` | cross-modal IoU threshold |
| `--device` | `auto` | compute device for BRISQUE (`auto` picks CUDA if available) |
| `--no-figures` | off | skip figure generation |

---

## Outputs

| File | Content |
|---|---|
| `output/results.json` | per-sample metrics (alignment, BRISQUE, cross-modal) |
| `output/summary.json` | aggregate statistics (anchors, by-time-of-day, diversity) |
| `output/report.md` | British-English report with Tables 1-3 |
| `output/figures/fig6_alignment.png` | four-panel alignment/quality figure |
| `output/figures/fig7_ablation.png` | ablation visual comparison (if ablation data present) |
| `output/ablation.json` / `ablation_summary.json` | ablation detail and summary (if present) |

---

## Ablation (optional)

To produce Figure 7 and Table 3, place the matched marker-injection and
hard-paste frames under `input/ablation/inject/` and `input/ablation/paste/`
and describe them in `input/ablation/manifest.json`.  Each pair must share the
same `nu`, `nv`, `target_px` and background so that the two arms differ only in
the repainting step.  If `input/ablation/manifest.json` is absent, the ablation
section is simply skipped.

---

## Dependencies

```bash
pip install numpy opencv-python pillow torch piq matplotlib
```

#!/usr/bin/env python3
"""
figures.py — Publication figures for Chapter 4
===============================================
Generates:

  * Figure 6 — alignment-precision and quality visualisation (2x2 panel)
  * Figure 7 — ablation visual comparison (marker injection vs hard paste)

All figure text follows British English conventions.  Matplotlib is used with
the Agg backend so that no display server is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_core as ec

# A clean, publication-friendly style.
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------


def make_figure6(out_dir: Path, samples: list, summary: dict) -> None:
    """Render the four-panel alignment/quality figure (Figure 6)."""
    n = len(samples)
    ids = list(range(1, n + 1))
    pos = [s["alignment"]["pos_err_px"] for s in samples]
    scale = [s["alignment"]["scale_err"] for s in samples]
    iou = [s["cross_modal"]["iou"] for s in samples]
    brisque = [s["brisque"] for s in samples]
    thr = summary["quality"]["threshold"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Alignment precision and generation quality (proof-of-concept batch)",
                 fontsize=12)

    # Top-left: position-error scatter --------------------------------------
    ax = axes[0, 0]
    pos_vals = [p if p is not None else np.nan for p in pos]
    ax.scatter(ids, pos_vals, s=30, c="#1f77b4", label="samples")
    m = _mean(pos)
    ax.axhline(m, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"mean = {m:.2f} px")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0,
               label="sub-pixel threshold (1 px)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Position error (px)")
    ax.set_title("(a) Bounding-box position error")
    ax.legend(fontsize=8)

    # Top-right: scale-error histogram --------------------------------------
    ax = axes[0, 1]
    scale_vals = [s for s in scale if s is not None]
    ax.hist(scale_vals, bins=10, color="#2ca02c", edgecolor="white", alpha=0.8)
    m = _mean(scale)
    ax.axvline(m, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"mean = {m * 100:.2f}%")
    ax.axvline(0.03, color="grey", linestyle=":", linewidth=1.0,
               label="3% threshold")
    ax.set_xlabel("Scale error (fraction)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Scale-error distribution")
    ax.legend(fontsize=8)

    # Bottom-left: cross-modal IoU bars -------------------------------------
    ax = axes[1, 0]
    iou_vals = [v if v is not None else np.nan for v in iou]
    ax.bar(ids, iou_vals, color="#9467bd", width=0.7)
    ax.axhline(0.95, color="grey", linestyle=":", linewidth=1.0,
               label="cross-modal threshold (0.95)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Cross-modal IoU")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("(c) Cross-modal IoU (RGB vs IR)")
    ax.legend(fontsize=8)

    # Bottom-right: BRISQUE distribution ------------------------------------
    ax = axes[1, 1]
    br_vals = [b for b in brisque if b is not None]
    ax.hist(br_vals, bins=10, color="#ff7f0e", edgecolor="white", alpha=0.8)
    m = _mean(brisque)
    ax.axvline(m, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"mean = {m:.1f}")
    ax.axvline(thr, color="grey", linestyle=":", linewidth=1.0,
               label=f"threshold ({thr:.0f})")
    ax.set_xlabel("BRISQUE score (lower is better)")
    ax.set_ylabel("Count")
    ax.set_title("(d) BRISQUE score distribution")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "figures" / "fig6_alignment.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure 6  -> {out}")


# ---------------------------------------------------------------------------
# Figure 7
# ---------------------------------------------------------------------------


def make_figure7(out_dir: Path, ablation_pairs: list) -> None:
    """Render the ablation visual comparison (Figure 7)."""
    if not ablation_pairs:
        return
    pairs_with_img = [p for p in ablation_pairs
                      if p.get("inject", {}).get("rgb_loaded")
                      and p.get("paste", {}).get("rgb_loaded")]
    if not pairs_with_img:
        return

    n = len(pairs_with_img)
    fig, axes = plt.subplots(n, 2, figsize=(6.5, 3.0 * n),
                             squeeze=False)

    for r, p in enumerate(pairs_with_img):
        inj_rgb = ec.load_image(p["inject"]["rgb_path"], colour="rgb")
        pas_rgb = ec.load_image(p["paste"]["rgb_path"], colour="rgb")
        for c, (arm, img) in enumerate((("paste", pas_rgb), ("inject", inj_rgb))):
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img)
                box = p[arm]["gt_bbox"]
                if box is not None:
                    ax.add_patch(plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        fill=False, edgecolor="lime", linewidth=1.5))
            else:
                ax.text(0.5, 0.5, "image missing", ha="centre", va="centre")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                title = ("(b) Marker injection + ControlNet" if arm == "inject"
                         else "(a) Hard-paste baseline")
                ax.set_title(title, fontsize=9)
            # per-arm metric annotation
            al = p[arm]["alignment"]
            se = al["scale_err"] if al["scale_err"] is not None else float("nan")
            pe = al["pos_err_px"] if al["pos_err_px"] is not None else float("nan")
            io = al["iou"] if al["iou"] is not None else float("nan")
            ax.set_xlabel(f"scale {se*100:.2f}%  |  pos {pe:.2f} px  |  IoU {io:.3f}",
                          fontsize=7)

    fig.suptitle("Ablation E1: hard paste vs marker injection", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "figures" / "fig7_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure 7  -> {out}")

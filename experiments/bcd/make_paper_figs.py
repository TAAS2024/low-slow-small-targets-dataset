#!/usr/bin/env python3
"""
make_paper_figs.py — regenerate Figures 6 and 7 from the real experiment artefacts
==================================================================================
Figure 6 (fig6_alignment.png)  : four-panel alignment / quality summary over the
                                 300-pair benchmark (out_D_eval/results.json).
Figure 7 (fig7_ablation.png)   : qualitative ablation panel for E1 — hard-paste
                                 baseline vs marker injection, three matched
                                 conditioning points (out_E1_full, out_E1_paste).

Everything plotted here is read from on-disk results; no number is hard-coded.
British English, Agg backend, publication-friendly style.

Usage:  python3 make_paper_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER_FIGS = HERE.parents[1] / "·重点节点总结笔记" / "论文写作" / "figures"
OUT_DIR = HERE / "out_paper_figs"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_d() -> tuple[list, dict]:
    res = json.loads((HERE / "out_D_eval" / "results.json").read_text(encoding="utf-8"))
    summ = json.loads((HERE / "out_D_eval" / "summary.json").read_text(encoding="utf-8"))
    return res["samples"], summ


def fig6(samples: list, summ: dict, out: Path) -> None:
    n = len(samples)
    pos = np.array([s["alignment"]["pos_err_px"] for s in samples], dtype=float)
    scale = np.array([s["alignment"]["scale_err"] for s in samples], dtype=float)
    iou = np.array([s["alignment"]["iou"] for s in samples], dtype=float)
    bris = np.array([s["brisque"] for s in samples], dtype=float)
    thr = summ["metadata"]["brisque_threshold"]
    frac_iou50 = float((iou >= 0.5).mean())
    frac_pass = float((bris <= thr).mean())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Alignment precision and generation quality over the %d-pair benchmark" % n,
                 fontsize=12)

    # (a) position error across the benchmark ------------------------------
    ax = axes[0, 0]
    ax.scatter(np.arange(1, n + 1), pos, s=8, c="#1f77b4", alpha=0.7, linewidths=0)
    ax.axhline(pos.mean(), color="#d62728", ls="--", lw=1.2,
               label="mean = %.3f px" % pos.mean())
    ax.axhline(1.0, color="grey", ls=":", lw=1.0, label="sub-pixel threshold (1 px)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Position error (px)")
    ax.set_title("(a) Bounding-box position error")
    ax.legend(fontsize=8)

    # (b) scale error -------------------------------------------------------
    ax = axes[0, 1]
    ax.hist(scale * 100, bins=25, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.axvline(scale.mean() * 100, color="#d62728", ls="--", lw=1.2,
               label="mean = %.2f%%" % (scale.mean() * 100))
    ax.axvline(3.0, color="grey", ls=":", lw=1.0, label="3% threshold")
    ax.set_xlabel("Relative scale error (%)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Scale-error distribution")
    ax.legend(fontsize=8)

    # (c) realised spatial agreement (IoU) ---------------------------------
    ax = axes[1, 0]
    ax.hist(iou, bins=25, color="#9467bd", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="grey", ls=":", lw=1.0, label="IoU = 0.5")
    ax.axvline(iou.mean(), color="#d62728", ls="--", lw=1.2,
               label="mean = %.4f" % iou.mean())
    ax.set_xlabel("Bounding-box IoU against the requested placement")
    ax.set_ylabel("Count")
    ax.set_title("(c) Spatial agreement (%.1f%% at IoU >= 0.5)" % (100 * frac_iou50))
    ax.legend(fontsize=8)

    # (d) perceptual quality ------------------------------------------------
    ax = axes[1, 1]
    ax.hist(bris, bins=25, color="#ff7f0e", edgecolor="white", alpha=0.85)
    ax.axvline(bris.mean(), color="#d62728", ls="--", lw=1.2,
               label="mean = %.2f" % bris.mean())
    ax.axvline(np.median(bris), color="#1f77b4", ls="-.", lw=1.0,
               label="median = %.2f" % np.median(bris))
    ax.axvline(thr, color="grey", ls=":", lw=1.0, label="threshold (%.0f)" % thr)
    ax.set_xlabel("BRISQUE score (lower is better)")
    ax.set_ylabel("Count")
    ax.set_title("(d) BRISQUE (%d/%d pass)" % (int(round(frac_pass * n)), n))
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("figure 6 ->", out)


def _pick_pairs(full: list, paste: list, bands=(72, 40)) -> list:
    """One matched pair per target-size band (near / mid / far)."""
    pmap = {s["sample_id"]: s for s in paste}
    rows = []
    for lo, hi, tag in ((bands[0], 10 ** 6, "near"), (bands[1], bands[0], "mid"),
                        (0, bands[1], "far")):
        for s in full:
            sid = s["sample_id"]
            if sid in pmap and lo <= s.get("target_px", 0) < hi:
                rows.append((tag, s, pmap[sid]))
                break
    return rows


def fig7(rows: list, out: Path, roots: dict) -> None:
    if not rows:
        print("figure 7: no matched pairs found")
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.6 * n), squeeze=False)

    for r, (tag, inj, pas) in enumerate(rows):
        for c, (arm, rec) in enumerate((("hard paste", pas), ("injection", inj))):
            root = roots["paste"] if c == 0 else roots["injection"]
            ax = axes[r, c]
            img = Image.open(root / rec["rgb"]).convert("RGB")
            ax.imshow(img)
            bbox = rec.get("bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, edgecolor="#00ff00", linewidth=1.4))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("(a) Hard-paste baseline" if c == 0
                             else "(b) Marker injection + ControlNet", fontsize=9)
            ax.set_xlabel("%s | %s %s | requested %d px"
                          % (rec.get("scene", ""), rec.get("time_of_day", ""),
                             rec.get("weather", ""), int(rec.get("target_px", 0))),
                          fontsize=7)
        axes[r, 0].set_ylabel("%s target" % tag, fontsize=8)

    fig.suptitle("Ablation E1: marker injection versus hard paste at matched conditioning points",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("figure 7 ->", out)


def main() -> None:
    samples, summ = load_d()
    full = json.loads((HERE / "out_E1_full" / "manifest.json").read_text(encoding="utf-8"))["samples"]
    paste = json.loads((HERE / "out_E1_paste" / "manifest.json").read_text(encoding="utf-8"))["samples"]

    for target in (OUT_DIR, PAPER_FIGS):
        fig6(samples, summ, target / "fig6_alignment.png")
        fig7(_pick_pairs(full, paste), target / "fig7_ablation.png",
             {"injection": HERE / "out_E1_full", "paste": HERE / "out_E1_paste"})

    meta = {
        "benchmark_pairs": len(samples),
        "pos_err_mean": float(np.mean([s["alignment"]["pos_err_px"] for s in samples])),
        "scale_err_mean": float(np.mean([s["alignment"]["scale_err"] for s in samples])),
        "iou_mean": float(np.mean([s["alignment"]["iou"] for s in samples])),
        "brisque_mean": float(np.mean([s["brisque"] for s in samples])),
        "e1_pairs_used": [r[1]["sample_id"] for r in _pick_pairs(full, paste)],
    }
    (OUT_DIR / "figure_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
evaluator.py — Chapter 4 experiment executor
=============================================
Reads a folder of generated RGB-IR pairs together with a ``manifest.json``
describing their insertion parameters, computes the three evaluation anchors
(alignment precision, generation quality, diversity) and writes:

    output/results.json   per-sample metrics
    output/summary.json   aggregate statistics
    output/report.md      British-English report with tables
    output/figures/       publication figures (Figure 6, and Figure 7 when
                          ablation pairs are present)

Usage
-----
    python evaluator.py                     # uses ./input and ./output
    python evaluator.py --input input --output output
    python evaluator.py --brisque-thr 45 --cross-modal-thr 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_core as ec

BRISQUE_THR = 45.0
CROSS_MODAL_IOU_THR = 0.95
SCENES = ["pure_sky", "coastal", "urban", "rural", "forest",
          "mountain", "desert", "industrial", "airfield"]


# ---------------------------------------------------------------------------
# Sample-level evaluation
# ---------------------------------------------------------------------------


def _resolve(path_spec, base: Path):
    """Resolve a manifest path (str or None) against the manifest directory."""
    if not path_spec:
        return None
    p = Path(path_spec)
    if not p.is_absolute():
        p = base / p
    return p


def _gt_bbox(sample: dict, base: Path):
    """Return the realised ground-truth box and its provenance.

    Priority: explicit ``bbox`` field > ``drone_mask`` image > ``None``.
    """
    if sample.get("bbox"):
        return np.array(sample["bbox"], dtype=np.float32), "explicit"
    mask = ec.load_image(_resolve(sample.get("drone_mask"), base), colour="grey")
    if mask is not None:
        box = ec.mask_to_bbox(mask)
        if box is not None:
            return box, "mask"
    return None, "unavailable"


def _ir_bbox(sample: dict, base: Path):
    """Return the IR box; ``None`` means it is carried across by construction."""
    if sample.get("ir_bbox"):
        return np.array(sample["ir_bbox"], dtype=np.float32)
    mask = ec.load_image(_resolve(sample.get("ir_drone_mask"), base), colour="grey")
    if mask is not None:
        return ec.mask_to_bbox(mask)
    return None


def evaluate_sample(sample: dict, base: Path, brisque_thr: float,
                    device: str) -> dict:
    """Evaluate a single sample described by a manifest entry."""
    sid = sample.get("sample_id", "unknown")
    rgb = ec.load_image(_resolve(sample.get("rgb"), base), colour="rgb")
    ir = ec.load_image(_resolve(sample.get("ir"), base), colour="rgb")

    out = {
        "sample_id": sid,
        "scene": sample.get("scene"),
        "time_of_day": sample.get("time_of_day"),
        "weather": sample.get("weather", "clear"),
        "nu": sample.get("nu"),
        "nv": sample.get("nv"),
        "target_px": sample.get("target_px"),
        "rgb_loaded": rgb is not None,
        "ir_loaded": ir is not None,
    }

    # -- dimensions ---------------------------------------------------------
    if rgb is not None:
        H, W = rgb.shape[:2]
        out["W"], out["H"] = W, H
    else:
        out["W"] = out["H"] = None

    # -- alignment precision -------------------------------------------------
    gt_bbox, provenance = _gt_bbox(sample, base)
    out["gt_bbox_source"] = provenance
    if gt_bbox is not None and out["W"] is not None and sample.get("nu") is not None:
        out["gt_bbox"] = gt_bbox.tolist()
        out["alignment"] = ec.compute_alignment(
            gt_bbox, out["W"], out["H"], sample["nu"], sample["nv"],
            sample["target_px"],
        )
    else:
        out["gt_bbox"] = None
        out["alignment"] = {"pos_err_px": None, "scale_err": None, "iou": None}

    # -- cross-modal alignment ----------------------------------------------
    ir_box = _ir_bbox(sample, base)
    out["cross_modal"] = ec.cross_modal_check(gt_bbox, ir_box, CROSS_MODAL_IOU_THR)

    # -- generation quality (BRISQUE) ---------------------------------------
    if rgb is not None:
        out["brisque"] = ec.brisque_score(rgb, device=device)
        out["brisque_passed"] = out["brisque"] <= brisque_thr
    else:
        out["brisque"] = None
        out["brisque_passed"] = False

    return out


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def _stat(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None}
    a = np.array(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def summarise(samples: list, brisque_thr: float) -> dict:
    """Aggregate a list of per-sample results into summary statistics."""
    n = len(samples)
    if n == 0:
        return {"n": 0}

    pos = [s["alignment"]["pos_err_px"] for s in samples]
    scale = [s["alignment"]["scale_err"] for s in samples]
    iou = [s["alignment"]["iou"] for s in samples]
    brisque = [s["brisque"] for s in samples]

    # acceptance rates
    pos_ok = sum(1 for p in pos if p is not None and p < 1.0)
    scale_ok = sum(1 for p in scale if p is not None and p < 0.03)
    iou_ok = sum(1 for p in iou if p is not None and p > 0.5)
    brisque_ok = sum(1 for s in samples if s["brisque_passed"])
    cross_ok = sum(1 for s in samples if s["cross_modal"]["passed"])

    # diversity
    nu = [s["nu"] for s in samples if s["nu"] is not None]
    nv = [s["nv"] for s in samples if s["nv"] is not None]
    px = [s["target_px"] for s in samples if s["target_px"] is not None]
    tuples = [(s["nu"], s["nv"], s["target_px"], s.get("time_of_day"))
              for s in samples if s["nu"] is not None]

    summary = {
        "n": n,
        "alignment": {
            "pos_err_px": _stat(pos),
            "scale_err": _stat(scale),
            "iou": _stat(iou),
            "sub_pixel_rate": pos_ok / n if pos else None,
            "scale_below_3pct_rate": scale_ok / n if scale else None,
            "iou_above_0p5_rate": iou_ok / n if iou else None,
        },
        "cross_modal_pass_rate": cross_ok / n,
        "quality": {
            "brisque": _stat(brisque),
            "pass_rate": brisque_ok / n,
            "threshold": brisque_thr,
        },
        "diversity": {
            "entropy_nu_bits": ec.shannon_entropy(nu),
            "entropy_nv_bits": ec.shannon_entropy(nv),
            "entropy_scale_bits": ec.shannon_entropy(px),
            "distinct_tuples": ec.count_distinct(tuples),
            "total_tuples": len(tuples),
        },
        "by_time_of_day": _by_time_of_day(samples, brisque_thr),
    }
    return summary


def _by_time_of_day(samples: list, brisque_thr: float) -> dict:
    out = {}
    for tod in ("afternoon", "dusk", "night"):
        grp = [s for s in samples if s.get("time_of_day") == tod]
        if not grp:
            continue
        pos = [s["alignment"]["pos_err_px"] for s in grp
               if s["alignment"]["pos_err_px"] is not None]
        scale = [s["alignment"]["scale_err"] for s in grp
                 if s["alignment"]["scale_err"] is not None]
        iou = [s["alignment"]["iou"] for s in grp if s["alignment"]["iou"] is not None]
        br = [s["brisque"] for s in grp if s["brisque"] is not None]
        out[tod] = {
            "n": len(grp),
            "pos_err_px": _stat(pos),
            "scale_err": _stat(scale),
            "iou": _stat(iou),
            "brisque": _stat(br),
            "brisque_pass": sum(1 for s in grp if s["brisque_passed"]),
        }
    return out


# ---------------------------------------------------------------------------
# Ablation (optional)
# ---------------------------------------------------------------------------


def evaluate_ablation(ablation_manifest: Path, brisque_thr: float,
                      device: str) -> list:
    """Evaluate ablation pairs from ``input/ablation/manifest.json``.

    Each pair compares marker injection against hard paste under identical
    insertion parameters.  Returns a list of per-pair dicts.
    """
    base = ablation_manifest.parent
    data = json.loads(ablation_manifest.read_text(encoding="utf-8"))
    pairs = []
    for p in data.get("pairs", []):
        pid = p.get("pair_id", "pair")
        nu, nv, px = p.get("nu"), p.get("nv"), p.get("target_px")
        entry = {"pair_id": pid, "nu": nu, "nv": nv, "target_px": px,
                 "scene": p.get("scene")}
        for arm in ("inject", "paste"):
            spec = p.get(arm, {})
            rgb_path = _resolve(spec.get("rgb"), base)
            rgb = ec.load_image(rgb_path, colour="rgb")
            box, _ = _gt_bbox(spec, base)
            H, W = (rgb.shape[:2] if rgb is not None else (None, None))
            align = (ec.compute_alignment(box, W, H, nu, nv, px)
                     if (box is not None and W is not None and nu is not None)
                     else {"pos_err_px": None, "scale_err": None, "iou": None})
            entry[arm] = {
                "rgb_loaded": rgb is not None,
                "rgb_path": str(rgb_path) if rgb_path is not None else None,
                "gt_bbox": box.tolist() if box is not None else None,
                "alignment": align,
                "brisque": ec.brisque_score(rgb, device=device) if rgb is not None else None,
                "brisque_passed": (ec.brisque_score(rgb, device=device) <= brisque_thr)
                                  if rgb is not None else False,
            }
        pairs.append(entry)
    return pairs


def summarise_ablation(pairs: list) -> dict:
    """Aggregate ablation pairs into per-arm statistics."""
    if not pairs:
        return {}
    out = {"n_pairs": len(pairs)}
    for arm in ("inject", "paste"):
        pos = [p[arm]["alignment"]["pos_err_px"] for p in pairs]
        scale = [p[arm]["alignment"]["scale_err"] for p in pairs]
        iou = [p[arm]["alignment"]["iou"] for p in pairs]
        br = [p[arm]["brisque"] for p in pairs]
        out[arm] = {
            "pos_err_px": _stat(pos),
            "scale_err": _stat(scale),
            "iou": _stat(iou),
            "brisque": _stat(br),
        }
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Chapter 4 experiment executor")
    ap.add_argument("--input", default="input", help="input directory (default: input)")
    ap.add_argument("--output", default="output", help="output directory (default: output)")
    ap.add_argument("--brisque-thr", type=float, default=BRISQUE_THR)
    ap.add_argument("--cross-modal-thr", type=float, default=CROSS_MODAL_IOU_THR)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    manifest = in_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"!! manifest not found: {manifest}")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    samples_raw = data.get("samples", [])
    print(f"Input manifest: {len(samples_raw)} samples")
    print(f"Device: {device}  |  BRISQUE threshold: {args.brisque_thr}")

    samples = [evaluate_sample(s, in_dir, args.brisque_thr, device)
               for s in samples_raw]
    summary = summarise(samples, args.brisque_thr)

    # optional ablation
    ablation_manifest = in_dir / "ablation" / "manifest.json"
    ablation_pairs = []
    ablation_summary = {}
    if ablation_manifest.exists():
        print("Ablation manifest found; evaluating ablation pairs.")
        ablation_pairs = evaluate_ablation(ablation_manifest, args.brisque_thr, device)
        ablation_summary = summarise_ablation(ablation_pairs)

    # write JSON outputs
    meta = {"brisque_threshold": args.brisque_thr,
            "cross_modal_iou_threshold": args.cross_modal_thr,
            "device": device,
            "background_pool_size": data.get("background_pool_size", 28000)}
    (out_dir / "results.json").write_text(
        json.dumps({"metadata": meta, "samples": samples}, indent=2),
        encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"metadata": meta, "summary": summary}, indent=2),
        encoding="utf-8")
    if ablation_pairs:
        (out_dir / "ablation.json").write_text(
            json.dumps({"metadata": meta, "pairs": ablation_pairs}, indent=2),
            encoding="utf-8")
        (out_dir / "ablation_summary.json").write_text(
            json.dumps({"metadata": meta, "summary": ablation_summary}, indent=2),
            encoding="utf-8")

    # report + figures
    import report
    import figures
    report.write_report(out_dir, samples, summary, ablation_pairs,
                        ablation_summary, meta)
    if not args.no_figures:
        figures.make_figure6(out_dir, samples, summary)
        if ablation_pairs:
            figures.make_figure7(out_dir, ablation_pairs)

    print("\nDone.")
    print(f"  results      -> {out_dir / 'results.json'}")
    print(f"  summary      -> {out_dir / 'summary.json'}")
    print(f"  report       -> {out_dir / 'report.md'}")
    print(f"  figures      -> {out_dir / 'figures'}")


if __name__ == "__main__":
    main()

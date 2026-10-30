#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralised path resolution for the CDFF archive.

Single source of truth for every filesystem anchor used by the generation and
validation sides.  Import this module instead of re-deriving paths with
``Path(__file__).parent.parent`` in individual files.

Resolution order for the project root:

1. ``LSS_PROJECT_ROOT`` environment variable (highest priority — use this when
   the archive is checked out somewhere else than the original project tree).
2. The nearest ancestor of this file that contains ``1-background-pool`` or
   ``0-model`` (i.e. the original project root, when the archive lives inside it).
3. The archive directory itself (degraded mode: helpers that only need the
   workspace still import cleanly, but model/pool paths will not exist).

Example
-------
>>> import lss_paths
>>> lss_paths.PROJECT_ROOT
PosixPath('/mnt/d/.../Paper-低慢小数据集生成架构')
>>> lss_paths.POOL_ROOT.is_dir()
True
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "WORKSPACE", "PROJECT_ROOT", "POOL_ROOT", "MODEL_DIR", "BEST_MODELS",
    "SD15_PATH", "CONTROLNET_DEPTH", "CONTROLNET_SEG", "LORA_PATH",
    "APP_DIR", "OUTPUT_DIR", "describe",
]

#: Directory holding this file — the clean archive root (``0-workspace``).
WORKSPACE: Path = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    """Locate the original project root that owns the datasets and weights.

    Two layouts are supported:

    * **Linked** — the archive is checked out inside the project tree and the
      three asset directories are symlinked (``0-model -> ../0-model`` …).
      In that case the symlink targets give the real root.
    * **Standalone** — no symlinks; the root is supplied via
      ``LSS_PROJECT_ROOT`` or falls back to the archive directory itself.
    """
    env = os.environ.get("LSS_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for cand in (WORKSPACE, *WORKSPACE.parents):
        for key in ("1-background-pool", "0-model"):
            p = cand / key
            if p.is_dir():
                if p.is_symlink():
                    # follow the link back to the project that owns the assets
                    return p.resolve().parent
                return cand
    return WORKSPACE


PROJECT_ROOT: Path = _resolve_project_root()

# ── data ──────────────────────────────────────────────────────────────────
POOL_ROOT: Path = PROJECT_ROOT / "1-background-pool" / "curated_backgrounds"

# ── model weights ─────────────────────────────────────────────────────────
MODEL_DIR: Path = PROJECT_ROOT / "0-model"
BEST_MODELS: Path = PROJECT_ROOT / "2-Lora training" / "best_models"
SD15_PATH: Path = MODEL_DIR / "stable-diffusion-v1-5"
CONTROLNET_DEPTH: Path = MODEL_DIR / "sd-controlnet-depth"
CONTROLNET_SEG: Path = MODEL_DIR / "sd-controlnet-seg"
LORA_PATH: Path = BEST_MODELS / "drn3_pocket_uav_v3_step2000.safetensors"

# ── runtime artefacts (kept inside the archive) ───────────────────────────
APP_DIR: Path = WORKSPACE / "app"
OUTPUT_DIR: Path = APP_DIR / "output_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # web 端静态目录，导入即就绪


def describe() -> str:
    """Human-readable report used by the smoke test and the README."""
    rows = [
        ("WORKSPACE", WORKSPACE, "OK  "),
        # SELF = standalone checkout, no external project root was found
        ("PROJECT_ROOT", PROJECT_ROOT,
         "OK  " if PROJECT_ROOT != WORKSPACE else "SELF"),
        ("POOL_ROOT", POOL_ROOT, POOL_ROOT.is_dir()),
        ("MODEL_DIR", MODEL_DIR, MODEL_DIR.is_dir()),
        ("SD15_PATH", SD15_PATH, SD15_PATH.is_dir()),
        ("CONTROLNET_DEPTH", CONTROLNET_DEPTH, CONTROLNET_DEPTH.is_dir()),
        ("CONTROLNET_SEG", CONTROLNET_SEG, CONTROLNET_SEG.is_dir()),
        ("LORA_PATH", LORA_PATH, LORA_PATH.is_file()),
    ]
    out = []
    for name, path, flag in rows:
        mark = flag if isinstance(flag, str) else ("OK  " if flag else "MISS")
        out.append(f"  {mark}  {name:<17} {path}")
    return "\n".join(out)


if __name__ == "__main__":
    print("CDFF path resolution")
    print(describe())

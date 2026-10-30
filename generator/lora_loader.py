#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load a kohya-format LoRA adapter and merge it into an SD 1.5 pipeline.

Kept separate from `lora_inpainter_v7.py` so that the adapter is an explicit,
inspectable extension point rather than a detail buried in the pipeline:

    from generator.lora_loader import inspect_lora, merge_lora_into_unet

    info = inspect_lora("path/to/adapter.safetensors")   # no torch model needed
    merge_lora_into_unet(pipe.unet, "path/to/adapter.safetensors")

What the merge does
-------------------
A kohya adapter stores, per adapted Linear layer, a down projection ``A`` of
shape ``[r, in]``, an up projection ``B`` of shape ``[out, r]`` and a scalar
``alpha``. The folded weight is

    ΔW = (alpha / r) · B @ A            and              W ← W + ΔW

which is exactly the low-rank update the adapter applies at inference, just
applied eagerly so that the ControlNet branches and the img2img pipeline can
share one UNet. The text-encoder half of the adapter (``lora_te_*`` keys) is
**skipped by default**: this pipeline conditions on a prompt that was already
written for the base model, and merging the CLIP half changes prompt weighting
in ways the paper's numbers did not include. Pass ``include_text_encoder=True``
if you want it.

Key naming
----------
`lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q` is the
dotted module path with ``.`` → ``_``, i.e.
`down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q`. The released
adapter uses this layout (576 ``lora_unet_*`` modules + 216 ``lora_te_*``).

Licence
-------
The adapter is a derivative of its base model (CreativeML OpenRAIL-M for
SD 1.5) and carries that licence, not this repository's MIT. See
THIRD_PARTY_LICENSES.md.

CLI
---
    python generator/lora_loader.py ADAPTER.safetensors            # inspect
    python generator/lora_loader.py ADAPTER.safetensors --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREFIX_UNET = "lora_unet_"
PREFIX_TE = "lora_te_"


# ══════════════════════════════════════════════════════════════════════════
# Inspection
# ══════════════════════════════════════════════════════════════════════════

def load_lora_state_dict(path: str | Path) -> dict:
    """Read a ``.safetensors`` adapter into memory (returns a plain dict)."""
    from safetensors.torch import load_file

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"expected .safetensors, got {path.suffix}")
    return load_file(str(path))


def read_lora_metadata(path: str | Path) -> dict:
    """Return the safetensors header metadata (kohya ``ss_*`` keys if present)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def _alpha_and_rank(state: dict, module: str) -> tuple[float, int]:
    alpha = state.get(f"{module}.alpha")
    alpha = float(alpha.item()) if alpha is not None else 1.0
    down = state.get(f"{module}.lora_down.weight")
    rank = int(down.shape[0]) if down is not None else 0
    return alpha, rank


def kohya_deltas(state: dict, prefix: str = PREFIX_UNET) -> dict[str, "object"]:
    """Compute the folded ``ΔW`` for every module under ``prefix``.

    Returns ``{module_name: delta}`` where ``module_name`` is the kohya name
    (``lora_unet_…``). Modules without both halves are skipped, mirroring the
    behaviour of the historical in-pipeline merge.
    """
    import torch  # noqa: F401  (imported for the side effect of a clear error)

    modules = sorted({k[: -len(".alpha")] for k in state if k.endswith(".alpha")})
    modules += sorted({k[: -len(".lora_down.weight")] for k in state
                       if k.endswith(".lora_down.weight")})
    deltas: dict[str, object] = {}
    for module in dict.fromkeys(modules):
        if not module.startswith(prefix):
            continue
        down = state.get(f"{module}.lora_down.weight")
        up = state.get(f"{module}.lora_up.weight")
        if down is None or up is None:
            continue
        a = down.squeeze() if down.dim() > 2 else down
        b = up.squeeze() if up.dim() > 2 else up
        alpha, rank = _alpha_and_rank(state, module)
        if rank == 0:
            rank = a.shape[0]
        deltas[module] = (alpha / rank) * (b @ a)
    return deltas


def inspect_lora(path: str | Path) -> dict:
    """Summarise an adapter without touching a model: keys, rank, halves, metadata."""
    state = load_lora_state_dict(path)
    unet_modules = {k[: -len(".alpha")] for k in state
                    if k.startswith(PREFIX_UNET) and k.endswith(".alpha")}
    te_modules = {k[: -len(".alpha")] for k in state
                  if k.startswith(PREFIX_TE) and k.endswith(".alpha")}
    ranks, alphas = set(), set()
    for module in list(unet_modules)[:200]:
        alpha, rank = _alpha_and_rank(state, module)
        ranks.add(rank)
        alphas.add(alpha)
    meta = read_lora_metadata(path)
    return {
        "path": str(path),
        "tensors": len(state),
        "unet_modules": len(unet_modules),
        "text_encoder_modules": len(te_modules),
        "ranks": sorted(ranks),
        "alphas": sorted(alphas),
        "base_model_version": meta.get("ss_base_model_version"),
        "resolution": meta.get("modelspec.resolution"),
        "trained_steps": meta.get("ss_max_train_steps"),
        "network_dim": meta.get("ss_network_dim"),
        "network_alpha": meta.get("ss_network_alpha"),
        "license": meta.get("modelspec.license", "CreativeML OpenRAIL-M (presumed — derivative of SD 1.5)"),
        "metadata": meta,
    }


# ══════════════════════════════════════════════════════════════════════════
# Merging
# ══════════════════════════════════════════════════════════════════════════

def merge_lora_into_module(model, state: dict, prefix: str = PREFIX_UNET,
                           verbose: bool = True) -> int:
    """Add every folded adapter delta into ``model``'s parameters in place.

    Returns the number of parameters that actually received an update. Shape
    mismatches are handled the way the original in-pipeline merge handled them:
    a 2-D delta is reshaped into a 4-D convolution kernel, and a transposed delta
    is used when that is what fits.
    """
    deltas = kohya_deltas(state, prefix=prefix)
    merged = 0
    for param_name, param in model.named_parameters():
        base = param_name.replace(".weight", "").replace(".bias", "")
        if base == param_name:            # a bare `.bias`, nothing to merge
            continue
        key = prefix + "_".join(base.split("."))
        delta = deltas.get(key)
        if delta is None:
            continue
        delta = delta.to(param.device, param.dtype)
        if delta.shape == param.data.shape:
            param.data += delta
            merged += 1
        elif param.data.dim() == 4 and delta.dim() == 2:
            param.data += delta.reshape(param.data.shape)
            merged += 1
        elif delta.T.shape == param.data.shape:
            param.data += delta.T
            merged += 1
    if verbose:
        print(f"  LoRA merged ({prefix}): {merged}/{len(deltas)}")
    return merged


def merge_lora_into_unet(unet, path: str | Path, verbose: bool = True) -> int:
    """Convenience wrapper: read the adapter, merge the UNet half into ``unet``."""
    state = load_lora_state_dict(path)
    return merge_lora_into_module(unet, state, prefix=PREFIX_UNET, verbose=verbose)


def merge_lora_into_text_encoder(text_encoder, path: str | Path,
                                 verbose: bool = True) -> int:
    """Merge the ``lora_te_*`` half. Off by default — see the module docstring."""
    state = load_lora_state_dict(path)
    return merge_lora_into_module(text_encoder, state, prefix=PREFIX_TE, verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a kohya-format LoRA adapter")
    ap.add_argument("adapter", help="path to *.safetensors")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--full-metadata", action="store_true",
                    help="include every metadata key (kohya writes ~80)")
    args = ap.parse_args(argv)

    info = inspect_lora(args.adapter)
    if not args.full_metadata:
        info.pop("metadata", None)
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    for key, value in info.items():
        print(f"{key:24s} {value}")
    print("\nLicence: this adapter is a derivative of its base model and carries "
          "that licence (CreativeML OpenRAIL-M for SD 1.5), not this "
          "repository's MIT licence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

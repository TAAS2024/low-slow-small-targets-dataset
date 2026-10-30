#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained checks for `generator/lora_loader.py` — no weights required.

    python tests/test_lora_loader.py

Covers the three things that can silently go wrong in an adapter merge:

1. key translation — kohya module names must map back onto SD 1.5 parameter names;
2. numerics — the folded delta must equal ``(alpha / rank) · B @ A``;
3. scope — the UNet half and the text-encoder half must not leak into each other,
   and a 1×1 convolution kernel must be handled by the reshape branch.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.lora_loader import (  # noqa: E402
    inspect_lora, kohya_deltas, load_lora_state_dict,
    merge_lora_into_module, merge_lora_into_unet,
)

ATTN = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
CONV = "down_blocks.0.attentions.0.proj_in"
ATTN_KEY = "lora_unet_" + ATTN.replace(".", "_")
CONV_KEY = "lora_unet_" + CONV.replace(".", "_")
TE_KEY = "lora_te_text_model_encoder_layers_0_self_attn_q_proj"


class FakeUNet(torch.nn.Module):
    """Two parameters whose names match real SD 1.5 UNet modules."""

    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = torch.nn.Module()
        self.down_blocks.add_module("0", torch.nn.Module())
        self.down_blocks.add_module("1", torch.nn.Module())
        blocks = torch.nn.ModuleDict({
            "attentions": torch.nn.ModuleDict({
                "0": torch.nn.ModuleDict({
                    "transformer_blocks": torch.nn.ModuleDict({
                        "0": torch.nn.ModuleDict({
                            "attn1": torch.nn.ModuleDict({
                                "to_q": torch.nn.Linear(4, 8, bias=False)})})})})})})
        self.down_blocks._modules["1"] = blocks
        self.down_blocks._modules["0"] = torch.nn.ModuleDict({
            "attentions": torch.nn.ModuleDict({
                "0": torch.nn.ModuleDict({
                    "proj_in": torch.nn.Conv2d(4, 8, 1, bias=False)})})})


def make_adapter(r: int = 2, alpha: float = 4.0, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    state = {
        f"{ATTN_KEY}.lora_down.weight": torch.randn(r, 4, generator=g),
        f"{ATTN_KEY}.lora_up.weight": torch.randn(8, r, generator=g),
        f"{ATTN_KEY}.alpha": torch.tensor(alpha),
        # conv branch: 4-D 1×1 kernels, as kohya stores them
        f"{CONV_KEY}.lora_down.weight": torch.randn(r, 4, 1, 1, generator=g),
        f"{CONV_KEY}.lora_up.weight": torch.randn(8, r, 1, 1, generator=g),
        f"{CONV_KEY}.alpha": torch.tensor(alpha),
        # text-encoder half must be ignored by the UNet merge
        f"{TE_KEY}.lora_down.weight": torch.randn(r, 4, generator=g),
        f"{TE_KEY}.lora_up.weight": torch.randn(4, r, generator=g),
        f"{TE_KEY}.alpha": torch.tensor(alpha),
    }
    return state


def expected_delta(state: dict, key: str, alpha: float, r: int) -> torch.Tensor:
    a = state[f"{key}.lora_down.weight"].squeeze()
    b = state[f"{key}.lora_up.weight"].squeeze()
    return (alpha / r) * (b @ a)


def main() -> int:
    state = make_adapter()
    failures = []

    # 1 — key translation
    deltas = kohya_deltas(state, prefix="lora_unet_")
    if set(deltas) != {ATTN_KEY, CONV_KEY}:
        failures.append(f"key translation: got {sorted(deltas)}")
    if any(k.startswith("lora_te_") for k in deltas):
        failures.append("text-encoder key leaked into the UNet delta set")

    # 2 — numerics, linear and conv
    unet = FakeUNet()
    before = {n: p.detach().clone() for n, p in unet.named_parameters()}
    merged = merge_lora_into_module(unet, state, verbose=False)
    if merged != 2:
        failures.append(f"merge count: expected 2, got {merged}")

    after = dict(unet.named_parameters())
    lin = after[f"{ATTN}.weight"].detach()
    conv = after[f"{CONV}.weight"].detach()
    if not torch.allclose(lin, before[f"{ATTN}.weight"] + expected_delta(state, ATTN_KEY, 4.0, 2),
                          atol=1e-6):
        failures.append("linear delta mismatch")
    conv_delta = expected_delta(state, CONV_KEY, 4.0, 2).reshape(8, 4, 1, 1)
    if not torch.allclose(conv, before[f"{CONV}.weight"] + conv_delta, atol=1e-6):
        failures.append("conv 1x1 delta mismatch")

    # 3 — file round trip + inspection
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adapter.safetensors"
        save_file(state, str(path), metadata={
            "modelspec.license": "CreativeML OpenRAIL-M",
            "ss_base_model_version": "sd_v1",
            "ss_network_dim": "2",
            "ss_network_alpha": "4.0",
        })
        info = inspect_lora(path)
        if info["unet_modules"] != 2:
            failures.append(f"inspect unet_modules: {info['unet_modules']}")
        if info["text_encoder_modules"] != 1:
            failures.append(f"inspect te_modules: {info['text_encoder_modules']}")
        if info["ranks"] != [2]:
            failures.append(f"inspect ranks: {info['ranks']}")
        if info["alphas"] != [4.0]:
            failures.append(f"inspect alphas: {info['alphas']}")
        if "OpenRAIL" not in info["license"]:
            failures.append(f"licence not surfaced: {info['license']}")

        fresh = FakeUNet()
        fresh_before = {n: p.detach().clone() for n, p in fresh.named_parameters()}
        n = merge_lora_into_unet(fresh, path, verbose=False)
        if n != 2:
            failures.append(f"merge_lora_into_unet merged {n} params")
        applied = (dict(fresh.named_parameters())[f"{ATTN}.weight"].detach()
                   - fresh_before[f"{ATTN}.weight"])
        if not torch.allclose(applied, expected_delta(state, ATTN_KEY, 4.0, 2), atol=1e-6):
            failures.append("file-based merge applies the wrong delta")

        # a missing file must fail loudly, not silently produce an unadapted model
        try:
            load_lora_state_dict(Path(tmp) / "nope.safetensors")
        except FileNotFoundError:
            pass
        else:
            failures.append("missing adapter did not raise")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("lora_loader: 3 groups, all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Models for the controlled Project 4 experiment suite."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.project1_ddpm.model.unet import UNet
from projects.project4_flow_matching.model.dit import SimpleDiT


class ConditionalProject1UNet(UNet):
    """Project 1 U-Net with an additive class embedding in its time features."""

    def __init__(self, *, num_classes: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.class_embedding = nn.Embedding(self.num_classes + 1, self.time_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.shape != (x.shape[0],):
            raise ValueError(f"labels must have shape {(x.shape[0],)}, got {tuple(y.shape)}")
        if torch.any((y < 0) | (y > self.num_classes)):
            raise ValueError(f"labels must be in [0, {self.num_classes}]")
        time_embedding = self.time_embedding(t) + self.class_embedding(y)
        return super().forward(x, t, time_embedding=time_embedding)


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    kind = model_cfg["type"]
    common = {
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "out_channels": int(model_cfg.get("out_channels", 3)),
        "base_channels": int(model_cfg.get("base_channels", 128)),
        "channel_mult": tuple(model_cfg.get("channel_mult", [1, 2, 2, 2])),
        "num_res_blocks": int(model_cfg.get("num_res_blocks", 2)),
        "attn_resolutions": tuple(model_cfg.get("attn_resolutions", [16])),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "image_size": int(model_cfg.get("image_size", 32)),
        "skip_layout": str(model_cfg.get("skip_layout", "full")),
    }
    if kind == "p1_unet":
        return UNet(**common)
    if kind == "p1_unet_conditional":
        return ConditionalProject1UNet(num_classes=int(model_cfg.get("num_classes", 10)), **common)
    if kind == "dit":
        return SimpleDiT(
            img_size=int(model_cfg.get("image_size", 32)),
            patch_size=int(model_cfg.get("patch_size", 2)),
            in_ch=int(model_cfg.get("in_channels", 3)),
            embed_dim=int(model_cfg["embed_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            num_classes=int(model_cfg.get("num_classes", 10)),
            time_scale=float(model_cfg.get("time_scale", 1000.0)),
        )
    raise ValueError(f"unsupported model.type: {kind}")


def predict(model: nn.Module, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    """Predict with normalized FM time, or integer DDPM time for the P1 U-Net."""
    if isinstance(model, ConditionalProject1UNet):
        if labels is None:
            raise ValueError("conditional Project1 U-Net requires labels")
        return model(x, t * 999.0, labels)
    if isinstance(model, UNet):
        return model(x, t * 999.0)
    if isinstance(model, SimpleDiT):
        if labels is None:
            raise ValueError("DiT requires class labels; use the null token for unconditional prediction")
        return model(x, t, labels)
    raise TypeError(f"unsupported model class: {type(model).__name__}")


def initial_state_sha256(model: nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

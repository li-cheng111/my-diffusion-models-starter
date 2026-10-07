"""
Diffusion Policy model for Project 5.

Architecture:
- Vision encoder: small CNN  (TODO 20: implement)
- Conditioning fusion: vision + state + time_embed
- Action denoiser: MLP that predicts noise on action chunk

Contains: TODO 20 (vision conditioning)
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal time embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


# =============================================================================
# TODO 20: Implement vision encoder for image observation
# =============================================================================
class VisionEncoder(nn.Module):
    """
    Small CNN encoder for 64×64 RGB images.

    Input:  (B, 3, 64, 64) image in [-1, 1]
    Output: (B, out_dim) feature

    要求实现一个简单但有效的 CNN:
        - 3-4 个 Conv2d + ReLU 模块，每个 stride 2 做下采样
        - 最后小型 AdaptiveAvgPool2d 空间网格 + Flatten + Linear → out_dim
        - 推荐参数量 < 1M (这是 toy task)

    Tips:
        - 不要用 BatchNorm（小 batch 不稳）— 用 GroupNorm 或 LayerNorm
        - GeLU / SiLU 比 ReLU 略好但 ReLU 也行
        - 如果遇到训练不收敛，先 print 一下 forward output 的均值与方差

    ⚠️ 下面的 forward 已经写好了，它调用的是 `self.net`——
       你的实现必须把网络赋值给 `self.net`（一个 nn.Sequential 就行），
       并且记得先调用 super().__init__()。
    """

    def __init__(self, out_dim=128, in_ch=3, image_size=64):
        super().__init__()
        # ====================================================================
        # TODO 20: 实现 CNN 视觉编码器 (≈ 10-15 行)
        # ====================================================================
        del image_size  # Pooling keeps the encoder resolution agnostic.
        channels = (32, 64, 128)
        layers = []
        current = in_ch
        for width in channels:
            groups = min(8, width)
            layers.extend([
                nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(groups, width),
                nn.SiLU(),
            ])
            current = width
        # Keep a small spatial grid instead of collapsing directly to 1x1.
        # The target and distractors are identified by their positions; a pure
        # global average would make translated scenes nearly indistinguishable.
        spatial_bins = 4
        layers.extend([
            nn.AdaptiveAvgPool2d((spatial_bins, spatial_bins)),
            nn.Flatten(),
            nn.Linear(current * spatial_bins * spatial_bins, out_dim),
            nn.LayerNorm(out_dim),
        ])
        self.net = nn.Sequential(*layers)
        # ====================================================================
        # END TODO 20
        # ====================================================================

    def forward(self, x):
        return self.net(x)


class ResNetVisionEncoder(nn.Module):
    """ImageNet-pretrained ResNet-18 adapter for the bonus comparison."""

    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, out_dim)
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Dataset images are in [-1, 1]; ImageNet weights expect [0, 1]
        # followed by channel-wise mean/std normalization.
        x = ((x + 1.0) * 0.5 - self.mean) / self.std
        return self.backbone(x)


# =============================================================================
# Main diffusion policy
# =============================================================================
class DiffusionPolicy(nn.Module):
    """
    Conditional action diffuser.

    Input:
        action_noisy: (B, H, A) noisy action chunk
        t: (B,) diffusion timestep
        cond: dict containing image (B, 3, H_img, W_img) and state (B, S)

    Output: (B, H, A) predicted noise
    """

    def __init__(
        self,
        horizon=32,
        action_dim=2,
        state_dim=2,
        image_size=64,
        vision_out_dim=128,
        time_emb_dim=64,
        hidden=256,
        use_vision=True,
        vision_encoder="small",
        vision_pretrained=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.use_vision = use_vision

        self.time_emb_dim = time_emb_dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Vision (TODO 20)
        if use_vision:
            if vision_encoder == "resnet18":
                self.vision = ResNetVisionEncoder(
                    out_dim=vision_out_dim, pretrained=vision_pretrained,
                )
            elif vision_encoder in {"small", "cnn"}:
                self.vision = VisionEncoder(out_dim=vision_out_dim, image_size=image_size)
            else:
                raise ValueError(f"Unknown vision_encoder: {vision_encoder}")
        else:
            self.vision = None
            vision_out_dim = 0

        # State encoder (small MLP)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 64), nn.SiLU(), nn.Linear(64, 64)
        )

        # Condition fusion
        cond_dim = time_emb_dim + 64 + vision_out_dim

        # Action denoiser (treat action chunk as flat vector for simplicity)
        flat_action_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(flat_action_dim + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_action_dim),
        )

    def forward(self, action_noisy, t, image=None, state=None):
        """
        action_noisy: (B, H, A)
        t: (B,) timestep
        image: (B, 3, H_img, W_img) or None if use_vision=False
        state: (B, S)
        """
        B = action_noisy.shape[0]
        a_flat = action_noisy.reshape(B, -1)

        # Time embedding
        # DDPM passes integer indices in [0, T). Flow Matching passes
        # continuous time in [0, 1]; scale the latter so the sinusoidal
        # embedding covers a useful frequency range instead of staying almost
        # constant throughout the path.
        embed_t = t * 1000.0 if t.dtype.is_floating_point else t
        t_emb = timestep_embedding(embed_t, self.time_emb_dim)
        t_emb = self.time_proj(t_emb)

        # State embedding
        s_emb = self.state_proj(state) if state is not None else torch.zeros(B, 64, device=a_flat.device)

        # Vision embedding
        cond_parts = [t_emb, s_emb]
        if self.use_vision:
            assert image is not None, "use_vision=True 但 image 是 None"
            v_emb = self.vision(image)
            cond_parts.append(v_emb)

        cond = torch.cat(cond_parts, dim=-1)
        x = torch.cat([a_flat, cond], dim=-1)
        eps = self.net(x)
        return eps.reshape(B, self.horizon, self.action_dim)


if __name__ == "__main__":
    # Sanity check (state-only, skip TODO 20)
    model = DiffusionPolicy(use_vision=False)
    a = torch.randn(4, 32, 2)
    t = torch.randint(0, 100, (4,))
    s = torch.randn(4, 2)
    eps = model(a, t, image=None, state=s)
    print(f"State-only forward: {eps.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")

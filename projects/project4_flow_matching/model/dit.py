"""
Simplified DiT model for CIFAR-10 Flow Matching.

Architecture:
- Input: (B, 3, 32, 32) image
- Patchify with patch_size=2 → 256 tokens
- N DiT blocks with AdaLN-Zero
- Output: (B, 3, 32, 32) velocity prediction

Note: This is a *teaching* implementation. Production DiT would use:
- Larger embed_dim
- Sinusoidal 2D pos embed (here we use learnable)
- T5/CLIP text conditioning (here we only use class label)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, scale, shift):
    """AdaLN modulation. x: (B, N, D), scale/shift: (B, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding. t in [0, 1] for FM, in [0, 1000] for DDPM."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_ch=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # AdaLN-Zero: 6 modulations per block
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        g1, b1, a1, g2, b2, a2 = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), g1, b1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + a1.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), g2, b2)
        x = x + a2.unsqueeze(1) * self.mlp(h)
        return x


class SimpleDiT(nn.Module):
    """DiT for CIFAR-10 (32x32, patch=2, 256 tokens)."""

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_ch=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,  # 真实类别数；null token 用索引 num_classes（与 ConditionalUNet 一致）
        time_scale=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.time_scale = float(time_scale)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.y_emb = nn.Embedding(num_classes + 1, embed_dim)  # +1 for null

        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.adaLN_final[1].weight)
        nn.init.zeros_(self.adaLN_final[1].bias)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, x):
        # x: (B, N, p²*C) → (B, C, H, W)
        B = x.shape[0]
        p = self.patch_size
        C = self.in_ch
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, h * p, w * p)
        return x

    def forward(self, x, t, y):
        """
        x: (B, 3, 32, 32) input image (or noisy image)
        t: (B,) time in [0, 1] for FM
        y: (B,) class label
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        # Condition embedding: time + label
        t_emb = timestep_embedding(t * self.time_scale, self.embed_dim)
        c = self.t_proj(t_emb) + self.y_emb(y)
        for block in self.blocks:
            tokens = block(tokens, c)
        # Final adaLN
        g, b = self.adaLN_final(c).chunk(2, dim=-1)
        tokens = modulate(self.norm_final(tokens), g, b)
        out = self.head(tokens)  # (B, N, p²*C)
        return self.unpatchify(out)


if __name__ == "__main__":
    # Sanity check
    model = SimpleDiT(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = torch.randint(0, 11, (2,))   # 0-9 是真实类别，10 是 null token
    v = model(x, t, y)
    print(f"Input: {x.shape}, Time: {t}, Class: {y}")
    print(f"Output: {v.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    # AdaLN-Zero identity check
    model.eval()
    with torch.no_grad():
        diff = (v - 0).abs().mean()  # head is zero-init, so init output ≈ 0
    print(f"Init output magnitude: {diff:.4e} (should be ~0)")

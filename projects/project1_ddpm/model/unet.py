"""Compact and full DDPM U-Net backbones for unconditional training."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn

from .embedding import SinusoidalPosEmb


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Choose a GroupNorm group count that divides the channel count."""

    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    """Residual convolution block with additive timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        time_bias = self.time_proj(torch.nn.functional.silu(t_emb))
        h = h + time_bias[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions at selected resolutions."""

    def __init__(self, channels: int, zero_init: bool = False) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, 3, channels, height * width)
        query, key, value = qkv.unbind(dim=1)
        query = query * (channels ** -0.5)
        attention = torch.einsum("bci,bcj->bij", query, key).softmax(dim=-1)
        attended = torch.einsum("bij,bcj->bci", attention, value)
        attended = attended.reshape(batch, channels, height, width)
        return x + self.proj(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        downsample: bool,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for block_index in range(num_res_blocks):
            blocks.append(
                ResBlock(
                    in_channels if block_index == 0 else out_channels,
                    out_channels,
                    time_dim,
                    dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        self.downsample = Downsample(out_channels) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attention(x)
        return x


class _UpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        upsample: bool,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for block_index in range(num_res_blocks):
            blocks.append(
                ResBlock(
                    in_channels + skip_channels if block_index == 0 else out_channels,
                    out_channels,
                    time_dim,
                    dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        self.upsample = Upsample(out_channels) if upsample else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "U-Net skip resolution mismatch: "
                f"{tuple(x.shape[-2:])} vs {tuple(skip.shape[-2:])}"
            )
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attention(x)
        return self.upsample(x)


class _FullDownStage(nn.Module):
    """DDPM-style encoder stage that exposes every block as a skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    time_dim,
                    dropout,
                    zero_init=True,
                )
                for index in range(num_res_blocks)
            ]
        )
        self.attention = nn.ModuleList(
            [
                AttentionBlock(out_channels, zero_init=True)
                if use_attention
                else nn.Identity()
                for _ in range(num_res_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for block, attention in zip(self.blocks, self.attention):
            x = attention(block(x, t_emb))
            skips.append(x)
        return x, skips


class _FullUpStage(nn.Module):
    """DDPM-style decoder stage with one skip fusion per residual block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        out_channels: int,
        time_dim: int,
        dropout: float,
        use_attention: bool,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attention = nn.ModuleList()
        current_channels = in_channels
        for skip in skip_channels:
            self.blocks.append(
                ResBlock(
                    current_channels + skip,
                    out_channels,
                    time_dim,
                    dropout,
                    zero_init=True,
                )
            )
            self.attention.append(
                AttentionBlock(out_channels, zero_init=True)
                if use_attention
                else nn.Identity()
            )
            current_channels = out_channels
        self.upsample = Upsample(out_channels) if upsample else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        for block, attention in zip(self.blocks, self.attention):
            if not skips:
                raise ValueError("Full U-Net ran out of skip tensors")
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    "U-Net skip resolution mismatch: "
                    f"{tuple(x.shape[-2:])} vs {tuple(skip.shape[-2:])}"
                )
            x = attention(block(torch.cat([x, skip], dim=1), t_emb))
        return self.upsample(x)


class UNet(nn.Module):
    """U-Net predicting epsilon_theta(x_t, t) with same-shape output."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: Iterable[int] = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
        time_dim: Optional[int] = None,
        skip_layout: str = "compact",
    ) -> None:
        super().__init__()
        if image_size <= 0 or image_size % (2 ** (len(channel_mult) - 1)) != 0:
            raise ValueError("image_size must be divisible by all configured downsampling levels")
        if not channel_mult:
            raise ValueError("channel_mult must contain at least one level")
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")
        skip_layout = skip_layout.lower()
        if skip_layout not in {"compact", "full"}:
            raise ValueError("skip_layout must be 'compact' or 'full'")
        self.skip_layout = skip_layout

        time_input_dim = base_channels
        self.time_dim = time_dim or base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_input_dim),
            nn.Linear(time_input_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        if skip_layout == "full":
            self._build_full(
                base_channels,
                stage_channels=[base_channels * multiplier for multiplier in channel_mult],
                num_res_blocks=num_res_blocks,
                attn_resolutions=set(attn_resolutions),
                dropout=dropout,
                image_size=image_size,
                out_channels=out_channels,
            )
            return

        attention_resolutions = set(attn_resolutions)
        stage_channels = [base_channels * multiplier for multiplier in channel_mult]
        self.down_stages = nn.ModuleList()
        current_channels = base_channels
        for index, stage_channels_out in enumerate(stage_channels):
            resolution = image_size // (2**index)
            self.down_stages.append(
                _DownStage(
                    current_channels,
                    stage_channels_out,
                    self.time_dim,
                    num_res_blocks,
                    dropout,
                    resolution in attention_resolutions,
                    downsample=index < len(stage_channels) - 1,
                )
            )
            current_channels = stage_channels_out

        self.mid = nn.ModuleList(
            [
                ResBlock(current_channels, current_channels, self.time_dim, dropout),
                AttentionBlock(current_channels),
                ResBlock(current_channels, current_channels, self.time_dim, dropout),
            ]
        )

        self.up_stages = nn.ModuleList()
        for index in reversed(range(len(stage_channels))):
            resolution = image_size // (2**index)
            out_channels_stage = stage_channels[index]
            self.up_stages.append(
                _UpStage(
                    current_channels,
                    stage_channels[index],
                    out_channels_stage,
                    self.time_dim,
                    num_res_blocks,
                    dropout,
                    resolution in attention_resolutions,
                    upsample=index > 0,
                )
            )
            current_channels = out_channels_stage

        self.output_norm = _group_norm(current_channels)
        self.output_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def _build_full(
        self,
        base_channels: int,
        stage_channels: list[int],
        num_res_blocks: int,
        attn_resolutions: set[int],
        dropout: float,
        image_size: int,
        out_channels: int,
    ) -> None:
        self.full_down_stages = nn.ModuleList()
        self.full_downsamples = nn.ModuleList()
        skip_channel_stack = [base_channels]
        current_channels = base_channels
        for index, out_channels_stage in enumerate(stage_channels):
            resolution = image_size // (2**index)
            self.full_down_stages.append(
                _FullDownStage(
                    current_channels,
                    out_channels_stage,
                    self.time_dim,
                    num_res_blocks,
                    dropout,
                    resolution in attn_resolutions,
                )
            )
            skip_channel_stack.extend([out_channels_stage] * num_res_blocks)
            current_channels = out_channels_stage
            if index < len(stage_channels) - 1:
                self.full_downsamples.append(Downsample(current_channels))
                skip_channel_stack.append(current_channels)

        self.full_mid = nn.ModuleList(
            [
                ResBlock(current_channels, current_channels, self.time_dim, dropout, zero_init=True),
                AttentionBlock(current_channels, zero_init=True),
                ResBlock(current_channels, current_channels, self.time_dim, dropout, zero_init=True),
            ]
        )

        self.full_up_stages = nn.ModuleList()
        for index in reversed(range(len(stage_channels))):
            count = num_res_blocks + 1
            skip_channels = [skip_channel_stack.pop() for _ in range(count)]
            resolution = image_size // (2**index)
            out_channels_stage = stage_channels[index]
            self.full_up_stages.append(
                _FullUpStage(
                    current_channels,
                    skip_channels,
                    out_channels_stage,
                    self.time_dim,
                    dropout,
                    resolution in attn_resolutions,
                    upsample=index > 0,
                )
            )
            current_channels = out_channels_stage
        if skip_channel_stack:
            raise RuntimeError("Full U-Net skip channel construction is unbalanced")
        self.output_norm = _group_norm(current_channels)
        self.output_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        h = self.input_conv(x)
        if self.skip_layout == "full":
            skips: list[torch.Tensor] = [h]
            for index, stage in enumerate(self.full_down_stages):
                h, stage_skips = stage(h, t_emb)
                skips.extend(stage_skips)
                if index < len(self.full_downsamples):
                    h = self.full_downsamples[index](h)
                    skips.append(h)
            for block in self.full_mid:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            for stage in self.full_up_stages:
                h = stage(h, skips, t_emb)
            if skips:
                raise ValueError("Full U-Net did not consume all skip tensors")
            return self.output_conv(torch.nn.functional.silu(self.output_norm(h)))

        skips: List[torch.Tensor] = []
        for stage in self.down_stages:
            h = stage(h, t_emb)
            skips.append(h)
            h = stage.downsample(h)

        for block in self.mid:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        for stage, skip in zip(self.up_stages, reversed(skips)):
            h = stage(h, skip, t_emb)

        return self.output_conv(torch.nn.functional.silu(self.output_norm(h)))


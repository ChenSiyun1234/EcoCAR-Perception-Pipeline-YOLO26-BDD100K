"""
Modular lane segmentation decoders.

Supports: CNN, Transformer (MUSTER-like), FPN, ASPP.
All take in_channels_list and produce (B, 1, mask_h, mask_w) logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ── CNN Decoder ───────────────────────────────────────────────────────────────

class CNNLaneHead(nn.Module):
    """Simple CNN decoder: concat features -> fuse -> upsample -> logits."""

    def __init__(self, in_channels_list: List[int], hidden_channels: int = 64,
                 mask_height: int = 180, mask_width: int = 320):
        super().__init__()
        self.mask_height = mask_height
        self.mask_width = mask_width
        total_in = sum(in_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_channels // 2, 1, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].shape[2:]
        aligned = []
        for feat in features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode="bilinear", align_corners=False)
            aligned.append(feat)
        x = self.fuse(torch.cat(aligned, dim=1))
        x = F.interpolate(x, size=(self.mask_height, self.mask_width),
                          mode="bilinear", align_corners=False)
        return self.head(x)


# ── FPN Decoder ───────────────────────────────────────────────────────────────

class FPNLaneHead(nn.Module):
    """
    FPN-style decoder: project each scale to same dim, progressive upsample+add,
    then decode to mask. More memory-efficient than concat and better for thin
    structures due to progressive spatial refinement.
    """

    def __init__(self, in_channels_list: List[int], hidden_channels: int = 64,
                 mask_height: int = 180, mask_width: int = 320):
        super().__init__()
        self.mask_height = mask_height
        self.mask_width = mask_width

        # Lateral projections (1x1 conv to common dim)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ) for ch in in_channels_list
        ])

        # Refinement convs after each add
        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ) for _ in in_channels_list
        ])

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Project all scales
        laterals = [proj(feat) for proj, feat in zip(self.laterals, features)]

        # Top-down pathway: start from smallest (last), upsample and add
        out = laterals[-1]
        out = self.refines[-1](out)

        for i in range(len(laterals) - 2, -1, -1):
            target_size = laterals[i].shape[2:]
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            out = out + laterals[i]
            out = self.refines[i](out)

        out = F.interpolate(out, size=(self.mask_height, self.mask_width),
                            mode="bilinear", align_corners=False)
        return self.head(out)


# ── ASPP Decoder ──────────────────────────────────────────────────────────────

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with multiple dilation rates."""

    def __init__(self, in_ch: int, out_ch: int, rates: List[int] = [6, 12, 18]):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        # Global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for conv in self.convs[:-1]:
            outs.append(conv(x))
        # Global pooling branch — upsample back
        gap = self.convs[-1](x)
        gap = F.interpolate(gap, size=x.shape[2:], mode="bilinear", align_corners=False)
        outs.append(gap)
        return self.project(torch.cat(outs, dim=1))


class ASPPLaneHead(nn.Module):
    """ASPP + simple decode head for lane segmentation."""

    def __init__(self, in_channels_list: List[int], hidden_channels: int = 64,
                 mask_height: int = 180, mask_width: int = 320):
        super().__init__()
        self.mask_height = mask_height
        self.mask_width = mask_width

        total_in = sum(in_channels_list)
        self.pre_fuse = nn.Sequential(
            nn.Conv2d(total_in, hidden_channels * 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPPModule(hidden_channels * 2, hidden_channels, rates=[3, 6, 9])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].shape[2:]
        aligned = []
        for feat in features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode="bilinear", align_corners=False)
            aligned.append(feat)
        x = self.pre_fuse(torch.cat(aligned, dim=1))
        x = self.aspp(x)
        x = F.interpolate(x, size=(self.mask_height, self.mask_width),
                          mode="bilinear", align_corners=False)
        return self.head(x)


# ── Transformer Decoder (cleaned-up MUSTER-like) ─────────────────────────────

class SpatialReductionAttention(nn.Module):
    """SRA: full-res queries, spatially-reduced keys/values."""

    def __init__(self, embed_dim: int, num_heads: int, reduction_ratio: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=reduction_ratio,
                            stride=reduction_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_sr = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
        x_sr = self.norm(x_sr)

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerLaneHead(nn.Module):
    """Lightweight transformer lane head with SRA for memory efficiency."""

    def __init__(self, in_channels_list: List[int], embed_dim: int = 96,
                 num_heads: int = 4, depth: int = 2,
                 mask_height: int = 180, mask_width: int = 320):
        super().__init__()
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.embed_dim = embed_dim

        self.projections = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, 1) for ch in in_channels_list
        ])
        self.fusion_conv = nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                SpatialReductionAttention(embed_dim, num_heads, reduction_ratio=8),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                ),
            ]))

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].shape[2:]
        projected = []
        for i, feat in enumerate(features):
            proj = self.projections[i](feat)
            if proj.shape[2:] != (target_h, target_w):
                proj = F.interpolate(proj, size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
            projected.append(proj)

        x = self.fusion_conv(torch.cat(projected, dim=1))
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        for norm1, attn, norm2, mlp in self.blocks:
            x_flat = x_flat + attn(norm1(x_flat), H, W)
            x_flat = x_flat + mlp(norm2(x_flat))

        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        x = F.interpolate(x, size=(self.mask_height, self.mask_width),
                          mode='bilinear', align_corners=False)
        return self.head(x)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_lane_head(head_cfg: dict, in_channels_list: List[int],
                    mask_height: int = 180, mask_width: int = 320) -> nn.Module:
    """Build lane head from config."""
    head_type = head_cfg.get("type", "fpn")

    kwargs = dict(
        in_channels_list=in_channels_list,
        mask_height=mask_height,
        mask_width=mask_width,
    )

    if head_type == "cnn":
        return CNNLaneHead(hidden_channels=head_cfg.get("hidden_channels", 64), **kwargs)
    elif head_type == "fpn":
        return FPNLaneHead(hidden_channels=head_cfg.get("hidden_channels", 64), **kwargs)
    elif head_type == "aspp":
        return ASPPLaneHead(hidden_channels=head_cfg.get("hidden_channels", 64), **kwargs)
    elif head_type == "transformer":
        return TransformerLaneHead(
            embed_dim=head_cfg.get("embed_dim", 96),
            num_heads=head_cfg.get("num_heads", 4),
            depth=head_cfg.get("depth", 2),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown lane head type: {head_type}")

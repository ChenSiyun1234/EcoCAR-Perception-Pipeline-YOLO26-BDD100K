"""
transformer_lane_head.py
Lightweight Transformer-based Lane Segmentation Head inspired by MUSTER.

This module implements a simplified, MMSeg-independent decode head designed
specifically for joint YOLO training in Colab environments.

Key Features:
- Multi-scale feature projection.
- Cross-scale fusion.
- Spatial Reduction Attention (SRA) to keep computation O(N) instead of O(N^2).
- Returns 1-channel binary mask logits.

Used by:
  - 08_joint_training.ipynb
  - 09_joint_inference.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialReductionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, reduction_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        
        self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Spatial reduction for K and V
        x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_sr = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)  # B, N', C
        x_sr = self.norm(x_sr)
        
        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B, num_heads, N', head_dim
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, reduction_ratio=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SpatialReductionAttention(embed_dim, num_heads, reduction_ratio)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class LightMUSTERLaneHead(nn.Module):
    def __init__(
        self, 
        in_channels_list, 
        embed_dim=96, 
        num_heads=4, 
        depth=2, 
        mask_height=180, 
        mask_width=320
    ):
        super().__init__()
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.embed_dim = embed_dim
        
        # 1. Project each scale to embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, 1) for in_channels in in_channels_list
        ])
        
        # 2. Linear projection after concatenating scaled features
        self.fusion_conv = nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1)
        
        # 3. Lightweight Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, reduction_ratio=8, mlp_ratio=4) 
            for _ in range(depth)
        ])
        
        # 4. Final upsampling & logits
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, 1)
        )

    def forward(self, features):
        """
        Args:
            features: List of [P3, P4, P5] tensors from neck.
        Returns:
            Logits tensor of shape (B, 1, mask_height, mask_width).
        """
        # Target spatial resolution from P3
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        
        projected = []
        for i, feat in enumerate(features):
            proj = self.projections[i](feat)
            if proj.shape[2:] != (target_h, target_w):
                proj = F.interpolate(proj, size=(target_h, target_w), mode='bilinear', align_corners=False)
            projected.append(proj)
            
        # Concatenate and fuse conceptually matching MUSTER multiscale awareness
        x = torch.cat(projected, dim=1)  # B, 3*embed_dim, H, W
        x = self.fusion_conv(x)          # B, embed_dim, H, W
        
        # Process with Transformer
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, H*W, C
        for block in self.blocks:
            x_flat = block(x_flat, H, W)
            
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Upsample to target mask resolution
        x = F.interpolate(x, size=(self.mask_height, self.mask_width), mode='bilinear', align_corners=False)
        logits = self.head(x)
        
        return logits

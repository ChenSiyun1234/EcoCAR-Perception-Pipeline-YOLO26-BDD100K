"""
Lightweight transformer encoder for multi-scale feature enhancement.

Inspired by RT-DETR's hybrid encoder: rather than encoding every scale
with heavy self-attention, we flatten + encode only P4 (mid resolution)
and fuse the result back into P3/P5 with simple addition.  This keeps
the encoder tractable for real-time inference.

Each branch (detection, lane) can optionally have its own encoder,
or share a single one.
"""

import math
import torch
import torch.nn as nn


class PositionEncoding2D(nn.Module):
    """Fixed 2-D sinusoidal position encoding (same as DETR)."""

    def __init__(self, d_model: int = 256, temperature: float = 10000.0,
                 max_h: int = 128, max_w: int = 128):
        super().__init__()
        pe = torch.zeros(d_model, max_h, max_w)
        half = d_model // 2
        y_pos = torch.arange(max_h).unsqueeze(1).float()
        x_pos = torch.arange(max_w).unsqueeze(0).float()
        div = torch.exp(torch.arange(0, half, 2).float() * (-math.log(temperature) / half))

        pe[0:half:2, :, :] = torch.sin(y_pos * div.view(-1, 1, 1))
        pe[1:half:2, :, :] = torch.cos(y_pos * div.view(-1, 1, 1))
        pe[half::2, :, :]  = torch.sin(x_pos * div.view(-1, 1, 1))
        pe[half+1::2, :, :] = torch.cos(x_pos * div.view(-1, 1, 1))
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position encoding to (B, D, H, W) tensor."""
        return x + self.pe[:, :, :x.shape[2], :x.shape[3]]


class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer: self-attention + FFN."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class HybridEncoder(nn.Module):
    """Lightweight hybrid encoder operating on flattened P4 features.

    Steps:
      1. Flatten P4 → (B, H*W, D)
      2. Add 2D position encoding
      3. Run N transformer encoder layers
      4. Reshape back to (B, D, H, W)

    P3 and P5 pass through unchanged.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 ffn_dim: int = 1024, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.pos_enc = PositionEncoding2D(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, features: list) -> list:
        """features: [P3, P4, P5] each (B, D, H, W)."""
        p3, p4, p5 = features

        # Encode P4 with self-attention
        p4_pe = self.pos_enc(p4)
        B, D, H, W = p4_pe.shape
        tokens = p4_pe.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
        for layer in self.layers:
            tokens = layer(tokens)
        p4_enc = tokens.permute(0, 2, 1).view(B, D, H, W)

        return [p3, p4_enc, p5]

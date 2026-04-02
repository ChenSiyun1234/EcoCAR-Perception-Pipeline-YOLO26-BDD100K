"""
Query-based transformer lane prediction head.

Each lane query predicts a structured lane representation:
  - existence probability (is there a lane?)
  - ordered polyline: N points (x, y) in normalized image coordinates
  - per-point visibility mask
  - lane type classification

Design references:
  - MapTR (query-based vectorized map elements)
  - CLRNet (lane detection with row anchors + refinement)

The decoder predicts lanes as ordered point sequences, NOT raster masks.
This is the key departure from the old pipeline.

TODO: future temporal extension
  - Add a temporal memory bank that caches lane queries across frames
  - StreamMapNet-style temporal propagation
  - The query_features output is designed to feed into such a memory bank
"""

import torch
import torch.nn as nn
from typing import Dict, List


class LaneDecoderLayer(nn.Module):
    """Same structure as detection decoder layer."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, memory, memory)[0]
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class LaneHead(nn.Module):
    """Query-based lane prediction head.

    Each lane query decodes to:
      - exist_logit: scalar (lane existence)
      - points: (N, 2) ordered polyline in normalized coords [0, 1]
      - visibility: (N,) per-point visibility logit
      - type_logits: (num_lane_types,) lane type classification

    Args:
        num_lane_types: number of lane type categories
        num_points: points per lane polyline
        d_model: feature / query dimension
        num_queries: number of lane queries (max lanes per image)
    """

    def __init__(self, num_lane_types: int = 7, num_points: int = 72,
                 d_model: int = 256, nhead: int = 8, ffn_dim: int = 1024,
                 num_layers: int = 3, num_queries: int = 10,
                 dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([
            LaneDecoderLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Prediction heads
        self.exist_head = nn.Linear(d_model, 1)
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_points * 2),
        )
        self.vis_head = nn.Linear(d_model, num_points)
        self.type_head = nn.Linear(d_model, num_lane_types)

        self._init_weights()

    def _init_weights(self):
        # Initialize existence head with negative bias (most queries = no lane)
        nn.init.constant_(self.exist_head.bias, -3.0)
        for m in self.point_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _flatten_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for feat in features:
            tokens.append(feat.flatten(2).permute(0, 2, 1))
        return torch.cat(tokens, dim=1)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: list of [P3, P4, P5] from backbone/encoder

        Returns:
            dict with:
              exist_logits: (B, Q, 1)
              pred_points:  (B, Q, N, 2) — sigmoid-activated normalized coords
              vis_logits:   (B, Q, N) — per-point visibility logits
              type_logits:  (B, Q, num_lane_types)
              query_features: (B, Q, D) — for cross-branch / temporal memory
        """
        B = features[0].shape[0]
        memory = self._flatten_features(features)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, memory)

        exist_logits = self.exist_head(queries)               # (B, Q, 1)
        raw_points = self.point_head(queries)                 # (B, Q, N*2)
        pred_points = raw_points.view(B, self.num_queries, self.num_points, 2).sigmoid()
        vis_logits = self.vis_head(queries)                   # (B, Q, N)
        type_logits = self.type_head(queries)                 # (B, Q, T)

        return {
            "exist_logits": exist_logits,
            "pred_points": pred_points,
            "vis_logits": vis_logits,
            "type_logits": type_logits,
            "query_features": queries,  # TODO: feed into temporal memory bank
        }

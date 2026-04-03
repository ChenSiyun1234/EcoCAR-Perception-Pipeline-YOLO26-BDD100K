"""
MapTRv2-style query-based lane prediction head with explicit polyline anchors.

Upgrades over the previous simplified head:
  - multi-scale 2D positional encoding
  - lane-anchor priors per query
  - iterative point refinement around anchors
  - auxiliary per-layer predictions

The output is still compatible with the rest of the project.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from .detection_head import build_2d_sincos_pos_embed


class LaneDecoderLayer(nn.Module):
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
    def __init__(self, num_lane_types: int = 7, num_points: int = 72,
                 d_model: int = 256, nhead: int = 8, ffn_dim: int = 1024,
                 num_layers: int = 3, num_queries: int = 10,
                 dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([
            LaneDecoderLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.exist_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_layers)])
        self.point_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_points * 2),
            ) for _ in range(num_layers)
        ])
        self.vis_heads = nn.ModuleList([nn.Linear(d_model, num_points) for _ in range(num_layers)])
        self.type_heads = nn.ModuleList([nn.Linear(d_model, num_lane_types) for _ in range(num_layers)])

        y = torch.linspace(0.15, 0.95, num_points)
        priors = []
        centers = torch.linspace(0.15, 0.85, num_queries)
        for cx in centers:
            x = torch.full_like(y, cx)
            priors.append(torch.stack([x, y], dim=-1))
        self.lane_priors = nn.Parameter(torch.stack(priors, dim=0))
        self._init_weights()

    def _init_weights(self):
        for head in self.exist_heads:
            nn.init.constant_(head.bias, -2.5)
        for mod in self.point_heads:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def _flatten_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for feat in features:
            b, c, h, w = feat.shape
            pos = build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)
            pos = pos.unsqueeze(0).expand(b, -1, -1)
            tok = feat.flatten(2).permute(0, 2, 1) + pos
            tokens.append(tok)
        return torch.cat(tokens, dim=1)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        b = features[0].shape[0]
        memory = self._flatten_features(features)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        queries = queries + self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
        priors = self.lane_priors.unsqueeze(0).expand(b, -1, -1, -1)

        aux_outputs = []
        exist_logits = None
        pred_points = None
        vis_logits = None
        type_logits = None
        ref_points = priors
        for layer, exist_head, point_head, vis_head, type_head in zip(
            self.layers, self.exist_heads, self.point_heads, self.vis_heads, self.type_heads
        ):
            queries = layer(queries, memory)
            exist_logits = exist_head(queries)
            point_delta = point_head(queries).view(b, self.num_queries, self.num_points, 2)
            pred_points = (ref_points + 0.15 * torch.tanh(point_delta)).clamp(0.0, 1.0)
            vis_logits = vis_head(queries)
            type_logits = type_head(queries)
            ref_points = pred_points.detach()
            aux_outputs.append({
                "exist_logits": exist_logits,
                "pred_points": pred_points,
                "vis_logits": vis_logits,
                "type_logits": type_logits,
            })

        return {
            "exist_logits": exist_logits,
            "pred_points": pred_points,
            "vis_logits": vis_logits,
            "type_logits": type_logits,
            "query_features": queries,
            "aux_outputs": aux_outputs[:-1],
        }

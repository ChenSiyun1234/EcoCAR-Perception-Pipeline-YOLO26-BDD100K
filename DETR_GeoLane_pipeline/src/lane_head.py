"""
CLRerNet / MapTR-inspired lane head.

Adapted ideas:
- learned lane priors / anchors rather than free-form MLP-only decoding
- iterative refinement of polyline geometry across decoder layers
- query-based set prediction compatible with Hungarian assignment
- outputs query features for future temporal memory and association
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


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
                 num_layers: int = 4, num_queries: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points
        self.d_model = d_model
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([LaneDecoderLayer(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)])
        self.exist_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_layers)])
        self.type_heads = nn.ModuleList([nn.Linear(d_model, num_lane_types) for _ in range(num_layers)])
        self.vis_heads = nn.ModuleList([nn.Linear(d_model, num_points) for _ in range(num_layers)])
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_points * 2),
            )
            for _ in range(num_layers)
        ])
        self.register_buffer("lane_priors", self._build_lane_priors(num_queries, num_points), persistent=False)
        self._init_weights()

    def _build_lane_priors(self, q: int, n: int) -> torch.Tensor:
        priors = []
        xs = torch.linspace(0.12, 0.88, q)
        ys = torch.linspace(0.20, 0.98, n)
        for i in range(q):
            x = torch.full((n,), xs[i])
            priors.append(torch.stack([x, ys], dim=-1))
        return torch.stack(priors, dim=0)

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        for head in self.exist_heads:
            nn.init.constant_(head.bias, -2.5)
        for block in self.delta_heads:
            for m in block:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def _flatten_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for feat in features:
            tokens.append(feat.flatten(2).permute(0, 2, 1))
        return torch.cat(tokens, dim=1)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = features[0].shape[0]
        memory = self._flatten_features(features)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        curve = self.lane_priors.unsqueeze(0).expand(B, -1, -1, -1).clone()

        aux_exist, aux_points, aux_vis, aux_type = [], [], [], []
        for layer, exist_head, delta_head, vis_head, type_head in zip(
                self.layers, self.exist_heads, self.delta_heads, self.vis_heads, self.type_heads):
            queries = layer(queries, memory)
            delta = delta_head(queries).view(B, self.num_queries, self.num_points, 2)
            curve = torch.sigmoid(torch.logit(curve.clamp(1e-4, 1 - 1e-4)) + delta)
            aux_exist.append(exist_head(queries))
            aux_points.append(curve)
            aux_vis.append(vis_head(queries))
            aux_type.append(type_head(queries))

        return {
            "exist_logits": aux_exist[-1],
            "pred_points": aux_points[-1],
            "vis_logits": aux_vis[-1],
            "type_logits": aux_type[-1],
            "aux_exist_logits": aux_exist[:-1],
            "aux_pred_points": aux_points[:-1],
            "aux_vis_logits": aux_vis[:-1],
            "aux_type_logits": aux_type[:-1],
            "query_features": queries,
        }

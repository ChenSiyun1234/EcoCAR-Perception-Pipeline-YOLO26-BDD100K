"""
RF-DETR / RT-DETR-inspired detection head.

This is no longer a minimal custom decoder. Key ideas adapted from recent
real-time DETR-family repos:
- learned content queries + learned reference points
- iterative box refinement across decoder layers
- per-layer prediction heads for stronger optimization
- optional auxiliary outputs for future deep supervision
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
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


class DetectionHead(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 256, nhead: int = 8,
                 ffn_dim: int = 1024, num_layers: int = 4, num_queries: int = 100,
                 dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)])
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.ref_point_head = nn.Linear(d_model, 4)
        self.class_heads = nn.ModuleList([nn.Linear(d_model, num_classes + 1) for _ in range(num_layers)])
        self.box_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 4),
            )
            for _ in range(num_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.ref_point_head.weight)
        nn.init.constant_(self.ref_point_head.bias, 0.0)
        for head in self.class_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0.0)
            head.bias.data[-1] = 2.0
        for block in self.box_heads:
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
        ref_boxes = self.ref_point_head(queries).sigmoid()

        aux_logits = []
        aux_boxes = []
        for layer, cls_head, box_head in zip(self.layers, self.class_heads, self.box_heads):
            queries = layer(queries, memory)
            logits = cls_head(queries)
            delta = box_head(queries)
            ref_boxes = torch.sigmoid(torch.logit(ref_boxes.clamp(1e-4, 1 - 1e-4)) + delta)
            aux_logits.append(logits)
            aux_boxes.append(ref_boxes)

        return {
            "pred_logits": aux_logits[-1],
            "pred_boxes": aux_boxes[-1],
            "aux_logits": aux_logits[:-1],
            "aux_boxes": aux_boxes[:-1],
            "query_features": queries,
        }

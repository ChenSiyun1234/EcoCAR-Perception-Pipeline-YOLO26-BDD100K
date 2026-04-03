"""
RT-DETRv2-style transformer detection decoder.

This version upgrades the original simplified decoder with:
  - multi-scale 2D positional encoding
  - query selection from encoder memory (content-aware init)
  - iterative box refinement with reference boxes
  - auxiliary per-layer outputs for easier future deep supervision

It is still lightweight enough to fit the existing project structure.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


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
                 ffn_dim: int = 1024, num_layers: int = 3, num_queries: int = 100,
                 dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)
        self.memory_score_head = nn.Linear(d_model, 1)
        self.memory_box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4),
        )
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.class_heads = nn.ModuleList([nn.Linear(d_model, num_classes + 1) for _ in range(num_layers)])
        self.box_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 4),
            ) for _ in range(num_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        for head in self.class_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0.0)
            nn.init.constant_(head.bias[-1], 2.0)
        for mod in [self.memory_box_head, *self.box_heads]:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
        nn.init.xavier_uniform_(self.memory_score_head.weight)
        nn.init.constant_(self.memory_score_head.bias, 0.0)

    def _flatten_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for feat in features:
            b, c, h, w = feat.shape
            pos = build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)
            pos = pos.unsqueeze(0).expand(b, -1, -1)
            tok = feat.flatten(2).permute(0, 2, 1) + pos
            tokens.append(tok)
        return torch.cat(tokens, dim=1)

    def _select_queries(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.memory_score_head(memory).squeeze(-1)
        topk = scores.topk(k=min(self.num_queries, memory.shape[1]), dim=1).indices
        gather_idx = topk.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        init_queries = torch.gather(memory, 1, gather_idx)
        ref_boxes = torch.gather(
            self.memory_box_head(memory).sigmoid(),
            1,
            topk.unsqueeze(-1).expand(-1, -1, 4),
        )
        return init_queries, ref_boxes

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        b = features[0].shape[0]
        memory = self._flatten_features(features)
        content_queries, ref_boxes = self._select_queries(memory)
        learned = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
        queries = content_queries + learned + query_pos

        aux_outputs = []
        ref_logits = inverse_sigmoid(ref_boxes)
        pred_logits = None
        pred_boxes = None
        for layer, class_head, box_head in zip(self.layers, self.class_heads, self.box_heads):
            queries = layer(queries, memory)
            pred_logits = class_head(queries)
            box_delta = box_head(queries)
            pred_boxes = (box_delta + ref_logits).sigmoid()
            ref_logits = inverse_sigmoid(pred_boxes.detach())
            aux_outputs.append({"pred_logits": pred_logits, "pred_boxes": pred_boxes})

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "query_features": queries,
            "aux_outputs": aux_outputs[:-1],
        }

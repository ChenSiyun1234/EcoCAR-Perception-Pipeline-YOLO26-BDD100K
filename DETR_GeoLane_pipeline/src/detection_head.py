"""
RT-DETR-inspired transformer detection decoder.

Key ideas borrowed from RT-DETR / DETR:
  - Learned object queries (no anchors)
  - Multi-scale cross-attention over FPN features
  - Direct set prediction with Hungarian matching
  - No NMS required at inference (optional light filtering)

Adapted for vehicle-only detection on BDD100K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class DecoderLayer(nn.Module):
    """Transformer decoder layer: self-attn → cross-attn → FFN."""

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
    """Transformer decoder that predicts vehicle detections.

    Each learned query decodes to:
      - class logits: (num_classes + 1)  [+1 for no-object / background]
      - box: (cx, cy, w, h) in normalized [0, 1] coordinates

    Args:
        num_classes: number of foreground classes (e.g. 5 for vehicle-only)
        d_model: feature / query dimension
        nhead: attention heads
        ffn_dim: FFN hidden dimension
        num_layers: decoder depth
        num_queries: number of object queries
    """

    def __init__(self, num_classes: int, d_model: int = 256, nhead: int = 8,
                 ffn_dim: int = 1024, num_layers: int = 3, num_queries: int = 100,
                 dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.constant_(self.class_head.bias, 0.0)
        # Bias the no-object class slightly positive for faster convergence
        nn.init.constant_(self.class_head.bias[-1], 2.0)
        for layer in self.box_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def _flatten_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Flatten multi-scale FPN features into a single token sequence."""
        tokens = []
        for feat in features:
            B, C, H, W = feat.shape
            tokens.append(feat.flatten(2).permute(0, 2, 1))  # (B, H*W, C)
        return torch.cat(tokens, dim=1)  # (B, total_tokens, C)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: list of [P3, P4, P5] from backbone/encoder

        Returns:
            dict with:
              pred_logits: (B, num_queries, num_classes+1)
              pred_boxes:  (B, num_queries, 4) — sigmoid-activated (cx,cy,w,h)
              query_features: (B, num_queries, d_model) — for cross-branch attention
        """
        B = features[0].shape[0]
        memory = self._flatten_features(features)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, memory)

        pred_logits = self.class_head(queries)
        pred_boxes = self.box_head(queries).sigmoid()

        return {
            "pred_logits": pred_logits,     # (B, Q, C+1)
            "pred_boxes": pred_boxes,       # (B, Q, 4)
            "query_features": queries,      # (B, Q, D)
        }

"""
DualPathNet — weakly-coupled dual-path perception model.

Architecture:

    Image
      |
    ResNet Backbone + FPN  (shared shallow features)
      |
    [P3, P4, P5]
      |
    +------- optional cross-branch attention -------+
    |                                                |
    Det Encoder ──→ Det Decoder (100 queries)        |
    |               → vehicle boxes + classes        |
    |                                                |
    Lane Encoder ──→ Lane Decoder (10 queries)       |
                    → lane polylines + existence

Design goals:
  - Weakly coupled: branches share backbone but have separate encoders/decoders
  - Optional cross-attention between branch queries reduces negative transfer
  - Future-ready: lane query_features can feed a temporal memory bank
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .backbone import BackboneFPN
from .encoder import HybridEncoder
from .detection_head import DetectionHead
from .lane_head import LaneHead
from .config import Config, NUM_CLASSES, NUM_LANE_TYPES


class CrossBranchAttention(nn.Module):
    """Lightweight cross-attention between detection and lane queries.

    Allows each branch to attend to the other's queries without sharing
    heavy encoder/decoder parameters.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 1):
        super().__init__()
        self.det_to_lane = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.lane_to_det = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm_det = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_lane = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, det_queries: torch.Tensor,
                lane_queries: torch.Tensor) -> tuple:
        for attn_d2l, attn_l2d, norm_d, norm_l in zip(
                self.det_to_lane, self.lane_to_det, self.norm_det, self.norm_lane):
            # Lane queries attend to detection queries
            lane_q = norm_l(lane_queries)
            lane_queries = lane_queries + attn_d2l(lane_q, det_queries, det_queries)[0]
            # Detection queries attend to lane queries
            det_q = norm_d(det_queries)
            det_queries = det_queries + attn_l2d(det_q, lane_queries, lane_queries)[0]
        return det_queries, lane_queries


class DualPathNet(nn.Module):
    """Full dual-path perception model.

    Args:
        cfg: Config dataclass with all hyperparameters
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        nc = len(cfg.use_expanded_classes and
                 __import__('src.config', fromlist=['EXPANDED_CLASSES']).EXPANDED_CLASSES
                 or __import__('src.config', fromlist=['VEHICLE_CLASSES']).VEHICLE_CLASSES) \
            if False else (7 if cfg.use_expanded_classes else NUM_CLASSES)

        self.num_classes = nc

        # Shared backbone
        self.backbone = BackboneFPN(
            name=cfg.backbone, pretrained=cfg.pretrained,
            fpn_channels=cfg.fpn_channels,
        )

        # Per-branch encoders (separate to reduce coupling)
        self.det_encoder = HybridEncoder(
            d_model=cfg.det_dim, nhead=cfg.det_nhead,
            ffn_dim=cfg.det_ffn_dim, num_layers=cfg.det_enc_layers,
            dropout=cfg.det_dropout,
        )
        self.lane_encoder = HybridEncoder(
            d_model=cfg.lane_dim, nhead=cfg.lane_nhead,
            ffn_dim=cfg.lane_ffn_dim, num_layers=cfg.lane_enc_layers,
            dropout=cfg.lane_dropout,
        )

        # Projection if FPN channels != branch dimension
        self.det_proj = (nn.Conv2d(cfg.fpn_channels, cfg.det_dim, 1)
                         if cfg.fpn_channels != cfg.det_dim else nn.Identity())
        self.lane_proj = (nn.Conv2d(cfg.fpn_channels, cfg.lane_dim, 1)
                          if cfg.fpn_channels != cfg.lane_dim else nn.Identity())

        # Task-specific decoders
        self.det_head = DetectionHead(
            num_classes=nc, d_model=cfg.det_dim, nhead=cfg.det_nhead,
            ffn_dim=cfg.det_ffn_dim, num_layers=cfg.det_dec_layers,
            num_queries=cfg.det_num_queries, dropout=cfg.det_dropout,
        )
        self.lane_head = LaneHead(
            num_lane_types=NUM_LANE_TYPES, num_points=cfg.lane_points,
            d_model=cfg.lane_dim, nhead=cfg.lane_nhead,
            ffn_dim=cfg.lane_ffn_dim, num_layers=cfg.lane_dec_layers,
            num_queries=cfg.lane_num_queries, dropout=cfg.lane_dropout,
        )

        # Optional cross-branch attention
        self.cross_attn = None
        if cfg.cross_attn:
            self.cross_attn = CrossBranchAttention(
                d_model=cfg.det_dim, nhead=cfg.det_nhead,
                num_layers=cfg.cross_attn_layers,
            )

        # Architecture config for checkpoint reproducibility
        self._arch_config = {
            "backbone": cfg.backbone,
            "num_classes": nc,
            "det_queries": cfg.det_num_queries,
            "lane_queries": cfg.lane_num_queries,
            "lane_points": cfg.lane_points,
            "det_dim": cfg.det_dim,
            "lane_dim": cfg.lane_dim,
            "cross_attn": cfg.cross_attn,
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared backbone
        fpn_features = self.backbone(images)  # [P3, P4, P5]

        # Branch-specific projection + encoding
        det_features = [self.det_proj(f) for f in fpn_features]
        lane_features = [self.lane_proj(f) for f in fpn_features]

        det_features = self.det_encoder(det_features)
        lane_features = self.lane_encoder(lane_features)

        # Task-specific decoders
        det_out = self.det_head(det_features)
        lane_out = self.lane_head(lane_features)

        # Optional cross-branch attention
        if self.cross_attn is not None:
            det_q, lane_q = self.cross_attn(
                det_out["query_features"], lane_out["query_features"])
            # Re-run prediction heads on refined queries
            det_out["pred_logits"] = self.det_head.class_head(det_q)
            det_out["pred_boxes"] = self.det_head.box_head(det_q).sigmoid()
            lane_out["exist_logits"] = self.lane_head.exist_head(lane_q)
            raw_pts = self.lane_head.point_head(lane_q)
            B = images.shape[0]
            lane_out["pred_points"] = raw_pts.view(
                B, self.lane_head.num_queries, self.lane_head.num_points, 2).sigmoid()
            lane_out["vis_logits"] = self.lane_head.vis_head(lane_q)
            lane_out["type_logits"] = self.lane_head.type_head(lane_q)

        return {**{f"det_{k}": v for k, v in det_out.items()},
                **{f"lane_{k}": v for k, v in lane_out.items()}}

    def print_summary(self):
        bb = sum(p.numel() for p in self.backbone.parameters())
        de = sum(p.numel() for p in self.det_encoder.parameters())
        dh = sum(p.numel() for p in self.det_head.parameters())
        le = sum(p.numel() for p in self.lane_encoder.parameters())
        lh = sum(p.numel() for p in self.lane_head.parameters())
        ca = sum(p.numel() for p in self.cross_attn.parameters()) if self.cross_attn else 0
        total = sum(p.numel() for p in self.parameters())
        print(f"DualPathNet summary:")
        print(f"  Backbone+FPN : {bb:>12,}")
        print(f"  Det encoder  : {de:>12,}")
        print(f"  Det decoder  : {dh:>12,}")
        print(f"  Lane encoder : {le:>12,}")
        print(f"  Lane decoder : {lh:>12,}")
        if ca: print(f"  Cross-attn   : {ca:>12,}")
        print(f"  Total        : {total:>12,}")
        print(f"  Det queries={self.cfg.det_num_queries}, "
              f"Lane queries={self.cfg.lane_num_queries}, "
              f"Lane points={self.cfg.lane_points}")


def build_model(cfg: Config) -> DualPathNet:
    """Build DualPathNet from config."""
    return DualPathNet(cfg)

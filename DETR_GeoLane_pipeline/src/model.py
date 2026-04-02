"""
DualPathNet — weakly-coupled dual-path perception model.

Updated to use stronger imported design ideas:
- RT-/RF-DETR-inspired detection head with iterative refinement
- CLRerNet / MapTR-inspired lane head with learned priors and iterative curve refinement
- weak cross-branch query communication to reduce destructive coupling
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import BackboneFPN
from .encoder import HybridEncoder
from .detection_head import DetectionHead
from .lane_head import LaneHead
from .config import Config, NUM_CLASSES, NUM_LANE_TYPES


class CrossBranchAttention(nn.Module):
    def __init__(self, det_dim: int = 256, lane_dim: int = 256, nhead: int = 8, num_layers: int = 1):
        super().__init__()
        self.det_to_lane_proj = nn.Linear(det_dim, lane_dim) if det_dim != lane_dim else nn.Identity()
        self.lane_to_det_proj = nn.Linear(lane_dim, det_dim) if det_dim != lane_dim else nn.Identity()
        self.det_to_lane = nn.ModuleList([nn.MultiheadAttention(lane_dim, nhead, batch_first=True) for _ in range(num_layers)])
        self.lane_to_det = nn.ModuleList([nn.MultiheadAttention(det_dim, nhead, batch_first=True) for _ in range(num_layers)])
        self.norm_det = nn.ModuleList([nn.LayerNorm(det_dim) for _ in range(num_layers)])
        self.norm_lane = nn.ModuleList([nn.LayerNorm(lane_dim) for _ in range(num_layers)])

    def forward(self, det_queries: torch.Tensor, lane_queries: torch.Tensor):
        det_ctx = self.det_to_lane_proj(det_queries)
        lane_ctx = self.lane_to_det_proj(lane_queries)
        for a_d2l, a_l2d, n_d, n_l in zip(self.det_to_lane, self.lane_to_det, self.norm_det, self.norm_lane):
            lane_q = n_l(lane_queries)
            lane_queries = lane_queries + a_d2l(lane_q, det_ctx, det_ctx)[0]
            det_q = n_d(det_queries)
            det_queries = det_queries + a_l2d(det_q, lane_ctx, lane_ctx)[0]
            det_ctx = self.det_to_lane_proj(det_queries)
            lane_ctx = self.lane_to_det_proj(lane_queries)
        return det_queries, lane_queries


class DualPathNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_classes = 7 if cfg.use_expanded_classes else NUM_CLASSES
        self.backbone = BackboneFPN(name=cfg.backbone, pretrained=cfg.pretrained, fpn_channels=cfg.fpn_channels)
        self.det_proj = nn.Conv2d(cfg.fpn_channels, cfg.det_dim, 1) if cfg.fpn_channels != cfg.det_dim else nn.Identity()
        self.lane_proj = nn.Conv2d(cfg.fpn_channels, cfg.lane_dim, 1) if cfg.fpn_channels != cfg.lane_dim else nn.Identity()
        self.det_encoder = HybridEncoder(cfg.det_dim, cfg.det_nhead, cfg.det_ffn_dim, cfg.det_enc_layers, cfg.det_dropout)
        self.lane_encoder = HybridEncoder(cfg.lane_dim, cfg.lane_nhead, cfg.lane_ffn_dim, cfg.lane_enc_layers, cfg.lane_dropout)
        self.det_head = DetectionHead(self.num_classes, cfg.det_dim, cfg.det_nhead, cfg.det_ffn_dim, cfg.det_dec_layers, cfg.det_num_queries, cfg.det_dropout)
        self.lane_head = LaneHead(NUM_LANE_TYPES, cfg.lane_points, cfg.lane_dim, cfg.lane_nhead, cfg.lane_ffn_dim, cfg.lane_dec_layers, cfg.lane_num_queries, cfg.lane_dropout)
        self.cross_attn = CrossBranchAttention(cfg.det_dim, cfg.lane_dim, min(cfg.det_nhead, cfg.lane_nhead), cfg.cross_attn_layers) if cfg.cross_attn else None

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        fpn_features = self.backbone(images)
        det_features = [self.det_proj(f) for f in fpn_features]
        lane_features = [self.lane_proj(f) for f in fpn_features]
        det_features = self.det_encoder(det_features)
        lane_features = self.lane_encoder(lane_features)
        det_out = self.det_head(det_features)
        lane_out = self.lane_head(lane_features)

        if self.cross_attn is not None:
            det_q, lane_q = self.cross_attn(det_out["query_features"], lane_out["query_features"])
            det_out["query_features"] = det_q
            lane_out["query_features"] = lane_q
            det_out["pred_logits"] = self.det_head.class_heads[-1](det_q)
            ref = det_out["pred_boxes"]
            delta = self.det_head.box_heads[-1](det_q)
            det_out["pred_boxes"] = torch.sigmoid(torch.logit(ref.clamp(1e-4, 1 - 1e-4)) + delta)
            lane_out["exist_logits"] = self.lane_head.exist_heads[-1](lane_q)
            lane_out["type_logits"] = self.lane_head.type_heads[-1](lane_q)
            lane_out["vis_logits"] = self.lane_head.vis_heads[-1](lane_q)
            delta_lane = self.lane_head.delta_heads[-1](lane_q).view(images.shape[0], self.lane_head.num_queries, self.lane_head.num_points, 2)
            ref_lane = lane_out["pred_points"]
            lane_out["pred_points"] = torch.sigmoid(torch.logit(ref_lane.clamp(1e-4, 1 - 1e-4)) + delta_lane)

        return {**{f"det_{k}": v for k, v in det_out.items()}, **{f"lane_{k}": v for k, v in lane_out.items()}}

    def print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        print("DualPathNet summary:")
        print(f"  Total params: {total:,}")
        print(f"  Detector queries={self.cfg.det_num_queries}, lane queries={self.cfg.lane_num_queries}, lane points={self.cfg.lane_points}")


def build_model(cfg: Config) -> DualPathNet:
    return DualPathNet(cfg)

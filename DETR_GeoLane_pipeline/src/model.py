"""
DualPathNet — weakly-coupled dual-path perception model.
"""

import torch
import torch.nn as nn
from typing import Dict

from .backbone import BackboneFPN
from .encoder import HybridEncoder
from .detection_head import DetectionHead, inverse_sigmoid
from .lane_head import LaneHead
from .config import Config, NUM_CLASSES, NUM_LANE_TYPES


class CrossBranchAttention(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 1):
        super().__init__()
        self.det_to_lane = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, batch_first=True) for _ in range(num_layers)])
        self.lane_to_det = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, batch_first=True) for _ in range(num_layers)])
        self.norm_det = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_lane = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, det_queries: torch.Tensor, lane_queries: torch.Tensor) -> tuple:
        for attn_d2l, attn_l2d, norm_d, norm_l in zip(self.det_to_lane, self.lane_to_det, self.norm_det, self.norm_lane):
            lane_q = norm_l(lane_queries)
            lane_queries = lane_queries + attn_d2l(lane_q, det_queries, det_queries)[0]
            det_q = norm_d(det_queries)
            det_queries = det_queries + attn_l2d(det_q, lane_queries, lane_queries)[0]
        return det_queries, lane_queries


class DualPathNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        nc = 7 if cfg.use_expanded_classes else NUM_CLASSES
        self.num_classes = nc
        self.backbone = BackboneFPN(name=cfg.backbone, pretrained=cfg.pretrained, fpn_channels=cfg.fpn_channels)
        self.det_encoder = HybridEncoder(d_model=cfg.det_dim, nhead=cfg.det_nhead, ffn_dim=cfg.det_ffn_dim, num_layers=cfg.det_enc_layers, dropout=cfg.det_dropout)
        self.lane_encoder = HybridEncoder(d_model=cfg.lane_dim, nhead=cfg.lane_nhead, ffn_dim=cfg.lane_ffn_dim, num_layers=cfg.lane_enc_layers, dropout=cfg.lane_dropout)
        self.det_proj = nn.Conv2d(cfg.fpn_channels, cfg.det_dim, 1) if cfg.fpn_channels != cfg.det_dim else nn.Identity()
        self.lane_proj = nn.Conv2d(cfg.fpn_channels, cfg.lane_dim, 1) if cfg.fpn_channels != cfg.lane_dim else nn.Identity()
        self.det_head = DetectionHead(num_classes=nc, d_model=cfg.det_dim, nhead=cfg.det_nhead, ffn_dim=cfg.det_ffn_dim, num_layers=cfg.det_dec_layers, num_queries=cfg.det_num_queries, dropout=cfg.det_dropout)
        self.lane_head = LaneHead(num_lane_types=NUM_LANE_TYPES, num_points=cfg.lane_points, d_model=cfg.lane_dim, nhead=cfg.lane_nhead, ffn_dim=cfg.lane_ffn_dim, num_layers=cfg.lane_dec_layers, num_queries=cfg.lane_num_queries, dropout=cfg.lane_dropout)
        self.cross_attn = None
        if cfg.cross_attn and cfg.det_dim == cfg.lane_dim:
            self.cross_attn = CrossBranchAttention(d_model=cfg.det_dim, nhead=cfg.det_nhead, num_layers=cfg.cross_attn_layers)
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
        fpn_features = self.backbone(images)
        det_features = self.det_encoder([self.det_proj(f) for f in fpn_features])
        lane_features = self.lane_encoder([self.lane_proj(f) for f in fpn_features])
        det_out = self.det_head(det_features)
        lane_out = self.lane_head(lane_features)

        if self.cross_attn is not None:
            det_q, lane_q = self.cross_attn(det_out["query_features"], lane_out["query_features"])
            det_out["query_features"] = det_q
            lane_out["query_features"] = lane_q
            det_logits = self.det_head.class_heads[-1](det_q)
            det_boxes = (self.det_head.box_heads[-1](det_q) + inverse_sigmoid(det_out["pred_boxes"].detach())).sigmoid()
            lane_exist = self.lane_head.exist_heads[-1](lane_q)
            lane_delta = self.lane_head.point_heads[-1](lane_q).view(images.shape[0], self.lane_head.num_queries, self.lane_head.num_points, 2)
            lane_points = (lane_out["pred_points"].detach() + 0.10 * torch.tanh(lane_delta)).clamp(0.0, 1.0)
            lane_vis = self.lane_head.vis_heads[-1](lane_q)
            lane_type = self.lane_head.type_heads[-1](lane_q)
            det_out["pred_logits"] = det_logits
            det_out["pred_boxes"] = det_boxes
            lane_out["exist_logits"] = lane_exist
            lane_out["pred_points"] = lane_points
            lane_out["vis_logits"] = lane_vis
            lane_out["type_logits"] = lane_type

        return {**{f"det_{k}": v for k, v in det_out.items()}, **{f"lane_{k}": v for k, v in lane_out.items()}}

    def print_summary(self):
        bb = sum(p.numel() for p in self.backbone.parameters())
        de = sum(p.numel() for p in self.det_encoder.parameters())
        dh = sum(p.numel() for p in self.det_head.parameters())
        le = sum(p.numel() for p in self.lane_encoder.parameters())
        lh = sum(p.numel() for p in self.lane_head.parameters())
        ca = sum(p.numel() for p in self.cross_attn.parameters()) if self.cross_attn else 0
        total = sum(p.numel() for p in self.parameters())
        print("DualPathNet summary:")
        print(f"  Backbone+FPN : {bb:>12,}")
        print(f"  Det encoder  : {de:>12,}")
        print(f"  Det decoder  : {dh:>12,}")
        print(f"  Lane encoder : {le:>12,}")
        print(f"  Lane decoder : {lh:>12,}")
        if ca:
            print(f"  Cross-attn   : {ca:>12,}")
        print(f"  Total        : {total:>12,}")
        print(f"  Det queries={self.cfg.det_num_queries}, Lane queries={self.cfg.lane_num_queries}, Lane points={self.cfg.lane_points}")


def build_model(cfg: Config) -> DualPathNet:
    return DualPathNet(cfg)

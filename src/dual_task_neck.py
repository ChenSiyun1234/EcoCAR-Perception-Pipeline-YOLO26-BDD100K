"""
dual_task_neck.py — Partially decoupled task-specific feature routing.

Routing:
  Detection branch  -> [P3*, P4*, P5]           -> native YOLO Detect head
  Lane branch       -> [early_proj, P3**, P4**]  -> LightMUSTER lane head

  * = task-adapted (and optionally interaction-enhanced)
  early_proj = high-resolution backbone feature projected to lane_proj_ch

Key fix in v3 vs v2:
  v2 picked early_lane_idx=10 which was 20x20 (stride 32) — same as P5.
  v3 uses stride-based probing to find the actual highest-resolution feature
  that is practically useful (stride <= 8, i.e., 80x80 at 640 input).

Design references:
  HybridNets: seg decoder consumes P2 from early encoder for spatial detail.
  YOLOP:      separate decoder pathways for detection vs segmentation.
  MTI-Net:    FPM gated exchange at each shared pyramid level.

Used by:
  - src/multitask_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.task_interaction import TaskInteractionBlock, VALID_INTERACT_MODES


class DualTaskNeck(nn.Module):
    """
    Routes shared FPN neck features into two task-specific feature pyramids.

    Detection:  [P3*, P4*, P5]           — native Detect head (channel-compatible)
    Lane:       [early_proj, P3**, P4**] — lane head (high-res + two FPN scales)
    """

    def __init__(
        self,
        det_channels: List[int],   # [P3_ch, P4_ch, P5_ch] from neck
        early_ch: int,             # channels of the early backbone feature
        lane_proj_ch: int = 128,   # projected dim for early backbone feature
        interact_mode: str = 'det_to_lane',  # see VALID_INTERACT_MODES
    ):
        super().__init__()
        p3_ch, p4_ch, p5_ch = det_channels

        if interact_mode not in VALID_INTERACT_MODES:
            raise ValueError(f"interact_mode must be one of {VALID_INTERACT_MODES}")
        self.interact_mode = interact_mode

        # Project early backbone feature -> lane_proj_ch
        self.early_proj = nn.Sequential(
            nn.Conv2d(early_ch, lane_proj_ch, 1, bias=False),
            nn.BatchNorm2d(lane_proj_ch),
            nn.ReLU(inplace=True),
        )

        # Task interaction at P3 and P4 (always instantiated — mode controls behavior)
        self.interact_p3 = TaskInteractionBlock(p3_ch, p3_ch, mode=interact_mode)
        self.interact_p4 = TaskInteractionBlock(p4_ch, p4_ch, mode=interact_mode)

        # Channel specs for downstream head construction
        self.lane_channels    = [lane_proj_ch, p3_ch, p4_ch]
        self.det_channels_out = list(det_channels)

    def forward(
        self,
        neck_features: List[torch.Tensor],
        early_feat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        p3, p4, p5 = neck_features

        early_projected = self.early_proj(early_feat)

        # Safety cap: if early feature is larger than 2x P3, downsample to P3 size
        # to prevent transformer OOM (stride-2 layers can be 320x320).
        p3_h, p3_w = p3.shape[2], p3.shape[3]
        if early_projected.shape[2] > p3_h * 2 or early_projected.shape[3] > p3_w * 2:
            early_projected = F.interpolate(
                early_projected, size=(p3_h, p3_w),
                mode='bilinear', align_corners=False,
            )

        p3_det, p3_lane = self.interact_p3(p3, p3)
        p4_det, p4_lane = self.interact_p4(p4, p4)

        det_features  = [p3_det, p4_det, p5]
        lane_features = [early_projected, p3_lane, p4_lane]

        return det_features, lane_features

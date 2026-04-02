"""
DualTaskNeck: routes shared FPN features into task-specific pyramids.

Detection: [P3*, P4*, P5] — adapter-only on P3/P4, P5 passed through
Lane:      [early_proj, P3**, P4**] — high-res early + interaction-enhanced
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.models.interaction import TaskInteractionBlock, VALID_INTERACT_MODES


class DualTaskNeck(nn.Module):

    def __init__(
        self,
        det_channels: List[int],
        early_ch: int,
        lane_proj_ch: int = 128,
        interact_mode: str = 'det_to_lane',
    ):
        super().__init__()
        p3_ch, p4_ch, p5_ch = det_channels

        if interact_mode not in VALID_INTERACT_MODES:
            raise ValueError(f"interact_mode must be one of {VALID_INTERACT_MODES}")
        self.interact_mode = interact_mode

        self.early_proj = nn.Sequential(
            nn.Conv2d(early_ch, lane_proj_ch, 1, bias=False),
            nn.BatchNorm2d(lane_proj_ch),
            nn.ReLU(inplace=True),
        )

        self.interact_p3 = TaskInteractionBlock(p3_ch, p3_ch, mode=interact_mode)
        self.interact_p4 = TaskInteractionBlock(p4_ch, p4_ch, mode=interact_mode)

        self.lane_channels = [lane_proj_ch, p3_ch, p4_ch]
        self.det_channels_out = list(det_channels)

    def forward(
        self,
        neck_features: List[torch.Tensor],
        early_feat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        p3, p4, p5 = neck_features

        early_projected = self.early_proj(early_feat)

        # Safety cap: avoid transformer OOM on stride-2 features
        p3_h, p3_w = p3.shape[2], p3.shape[3]
        if early_projected.shape[2] > p3_h * 2 or early_projected.shape[3] > p3_w * 2:
            early_projected = F.interpolate(
                early_projected, size=(p3_h, p3_w),
                mode='bilinear', align_corners=False,
            )

        p3_det, p3_lane = self.interact_p3(p3, p3)
        p4_det, p4_lane = self.interact_p4(p4, p4)

        det_features = [p3_det, p4_det, p5]
        lane_features = [early_projected, p3_lane, p4_lane]

        return det_features, lane_features

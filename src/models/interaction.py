"""
Configurable cross-task feature interaction for DualTaskNeck.

Modes: none, det_to_lane, lane_to_det, bidirectional.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

VALID_INTERACT_MODES = ('none', 'det_to_lane', 'lane_to_det', 'bidirectional')


class GatedExchange(nn.Module):
    """One-way gated feature exchange: target += gate * proj(source)."""

    def __init__(self, ch_target: int, ch_source: int, gate_reduction: int = 4):
        super().__init__()
        mid = max(ch_target // gate_reduction, 16)
        self.proj = nn.Sequential(
            nn.Conv2d(ch_source, ch_target, 1, bias=False),
            nn.BatchNorm2d(ch_target),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(ch_target * 2, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch_target, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if source.shape[2:] != target.shape[2:]:
            source = F.interpolate(source, size=target.shape[2:],
                                   mode='bilinear', align_corners=False)
        s = self.proj(source)
        g = self.gate(torch.cat([target, s], dim=1))
        return target + g * s


class TaskInteractionBlock(nn.Module):
    """
    Task-specific adaptation + optional gated cross-task exchange at one scale.

    Always applies independent 3x3 adapter convs per branch. Then optionally
    applies gated exchange in the configured direction(s).
    """

    def __init__(self, ch_det: int, ch_lane: int,
                 mode: str = 'det_to_lane', gate_reduction: int = 4):
        super().__init__()
        if mode not in VALID_INTERACT_MODES:
            raise ValueError(f"mode must be one of {VALID_INTERACT_MODES}, got '{mode}'")
        self.mode = mode

        self.det_adapt = nn.Sequential(
            nn.Conv2d(ch_det, ch_det, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_det), nn.ReLU(inplace=True),
        )
        self.lane_adapt = nn.Sequential(
            nn.Conv2d(ch_lane, ch_lane, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_lane), nn.ReLU(inplace=True),
        )

        if mode in ('det_to_lane', 'bidirectional'):
            self.d2l = GatedExchange(ch_lane, ch_det, gate_reduction)
        if mode in ('lane_to_det', 'bidirectional'):
            self.l2d = GatedExchange(ch_det, ch_lane, gate_reduction)

    def forward(self, x_det: torch.Tensor, x_lane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_lane.shape[2:] != x_det.shape[2:]:
            x_lane = F.interpolate(x_lane, size=x_det.shape[2:],
                                   mode='bilinear', align_corners=False)

        det_a = self.det_adapt(x_det)
        lane_a = self.lane_adapt(x_lane)

        det_out, lane_out = det_a, lane_a

        if self.mode in ('det_to_lane', 'bidirectional'):
            lane_out = self.d2l(lane_a, det_a)
        if self.mode in ('lane_to_det', 'bidirectional'):
            det_out = self.l2d(det_a, lane_a)

        return det_out, lane_out

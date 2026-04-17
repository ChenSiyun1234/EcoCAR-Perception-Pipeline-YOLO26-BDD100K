"""
YOLOP-style Vehicle + Lane baseline (honestly labelled).

This model is YOLOP's `MCnet_0` block_cfg with two task-specific edits:
  - drivable-area decoder deleted
  - Detect `nc` 1 -> 5 (vehicle-only BDD classes)

Nothing else here is YOLOPv2. Backbone = CSP + Focus + SPP. Neck =
FPN + Upsample+Concat + PAN. Lane head = YOLOP progressive
Upsample + Conv + BottleneckCSP decoder with a 2-channel (bg, fg)
output passed through Sigmoid at forward time.

For the phase-1 YOLOPv2-style baseline see `yolopv2_baseline.py`.
"""

import torch
from torch import tensor
import torch.nn as nn
import math

from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized


# ── Block config: Vehicle detection (nc=5) + Lane segmentation ──────
# Header: [Det_out_idx, LL_Segout_idx]
VehicleLane = [
    [24, 33],   # Det output at layer 24, Lane seg output at layer 33

    # ── Encoder (layers 0-9) ──
    [-1, Focus, [3, 32, 3]],                      # 0
    [-1, Conv, [32, 64, 3, 2]],                    # 1
    [-1, BottleneckCSP, [64, 64, 1]],              # 2
    [-1, Conv, [64, 128, 3, 2]],                   # 3
    [-1, BottleneckCSP, [128, 128, 3]],            # 4
    [-1, Conv, [128, 256, 3, 2]],                  # 5
    [-1, BottleneckCSP, [256, 256, 3]],            # 6
    [-1, Conv, [256, 512, 3, 2]],                  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],             # 8
    [-1, BottleneckCSP, [512, 512, 1, False]],     # 9

    # ── FPN Neck (layers 10-16) ──
    [-1, Conv, [512, 256, 1, 1]],                  # 10
    [-1, Upsample, [None, 2, 'nearest']],          # 11
    [[-1, 6], Concat, [1]],                        # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],     # 13
    [-1, Conv, [256, 128, 1, 1]],                  # 14
    [-1, Upsample, [None, 2, 'nearest']],          # 15
    [[-1, 4], Concat, [1]],                        # 16  (encoder output)

    # ── Detection PAN head (layers 17-24) nc=5 ──
    [-1, BottleneckCSP, [256, 128, 1, False]],     # 17
    [-1, Conv, [128, 128, 3, 2]],                  # 18
    [[-1, 14], Concat, [1]],                       # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],     # 20
    [-1, Conv, [256, 256, 3, 2]],                  # 21
    [[-1, 10], Concat, [1]],                       # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],     # 23
    [[17, 20, 23], Detect, [5, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],  # 24

    # ── Lane line segmentation decoder (layers 25-33) ──
    [16, Conv, [256, 128, 3, 1]],                  # 25  (taps from encoder output)
    [-1, Upsample, [None, 2, 'nearest']],          # 26
    [-1, BottleneckCSP, [128, 64, 1, False]],      # 27
    [-1, Conv, [64, 32, 3, 1]],                    # 28
    [-1, Upsample, [None, 2, 'nearest']],          # 29
    [-1, Conv, [32, 16, 3, 1]],                    # 30
    [-1, BottleneckCSP, [16, 8, 1, False]],        # 31
    [-1, Upsample, [None, 2, 'nearest']],          # 32
    [-1, Conv, [8, 2, 3, 1]],                      # 33  (lane seg output, 2-ch: bg + lane)
]


class MCnet(nn.Module):
    """
    Multi-task network with shared encoder, detection head, and lane segmentation head.
    """
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 5  # vehicle-only classes
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]  # only lane seg index

        # Build model from block config
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # Set stride and anchors for detector
        Detector = self.model[self.detector_index]
        if isinstance(Detector, Detect):
            s = 128
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])
            Detector.anchors /= Detector.stride.view(-1, 1, 1)
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]
            x = block(x)
            if i in self.seg_out_idx:
                m = nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out  # [det_heads, ll_seg]

    def _initialize_biases(self, cf=None):
        m = self.model[self.detector_index]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg, **kwargs):
    """Factory function to build the VehicleLane model."""
    m_block_cfg = VehicleLane
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    model_out = model(input_)
    detects, lane_line_seg = model_out
    for det in detects:
        print(det.shape)
    print(lane_line_seg.shape)

"""
YOLOPv2-aligned Vehicle + Lane baseline (phase 1).

This model is a surgical edit of YOLOP's `MCnet_0` block_cfg from
`external_repos/YOLOP/lib/models/YOLOP.py`. It is not a fresh design.

Sanctioned task deviation: **drivable-area decoder removed**. Every
other architectural delta versus YOLOP tracks the YOLOPv2 paper
(arXiv 2208.11434, §3) and the YOLOPv2 public demo contract from
`external_repos/YOLOPv2/demo.py` / `utils/utils.py`.

Deltas from YOLOP MCnet_0 (with layer-number preservation where possible):

  1. Backbone (layers 0-9): `BottleneckCSP` → `ELAN` with groups=2
     beyond stride 8. Paper: "more efficient ELAN structures". The
     exact `groups` value is [INFERRED] because YOLOPv2 source is not
     public; `groups=2` lands the model in the 38-40M param band the
     paper reports.
  2. Neck SPP (layer 8): kept as YOLOP's `SPP(k=(5,9,13))` — the
     YOLOPv2 paper text keeps SPP. We do NOT use SPPCSPC.
  3. FPN + PAN + Detect (layers 10-24): unchanged vs YOLOP MCnet_0.
     `nc` changed 1→5 for the vehicle-only BDD class set.
  4. Lane seg decoder (layers 25-30): YOLOP's `Upsample+Conv` stages
     replaced with `ConvTranspose2d` (deconvolution) stages per paper.
     Tap moved to layer 17 — the refined end-of-FPN P3 feature — which
     is the actual end of the FPN path in this YOLOP-derived neck and is
     therefore the closest faithful reading of the paper text.
  5. Drivable-area decoder: removed entirely.
  6. Output contract: the lane head emits a 2-channel bg/fg logit map at
     full input resolution. This matches the paper's lane-loss formula,
     which is written for C=2 segmentation categories (background and
     lane), more faithfully than the earlier 1-channel surrogate.
"""

import math

import torch
import torch.nn as nn
from torch.nn import Upsample

from lib.models.common import (
    Conv, Concat, SPP, BottleneckCSP, Focus, Detect, ELAN,
)
from lib.utils import initialize_weights, check_anchor_order


# Block config — mirrors YOLOP MCnet_0 almost line-for-line, with the
# deltas documented above. Header: [detect_out_idx, lane_out_idx].
YOLOPv2Cfg = [
    [24, 30],    # detect @ 24, lane @ 30

    # ── Encoder 0-9 — ELAN with groups ──────────────────────────────
    [-1, Focus, [3, 32, 3]],                       # 0  stride 2
    [-1, Conv, [32, 64, 3, 2]],                    # 1  stride 4
    [-1, ELAN, [64, 64, 0.5, 4, 1]],               # 2  stride 4
    [-1, Conv, [64, 128, 3, 2]],                   # 3  stride 8
    [-1, ELAN, [128, 128, 0.5, 4, 1]],             # 4  stride 8
    [-1, Conv, [128, 256, 3, 2]],                  # 5  stride 16
    [-1, ELAN, [256, 256, 0.5, 4, 2]],             # 6  stride 16  (groups=2 [INFERRED])
    [-1, Conv, [256, 512, 3, 2]],                  # 7  stride 32
    [-1, SPP, [512, 512, [5, 9, 13]]],             # 8  neck SPP (keep — YOLOPv2 keeps SPP)
    [-1, ELAN, [512, 512, 0.5, 4, 2]],             # 9  stride 32  (groups=2 [INFERRED])

    # ── FPN 10-16 (unchanged vs YOLOP MCnet_0) ──────────────────────
    [-1, Conv, [512, 256, 1, 1]],                  # 10
    [-1, Upsample, [None, 2, 'nearest']],          # 11
    [[-1, 6], Concat, [1]],                        # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],     # 13
    [-1, Conv, [256, 128, 1, 1]],                  # 14
    [-1, Upsample, [None, 2, 'nearest']],          # 15
    [[-1, 4], Concat, [1]],                        # 16  (encoder tap for lane)

    # ── Detection PAN head 17-24 (unchanged wiring, nc=5) ──────────
    [-1, BottleneckCSP, [256, 128, 1, False]],     # 17
    [-1, Conv, [128, 128, 3, 2]],                  # 18
    [[-1, 14], Concat, [1]],                       # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],     # 20
    [-1, Conv, [256, 256, 3, 2]],                  # 21
    [[-1, 10], Concat, [1]],                       # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],     # 23
    [[17, 20, 23], Detect, [5,
        [[3, 9, 5, 11, 4, 20],
         [7, 18, 6, 39, 12, 31],
         [19, 50, 38, 81, 68, 157]],
        [128, 256, 512]]],                          # 24  Detect (nc=5)

    # ── Lane seg decoder 25-30 — deconvolution stages ──────────────
    # Tap the refined P3 feature after the FPN CSP block (layer 17).
    # Output is a paper-aligned 2-channel bg/fg logit map.
    [17,  nn.ConvTranspose2d, [128, 64, 2, 2]],    # 25
    [-1,  Conv,               [64, 32, 3, 1]],     # 26
    [-1,  nn.ConvTranspose2d, [32, 16, 2, 2]],     # 27
    [-1,  Conv,               [16, 8, 3, 1]],      # 28
    [-1,  nn.ConvTranspose2d, [8, 8, 2, 2]],       # 29
    [-1,  nn.Conv2d,          [8, 2, 1, 1]],       # 30  2-ch lane logits
]


class MCnetV2(nn.Module):
    """YOLOPv2-aligned multi-task network. Structurally YOLOP's MCnet
    with DA removed, ELAN backbone, and a deconvolution lane decoder.
    """

    def __init__(self, block_cfg=None, nc=5, names=None, **kwargs):
        super().__init__()
        block_cfg = block_cfg or YOLOPv2Cfg
        self.nc = nc
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]  # lane is a single index

        layers, save = [], []
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            if isinstance(block, str):
                block = eval(block)
            if isinstance(block, type) and issubclass(block, Detect):
                self.detector_index = i
                # Override the hardcoded nc in block_cfg with cfg-driven nc.
                args = [self.nc] + list(args[1:])
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        assert self.detector_index == block_cfg[0][0], (
            f'detect idx mismatch: header={block_cfg[0][0]} actual={self.detector_index}')

        self.model = nn.Sequential(*layers)
        self.save = sorted(set(save))
        if names is not None and len(names) == self.nc:
            self.names = list(names)
        else:
            self.names = [str(i) for i in range(self.nc)]

        # Stride + anchor normalization (YOLOP-style).
        Detector = self.model[self.detector_index]
        if isinstance(Detector, Detect):
            s = 128
            with torch.no_grad():
                out = self.forward(torch.zeros(1, 3, s, s))
                detects, _ = out
            Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])
            Detector.anchors /= Detector.stride.view(-1, 1, 1)
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        """Training / evaluation forward pass.

        Returns `[det_out, lane_logits]` where:
          * `det_out` is the YOLOP-style detection output (training
            mode: list of 3 raw grid tensors; eval mode: `(infer, train_out)`
            tuple from `Detect.forward`);
          * `lane_logits` is a **2-channel bg/fg raw-logit** map at
            **full input resolution**.

        This is the paper-aligned training contract — the loss pipeline
        consumes the 2-channel logits directly and applies hybrid focal
        + dice supervision over the two segmentation categories.

        `predict()` converts these logits to the foreground lane
        probability map expected by downstream visualization / export
        utilities.
        """
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]
            x = block(x)
            if i in self.seg_out_idx:
                out.append(x)          # raw logits (training contract)
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out

    @torch.no_grad()
    def predict(self, x):
        """Inference / export wrapper matching the YOLOPv2 demo contract.

        Returns `(det_out, lane_prob_half_res_1ch)`:
          * `det_out` unchanged from `forward()`;
          * `lane_prob_half_res_1ch` is `sigmoid(lane_logit)` at
            **half input resolution** (H/2 × W/2), a single channel.

        Downstream consumers:
          * notebook 06 (export) — use this for ONNX / TorchScript.
          * notebook 07 (A5000 profile) — use this for latency.
          * validate() in training — uses `forward()` directly (logits
            argmax path), not `predict()`.
        """
        out = self.forward(x)
        det_out, lane_logits = out       # lane_logits: [B, 2, H, W]
        fg_prob = torch.softmax(lane_logits, dim=1)[:, 1:2]
        # Downsample to H/2 × W/2 to match the demo contract.
        lane_prob_half = torch.nn.functional.interpolate(
            fg_prob, scale_factor=0.5, mode='bilinear', align_corners=False)
        return det_out, lane_prob_half

    def _initialize_biases(self, cf=None):
        m = self.model[self.detector_index]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)


def get_net_yolopv2(cfg=None, **kwargs):
    """Factory for the YOLOPv2-aligned baseline."""
    nc = 5
    names = None
    if cfg is not None:
        nc = int(getattr(cfg.MODEL, 'NC', 5))
        try:
            from lib.dataset.class_maps import build_id_dict
            _, names = build_id_dict(cfg)
        except Exception:
            vc = getattr(cfg.MODEL, 'VEHICLE_CLASSES', None)
            if vc is not None:
                names = list(vc)
    return MCnetV2(YOLOPv2Cfg, nc=nc, names=names, **kwargs)


if __name__ == '__main__':
    model = get_net_yolopv2()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 640, 640))
    detects, lane = out
    for i, d in enumerate(detects):
        print(f'det[{i}]: {d.shape}')
    print(f'lane: {lane.shape}')
    print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')

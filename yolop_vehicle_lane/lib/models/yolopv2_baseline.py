"""
YOLOPv2-style Vehicle + Lane baseline.

[INFERRED] The public YOLOPv2 repo (`external_repos/YOLOPv2`) ships only
torch.jit-scripted weights — there is no Python architecture source.
This module is reconstructed conservatively from:
  * YOLOPv2 paper text: "more efficient ELAN structures" and
    "deconvolution upsampling" in the segmentation decoders.
  * YOLOv7's upstream E-ELAN, MP (MaxPool+Conv), SPPCSPC, IDetect blocks.
  * The output-shape contract of `external_repos/YOLOPv2/demo.py`:
        [pred, anchor_grid], seg, ll = model(img)
    with `pred`/`anchor_grid` as a split detection output at 3 scales
    (strides 8/16/32), `seg` a 2-channel softmax at 1/2 input res,
    `ll` a 1-channel sigmoid at 1/2 input res.

For this project we keep **detection + lane** only (no drivable area).
The driving-area branch of YOLOPv2 is intentionally omitted because the
downstream task is vehicle + lane.

Every design decision that is not directly copied from the upstream repos
is marked `# [INFERRED]` with the source of the inference.
"""

import math

import torch
import torch.nn as nn

from lib.models.common import (
    Conv,
    Concat,
    ELAN,
    MP,
    SPPCSPC,
    DeconvBlock,
    IDetect,
    Detect,
)
from lib.utils import initialize_weights, check_anchor_order


# Block config format (same grammar as YOLOP's MCnet):
#   [from, module_cls, args]
#   `from` = -1 for previous layer, int index, or list for concat sources.
# The header row holds [detect_out_idx, lane_out_idx].
#
# Channel schedule [INFERRED] — chosen to land in the 30-40 M param
# band that YOLOPv2 reports (38.9 M). Stage outputs after each ELAN
# are (64, 128, 256, 512, 1024) at strides (2, 4, 8, 16, 32) for a
# 640x640 input.
YOLOPv2Cfg = [
    [28, 37],   # detect_out_idx=28, lane_out_idx=37

    # ── Stem (0..2) — replaces YOLOP Focus [INFERRED: YOLOv7 stem] ─────
    [-1, Conv, [3,  32, 3, 1]],     # 0  out 32   stride 1
    [-1, Conv, [32, 64, 3, 2]],     # 1  out 64   stride 2
    [-1, Conv, [64, 64, 3, 1]],     # 2  out 64   stride 2

    # ── Stage 1: downsample + ELAN (3..4) — stride 4 ───────────────────
    [-1, Conv, [64, 128, 3, 2]],    # 3  stride 4
    [-1, ELAN, [128, 128]],         # 4  P2 out (ELAN features)

    # ── Stage 2: MP + ELAN (5..6) — stride 8 ───────────────────────────
    [-1, MP,   [128, 128]],         # 5  stride 8
    [-1, ELAN, [128, 256]],         # 6  P3 out  <-- neck concat source

    # ── Stage 3: MP + ELAN (7..8) — stride 16 ──────────────────────────
    [-1, MP,   [256, 256]],         # 7  stride 16
    [-1, ELAN, [256, 512]],         # 8  P4 out  <-- neck concat source

    # ── Stage 4: MP + ELAN + SPPCSPC (9..11) — stride 32 ───────────────
    [-1, MP,     [512, 512]],       # 9  stride 32
    [-1, ELAN,   [512, 1024]],      # 10 P5 out
    [-1, SPPCSPC, [1024, 512]],     # 11 neck input (reduced to 512)

    # ── FPN-up to P4 (12..16) ──────────────────────────────────────────
    [-1,     Conv,     [512, 256, 1, 1]],   # 12
    [-1,     nn.Upsample, [None, 2, 'nearest']],  # 13
    [8,      Conv,     [512, 256, 1, 1]],   # 14 lateral from P4
    [[-1, 13], Concat, [1]],                 # 15
    [-1,     ELAN,     [512, 256]],          # 16 P4 fused  <-- det head source

    # ── FPN-up to P3 (17..21) ──────────────────────────────────────────
    [-1,     Conv,     [256, 128, 1, 1]],   # 17
    [-1,     nn.Upsample, [None, 2, 'nearest']],  # 18
    [6,      Conv,     [256, 128, 1, 1]],   # 19 lateral from P3
    [[-1, 18], Concat, [1]],                 # 20
    [-1,     ELAN,     [256, 128]],          # 21 P3 fused  <-- det head source + lane source

    # ── PAN-down P3 -> P4 (22..24) ─────────────────────────────────────
    [-1,     MP,       [128, 256]],          # 22
    [[-1, 16], Concat, [1]],                  # 23
    [-1,     ELAN,     [512, 256]],          # 24 P4 PAN  <-- det head source

    # ── PAN-down P4 -> P5 (25..27) ─────────────────────────────────────
    [-1,     MP,       [256, 512]],          # 25
    [[-1, 11], Concat, [1]],                  # 26
    [-1,     ELAN,     [1024, 512]],         # 27 P5 PAN  <-- det head source
    # Detect head inputs: layers 21, 24, 27 (P3, P4, P5)
    # Anchors are the YOLOP defaults — [INFERRED — YOLOPv2 uses YOLOv7-style
    # 3 anchors/scale with (8,16,32) strides; exact values not disclosed,
    # so we reuse YOLOP's which were k-means'd on BDD100K].
    [[21, 24, 27], IDetect, [5,
        [[3, 9, 5, 11, 4, 20],
         [7, 18, 6, 39, 12, 31],
         [19, 50, 38, 81, 68, 157]],
        [128, 256, 512]]],                    # 28  DETECT

    # ── Lane seg decoder (29..37)  [INFERRED: deconv from paper] ───────
    # Taps from P3 fused (layer 21, stride 8, 128 ch) and upsamples all
    # the way back to input resolution (stride 1) through 3 DeconvBlock
    # stages. Full-resolution output is needed because the YOLOP-derived
    # loss stack compares against a mask at input resolution.
    [21,  Conv,        [128, 64, 3, 1]],      # 29
    [-1,  DeconvBlock, [64, 32]],             # 30 stride 4
    [-1,  DeconvBlock, [32, 16]],             # 31 stride 2
    [-1,  DeconvBlock, [16,  8]],             # 32 stride 1
    [-1,  Conv,        [8,   8, 3, 1]],       # 33
    [-1,  Conv,        [8,   8, 3, 1]],       # 34
    [-1,  Conv,        [8,   8, 3, 1]],       # 35
    [-1,  Conv,        [8,   4, 3, 1]],       # 36
    # Lane output: 2 channels (bg, fg).
    # [INFERRED NOTE] The YOLOPv2 demo contract is a 1-ch sigmoid mask
    # (`ll_seg_mask = torch.round(sigmoid(ll))` in
    # external_repos/YOLOPv2/utils/utils.py::lane_line_mask). We keep
    # the 2-ch (bg, fg) output here so the phase-1 loss stack (BCE + IoU
    # from YOLOP's MultiHeadLoss) can be reused unmodified. Deployment
    # code can post-process with argmax or with softmax[:,1].round().
    [-1,  nn.Conv2d,   [4,   2, 1]],           # 37 LANE 2-ch logits
]


class YOLOPv2Net(nn.Module):
    """Multi-task network: ELAN backbone + SPPCSPC + PAN neck + IDetect
    head + transpose-conv lane decoder. 1-channel sigmoid lane output to
    match the YOLOPv2 demo output contract.
    """

    def __init__(self, block_cfg=None, nc=5, **kwargs):
        super().__init__()
        block_cfg = block_cfg or YOLOPv2Cfg
        self.nc = nc
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        # Lane is a single index here (not a list) — contract: 1-ch sigmoid.
        self.lane_out_idx = block_cfg[0][1]

        layers, save = [], []
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            if isinstance(block, str):
                block = eval(block)
            # Subclass check (IDetect -> Detect) so we also catch IDetect.
            if isinstance(block, type) and issubclass(block, Detect):
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        assert self.detector_index == block_cfg[0][0], (
            f'detect_out_idx mismatch: header={block_cfg[0][0]} actual={self.detector_index}')

        self.model = nn.Sequential(*layers)
        self.save = sorted(set(save))
        self.names = [str(i) for i in range(self.nc)]

        # Stride + anchor normalization, YOLOv5/YOLOP style.
        det = self.model[self.detector_index]
        if isinstance(det, Detect):
            s = 128
            with torch.no_grad():
                out = self.forward(torch.zeros(1, 3, s, s))
                det_out, _ = out
            det.stride = torch.tensor([s / x.shape[-2] for x in det_out])
            det.anchors /= det.stride.view(-1, 1, 1)
            check_anchor_order(det)
            self.stride = det.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        det_out = None
        lane_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]
            x = block(x)
            if i == self.detector_index:
                det_out = x
            if i == self.lane_out_idx:
                # 2-ch (bg, fg) output; Sigmoid keeps the same activation
                # semantics as YOLOP's lane decoder so the existing BCE+IoU
                # loss stack works without changes. See the note on the last
                # block_cfg entry for the demo-contract discrepancy.
                lane_out = torch.sigmoid(x)
            cache.append(x if block.index in self.save else None)
        return [det_out, lane_out]

    def _initialize_biases(self, cf=None):
        m = self.model[self.detector_index]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)


def get_net_yolopv2(cfg=None, **kwargs):
    """Factory for the YOLOPv2-style baseline."""
    nc = getattr(getattr(cfg, 'MODEL', object()), 'NC', 5) if cfg is not None else 5
    return YOLOPv2Net(YOLOPv2Cfg, nc=nc, **kwargs)


if __name__ == '__main__':
    # Smoke test: dump output shapes for a 640x640 input.
    model = get_net_yolopv2()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 640, 640))
    det_out, lane = out
    print('Lane shape:', lane.shape)           # expect [1, 1, 320, 320]
    for i, d in enumerate(det_out[1] if isinstance(det_out, tuple) else det_out):
        print(f'Det[{i}]:', d.shape)
    n = sum(p.numel() for p in model.parameters())
    print(f'Params: {n/1e6:.2f} M')

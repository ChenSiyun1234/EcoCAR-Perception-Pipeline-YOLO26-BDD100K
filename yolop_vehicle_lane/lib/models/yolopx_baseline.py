import torch
import torch.nn as nn
from torch.nn import Upsample

from lib.models.yolopx_common import (
    Conv,
    ELANNet,
    ELANBlock_Head,
    FPN_C2,
    FPN_C3,
    MergeBlock,
    PaFPNELAN,
    PSA_p,
    RepConv,
    IDetect,
    seg_head)
from lib.models.yolopx_head import YOLOXHead
from lib.utils.utils import initialize_weights


class YOLOPXVehicleLaneNet(nn.Module):
    """YOLOPX baseline adapted to the stage-1 vehicle + lane protocol.

    The upstream YOLOPX model has three tasks: traffic-object detection,
    drivable-area segmentation, and lane-line segmentation. This stage-1
    variant intentionally removes the drivable-area branch and keeps only
    the anchor-free YOLOX detector plus the C2-assisted lane branch.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.nc = int(getattr(getattr(cfg, 'MODEL', object()), 'NC', 1) or 1)
        names = list(getattr(getattr(cfg, 'MODEL', object()), 'VEHICLE_CLASSES', []) or [])
        self.names = names if len(names) == self.nc else [str(i) for i in range(self.nc)]
        self.gr = 1.0

        self.backbone = ELANNet(use_C2=True)
        self.neck = PaFPNELAN()
        self.det_head = YOLOXHead(num_classes=self.nc, width=0.75, strides=(8, 16, 32), in_channels=(128, 256, 512))

        self.lane_c3 = FPN_C3()
        self.lane_c2 = FPN_C2()
        self.c2_reduce = Conv(256, 128, 3, 1)
        self.p3_reduce = Conv(256, 128, 3, 1)
        self.p3_up = Upsample(None, 2, 'bilinear')
        self.c2_p3_merge = MergeBlock('add')
        self.lane_elan_1 = ELANBlock_Head(128, 64)
        self.lane_context_1 = PSA_p(64, 64)
        self.lane_reduce_1 = Conv(64, 32, 3, 1)
        self.lane_up_1 = Upsample(None, 2, 'bilinear')
        self.lane_reduce_2 = Conv(32, 16, 3, 1)
        self.lane_elan_2 = ELANBlock_Head(16, 8)
        self.lane_context_2 = PSA_p(8, 8)
        self.lane_up_2 = Upsample(None, 2, 'bilinear')
        self.lane_pred = Conv(8, 2, 3, 1)
        self.lane_seg_head = seg_head('sigmoid')

        self.stride = self.det_head.strides
        self.det_head.initialize_biases(1e-2)
        initialize_weights(self)

    def forward(self, x):
        features = self.backbone(x)
        pyramid = self.neck(features)
        det_out = self.det_head(pyramid)

        c2 = self.c2_reduce(self.lane_c2(pyramid))
        p3 = self.p3_up(self.p3_reduce(self.lane_c3(pyramid)))
        lane = self.c2_p3_merge([p3, c2])
        lane = self.lane_elan_1(lane)
        lane = self.lane_context_1(lane)
        lane = self.lane_reduce_1(lane)
        lane = self.lane_up_1(lane)
        lane = self.lane_reduce_2(lane)
        lane = self.lane_elan_2(lane)
        lane = self.lane_context_2(lane)
        lane = self.lane_up_2(lane)
        lane = self.lane_pred(lane)
        lane = self.lane_seg_head(lane)
        return det_out, lane

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
            elif isinstance(m, IDetect):
                m.fuse()
                m.forward = m.fuseforward
        return self


def fuse_conv_and_bn(conv, bn):
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


def get_net_yolopx(cfg=None, **kwargs):
    del kwargs
    return YOLOPXVehicleLaneNet(cfg)

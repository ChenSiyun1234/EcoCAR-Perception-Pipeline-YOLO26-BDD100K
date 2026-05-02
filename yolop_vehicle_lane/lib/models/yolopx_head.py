import math

import torch
import torch.nn as nn

from lib.models.yolopx_common import Conv, GhostConv


def meshgrid_ij(y, x):
    try:
        return torch.meshgrid(y, x, indexing='ij')
    except TypeError:
        return torch.meshgrid(y, x)


class YOLOXHead(nn.Module):
    """Decoupled anchor-free YOLOX detection head used by YOLOPX."""

    def __init__(self, num_classes, width=0.75, strides=(8, 16, 32), in_channels=(128, 256, 512), depthwise=False):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = int(num_classes)
        self.decode_in_inference = True
        self.strides = list(strides)

        base_channels = int(256 * width)
        base_block = GhostConv if depthwise else Conv

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for channels in in_channels:
            self.stems.append(Conv(int(channels), base_channels, k=1, s=1, act=True))
            self.cls_convs.append(nn.Sequential(
                base_block(base_channels, base_channels, k=3, s=1, act=True),
                base_block(base_channels, base_channels, k=3, s=1, act=True),
            ))
            self.reg_convs.append(nn.Sequential(
                base_block(base_channels, base_channels, k=3, s=1, act=True),
                base_block(base_channels, base_channels, k=3, s=1, act=True),
            ))
            self.cls_preds.append(nn.Conv2d(base_channels, self.n_anchors * self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(base_channels, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(base_channels, self.n_anchors, 1, 1, 0))

    def initialize_biases(self, prior_prob=1e-2):
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(bias_value)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(bias_value)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin):
        outputs = []
        feature_maps = xin[4:]
        for cls_conv, reg_conv, stride_this_level, x, stem, cls_pred, reg_pred, obj_pred in zip(
            self.cls_convs,
            self.reg_convs,
            self.strides,
            feature_maps,
            self.stems,
            self.cls_preds,
            self.reg_preds,
            self.obj_preds,
        ):
            del stride_this_level
            x = stem(x)
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)
            cls_output = cls_pred(cls_feat)
            reg_output = reg_pred(reg_feat)
            obj_output = obj_pred(reg_feat)
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], dim=1)
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], dim=1)
            outputs.append(output)

        if self.training:
            return outputs

        self.hw = [x.shape[-2:] for x in outputs]
        flattened = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        if self.decode_in_inference:
            return self.decode_outputs(flattened, dtype=feature_maps[0].type()), outputs
        return outputs

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid_ij(torch.arange(hsize), torch.arange(wsize))
            grid = torch.stack((xv, yv), dim=2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((*grid.shape[:2], 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

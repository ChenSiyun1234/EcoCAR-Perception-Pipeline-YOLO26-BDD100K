"""
Shared visual backbone — ResNet + FPN.

Produces multi-scale feature maps [P3, P4, P5] that feed both the
detection and lane branches.  Only the shallow ResNet stem is truly
"shared" in the forward pass; the FPN can optionally be duplicated
per task (future work).

Design reference: RT-DETR hybrid encoder concept — but here we keep
a standard FPN for simplicity and proven performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPN(nn.Module):
    """Simple Feature Pyramid Network.

    Takes C3, C4, C5 from ResNet and produces P3, P4, P5 with
    uniform channel dimension.
    """

    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        self.lateral3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        return self.smooth3(p3), self.smooth4(p4), self.smooth5(p5)


# Channel counts for ResNet variants at stages 2, 3, 4 (C3, C4, C5)
_RESNET_CHANNELS = {
    "resnet18":  [128, 256, 512],
    "resnet34":  [128, 256, 512],
    "resnet50":  [512, 1024, 2048],
    "resnet101": [512, 1024, 2048],
}


class BackboneFPN(nn.Module):
    """ResNet backbone + FPN producing multi-scale features.

    Output: list of 3 tensors [P3, P4, P5] with shape
    [B, fpn_channels, H/8, W/8], [B, fpn_channels, H/16, W/16],
    [B, fpn_channels, H/32, W/32].
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True,
                 fpn_channels: int = 256):
        super().__init__()
        self.name = name
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = getattr(models, name)(weights=weights)

        # Shared stem: conv1 + bn1 + relu + maxpool + layer1
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
        )
        self.layer2 = resnet.layer2  # stride 8  → C3
        self.layer3 = resnet.layer3  # stride 16 → C4
        self.layer4 = resnet.layer4  # stride 32 → C5

        in_ch = _RESNET_CHANNELS[name]
        self.fpn = FPN(in_ch, fpn_channels)
        self.out_channels = fpn_channels

    def forward(self, x: torch.Tensor) -> list:
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        return [p3, p4, p5]

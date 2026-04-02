"""
Shared visual backbone — ResNet + FPN, with a safe fallback lightweight CNN.

Primary path uses torchvision ResNet to stay compatible with common Colab
training environments. If torchvision ops are broken in a local environment,
a simple ConvNet fallback keeps imports and debugging functional.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
    _HAS_TORCHVISION = True
except Exception:
    tv_models = None
    _HAS_TORCHVISION = False


class FPN(nn.Module):
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


_RESNET_CHANNELS = {
    "resnet18": [128, 256, 512],
    "resnet34": [128, 256, 512],
    "resnet50": [512, 1024, 2048],
    "resnet101": [512, 1024, 2048],
}


class TinyConvBackbone(nn.Module):
    def __init__(self, fpn_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GroupNorm(8, 512), nn.ReLU(inplace=True))
        self.fpn = FPN([128, 256, 512], fpn_channels)
        self.out_channels = fpn_channels

    def forward(self, x):
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return list(self.fpn(c3, c4, c5))


class BackboneFPN(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, fpn_channels: int = 256):
        super().__init__()
        self.name = name
        if _HAS_TORCHVISION and name in _RESNET_CHANNELS:
            weights = "IMAGENET1K_V1" if pretrained else None
            resnet = getattr(tv_models, name)(weights=weights)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.fpn = FPN(_RESNET_CHANNELS[name], fpn_channels)
            self.out_channels = fpn_channels
            self._fallback = None
        else:
            self._fallback = TinyConvBackbone(fpn_channels)
            self.out_channels = fpn_channels

    def forward(self, x: torch.Tensor) -> list:
        if self._fallback is not None:
            return self._fallback(x)
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return list(self.fpn(c3, c4, c5))

"""
multitask_model.py — Multi-task YOLO26: shared backbone/neck + detection head + lane seg head.

Architecture:
  Input → YOLO26 Backbone → YOLO26 Neck → ┬→ Native Detect Head (bboxes)
                                           └→ LaneSegHead (binary mask)

Used by:
  - 08_joint_training.ipynb
  - 09_joint_inference.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LaneSegHead(nn.Module):
    """
    Lightweight lane segmentation head.

    Takes multi-scale neck features (P3, P4, P5), fuses them, and produces
    a binary segmentation mask at the target resolution.

    Architecture:
      - Upsample P4, P5 to P3 resolution
      - Concatenate along channel dim
      - 3× (Conv3×3 + BN + ReLU) to reduce channels
      - Bilinear upsample to target mask size
      - 1×1 conv → 1-channel logits
    """

    def __init__(
        self,
        in_channels_list: List[int],
        hidden_channels: int = 64,
        mask_height: int = 180,
        mask_width: int = 320,
    ):
        """
        Args:
            in_channels_list: Channel counts for [P3, P4, P5] neck outputs.
            hidden_channels:  Intermediate conv channels.
            mask_height:      Output mask height.
            mask_width:       Output mask width.
        """
        super().__init__()
        self.mask_height = mask_height
        self.mask_width = mask_width

        total_in = sum(in_channels_list)

        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(hidden_channels // 2, 1, 1)  # 1-channel logits

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [P3, P4, P5] tensors from neck.

        Returns:
            Logits tensor of shape (B, 1, mask_height, mask_width).
        """
        # Upsample all features to P3 spatial size
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        upsampled = []
        for feat in features:
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode="bilinear", align_corners=False)
            upsampled.append(feat)

        # Concatenate along channels
        x = torch.cat(upsampled, dim=1)

        # Conv fusion
        x = self.fuse(x)

        # Upsample to mask resolution
        x = F.interpolate(x, size=(self.mask_height, self.mask_width),
                          mode="bilinear", align_corners=False)

        # 1×1 conv → logits
        x = self.head(x)
        return x


class MultiTaskYOLO(nn.Module):
    """
    Multi-task wrapper around a YOLO26 model.

    Shares the backbone and neck, then branches to:
      - Native YOLO detection head (bounding boxes)
      - Lightweight lane segmentation head (binary mask)

    The detection head is the original Ultralytics Detect module,
    kept frozen or jointly trained.
    """

    def __init__(
        self,
        yolo_model,
        neck_output_indices: Optional[List[int]] = None,
        lane_hidden_channels: int = 64,
        mask_height: int = 180,
        mask_width: int = 320,
    ):
        """
        Args:
            yolo_model:           An Ultralytics YOLO model (e.g. YOLO("yolo26s.pt")).
            neck_output_indices:  Indices of the layers whose outputs feed into
                                  the detection head (auto-detected if None).
            lane_hidden_channels: Hidden channels in the lane head.
            mask_height:          Lane mask output height.
            mask_width:           Lane mask output width.
        """
        super().__init__()

        # Extract the inner nn.Module
        inner = yolo_model.model
        self.model_layers = inner.model  # nn.Sequential-like

        # Auto-detect architecture structure
        self._detect_head_idx = len(self.model_layers) - 1
        self.detect_head = self.model_layers[self._detect_head_idx]

        # Determine which neck outputs feed the detect head
        # The Detect module's forward() receives a list of feature maps
        # We need to identify which layer outputs become those inputs
        if neck_output_indices is not None:
            self.neck_output_indices = neck_output_indices
        else:
            # Auto-detect from the detect head's input spec
            self.neck_output_indices = self._find_neck_output_indices()

        # Build backbone + neck (everything except detect head)
        self.backbone_neck = nn.ModuleList(
            [self.model_layers[i] for i in range(self._detect_head_idx)]
        )

        # Determine channel counts for the neck outputs that go to detection head
        # We'll probe these by looking at the detect head's expected inputs
        self._neck_channels = self._probe_neck_channels(yolo_model)

        # Build lane segmentation head
        self.lane_head = LaneSegHead(
            in_channels_list=self._neck_channels,
            hidden_channels=lane_hidden_channels,
            mask_height=mask_height,
            mask_width=mask_width,
        )

        # Store the ultralytics model reference for loss computation
        self._yolo_model = yolo_model

        # Copy model config attributes needed by ultralytics internals
        if hasattr(inner, 'args'):
            self.args = inner.args
        if hasattr(inner, 'stride'):
            self.stride = inner.stride

        print(f"✅ MultiTaskYOLO built:")
        print(f"   Backbone+Neck layers: {self._detect_head_idx}")
        print(f"   Detect head: {self.detect_head.__class__.__name__}")
        print(f"   Neck output indices: {self.neck_output_indices}")
        print(f"   Neck channels: {self._neck_channels}")
        print(f"   Lane head params: {sum(p.numel() for p in self.lane_head.parameters()):,}")

    def _find_neck_output_indices(self) -> List[int]:
        """
        Find which backbone/neck layer indices produce the feature maps
        that feed into the detection head.

        In Ultralytics YOLO, the detect head receives features from specific
        layers defined in the YAML config. We identify these by looking at
        the model's `save` attribute or by finding Conv/C3k2 layers just
        before Concat/Upsample transitions.
        """
        # The Detect module typically expects 3 feature maps (P3, P4, P5)
        # In the model YAML, these are specified via 'from' indices
        # As a robust fallback, we look at the detect head's `f` attribute
        if hasattr(self.detect_head, 'f'):
            # `f` is the list of "from" indices in the YAML
            from_indices = self.detect_head.f
            if isinstance(from_indices, list):
                return from_indices

        # Heuristic: search for the layer indices that the detect head uses
        # Typical YOLO26 uses layers at roughly [15, 18, 21] or similar
        # We default to a reasonable set
        n = self._detect_head_idx
        # Common pattern: detect head takes from [-4, -2, -1] relative positions
        return [n - 6, n - 3, n - 1]

    def _probe_neck_channels(self, yolo_model) -> List[int]:
        """
        Determine the channel counts of neck outputs by looking at the
        detection head's input expectations.
        """
        # Detect heads in Ultralytics store their input channel info
        if hasattr(self.detect_head, 'cv2'):  # Standard detect
            ch_list = []
            for cv2_block in self.detect_head.cv2:
                # cv2 is a ModuleList of conv sequences, first conv has in_channels
                first_conv = None
                for m in cv2_block.modules():
                    if isinstance(m, nn.Conv2d):
                        first_conv = m
                        break
                if first_conv:
                    ch_list.append(first_conv.in_channels)
            if ch_list:
                return ch_list

        # Fallback: inspect actual layer outputs via a dummy forward pass
        try:
            dummy = torch.randn(1, 3, 640, 640)
            if next(self.backbone_neck.parameters()).is_cuda:
                dummy = dummy.cuda()
            with torch.no_grad():
                _ = self._forward_backbone_neck(dummy)
            neck_outs = [self._layer_outputs[i] for i in self.neck_output_indices
                         if i in self._layer_outputs]
            return [f.shape[1] for f in neck_outs]
        except Exception:
            pass

        # Final fallback for yolo26s
        return [128, 256, 512]

    def _forward_backbone_neck(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Run forward through backbone + neck layers, tracking intermediate outputs.

        Returns:
            Tuple of (final_output, dict of {layer_idx: output_tensor}).
        """
        self._layer_outputs = {}

        # Ultralytics models use a 'from' mechanism where some layers
        # take input from earlier layers (skip connections).
        # We need to replicate this routing.
        outputs = []  # Store output of each layer

        for idx, layer in enumerate(self.backbone_neck):
            # Determine input for this layer
            # Check if layer has a 'f' (from) attribute indicating skip connection
            f = getattr(layer, 'f', -1)  # default: previous layer output

            if isinstance(f, int):
                if f == -1:
                    x_in = x if idx == 0 else outputs[-1]
                else:
                    x_in = outputs[f]
            elif isinstance(f, list):
                x_in = [outputs[j] for j in f]
            else:
                x_in = x if idx == 0 else outputs[-1]

            # Handle layers that expect list inputs (Concat)
            if isinstance(x_in, list):
                x = layer(x_in)
            else:
                x = layer(x_in)

            outputs.append(x)
            self._layer_outputs[idx] = x

        return x, self._layer_outputs

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full multi-task model.

        Args:
            x:       Input images tensor (B, 3, H, W).
            targets: Optional detection targets for loss computation.

        Returns:
            Dict with keys:
              'det_output': detection head output
              'lane_logits': lane segmentation logits (B, 1, mask_h, mask_w)
              'neck_features': list of neck feature tensors (for loss computation)
        """
        # Forward through backbone + neck
        _, layer_outputs = self._forward_backbone_neck(x)

        # Collect neck outputs for detection and lane heads
        neck_features = []
        for idx in self.neck_output_indices:
            if idx in layer_outputs:
                neck_features.append(layer_outputs[idx])

        # Detection head
        det_output = self.detect_head(neck_features)

        # Lane segmentation head
        lane_logits = self.lane_head(neck_features)

        return {
            "det_output": det_output,
            "lane_logits": lane_logits,
            "neck_features": neck_features,
        }


def build_multitask_model(
    weights: str = "yolo26s.pt",
    mask_height: int = 180,
    mask_width: int = 320,
    lane_hidden_channels: int = 64,
) -> MultiTaskYOLO:
    """
    Convenience function to build a MultiTaskYOLO from pretrained weights.

    Args:
        weights:              Path to YOLO26 weights or model name.
        mask_height:          Lane mask output height.
        mask_width:           Lane mask output width.
        lane_hidden_channels: Hidden channels in lane head.

    Returns:
        MultiTaskYOLO model.
    """
    from ultralytics import YOLO

    yolo_model = YOLO(weights)
    model = MultiTaskYOLO(
        yolo_model,
        lane_hidden_channels=lane_hidden_channels,
        mask_height=mask_height,
        mask_width=mask_width,
    )
    return model

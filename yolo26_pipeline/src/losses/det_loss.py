"""
Detection loss wrapper — delegates to Ultralytics v8/v11 native loss.

The detect head now uses nc=5 (vehicle-only). BDD100K label files still use
the original BDD class IDs (0-9), so targets are filtered and remapped to
vehicle class IDs (0-4) before loss computation.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.utils.class_map import remap_targets_bdd_to_vehicle


class DetectionLossWrapper(nn.Module):
    """
    Wraps Ultralytics' native detection loss.
    Handles lazy initialization, format differences, and vehicle class remapping.
    """

    def __init__(self, model: nn.Module, remap_classes: bool = True):
        super().__init__()
        self.model = model
        self.remap_classes = remap_classes
        self._native_loss: Optional[nn.Module] = None
        self._polyfilled = False

    def _ensure_initialized(self, det_output):
        if self._native_loss is not None:
            return

        # Polyfill args dict -> object with attribute access
        if not self._polyfilled and hasattr(self.model, 'args') and isinstance(self.model.args, dict):
            class ObjectDict(dict):
                def __getattr__(self, key):
                    return self.get(key, 1.0)
                def __setattr__(self, key, value):
                    self[key] = value
            self.model.args = ObjectDict(self.model.args)
            self._polyfilled = True

        if not hasattr(self.model, 'model'):
            self.model.model = self.model.model_layers

        loss_cls = None

        # Try v11 loss for one2many outputs
        if isinstance(det_output, dict) and "one2many" in det_output:
            try:
                from ultralytics.utils.loss import v11DetectionLoss
                loss_cls = v11DetectionLoss
            except ImportError:
                pass

        # Fallback to v8 loss
        if loss_cls is None:
            try:
                from ultralytics.utils.loss import v8DetectionLoss
                loss_cls = v8DetectionLoss
            except ImportError:
                from ultralytics.models.yolo.detect.train import v8DetectionLoss
                loss_cls = v8DetectionLoss

        self._native_loss = loss_cls(self.model)

    def forward(self, det_output, targets: torch.Tensor) -> torch.Tensor:
        # Remap BDD class IDs to vehicle IDs and filter non-vehicle
        if self.remap_classes:
            targets = remap_targets_bdd_to_vehicle(targets)

        if targets.shape[0] == 0:
            dev = det_output[0].device if isinstance(det_output, list) else det_output.device
            return torch.tensor(0.0, device=dev, requires_grad=True)

        self._ensure_initialized(det_output)

        batch = {
            'batch_idx': targets[:, 0],
            'cls': targets[:, 1:2],
            'bboxes': targets[:, 2:],
        }

        try:
            loss, _ = self._native_loss(det_output, batch)
        except (KeyError, TypeError):
            raw = det_output[1] if isinstance(det_output, tuple) else det_output

            if isinstance(raw, dict) and "one2many" in raw:
                loss, _ = self._native_loss(raw["one2many"], batch)
            elif isinstance(raw, (list, tuple)):
                bs = raw[0].shape[0]
                nc = self.model.detect_head.nc
                reg_max = self.model.detect_head.reg_max
                boxes, scores = [], []
                for xi in raw:
                    b, s = xi.split([4 * reg_max, nc], dim=1)
                    boxes.append(b.reshape(bs, 4 * reg_max, -1))
                    scores.append(s.reshape(bs, nc, -1))
                formatted = {
                    "boxes": torch.cat(boxes, dim=-1),
                    "scores": torch.cat(scores, dim=-1),
                    "feats": raw,
                }
                loss, _ = self._native_loss(formatted, batch)
            else:
                raise

        if hasattr(loss, "sum") and loss.numel() > 1:
            loss = loss.sum()

        return loss

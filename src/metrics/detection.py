"""
Detection metrics: mAP@50, mAP@50-95 for vehicle-only detection.

Uses torchmetrics MeanAveragePrecision with safe handling of sentinel values.
With native nc=5, all predictions are vehicle classes — no filtering needed.
"""

import math
import torch
from typing import Dict, List, Optional

from src.utils.class_map import NUM_VEHICLE_CLASSES, VEHICLE_CLASSES


class DetectionMetrics:
    """Accumulates detection predictions and ground truth, then computes mAP."""

    def __init__(self, device: str = "cpu",
                 max_detection_threshold: int = 300):
        self.device = device
        self.max_detection_threshold = max_detection_threshold

        try:
            from torchmetrics.detection import MeanAveragePrecision

            kwargs = dict(box_format='xyxy', iou_type='bbox')
            try:
                kwargs["max_detection_thresholds"] = [1, 10, max_detection_threshold]
            except Exception:
                pass

            self._map = MeanAveragePrecision(**kwargs).to(device)
            if hasattr(self._map, "warn_on_many_detections"):
                self._map.warn_on_many_detections = False
            self._has_torchmetrics = True
        except ImportError:
            self._has_torchmetrics = False
            self._preds = []
            self._targets = []

    def reset(self):
        if self._has_torchmetrics:
            self._map.reset()
        else:
            self._preds = []
            self._targets = []

    @staticmethod
    def _empty_boxes(dev: str):
        return torch.empty((0, 4), device=dev)

    @staticmethod
    def _empty_scores(dev: str):
        return torch.empty((0,), device=dev)

    @staticmethod
    def _empty_labels(dev: str):
        return torch.empty((0,), dtype=torch.int64, device=dev)

    @staticmethod
    def _safe_metric_value(x) -> float:
        try:
            v = float(x.item() if hasattr(x, 'item') else x)
        except Exception:
            return 0.0
        if not math.isfinite(v) or v < 0:
            return 0.0
        return v

    @torch.no_grad()
    def update(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        if not self._has_torchmetrics:
            return

        dev = self.device
        pred_dict = {
            'boxes': pred_boxes[:, :4].to(dev) if len(pred_boxes) else self._empty_boxes(dev),
            'scores': pred_boxes[:, 4].to(dev) if len(pred_boxes) else self._empty_scores(dev),
            'labels': pred_boxes[:, 5].long().to(dev) if len(pred_boxes) else self._empty_labels(dev),
        }
        target_dict = {
            'boxes': gt_boxes.to(dev) if len(gt_boxes) else self._empty_boxes(dev),
            'labels': gt_labels.long().to(dev) if len(gt_labels) else self._empty_labels(dev),
        }
        self._map.update([pred_dict], [target_dict])

    def compute(self) -> Dict[str, float]:
        if not self._has_torchmetrics:
            return {"det_map50": 0.0, "det_map50_95": 0.0}

        result = self._map.compute()
        det_map50_95 = self._safe_metric_value(result.get('map', 0.0))
        det_map50 = self._safe_metric_value(result.get('map_50', 0.0))

        # Per-class AP
        per_class = {}
        map_per_class = result.get('map_per_class', None)
        if map_per_class is not None and len(map_per_class) == NUM_VEHICLE_CLASSES:
            for i, name in enumerate(VEHICLE_CLASSES):
                per_class[f'ap_{name}'] = self._safe_metric_value(map_per_class[i])

        return {
            'det_map50_95': det_map50_95,
            'det_map50': det_map50,
            **per_class,
        }

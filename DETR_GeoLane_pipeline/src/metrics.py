"""
Evaluation metrics.

Detection:
- torchmetrics mAP when available

Lane:
- curve-aware F1 using bidirectional point-to-curve distance for matching
- average curve distance over true matches
"""

from __future__ import annotations

from typing import Dict

import torch

from .losses import bidirectional_point_to_curve_distance


class DetectionMetrics:
    def __init__(self, num_classes: int = 5, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        try:
            from torchmetrics.detection import MeanAveragePrecision
            self._map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(device)
            self._has_tm = True
        except Exception:
            self._has_tm = False
            print("  torchmetrics not available — detection metrics disabled")

    def reset(self):
        if self._has_tm:
            self._map.reset()

    @torch.no_grad()
    def update(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
               gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        if not self._has_tm:
            return
        dev = self.device
        preds = [{
            "boxes": pred_boxes.to(dev) if pred_boxes.numel() else torch.empty((0, 4), device=dev),
            "scores": pred_scores.to(dev) if pred_scores.numel() else torch.empty(0, device=dev),
            "labels": pred_labels.long().to(dev) if pred_labels.numel() else torch.empty(0, dtype=torch.long, device=dev),
        }]
        targets = [{
            "boxes": gt_boxes.to(dev) if gt_boxes.numel() else torch.empty((0, 4), device=dev),
            "labels": gt_labels.long().to(dev) if gt_labels.numel() else torch.empty(0, dtype=torch.long, device=dev),
        }]
        self._map.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        if not self._has_tm:
            return {"det_map50": 0.0, "det_map50_95": 0.0}
        result = self._map.compute()
        def safe(x):
            return max(0.0, float(x.item())) if torch.isfinite(x).item() else 0.0
        return {"det_map50": safe(result.get("map_50", torch.tensor(0.0))), "det_map50_95": safe(result.get("map", torch.tensor(0.0)))}


class LaneMetrics:
    def __init__(self, match_thresh_px: float = 15.0, img_size: int = 640):
        self.match_thresh = match_thresh_px / img_size
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.curve_sum = 0.0
        self.curve_count = 0

    @torch.no_grad()
    def update(self, pred_points: torch.Tensor, pred_exist: torch.Tensor,
               gt_points: torch.Tensor, gt_exist: torch.Tensor,
               gt_vis: torch.Tensor | None = None,
               exist_thresh: float = 0.5):
        pred_mask = pred_exist[:, 0].sigmoid() > exist_thresh
        gt_mask = gt_exist > 0.5
        n_pred = int(pred_mask.sum().item())
        n_gt = int(gt_mask.sum().item())
        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_gt
            return
        pred_pts = pred_points[pred_mask]
        gt_pts = gt_points[gt_mask]
        gt_vis_sel = gt_vis[gt_mask] if gt_vis is not None else [None] * len(gt_pts)
        dist = torch.zeros(n_pred, n_gt)
        for i in range(n_pred):
            for j in range(n_gt):
                dist[i, j] = bidirectional_point_to_curve_distance(pred_pts[i], gt_pts[j], gt_vis=gt_vis_sel[j] if gt_vis is not None else None)
        matched_pred = set()
        matched_gt = set()
        flat = [(float(dist[i, j].item()), i, j) for i in range(n_pred) for j in range(n_gt)]
        flat.sort()
        for d, i, j in flat:
            if i in matched_pred or j in matched_gt:
                continue
            if d < self.match_thresh:
                matched_pred.add(i)
                matched_gt.add(j)
                self.tp += 1
                self.curve_sum += d
                self.curve_count += 1
        self.fp += n_pred - len(matched_pred)
        self.fn += n_gt - len(matched_gt)

    def compute(self) -> Dict[str, float]:
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        curve = self.curve_sum / max(self.curve_count, 1)
        return {"lane_f1": f1, "lane_precision": prec, "lane_recall": rec, "lane_curve_dist": curve}

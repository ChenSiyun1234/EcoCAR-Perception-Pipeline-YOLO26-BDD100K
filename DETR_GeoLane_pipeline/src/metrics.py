"""
Evaluation metrics for both tasks.

Detection: mAP@50, mAP@50-95 using torchmetrics
Lane: F1-score based on curve-to-curve geometric matching
"""

import math
import torch
import numpy as np
from typing import Dict, List


def _segments_from_points(points: torch.Tensor):
    if points.shape[0] < 2:
        return points[:1], points[:1]
    return points[:-1], points[1:]


def point_to_polyline_distance(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    if polyline.shape[0] == 0:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    if polyline.shape[0] == 1:
        return torch.norm(points - polyline[0], dim=-1)
    a, b = _segments_from_points(polyline)
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-8)
    t = (ap * ab[None, :, :]).sum(dim=-1) / denom[None, :]
    t = t.clamp(0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    dist = torch.norm(points[:, None, :] - proj, dim=-1)
    return dist.min(dim=1).values


def resample_polyline(points: torch.Tensor, num: int) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.zeros(num, 2, device=points.device, dtype=points.dtype)
    if points.shape[0] == 1:
        return points.repeat(num, 1)
    seg = torch.norm(points[1:] - points[:-1], dim=-1)
    cum = torch.cat([torch.zeros(1, device=points.device, dtype=points.dtype), seg.cumsum(0)])
    total = cum[-1]
    if total < 1e-8:
        return points[:1].repeat(num, 1)
    t = torch.linspace(0.0, float(total.item()), num, device=points.device, dtype=points.dtype)
    idx = torch.searchsorted(cum, t, right=True) - 1
    idx = idx.clamp(min=0, max=points.shape[0] - 2)
    left = points[idx]
    right = points[idx + 1]
    left_t = cum[idx]
    right_t = cum[idx + 1]
    alpha = ((t - left_t) / (right_t - left_t).clamp(min=1e-8)).unsqueeze(-1)
    return left + alpha * (right - left)


def curve_distance(pred_points: torch.Tensor, gt_points: torch.Tensor, resample_n: int = 96) -> torch.Tensor:
    pred_rs = resample_polyline(pred_points, resample_n)
    gt_rs = resample_polyline(gt_points, resample_n)
    d1 = point_to_polyline_distance(pred_rs, gt_rs).mean()
    d2 = point_to_polyline_distance(gt_rs, pred_rs).mean()
    return 0.5 * (d1 + d2)


class DetectionMetrics:
    """mAP computation using torchmetrics MeanAveragePrecision."""

    def __init__(self, num_classes: int = 5, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        try:
            from torchmetrics.detection import MeanAveragePrecision
            self._map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(device)
            self._has_tm = True
        except ImportError:
            self._has_tm = False
            print("  torchmetrics not available — detection metrics disabled")

    def reset(self):
        if self._has_tm:
            self._map.reset()

    @torch.no_grad()
    def update(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
               pred_labels: torch.Tensor, gt_boxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """Update with predictions and GT for one image."""
        if not self._has_tm:
            return
        dev = self.device
        preds = [{
            "boxes": pred_boxes.to(dev) if pred_boxes.shape[0] > 0 else torch.empty((0, 4), device=dev),
            "scores": pred_scores.to(dev) if pred_scores.shape[0] > 0 else torch.empty(0, device=dev),
            "labels": pred_labels.long().to(dev) if pred_labels.shape[0] > 0 else torch.empty(0, dtype=torch.long, device=dev),
        }]
        targets = [{
            "boxes": gt_boxes.to(dev) if gt_boxes.shape[0] > 0 else torch.empty((0, 4), device=dev),
            "labels": gt_labels.long().to(dev) if gt_labels.shape[0] > 0 else torch.empty(0, dtype=torch.long, device=dev),
        }]
        self._map.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        if not self._has_tm:
            return {"det_map50": 0.0, "det_map50_95": 0.0}
        result = self._map.compute()
        safe = lambda x: max(0.0, float(x.item())) if torch.isfinite(x) else 0.0
        return {
            "det_map50": safe(result.get("map_50", torch.tensor(0.0))),
            "det_map50_95": safe(result.get("map", torch.tensor(0.0))),
        }


class LaneMetrics:
    """Lane evaluation based on curve-to-curve geometric matching.

    A predicted lane is matched to a GT lane if the symmetric point-to-polyline
    distance between the two piecewise-linear curves is below a threshold.
    """

    def __init__(self, match_thresh_px: float = 15.0, img_size: int = 640):
        self.match_thresh = match_thresh_px / img_size  # normalized threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.chamfer_sum = 0.0
        self.chamfer_count = 0

    def reset(self):
        self.tp = self.fp = self.fn = 0
        self.chamfer_sum = self.chamfer_count = 0

    @torch.no_grad()
    def update(self, pred_points: torch.Tensor, pred_exist: torch.Tensor,
               gt_points: torch.Tensor, gt_exist: torch.Tensor,
               exist_thresh: float = 0.5):
        """Update for one image.

        Args:
            pred_points: (Q, N, 2) normalized
            pred_exist: (Q, 1) logits
            gt_points: (max_lanes, N, 2) normalized
            gt_exist: (max_lanes,) float
        """
        pred_mask = pred_exist[:, 0].sigmoid() > exist_thresh
        gt_mask = gt_exist > 0.5

        n_pred = pred_mask.sum().item()
        n_gt = gt_mask.sum().item()

        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_gt
            return

        pred_pts = pred_points[pred_mask]  # (n_pred, N, 2)
        gt_pts = gt_points[gt_mask]        # (n_gt, N, 2)

                # Distance matrix: curve-to-curve geometric distance
        dist = torch.zeros(n_pred, n_gt)
        for i in range(n_pred):
            for j in range(n_gt):
                dist[i, j] = curve_distance(pred_pts[i], gt_pts[j])

        # Greedy matching (simple, fast)
        matched_pred = set()
        matched_gt = set()
        # Sort all (pred, gt) pairs by distance
        flat = [(dist[i, j].item(), i, j) for i in range(n_pred) for j in range(n_gt)]
        flat.sort()
        for d, i, j in flat:
            if i in matched_pred or j in matched_gt:
                continue
            if d < self.match_thresh:
                matched_pred.add(i)
                matched_gt.add(j)
                self.tp += 1
                self.chamfer_sum += d
                self.chamfer_count += 1

        self.fp += n_pred - len(matched_pred)
        self.fn += n_gt - len(matched_gt)

    def compute(self) -> Dict[str, float]:
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        chamfer = self.chamfer_sum / max(self.chamfer_count, 1)
        return {
            "lane_f1": f1,
            "lane_precision": prec,
            "lane_recall": rec,
            "lane_curve_dist": chamfer,
        }

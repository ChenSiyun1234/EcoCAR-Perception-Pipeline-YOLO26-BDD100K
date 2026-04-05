"""
Evaluation metrics for both tasks.

Detection: mAP@50, mAP@50-95 using torchmetrics
Lane: thick-polyline overlap metrics (mIoU / F1) plus legacy curve distance
"""

import torch
from typing import Dict
from .losses import aggregate_lane_mask, curve_to_curve_distance


class DetectionMetrics:
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
        def safe(x):
            return max(0.0, float(x.item())) if torch.isfinite(x) else 0.0
        return {"det_map50": safe(result.get("map_50", torch.tensor(0.0))), "det_map50_95": safe(result.get("map", torch.tensor(0.0)))}


class LaneMetrics:
    def __init__(self, match_thresh_px: float = 15.0, img_size: int = 640,
                 raster_h: int = 72, raster_w: int = 128, raster_thickness: float = 0.03):
        self.match_thresh = match_thresh_px / img_size
        self.raster_h = raster_h
        self.raster_w = raster_w
        self.raster_thickness = raster_thickness
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.chamfer_sum = 0.0
        self.chamfer_count = 0
        self.intersection = 0.0
        self.union = 0.0

    @torch.no_grad()
    def update(self, pred_points: torch.Tensor, pred_exist: torch.Tensor,
               gt_points: torch.Tensor, gt_exist: torch.Tensor,
               gt_visibility: torch.Tensor | None = None,
               exist_thresh: float = 0.5):
        pred_mask_img = aggregate_lane_mask(
            pred_points, pred_exist[:, 0], None,
            height=self.raster_h, width=self.raster_w,
            thickness=self.raster_thickness, exist_thresh=exist_thresh,
            use_logits=True,
        )
        gt_mask_img = aggregate_lane_mask(
            gt_points, gt_exist, gt_visibility,
            height=self.raster_h, width=self.raster_w,
            thickness=self.raster_thickness, exist_thresh=0.5,
            use_logits=False,
        )
        pred_bin = pred_mask_img > 0.5
        gt_bin = gt_mask_img > 0.5
        self.intersection += float((pred_bin & gt_bin).sum().item())
        self.union += float((pred_bin | gt_bin).sum().item())

        pred_keep = pred_exist[:, 0].sigmoid() > exist_thresh
        gt_keep = gt_exist > 0.5
        n_pred = int(pred_keep.sum().item())
        n_gt = int(gt_keep.sum().item())
        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_gt
            return
        pred_pts = pred_points[pred_keep]
        gt_pts = gt_points[gt_keep]
        gt_vis = gt_visibility[gt_keep] if gt_visibility is not None else None
        dist = torch.zeros(n_pred, n_gt)
        for i in range(n_pred):
            for j in range(n_gt):
                geom = curve_to_curve_distance(pred_pts[i], gt_pts[j], None, gt_vis[j] if gt_vis is not None else None, 48)
                dist[i, j] = geom["sym_dist"]
        matched_pred = set()
        matched_gt = set()
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
        miou = self.intersection / max(self.union, 1.0)
        overlap_f1 = 2.0 * self.intersection / max((self.intersection + self.union), 1.0)
        return {
            "lane_f1": f1,
            "lane_precision": prec,
            "lane_recall": rec,
            "lane_curve_dist": chamfer,
            "lane_miou": miou,
            "lane_overlap_f1": overlap_f1,
        }

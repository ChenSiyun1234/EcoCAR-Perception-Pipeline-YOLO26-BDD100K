
"""Metrics for detection and geometric lane evaluation."""
import torch
from typing import Dict
from .losses import bidirectional_curve_distance, box_cxcywh_to_xyxy

class DetectionMetrics:
    def __init__(self, num_classes: int, device: str = 'cuda', iou_thresh: float = 0.5):
        self.num_classes = num_classes; self.device = device; self.iou_thresh = iou_thresh; self.reset()
    def reset(self):
        self.tp = 0; self.fp = 0; self.fn = 0
    def _iou(self, boxes1, boxes2):
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        lt = torch.max(boxes1[:,None,:2], boxes2[None,:,:2]); rb = torch.min(boxes1[:,None,2:], boxes2[None,:,2:])
        inter = (rb-lt).clamp(min=0).prod(dim=-1)
        a1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0)*(boxes1[:,3]-boxes1[:,1]).clamp(min=0)
        a2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0)*(boxes2[:,3]-boxes2[:,1]).clamp(min=0)
        union = a1[:,None] + a2[None,:] - inter
        return inter / union.clamp(min=1e-6)
    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        if pred_boxes.numel() == 0 and gt_boxes.numel() == 0: return
        matched_gt = set()
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]; pred_labels = pred_labels[order]
        for pb, pl in zip(pred_boxes, pred_labels):
            candidates = torch.where(gt_labels == pl)[0]
            if len(candidates) == 0:
                self.fp += 1; continue
            ious = self._iou(pb.unsqueeze(0), gt_boxes[candidates]).squeeze(0)
            best_rel = int(torch.argmax(ious).item()); best_iou = float(ious[best_rel].item()); best_idx = int(candidates[best_rel].item())
            if best_iou >= self.iou_thresh and best_idx not in matched_gt:
                self.tp += 1; matched_gt.add(best_idx)
            else:
                self.fp += 1
        self.fn += max(0, gt_boxes.shape[0] - len(matched_gt))
    def compute(self) -> Dict[str,float]:
        prec = self.tp / max(self.tp + self.fp, 1); rec = self.tp / max(self.tp + self.fn, 1)
        ap50 = prec * rec
        return {'det_precision': prec, 'det_recall': rec, 'det_map50': ap50}

class LaneMetrics:
    def __init__(self, match_thresh_px: float = 15.0, img_size: int = 640):
        self.match_thresh_px = match_thresh_px; self.img_size = img_size; self.reset()
    def reset(self):
        self.tp = 0; self.fp = 0; self.fn = 0; self.curve_sum = 0.0; self.curve_count = 0; self.miou_sum = 0.0; self.miou_count = 0
    def _rasterize(self, pts: torch.Tensor, h: int, w: int, thickness: int = 2):
        import cv2, numpy as np
        mask = np.zeros((h,w), dtype=np.uint8)
        arr = pts.detach().cpu().numpy().copy()
        arr[:,0] = np.clip(arr[:,0] * w, 0, w-1); arr[:,1] = np.clip(arr[:,1] * h, 0, h-1)
        arr = arr.astype(np.int32)
        if arr.shape[0] >= 2:
            cv2.polylines(mask, [arr], False, 1, thickness)
        return torch.from_numpy(mask)
    def update(self, pred_points, pred_exist, gt_points, gt_exist):
        pred_keep = (pred_exist[:,0].sigmoid() > 0.5)
        pred_pts = pred_points[pred_keep]
        gt_mask = gt_exist > 0.5
        gt_pts = gt_points[gt_mask]
        n_pred, n_gt = pred_pts.shape[0], gt_pts.shape[0]
        if n_pred == 0 and n_gt == 0: return
        matched_gt = set()
        for i in range(n_pred):
            best_j, best_d = -1, 1e9
            for j in range(n_gt):
                d = float((bidirectional_curve_distance(pred_pts[i] * self.img_size, gt_pts[j] * self.img_size)).item())
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= self.match_thresh_px and best_j not in matched_gt:
                self.tp += 1; matched_gt.add(best_j); self.curve_sum += best_d; self.curve_count += 1
                pm = self._rasterize(pred_pts[i], self.img_size, self.img_size)
                gm = self._rasterize(gt_pts[best_j], self.img_size, self.img_size)
                inter = ((pm > 0) & (gm > 0)).sum().item(); union = ((pm > 0) | (gm > 0)).sum().item()
                self.miou_sum += inter / max(union, 1); self.miou_count += 1
            else:
                self.fp += 1
        self.fn += max(0, n_gt - len(matched_gt))
    def compute(self) -> Dict[str,float]:
        prec = self.tp / max(self.tp + self.fp, 1); rec = self.tp / max(self.tp + self.fn, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
        curve = self.curve_sum / max(self.curve_count, 1); miou = self.miou_sum / max(self.miou_count, 1)
        return {'lane_f1': f1, 'lane_precision': prec, 'lane_recall': rec, 'lane_curve_dist_px': curve, 'lane_miou': miou}

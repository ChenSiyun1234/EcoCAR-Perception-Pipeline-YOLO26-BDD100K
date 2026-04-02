"""
Losses for dual-path training.

Detection:
- Hungarian matching
- cross-entropy over classes
- L1 + GIoU over boxes

Lane:
- Hungarian matching using curve-aware assignment cost inspired by recent
  vectorized map and lane work
- existence BCE
- bidirectional point-to-curve loss
- tangent consistency loss
- smoothness regularization
- optional lane type CE
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area_enc = (rb_enc - lt_enc).clamp(min=0).prod(dim=-1)
    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


def _polyline_segments(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return points[..., :-1, :], points[..., 1:, :]


def point_to_segment_distance(points: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor) -> torch.Tensor:
    """Distance from points (..., P, 2) to segments (..., S, 2) => (..., P, S)."""
    p = points.unsqueeze(-2)
    a = seg_start.unsqueeze(-3)
    b = seg_end.unsqueeze(-3)
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    t = (ap * ab).sum(dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    proj = a + t * ab
    return torch.norm(p - proj, dim=-1)


def bidirectional_point_to_curve_distance(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                                          pred_vis: torch.Tensor | None = None,
                                          gt_vis: torch.Tensor | None = None) -> torch.Tensor:
    pred_start, pred_end = _polyline_segments(pred_pts)
    gt_start, gt_end = _polyline_segments(gt_pts)

    d_pred_to_gt = point_to_segment_distance(pred_pts, gt_start, gt_end).min(dim=-1).values
    d_gt_to_pred = point_to_segment_distance(gt_pts, pred_start, pred_end).min(dim=-1).values

    if pred_vis is not None:
        pred_vis = pred_vis.float()
        d_pred_to_gt = (d_pred_to_gt * pred_vis).sum() / pred_vis.sum().clamp(min=1.0)
    else:
        d_pred_to_gt = d_pred_to_gt.mean()

    if gt_vis is not None:
        gt_vis = gt_vis.float()
        d_gt_to_pred = (d_gt_to_pred * gt_vis).sum() / gt_vis.sum().clamp(min=1.0)
    else:
        d_gt_to_pred = d_gt_to_pred.mean()

    return 0.5 * (d_pred_to_gt + d_gt_to_pred)


def tangent_consistency_loss(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                             vis: torch.Tensor | None = None) -> torch.Tensor:
    pred_t = pred_pts[1:] - pred_pts[:-1]
    gt_t = gt_pts[1:] - gt_pts[:-1]
    pred_t = pred_t / pred_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gt_t = gt_t / gt_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos = (pred_t * gt_t).sum(dim=-1)
    loss = 1.0 - cos
    if vis is not None:
        seg_vis = (vis[:-1] * vis[1:]).float()
        return (loss * seg_vis).sum() / seg_vis.sum().clamp(min=1.0)
    return loss.mean()


def smoothness_regularization(points: torch.Tensor, vis: torch.Tensor | None = None) -> torch.Tensor:
    if points.shape[0] < 3:
        return points.new_tensor(0.0)
    second = points[2:] - 2.0 * points[1:-1] + points[:-2]
    mag = torch.norm(second, dim=-1)
    if vis is not None and len(vis) >= 3:
        curv_vis = (vis[:-2] * vis[1:-1] * vis[2:]).float()
        return (mag * curv_vis).sum() / curv_vis.sum().clamp(min=1.0)
    return mag.mean()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, cls_weight: float = 2.0, l1_weight: float = 5.0, giou_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def _hungarian_match(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                         gt_classes: torch.Tensor, gt_boxes: torch.Tensor) -> list:
        B = pred_logits.shape[0]
        matches = []
        for bi in range(B):
            mask = gt_classes[bi] >= 0
            gt_cls_i = gt_classes[bi][mask]
            gt_box_i = gt_boxes[bi][mask]
            if len(gt_cls_i) == 0:
                matches.append(([], []))
                continue
            prob = pred_logits[bi].softmax(-1)
            cls_cost = -prob[:, gt_cls_i.long()]
            l1_cost = torch.cdist(pred_boxes[bi], gt_box_i, p=1)
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes[bi]), box_cxcywh_to_xyxy(gt_box_i))
            cost = self.cls_weight * cls_cost + self.l1_weight * l1_cost + self.giou_weight * (-giou)
            pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pi.tolist(), gi.tolist()))
        return matches

    def forward(self, outputs: dict, gt_classes: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        pred_logits = outputs["det_pred_logits"]
        pred_boxes = outputs["det_pred_boxes"]
        B, Q = pred_logits.shape[:2]
        device = pred_logits.device
        matches = self._hungarian_match(pred_logits, pred_boxes, gt_classes, gt_boxes)

        target_cls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=device)
        total_l1 = pred_logits.new_tensor(0.0)
        total_giou = pred_logits.new_tensor(0.0)
        n_matched = 0
        for bi, (pi, gi) in enumerate(matches):
            if len(pi) == 0:
                continue
            mask = gt_classes[bi] >= 0
            gt_cls_i = gt_classes[bi][mask]
            gt_box_i = gt_boxes[bi][mask]
            pi_t = torch.tensor(pi, dtype=torch.long, device=device)
            gi_t = torch.tensor(gi, dtype=torch.long, device=device)
            target_cls[bi, pi_t] = gt_cls_i[gi_t].long()
            total_l1 += F.l1_loss(pred_boxes[bi, pi_t], gt_box_i[gi_t], reduction="sum")
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes[bi, pi_t]), box_cxcywh_to_xyxy(gt_box_i[gi_t]))
            total_giou += (1.0 - giou.diag()).sum()
            n_matched += len(pi)

        cls_loss = F.cross_entropy(pred_logits.view(-1, self.num_classes + 1), target_cls.view(-1), weight=self._class_weights(device))
        n_matched = max(n_matched, 1)
        l1_loss = total_l1 / n_matched
        giou_loss = total_giou / n_matched
        total = self.cls_weight * cls_loss + self.l1_weight * l1_loss + self.giou_weight * giou_loss
        return total, {"det_cls": float(cls_loss.item()), "det_l1": float(l1_loss.item()), "det_giou": float(giou_loss.item())}

    def _class_weights(self, device) -> torch.Tensor:
        w = torch.ones(self.num_classes + 1, device=device)
        w[-1] = 0.1
        return w


class LaneLoss(nn.Module):
    def __init__(self, num_lane_types: int = 7, exist_weight: float = 1.5,
                 pts_weight: float = 3.0, type_weight: float = 0.5,
                 curve_weight: float = 5.0, dir_weight: float = 1.5,
                 smooth_weight: float = 0.25):
        super().__init__()
        self.exist_weight = exist_weight
        self.pts_weight = pts_weight
        self.type_weight = type_weight
        self.curve_weight = curve_weight
        self.dir_weight = dir_weight
        self.smooth_weight = smooth_weight
        self.num_lane_types = num_lane_types

    @torch.no_grad()
    def _hungarian_match(self, pred_pts: torch.Tensor, pred_exist: torch.Tensor,
                         gt_pts: torch.Tensor, gt_exist: torch.Tensor,
                         gt_vis: torch.Tensor) -> list:
        B = pred_pts.shape[0]
        matches = []
        for bi in range(B):
            gt_mask = gt_exist[bi] > 0.5
            n_gt = int(gt_mask.sum().item())
            if n_gt == 0:
                matches.append(([], []))
                continue
            gt_pts_i = gt_pts[bi][gt_mask]
            gt_vis_i = gt_vis[bi][gt_mask]
            gt_indices = torch.where(gt_mask)[0]
            Q = pred_pts.shape[1]
            cost = torch.zeros(Q, n_gt, device=pred_pts.device)
            exist_prob = pred_exist[bi, :, 0].sigmoid()
            for qi in range(Q):
                for gi in range(n_gt):
                    curve = bidirectional_point_to_curve_distance(pred_pts[bi, qi], gt_pts_i[gi], gt_vis=gt_vis_i[gi])
                    direction = tangent_consistency_loss(pred_pts[bi, qi], gt_pts_i[gi], vis=gt_vis_i[gi])
                    cost[qi, gi] = 1.0 * curve + 0.35 * direction - 0.25 * exist_prob[qi]
            pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pi.tolist(), gt_indices[gi].tolist()))
        return matches

    def forward(self, outputs: dict, gt_existence: torch.Tensor, gt_points: torch.Tensor,
                gt_visibility: torch.Tensor, gt_lane_type: torch.Tensor, has_lanes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        pred_exist = outputs["lane_exist_logits"]
        pred_pts = outputs["lane_pred_points"]
        pred_type = outputs["lane_type_logits"]
        B, Q = pred_exist.shape[:2]
        device = pred_exist.device

        if not (has_lanes > 0.5).any():
            z = pred_exist.new_tensor(0.0)
            return z, {"lane_exist": 0.0, "lane_pts": 0.0, "lane_type": 0.0, "lane_curve": 0.0, "lane_dir": 0.0, "lane_smooth": 0.0}

        matches = self._hungarian_match(pred_pts, pred_exist, gt_points, gt_existence, gt_visibility)
        exist_target = torch.zeros(B, Q, device=device)
        total_pts = pred_exist.new_tensor(0.0)
        total_curve = pred_exist.new_tensor(0.0)
        total_dir = pred_exist.new_tensor(0.0)
        total_smooth = pred_exist.new_tensor(0.0)
        total_type = pred_exist.new_tensor(0.0)
        n_matched = 0

        for bi, (pi, gi) in enumerate(matches):
            if len(pi) == 0:
                continue
            pi_t = torch.tensor(pi, dtype=torch.long, device=device)
            gi_t = torch.tensor(gi, dtype=torch.long, device=device)
            exist_target[bi, pi_t] = 1.0
            for p, g in zip(pi_t.tolist(), gi_t.tolist()):
                pred_curve = pred_pts[bi, p]
                gt_curve = gt_points[bi, g]
                gt_vis = gt_visibility[bi, g]
                vis2 = gt_vis.unsqueeze(-1)
                total_pts += (F.smooth_l1_loss(pred_curve, gt_curve, reduction="none") * vis2).sum() / vis2.sum().clamp(min=1.0)
                total_curve += bidirectional_point_to_curve_distance(pred_curve, gt_curve, gt_vis=gt_vis)
                total_dir += tangent_consistency_loss(pred_curve, gt_curve, vis=gt_vis)
                total_smooth += smoothness_regularization(pred_curve, vis=gt_vis)
                total_type += F.cross_entropy(pred_type[bi, p].unsqueeze(0), gt_lane_type[bi, g].long().unsqueeze(0))
                n_matched += 1

        exist_loss = F.binary_cross_entropy_with_logits(pred_exist.squeeze(-1), exist_target)
        n_matched = max(n_matched, 1)
        pts_loss = total_pts / n_matched
        curve_loss = total_curve / n_matched
        dir_loss = total_dir / n_matched
        smooth_loss = total_smooth / n_matched
        type_loss = total_type / n_matched if n_matched > 0 else pred_exist.new_tensor(0.0)

        total = (
            self.exist_weight * exist_loss +
            self.pts_weight * pts_loss +
            self.curve_weight * curve_loss +
            self.dir_weight * dir_loss +
            self.smooth_weight * smooth_loss +
            self.type_weight * type_loss
        )
        return total, {
            "lane_exist": float(exist_loss.item()),
            "lane_pts": float(pts_loss.item()),
            "lane_curve": float(curve_loss.item()),
            "lane_dir": float(dir_loss.item()),
            "lane_smooth": float(smooth_loss.item()),
            "lane_type": float(type_loss.item() if torch.is_tensor(type_loss) else type_loss),
        }


class DualPathLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = 7 if cfg.use_expanded_classes else 5
        self.det_loss = DetectionLoss(num_classes, cfg.det_cls_weight, cfg.det_l1_weight, cfg.det_giou_weight)
        self.lane_loss = LaneLoss(
            num_lane_types=7,
            exist_weight=cfg.lane_exist_weight,
            pts_weight=cfg.lane_pts_weight,
            type_weight=cfg.lane_type_weight,
            curve_weight=cfg.lane_curve_weight,
            dir_weight=cfg.lane_dir_weight,
            smooth_weight=cfg.lane_smooth_weight,
        )
        self.det_task_weight = cfg.det_task_weight
        self.lane_task_weight = cfg.lane_task_weight

    def _prepare_det_gt(self, det_targets: torch.Tensor, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_gt = 0
        per_image = []
        for bi in range(batch_size):
            mask = det_targets[:, 0] == bi if det_targets.numel() > 0 else torch.zeros(0, dtype=torch.bool, device=device)
            tgt = det_targets[mask] if det_targets.numel() > 0 else det_targets.new_zeros((0, 6))
            per_image.append(tgt)
            max_gt = max(max_gt, int(tgt.shape[0]))
        gt_classes = torch.full((batch_size, max_gt), -1, dtype=torch.long, device=device)
        gt_boxes = torch.zeros((batch_size, max_gt, 4), dtype=torch.float32, device=device)
        for bi, tgt in enumerate(per_image):
            n = tgt.shape[0]
            if n > 0:
                gt_classes[bi, :n] = tgt[:, 1].long()
                gt_boxes[bi, :n] = tgt[:, 2:6]
        return gt_classes, gt_boxes

    def forward(self, outputs: dict, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        images = batch["images"]
        det_targets = batch["det_targets"]
        device = images.device
        B = images.shape[0]
        gt_classes, gt_boxes = self._prepare_det_gt(det_targets, B, device)
        det_loss, det_info = self.det_loss(outputs, gt_classes, gt_boxes)
        lane_loss, lane_info = self.lane_loss(
            outputs,
            batch["lane_existence"],
            batch["lane_points"],
            batch["lane_visibility"],
            batch["lane_type"],
            batch["has_lanes"],
        )
        total = self.det_task_weight * det_loss + self.lane_task_weight * lane_loss
        info = {**det_info, **lane_info, "det_total": float(det_loss.item()), "lane_total": float(lane_loss.item())}
        return total, info

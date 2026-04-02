"""
Losses for dual-path training.

Detection loss: Focal + L1 + GIoU (DETR-style with Hungarian matching)
Lane loss: existence BCE + point smooth-L1 + type CE (with Hungarian matching)

Both use scipy's linear_sum_assignment for optimal bipartite matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple


# ── Utilities ────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two sets of boxes in xyxy format. Returns (N, M) matrix."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area_enc = (rb_enc - lt_enc).clamp(min=0).prod(dim=-1)

    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


# ── Detection Loss ──────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """DETR-style detection loss with Hungarian matching."""

    def __init__(self, num_classes: int, cls_weight: float = 2.0,
                 l1_weight: float = 5.0, giou_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def _hungarian_match(self, pred_logits: torch.Tensor,
                         pred_boxes: torch.Tensor,
                         gt_classes: torch.Tensor,
                         gt_boxes: torch.Tensor) -> list:
        """Match predictions to GT using Hungarian algorithm.

        Returns list of (pred_indices, gt_indices) tuples, one per image.
        """
        B = pred_logits.shape[0]
        matches = []

        for bi in range(B):
            # Get GT for this image
            mask = (gt_classes[bi] >= 0)  # valid GT flag
            gt_cls_i = gt_classes[bi][mask]
            gt_box_i = gt_boxes[bi][mask]

            if len(gt_cls_i) == 0:
                matches.append(([], []))
                continue

            # Cost matrix
            # Classification cost: -prob of correct class
            prob = pred_logits[bi].softmax(-1)
            cls_cost = -prob[:, gt_cls_i.long()]  # (Q, num_gt)

            # L1 box cost
            l1_cost = torch.cdist(pred_boxes[bi], gt_box_i, p=1)  # (Q, num_gt)

            # GIoU cost
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[bi]),
                box_cxcywh_to_xyxy(gt_box_i),
            )
            giou_cost = -giou  # (Q, num_gt)

            cost = (self.cls_weight * cls_cost +
                    self.l1_weight * l1_cost +
                    self.giou_weight * giou_cost)

            pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
            matches.append((pred_idx.tolist(), gt_idx.tolist()))

        return matches

    def forward(self, outputs: dict,
                gt_classes: torch.Tensor,
                gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: dict with pred_logits (B, Q, C+1), pred_boxes (B, Q, 4)
            gt_classes: (B, max_gt) class IDs, -1 for padding
            gt_boxes: (B, max_gt, 4) in cxcywh normalized
        """
        pred_logits = outputs["det_pred_logits"]
        pred_boxes = outputs["det_pred_boxes"]
        B, Q = pred_logits.shape[:2]
        device = pred_logits.device

        matches = self._hungarian_match(pred_logits, pred_boxes, gt_classes, gt_boxes)

        # Build target tensors
        target_cls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=device)
        # no-object class = num_classes (last index)

        total_l1 = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)
        n_matched = 0

        for bi, (pi, gi) in enumerate(matches):
            if len(pi) == 0:
                continue
            mask = (gt_classes[bi] >= 0)
            gt_cls_i = gt_classes[bi][mask]
            gt_box_i = gt_boxes[bi][mask]

            for p, g in zip(pi, gi):
                target_cls[bi, p] = gt_cls_i[g].long()

            pi_t = torch.tensor(pi, dtype=torch.long, device=device)
            gi_t = torch.tensor(gi, dtype=torch.long, device=device)
            total_l1 += F.l1_loss(pred_boxes[bi, pi_t], gt_box_i[gi_t], reduction="sum")

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[bi, pi_t]),
                box_cxcywh_to_xyxy(gt_box_i[gi_t]),
            )
            total_giou += (1 - giou.diag()).sum()
            n_matched += len(pi)

        # Classification: focal-like CE
        cls_loss = F.cross_entropy(
            pred_logits.view(-1, self.num_classes + 1),
            target_cls.view(-1),
            weight=self._class_weights(device),
        )

        n_matched = max(n_matched, 1)
        l1_loss = total_l1 / n_matched
        giou_loss = total_giou / n_matched

        total = (self.cls_weight * cls_loss +
                 self.l1_weight * l1_loss +
                 self.giou_weight * giou_loss)

        return total, {
            "det_cls": cls_loss.item(),
            "det_l1": l1_loss.item(),
            "det_giou": giou_loss.item(),
        }

    def _class_weights(self, device) -> torch.Tensor:
        """Down-weight the no-object class to handle class imbalance."""
        w = torch.ones(self.num_classes + 1, device=device)
        w[-1] = 0.1  # no-object gets lower weight
        return w


# ── Lane Loss ────────────────────────────────────────────────────────

class LaneLoss(nn.Module):
    """Query-based lane loss with Hungarian matching on polylines."""

    def __init__(self, num_lane_types: int = 7,
                 exist_weight: float = 2.0,
                 pts_weight: float = 5.0,
                 type_weight: float = 1.0):
        super().__init__()
        self.exist_weight = exist_weight
        self.pts_weight = pts_weight
        self.type_weight = type_weight
        self.num_lane_types = num_lane_types

    @torch.no_grad()
    def _hungarian_match(self, pred_pts: torch.Tensor,
                         pred_exist: torch.Tensor,
                         gt_pts: torch.Tensor,
                         gt_exist: torch.Tensor) -> list:
        """Match predicted lanes to GT lanes based on point distance."""
        B = pred_pts.shape[0]
        matches = []

        for bi in range(B):
            gt_mask = gt_exist[bi] > 0.5
            n_gt = gt_mask.sum().item()
            if n_gt == 0:
                matches.append(([], []))
                continue

            gt_pts_i = gt_pts[bi][gt_mask]  # (n_gt, N, 2)
            gt_indices = torch.where(gt_mask)[0]

            # Cost: average point distance between predicted and GT lanes
            # pred_pts: (Q, N, 2), gt_pts_i: (n_gt, N, 2)
            Q = pred_pts.shape[1]
            cost = torch.zeros(Q, n_gt, device=pred_pts.device)
            for qi in range(Q):
                for gi in range(n_gt):
                    cost[qi, gi] = (pred_pts[bi, qi] - gt_pts_i[gi]).abs().mean()

            # Add existence cost: penalize matching to low-confidence predictions
            exist_prob = pred_exist[bi, :, 0].sigmoid()
            cost -= 0.5 * exist_prob.unsqueeze(1)  # prefer higher confidence

            pi, gi = linear_sum_assignment(cost.cpu().numpy())
            matches.append((pi.tolist(), gt_indices[gi].tolist()))

        return matches

    def forward(self, outputs: dict,
                gt_existence: torch.Tensor,
                gt_points: torch.Tensor,
                gt_visibility: torch.Tensor,
                gt_lane_type: torch.Tensor,
                has_lanes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: dict with lane_exist_logits, lane_pred_points, etc.
            gt_existence: (B, max_lanes) float
            gt_points: (B, max_lanes, N, 2) normalized
            gt_visibility: (B, max_lanes, N) float
            gt_lane_type: (B, max_lanes) long
            has_lanes: (B,) float — 1.0 if image has lane annotations
        """
        pred_exist = outputs["lane_exist_logits"]    # (B, Q, 1)
        pred_pts = outputs["lane_pred_points"]       # (B, Q, N, 2)
        pred_type = outputs["lane_type_logits"]      # (B, Q, T)

        B, Q = pred_exist.shape[:2]
        device = pred_exist.device

        # Only compute lane loss for images with lane annotations
        lane_mask = has_lanes > 0.5
        if not lane_mask.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {"lane_exist": 0.0, "lane_pts": 0.0, "lane_type": 0.0}

        matches = self._hungarian_match(pred_pts, pred_exist, gt_points, gt_existence)

        # Existence targets
        exist_target = torch.zeros(B, Q, device=device)
        total_pts_loss = torch.tensor(0.0, device=device)
        total_type_loss = torch.tensor(0.0, device=device)
        n_matched = 0

        for bi, (pi, gi) in enumerate(matches):
            if not lane_mask[bi]:
                continue
            for p, g in zip(pi, gi):
                exist_target[bi, p] = 1.0

                # Point loss: smooth L1 weighted by visibility
                vis = gt_visibility[bi, g]  # (N,)
                pts_diff = F.smooth_l1_loss(
                    pred_pts[bi, p], gt_points[bi, g], reduction="none")
                pts_diff = (pts_diff * vis.unsqueeze(-1)).sum() / vis.sum().clamp(min=1)
                total_pts_loss += pts_diff

                # Type loss
                total_type_loss += F.cross_entropy(
                    pred_type[bi, p].unsqueeze(0),
                    gt_lane_type[bi, g].unsqueeze(0).to(device),
                )
                n_matched += 1

        # Existence loss (only for images with lane annotations)
        exist_loss = F.binary_cross_entropy_with_logits(
            pred_exist[lane_mask, :, 0], exist_target[lane_mask],
            pos_weight=torch.tensor(3.0, device=device),
        )

        n_matched = max(n_matched, 1)
        pts_loss = total_pts_loss / n_matched
        type_loss = total_type_loss / n_matched

        total = (self.exist_weight * exist_loss +
                 self.pts_weight * pts_loss +
                 self.type_weight * type_loss)

        return total, {
            "lane_exist": exist_loss.item(),
            "lane_pts": pts_loss.item(),
            "lane_type": type_loss.item(),
        }


# ── Combined Loss ────────────────────────────────────────────────────

class DualPathLoss(nn.Module):
    """Combined detection + lane loss."""

    def __init__(self, cfg):
        super().__init__()
        nc = 7 if cfg.use_expanded_classes else 5
        self.det_loss = DetectionLoss(
            num_classes=nc,
            cls_weight=cfg.det_cls_weight,
            l1_weight=cfg.det_l1_weight,
            giou_weight=cfg.det_giou_weight,
        )
        from .config import NUM_LANE_TYPES
        self.lane_loss = LaneLoss(
            num_lane_types=NUM_LANE_TYPES,
            exist_weight=cfg.lane_exist_weight,
            pts_weight=cfg.lane_pts_weight,
            type_weight=cfg.lane_type_weight,
        )
        self.det_weight = cfg.det_task_weight
        self.lane_weight = cfg.lane_task_weight

    def forward(self, outputs: dict, batch: dict) -> Tuple[torch.Tensor, dict]:
        # Prepare detection GT in (B, max_gt) format
        det_gt = self._prepare_det_gt(outputs, batch)

        det_loss, det_info = self.det_loss(outputs, det_gt["classes"], det_gt["boxes"])

        lane_loss, lane_info = self.lane_loss(
            outputs,
            batch["lane_existence"],
            batch["lane_points"],
            batch["lane_visibility"],
            batch["lane_type"],
            batch["has_lanes"],
        )

        total = self.det_weight * det_loss + self.lane_weight * lane_loss
        info = {**det_info, **lane_info,
                "det_total": det_loss.item(), "lane_total": lane_loss.item()}
        return total, info

    def _prepare_det_gt(self, outputs, batch) -> dict:
        """Convert batched detection targets to (B, max_gt, ...) format."""
        det_targets = batch["det_targets"]  # (total, 6): batch_idx, cls, cx, cy, w, h
        B = outputs["det_pred_logits"].shape[0]
        device = outputs["det_pred_logits"].device

        # Find max GT per image
        if det_targets.shape[0] == 0:
            return {
                "classes": torch.full((B, 1), -1, dtype=torch.long, device=device),
                "boxes": torch.zeros(B, 1, 4, device=device),
            }

        max_gt = max(1, max(int((det_targets[:, 0] == bi).sum().item())
                         for bi in range(B)))

        gt_classes = torch.full((B, max_gt), -1, dtype=torch.long, device=device)
        gt_boxes = torch.zeros(B, max_gt, 4, device=device)

        for bi in range(B):
            mask = det_targets[:, 0] == bi
            tgt = det_targets[mask]
            n = tgt.shape[0]
            if n > 0:
                gt_classes[bi, :n] = tgt[:, 1].long().to(device)
                gt_boxes[bi, :n] = tgt[:, 2:6].to(device)

        return {"classes": gt_classes, "boxes": gt_boxes}

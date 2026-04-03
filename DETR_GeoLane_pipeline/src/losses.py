"""
Losses for dual-path training.

Detection loss: DETR / RT-DETRv2-style matching with classification + box loss
Lane loss: hybrid geometric + raster-overlap supervision

The lane branch now mixes two complementary signals:
  1) vector geometry loss on matched polylines
  2) thick-line overlap loss via low-resolution soft rasterization

This makes optimization much less brittle early in training while keeping the
structured polyline representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
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


def _valid_curve_points(points: torch.Tensor, visibility: torch.Tensor | None = None) -> torch.Tensor:
    if visibility is None:
        return points
    mask = visibility > 0.5
    if mask.sum() >= 2:
        return points[mask]
    return points


def _segments_from_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] < 2:
        return points[:1], points[:1]
    return points[:-1], points[1:]


def point_to_polyline_distance(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    if polyline.shape[0] == 0:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    if polyline.shape[0] == 1:
        return torch.norm(points - polyline[0], dim=-1)
    seg_a, seg_b = _segments_from_points(polyline)
    ab = seg_b - seg_a
    ap = points[:, None, :] - seg_a[None, :, :]
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-8)
    t = (ap * ab[None, :, :]).sum(dim=-1) / denom[None, :]
    t = t.clamp(0.0, 1.0)
    proj = seg_a[None, :, :] + t[..., None] * ab[None, :, :]
    dist = torch.norm(points[:, None, :] - proj, dim=-1)
    return dist.min(dim=1).values


def polyline_tangents(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] < 2:
        return torch.zeros_like(points)
    fwd = torch.zeros_like(points)
    fwd[1:-1] = points[2:] - points[:-2]
    fwd[0] = points[1] - points[0]
    fwd[-1] = points[-1] - points[-2]
    norm = torch.norm(fwd, dim=-1, keepdim=True).clamp(min=1e-8)
    return fwd / norm


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


def curve_to_curve_distance(pred_points: torch.Tensor, gt_points: torch.Tensor,
                            pred_visibility: torch.Tensor | None = None,
                            gt_visibility: torch.Tensor | None = None,
                            resample_n: int = 96) -> Dict[str, torch.Tensor]:
    pred_curve = _valid_curve_points(pred_points, pred_visibility)
    gt_curve = _valid_curve_points(gt_points, gt_visibility)
    if pred_curve.shape[0] < 2:
        pred_curve = pred_points
    if gt_curve.shape[0] < 2:
        gt_curve = gt_points
    pred_rs = resample_polyline(pred_curve, resample_n)
    gt_rs = resample_polyline(gt_curve, resample_n)
    d_pred_to_gt = point_to_polyline_distance(pred_rs, gt_rs).mean()
    d_gt_to_pred = point_to_polyline_distance(gt_rs, pred_rs).mean()
    sym_dist = 0.5 * (d_pred_to_gt + d_gt_to_pred)
    pred_tan = polyline_tangents(pred_rs)
    gt_tan = polyline_tangents(gt_rs)
    tan_align = 1.0 - (pred_tan * gt_tan).sum(dim=-1).abs().mean()
    pred_second = pred_rs[2:] - 2 * pred_rs[1:-1] + pred_rs[:-2]
    gt_second = gt_rs[2:] - 2 * gt_rs[1:-1] + gt_rs[:-2]
    curvature_gap = F.smooth_l1_loss(pred_second, gt_second, reduction="mean")
    return {"sym_dist": sym_dist, "tan": tan_align, "curvature": curvature_gap}


def _grid_xy(height: int, width: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def soft_polyline_mask(points: torch.Tensor,
                       visibility: torch.Tensor | None = None,
                       height: int = 72,
                       width: int = 128,
                       thickness: float = 0.03,
                       sharpness: float = 80.0) -> torch.Tensor:
    curve = _valid_curve_points(points, visibility)
    if curve.shape[0] < 2:
        curve = points
    grid = _grid_xy(height, width, points.device, points.dtype)
    dist = point_to_polyline_distance(grid, curve)
    mask = torch.sigmoid((thickness - dist) * sharpness)
    return mask.view(height, width)


def aggregate_lane_mask(points: torch.Tensor,
                        existence: torch.Tensor,
                        visibility: torch.Tensor | None = None,
                        height: int = 72,
                        width: int = 128,
                        thickness: float = 0.03,
                        sharpness: float = 80.0,
                        exist_thresh: float = 0.5,
                        use_logits: bool = False) -> torch.Tensor:
    device = points.device
    dtype = points.dtype
    out = torch.zeros(height, width, device=device, dtype=dtype)
    q = points.shape[0]
    for i in range(q):
        ex = existence[i]
        score = ex.sigmoid() if use_logits else ex
        score = score.squeeze()
        if float(score.detach().item()) <= exist_thresh:
            continue
        vis_i = visibility[i] if visibility is not None else None
        lane_mask = soft_polyline_mask(points[i], vis_i, height, width, thickness, sharpness)
        out = torch.maximum(out, lane_mask * score.clamp(0.0, 1.0))
    return out


class DetectionLoss(nn.Module):
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
        b = pred_logits.shape[0]
        matches = []
        for bi in range(b):
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
            giou_cost = -giou
            cost = self.cls_weight * cls_cost + self.l1_weight * l1_cost + self.giou_weight * giou_cost
            pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pred_idx.tolist(), gt_idx.tolist()))
        return matches

    def _loss_single(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                     gt_classes: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        b, q = pred_logits.shape[:2]
        device = pred_logits.device
        matches = self._hungarian_match(pred_logits, pred_boxes, gt_classes, gt_boxes)
        target_cls = torch.full((b, q), self.num_classes, dtype=torch.long, device=device)
        total_l1 = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)
        n_matched = 0
        for bi, (pi, gi) in enumerate(matches):
            if len(pi) == 0:
                continue
            mask = gt_classes[bi] >= 0
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
            total_giou += (1.0 - giou.diag()).sum()
            n_matched += len(pi)
        cls_loss = F.cross_entropy(pred_logits.reshape(-1, self.num_classes + 1), target_cls.view(-1))
        n = max(n_matched, 1)
        l1_loss = total_l1 / n
        giou_loss = total_giou / n
        total = self.cls_weight * cls_loss + self.l1_weight * l1_loss + self.giou_weight * giou_loss
        return total, {
            "det_cls": float(cls_loss.item()),
            "det_l1": float(l1_loss.item()),
            "det_giou": float(giou_loss.item()),
        }

    def forward(self, outputs: dict, gt_classes: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        total, info = self._loss_single(outputs["det_pred_logits"], outputs["det_pred_boxes"], gt_classes, gt_boxes)
        aux_outputs = outputs.get("det_aux_outputs", [])
        if aux_outputs:
            aux_total = torch.tensor(0.0, device=gt_boxes.device)
            for aux in aux_outputs:
                aux_loss, _ = self._loss_single(aux["pred_logits"], aux["pred_boxes"], gt_classes, gt_boxes)
                aux_total += aux_loss
            total = total + 0.5 * aux_total / max(len(aux_outputs), 1)
        return total, info


class LaneLoss(nn.Module):
    def __init__(self, num_lane_types: int = 7,
                 exist_weight: float = 2.0,
                 pts_weight: float = 5.0,
                 type_weight: float = 1.0,
                 tangent_weight: float = 1.0,
                 curvature_weight: float = 0.5,
                 overlap_weight: float = 2.0,
                 match_resample_n: int = 64,
                 loss_resample_n: int = 96,
                 raster_h: int = 72,
                 raster_w: int = 128,
                 raster_thickness: float = 0.03):
        super().__init__()
        self.exist_weight = exist_weight
        self.pts_weight = pts_weight
        self.type_weight = type_weight
        self.tangent_weight = tangent_weight
        self.curvature_weight = curvature_weight
        self.overlap_weight = overlap_weight
        self.num_lane_types = num_lane_types
        self.match_resample_n = match_resample_n
        self.loss_resample_n = loss_resample_n
        self.raster_h = raster_h
        self.raster_w = raster_w
        self.raster_thickness = raster_thickness

    @torch.no_grad()
    def _hungarian_match(self, pred_pts: torch.Tensor, pred_exist: torch.Tensor,
                         gt_pts: torch.Tensor, gt_exist: torch.Tensor,
                         gt_visibility: torch.Tensor | None = None) -> list:
        b = pred_pts.shape[0]
        matches = []
        for bi in range(b):
            gt_mask = gt_exist[bi] > 0.5
            n_gt = int(gt_mask.sum().item())
            if n_gt == 0:
                matches.append(([], []))
                continue
            gt_pts_i = gt_pts[bi][gt_mask]
            gt_vis_i = gt_visibility[bi][gt_mask] if gt_visibility is not None else None
            gt_indices = torch.where(gt_mask)[0]
            q = pred_pts.shape[1]
            cost = torch.zeros(q, n_gt, device=pred_pts.device)
            for qi in range(q):
                for gi in range(n_gt):
                    geom = curve_to_curve_distance(
                        pred_pts[bi, qi], gt_pts_i[gi], None,
                        gt_vis_i[gi] if gt_vis_i is not None else None,
                        self.match_resample_n,
                    )
                    pred_mask = soft_polyline_mask(pred_pts[bi, qi], None, self.raster_h, self.raster_w, self.raster_thickness)
                    gt_mask_img = soft_polyline_mask(
                        gt_pts_i[gi],
                        gt_vis_i[gi] if gt_vis_i is not None else None,
                        self.raster_h, self.raster_w, self.raster_thickness,
                    )
                    inter = torch.minimum(pred_mask, gt_mask_img).sum()
                    union = torch.maximum(pred_mask, gt_mask_img).sum().clamp(min=1e-6)
                    overlap_cost = 1.0 - inter / union
                    cost[qi, gi] = geom["sym_dist"] + 0.35 * geom["tan"] + 0.20 * geom["curvature"] + 0.75 * overlap_cost
            exist_prob = pred_exist[bi, :, 0].sigmoid()
            cost -= 0.25 * exist_prob.unsqueeze(1)
            pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pi.tolist(), gt_indices[gi].tolist()))
        return matches

    def _loss_single(self, outputs: dict,
                     gt_existence: torch.Tensor,
                     gt_points: torch.Tensor,
                     gt_visibility: torch.Tensor,
                     gt_lane_type: torch.Tensor,
                     has_lanes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        pred_exist = outputs["lane_exist_logits"]
        pred_pts = outputs["lane_pred_points"]
        pred_type = outputs["lane_type_logits"]
        pred_vis = outputs.get("lane_vis_logits")
        b, q = pred_exist.shape[:2]
        device = pred_exist.device
        lane_mask = has_lanes > 0.5
        if not lane_mask.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "lane_exist": 0.0,
                "lane_curve": 0.0,
                "lane_tangent": 0.0,
                "lane_curvature": 0.0,
                "lane_overlap": 0.0,
                "lane_type": 0.0,
            }
        matches = self._hungarian_match(pred_pts, pred_exist, gt_points, gt_existence, gt_visibility)
        exist_target = torch.zeros(b, q, device=device)
        total_curve = torch.tensor(0.0, device=device)
        total_tangent = torch.tensor(0.0, device=device)
        total_curvature = torch.tensor(0.0, device=device)
        total_overlap = torch.tensor(0.0, device=device)
        total_type_loss = torch.tensor(0.0, device=device)
        total_vis_loss = torch.tensor(0.0, device=device)
        n_matched = 0
        for bi, (pi, gi) in enumerate(matches):
            if not lane_mask[bi]:
                continue
            for p, g in zip(pi, gi):
                exist_target[bi, p] = 1.0
                geom = curve_to_curve_distance(pred_pts[bi, p], gt_points[bi, g], None, gt_visibility[bi, g], self.loss_resample_n)
                total_curve += geom["sym_dist"]
                total_tangent += geom["tan"]
                total_curvature += geom["curvature"]
                pred_mask_img = soft_polyline_mask(pred_pts[bi, p], None, self.raster_h, self.raster_w, self.raster_thickness)
                gt_mask_img = soft_polyline_mask(gt_points[bi, g], gt_visibility[bi, g], self.raster_h, self.raster_w, self.raster_thickness)
                inter = (pred_mask_img * gt_mask_img).sum()
                union = (pred_mask_img + gt_mask_img - pred_mask_img * gt_mask_img).sum().clamp(min=1e-6)
                dice = 1.0 - (2.0 * inter + 1e-6) / (pred_mask_img.sum() + gt_mask_img.sum() + 1e-6)
                iou = 1.0 - inter / union
                total_overlap += 0.5 * (iou + dice)
                total_type_loss += F.cross_entropy(pred_type[bi, p].unsqueeze(0), gt_lane_type[bi, g].unsqueeze(0).to(device))
                if pred_vis is not None:
                    total_vis_loss += F.binary_cross_entropy_with_logits(pred_vis[bi, p], gt_visibility[bi, g].to(device))
                n_matched += 1
        exist_loss = F.binary_cross_entropy_with_logits(
            pred_exist[lane_mask, :, 0], exist_target[lane_mask],
            pos_weight=torch.tensor(3.0, device=device),
        )
        n = max(n_matched, 1)
        curve_loss = total_curve / n
        tangent_loss = total_tangent / n
        curvature_loss = total_curvature / n
        overlap_loss = total_overlap / n
        type_loss = total_type_loss / n
        vis_loss = total_vis_loss / n
        total = (
            self.exist_weight * exist_loss +
            self.pts_weight * curve_loss +
            self.tangent_weight * tangent_loss +
            self.curvature_weight * curvature_loss +
            self.overlap_weight * overlap_loss +
            self.type_weight * type_loss +
            0.5 * vis_loss
        )
        return total, {
            "lane_exist": float(exist_loss.item()),
            "lane_curve": float(curve_loss.item()),
            "lane_tangent": float(tangent_loss.item()),
            "lane_curvature": float(curvature_loss.item()),
            "lane_overlap": float(overlap_loss.item()),
            "lane_type": float(type_loss.item()),
            "lane_vis": float(vis_loss.item()),
        }

    def forward(self, outputs: dict,
                gt_existence: torch.Tensor,
                gt_points: torch.Tensor,
                gt_visibility: torch.Tensor,
                gt_lane_type: torch.Tensor,
                has_lanes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        total, info = self._loss_single(outputs, gt_existence, gt_points, gt_visibility, gt_lane_type, has_lanes)
        aux_outputs = outputs.get("lane_aux_outputs", [])
        if aux_outputs:
            aux_total = torch.tensor(0.0, device=gt_points.device)
            for aux in aux_outputs:
                aux_pack = {
                    "lane_exist_logits": aux["exist_logits"],
                    "lane_pred_points": aux["pred_points"],
                    "lane_vis_logits": aux["vis_logits"],
                    "lane_type_logits": aux["type_logits"],
                }
                aux_loss, _ = self._loss_single(aux_pack, gt_existence, gt_points, gt_visibility, gt_lane_type, has_lanes)
                aux_total += aux_loss
            total = total + 0.5 * aux_total / max(len(aux_outputs), 1)
        return total, info


class DualPathLoss(nn.Module):
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
            overlap_weight=getattr(cfg, "lane_overlap_weight", 2.0),
            raster_h=getattr(cfg, "lane_raster_h", 72),
            raster_w=getattr(cfg, "lane_raster_w", 128),
            raster_thickness=getattr(cfg, "lane_raster_thickness", 0.03),
        )
        self.det_weight = cfg.det_task_weight
        self.lane_weight = cfg.lane_task_weight

    def forward(self, outputs: dict, batch: dict) -> Tuple[torch.Tensor, dict]:
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
        info = {**det_info, **lane_info, "det_total": det_loss.item(), "lane_total": lane_loss.item()}
        return total, info

    def _prepare_det_gt(self, outputs, batch) -> dict:
        det_targets = batch["det_targets"]
        b = outputs["det_pred_logits"].shape[0]
        device = outputs["det_pred_logits"].device
        if det_targets.shape[0] == 0:
            return {"classes": torch.full((b, 1), -1, dtype=torch.long, device=device), "boxes": torch.zeros(b, 1, 4, device=device)}
        max_gt = max(1, max(int((det_targets[:, 0] == bi).sum().item()) for bi in range(b)))
        gt_classes = torch.full((b, max_gt), -1, dtype=torch.long, device=device)
        gt_boxes = torch.zeros(b, max_gt, 4, device=device)
        for bi in range(b):
            mask = det_targets[:, 0] == bi
            tgt = det_targets[mask]
            n = tgt.shape[0]
            if n > 0:
                gt_classes[bi, :n] = tgt[:, 1].long()
                gt_boxes[bi, :n] = tgt[:, 2:6]
        return {"classes": gt_classes, "boxes": gt_boxes}

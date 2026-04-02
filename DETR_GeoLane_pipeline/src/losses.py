
"""Detection and geometry-aware lane losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    lt = torch.max(boxes1[:,None,:2], boxes2[None,:,:2])
    rb = torch.min(boxes1[:,None,2:], boxes2[None,:,2:])
    inter = (rb-lt).clamp(min=0).prod(dim=-1)
    union = area1[:,None] + area2[None,:] - inter
    iou = inter / union.clamp(min=1e-6)
    lt_enc = torch.min(boxes1[:,None,:2], boxes2[None,:,:2])
    rb_enc = torch.max(boxes1[:,None,2:], boxes2[None,:,2:])
    area_enc = (rb_enc-lt_enc).clamp(min=0).prod(dim=-1)
    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


def _segments(poly: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return poly[:-1], poly[1:]


def point_to_polyline_distance(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """Distance from each point to the closest point on any line segment in the polyline."""
    if polyline.shape[0] < 2:
        return torch.cdist(points, polyline[:1]).squeeze(-1)
    a, b = _segments(polyline)
    ab = b - a  # (S,2)
    ap = points[:,None,:] - a[None,:,:]  # (P,S,2)
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-9)  # (S,)
    t = (ap * ab[None,:,:]).sum(dim=-1) / denom[None,:]
    t = t.clamp(0.0, 1.0)
    proj = a[None,:,:] + t[:,:,None] * ab[None,:,:]
    d = ((points[:,None,:] - proj) ** 2).sum(dim=-1).sqrt()
    return d.min(dim=1).values


def bidirectional_curve_distance(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                                 pred_vis: torch.Tensor = None, gt_vis: torch.Tensor = None) -> torch.Tensor:
    if pred_vis is None:
        pred_vis = torch.ones(pred_pts.shape[0], device=pred_pts.device, dtype=torch.bool)
    if gt_vis is None:
        gt_vis = torch.ones(gt_pts.shape[0], device=gt_pts.device, dtype=torch.bool)
    pred_poly = pred_pts[pred_vis]
    gt_poly = gt_pts[gt_vis]
    if pred_poly.shape[0] == 0 or gt_poly.shape[0] == 0:
        return pred_pts.new_tensor(1.0)
    d1 = point_to_polyline_distance(pred_poly, gt_poly).mean()
    d2 = point_to_polyline_distance(gt_poly, pred_poly).mean()
    return d1 + d2


def tangent_direction_loss(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                           pred_vis: torch.Tensor = None, gt_vis: torch.Tensor = None) -> torch.Tensor:
    if pred_vis is None: pred_vis = torch.ones(pred_pts.shape[0], dtype=torch.bool, device=pred_pts.device)
    if gt_vis is None: gt_vis = torch.ones(gt_pts.shape[0], dtype=torch.bool, device=gt_pts.device)
    pred = pred_pts[pred_vis]
    gt = gt_pts[gt_vis]
    n = min(pred.shape[0], gt.shape[0])
    if n < 3:
        return pred_pts.new_tensor(0.0)
    pred = pred[:n]; gt = gt[:n]
    tp = F.normalize(pred[1:] - pred[:-1], dim=-1, eps=1e-6)
    tg = F.normalize(gt[1:] - gt[:-1], dim=-1, eps=1e-6)
    return (1.0 - (tp * tg).sum(dim=-1)).mean()


def curve_smoothness_loss(pts: torch.Tensor, vis: torch.Tensor = None) -> torch.Tensor:
    if vis is None:
        vis = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
    p = pts[vis]
    if p.shape[0] < 3:
        return pts.new_tensor(0.0)
    second = p[2:] - 2*p[1:-1] + p[:-2]
    return second.norm(dim=-1).mean()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, cls_weight: float = 2.0, l1_weight: float = 5.0, giou_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes; self.cls_weight = cls_weight; self.l1_weight = l1_weight; self.giou_weight = giou_weight

    @torch.no_grad()
    def _hungarian_match(self, pred_logits, pred_boxes, gt_classes, gt_boxes):
        B = pred_logits.shape[0]; matches = []
        for bi in range(B):
            mask = gt_classes[bi] >= 0
            gt_cls_i = gt_classes[bi][mask]; gt_box_i = gt_boxes[bi][mask]
            if len(gt_cls_i) == 0:
                matches.append(([], [])); continue
            prob = pred_logits[bi].softmax(-1)
            cls_cost = -prob[:, gt_cls_i.long()]
            l1_cost = torch.cdist(pred_boxes[bi], gt_box_i, p=1)
            giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes[bi]), box_cxcywh_to_xyxy(gt_box_i))
            cost = self.cls_weight*cls_cost + self.l1_weight*l1_cost + self.giou_weight*giou_cost
            pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pi.tolist(), gi.tolist()))
        return matches

    def forward(self, outputs, gt_classes, gt_boxes):
        pred_logits = outputs['det_pred_logits']; pred_boxes = outputs['det_pred_boxes']
        B, Q = pred_logits.shape[:2]; device = pred_logits.device
        matches = self._hungarian_match(pred_logits, pred_boxes, gt_classes, gt_boxes)
        target_cls = torch.full((B,Q), self.num_classes, dtype=torch.long, device=device)
        total_l1 = pred_boxes.new_tensor(0.0); total_giou = pred_boxes.new_tensor(0.0); n_matched = 0
        for bi,(pi,gi) in enumerate(matches):
            if len(pi) == 0: continue
            mask = gt_classes[bi] >= 0
            gt_cls_i = gt_classes[bi][mask]; gt_box_i = gt_boxes[bi][mask]
            pi_t = torch.tensor(pi, dtype=torch.long, device=device)
            gi_t = torch.tensor(gi, dtype=torch.long, device=device)
            target_cls[bi, pi_t] = gt_cls_i[gi_t].long()
            total_l1 += F.l1_loss(pred_boxes[bi, pi_t], gt_box_i[gi_t], reduction='sum')
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes[bi, pi_t]), box_cxcywh_to_xyxy(gt_box_i[gi_t]))
            total_giou += (1 - giou.diag()).sum(); n_matched += len(pi)
        w = torch.ones(self.num_classes + 1, device=device); w[-1] = 0.1
        cls_loss = F.cross_entropy(pred_logits.view(-1, self.num_classes + 1), target_cls.view(-1), weight=w)
        n_matched = max(n_matched, 1)
        l1_loss = total_l1 / n_matched; giou_loss = total_giou / n_matched
        total = self.cls_weight*cls_loss + self.l1_weight*l1_loss + self.giou_weight*giou_loss
        return total, {'det_cls': cls_loss.item(), 'det_l1': l1_loss.item(), 'det_giou': giou_loss.item()}


class LaneLoss(nn.Module):
    def __init__(self, num_lane_types: int = 7, exist_weight: float = 2.0, pts_weight: float = 5.0,
                 type_weight: float = 1.0, dir_weight: float = 1.5, smooth_weight: float = 0.5):
        super().__init__()
        self.exist_weight = exist_weight; self.pts_weight = pts_weight; self.type_weight = type_weight
        self.dir_weight = dir_weight; self.smooth_weight = smooth_weight; self.num_lane_types = num_lane_types

    @torch.no_grad()
    def _hungarian_match(self, pred_pts, pred_exist, gt_pts, gt_exist, gt_visibility):
        B = pred_pts.shape[0]; matches = []
        for bi in range(B):
            gt_mask = gt_exist[bi] > 0.5
            n_gt = int(gt_mask.sum().item())
            if n_gt == 0:
                matches.append(([], [])); continue
            gt_pts_i = gt_pts[bi][gt_mask]; gt_vis_i = gt_visibility[bi][gt_mask]; gt_indices = torch.where(gt_mask)[0]
            Q = pred_pts.shape[1]
            cost = torch.zeros(Q, n_gt, device=pred_pts.device)
            exist_prob = pred_exist[bi,:,0].sigmoid()
            for qi in range(Q):
                pred_vis = torch.ones(pred_pts.shape[2], device=pred_pts.device, dtype=torch.bool)
                for gj in range(n_gt):
                    geom = bidirectional_curve_distance(pred_pts[bi, qi], gt_pts_i[gj], pred_vis, gt_vis_i[gj] > 0.5)
                    cost[qi, gj] = geom
            cost = cost - 0.2 * exist_prob.unsqueeze(1)
            pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pi.tolist(), gt_indices[gi].tolist()))
        return matches

    def forward(self, outputs, gt_existence, gt_points, gt_visibility, gt_lane_type, has_lanes):
        pred_exist = outputs['lane_exist_logits']; pred_pts = outputs['lane_pred_points']; pred_type = outputs['lane_type_logits']
        B, Q = pred_exist.shape[:2]; device = pred_exist.device
        lane_mask = has_lanes > 0.5
        if not lane_mask.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {'lane_exist':0.0,'lane_curve':0.0,'lane_dir':0.0,'lane_smooth':0.0,'lane_type':0.0}
        matches = self._hungarian_match(pred_pts, pred_exist, gt_points, gt_existence, gt_visibility)
        exist_target = torch.zeros(B,Q, device=device)
        total_curve = pred_pts.new_tensor(0.0); total_dir = pred_pts.new_tensor(0.0); total_smooth = pred_pts.new_tensor(0.0); total_type = pred_pts.new_tensor(0.0)
        n_matched = 0
        for bi,(pi,gi) in enumerate(matches):
            if not lane_mask[bi]:
                continue
            for p,g in zip(pi,gi):
                exist_target[bi,p] = 1.0
                gt_vis = gt_visibility[bi,g] > 0.5
                pred_vis = torch.ones(pred_pts.shape[2], device=device, dtype=torch.bool)
                total_curve += bidirectional_curve_distance(pred_pts[bi,p], gt_points[bi,g], pred_vis, gt_vis)
                total_dir += tangent_direction_loss(pred_pts[bi,p], gt_points[bi,g], pred_vis, gt_vis)
                total_smooth += curve_smoothness_loss(pred_pts[bi,p], pred_vis)
                total_type += F.cross_entropy(pred_type[bi,p].unsqueeze(0), gt_lane_type[bi,g].unsqueeze(0).to(device))
                n_matched += 1
        exist_loss = F.binary_cross_entropy_with_logits(pred_exist[lane_mask,:,0], exist_target[lane_mask], pos_weight=torch.tensor(3.0, device=device))
        n_matched = max(n_matched, 1)
        curve_loss = total_curve / n_matched; dir_loss = total_dir / n_matched; smooth_loss = total_smooth / n_matched; type_loss = total_type / n_matched
        total = self.exist_weight*exist_loss + self.pts_weight*curve_loss + self.dir_weight*dir_loss + self.smooth_weight*smooth_loss + self.type_weight*type_loss
        return total, {'lane_exist': exist_loss.item(), 'lane_curve': curve_loss.item(), 'lane_dir': dir_loss.item(), 'lane_smooth': smooth_loss.item(), 'lane_type': type_loss.item()}


class DualPathLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nc = 7 if cfg.use_expanded_classes else 5
        self.det_loss = DetectionLoss(num_classes=nc, cls_weight=cfg.det_cls_weight, l1_weight=cfg.det_l1_weight, giou_weight=cfg.det_giou_weight)
        from .config import NUM_LANE_TYPES
        self.lane_loss = LaneLoss(num_lane_types=NUM_LANE_TYPES, exist_weight=cfg.lane_exist_weight, pts_weight=cfg.lane_pts_weight, type_weight=cfg.lane_type_weight, dir_weight=cfg.lane_dir_weight, smooth_weight=cfg.lane_smooth_weight)
        self.det_task_weight = cfg.det_task_weight; self.lane_task_weight = cfg.lane_task_weight

    def _pack_det_gt(self, det_targets, batch_size, device):
        max_gt = 0
        for bi in range(batch_size):
            max_gt = max(max_gt, int((det_targets[:,0] == bi).sum().item()) if det_targets.numel() > 0 else 0)
        max_gt = max(max_gt, 1)
        gt_classes = torch.full((batch_size, max_gt), -1, dtype=torch.long, device=device)
        gt_boxes = torch.zeros((batch_size, max_gt, 4), dtype=torch.float32, device=device)
        if det_targets.numel() == 0:
            return gt_classes, gt_boxes
        for bi in range(batch_size):
            rows = det_targets[det_targets[:,0] == bi]
            n = rows.shape[0]
            if n > 0:
                gt_classes[bi,:n] = rows[:,1].long()
                gt_boxes[bi,:n] = rows[:,2:6]
        return gt_classes, gt_boxes

    def forward(self, outputs, batch):
        device = outputs['det_pred_logits'].device
        B = outputs['det_pred_logits'].shape[0]
        gt_classes, gt_boxes = self._pack_det_gt(batch['det_targets'], B, device)
        det_total, det_info = self.det_loss(outputs, gt_classes, gt_boxes)
        lane_total, lane_info = self.lane_loss(outputs, batch['lane_existence'].to(device), batch['lane_points'].to(device), batch['lane_visibility'].to(device), batch['lane_type'].to(device), batch['has_lanes'].to(device))
        total = self.det_task_weight * det_total + self.lane_task_weight * lane_total
        info = dict(det_info); info.update(lane_info); info['det_total'] = det_total.item(); info['lane_total'] = lane_total.item(); info['total'] = total.item()
        return total, info

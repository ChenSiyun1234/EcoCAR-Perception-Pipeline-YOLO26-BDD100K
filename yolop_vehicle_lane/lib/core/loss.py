"""
Multi-head loss for Vehicle + Lane detection.
Adapted from YOLOP loss with drivable-area segmentation removed.

Loss components:
  - Detection: BCEcls + BCEobj + CIoU box regression
  - Lane: paper-aligned focal/CE + optional Dice (+ optional IoU legacy term)

Lambda order: [cls, obj, iou, ll_seg, ll_iou] - 5 elements
"""

import torch
import torch.nn as nn
from .general import bbox_iou
from .postprocess import build_targets


class MultiHeadLoss(nn.Module):
    def __init__(self, losses, cfg, lambdas=None):
        """
        Args:
            losses: [BCEcls, BCEobj, BCEseg]
            cfg: config object
            lambdas: [cls, obj, iou, ll_seg, ll_iou] weights
        """
        super().__init__()
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 2)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """
        Args:
            predictions: [det_heads, lane_line_seg_head]  (2 elements, no DA)
            targets: [det_targets, lane_targets]  (2 elements, no DA)
            model: model instance (for nc, gr, etc.)
        Returns:
            total_loss, (lbox, lobj, lcls, lseg_ll, liou_ll, loss)
        """
        cfg = self.cfg
        device = targets[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)

        # Label smoothing
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, LaneSegLoss = self.losses

        # Detection losses
        nt = 0
        no = len(predictions[0])
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]

        for i, pi in enumerate(predictions[0]):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                nt += n
                ps = pi[b, a, gj, gi]

                # Box regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification
                if model.nc > 1:
                    t = torch.full_like(ps[:, 5:], cn, device=device)
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]

        # ── Lane seg loss ────────────────────────────────────────────
        lane_logits = predictions[1]
        lane_targets = targets[1].to(lane_logits.dtype)

        # Paper-aligned 2-class lane segmentation: background + lane.
        # Keep the 1-channel route only as a backward-compatibility fallback.
        if lane_logits.shape[1] == 2:
            lseg_ll = LaneSegLoss(lane_logits, lane_targets)
            lane_prob = torch.softmax(lane_logits, dim=1)
            lane_fg_prob = lane_prob[:, 1:2]
            lane_fg_tgt = lane_targets[:, 1:2]
        else:
            lane_fg_tgt = lane_targets[:, 1:2]
            lane_logits_for_loss = lane_logits if lane_logits.shape[1] == 1 else lane_logits[:, 1:2]
            lseg_ll = LaneSegLoss(lane_logits_for_loss, lane_fg_tgt)
            lane_fg_prob = torch.sigmoid(lane_logits_for_loss)

        ldice_ll = torch.zeros(1, device=device)
        dice_gain = float(getattr(cfg.LOSS, 'LL_DICE_GAIN', 0.0) or 0.0)
        if dice_gain > 0:
            if lane_logits.shape[1] == 2:
                inter = (lane_prob * lane_targets).sum(dim=(2, 3))
                denom = lane_prob.sum(dim=(2, 3)) + lane_targets.sum(dim=(2, 3))
                dice = (2.0 * inter + 1.0) / (denom + 1.0)
                ldice_ll = (1.0 - dice.mean(dim=1)).mean().unsqueeze(0)
            else:
                inter = (lane_fg_prob * lane_fg_tgt).sum(dim=(1, 2, 3))
                denom = lane_fg_prob.sum(dim=(1, 2, 3)) + lane_fg_tgt.sum(dim=(1, 2, 3))
                dice = (2.0 * inter + 1.0) / (denom + 1.0)
                ldice_ll = (1.0 - dice).mean().unsqueeze(0)

        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        fg_prob = lane_fg_prob[:, 0]
        fg_tgt = lane_fg_tgt[:, 0]
        if pad_h > 0 or pad_w > 0:
            fg_prob = fg_prob[:, pad_h:height - pad_h, pad_w:width - pad_w]
            fg_tgt = fg_tgt[:, pad_h:height - pad_h, pad_w:width - pad_w]
        inter = (fg_prob * fg_tgt).sum(dim=(1, 2))
        union = (fg_prob + fg_tgt - fg_prob * fg_tgt).sum(dim=(1, 2))
        soft_iou = (inter + 1.0) / (union + 1.0)
        liou_ll = (1.0 - soft_iou).mean().unsqueeze(0)

        # Scale detection losses
        s = 3 / no
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        # Scale lane losses
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[3]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[4]
        if dice_gain > 0:
            ldice_ll = ldice_ll * dice_gain

        # Task-specific training modes
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY:
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            ldice_ll = 0 * ldice_ll

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox

        loss = lbox + lobj + lcls + lseg_ll + liou_ll + ldice_ll
        return loss, (lbox.item(), lobj.item(), lcls.item(),
                      lseg_ll.item(), liou_ll.item(), loss.item())


def get_loss(cfg, device):
    """Build the stage-1 multi-task loss selected by MODEL.NAME."""
    model_name = str(getattr(cfg.MODEL, 'NAME', '')).lower()
    if model_name == 'yolopx':
        return YOLOPXMultiHeadLoss(cfg, device)

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)

    gamma = float(getattr(cfg.LOSS, 'FL_GAMMA', 0.0) or 0.0)
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    ll_gamma = float(getattr(cfg.LOSS, 'LL_FL_GAMMA', 0.0) or 0.0)
    LaneSeg = LaneSegCriterion(pos_weight=float(cfg.LOSS.SEG_POS_WEIGHT), gamma=ll_gamma).to(device)

    loss_list = [BCEcls, BCEobj, LaneSeg]
    return MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)


class LaneSegCriterion(nn.Module):
    """Lane segmentation loss router.

    - 2-channel logits + 2-channel one-hot targets  -> softmax focal / CE
    - 1-channel logits + fg target                  -> BCE / focal-BCE

    This lets the repaired training code stay backward compatible while
    making the default YOLOPv2 path faithful to the paper's C=2 lane-loss
    formula.
    """

    def __init__(self, pos_weight=1.0, gamma=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.binary = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        if self.gamma > 0:
            self.binary = FocalLoss(self.binary, self.gamma)

    def forward(self, pred, true):
        if pred.shape[1] == 2 and true.shape[1] == 2:
            return self._softmax_focal(pred, true)
        return self.binary(pred, true)

    def _softmax_focal(self, logits, targets):
        import torch.nn.functional as F
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        if self.gamma > 0:
            loss = -targets * ((1.0 - prob).clamp(min=1e-6) ** self.gamma) * log_prob
        else:
            loss = -targets * log_prob
        loss = loss.sum(dim=1)
        return loss.mean()


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def meshgrid_ij(y, x):
    try:
        return torch.meshgrid(y, x, indexing='ij')
    except TypeError:
        return torch.meshgrid(y, x)


def bboxes_iou_xywh(boxes_a, boxes_b):
    if boxes_a.shape[1] != 4 or boxes_b.shape[1] != 4:
        raise ValueError('boxes must have shape [N, 4] and [M, 4]')
    a_xy1 = boxes_a[:, None, :2] - boxes_a[:, None, 2:] / 2
    a_xy2 = boxes_a[:, None, :2] + boxes_a[:, None, 2:] / 2
    b_xy1 = boxes_b[:, :2] - boxes_b[:, 2:] / 2
    b_xy2 = boxes_b[:, :2] + boxes_b[:, 2:] / 2
    tl = torch.max(a_xy1, b_xy1)
    br = torch.min(a_xy2, b_xy2)
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (boxes_a[:, 2] * boxes_a[:, 3])[:, None]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    return inter / (area_a + area_b - inter + 1e-8)


class YOLOXIOULoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        pred_xy1 = pred[:, :2] - pred[:, 2:] / 2
        pred_xy2 = pred[:, :2] + pred[:, 2:] / 2
        tgt_xy1 = target[:, :2] - target[:, 2:] / 2
        tgt_xy2 = target[:, :2] + target[:, 2:] / 2
        tl = torch.max(pred_xy1, tgt_xy1)
        br = torch.min(pred_xy2, tgt_xy2)
        wh = (br - tl).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area_p = pred[:, 2] * pred[:, 3]
        area_t = target[:, 2] * target[:, 3]
        iou = inter / (area_p + area_t - inter + 1e-8)
        loss = 1.0 - iou.pow(2)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss


class YOLOXDetectionLoss(nn.Module):
    """YOLOPX/YOLOX dynamic-assignment detection loss.

    This branch intentionally follows the public YOLOPX implementation for
    single-class detection: dynamic-k assignment, IoU-squared regression loss,
    focal objectness loss with gamma=2.0, and total detection loss
    `5 * loss_iou + loss_obj`. Classification loss is computed only for
    reporting when nc > 1, but it is not part of the YOLOPX single-class loss.
    """

    def __init__(self, cfg, device, strides=(8, 16, 32)):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = int(cfg.MODEL.NC)
        self.strides = list(strides)
        self.grids = [torch.zeros(1, device=device)] * len(self.strides)
        bce_obj = nn.BCEWithLogitsLoss(reduction='none').to(device)
        obj_gamma = float(getattr(cfg.LOSS, 'YOLOPX_OBJ_FL_GAMMA', 2.0) or 2.0)
        self.obj_loss_fn = FocalLoss(bce_obj, obj_gamma)
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.iou_loss = YOLOXIOULoss(reduction='none').to(device)
        self.reg_weight = float(getattr(cfg.LOSS, 'YOLOPX_REG_WEIGHT', 5.0) or 5.0)

    def forward(self, preds, labels):
        decoded = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for level, (stride, pred) in enumerate(zip(self.strides, preds)):
            out, grid = self._decode_train_output(pred, level, stride)
            decoded.append(out)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.full((1, grid.shape[1]), stride, device=pred.device, dtype=pred.dtype))

        outputs = torch.cat(decoded, dim=1)
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]
        total_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, dim=1)
        y_shifts = torch.cat(y_shifts, dim=1)
        expanded_strides = torch.cat(expanded_strides, dim=1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0

        nlabels = (labels.sum(dim=2) > 0).sum(dim=1)
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabels[batch_idx])
            if num_gt == 0:
                cls_targets.append(outputs.new_zeros((0, self.num_classes)))
                reg_targets.append(outputs.new_zeros((0, 4)))
                obj_targets.append(outputs.new_zeros((total_anchors, 1)))
                fg_masks.append(outputs.new_zeros(total_anchors).bool())
                continue

            gt_bboxes = labels[batch_idx, :num_gt, 1:5]
            gt_classes = labels[batch_idx, :num_gt, 0]
            matched_classes, fg_mask, matched_ious, matched_gt_inds, num_fg_img = self._get_assignments(
                gt_bboxes,
                gt_classes,
                bbox_preds[batch_idx],
                cls_preds[batch_idx],
                obj_preds[batch_idx],
                expanded_strides,
                x_shifts,
                y_shifts,
                total_anchors,
            )
            num_fg += num_fg_img
            cls_target = torch.nn.functional.one_hot(matched_classes.to(torch.int64), self.num_classes).to(outputs.dtype)
            cls_target = cls_target * matched_ious.unsqueeze(-1)
            obj_target = fg_mask.unsqueeze(-1).to(outputs.dtype)
            reg_target = gt_bboxes[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)
        obj_targets = torch.cat(obj_targets, dim=0)
        fg_masks = torch.cat(fg_masks, dim=0)
        num_fg = max(float(num_fg), 1.0)

        loss_iou = self.iou_loss(bbox_preds.reshape(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.obj_loss_fn(obj_preds.reshape(-1, 1), obj_targets).sum() / num_fg
        if self.num_classes > 1 and cls_targets.numel() > 0:
            loss_cls = self.cls_loss_fn(cls_preds.reshape(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg
        else:
            loss_cls = outputs.new_zeros(1)

        total = self.reg_weight * loss_iou + loss_obj
        return total, loss_iou, loss_obj, loss_cls

    def _decode_train_output(self, output, level, stride):
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        grid = self.grids[level]
        if grid.shape[2:4] != output.shape[2:4] or grid.device != output.device:
            yv, xv = meshgrid_ij(torch.arange(hsize, device=output.device), torch.arange(wsize, device=output.device))
            grid = torch.stack((xv, yv), dim=2).view(1, 1, hsize, wsize, 2).type_as(output)
            self.grids[level] = grid
        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    @torch.no_grad()
    def _get_assignments(self, gt_bboxes, gt_classes, bbox_preds, cls_preds, obj_preds,
                         expanded_strides, x_shifts, y_shifts, total_anchors):
        fg_mask, is_in_boxes_and_center = self._get_in_boxes_info(
            gt_bboxes,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_anchors,
            gt_bboxes.shape[0],
        )
        bbox_preds_in = bbox_preds[fg_mask]
        cls_preds_in = cls_preds[fg_mask]
        obj_preds_in = obj_preds[fg_mask]
        num_in_boxes = bbox_preds_in.shape[0]
        if num_in_boxes == 0:
            fg_mask = torch.zeros(total_anchors, device=gt_bboxes.device, dtype=torch.bool)
            return gt_classes.new_zeros((0,)), fg_mask, gt_bboxes.new_zeros((0,)), gt_classes.new_zeros((0,), dtype=torch.long), 0

        pair_ious = bboxes_iou_xywh(gt_bboxes, bbox_preds_in)
        gt_cls = torch.nn.functional.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        gt_cls = gt_cls.unsqueeze(1).repeat(1, num_in_boxes, 1)
        pair_iou_loss = -torch.log(pair_ious + 1e-8)

        device_type = 'cuda' if gt_bboxes.is_cuda else 'cpu'
        with torch.amp.autocast(device_type, enabled=False):
            cls_prob = cls_preds_in.float().unsqueeze(0).repeat(gt_bboxes.shape[0], 1, 1).sigmoid()
            obj_prob = obj_preds_in.float().unsqueeze(0).repeat(gt_bboxes.shape[0], 1, 1).sigmoid()
            pair_cls_loss = torch.nn.functional.binary_cross_entropy(
                (cls_prob * obj_prob).sqrt_(), gt_cls, reduction='none'
            ).sum(-1)

        cost = pair_cls_loss + 3.0 * pair_iou_loss + 100000.0 * (~is_in_boxes_and_center)
        num_fg, matched_classes, matched_ious, matched_gt_inds = self._dynamic_k_matching(
            cost, pair_ious, gt_classes, fg_mask
        )
        return matched_classes, fg_mask, matched_ious, matched_gt_inds, num_fg

    def _get_in_boxes_info(self, gt_bboxes, expanded_strides, x_shifts, y_shifts, total_anchors, num_gt):
        strides = expanded_strides[0]
        x_centers = (x_shifts[0] + 0.5) * strides
        y_centers = (y_shifts[0] + 0.5) * strides
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        left = gt_bboxes[:, 0].unsqueeze(1) - gt_bboxes[:, 2].unsqueeze(1) / 2
        right = gt_bboxes[:, 0].unsqueeze(1) + gt_bboxes[:, 2].unsqueeze(1) / 2
        top = gt_bboxes[:, 1].unsqueeze(1) - gt_bboxes[:, 3].unsqueeze(1) / 2
        bottom = gt_bboxes[:, 1].unsqueeze(1) + gt_bboxes[:, 3].unsqueeze(1) / 2

        bbox_deltas = torch.stack([x_centers - left, y_centers - top, right - x_centers, bottom - y_centers], dim=2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_radius = 2.5
        c_left = gt_bboxes[:, 0].unsqueeze(1) - center_radius * strides.unsqueeze(0)
        c_right = gt_bboxes[:, 0].unsqueeze(1) + center_radius * strides.unsqueeze(0)
        c_top = gt_bboxes[:, 1].unsqueeze(1) - center_radius * strides.unsqueeze(0)
        c_bottom = gt_bboxes[:, 1].unsqueeze(1) + center_radius * strides.unsqueeze(0)
        center_deltas = torch.stack([x_centers - c_left, y_centers - c_top, c_right - x_centers, c_bottom - y_centers], dim=2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_anchor] & is_in_centers[:, is_in_anchor]
        return is_in_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(self, cost, pair_ious, gt_classes, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        n_candidate = min(10, pair_ious.size(1))
        topk_ious, _ = torch.topk(pair_ious, n_candidate, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1).tolist()
        for gt_idx, dynamic_k in enumerate(dynamic_ks):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_k, largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] = 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = int(fg_mask_inboxes.sum().item())
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        matched_classes = gt_classes[matched_gt_inds]
        matched_ious = (matching_matrix * pair_ious).sum(0)[fg_mask_inboxes]
        return num_fg, matched_classes, matched_ious, matched_gt_inds


class FocalLossSeg(nn.Module):
    """Segmentation focal wrapper used by the public YOLOPX lane branch."""

    def __init__(self, loss_fcn, gamma=2.0, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        p_t = torch.exp(-loss)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.sigmoid()
        batch_size = y_true.size(0)
        channels = y_true.size(1)
        dims = (0, 2)
        y_pred = y_pred.view(batch_size, channels, -1)
        y_true = y_true.view(batch_size, channels, -1)
        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)
        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        raise NotImplementedError


def soft_tversky_score(output, target, alpha, beta, smooth=0.0, eps=1e-7, dims=None):
    if output.size() != target.size():
        raise ValueError('output and target must have the same shape')
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1.0 - output) * target)
    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)


class TverskyLoss(DiceLoss):
    """YOLOPX lane Tversky loss: alpha=0.7, beta=0.3, gamma=4/3."""

    def __init__(self, smooth=0.0, eps=1e-7, alpha=0.7, beta=0.3, gamma=1.0):
        super().__init__(smooth=smooth, eps=eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)


class YOLOPXMultiHeadLoss(nn.Module):
    """Two-task YOLOPX loss adapted from the public three-task recipe.

    Upstream YOLOPX uses:
      detection loss  -> 0.02 * (5 * IoU loss + focal objectness loss)
      lane BCE/focal  -> 0.20 * focal BCE on the lane branch
      lane Tversky    -> 0.20 * Tversky(alpha=0.7, beta=0.3, gamma=4/3)

    The drivable-area branch is intentionally removed for this stage-1
    vehicle + lane experiment. The remaining detection and lane terms keep
    YOLOPX's original scaling.
    """

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.det_loss = YOLOXDetectionLoss(cfg, device)
        lane_bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
        gamma = float(getattr(cfg.LOSS, 'FL_GAMMA', 2.0) or 2.0)
        self.lane_focal = FocalLossSeg(lane_bce, gamma).to(device) if gamma > 0 else lane_bce
        self.lane_tversky = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0 / 3).to(device)
        lambdas = getattr(cfg.LOSS, 'MULTI_HEAD_LAMBDA', None)
        if not lambdas:
            lambdas = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.lambdas = list(lambdas)

    def forward(self, head_fields, head_targets, shapes, model):
        del shapes, model
        det_preds, lane_logits = head_fields
        det_targets, lane_targets = head_targets
        labels = self._labels_to_yolox(det_targets, lane_targets.shape[0], lane_targets.shape[2], lane_targets.shape[3])

        det_raw, loss_iou, loss_obj, loss_cls = self.det_loss(det_preds, labels)
        lane_targets = lane_targets.to(lane_logits.dtype)

        lane_focal = self.lane_focal(lane_logits.reshape(-1), lane_targets.reshape(-1))
        lane_tversky = self.lane_tversky(lane_logits, lane_targets)

        det_scaled = det_raw * 0.02 * self._lambda(1)
        lane_focal_scaled = lane_focal * 0.2 * self._lambda(3)
        lane_tversky_scaled = lane_tversky * 0.2 * self._lambda(4)

        if self.cfg.TRAIN.DET_ONLY or self.cfg.TRAIN.ENC_DET_ONLY:
            lane_focal_scaled = 0 * lane_focal_scaled
            lane_tversky_scaled = 0 * lane_tversky_scaled
        if self.cfg.TRAIN.LANE_ONLY:
            det_scaled = 0 * det_scaled
            loss_iou = 0 * loss_iou
            loss_obj = 0 * loss_obj
            loss_cls = 0 * loss_cls

        total = det_scaled + lane_focal_scaled + lane_tversky_scaled
        return total, (
            det_scaled.item(),
            loss_obj.item() if torch.is_tensor(loss_obj) else float(loss_obj),
            loss_cls.item() if torch.is_tensor(loss_cls) else float(loss_cls),
            lane_focal_scaled.item(),
            lane_tversky_scaled.item(),
            total.item(),
        )

    def _lambda(self, index):
        return float(self.lambdas[index]) if index < len(self.lambdas) else 1.0

    @staticmethod
    def _labels_to_yolox(labels, batch_size, height, width):
        counts = []
        for batch_idx in range(batch_size):
            counts.append(int((labels[:, 0] == batch_idx).sum().item()))
        max_gt = max(counts) if counts else 0
        max_gt = max(max_gt, 1)
        out = labels.new_zeros((batch_size, max_gt, 5))
        for batch_idx in range(batch_size):
            rows = labels[labels[:, 0] == batch_idx]
            if rows.numel() == 0:
                continue
            n = rows.shape[0]
            out[batch_idx, :n, 0] = rows[:, 1]
            out[batch_idx, :n, 1] = rows[:, 2] * width
            out[batch_idx, :n, 2] = rows[:, 3] * height
            out[batch_idx, :n, 3] = rows[:, 4] * width
            out[batch_idx, :n, 4] = rows[:, 5] * height
        return out

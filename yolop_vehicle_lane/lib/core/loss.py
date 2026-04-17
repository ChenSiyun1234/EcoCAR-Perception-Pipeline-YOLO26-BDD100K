"""
Multi-head loss for Vehicle + Lane detection.
Adapted from YOLOP loss with drivable-area segmentation removed.

Loss components:
  - Detection: BCEcls + BCEobj + CIoU box regression
  - Lane: BCE segmentation + IoU

Lambda order: [cls, obj, iou, ll_seg, ll_iou] - 5 elements
"""

import torch
import torch.nn as nn
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric


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

        BCEcls, BCEobj, BCEseg = self.losses

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
        # predictions[1] is RAW logits (see model.forward docstring —
        # the YOLOP-upstream double-sigmoid bug is fixed). BCEseg here
        # is `BCEWithLogitsLoss` (optionally focal-wrapped if
        # LOSS.LL_FL_GAMMA > 0), which consumes logits correctly.
        # targets[1] is a 2-channel (bg, fg) binary tensor.
        lane_line_seg_predicts = predictions[1].view(-1)
        lane_line_seg_targets = targets[1].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # Hybrid focal + dice variant (YOLOPv2 paper §3 ablation).
        # LOSS.LL_DICE_GAIN > 0 turns it on. Dice operates on the
        # sigmoided foreground-channel probabilities.
        ldice_ll = torch.zeros(1, device=device)
        dice_gain = float(getattr(cfg.LOSS, 'LL_DICE_GAIN', 0.0) or 0.0)
        if dice_gain > 0:
            # predictions[1]: [B, 2, H, W] logits. We take channel 1 (fg)
            # as the lane-presence logit and sigmoid it to [0,1].
            fg_prob = torch.sigmoid(predictions[1][:, 1])              # [B, H, W]
            fg_tgt = targets[1][:, 1].to(fg_prob.dtype)                # [B, H, W]
            inter = (fg_prob * fg_tgt).sum(dim=(1, 2))
            denom = fg_prob.sum(dim=(1, 2)) + fg_tgt.sum(dim=(1, 2))
            dice = (2.0 * inter + 1.0) / (denom + 1.0)
            ldice_ll = (1.0 - dice).mean().unsqueeze(0)

        # Lane line IoU loss
        metric = SegmentationMetric(2)
        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _, lane_line_pred = torch.max(predictions[1], 1)
        _, lane_line_gt = torch.max(targets[1], 1)
        lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

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
    """Build MultiHeadLoss from config.

    YOLOP applies focal loss only to BCEcls / BCEobj when FL_GAMMA > 0.
    YOLOPv2 paper §3 applies focal loss to the lane segmentation head
    as well. We keep YOLOP's behavior by default and add a separate
    `LOSS.LL_FL_GAMMA` knob: when > 0, BCEseg is focal-wrapped. Set it
    in the YOLOPv2 YAML; leave it 0 for the YOLOP baseline.
    """
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)

    gamma = float(getattr(cfg.LOSS, 'FL_GAMMA', 0.0) or 0.0)
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    ll_gamma = float(getattr(cfg.LOSS, 'LL_FL_GAMMA', 0.0) or 0.0)
    if ll_gamma > 0:
        BCEseg = FocalLoss(BCEseg, ll_gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss


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

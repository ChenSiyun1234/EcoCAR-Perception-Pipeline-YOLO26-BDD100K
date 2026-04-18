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
    """Build MultiHeadLoss from config.

    YOLOP applies focal loss only to BCEcls / BCEobj when FL_GAMMA > 0.
    YOLOPv2 paper §3 applies focal loss to the lane segmentation head
    as well. We keep YOLOP's behavior by default and add a separate
    `LOSS.LL_FL_GAMMA` knob: when > 0, BCEseg is focal-wrapped. Set it
    in the YOLOPv2 YAML; leave it 0 for the YOLOP baseline.
    """
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)

    gamma = float(getattr(cfg.LOSS, 'FL_GAMMA', 0.0) or 0.0)
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    ll_gamma = float(getattr(cfg.LOSS, 'LL_FL_GAMMA', 0.0) or 0.0)
    LaneSeg = LaneSegCriterion(pos_weight=float(cfg.LOSS.SEG_POS_WEIGHT), gamma=ll_gamma).to(device)

    loss_list = [BCEcls, BCEobj, LaneSeg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss


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

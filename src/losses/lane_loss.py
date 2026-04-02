"""
Lane segmentation losses: BCE+Dice, Focal+Dice, soft-target variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BCEDiceLoss(nn.Module):
    """BCE + Dice loss with configurable pos_weight and label smoothing."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: float = 10.0, label_smoothing: float = 0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        dice = self._dice_loss(logits, targets)
        total = self.bce_weight * bce + self.dice_weight * dice
        return total, {"bce": bce.item(), "dice": dice.item()}

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()
        smooth = 1e-6
        intersection = (probs * targets_f).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_f.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()


class FocalDiceLoss(nn.Module):
    """Focal loss (for sparse positives) + Dice loss."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 dice_weight: float = 1.0, focal_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        focal = self._focal_loss(logits, targets)
        dice = BCEDiceLoss._dice_loss(logits, targets)
        total = self.focal_weight * focal + self.dice_weight * dice
        return total, {"focal": focal.item(), "dice": dice.item()}

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits.float(), targets_f, reduction='none')

        # p_t = p for positive, 1-p for negative
        p_t = probs * targets_f + (1 - probs) * (1 - targets_f)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


def build_lane_loss(cfg: dict) -> nn.Module:
    """Factory for lane loss from config dict."""
    loss_type = cfg.get("type", "bce_dice")

    if loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=cfg.get("bce_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            pos_weight=cfg.get("pos_weight", 10.0),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    elif loss_type in ("focal_dice", "focal"):
        return FocalDiceLoss(
            alpha=cfg.get("focal_alpha", 0.75),
            gamma=cfg.get("focal_gamma", 2.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            focal_weight=cfg.get("bce_weight", 1.0),
        )
    elif loss_type == "soft_bce_dice":
        return BCEDiceLoss(
            bce_weight=cfg.get("bce_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            pos_weight=cfg.get("pos_weight", 5.0),
            label_smoothing=cfg.get("label_smoothing", 0.05),
        )
    else:
        raise ValueError(f"Unknown lane loss type: {loss_type}")

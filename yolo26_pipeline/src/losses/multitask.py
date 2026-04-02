"""
Multi-task loss weighting strategies.

- FixedWeighting: static manual weights
- UncertaintyWeighting: Kendall et al. homoscedastic uncertainty
- PCGrad: Yu et al. gradient surgery for conflicting gradients
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import copy
import random


class FixedWeighting(nn.Module):
    """Weighted sum with fixed coefficients."""

    def __init__(self, det_weight: float = 1.0, lane_weight: float = 0.5):
        super().__init__()
        self.det_weight = det_weight
        self.lane_weight = lane_weight

    def forward(self, det_loss: torch.Tensor, lane_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        total = self.det_weight * det_loss + self.lane_weight * lane_loss
        return total, {
            "det_weight": self.det_weight,
            "lane_weight": self.lane_weight,
        }


class UncertaintyWeighting(nn.Module):
    """
    Homoscedastic uncertainty weighting (Kendall et al., 2018).

    loss = 1/(2*sigma_det^2) * L_det + 1/(2*sigma_lane^2) * L_lane + log(sigma_det) + log(sigma_lane)

    The log-variance parameters are learned during training, automatically
    balancing the contribution of each task.
    """

    def __init__(self):
        super().__init__()
        # log(sigma^2) parameters — initialized to 0 (sigma=1, weight=0.5)
        self.log_var_det = nn.Parameter(torch.zeros(1))
        self.log_var_lane = nn.Parameter(torch.zeros(1))

    def forward(self, det_loss: torch.Tensor, lane_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        precision_det = torch.exp(-self.log_var_det)
        precision_lane = torch.exp(-self.log_var_lane)

        total = (precision_det * det_loss + self.log_var_det +
                 precision_lane * lane_loss + self.log_var_lane)

        return total, {
            "det_weight": precision_det.item(),
            "lane_weight": precision_lane.item(),
            "log_var_det": self.log_var_det.item(),
            "log_var_lane": self.log_var_lane.item(),
        }



class StagedWeighting(nn.Module):
    """Detection-first schedule with optional hold period before ramping lane loss."""

    def __init__(self, det_weight: float = 1.0, lane_weight_start: float = 0.0,
                 lane_weight_end: float = 0.2, ramp_epochs: int = 8,
                 hold_epochs: int = 2):
        super().__init__()
        self.det_weight = det_weight
        self.lane_weight_start = lane_weight_start
        self.lane_weight_end = lane_weight_end
        self.ramp_epochs = ramp_epochs
        self.hold_epochs = hold_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def current_lane_weight(self) -> float:
        if self._current_epoch < self.hold_epochs:
            return self.lane_weight_start
        progress = min(
            1.0,
            (self._current_epoch - self.hold_epochs) / max(1, self.ramp_epochs)
        )
        return self.lane_weight_start + (self.lane_weight_end - self.lane_weight_start) * progress

    def forward(self, det_loss: torch.Tensor, lane_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        lane_weight = self.current_lane_weight()
        if not torch.isfinite(det_loss):
            total = lane_weight * lane_loss
        elif not torch.isfinite(lane_loss):
            total = self.det_weight * det_loss
        else:
            total = self.det_weight * det_loss + lane_weight * lane_loss
        return total, {
            "det_weight": self.det_weight,
            "lane_weight": lane_weight,
            "hold_epochs": self.hold_epochs,
            "ramp_epochs": self.ramp_epochs,
        }


class PCGrad:
    """
    PCGrad (Yu et al., 2020): Project Conflicting Gradients.

    When gradients from two tasks conflict (negative cosine similarity),
    project one onto the normal plane of the other to remove the conflicting
    component. This prevents one task from degrading the other.

    Usage: call pcgrad_backward() instead of loss.backward().
    """

    @staticmethod
    def pcgrad_backward(
        losses: List[torch.Tensor],
        shared_params: List[nn.Parameter],
        scaler=None,
    ) -> None:
        """
        Compute PCGrad-corrected gradients and accumulate onto shared_params.

        Args:
            losses: list of per-task scalar losses
            shared_params: parameters that receive conflicting gradients
            scaler: GradScaler for AMP (optional)
        """
        grads = []
        for loss in losses:
            # Zero existing grads
            for p in shared_params:
                if p.grad is not None:
                    p.grad.zero_()

            if scaler is not None:
                scaler.scale(loss).backward(retain_graph=True)
                # Manually unscale: divide grads by the current scale factor
                inv_scale = 1.0 / scaler.get_scale()
                for p in shared_params:
                    if p.grad is not None:
                        p.grad.mul_(inv_scale)
            else:
                loss.backward(retain_graph=True)

            grad_vec = torch.cat([
                p.grad.detach().flatten() if p.grad is not None
                else torch.zeros(p.numel(), device=p.device)
                for p in shared_params
            ])
            grads.append(grad_vec)

        # Project conflicting gradients
        corrected = []
        order = list(range(len(grads)))
        random.shuffle(order)

        for i in order:
            g_i = grads[i].clone()
            for j in order:
                if i == j:
                    continue
                g_j = grads[j]
                dot = (g_i * g_j).sum()
                if dot < 0:
                    # Project g_i onto normal plane of g_j
                    g_i = g_i - (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
            corrected.append(g_i)

        # Average corrected gradients
        merged = torch.stack(corrected).mean(dim=0)

        # Assign back to parameters
        offset = 0
        for p in shared_params:
            numel = p.numel()
            if p.grad is None:
                p.grad = merged[offset:offset + numel].reshape(p.shape).clone()
            else:
                p.grad.copy_(merged[offset:offset + numel].reshape(p.shape))
            offset += numel


def build_multitask_strategy(cfg: dict) -> nn.Module:
    """Factory for multi-task weighting strategy."""
    strategy = cfg.get("strategy", "fixed")

    if strategy == "fixed":
        return FixedWeighting(
            det_weight=cfg.get("det_weight", 1.0),
            lane_weight=cfg.get("lane_weight", 0.5),
        )
    elif strategy == "uncertainty":
        return UncertaintyWeighting()
    elif strategy == "pcgrad":
        # PCGrad is not an nn.Module — return a FixedWeighting for the non-PCGrad path
        # and handle PCGrad in the trainer
        return FixedWeighting(
            det_weight=cfg.get("det_weight", 1.0),
            lane_weight=cfg.get("lane_weight", 0.5),
        )
    elif strategy == "staged":
        return StagedWeighting(
            det_weight=cfg.get("det_weight", 1.0),
            lane_weight_start=cfg.get("lane_weight_start", 0.0),
            lane_weight_end=cfg.get("lane_weight_end", 0.2),
            ramp_epochs=cfg.get("ramp_epochs", 8),
            hold_epochs=cfg.get("hold_epochs", 2),
        )
    else:
        raise ValueError(f"Unknown multitask strategy: {strategy}")

"""
Exponential Moving Average model for stable evaluation and checkpointing.
"""

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        # after each optimizer step:
        ema.update(model)
        # for evaluation:
        ema.apply(model)   # copy EMA weights into model
        ema.restore(model) # restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Copy EMA weights into model (save originals for restore)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original (non-EMA) weights."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()}, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        for k, v in state_dict["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

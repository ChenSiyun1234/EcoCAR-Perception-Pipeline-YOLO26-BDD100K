"""
feature_hooks.py — Hook-based feature extraction from YOLO26 backbone/neck.

Used by:
  - 05_extract_backbone_features.ipynb
  - Future lane-marking integration work
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class FeatureExtractor:
    """
    Register forward hooks on selected layers of a YOLO model to capture
    intermediate feature maps during inference.

    Usage:
        from ultralytics import YOLO
        model = YOLO("best.pt")

        extractor = FeatureExtractor(model.model)
        extractor.register_hooks(layer_indices=[2, 4, 6, 8, 10])

        results = model("image.jpg")
        features = extractor.get_features()

        for name, feat in features.items():
            print(f"{name}: {feat.shape}")

        extractor.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: The inner PyTorch model (e.g. yolo_model.model for
                   an Ultralytics YOLO object).
        """
        self.model = model
        self._features: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ── Hook registration ────────────────────────────────────────────────

    def _make_hook(self, name: str):
        """Create a forward-hook closure that stores the output tensor."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._features[name] = output.detach()
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                # Some modules return lists; grab the first tensor
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self._features[name] = item.detach()
                        break
        return hook_fn

    def register_hooks(
        self,
        layer_indices: Optional[List[int]] = None,
        layer_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Register forward hooks on selected layers.

        Args:
            layer_indices: List of integer indices into model.model (the
                           Sequential backbone/neck/head).  Typical backbone
                           layers for YOLO26 are [0..9], neck [10..22],
                           head [23+].
            layer_names:   Alternative: list of submodule names
                           (dot-separated, e.g. 'model.2').

        Returns:
            List of hook names that were successfully registered.
        """
        self.remove_hooks()
        self._features.clear()
        registered = []

        if layer_indices is not None:
            # Ultralytics models store the architecture as self.model.model
            # which is a nn.Sequential-like list.
            try:
                layers = list(self.model.model)
            except TypeError:
                layers = list(self.model.children())

            for idx in layer_indices:
                if idx < 0 or idx >= len(layers):
                    print(f"⚠ Layer index {idx} out of range (0..{len(layers)-1})")
                    continue
                name = f"layer_{idx}_{layers[idx].__class__.__name__}"
                h = layers[idx].register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
                registered.append(name)

        if layer_names is not None:
            for lname in layer_names:
                try:
                    module = dict(self.model.named_modules())[lname]
                    name = f"{lname}_{module.__class__.__name__}"
                    h = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(h)
                    registered.append(name)
                except KeyError:
                    print(f"⚠ Module '{lname}' not found")

        if registered:
            print(f"✅ Registered {len(registered)} hooks: {registered}")
        return registered

    # ── Retrieve captured features ───────────────────────────────────────

    def get_features(self) -> OrderedDict[str, torch.Tensor]:
        """Return captured feature tensors from the last forward pass."""
        return self._features

    def get_feature_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return {name: shape} for all captured features."""
        return {k: tuple(v.shape) for k, v in self._features.items()}

    def clear_features(self) -> None:
        """Clear stored features (free memory)."""
        self._features.clear()

    # ── Cleanup ──────────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


# ── Convenience functions ────────────────────────────────────────────────────

def get_backbone_layer_indices() -> List[int]:
    """
    Return typical backbone layer indices for YOLO26.
    These correspond to the early convolutional / C3k2 blocks.
    Adjust after inspecting your specific model with 06_model_inspection.
    """
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def get_neck_layer_indices() -> List[int]:
    """
    Return typical neck (FPN/PAN) layer indices for YOLO26.
    Adjust after inspecting your specific model with 06_model_inspection.
    """
    return [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def extract_features(
    model,
    image_path: str,
    layer_indices: Optional[List[int]] = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    One-shot helper: load image, run inference, return backbone/neck features.

    Args:
        model:          An Ultralytics YOLO model object.
        image_path:     Path to input image.
        layer_indices:  Layer indices to hook (defaults to backbone).

    Returns:
        OrderedDict of {layer_name: feature_tensor}.
    """
    if layer_indices is None:
        layer_indices = get_backbone_layer_indices()

    extractor = FeatureExtractor(model.model)
    extractor.register_hooks(layer_indices=layer_indices)

    # Run inference (this triggers the forward hooks)
    _ = model(image_path, verbose=False)

    features = extractor.get_features()
    extractor.remove_hooks()
    return features

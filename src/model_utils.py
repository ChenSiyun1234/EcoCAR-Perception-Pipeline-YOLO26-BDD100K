"""
model_utils.py — YOLO26 model loading, info, and component identification.

Used by:
  - 00_quick_pretrained_yolo_baseline.ipynb
  - 03_train_yolo26_on_bdd.ipynb
  - 04_inference_yolo26.ipynb
  - 05_extract_backbone_features.ipynb
  - 06_model_inspection.ipynb
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch


def load_model(
    weights: str = "yolo26n.pt",
    task: str = "detect",
    verbose: bool = True,
):
    """
    Load a YOLO26 model via Ultralytics.

    Args:
        weights: Path to .pt weights or model name (e.g. 'yolo26n.pt').
        task:    Task type — 'detect', 'segment', etc.
        verbose: Print model info after loading.

    Returns:
        Ultralytics YOLO model object.
    """
    from ultralytics import YOLO

    model = YOLO(weights, task=task)
    if verbose:
        print(f"✅ Loaded model: {weights}")
        print(f"   Task: {task}")
        if hasattr(model, "names"):
            print(f"   Classes: {len(model.names)}")
    return model


def get_model_info(model) -> Dict[str, Any]:
    """
    Extract summary information about a loaded YOLO model.

    Returns dict with: num_params, num_layers, class_names, task, etc.
    """
    inner = model.model if hasattr(model, "model") else model

    # Count parameters
    total_params = sum(p.numel() for p in inner.parameters())
    trainable_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)

    # Count layers / modules
    num_modules = sum(1 for _ in inner.modules())

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "num_modules": num_modules,
        "class_names": model.names if hasattr(model, "names") else {},
        "num_classes": len(model.names) if hasattr(model, "names") else 0,
        "task": model.task if hasattr(model, "task") else "unknown",
    }
    return info


def print_model_info(model) -> None:
    """Pretty-print model summary."""
    info = get_model_info(model)
    print(f"\n{'='*50}")
    print(f" YOLO Model Summary")
    print(f"{'='*50}")
    print(f" Task:             {info['task']}")
    print(f" Classes:          {info['num_classes']}")
    print(f" Total params:     {info['total_params']:,}")
    print(f" Trainable params: {info['trainable_params']:,}")
    print(f" Modules:          {info['num_modules']}")
    print(f"{'='*50}\n")


def identify_model_components(model) -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Walk the YOLO model Sequential structure and classify each layer
    as belonging to backbone, neck, or head.

    Returns:
        Dict with keys 'backbone', 'neck', 'head', each containing a list
        of (index, module_class_name, short_description).

    Note: The exact boundary indices depend on the YOLO26 variant.
    This function uses heuristics based on known YOLO architecture patterns.
    Review the output against 06_model_inspection.ipynb for your model.
    """
    inner = model.model if hasattr(model, "model") else model

    try:
        layers = list(inner.model)
    except (TypeError, AttributeError):
        layers = list(inner.children())

    components = {
        "backbone": [],
        "neck": [],
        "head": [],
    }

    detect_keywords = {"Detect", "Segment", "Classify", "OBB", "Pose", "WorldDetect", "E2EDetect"}
    upsample_keywords = {"Upsample", "Concat", "BiFPN"}

    found_upsample = False
    found_head = False

    for idx, layer in enumerate(layers):
        cls_name = layer.__class__.__name__
        desc = f"[{idx}] {cls_name}"

        # Try to get more info
        if hasattr(layer, "cv1"):
            desc += " (conv block)"
        elif hasattr(layer, "m"):
            desc += " (bottleneck)"

        # Classification heuristic
        if cls_name in detect_keywords or "Detect" in cls_name:
            components["head"].append((idx, cls_name, desc))
            found_head = True
        elif found_upsample or cls_name in upsample_keywords or "Concat" in cls_name:
            if not found_head:
                components["neck"].append((idx, cls_name, desc))
                found_upsample = True
        else:
            if not found_upsample and not found_head:
                components["backbone"].append((idx, cls_name, desc))
            elif not found_head:
                components["neck"].append((idx, cls_name, desc))

    return components


def print_model_architecture(model) -> None:
    """Print the full sequential architecture with layer indices."""
    inner = model.model if hasattr(model, "model") else model

    try:
        layers = list(inner.model)
    except (TypeError, AttributeError):
        layers = list(inner.children())

    print(f"\n{'='*60}")
    print(f" Model Architecture — {len(layers)} top-level layers")
    print(f"{'='*60}")

    for idx, layer in enumerate(layers):
        cls_name = layer.__class__.__name__
        params = sum(p.numel() for p in layer.parameters())
        print(f"  [{idx:>2}] {cls_name:<30} params={params:>10,}")

    print(f"{'='*60}\n")


def check_gpu() -> Dict[str, Any]:
    """Check GPU availability and print info. Useful for Colab."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": "",
        "memory_gb": 0.0,
    }

    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem
        info["memory_gb"] = mem / (1024 ** 3)

        print(f"✅ GPU available: {info['device_name']}")
        print(f"   Memory: {info['memory_gb']:.1f} GB")
    else:
        print("⚠ No GPU detected! Training will be very slow.")
        print("  → In Colab: Runtime → Change runtime type → GPU")

    return info

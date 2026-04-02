"""
Vehicle-only class definitions for EcoCAR perception.

The detect head uses nc=5 (native vehicle classes). No COCO remapping needed.
The head is initialized from COCO-pretrained weights by copying the relevant
channel weights for each vehicle class.

Vehicle classes:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle

Dropped classes (not vehicle-relevant for EcoCAR):
  - person, rider, traffic light, traffic sign (non-vehicle)
  - train (too rare: ~50 annotations in 70K images, unstable metrics)
"""

import torch
from typing import Dict, List, Tuple

# ── Vehicle class definitions ─────────────────────────────────────────────
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
NUM_VEHICLE_CLASSES = len(VEHICLE_CLASSES)

VEHICLE_CLASS_TO_ID = {name: idx for idx, name in enumerate(VEHICLE_CLASSES)}
VEHICLE_ID_TO_CLASS = {idx: name for idx, name in enumerate(VEHICLE_CLASSES)}

# ── BDD100K original 10-class mapping ────────────────────────────────────
BDD_CLASSES_FULL = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# BDD100K class ID -> Vehicle class ID (only for vehicle classes)
BDD_TO_VEHICLE = {
    2: 0,   # car -> 0
    3: 1,   # truck -> 1
    4: 2,   # bus -> 2
    6: 3,   # motorcycle -> 3
    7: 4,   # bicycle -> 4
}

# Reverse mapping
VEHICLE_TO_BDD = {v: k for k, v in BDD_TO_VEHICLE.items()}

# COCO class IDs for each vehicle class (for weight initialization)
VEHICLE_TO_COCO = {
    0: 2,   # car -> COCO car
    1: 7,   # truck -> COCO truck
    2: 5,   # bus -> COCO bus
    3: 3,   # motorcycle -> COCO motorcycle
    4: 1,   # bicycle -> COCO bicycle
}


def remap_targets_bdd_to_vehicle(targets: torch.Tensor) -> torch.Tensor:
    """Remap BDD100K class IDs to vehicle class IDs, dropping non-vehicle.

    Args:
        targets: (N, 6) tensor — [batch_idx, class_id, x, y, w, h]
                 class_id is BDD100K (0-9)
    Returns:
        Filtered targets with only vehicle classes, remapped to 0-4.
    """
    if targets.shape[0] == 0:
        return targets

    remapped = targets.clone()
    cls_col = remapped[:, 1].long()

    new_cls = torch.full_like(cls_col, -1)
    for bdd_id, veh_id in BDD_TO_VEHICLE.items():
        mask = cls_col == bdd_id
        new_cls[mask] = veh_id

    valid = new_cls >= 0
    result = remapped[valid]
    result[:, 1] = new_cls[valid].float()
    return result


def remap_preds_to_vehicle_names(pred_boxes: torch.Tensor) -> torch.Tensor:
    """No remapping needed — predictions already use vehicle class IDs 0-4.

    This function exists for API compatibility. It simply passes through
    and filters any out-of-range class IDs (shouldn't happen with nc=5).
    """
    if len(pred_boxes) == 0:
        return pred_boxes
    valid = pred_boxes[:, 5] < NUM_VEHICLE_CLASSES
    return pred_boxes[valid]


def get_vehicle_class_info() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Return (class_list, name->id, id->name) for vehicle detection."""
    return VEHICLE_CLASSES, VEHICLE_CLASS_TO_ID, VEHICLE_ID_TO_CLASS

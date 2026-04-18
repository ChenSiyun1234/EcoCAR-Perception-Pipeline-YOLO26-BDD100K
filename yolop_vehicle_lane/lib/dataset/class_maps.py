"""
Class-taxonomy maps.

Stage1 uses the strict YOLOP-style merged-vehicle protocol. Stage2 can
opt in to an extended 3-class taxonomy that keeps motorcycle / bicycle
as separate classes. Other modules read the right map from config via
`build_id_dict(cfg)`.

Every map is `{bdd_category_name: class_id}`. Class IDs must be dense
(0..N-1) because the detection head channel count is NC.
"""

from typing import Dict, List, Tuple


# Stage 1 — YOLOP-style strict fair-comparison protocol.
# Paper protocol: one "vehicle" class merged from car / bus / truck / train.
# motorcycle, bicycle, and other VRU categories are NOT included.
STAGE1_VEHICLE_MERGED: Dict[str, int] = {
    'car':   0,
    'bus':   0,
    'truck': 0,
    'train': 0,
}
STAGE1_VEHICLE_MERGED_NAMES: List[str] = ['vehicle']

# Stage 2 — 1-class variant (identical to stage1; for direct control-group runs).
STAGE2_1C_VEHICLE_MERGED: Dict[str, int] = dict(STAGE1_VEHICLE_MERGED)
STAGE2_1C_VEHICLE_MERGED_NAMES: List[str] = list(STAGE1_VEHICLE_MERGED_NAMES)

# Stage 2 — 3-class extended taxonomy.
# vehicle (merged) + motorcycle + bicycle. Useful when we want to study
# VRU detection as an added capability. motor/bike are aliases BDD uses.
STAGE2_3C_EXTENDED: Dict[str, int] = {
    'car':        0,
    'bus':        0,
    'truck':      0,
    'train':      0,
    'motorcycle': 1,
    'motor':      1,
    'bicycle':    2,
    'bike':       2,
}
STAGE2_3C_EXTENDED_NAMES: List[str] = ['vehicle', 'motorcycle', 'bicycle']


# Strict BDD100K / YOLOP / YOLOPv2 detection taxonomy (10 classes).
BDD100K_10CLASS: Dict[str, int] = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6, 'motor': 6,
    'bicycle': 7, 'bike': 7,
    'traffic light': 8,
    'traffic sign': 9,
}
BDD100K_10CLASS_NAMES: List[str] = [
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

# Legacy 5-class — kept for backward compat with early runs.
LEGACY_5C_VEHICLE: Dict[str, int] = {
    'car':        0,
    'truck':      1,
    'bus':        2,
    'motorcycle': 3, 'motor': 3,
    'bicycle':    4, 'bike':  4,
}
LEGACY_5C_VEHICLE_NAMES: List[str] = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']


_PROTOCOL_REGISTRY = {
    'stage1_vehicle_merged':    (STAGE1_VEHICLE_MERGED,    STAGE1_VEHICLE_MERGED_NAMES),
    'stage2_1c_vehicle_merged': (STAGE2_1C_VEHICLE_MERGED, STAGE2_1C_VEHICLE_MERGED_NAMES),
    'stage2_3c_extended':       (STAGE2_3C_EXTENDED,       STAGE2_3C_EXTENDED_NAMES),
    'bdd100k_10class':          (BDD100K_10CLASS,          BDD100K_10CLASS_NAMES),
    'legacy_5c_vehicle':        (LEGACY_5C_VEHICLE,        LEGACY_5C_VEHICLE_NAMES),
}


def build_id_dict(cfg) -> Tuple[Dict[str, int], List[str]]:
    """Resolve `(id_dict, class_names)` from cfg.

    `cfg.DATASET.CLASS_PROTOCOL` selects the protocol (string). If the
    key is missing we fall back to the legacy 5-class mapping so old
    runs still work.
    """
    proto = getattr(getattr(cfg, 'DATASET', object()), 'CLASS_PROTOCOL', '') or ''
    proto = str(proto).strip().lower()
    if proto not in _PROTOCOL_REGISTRY:
        # Back-compat: the classic 5-class mapping from the earliest
        # runs. Emit a soft warning so callers notice when nothing is
        # set in the YAML.
        if proto:
            print(f'[class_maps] unknown CLASS_PROTOCOL={proto!r}; '
                  f'falling back to legacy_5c_vehicle')
        return _PROTOCOL_REGISTRY['legacy_5c_vehicle']
    return _PROTOCOL_REGISTRY[proto]


def available_protocols() -> List[str]:
    return sorted(_PROTOCOL_REGISTRY.keys())

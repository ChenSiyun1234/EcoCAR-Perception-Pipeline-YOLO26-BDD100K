"""
Label conversion utilities for vehicle-only detection on BDD100K.
Maps BDD100K categories to a compact 5-class vehicle label space.
"""

# Vehicle-only class mapping (5 classes)
id_dict = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'motorcycle': 3, 'motor': 3,
    'bicycle': 4, 'bike': 4,
}

# Reverse mapping for display
id_to_name = {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'bicycle'}


def convert(size, box):
    """Convert (x1, x2, y1, y2) to normalized (cx, cy, w, h)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

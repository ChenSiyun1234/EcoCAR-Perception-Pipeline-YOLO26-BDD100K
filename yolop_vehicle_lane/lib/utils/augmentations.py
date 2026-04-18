"""
Data augmentation utilities adapted from YOLOP.
Modified to handle 2-tuple (img, lane_mask) instead of 3-tuple (img, da_mask, lane_mask)
since drivable-area segmentation is removed.
"""

import numpy as np
import cv2
import random
import math


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """Change color hue, saturation, value."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_perspective(combination, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """
    Combined image transform: perspective, rotation, scale, shear, translation.
    combination is (img, lane_mask) - 2 elements (no drivable-area mask).
    """
    img, line = combination
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined rotation matrix
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        # REPAIR (v5): images use linear interp (default) but thin lane
        # masks MUST use nearest — otherwise the warp smears the 8-pixel
        # line into uint8 intermediates that don't survive the downstream
        # `threshold(>1)` binarization consistently.
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            line = cv2.warpPerspective(line, M, dsize=(width, height),
                                        flags=cv2.INTER_NEAREST, borderValue=0)
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            line = cv2.warpAffine(line, M[:2], dsize=(width, height),
                                   flags=cv2.INTER_NEAREST, borderValue=0)

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        else:
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = _box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    combination = (img, line)
    return combination, targets


def cutout(combination, labels):
    """Applies image cutout augmentation."""
    image, line = combination
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        box2 = box2.transpose()
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
        return inter_area / box2_area

    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        line[ymin:ymax, xmin:xmax] = 0
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])
            labels = labels[ioa < 0.60]

    return image, line, labels


def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Resize image + lane mask with letterbox padding.
    combination is (img, lane_mask) - 2 elements.
    """
    img, line = combination
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        # REPAIR (v5): lane mask must use NEAREST to preserve the thin
        # binary supervision (see top-of-file note in AutoDriveDataset).
        line = cv2.resize(line, new_unpad, interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    line = cv2.copyMakeBorder(line, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    combination = (img, line)
    return combination, ratio, (dw, dh)


def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize image only with letterbox padding (for inference)."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)


# ─────────────────────────────────────────────────────────────────────
# Mosaic + MixUp for YOLOPv2-style training.
# [INFERRED] YOLOPv2 does not publish training code. Mosaic and MixUp are
# YOLOv7's default recipe (ultralytics style) and YOLOPv2 inherits its
# lineage from YOLOv7 backbone. The parameters below mirror YOLOv7's
# hyp.scratch.yaml defaults.
#
# Both helpers operate on a 2-tuple (img, lane_mask); labels are in the
# pre-normalization xyxy pixel format used by AutoDriveDataset just
# before the `labels[:, 1:5] = xyxy2xywh(...)` line.
# ─────────────────────────────────────────────────────────────────────


def load_mosaic(dataset, index, s=640):
    """Build a 4-image mosaic centered at a random pivot inside a 2s x 2s
    canvas, then sample back to s x s via random_perspective with a
    negative border (standard YOLOv5/YOLOv7 recipe).

    Args:
        dataset: AutoDriveDataset-like object exposing
                 `_load_mosaic_sample(idx)` that returns
                 (img_rgb, lane_mask, labels_xyxy_abs).
        index:   base index for the first tile.
        s:       target mosaic size (before final perspective).
    Returns:
        (img_out, lane_out), labels_out
    """
    labels4 = []
    yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
    indices = [index] + random.choices(range(len(dataset)), k=3)

    img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    lane4 = np.zeros((s * 2, s * 2), dtype=np.uint8)

    for i, idx in enumerate(indices):
        img, lane, labels = dataset._load_mosaic_sample(idx)
        h, w = img.shape[:2]
        r = s / max(h, w)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)
            # REPAIR (v5): lane mask uses NEAREST regardless of the image
            # interp choice — see AutoDriveDataset note.
            lane = cv2.resize(lane, (int(w * r), int(h * r)),
                               interpolation=cv2.INTER_NEAREST)
            if labels.size:
                labels = labels.copy()
                labels[:, 1:5] *= r
        h, w = img.shape[:2]

        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        lane4[y1a:y2a, x1a:x2a] = lane[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b

        if labels.size:
            lab = labels.copy()
            lab[:, 1] = labels[:, 1] + padw
            lab[:, 2] = labels[:, 2] + padh
            lab[:, 3] = labels[:, 3] + padw
            lab[:, 4] = labels[:, 4] + padh
            labels4.append(lab)

    labels4 = np.concatenate(labels4, 0) if labels4 else np.zeros((0, 5), dtype=np.float32)
    # Clip to canvas before the affine warp consumes the 2s canvas.
    if labels4.size:
        np.clip(labels4[:, 1::2], 0, 2 * s, out=labels4[:, 1::2])
        np.clip(labels4[:, 2::2], 0, 2 * s, out=labels4[:, 2::2])
    return (img4, lane4), labels4


def mixup(img1, lane1, labels1, img2, lane2, labels2, alpha=8.0, beta=8.0):
    """Beta-weighted image blend; labels are concatenated.
    [INFERRED] parameters (alpha=beta=8.0) taken from YOLOv7 default.
    Lane masks are combined by logical OR since both are binary foreground.
    """
    r = np.random.beta(alpha, beta)
    img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(img1.dtype)
    lane = np.maximum(lane1, lane2)  # foreground OR
    labels = np.concatenate([labels1, labels2], 0) if (labels1.size and labels2.size) \
             else (labels1 if labels1.size else labels2)
    return img, lane, labels

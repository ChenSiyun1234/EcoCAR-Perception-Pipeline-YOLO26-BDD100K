"""
Synchronized augmentation transforms for joint detection + lane segmentation.

All transforms operate on (image, boxes, mask) tuples to keep spatial
transforms consistent across modalities.

Geometry note — aspect-ratio consistency
-----------------------------------------
BDD100K images are natively 1280x720 (16:9). YOLO-style training resizes
(stretches) images to a square target, e.g. 640x640. This changes the
aspect ratio: circles become ovals, etc. The lane mask must undergo the
**same** geometric distortion so that spatial correspondence is preserved.

Both the image resize and the mask resize start from the same augmented
(H, W) source and are independently resized to their respective targets.
Because boxes are stored in normalized [0, 1] coordinates, a point at
relative position (ry, rx) in the source maps to (ry, rx) in both the
resized image and the resized mask, regardless of the absolute pixel
dimensions of each target.

However, when the image target is square (img_size x img_size) but the
mask target is non-square (mask_height x mask_width), the lane
segmentation head must implicitly learn that horizontal and vertical
spatial scales differ between its input feature maps (derived from the
square image) and its output (the non-square mask). This is learnable
but sub-optimal.

**Recommendation:** set mask_height == mask_width (both square) so that
the lane head's input and output share the same aspect ratio as the
square image. The config (default.yaml) controls these values.
"""

import random
import warnings
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class JointTransform:
    """
    Synchronized transform pipeline for image + detection boxes + lane mask.

    Boxes are in YOLO normalized format: [class_id, xc, yc, w, h].
    Masks are single-channel uint8 (0 or 255).
    """

    def __init__(
        self,
        img_size: int = 640,
        mask_height: int = 180,
        mask_width: int = 320,
        augment: bool = True,
        aug_cfg: Optional[Dict] = None,
        lane_target_cfg: Optional[Dict] = None,
    ):
        self.img_size = img_size
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.augment = augment

        # --- Geometry sanity check ---
        # When the image is resized to a square, a non-square mask means the
        # lane head must learn a different spatial scale for each axis.  This
        # is technically correct in normalised coordinates but sub-optimal.
        _img_is_square = True  # img_size produces (img_size, img_size)
        if _img_is_square and mask_height != mask_width:
            warnings.warn(
                f"Image target is square ({img_size}x{img_size}) but lane mask "
                f"target is non-square ({mask_height}x{mask_width}). Spatial "
                f"correspondence is preserved in normalised coordinates, but "
                f"the lane head must learn differing horizontal/vertical "
                f"scales. Consider setting mask_height == mask_width for "
                f"optimal geometry alignment.",
                stacklevel=2,
            )

        cfg = aug_cfg or {}
        self.flip_prob = cfg.get("horizontal_flip", 0.5)
        self.scale_range = cfg.get("scale_range", [1.0, 1.0])

        cj = cfg.get("color_jitter", {})
        self.brightness = cj.get("brightness", 0.0)
        self.contrast = cj.get("contrast", 0.0)
        self.saturation = cj.get("saturation", 0.0)
        self.hue = cj.get("hue", 0.0)

        self.blur_prob = cfg.get("blur_prob", 0.0)
        self.blur_kernel = cfg.get("blur_kernel", 3)
        self.noise_prob = cfg.get("noise_prob", 0.0)
        self.noise_std = cfg.get("noise_std", 10.0)

        lt = lane_target_cfg or {}
        self.lane_mode = lt.get("mode", "binary")
        self.dilation_kernel = lt.get("dilation_kernel", 3)
        self.distance_clip = lt.get("distance_clip", 10.0)
        self.gaussian_sigma = lt.get("gaussian_sigma", 2.0)

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image: (H, W, 3) uint8 RGB
            boxes: (N, 5) float32 — [class_id, xc, yc, w, h] normalized
            mask:  (H, W) uint8 — lane mask (0 or 255)
        Returns:
            image: (3, img_size, img_size) float32 [0,1]
            boxes: (N, 5) float32
            mask:  (1, mask_height, mask_width) float32
        """
        h, w = image.shape[:2]

        if self.augment:
            # ----- Spatial augmentations (applied to BOTH image and mask) -----

            # Random scale
            lo, hi = self.scale_range
            if lo != 1.0 or hi != 1.0:
                scale = random.uniform(lo, hi)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                h, w = new_h, new_w

            # Horizontal flip (image, mask, AND boxes — synchronised)
            if random.random() < self.flip_prob:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                if len(boxes) > 0:
                    boxes = boxes.copy()
                    boxes[:, 1] = 1.0 - boxes[:, 1]

            # ----- Photometric augmentations (image only) -----
            # These do not affect spatial alignment with the mask.

            # Color jitter
            image = self._color_jitter(image)

            # Gaussian blur
            if random.random() < self.blur_prob:
                k = self.blur_kernel
                if k % 2 == 0:
                    k += 1
                image = cv2.GaussianBlur(image, (k, k), 0)

            # Additive Gaussian noise
            if random.random() < self.noise_prob:
                noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # ----- Final resize -----
        # Both image and mask are resized from the same (H, W) source to
        # their respective target resolutions.  Because YOLO boxes use
        # normalised [0,1] coordinates, a point at relative position
        # (ry, rx) in the source maps identically in both targets.
        # The resize is a simple stretch (no letterboxing / padding).

        # Image -> (img_size, img_size)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Mask -> (mask_height, mask_width) using nearest-neighbour to
        # preserve hard label boundaries.
        mask = cv2.resize(mask, (self.mask_width, self.mask_height),
                          interpolation=cv2.INTER_NEAREST)

        # Process lane targets (binary, dilated, distance, gaussian)
        mask_float = self._process_lane_target(mask)

        # Normalize image to [0, 1] and transpose to CHW
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return image, boxes, mask_float

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply random color jitter in HSV space."""
        if self.brightness == 0 and self.contrast == 0 and self.saturation == 0 and self.hue == 0:
            return image

        img = image.astype(np.float32)

        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean()
            img = (img - mean) * factor + mean

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Saturation and hue in HSV
        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            if self.saturation > 0:
                factor = 1.0 + random.uniform(-self.saturation, self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            if self.hue > 0:
                shift = random.uniform(-self.hue, self.hue) * 180
                hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img

    def _process_lane_target(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to the configured target representation.
        Returns (1, H, W) float32 array.
        """
        binary = (mask > 128).astype(np.uint8)

        if self.lane_mode == "binary":
            target = binary.astype(np.float32)

        elif self.lane_mode == "dilated":
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.dilation_kernel, self.dilation_kernel))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            target = dilated.astype(np.float32)

        elif self.lane_mode == "distance":
            if binary.sum() == 0:
                target = np.zeros_like(binary, dtype=np.float32)
            else:
                dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
                target = np.clip(1.0 - dist / self.distance_clip, 0, 1).astype(np.float32)

        elif self.lane_mode == "gaussian":
            if binary.sum() == 0:
                target = np.zeros_like(binary, dtype=np.float32)
            else:
                dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
                target = np.exp(-0.5 * (dist / self.gaussian_sigma) ** 2).astype(np.float32)

        else:
            target = binary.astype(np.float32)

        return target[np.newaxis, ...]  # (1, H, W)

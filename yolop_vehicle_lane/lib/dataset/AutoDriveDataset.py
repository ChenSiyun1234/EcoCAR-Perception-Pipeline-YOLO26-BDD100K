"""
Base dataset for vehicle detection + lane segmentation.

Updated to support two layouts:
1) Raw BDD100K-style roots:
   DATAROOT=/content/bdd100k/images/100k
   LABELROOT=/content/bdd100k/labels/100k
   LANEROOT=/content/bdd100k/lane_masks

2) Packaged DETR_GeoLane-style root:
   ROOT=/content/bdd100k_vehicle5
   /images/{train,val}/*.jpg
   /labels/{train,val}/*.txt
   /masks/{train,val}/*.png   or /lane_masks/{train,val}/*.png

The goal is to keep the YOLOP training code intact while making the
filesystem resolution robust and deterministic.
"""

import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, load_mosaic, mixup


class AutoDriveDataset(Dataset):
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()

        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.split = indicator

        self.img_root, self.label_root, self.lane_root, self.layout_name = self._resolve_split_paths(cfg, indicator)
        print(f"[Dataset] split={indicator} | layout={self.layout_name}")
        print(f"[Dataset] images={self.img_root}")
        print(f"[Dataset] labels={self.label_root}")
        print(f"[Dataset] lanes ={self.lane_root}")

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    @staticmethod
    def _to_path(value):
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            return None
        return Path(value)

    @staticmethod
    def _existing_dir(candidates):
        for candidate in candidates:
            if candidate is None:
                continue
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _resolve_split_paths(self, cfg, split):
        dataset_root = self._to_path(getattr(cfg.DATASET, 'ROOT', ''))
        data_root = self._to_path(getattr(cfg.DATASET, 'DATAROOT', ''))
        label_root = self._to_path(getattr(cfg.DATASET, 'LABELROOT', ''))
        lane_root = self._to_path(getattr(cfg.DATASET, 'LANEROOT', ''))
        lane_dir_candidates = list(getattr(cfg.DATASET, 'LANE_DIR_CANDIDATES', ['masks', 'lane_masks']))

        # Case A: packaged root like DETR_GeoLane_pipeline
        packaged_img = None
        packaged_label = None
        packaged_lane = None
        if dataset_root is not None:
            packaged_img = self._existing_dir([
                dataset_root / 'images' / split,
                dataset_root / split / 'images',
            ])
            packaged_label = self._existing_dir([
                dataset_root / 'labels' / split,
                dataset_root / split / 'labels',
            ])
            lane_candidates = []
            for lane_name in lane_dir_candidates:
                lane_candidates.extend([
                    dataset_root / lane_name / split,
                    dataset_root / split / lane_name,
                ])
            packaged_lane = self._existing_dir(lane_candidates)
            if packaged_img is not None and packaged_label is not None and packaged_lane is not None:
                return packaged_img, packaged_label, packaged_lane, 'packaged_root'

        # Case B: user passed dataset root into DATAROOT/LABELROOT/LANEROOT directly
        direct_img = self._existing_dir([
            data_root / 'images' / split if data_root is not None else None,
            data_root / split if data_root is not None else None,
        ])
        direct_label = self._existing_dir([
            label_root / 'labels' / split if label_root is not None else None,
            label_root / split if label_root is not None else None,
        ])
        direct_lane_candidates = []
        for lane_name in lane_dir_candidates:
            direct_lane_candidates.extend([
                lane_root / lane_name / split if lane_root is not None else None,
                lane_root / split if lane_root is not None else None,
            ])
        direct_lane = self._existing_dir(direct_lane_candidates)
        if direct_img is not None and direct_label is not None and direct_lane is not None:
            return direct_img, direct_label, direct_lane, 'explicit_packaged_like'

        # Case C: raw BDD roots with train/val subfolders
        raw_img = self._existing_dir([
            data_root / split if data_root is not None else None,
        ])
        raw_label = self._existing_dir([
            label_root / split if label_root is not None else None,
        ])
        raw_lane = self._existing_dir([
            lane_root / split if lane_root is not None else None,
        ])
        if raw_img is not None and raw_label is not None and raw_lane is not None:
            return raw_img, raw_label, raw_lane, 'raw_bdd_roots'

        # Final fallback: preserve the original join behavior for debugging clarity.
        fallback_img = (data_root / split) if data_root is not None else Path(f'./images/{split}')
        fallback_label = (label_root / split) if label_root is not None else Path(f'./labels/{split}')
        fallback_lane = (lane_root / split) if lane_root is not None else Path(f'./masks/{split}')
        return fallback_img, fallback_label, fallback_lane, 'fallback'

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    # ── Helpers for Mosaic (YOLOv7-derived) ────────────────────────────
    def _load_mosaic_sample(self, idx):
        """Load one sample for mosaic: returns (img_rgb, lane_mask, labels_xyxy_abs).
        Labels are in pixel-absolute xyxy format in the source image's own
        coordinate frame (no letterbox applied here)."""
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {data['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lane = cv2.imread(data["lane"], 0)
        if lane is None:
            raise FileNotFoundError(f"Failed to read lane mask: {data['lane']}")
        h, w = img.shape[:2]
        det = data["label"]
        if det.size > 0:
            labels = det.copy()
            # det in normalized xywh → absolute xyxy
            labels[:, 1] = w * (det[:, 1] - det[:, 3] / 2)
            labels[:, 2] = h * (det[:, 2] - det[:, 4] / 2)
            labels[:, 3] = w * (det[:, 1] + det[:, 3] / 2)
            labels[:, 4] = h * (det[:, 2] + det[:, 4] / 2)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        return img, lane, labels

    def __getitem__(self, idx):
        """Training-time pipeline depends on cfg flags:
          * Mosaic (DATASET.MOSAIC) gates on with prob p, composes a 4-tile
            image, then runs random_perspective with negative border — the
            standard YOLOv5/YOLOv7 recipe.
          * MixUp (DATASET.MIXUP) blends two mosaic-warped samples.
          * Both flags default False to keep parity with YOLOP's original
            recipe. Switch them on in the YOLOPv2-style YAML.
        [INFERRED] YOLOPv2 training code is not public. Mosaic/MixUp
        follow YOLOv7 conventions which YOLOPv2 inherits its backbone
        from.
        """
        use_mosaic = bool(self.is_train and getattr(self.cfg.DATASET, 'MOSAIC', False)
                          and random.random() < getattr(self.cfg.DATASET, 'MOSAIC_PROB', 1.0))
        use_mixup = bool(self.is_train and getattr(self.cfg.DATASET, 'MIXUP', False)
                         and random.random() < getattr(self.cfg.DATASET, 'MIXUP_PROB', 0.15))

        # `inputsize` can be:
        #   int                → square letterbox with auto=True
        #                        (YOLOP style; for 1280x720 this produces
        #                         H=384 naturally via stride=32 rounding).
        #   tuple / list (H,W) → explicit rectangular letterbox
        #                        (auto=False). Used for YOLOPv2 val @ 384×640.
        explicit_shape = None
        if isinstance(self.inputsize, (list, tuple)):
            explicit_shape = (int(self.inputsize[0]), int(self.inputsize[1]))
            resized_shape = max(explicit_shape)
        else:
            resized_shape = int(self.inputsize)
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)

        if use_mosaic:
            (img, lane_label), labels = load_mosaic(self, idx, s=resized_shape)
            # `random_perspective` with a negative border cuts the 2s canvas
            # back down to s-x-s while also applying rotate/scale/shear.
            border = (-resized_shape // 2, -resized_shape // 2)
            (img, lane_label), labels = random_perspective(
                combination=(img, lane_label),
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR,
                border=border,
            )
            if use_mixup:
                idx2 = random.randint(0, len(self) - 1)
                (img2, lane2), labels2 = load_mosaic(self, idx2, s=resized_shape)
                (img2, lane2), labels2 = random_perspective(
                    combination=(img2, lane2), targets=labels2,
                    degrees=self.cfg.DATASET.ROT_FACTOR,
                    translate=self.cfg.DATASET.TRANSLATE,
                    scale=self.cfg.DATASET.SCALE_FACTOR,
                    shear=self.cfg.DATASET.SHEAR,
                    border=border,
                )
                img, lane_label, labels = mixup(img, lane_label, labels, img2, lane2, labels2)
            h, w = img.shape[:2]
            augment_hsv(img,
                        hgain=self.cfg.DATASET.HSV_H,
                        sgain=self.cfg.DATASET.HSV_S,
                        vgain=self.cfg.DATASET.HSV_V)
            if len(labels):
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, [2, 4]] /= h
                labels[:, [1, 3]] /= w
            if random.random() < 0.5:
                img = np.fliplr(img)
                lane_label = np.fliplr(lane_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
            shapes = (h, w), ((1.0, 1.0), (0.0, 0.0))  # synthesized shape; not used by val

        else:
            # ── Original YOLOP path: resize + letterbox + perspective/HSV/flip ──
            data = self.db[idx]
            img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {data['image']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            lane_label = cv2.imread(data["lane"], 0)
            if lane_label is None:
                raise FileNotFoundError(f"Failed to read lane mask: {data['lane']}")

            h0, w0 = img.shape[:2]
            r = resized_shape / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
                # REPAIR (v5): lane masks are thin binary supervision (8
                # px train / 2 px test per YOLOPv2 paper §3). Bilinear /
                # area interp produces intermediate uint8 values that then
                # get over- or under-selected by the downstream
                # `threshold(>1)` binarizer. Use NEAREST everywhere a
                # lane mask is resized / warped — in this file,
                # load_mosaic, letterbox, and random_perspective.
                lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)),
                                        interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]

            if explicit_shape is not None:
                # Explicit rectangular target (e.g. val @ 384×640). Uses
                # letterbox auto=False so we get exactly the requested
                # output shape, not the YOLOP `stride-rounded` shape.
                (img, lane_label), ratio, pad = letterbox(
                    (img, lane_label), explicit_shape,
                    auto=False, scaleup=self.is_train)
            else:
                (img, lane_label), ratio, pad = letterbox(
                    (img, lane_label), resized_shape,
                    auto=True, scaleup=self.is_train)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            det_label = data["label"]
            labels = []
            if det_label.size > 0:
                labels = det_label.copy()
                labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]
                labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]
                labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

            if self.is_train:
                (img, lane_label), labels = random_perspective(
                    combination=(img, lane_label),
                    targets=labels,
                    degrees=self.cfg.DATASET.ROT_FACTOR,
                    translate=self.cfg.DATASET.TRANSLATE,
                    scale=self.cfg.DATASET.SCALE_FACTOR,
                    shear=self.cfg.DATASET.SHEAR,
                )
                augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
                if len(labels):
                    labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                    labels[:, [2, 4]] /= img.shape[0]
                    labels[:, [1, 3]] /= img.shape[1]
                if random.random() < 0.5:
                    img = np.fliplr(img)
                    lane_label = np.fliplr(lane_label)
                    if len(labels):
                        labels[:, 1] = 1 - labels[:, 1]
            else:
                if len(labels):
                    labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                    labels[:, [2, 4]] /= img.shape[0]
                    labels[:, [1, 3]] /= img.shape[1]

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(np.asarray(labels))

        img = np.ascontiguousarray(img)

        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)
        lane_label = torch.stack((lane2[0], lane1[0]), 0)

        target = [labels_out, lane_label]
        img = self.transform(img)

        path = self.db[idx]["image"]
        return img, target, path, shapes

    def select_data(self, db):
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_det, label_lane = [], []
        for i, l in enumerate(label):
            l_det, l_lane = l
            l_det[:, 0] = i
            label_det.append(l_det)
            label_lane.append(l_lane)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_lane, 0)], paths, shapes

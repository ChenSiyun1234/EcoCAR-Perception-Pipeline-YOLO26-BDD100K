"""
Base dataset for vehicle detection + lane segmentation.
Adapted from YOLOP AutoDriveDataset with drivable-area mask removed.
Target is now [det_labels, lane_label] instead of [det_labels, seg_label, lane_label].
"""

import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh


class AutoDriveDataset(Dataset):
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.lane_root = lane_root / indicator

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lane_label = cv2.imread(data["lane"], 0)

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]
        r = resized_shape / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        # Letterbox: 2-tuple (img, lane_label)
        (img, lane_label), ratio, pad = letterbox((img, lane_label), resized_shape, auto=True, scaleup=self.is_train)
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
            # random_perspective: 2-tuple (img, lane_label)
            combination = (img, lane_label)
            (img, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )

            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)

            if len(labels):
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, [2, 4]] /= img.shape[0]
                labels[:, [1, 3]] /= img.shape[1]

            # Random horizontal flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
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
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = np.ascontiguousarray(img)

        # Lane: binary 2-channel (background + lane)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)
        lane_label = torch.stack((lane2[0], lane1[0]), 0)

        # Target: [det_labels, lane_label] (2 elements, no DA)
        target = [labels_out, lane_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_det, label_lane = [], []
        for i, l in enumerate(label):
            l_det, l_lane = l
            l_det[:, 0] = i  # add target image index
            label_det.append(l_det)
            label_lane.append(l_lane)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_lane, 0)], paths, shapes

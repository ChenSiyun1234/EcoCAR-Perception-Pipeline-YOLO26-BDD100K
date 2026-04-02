"""
Lane segmentation metrics: IoU, F1/Dice, precision, recall, threshold sweep.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class LaneMetrics:
    """Accumulates lane predictions and computes metrics at the end."""

    def __init__(self, thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]):
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self._ious = {t: [] for t in self.thresholds}
        self._f1s = {t: [] for t in self.thresholds}
        self._precisions = {t: [] for t in self.thresholds}
        self._recalls = {t: [] for t in self.thresholds}

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) ground truth (float, 0 to 1)
        """
        probs = torch.sigmoid(logits).cpu().numpy()
        gt = (targets.cpu().numpy() > 0.5).astype(np.uint8)

        for b in range(probs.shape[0]):
            pred_prob = probs[b, 0]
            gt_bin = gt[b, 0]

            for t in self.thresholds:
                pred_bin = (pred_prob > t).astype(np.uint8)
                tp = np.logical_and(pred_bin, gt_bin).sum()
                fp = np.logical_and(pred_bin, ~gt_bin.astype(bool)).sum()
                fn = np.logical_and(~pred_bin.astype(bool), gt_bin).sum()

                intersection = tp
                union = tp + fp + fn

                # Skip images with no lane GT — they inflate IoU artificially
                if gt_bin.sum() == 0 and pred_bin.sum() == 0:
                    continue
                iou = intersection / union if union > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                self._ious[t].append(iou)
                self._f1s[t].append(f1)
                self._precisions[t].append(precision)
                self._recalls[t].append(recall)

    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics across all thresholds."""
        results = {}

        for t in self.thresholds:
            suffix = f"@{t:.1f}"
            n = len(self._ious[t])
            if n == 0:
                continue
            results[f"lane_iou{suffix}"] = np.mean(self._ious[t])
            results[f"lane_f1{suffix}"] = np.mean(self._f1s[t])
            results[f"lane_precision{suffix}"] = np.mean(self._precisions[t])
            results[f"lane_recall{suffix}"] = np.mean(self._recalls[t])

        # Primary metrics at threshold=0.5
        if 0.5 in self.thresholds and self._ious.get(0.5):
            results["lane_miou"] = np.mean(self._ious[0.5])
            results["lane_f1"] = np.mean(self._f1s[0.5])

        # Best threshold by F1
        best_t, best_f1 = 0.5, 0.0
        for t in self.thresholds:
            if self._f1s.get(t):
                f1 = np.mean(self._f1s[t])
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
        results["lane_best_thresh"] = best_t
        results["lane_best_f1"] = best_f1

        return results

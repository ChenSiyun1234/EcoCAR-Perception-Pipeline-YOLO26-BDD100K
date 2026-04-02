"""
Training loop for DualPathNet.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from .config import Config, VEHICLE_CLASSES, EXPANDED_CLASSES, WEIGHTS_DIR, ensure_dirs
from .losses import DualPathLoss, box_cxcywh_to_xyxy
from .metrics import DetectionMetrics, LaneMetrics


class Trainer:
    def __init__(self, cfg: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        ensure_dirs(cfg)
        self.criterion = DualPathLoss(cfg).to(self.device)
        backbone_params, other_params = [], []
        for name, p in model.named_parameters():
            (backbone_params if "backbone" in name else other_params).append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_scale, "name": "backbone"},
            {"params": other_params, "lr": cfg.lr, "name": "heads"},
        ], weight_decay=cfg.weight_decay)
        self.scaler = GradScaler("cuda", enabled=(cfg.amp and cfg.device == "cuda")) if cfg.device == "cuda" else None
        nc = 7 if cfg.use_expanded_classes else 5
        self.det_metrics = DetectionMetrics(num_classes=nc, device=cfg.device)
        self.lane_metrics = LaneMetrics(match_thresh_px=cfg.lane_match_thresh, img_size=cfg.img_size)
        self.history = []
        self.best_scores = {"det": 0.0, "lane": 0.0, "joint": 0.0}
        self.save_dir = cfg.save_dir
        os.makedirs(os.path.join(self.save_dir, "weights"), exist_ok=True)

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.cfg.warmup_epochs:
            return self.cfg.lr * (epoch + 1) / max(self.cfg.warmup_epochs, 1)
        progress = (epoch - self.cfg.warmup_epochs) / max(1, self.cfg.epochs - self.cfg.warmup_epochs)
        return self.cfg.lr * (self.cfg.min_lr_ratio + (1 - self.cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    def _set_lr(self, epoch: int):
        lr = self._get_lr(epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * self.cfg.backbone_lr_scale if pg.get("name") == "backbone" else lr

    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=(self.cfg.amp and self.device == "cuda"))

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals, n = {}, 0
        for batch in self.train_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                outputs = self.model(images)
                loss, info = self.criterion(outputs, batch_gpu)
            if not torch.isfinite(loss):
                continue
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            for k, v in info.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item())
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.det_metrics.reset()
        self.lane_metrics.reset()
        val_loss_sum = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with self._autocast():
                outputs = self.model(images)
                loss, _ = self.criterion(outputs, batch_gpu)
            val_loss_sum += float(loss.item())
            n += 1
            B = images.shape[0]
            pred_logits = outputs["det_pred_logits"]
            pred_boxes = outputs["det_pred_boxes"]
            det_targets = batch_gpu["det_targets"]
            for bi in range(B):
                scores, labels = pred_logits[bi, :, :-1].max(dim=-1)
                scores = scores.sigmoid()
                keep = scores > self.cfg.conf_thresh
                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[bi, keep]) * self.cfg.img_size
                if det_targets.shape[0] > 0:
                    mask = det_targets[:, 0] == bi
                    tgt = det_targets[mask]
                    gt_xyxy = box_cxcywh_to_xyxy(tgt[:, 2:6]) * self.cfg.img_size if tgt.shape[0] > 0 else torch.empty((0, 4), device=self.device)
                    gt_cls = tgt[:, 1].long() if tgt.shape[0] > 0 else torch.empty(0, dtype=torch.long, device=self.device)
                else:
                    gt_xyxy = torch.empty((0, 4), device=self.device)
                    gt_cls = torch.empty(0, dtype=torch.long, device=self.device)
                self.det_metrics.update(pred_xyxy, scores[keep], labels[keep], gt_xyxy, gt_cls)
            if batch_gpu["has_lanes"].sum() > 0:
                for bi in range(B):
                    if batch_gpu["has_lanes"][bi] > 0.5:
                        self.lane_metrics.update(
                            outputs["lane_pred_points"][bi].detach().cpu(),
                            outputs["lane_exist_logits"][bi].detach().cpu(),
                            batch_gpu["lane_points"][bi].detach().cpu(),
                            batch_gpu["lane_existence"][bi].detach().cpu(),
                            batch_gpu["lane_visibility"][bi].detach().cpu(),
                        )
        res = {"val_loss": val_loss_sum / max(n, 1)}
        res.update(self.det_metrics.compute())
        res.update(self.lane_metrics.compute())
        return res

    def _save(self, name: str, epoch: int, metrics: Dict[str, float]):
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg.to_dict(),
            "metrics": metrics,
        }
        run_path = os.path.join(self.save_dir, "weights", f"{name}.pt")
        torch.save(payload, run_path)
        mirror_path = os.path.join(WEIGHTS_DIR, f"{self.cfg.run_name}_{name}.pt")
        try:
            shutil.copy2(run_path, mirror_path)
        except Exception:
            pass

    def _save_history(self):
        csv_path = os.path.join(self.save_dir, "history.csv")
        json_path = os.path.join(self.save_dir, "history.json")
        if self.history:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                writer.writeheader()
                writer.writerows(self.history)
            with open(json_path, "w") as f:
                json.dump(self.history, f, indent=2)

    def train(self) -> list:
        epochs = self.cfg.epochs
        classes = EXPANDED_CLASSES if self.cfg.use_expanded_classes else VEHICLE_CLASSES
        print(f"\n{'='*65}")
        print(f"  DualPathNet Training: {epochs} epochs")
        print(f"  Classes: {', '.join(classes)}")
        print(f"  Save: {self.save_dir}")
        print(f"{'='*65}")
        patience_counter = 0
        start = time.time()
        last_val = {}
        for epoch in range(epochs):
            self._set_lr(epoch)
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            last_val = val_metrics
            dt = time.time() - t0
            record = {"epoch": epoch + 1, "time": dt, "lr": self.optimizer.param_groups[-1]["lr"]}
            record.update({f"train_{k}": v for k, v in train_metrics.items()})
            record.update(val_metrics)
            self.history.append(record)
            det_map = val_metrics.get("det_map50", 0.0)
            lane_f1 = val_metrics.get("lane_f1", 0.0)
            print(f"Epoch {epoch+1:>3}/{epochs} | loss={train_metrics.get('loss', 0):.3f} | val={val_metrics.get('val_loss', 0):.3f} | mAP50={det_map*100:.1f}% | laneF1={lane_f1*100:.1f}% | {dt:.0f}s")
            improved = False
            if det_map > self.best_scores["det"]:
                self.best_scores["det"] = det_map
                self._save("best_det", epoch + 1, val_metrics)
                improved = True
            if lane_f1 > self.best_scores["lane"]:
                self.best_scores["lane"] = lane_f1
                self._save("best_lane", epoch + 1, val_metrics)
                improved = True
            joint = 0.6 * det_map + 0.4 * lane_f1
            if joint > self.best_scores["joint"]:
                self.best_scores["joint"] = joint
                self._save("best_joint", epoch + 1, val_metrics)
                improved = True
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        self._save("last", len(self.history), last_val)
        self._save_history()
        total_time = time.time() - start
        print(f"\n{'='*65}")
        print(f"  Training complete — {total_time/60:.1f} min")
        print(f"  Best det mAP@50: {self.best_scores['det']*100:.2f}%")
        print(f"  Best lane F1:    {self.best_scores['lane']*100:.2f}%")
        return self.history

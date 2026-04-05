
import csv
import json
import math
import os
import time
from typing import Dict, Optional

import torch
from torch.amp import GradScaler

from .config import VEHICLE_CLASSES, EXPANDED_CLASSES
from .losses import DualPathLoss, box_cxcywh_to_xyxy
from .metrics import DetectionMetrics, LaneMetrics


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = cfg.device
        self.criterion = DualPathLoss(cfg)

        backbone_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(p)
            else:
                other_params.append(p)
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_scale, 'name': 'backbone'},
            {'params': other_params, 'lr': cfg.lr, 'name': 'heads'},
        ], weight_decay=cfg.weight_decay)
        self.scaler = GradScaler('cuda', enabled=cfg.amp) if cfg.device == 'cuda' else None
        nc = 7 if cfg.use_expanded_classes else 5
        self.det_metrics = DetectionMetrics(num_classes=nc, device=cfg.device)
        self.lane_metrics = LaneMetrics(match_thresh_px=cfg.lane_match_thresh, img_size=cfg.img_size, raster_h=cfg.lane_raster_h, raster_w=cfg.lane_raster_w, raster_thickness=cfg.lane_raster_thickness)
        self.history = []
        self.best_scores = {'det': 0.0, 'lane': 0.0, 'joint': 0.0}
        self.save_dir = cfg.save_dir
        self.start_epoch = 0
        self.last_val_metrics: Optional[dict] = None
        os.makedirs(os.path.join(self.save_dir, 'weights'), exist_ok=True)
        if getattr(cfg, 'auto_resume', True):
            self._maybe_resume()

    def _maybe_resume(self):
        candidate_paths = []
        resume_path = getattr(self.cfg, 'resume_path', '')
        if resume_path:
            candidate_paths.append(resume_path)
        candidate_paths.extend([
            os.path.join(self.save_dir, 'weights', 'last.pt'),
            os.path.join(self.save_dir, 'weights', 'best_joint.pt'),
            os.path.join(self.save_dir, 'weights', 'best_det.pt'),
            os.path.join(self.save_dir, 'last.pt'),
        ])
        ckpt_path = next((p for p in candidate_paths if p and os.path.isfile(p)), None)
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model'
        opt_key = 'optimizer_state_dict' if 'optimizer_state_dict' in ckpt else 'optimizer'
        scaler_key = 'scaler_state_dict' if 'scaler_state_dict' in ckpt else 'scaler'
        try:
            self.model.load_state_dict(ckpt[model_key], strict=True)
        except Exception:
            self.model.load_state_dict(ckpt[model_key], strict=False)
        if opt_key in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt[opt_key])
            except Exception:
                pass
        if self.scaler is not None and scaler_key in ckpt:
            try:
                self.scaler.load_state_dict(ckpt[scaler_key])
            except Exception:
                pass
        self.best_scores = ckpt.get('best_scores', self.best_scores)
        self.start_epoch = int(ckpt.get('epoch', 0))
        self.last_val_metrics = ckpt.get('metrics')
        hist_path = os.path.join(self.save_dir, 'history.csv')
        if os.path.isfile(hist_path):
            with open(hist_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self.history = [{k: self._parse_scalar(v) for k, v in row.items()} for row in reader]
        print(f'Resumed from checkpoint: {ckpt_path} (next epoch={self.start_epoch + 1})')

    @staticmethod
    def _parse_scalar(v):
        if isinstance(v, (int, float)):
            return v
        if v is None:
            return v
        s = str(v)
        try:
            if s.lower() in {'nan', 'inf', '-inf'}:
                return float(s)
            if '.' in s or 'e' in s.lower():
                return float(s)
            return int(s)
        except Exception:
            return v

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.cfg.warmup_epochs:
            return self.cfg.lr * (epoch + 1) / self.cfg.warmup_epochs
        progress = (epoch - self.cfg.warmup_epochs) / max(1, self.cfg.epochs - self.cfg.warmup_epochs)
        return self.cfg.lr * (self.cfg.min_lr_ratio + (1 - self.cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    def _set_lr(self, epoch: int):
        lr = self._get_lr(epoch)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * self.cfg.backbone_lr_scale if pg.get('name') == 'backbone' else lr

    def _set_task_weights(self, epoch: int):
        warm_epochs = max(int(getattr(self.cfg, 'task_warmup_epochs', 0)), 0)
        if epoch < warm_epochs:
            self.criterion.det_weight = float(getattr(self.cfg, 'det_task_warmup_weight', self.cfg.det_task_weight))
            self.criterion.lane_weight = float(getattr(self.cfg, 'lane_task_warmup_weight', self.cfg.lane_task_weight))
        else:
            self.criterion.det_weight = float(self.cfg.det_task_weight)
            self.criterion.lane_weight = float(self.cfg.lane_task_weight)
        self.criterion.set_epoch(epoch)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._set_task_weights(epoch)
        totals = {}
        n = 0
        for batch in self.train_loader:
            images = batch['images'].to(self.device, non_blocking=True)
            batch_gpu = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
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
            totals['loss'] = totals.get('loss', 0.0) + float(loss.item())
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.eval()
        self.det_metrics.reset()
        self.lane_metrics.reset()
        val_loss_sum = 0.0
        n = 0
        batch_limit = max_batches if max_batches is not None else int(getattr(self.cfg, 'max_val_batches', 0) or 0)
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_limit > 0 and batch_idx >= batch_limit:
                break
            images = batch['images'].to(self.device, non_blocking=True)
            batch_gpu = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                outputs = self.model(images)
                loss, _ = self.criterion(outputs, batch_gpu)
            val_loss_sum += float(loss.item())
            n += 1
            b = images.shape[0]
            pred_logits = outputs['det_pred_logits']
            pred_boxes = outputs['det_pred_boxes']
            det_targets = batch_gpu['det_targets']
            for bi in range(b):
                probs = pred_logits[bi].softmax(dim=-1)
                scores, labels = probs[:, :-1].max(dim=-1)
                keep = scores > self.cfg.conf_thresh
                keep_idx = torch.where(keep)[0]
                if keep_idx.numel() > 150:
                    topk = scores[keep_idx].topk(150).indices
                    keep_idx = keep_idx[topk]
                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[bi, keep_idx]) * self.cfg.img_size
                if det_targets.shape[0] > 0:
                    mask = det_targets[:, 0] == bi
                    tgt = det_targets[mask]
                    if tgt.shape[0] > 0:
                        gt_xyxy = box_cxcywh_to_xyxy(tgt[:, 2:6]) * self.cfg.img_size
                        gt_cls = tgt[:, 1].long()
                    else:
                        gt_xyxy = torch.empty((0, 4), device=self.device)
                        gt_cls = torch.empty(0, dtype=torch.long, device=self.device)
                else:
                    gt_xyxy = torch.empty((0, 4), device=self.device)
                    gt_cls = torch.empty(0, dtype=torch.long, device=self.device)
                self.det_metrics.update(pred_xyxy, scores[keep_idx], labels[keep_idx], gt_xyxy, gt_cls)
            has_lane_mask = batch_gpu['has_lanes'] > 0.5
            if has_lane_mask.any():
                for bi in torch.where(has_lane_mask)[0].tolist():
                    self.lane_metrics.update(
                        outputs['lane_pred_points'][bi].detach().cpu(),
                        outputs['lane_exist_logits'][bi].detach().cpu(),
                        batch_gpu['lane_points'][bi].detach().cpu(),
                        batch_gpu['lane_existence'][bi].detach().cpu(),
                        batch_gpu['lane_visibility'][bi].detach().cpu(),
                    )
        results = {'val_loss': val_loss_sum / max(n, 1)}
        results.update(self.det_metrics.compute())
        results.update(self.lane_metrics.compute())
        results['lane_geom_runtime_scale'] = float(self.criterion.lane_loss.geom_runtime_scale)
        results['lane_raster_runtime_scale'] = float(self.criterion.lane_loss.raster_runtime_scale)
        self.last_val_metrics = results
        return results

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
        val_interval = max(int(getattr(self.cfg, 'val_interval', 1)), 1)
        for epoch in range(self.start_epoch, epochs):
            self._set_lr(epoch)
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            do_val = ((epoch == self.start_epoch) or ((epoch + 1) % val_interval == 0) or (epoch + 1 == epochs))
            if do_val:
                val_metrics = self.validate()
            else:
                val_metrics = self.last_val_metrics or {'val_loss': float('nan'), 'det_map50': 0.0, 'lane_miou': 0.0, 'lane_overlap_f1': 0.0}
            dt = time.time() - t0
            record = {'epoch': epoch + 1, 'time': dt, 'lr': self.optimizer.param_groups[-1]['lr'], 'det_task_weight': self.criterion.det_weight, 'lane_task_weight': self.criterion.lane_weight}
            record.update({f'train_{k}': v for k, v in train_metrics.items()})
            record.update(val_metrics)
            self.history.append(record)
            det_map = float(val_metrics.get('det_map50', 0) or 0)
            lane_miou = float(val_metrics.get('lane_miou', 0) or 0)
            lane_f1 = float(val_metrics.get('lane_overlap_f1', 0) or 0)
            val_tag = f"val={val_metrics.get('val_loss', float('nan')):.3f}" if do_val else 'val=skip'
            print(f"Epoch {epoch+1:>3}/{epochs} | loss={train_metrics.get('loss', 0):.3f} | {val_tag} | mAP50={det_map*100:.1f}% | lane mIoU={lane_miou*100:.1f}% | lane overlapF1={lane_f1*100:.1f}% | {dt:.0f}s")
            if do_val:
                improved = False
                if det_map > self.best_scores['det']:
                    self.best_scores['det'] = det_map
                    self._save('best_det', epoch + 1, val_metrics)
                    improved = True
                lane_score = lane_miou
                if lane_score > self.best_scores['lane']:
                    self.best_scores['lane'] = lane_score
                    self._save('best_lane', epoch + 1, val_metrics)
                    improved = True
                joint = det_map * 0.6 + lane_score * 0.4
                if joint > self.best_scores['joint']:
                    self.best_scores['joint'] = joint
                    self._save('best_joint', epoch + 1, val_metrics)
                    improved = True
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
            self._save('last', epoch + 1, val_metrics)
            self._save_history()
        total_time = time.time() - start
        print(f"\n{'='*65}")
        print(f"  Training complete — {total_time/60:.1f} min")
        print(f"  Best det mAP@50: {self.best_scores['det']*100:.2f}%")
        print(f"  Best lane mIoU:  {self.best_scores['lane']*100:.2f}%")
        print(f"  Best joint:      {self.best_scores['joint']:.4f}")
        print(f"{'='*65}")
        return self.history

    def _save(self, name: str, epoch: int, metrics: dict):
        path = os.path.join(self.save_dir, 'weights', f'{name}.pt')
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_scores': self.best_scores,
            'arch_config': self.model._arch_config,
            'config': self.cfg.to_dict(),
        }
        if self.scaler is not None:
            payload['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(payload, path)

    def _save_history(self):
        if not self.history:
            return
        csv_path = os.path.join(self.save_dir, 'history.csv')
        keys = list(self.history[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
        summary = {'run_name': self.cfg.run_name, 'epochs': len(self.history), 'best_scores': self.best_scores}
        with open(os.path.join(self.save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

"""
Metric-driven joint trainer for MultiTaskYOLO.

Key improvements over legacy joint_trainer.py:
- Checkpoint selection by actual task metrics (mAP, mIoU), not val loss
- EMA model for stable evaluation
- Multi-task weighting strategies (fixed, uncertainty, PCGrad)
- Detection preservation via teacher distillation or L2 regularization
- Warmup + cosine scheduler
- CSV + JSON logging per epoch
- Saves best_det.pt, best_lane.pt, best_joint.pt, last.pt
"""

import copy
import csv
import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.losses.det_loss import DetectionLossWrapper
from src.losses.lane_loss import build_lane_loss
from src.losses.multitask import build_multitask_strategy, PCGrad
from src.metrics.detection import DetectionMetrics
from src.metrics.lane import LaneMetrics
from src.trainers.ema import ModelEMA
from src.utils.class_map import remap_preds_to_vehicle_names, remap_targets_bdd_to_vehicle, NUM_VEHICLE_CLASSES


class JointTrainer:

    def __init__(self, cfg: dict, model: nn.Module,
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.amp = cfg.get("amp", True)

        train_cfg = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})
        eval_cfg = cfg.get("eval", {})

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── Losses ────────────────────────────────────────────────────────
        self.det_loss_fn = DetectionLossWrapper(self.model)
        self.lane_loss_fn = build_lane_loss(loss_cfg.get("lane", {})).to(self.device)
        self.mt_strategy = build_multitask_strategy(loss_cfg.get("multitask", {})).to(self.device)
        self.use_pcgrad = loss_cfg.get("multitask", {}).get("strategy") == "pcgrad"
        self.use_staged = loss_cfg.get("multitask", {}).get("strategy") == "staged"
        if self.use_staged:
            self.mt_strategy.set_epoch(0)

        # ── Detection preservation ────────────────────────────────────────
        det_pres = loss_cfg.get("detection_preservation", {})
        self.det_preservation_mode = det_pres.get("mode", "none") if det_pres.get("enabled") else "none"
        self.teacher_weight = det_pres.get("teacher_weight", 0.5)
        self.l2_weight = det_pres.get("l2_weight", 0.001)
        self.teacher_model = None
        self._init_weights_snapshot = None

        # With native nc=5, all classes are relevant (0-4).
        self.teacher_relevant_classes = det_pres.get(
            "relevant_classes", list(range(NUM_VEHICLE_CLASSES)))
        self.teacher_conf_thresh = det_pres.get("teacher_conf_thresh", 0.25)

        if self.det_preservation_mode == "teacher":
            teacher_ckpt = det_pres.get("teacher_checkpoint")
            if teacher_ckpt and os.path.isfile(teacher_ckpt):
                # Load teacher using the same robust loading as warm_start
                self.teacher_model = copy.deepcopy(self.model)
                loaded = self.teacher_model.warm_start_from_checkpoint(
                    teacher_ckpt, device=str(self.device))
                print(f"Detection preservation: teacher loaded from {teacher_ckpt} ({loaded} params)")
            else:
                # Fallback: snapshot current model (works if warm_start already applied)
                self.teacher_model = copy.deepcopy(self.model)
                print("\n" + "!" * 70)
                print("WARNING: No teacher_checkpoint provided. Detection preservation")
                print("will use a snapshot of the initial model (backbone pretrained only).")
                print("For best results, provide a strong detection-only checkpoint")
                print("from Notebook 03.")
                print("!" * 70 + "\n")
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        elif self.det_preservation_mode == "l2_reg":
            self._init_weights_snapshot = {
                n: p.data.clone() for n, p in self.model.named_parameters()
                if 'backbone_neck' in n or 'detect_head' in n
            }
            print(f"Detection preservation: L2 reg on {len(self._init_weights_snapshot)} params")

        # ── Optimizer ─────────────────────────────────────────────────────
        # Support optimizer config at cfg["optimizer"] or cfg["training"]["optimizer"]
        opt_cfg = train_cfg.get("optimizer", cfg.get("optimizer", {}))
        lr = opt_cfg.get("lr", 1e-3)
        self.base_lr = lr

        param_groups = self._build_param_groups(lr, opt_cfg)

        # Add uncertainty weighting params if applicable
        if hasattr(self.mt_strategy, 'log_var_det'):
            param_groups.append({
                "params": list(self.mt_strategy.parameters()),
                "lr": lr, "name": "mt_weights",
            })

        opt_type = opt_cfg.get("type", "adamw").lower()
        if opt_type == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=opt_cfg.get("momentum", 0.937),
                weight_decay=opt_cfg.get("weight_decay", 5e-4),
                nesterov=opt_cfg.get("nesterov", True),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups, weight_decay=opt_cfg.get("weight_decay", 1e-4))

        # ── Scheduler ─────────────────────────────────────────────────────
        self.epochs = train_cfg.get("epochs", 30)
        sched_cfg = train_cfg.get("scheduler", cfg.get("scheduler", {}))
        self.warmup_epochs = sched_cfg.get("warmup_epochs", 3)
        self.min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)
        self.scheduler = None  # Built in train()

        # ── Gradient clipping / stability ─────────────────────────────────
        self.grad_clip = train_cfg.get("gradient_clip", 5.0)
        self.max_skip_batches = train_cfg.get("max_skip_batches", 50)
        self.max_skip_ratio = train_cfg.get("max_skip_ratio", 0.25)
        self.disable_teacher_on_instability = train_cfg.get("disable_teacher_on_instability", True)
        self._teacher_disabled = False

        # ── AMP ───────────────────────────────────────────────────────────
        self.scaler = GradScaler('cuda', enabled=self.amp) if str(self.device).startswith('cuda') else None

        # ── EMA ───────────────────────────────────────────────────────────
        ema_cfg = train_cfg.get("ema", {})
        self.ema = None
        if ema_cfg.get("enabled", True):
            self.ema = ModelEMA(self.model, decay=ema_cfg.get("decay", 0.9999))
            print(f"EMA enabled (decay={ema_cfg.get('decay', 0.9999)})")

        # ── Metrics ───────────────────────────────────────────────────────
        self.lane_metrics = LaneMetrics(
            thresholds=eval_cfg.get("lane_thresholds", [0.3, 0.4, 0.5, 0.6, 0.7]))
        self.det_metrics = DetectionMetrics(
            device=self.device,
            max_detection_threshold=eval_cfg.get("max_det", 300))

        # ── Checkpointing ─────────────────────────────────────────────────
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.save_period = ckpt_cfg.get("save_period", 5)
        self.joint_score_weights = ckpt_cfg.get("joint_score_weights", {
            "det_map50": 0.4, "det_map50_95": 0.2, "lane_miou": 0.3, "lane_f1": 0.1,
        })

        self.save_dir = os.path.join("training_runs", cfg.get("run_name", "joint_default"))
        os.makedirs(os.path.join(self.save_dir, "weights"), exist_ok=True)

        # ── Logging ───────────────────────────────────────────────────────
        self.history = []
        self.best_scores = {"det": 0.0, "lane": 0.0, "joint": 0.0}

        # NMS for evaluation
        self.conf_thresh = eval_cfg.get("conf_thresh", 0.001)
        self.nms_iou = eval_cfg.get("nms_iou_thresh", 0.6)
        self.max_det = eval_cfg.get("max_det", 300)

    def _build_param_groups(self, lr: float, opt_cfg: dict):
        backbone_scale = opt_cfg.get("backbone_lr_scale", 0.1)
        neck_scale = opt_cfg.get("neck_lr_scale", 0.5)

        backbone, dual_neck, lane_head, detect_head = [], [], [], []
        for name, param in self.model.named_parameters():
            if "lane_head" in name:
                lane_head.append(param)
            elif "detect_head" in name:
                detect_head.append(param)
            elif "dual_neck" in name:
                dual_neck.append(param)
            else:
                backbone.append(param)

        groups = [
            {"params": backbone, "lr": lr * backbone_scale, "name": "backbone"},
            {"params": detect_head, "lr": lr, "name": "detect_head"},
            {"params": lane_head, "lr": lr, "name": "lane_head"},
        ]
        if dual_neck:
            groups.append({"params": dual_neck, "lr": lr * neck_scale, "name": "dual_neck"})
        return groups

    def _get_lr(self, epoch: int) -> float:
        """Warmup + cosine annealing schedule."""
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
        import math
        return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    def _set_lr(self, epoch: int):
        lr = self._get_lr(epoch)
        opt_cfg = self.cfg.get("training", {}).get("optimizer", {})
        backbone_scale = opt_cfg.get("backbone_lr_scale", 0.1)
        neck_scale = opt_cfg.get("neck_lr_scale", 0.5)

        for pg in self.optimizer.param_groups:
            name = pg.get("name", "")
            if name == "backbone":
                pg["lr"] = lr * backbone_scale
            elif name == "dual_neck":
                pg["lr"] = lr * neck_scale
            elif name == "mt_weights":
                pg["lr"] = lr
            else:
                pg["lr"] = lr

    def _det_preservation_loss(self, outputs: dict, images: torch.Tensor) -> torch.Tensor:
        """Compute detection preservation regularization loss.

        For teacher mode: only distill features in spatial regions where the
        teacher detects relevant classes (COCO classes that overlap with BDD100K).
        This avoids forcing the student to preserve features for irrelevant COCO
        classes (e.g., banana, airplane) that would hurt BDD100K performance.
        """
        if self.det_preservation_mode == "teacher" and self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_out = self.teacher_model(images)

            # Get teacher detections to build spatial masks
            spatial_masks = self._build_teacher_spatial_masks(
                teacher_out, images.shape)

            # Masked feature distillation — only in relevant regions
            loss = torch.tensor(0.0, device=self.device)
            n_feats = 0
            for i, (sf, tf) in enumerate(zip(
                    outputs["det_features"], teacher_out["det_features"])):
                if sf.shape != tf.shape:
                    continue
                mask = spatial_masks[i] if i < len(spatial_masks) else None
                if mask is not None and mask.any():
                    # Expand mask to match feature channels: (B, 1, H, W) -> (B, C, H, W)
                    mask_expanded = mask.unsqueeze(1).expand_as(sf)
                    diff = (sf - tf) ** 2
                    loss = loss + (diff * mask_expanded).sum() / mask_expanded.sum().clamp(min=1.0)
                elif mask is None:
                    # No NMS available — fall back to full MSE
                    loss = loss + F.mse_loss(sf, tf)
                # If mask exists but is all-False, skip (no relevant objects)
                n_feats += 1

            return self.teacher_weight * loss if n_feats > 0 else torch.tensor(0.0, device=self.device)

        elif self.det_preservation_mode == "l2_reg" and self._init_weights_snapshot:
            loss = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if name in self._init_weights_snapshot:
                    loss = loss + F.mse_loss(param, self._init_weights_snapshot[name])
            return self.l2_weight * loss

        return torch.tensor(0.0, device=self.device)

    def _build_teacher_spatial_masks(self, teacher_out: dict,
                                     img_shape: torch.Size) -> list:
        """Build per-scale boolean masks from teacher's relevant-class detections.

        Returns a list of (B, H_feat, W_feat) boolean tensors, one per det_feature.
        """
        try:
            try:
                from ultralytics.utils.nms import non_max_suppression
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression

            det_out = teacher_out["det_output"]
            if isinstance(det_out, (tuple, list)):
                det_out = det_out[0]

            preds = non_max_suppression(
                det_out, conf_thres=self.teacher_conf_thresh,
                iou_thres=0.45, max_det=100)
        except Exception:
            # If teacher NMS fails, return None masks (fallback to full MSE)
            return [None] * len(teacher_out.get("det_features", []))

        B = img_shape[0]
        _, _, img_h, img_w = img_shape
        relevant = set(self.teacher_relevant_classes)

        masks = []
        for feat in teacher_out["det_features"]:
            _, _, fh, fw = feat.shape
            mask = torch.zeros(B, fh, fw, dtype=torch.bool, device=feat.device)

            for bi in range(B):
                boxes = preds[bi]  # (N, 6): x1, y1, x2, y2, conf, cls
                if len(boxes) == 0:
                    continue
                for box in boxes:
                    cls_id = int(box[5].item())
                    if cls_id not in relevant:
                        continue
                    # Project box coords to feature map scale
                    x1 = max(0, int(box[0].item() / img_w * fw))
                    y1 = max(0, int(box[1].item() / img_h * fh))
                    x2 = min(fw, int(box[2].item() / img_w * fw) + 1)
                    y2 = min(fh, int(box[3].item() / img_h * fh) + 1)
                    # Pad by 1 cell for context
                    x1 = max(0, x1 - 1)
                    y1 = max(0, y1 - 1)
                    x2 = min(fw, x2 + 1)
                    y2 = min(fh, y2 + 1)
                    mask[bi, y1:y2, x1:x2] = True

            masks.append(mask)

        return masks

    # ── Detection output decoding ────────────────────────────────────────

    def _decode_det_output(self, det_out, batch_size: int) -> list:
        """Decode detect head output into a list of per-image prediction tensors.

        Handles all YOLO output formats:
        - Eval end2end: tuple (y_postprocessed, preds_dict)
        - Eval standard: single tensor for NMS
        - Train dict: {"one2many": ..., "one2one": ...}
        """
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression

        # Case 1: tuple (y_postprocessed, preds_dict) from end2end eval
        if isinstance(det_out, (tuple, list)):
            # Check if first element is postprocessed predictions
            first = det_out[0]
            if isinstance(first, torch.Tensor):
                if first.dim() == 3 and first.shape[-1] == 6:
                    # Already postprocessed [B, max_det, 6]
                    preds = []
                    for bi in range(first.shape[0]):
                        valid = first[bi][:, 4] > self.conf_thresh
                        preds.append(first[bi][valid])
                    return preds
                elif first.dim() >= 2:
                    # Raw predictions — run NMS
                    return non_max_suppression(
                        first, conf_thres=self.conf_thresh,
                        iou_thres=self.nms_iou, max_det=self.max_det)

        # Case 2: single tensor
        if isinstance(det_out, torch.Tensor) and det_out.dim() >= 2:
            return non_max_suppression(
                det_out, conf_thres=self.conf_thresh,
                iou_thres=self.nms_iou, max_det=self.max_det)

        # Case 3: dict (train mode — shouldn't happen in validate, but handle it)
        if isinstance(det_out, dict):
            key = "one2one" if "one2one" in det_out else "one2many"
            raw = det_out[key]
            if isinstance(raw, dict) and "feats" in raw:
                feats = raw["feats"]
                # Manually concat features for NMS
                nc = self.model.detect_head.nc
                reg_max = self.model.detect_head.reg_max
                bs = feats[0].shape[0]
                boxes, scores = [], []
                for xi in feats:
                    b, s = xi.split([4 * reg_max, nc], dim=1)
                    boxes.append(b.reshape(bs, 4 * reg_max, -1))
                    scores.append(s.reshape(bs, nc, -1))
                formatted = torch.cat([
                    torch.cat(boxes, dim=-1),
                    torch.cat(scores, dim=-1),
                ], dim=1).permute(0, 2, 1)
                return non_max_suppression(
                    formatted, conf_thres=self.conf_thresh,
                    iou_thres=self.nms_iou, max_det=self.max_det)

        # Fallback: empty predictions
        return [torch.empty((0, 6))] * batch_size

    # ── Training ──────────────────────────────────────────────────────────

    def _backward(self, loss: torch.Tensor):
        if self.amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _maybe_unscale_and_clip(self) -> torch.Tensor:
        """
        Unscale gradients once when AMP is enabled, then clip. If GradScaler's
        internal state says gradients were already unscaled, do not crash; just
        clip the current grads. This avoids the runtime failure:
        "unscale_() has already been called on this optimizer since the last update()."
        """
        if self.amp and self.scaler is not None:
            try:
                self.scaler.unscale_(self.optimizer)
            except RuntimeError as e:
                if "already been called on this optimizer" not in str(e):
                    raise
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    def _optimizer_step(self):
        if self.amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.use_staged:
            self.mt_strategy.set_epoch(epoch)

        self.model.train()
        totals = {"loss": 0.0, "det_loss": 0.0, "lane_loss": 0.0, "pres_loss": 0.0}
        last_mt_info = {}
        n_batches = 0
        skipped_batches = 0
        nan_det_batches = 0
        nan_lane_batches = 0
        nan_pres_batches = 0
        nan_total_batches = 0
        teacher_disabled_this_epoch = False

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True)
            det_targets = batch["det_targets"].to(self.device)
            lane_masks = batch["lane_masks"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda' if str(self.device).startswith('cuda') else 'cpu', enabled=self.amp):
                outputs = self.model(images)
                det_loss = self.det_loss_fn(outputs["det_output"], det_targets)

            # Compute lane + preservation losses in FP32 for stability
            lane_loss, lane_components = self.lane_loss_fn(
                outputs["lane_logits"].float(), lane_masks.float()
            )
            pres_loss = self._det_preservation_loss(outputs, images)

            # If teacher preservation goes unstable, drop it instead of poisoning the epoch
            if not torch.isfinite(pres_loss):
                nan_pres_batches += 1
                skipped_batches += 1
                if self.det_preservation_mode == "teacher" and self.disable_teacher_on_instability:
                    self.teacher_model = None
                    self.det_preservation_mode = "none"
                    self._teacher_disabled = True
                    teacher_disabled_this_epoch = True
                    print(f"[epoch {epoch+1}] disabled teacher preservation after non-finite pres_loss on batch {batch_idx+1}")
                    pres_loss = torch.tensor(0.0, device=self.device)
                else:
                    continue

            if not torch.isfinite(det_loss):
                nan_det_batches += 1
                skipped_batches += 1
                continue

            if not torch.isfinite(lane_loss):
                nan_lane_batches += 1
                skipped_batches += 1
                continue

            if self.use_pcgrad:
                shared_params = [p for p in self.model.backbone_neck.parameters() if p.requires_grad]
                shared_set = set(id(p) for p in shared_params)

                PCGrad.pcgrad_backward([det_loss, lane_loss], shared_params, self.scaler if self.amp else None)
                saved_grads = {id(p): p.grad.clone() for p in shared_params if p.grad is not None}

                self.optimizer.zero_grad(set_to_none=True)
                combined = det_loss + lane_loss + pres_loss
                if not torch.isfinite(combined):
                    nan_total_batches += 1
                    skipped_batches += 1
                    continue

                if self.amp:
                    self._backward(combined)

                if self.amp:
                    inv_scale = 1.0 / self.scaler.get_scale()
                for p in self.model.parameters():
                    if id(p) in shared_set and id(p) in saved_grads:
                        p.grad = saved_grads[id(p)]
                    elif id(p) not in shared_set and p.grad is not None and self.amp:
                        p.grad.mul_(inv_scale)
                total = combined
            else:
                total, mt_info = self.mt_strategy(det_loss, lane_loss)
                total = total + pres_loss
                last_mt_info = mt_info

                if not torch.isfinite(total):
                    nan_total_batches += 1
                    skipped_batches += 1
                    continue

                self._backward(total)

            grad_norm = self._maybe_unscale_and_clip()
            if not torch.isfinite(grad_norm):
                nan_total_batches += 1
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.use_pcgrad:
                self.optimizer.step()
                if self.amp and self.scaler is not None:
                    self.scaler.update()
            else:
                self._optimizer_step()

            if self.ema:
                self.ema.update(self.model)

            totals["loss"] += float(total.detach().item())
            totals["det_loss"] += float(det_loss.detach().item())
            totals["lane_loss"] += float(lane_loss.detach().item())
            totals["pres_loss"] += float(pres_loss.detach().item())
            n_batches += 1

            processed = n_batches + skipped_batches
            if processed > 0 and (
                skipped_batches >= self.max_skip_batches or
                skipped_batches / processed > self.max_skip_ratio
            ):
                raise RuntimeError(
                    f"Training became numerically unstable at epoch {epoch+1}: "
                    f"skipped {skipped_batches}/{processed} batches "
                    f"(det={nan_det_batches}, lane={nan_lane_batches}, pres={nan_pres_batches}, total={nan_total_batches})."
                )

        result = {k: v / max(1, n_batches) for k, v in totals.items()}
        if last_mt_info:
            result["det_weight"] = last_mt_info.get("det_weight", 1.0)
            result["lane_weight"] = last_mt_info.get("lane_weight", 0.5)
        result["skipped_batches"] = skipped_batches
        result["good_batches"] = n_batches
        result["teacher_disabled"] = 1.0 if teacher_disabled_this_epoch else 0.0
        return result

    # ── Validation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        # Use EMA weights for validation if available
        if self.ema:
            self.ema.apply(self.model)

        self.lane_metrics.reset()
        self.det_metrics.reset()

        val_loss_total = 0.0
        val_det_total = 0.0
        val_lane_total = 0.0
        n_batches = 0
        total_preds = 0
        total_conf = 0.0
        total_images = 0

        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            det_targets = batch["det_targets"].to(self.device)
            lane_masks = batch["lane_masks"].to(self.device)

            # ── Forward in TRAIN mode for loss (detect head returns raw features) ──
            self.model.train()
            with torch.amp.autocast('cuda' if str(self.device).startswith('cuda') else 'cpu', enabled=self.amp):
                outputs_train = self.model(images)
                det_loss = self.det_loss_fn(outputs_train["det_output"], det_targets)
                lane_loss, _ = self.lane_loss_fn(outputs_train["lane_logits"], lane_masks)

            val_det_total += det_loss.item()
            val_lane_total += lane_loss.item()
            val_loss_total += det_loss.item() + lane_loss.item()
            n_batches += 1

            # Lane metrics (logits are the same in train/eval mode)
            self.lane_metrics.update(outputs_train["lane_logits"], lane_masks)

            # ── Forward in EVAL mode for detection metrics (detect head decodes boxes) ──
            self.model.eval()
            with torch.amp.autocast('cuda' if str(self.device).startswith('cuda') else 'cpu', enabled=self.amp):
                outputs_eval = self.model(images)

            det_out = outputs_eval["det_output"]

            # Handle YOLO26/v11 end2end dual-branch output format:
            # - Train mode: dict {"one2many": ..., "one2one": ...}
            # - Eval mode:  tuple (y_postprocessed, preds_dict)
            #   where y_postprocessed is [B, max_det, 6] (x1,y1,x2,y2,conf,cls)
            preds = self._decode_det_output(det_out, images.shape[0])

            for bi in range(images.shape[0]):
                pred_boxes = preds[bi].cpu() if preds[bi].dim() > 0 else torch.empty((0, 6))
                # Track detection diagnostics
                total_images += 1
                n_preds = pred_boxes.shape[0] if pred_boxes.dim() > 0 else 0
                total_preds += n_preds
                if n_preds > 0:
                    total_conf += pred_boxes[:, 4].sum().item()
                # Filter predictions to valid vehicle classes (0-4)
                pred_boxes = remap_preds_to_vehicle_names(pred_boxes)

                # Get GT for this image
                if det_targets.shape[0] > 0:
                    img_mask = det_targets[:, 0] == bi
                    img_gt = det_targets[img_mask]
                    if img_gt.shape[0] > 0:
                        gt_cls = img_gt[:, 1].long()
                        gt_xywh = img_gt[:, 2:6]
                        # Convert YOLO normalized to pixel coords
                        _, _, h, w = images.shape
                        gt_boxes = torch.zeros(gt_xywh.shape[0], 4)
                        gt_boxes[:, 0] = (gt_xywh[:, 0] - gt_xywh[:, 2] / 2) * w
                        gt_boxes[:, 1] = (gt_xywh[:, 1] - gt_xywh[:, 3] / 2) * h
                        gt_boxes[:, 2] = (gt_xywh[:, 0] + gt_xywh[:, 2] / 2) * w
                        gt_boxes[:, 3] = (gt_xywh[:, 1] + gt_xywh[:, 3] / 2) * h
                    else:
                        gt_boxes = torch.empty((0, 4))
                        gt_cls = torch.empty((0,), dtype=torch.int64)
                else:
                    gt_boxes = torch.empty((0, 4))
                    gt_cls = torch.empty((0,), dtype=torch.int64)

                self.det_metrics.update(pred_boxes, gt_boxes, gt_cls)

        # Compute final metrics
        lane_results = self.lane_metrics.compute()
        det_results = self.det_metrics.compute()

        results = {
            "val_loss": val_loss_total / max(1, n_batches),
            "val_det_loss": val_det_total / max(1, n_batches),
            "val_lane_loss": val_lane_total / max(1, n_batches),
        }
        results.update(lane_results)
        results.update(det_results)

        # Detection diagnostics
        results["avg_preds_per_image"] = total_preds / max(1, total_images)
        results["avg_conf"] = total_conf / max(1, total_preds)

        # Restore non-EMA weights for continued training
        if self.ema:
            self.ema.restore(self.model)

        return results

    # ── Main loop ─────────────────────────────────────────────────────────

    def train(self, epochs: Optional[int] = None) -> list:
        epochs = epochs or self.epochs
        print(f"\n{'=' * 65}")
        print(f"  Joint Training: {epochs} epochs — {self.cfg.get('run_name', '?')}")
        print(f"  Multi-task strategy: {self.cfg.get('loss', {}).get('multitask', {}).get('strategy', 'fixed')}")
        print(f"  Det preservation: {self.det_preservation_mode}")
        print(f"  Save dir: {self.save_dir}")
        print(f"{'=' * 65}")

        # Dump config
        config_path = os.path.join(self.save_dir, "config.yaml")
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.cfg, f, default_flow_style=False)
        except ImportError:
            with open(config_path.replace('.yaml', '.json'), 'w') as f:
                json.dump(self.cfg, f, indent=2)

        start_time = time.time()

        for epoch in range(epochs):
            self._set_lr(epoch)
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate with real metrics
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # Compile epoch record
            record = {"epoch": epoch + 1, "time": epoch_time}
            record.update({f"train_{k}": v for k, v in train_metrics.items()})
            record.update(val_metrics)
            record["lr"] = self.optimizer.param_groups[0]["lr"]
            self.history.append(record)

            # Print summary
            det_map = val_metrics.get("det_map50", 0)
            lane_iou = val_metrics.get("lane_miou", 0)
            lane_f1 = val_metrics.get("lane_f1", 0)
            lane_best_f1 = val_metrics.get("lane_best_f1", 0)
            lane_best_t = val_metrics.get("lane_best_thresh", 0.5)
            val_det = val_metrics.get("val_det_loss", 0)
            val_lane = val_metrics.get("val_lane_loss", 0)
            skipped = int(train_metrics.get("skipped_batches", 0))
            lane_w = train_metrics.get("lane_weight", None)
            lane_w_str = f" | w_lane={lane_w:.3f}" if lane_w is not None else ""
            skip_str = f" | skipped={skipped}" if skipped > 0 else ""
            print(
                f"Epoch {epoch+1:>3}/{epochs} | "
                f"train: det={train_metrics['det_loss']:.3f} lane={train_metrics['lane_loss']:.3f} | "
                f"val: det={val_det:.3f} lane={val_lane:.3f} | "
                f"mAP50={det_map*100:.1f}% mIoU={lane_iou*100:.1f}% "
                f"F1={lane_f1*100:.1f}% (best@{lane_best_t}={lane_best_f1*100:.1f}%)"
                f"{lane_w_str}{skip_str} | "
                f"{epoch_time:.0f}s"
            )

            # ── Metric-driven checkpointing ───────────────────────────────
            # Best detection — prefer mAP@50 for checkpoint selection because
            # it is much more stable than mAP@50-95 in our current joint setup.
            det_score = max(0.0, float(val_metrics.get("det_map50", 0) or 0))
            if det_score > self.best_scores["det"]:
                self.best_scores["det"] = det_score
                self._save_checkpoint(epoch + 1, "best_det", val_metrics)

            # Best lane
            lane_score = val_metrics.get("lane_miou", 0)
            if lane_score > self.best_scores["lane"]:
                self.best_scores["lane"] = lane_score
                self._save_checkpoint(epoch + 1, "best_lane", val_metrics)

            # Best joint (weighted combination of task metrics)
            joint_score = sum(
                max(0.0, float(val_metrics.get(k, 0) or 0)) * w
                for k, w in self.joint_score_weights.items()
            )
            record["joint_score"] = joint_score
            if joint_score > self.best_scores["joint"]:
                self.best_scores["joint"] = joint_score
                self._save_checkpoint(epoch + 1, "best_joint", val_metrics)

            # Periodic + last
            if (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(epoch + 1, f"epoch_{epoch+1}", val_metrics)

        # Save last
        self._save_checkpoint(epochs, "last", val_metrics if val_metrics else {})

        total_time = time.time() - start_time

        # Save history CSV
        self._save_history_csv()

        # Save run summary
        summary = {
            "run_name": self.cfg.get("run_name", "?"),
            "epochs": epochs,
            "total_time_min": total_time / 60,
            "best_scores": self.best_scores,
            "final_metrics": self.history[-1] if self.history else {},
        }
        with open(os.path.join(self.save_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 65}")
        print(f"  Training Complete — {total_time/60:.1f} min")
        print(f"  Best det mAP@50:    {self.best_scores['det']*100:.2f}%")
        print(f"  Best lane mIoU:     {self.best_scores['lane']*100:.2f}%")
        print(f"  Best joint score:   {self.best_scores['joint']:.4f}")
        print(f"{'=' * 65}")

        return self.history

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, name: str, metrics: dict):
        path = os.path.join(self.save_dir, "weights", f"{name}.pt")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_scores": self.best_scores,
        }
        if hasattr(self.model, '_arch_config'):
            save_dict["arch_config"] = self.model._arch_config
        if self.ema:
            save_dict["ema_state_dict"] = self.ema.state_dict()
        if hasattr(self.mt_strategy, 'state_dict'):
            save_dict["mt_strategy_state_dict"] = self.mt_strategy.state_dict()

        torch.save(save_dict, path)

    def _save_history_csv(self):
        if not self.history:
            return
        csv_path = os.path.join(self.save_dir, "history.csv")
        keys = list(self.history[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt and self.ema:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        if "best_scores" in ckpt:
            self.best_scores = ckpt["best_scores"]
        return ckpt.get("epoch", 0)
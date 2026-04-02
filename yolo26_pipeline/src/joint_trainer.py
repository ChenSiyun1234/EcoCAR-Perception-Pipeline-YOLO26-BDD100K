"""
joint_trainer.py — Custom training loop for joint detection + lane segmentation.

Uses the YOLO26 detection loss internally (via Ultralytics internals) and adds
a lane segmentation loss (BCE + Dice). Combined: α·det_loss + β·lane_loss.

Used by:
  - 08_joint_training.ipynb
"""

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


# ── Lane Loss ────────────────────────────────────────────────────────────────

class LaneLoss(nn.Module):
    """
    Combined BCE + Dice loss for binary lane segmentation.

    The Dice component helps with class imbalance (lanes are thin lines
    occupying a small fraction of the image).
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # Sparse-Positive Fix: Penalize lane pixel misses 10x harder.
        # Lane pixels only occupy ~3-5% of an image. Without this, the network conservatively outputs zero.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits:  (B, 1, H, W) raw logits from lane head.
            targets: (B, 1, H, W) binary ground truth.
        """
        bce_loss = self.bce(logits, targets)

        # Dice loss (cast to FP32 to prevent FP16 max-value overflow since H*W = 57600)
        probs = torch.sigmoid(logits.float())
        targets_f32 = targets.float()
        smooth = 1e-6
        intersection = (probs * targets_f32).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_f32.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice.mean()

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss


# ── Joint Trainer ────────────────────────────────────────────────────────────

class JointTrainer:
    """
    Custom training loop for multi-task YOLO26 (detection + lane segmentation).

    Since Ultralytics' .train() only supports built-in tasks, this trainer
    handles the forward pass, loss computation, and optimisation manually.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        det_loss_weight: float = 1.0,       # Stage 2 primary
        lane_loss_weight: float = 0.5,      # Stage 2 primary
        stage1_epochs: int = 0,             # Epochs for Detection-Prioritized Warmup
        stage2_epochs: int = 0,             # Epochs for Smooth Transition Ramp
        stage1_det_weight: float = 1.0,
        stage1_lane_weight: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        amp: bool = True,
        save_dir: str = "runs/joint",
        save_period: int = 5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.amp = amp
        self.save_dir = save_dir
        self.save_period = save_period
        self.base_lr = lr

        # Training Stages Configuration
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage1_det_weight = stage1_det_weight
        self.stage1_lane_weight = stage1_lane_weight
        self.stage2_det_weight = det_loss_weight    # Target final weights
        self.stage2_lane_weight = lane_loss_weight  # Target final weights
        
        # Initial active weights
        self.det_loss_weight = stage1_det_weight if stage1_epochs > 0 else det_loss_weight
        self.lane_loss_weight = stage1_lane_weight if stage1_epochs > 0 else lane_loss_weight
        self.current_stage_name = "Initialization"
        self.invalid_batch_count = 0

        # Losses
        self.lane_criterion = LaneLoss().to(device)

        # Optimiser — different LR for backbone/neck vs heads vs interaction blocks
        backbone_params    = []
        dual_neck_params   = []   # DualTaskNeck interaction blocks — mid LR
        lane_head_params   = []
        detect_head_params = []

        for name, param in model.named_parameters():
            if "lane_head" in name:
                lane_head_params.append(param)
            elif "detect_head" in name:
                detect_head_params.append(param)
            elif "dual_neck" in name:
                dual_neck_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params,    "lr": lr * 0.1, "name": "backbone"},
            {"params": detect_head_params, "lr": lr,       "name": "detect_head"},
            {"params": lane_head_params,   "lr": lr,       "name": "lane_head"},
        ]
        if dual_neck_params:
            # Interaction blocks are new parameters — train at mid LR (0.5×)
            # so they adapt faster than the frozen backbone but don't destabilise heads
            param_groups.append(
                {"params": dual_neck_params, "lr": lr * 0.5, "name": "dual_neck"}
            )

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )

        self.scaler = GradScaler(enabled=amp)

        # Logging
        self.history = {
            "train_loss": [], "train_det_loss": [], "train_lane_loss": [],
            "train_bce": [], "train_dice": [], "train_logit_mean": [], "train_pred_coverage": [],
            "val_loss": [], "val_det_loss": [], "val_lane_loss": [],
        }

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)

        has_dual = bool(dual_neck_params)
        print(f"JointTrainer ready:")
        print(f"  Device        : {device}")
        print(f"  AMP           : {amp}")
        print(f"  Staged training: {'Yes' if stage1_epochs > 0 else 'No'} ({stage1_epochs} stage-1 epochs)")
        print(f"  Backbone LR   : {lr * 0.1:.2e}")
        print(f"  Detect head LR: {lr:.2e}")
        print(f"  Lane head LR  : {lr:.2e}")
        if has_dual:
            print(f"  Dual neck LR  : {lr * 0.5:.2e}")
        print(f"  Save dir      : {save_dir}")

    def _compute_det_loss(
        self, det_output: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute detection loss using Ultralytics native v8DetectionLoss.
        
        This avoids catastrophic forgetting by properly decoding the YOLOv8 
        detection head grid and calculating Box, Cls, and DFL losses exactly 
        as the native engine would.
        """
        if targets.shape[0] == 0:
            dev = det_output[0].device if isinstance(det_output, list) else det_output.device
            return torch.tensor(0.0, device=dev, requires_grad=True)

        # Lazy initialize the native YOLO loss
        if not hasattr(self, 'native_loss'):
            if not hasattr(self.model, 'model'):
                self.model.model = self.model.model_layers
                
            # Polyfill for AttributeError: 'dict' object has no attribute 'box'
            # Native loss requires deep property lookup from args that was lost when serialized
            if hasattr(self.model, 'args') and isinstance(self.model.args, dict):
                class ObjectDict(dict):
                    def __getattr__(self, key):
                        return self.get(key, 1.0)
                    def __setattr__(self, key, value):
                        self[key] = value
                self.model.args = ObjectDict(self.model.args)
                
            loss_cls = None
            
            # Detect YOLO11 architecture dynamically based on standard dictionary outputs
            if isinstance(det_output, dict) and "one2many" in det_output:
                try:
                    from ultralytics.utils.loss import v11DetectionLoss
                    loss_cls = v11DetectionLoss
                except ImportError:
                    pass
            
            # Fallback to YOLOv8 loss
            if loss_cls is None:
                try:
                    from ultralytics.utils.loss import v8DetectionLoss
                    loss_cls = v8DetectionLoss
                except ImportError:
                    from ultralytics.models.yolo.detect.train import v8DetectionLoss
                    loss_cls = v8DetectionLoss
                    
            self.native_loss = loss_cls(self.model)
            
        # Format targets into the dictionary format expected by native losses
        # targets is shape [N, 6] -> (batch_idx, cls, x, y, w, h)
        ultralytics_batch = {
            'batch_idx': targets[:, 0],
            'cls': targets[:, 1:2],
            'bboxes': targets[:, 2:]
        }
        
        # Native YOLO Loss returns (total_loss, [box_loss, cls_loss, dfl_loss])
        try:
            # Standard execution (works natively for YOLOv8 and well-mapped environments)
            loss, _ = self.native_loss(det_output, ultralytics_batch)
        except (KeyError, TypeError) as e:
            # ----------------- FAILSAFE PROTECTIONS -----------------
            # Extract features from tuple if necessary (Validation mode returns (inference_output, raw_features))
            raw_feats = det_output[1] if isinstance(det_output, tuple) else det_output
            
            # Condition A: YOLO11 architecture dict but `v11DetectionLoss` was missing. 
            # `v8DetectionLoss` cannot read `{"one2many": ...}`. We extract the compatible branch.
            if isinstance(raw_feats, dict) and "one2many" in raw_feats:
                loss, _ = self.native_loss(raw_feats["one2many"], ultralytics_batch)
            
            # Condition B: Older YOLO architecture returns raw feature lists `[P3, P4, P5]`, 
            # but newer Ultralytics (8.3+) `v8DetectionLoss` strictly expects a formatted dict.
            elif isinstance(raw_feats, (list, tuple)):
                bs = raw_feats[0].shape[0]
                nc = self.model.detect_head.nc
                reg_max = self.model.detect_head.reg_max
                
                boxes, scores = [], []
                for xi in raw_feats:
                    b, s = xi.split([4 * reg_max, nc], dim=1)
                    boxes.append(b.reshape(bs, 4 * reg_max, -1))
                    scores.append(s.reshape(bs, nc, -1))
                    
                formatted_preds = {
                    "boxes": torch.cat(boxes, dim=-1),
                    "scores": torch.cat(scores, dim=-1),
                    "feats": raw_feats
                }
                loss, _ = self.native_loss(formatted_preds, ultralytics_batch)
            else:
                raise e
        
        # YOLO native loss functions return a vector of components (e.g., [box_loss, cls_loss, dfl_loss]).
        # PyTorch requires a single scalar value for .backward().
        if hasattr(loss, "sum") and loss.numel() > 1:
            loss = loss.sum()
            
        return loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_det_loss = 0.0
        total_lane_loss = 0.0
        
        # Diagnostics
        total_bce = 0.0
        total_dice = 0.0
        total_logit_mean = 0.0
        total_logit_max = 0.0
        total_logit_min = 0.0
        total_pred_coverage = 0.0
        
        n_batches = 0

        for batch in self.train_loader:
            images = batch["images"].to(self.device)
            det_targets = batch["det_targets"].to(self.device)
            lane_masks = batch["lane_masks"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.amp):
                # Forward pass
                outputs = self.model(images)

                # Detection loss
                det_loss = self._compute_det_loss(outputs["det_output"], det_targets)

                # Lane segmentation loss
                lane_logits = outputs["lane_logits"]
                lane_loss, bce_loss, dice_loss = self.lane_criterion(lane_logits, lane_masks)

                # Combined loss
                loss = (self.det_loss_weight * det_loss +
                        self.lane_loss_weight * lane_loss)

            # ----------------- FAILSAFE PROTECTIONS -----------------
            if torch.isnan(loss) or torch.isinf(loss):
                self.invalid_batch_count += 1
                print(f"\\n[!] 🚨 INVALID LOSS DETECTED (Batch {n_batches+1})")
                print(f"    Epoch: {epoch} | {self.current_stage_name}")
                print(f"    Det Loss: {det_loss.item():.4f} (Weight: {self.det_loss_weight:.3f})")
                print(f"    Lane Loss: {lane_loss.item():.4f} (Weight: {self.lane_loss_weight:.3f})")
                print(f"    Total Loss: {loss.item():.4f}")
                
                if "lane_logits" in locals() or "lane_logits" in outputs:
                    logits_f32 = lane_logits.detach().float()
                    print(f"    Lane Logits -> Min: {logits_f32.min().item():.3f}, Max: {logits_f32.max().item():.3f}, Mean: {logits_f32.mean().item():.3f}")
                
                if self.invalid_batch_count > 10:
                    raise RuntimeError("🚨 Hard Stop: Over 10 invalid batches detected. Aborting to prevent model corruption.")
                
                print("    ⚠️ Safely skipping this batch to protect network weights.\\n")
                self.optimizer.zero_grad()
                continue

            # Backward + step with Gradient Clipping
            self.scaler.scale(loss).backward()
            
            # Unscale before clipping to ensure threshold accuracy
            self.scaler.unscale_(self.optimizer)
            clipped_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            if torch.isnan(clipped_norm) or torch.isinf(clipped_norm):
                print(f"    ⚠️ Invalid gradient norm ({clipped_norm.item()}). Auto-skipping optimizer step via Scaler.\\n")
                
            # scaler.step() intrinsically checks for infs/NaNs and safely skips optimizer.step() if found.
            self.scaler.step(self.optimizer)
            
            # scaler.update() adjusts the scale factor based on whether the step was skipped or not. 
            self.scaler.update()

            total_loss += loss.item()
            total_det_loss += det_loss.item()
            total_lane_loss += lane_loss.item()
            
            with torch.no_grad():
                total_bce += bce_loss.item()
                total_dice += dice_loss.item()
                logits_f32 = lane_logits.detach().float()
                total_logit_mean += logits_f32.mean().item()
                total_logit_max += logits_f32.max().item()
                total_logit_min += logits_f32.min().item()
                total_pred_coverage += (torch.sigmoid(logits_f32) > 0.5).float().mean().item()
                
            n_batches += 1

        self.scheduler.step()

        # Compile epoch averages
        avg = {
            "loss": total_loss / max(1, n_batches),
            "det_loss": total_det_loss / max(1, n_batches),
            "lane_loss": total_lane_loss / max(1, n_batches),
            "bce_loss": total_bce / max(1, n_batches),
            "dice_loss": total_dice / max(1, n_batches),
            "logit_mean": total_logit_mean / max(1, n_batches),
            "logit_max": total_logit_max / max(1, n_batches),
            "logit_min": total_logit_min / max(1, n_batches),
            "pred_coverage": total_pred_coverage / max(1, n_batches),
        }

        self.history["train_loss"].append(avg["loss"])
        self.history["train_det_loss"].append(avg["det_loss"])
        self.history["train_lane_loss"].append(avg["lane_loss"])
        self.history["train_bce"].append(avg["bce_loss"])
        self.history["train_dice"].append(avg["dice_loss"])
        self.history["train_logit_mean"].append(avg["logit_mean"])
        self.history["train_pred_coverage"].append(avg["pred_coverage"])

        return avg

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {"loss": 0, "det_loss": 0, "lane_loss": 0}

        self.model.eval()

        total_loss = 0.0
        total_det_loss = 0.0
        total_lane_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            det_targets = batch["det_targets"].to(self.device)
            lane_masks = batch["lane_masks"].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.amp):
                outputs = self.model(images)
                det_loss = self._compute_det_loss(outputs["det_output"], det_targets)
                lane_loss, _, _ = self.lane_criterion(outputs["lane_logits"], lane_masks)
                loss = (self.det_loss_weight * det_loss +
                        self.lane_loss_weight * lane_loss)

            total_loss += loss.item()
            total_det_loss += det_loss.item()
            total_lane_loss += lane_loss.item()
            n_batches += 1

        avg = {
            "loss": total_loss / max(1, n_batches),
            "det_loss": total_det_loss / max(1, n_batches),
            "lane_loss": total_lane_loss / max(1, n_batches),
        }

        self.history["val_loss"].append(avg["loss"])
        self.history["val_det_loss"].append(avg["det_loss"])
        self.history["val_lane_loss"].append(avg["lane_loss"])

        return avg

    def train(self, epochs: int = 10) -> Dict:
        """
        Full training loop with YOLOP-inspired staged training support.

        Args:
            epochs: Number of epochs to train.

        Returns:
            Training history dict.
        """
        print(f"\n{'='*60}")
        print(f" Starting Joint Training — {epochs} epochs")
        print(f"{'='*60}")

        best_val_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # --- Staged Training Logic (YOLOP / HybridNets style) ---
            if epoch <= self.stage1_epochs:
                self.det_loss_weight = self.stage1_det_weight
                self.lane_loss_weight = self.stage1_lane_weight
                self.current_stage_name = "Stage 1: Detection-Prioritized Warmup"
                
                # Throttle lane branch during detection-prioritized warmup
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] in ("lane_head", "dual_neck"):
                        param_group["lr"] = self.base_lr * 0.01  # Severely throttle lane/interaction

            elif epoch <= self.stage1_epochs + self.stage2_epochs:
                # Smooth Transition Ramp
                progress = (epoch - self.stage1_epochs) / max(1, self.stage2_epochs)
                self.det_loss_weight = self.stage1_det_weight + progress * (self.stage2_det_weight - self.stage1_det_weight)
                self.lane_loss_weight = self.stage1_lane_weight + progress * (self.stage2_lane_weight - self.stage1_lane_weight)
                self.current_stage_name = f"Stage 2: Smooth Transition Ramp ({progress*100:.0f}%)"

                # Restore learning rates smoothly
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] == "lane_head":
                        param_group["lr"] = self.base_lr * (0.1 + 0.9 * progress)
                    elif param_group["name"] == "dual_neck":
                        param_group["lr"] = self.base_lr * 0.5 * (0.1 + 0.9 * progress)
            else:
                self.det_loss_weight = self.stage2_det_weight
                self.lane_loss_weight = self.stage2_lane_weight
                self.current_stage_name = "Stage 3: Balanced Joint Pipeline"
                
                # Full learning rate restored
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] == "lane_head":
                        param_group["lr"] = self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else self.base_lr

            # Print stage banner whenever we enter a new discrete phase or are ramping
            print(f"\\n{'*'*60}")
            print(f" 🔀 {self.current_stage_name.upper()}")
            print(f"    ↳ Det Weight:  {self.det_loss_weight:.3f} | Lane Weight: {self.lane_loss_weight:.3f}")
            print(f"{'*'*60}\\n")
            # -------------------------------------------

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # Print progress
            print(
                f"Epoch {epoch:>3}/{epochs} │ "
                f"train: {train_metrics['loss']:.4f} "
                f"(det={train_metrics['det_loss']:.4f}, lane={train_metrics['lane_loss']:.4f} "
                f"[bce={train_metrics['bce_loss']:.4f}, dice={train_metrics['dice_loss']:.4f}]) │ "
                f"val: {val_metrics['loss']:.4f} "
                f"(det={val_metrics['det_loss']:.4f}, lane={val_metrics['lane_loss']:.4f}) │ "
                f"cov: {train_metrics['pred_coverage']*100:.1f}% │ "
                f"{epoch_time:.1f}s"
            )

            # Save checkpoint periodically
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, "checkpoint")

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, "best")

        # Save final model
        self._save_checkpoint(epochs, "last")

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f" Training Complete — {total_time/60:.1f} minutes")
        print(f" Best val loss: {best_val_loss:.4f}")
        print(f"{'='*60}")

        return self.history

    def _save_checkpoint(self, epoch: int, name: str) -> str:
        """Save model checkpoint with architecture config for reproducible loading."""
        path = os.path.join(self.save_dir, "weights", f"{name}.pt")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        # Save architecture config so nb09 can rebuild a matching model
        if hasattr(self.model, '_arch_config'):
            save_dict["arch_config"] = self.model._arch_config
        torch.save(save_dict, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint. Returns the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        return ckpt.get("epoch", 0)


def plot_joint_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training curves for joint detection + lane training."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Total loss
    axes[0].plot(history["train_loss"], label="train")
    if history["val_loss"]:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Detection loss
    axes[1].plot(history["train_det_loss"], label="train")
    if history["val_det_loss"]:
        axes[1].plot(history["val_det_loss"], label="val")
    axes[1].set_title("Detection Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Lane loss
    axes[2].plot(history["train_lane_loss"], label="train")
    if history["val_lane_loss"]:
        axes[2].plot(history["val_lane_loss"], label="val")
    axes[2].set_title("Lane Segmentation Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.suptitle("Joint Training History", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"💾 {save_path}")
    plt.show()

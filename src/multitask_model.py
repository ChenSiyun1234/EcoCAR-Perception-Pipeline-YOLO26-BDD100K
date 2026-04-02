"""
multitask_model.py — Vehicle Detection + Lane Segmentation joint model.

Architecture (v4 — simplified, native vehicle classes):
  Input
    -> YOLO26 Backbone+Neck (shared)
         +-- FPN neck [P3, P4, P5]
                |
                +-- [P3, P4, P5]  -> Vehicle Detect Head (nc=5)
                +-- [P3, P4, P5]  -> Transformer Lane Head

Key changes from v3:
  - Native nc=5 detect head (car, truck, bus, motorcycle, bicycle)
  - No COCO class remapping — direct class IDs everywhere
  - No DualTaskNeck — both tasks share FPN features directly
  - No teacher distillation or detection preservation
  - COCO pretrained weight transfer for vehicle channels at init
  - Square mask (160x160) by default for aspect ratio consistency

Used by:
  - train.py, eval.py, infer.py
  - 08_joint_training.ipynb, 09_joint_inference.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.utils.class_map import VEHICLE_CLASSES, VEHICLE_TO_COCO, NUM_VEHICLE_CLASSES


class MultiTaskYOLO(nn.Module):
    """
    Multi-task YOLO26 with native vehicle detection head + transformer lane head.

    Both tasks consume the same shared [P3, P4, P5] features from the YOLO neck.
    No intermediate routing, no gated exchange — maximum simplicity.
    """

    def __init__(
        self,
        yolo_model,
        neck_output_indices: Optional[List[int]] = None,
        lane_head_type: str = "transformer",
        lane_embed_dim: int = 128,
        lane_num_heads: int = 4,
        lane_depth: int = 2,
        lane_hidden_channels: int = 64,
        mask_height: int = 160,
        mask_width: int = 160,
    ):
        super().__init__()

        inner = yolo_model.model
        self.model_layers = inner.model

        self._detect_head_idx = len(self.model_layers) - 1
        self.detect_head = self.model_layers[self._detect_head_idx]

        if neck_output_indices is not None:
            self.neck_output_indices = neck_output_indices
        else:
            self.neck_output_indices = self._find_neck_output_indices()

        self.backbone_neck = nn.ModuleList(
            [self.model_layers[i] for i in range(self._detect_head_idx)]
        )

        # Save original nc and head weights before replacing
        self._orig_nc = self.detect_head.nc
        self._orig_detect_state = {
            k: v.clone() for k, v in self.detect_head.state_dict().items()
        }

        # Replace the detect head with nc=5 (vehicle-only)
        self._replace_detect_head_native(NUM_VEHICLE_CLASSES)

        # Probe layer shapes
        self._all_layer_info = self._probe_all_layers()
        self._neck_channels = [
            self._all_layer_info[i]['ch'] for i in self.neck_output_indices
        ]

        # ── Lane segmentation head ────────────────────────────────────────
        from src.models.lane_heads import build_lane_head
        lane_head_cfg = {
            "type": lane_head_type,
            "embed_dim": lane_embed_dim,
            "num_heads": lane_num_heads,
            "depth": lane_depth,
            "hidden_channels": lane_hidden_channels,
        }
        self.lane_head = build_lane_head(
            lane_head_cfg, self._neck_channels, mask_height, mask_width,
        )

        # Ultralytics internals
        if hasattr(inner, 'args'):
            self.args = inner.args
        if hasattr(inner, 'stride'):
            self.stride = inner.stride

        if hasattr(self.detect_head, 'dynamic'):
            self.detect_head.dynamic = True
        self.train()

        # Architecture config for checkpoint reproducibility
        self._arch_config = {
            'nc': NUM_VEHICLE_CLASSES,
            'classes': VEHICLE_CLASSES,
            'lane_head_type': lane_head_type,
            'lane_embed_dim': lane_embed_dim,
            'lane_num_heads': lane_num_heads,
            'lane_depth': lane_depth,
            'lane_hidden_channels': lane_hidden_channels,
            'mask_height': mask_height,
            'mask_width': mask_width,
            'neck_output_indices': self.neck_output_indices,
            'neck_channels': self._neck_channels,
        }

        # ── Summary ───────────────────────────────────────────────────────
        det_params = sum(p.numel() for p in self.detect_head.parameters())
        lane_params = sum(p.numel() for p in self.lane_head.parameters())
        backbone_params = sum(p.numel() for p in self.backbone_neck.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"MultiTaskYOLO v4 (vehicle-only) built:")
        print(f"  Backbone+Neck    : {self._detect_head_idx} layers, {backbone_params:,} params")
        print(f"  Detect head      : nc={NUM_VEHICLE_CLASSES} ({', '.join(VEHICLE_CLASSES)}), {det_params:,} params")
        print(f"  Lane head        : {lane_head_type}, {lane_params:,} params")
        print(f"  Total params     : {total:,}")
        print(f"  Mask size        : {mask_height}x{mask_width}")
        print(f"  Neck indices     : {self.neck_output_indices}")
        print(f"  Neck channels    : {self._neck_channels}")

    # ── Detect head replacement ──────────────────────────────────────────

    def _replace_detect_head_native(self, nc: int):
        """Replace the COCO-80 detect head with a native nc-class head.

        Copies pretrained weights for vehicle classes from COCO channels.
        Non-vehicle channels are discarded. Backbone/neck weights are untouched.
        """
        head = self.detect_head
        orig_nc = self._orig_nc

        # Find classification conv layers in the detect head.
        # YOLO26/v11 has dual-branch end2end: one2many (cv2/cv3) and
        # one2one (one2one_cv2/one2one_cv3). Must replace ALL branches.
        for attr_name in ['cv3', 'cv2', 'one2one_cv3', 'one2one_cv2']:
            module_list = getattr(head, attr_name, None)
            if module_list is None:
                continue
            for branch in module_list:
                # Each branch is a Sequential ending with a Conv2d
                # The last conv has out_channels = nc (for cv3) or 4*reg_max (for cv2)
                last_conv = self._find_last_conv(branch)
                if last_conv is None:
                    continue
                out_ch = last_conv.out_channels
                if out_ch == orig_nc:
                    # This is a classification conv — replace it
                    new_conv = nn.Conv2d(
                        last_conv.in_channels, nc,
                        last_conv.kernel_size, last_conv.stride,
                        last_conv.padding, bias=last_conv.bias is not None,
                    )
                    # Transfer COCO weights for vehicle classes
                    with torch.no_grad():
                        for veh_id, coco_id in VEHICLE_TO_COCO.items():
                            if coco_id < orig_nc:
                                new_conv.weight[veh_id] = last_conv.weight[coco_id]
                                if new_conv.bias is not None and last_conv.bias is not None:
                                    new_conv.bias[veh_id] = last_conv.bias[coco_id]
                    self._replace_last_conv(branch, new_conv)

        head.nc = nc
        head.no = nc + head.reg_max * 4

    @staticmethod
    def _find_last_conv(module):
        """Find the last Conv2d in a Sequential or nested module."""
        last = None
        if isinstance(module, nn.Sequential):
            for m in module:
                if isinstance(m, nn.Conv2d):
                    last = m
                elif isinstance(m, nn.Sequential):
                    sub = MultiTaskYOLO._find_last_conv(m)
                    if sub is not None:
                        last = sub
                elif hasattr(m, 'conv'):
                    if isinstance(m.conv, nn.Conv2d):
                        last = m.conv
        elif hasattr(module, 'conv'):
            if isinstance(module.conv, nn.Conv2d):
                last = module.conv
        return last

    @staticmethod
    def _replace_last_conv(module, new_conv):
        """Replace the last Conv2d in a Sequential."""
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 1, -1, -1):
                if isinstance(module[i], nn.Conv2d):
                    module[i] = new_conv
                    return True
                elif isinstance(module[i], nn.Sequential):
                    if MultiTaskYOLO._replace_last_conv(module[i], new_conv):
                        return True
                elif hasattr(module[i], 'conv') and isinstance(module[i].conv, nn.Conv2d):
                    module[i].conv = new_conv
                    return True
        return False

    # ── Architecture helpers ──────────────────────────────────────────────

    def _find_neck_output_indices(self) -> List[int]:
        if hasattr(self.detect_head, 'f'):
            from_indices = self.detect_head.f
            if isinstance(from_indices, list):
                return from_indices
        n = self._detect_head_idx
        return [n - 6, n - 3, n - 1]

    def _probe_all_layers(self) -> Dict[int, Dict]:
        info = {}
        try:
            dummy = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                self._forward_backbone_neck(dummy)
            for idx, t in self._layer_outputs.items():
                info[idx] = {'ch': t.shape[1], 'hw': (t.shape[2], t.shape[3])}
        except Exception:
            for i, idx in enumerate(self.neck_output_indices):
                ch = [128, 256, 512][i] if i < 3 else 256
                hw = [(80, 80), (40, 40), (20, 20)][i] if i < 3 else (20, 20)
                info[idx] = {'ch': ch, 'hw': hw}
        return info

    # ── Forward ───────────────────────────────────────────────────────────

    def _forward_backbone_neck(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        self._layer_outputs: Dict[int, torch.Tensor] = {}
        outputs: List[torch.Tensor] = []

        for idx, layer in enumerate(self.backbone_neck):
            f = getattr(layer, 'f', -1)
            if isinstance(f, int):
                x_in = (x if idx == 0 else outputs[-1]) if f == -1 else outputs[f]
            elif isinstance(f, list):
                x_in = [outputs[j] for j in f]
            else:
                x_in = x if idx == 0 else outputs[-1]

            x = layer(x_in)
            outputs.append(x)
            self._layer_outputs[idx] = x

        return x, self._layer_outputs

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        _, layer_outputs = self._forward_backbone_neck(x)

        neck_features = [layer_outputs[idx] for idx in self.neck_output_indices
                         if idx in layer_outputs]

        # Both tasks get the same shared features — no routing needed
        det_output = self.detect_head(neck_features)
        lane_logits = self.lane_head(neck_features)

        return {
            "det_output": det_output,
            "lane_logits": lane_logits,
            "neck_features": neck_features,
        }

    # ── Warm-start ────────────────────────────────────────────────────────

    def warm_start_from_checkpoint(self, ckpt_path: str, device: str = 'cpu') -> int:
        """Load backbone weights from a checkpoint (detection-only or joint).

        Handles our own format, Ultralytics format, and partial loading.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            state_dict = ckpt['model'].state_dict()
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            print(f"Warm-start: unrecognised checkpoint format. Keys: {list(ckpt.keys())[:10]}")
            return 0

        own_state = self.state_dict()
        loaded = 0

        # Try direct key match (our own format)
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                if name.startswith('backbone_neck.'):
                    own_state[name].copy_(param)
                    loaded += 1

        # If no direct matches, try Ultralytics key remapping
        if loaded == 0:
            for name, param in state_dict.items():
                if name.startswith('model.'):
                    parts = name.split('.', 2)
                    if len(parts) >= 3:
                        layer_idx = int(parts[1])
                        rest = parts[2]
                        if layer_idx < self._detect_head_idx:
                            mapped = f"backbone_neck.{layer_idx}.{rest}"
                            if mapped in own_state and own_state[mapped].shape == param.shape:
                                own_state[mapped].copy_(param)
                                loaded += 1

        print(f"Warm-start: loaded {loaded} backbone params from {ckpt_path}")
        return loaded

    # ── Sanity check ──────────────────────────────────────────────────────

    @torch.no_grad()
    def sanity_check(self, img_size: Tuple[int, int] = (640, 640)) -> None:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        dummy = torch.zeros(1, 3, *img_size, device=device)

        print(f"\n{'='*60}")
        print(f"  MultiTaskYOLO v4 Sanity Check")
        print(f"{'='*60}")

        out = self.forward(dummy)

        print(f"\n[Neck features]")
        for i, f in enumerate(out["neck_features"]):
            print(f"  P{i+3}: {tuple(f.shape)}")

        det = out["det_output"]
        print(f"\n[Detection output]")
        if isinstance(det, (list, tuple)):
            for i, d in enumerate(det):
                if isinstance(d, torch.Tensor):
                    print(f"  det_out[{i}]: {tuple(d.shape)}")
        elif isinstance(det, dict):
            for k, v in det.items():
                if isinstance(v, torch.Tensor):
                    print(f"  det_out[{k}]: {tuple(v.shape)}")
        else:
            print(f"  det_out: {type(det)}")

        print(f"\n[Lane output]")
        print(f"  lane_logits: {tuple(out['lane_logits'].shape)}")

        print(f"\n[Config]")
        for k, v in self._arch_config.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")

        if was_training:
            self.train()

    def print_summary(self):
        det_params = sum(p.numel() for p in self.detect_head.parameters())
        lane_params = sum(p.numel() for p in self.lane_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"MultiTaskYOLO v4:")
        print(f"  Detect: nc={NUM_VEHICLE_CLASSES}, {det_params:,} params")
        print(f"  Lane:   {self.lane_head.__class__.__name__}, {lane_params:,} params")
        print(f"  Total:  {total:,} params")


# ── Convenience builder ───────────────────────────────────────────────────

def build_multitask_model(
    cfg: dict = None,
    weights: str = "yolo26s.pt",
    **kwargs,
) -> MultiTaskYOLO:
    """Build MultiTaskYOLO from config dict or keyword arguments."""
    from ultralytics import YOLO

    if cfg is not None:
        model_cfg = cfg.get("model", cfg)
        data_cfg = cfg.get("data", {})
        weights = model_cfg.get("backbone_weights", weights)

        lane_head_cfg = model_cfg.get("lane_head", {})

        build_kwargs = {
            "lane_head_type": lane_head_cfg.get("type", "transformer"),
            "lane_embed_dim": lane_head_cfg.get("embed_dim", 128),
            "lane_num_heads": lane_head_cfg.get("num_heads", 4),
            "lane_depth": lane_head_cfg.get("depth", 2),
            "lane_hidden_channels": lane_head_cfg.get("hidden_channels", 64),
            "mask_height": data_cfg.get("mask_height", 160),
            "mask_width": data_cfg.get("mask_width", 160),
        }
    else:
        build_kwargs = kwargs

    yolo_model = YOLO(weights)
    model = MultiTaskYOLO(yolo_model, **build_kwargs)
    return model

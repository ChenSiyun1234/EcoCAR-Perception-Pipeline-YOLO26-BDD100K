# Training-log findings (2026-04-18)

Observed from the two uploaded logs:

- `training output.txt`
  - epoch 0: `mAP50=0.0011`, `LL_IoU=0.0000`, `Loss=0.6886`
  - epoch 1: `mAP50=0.0620`, `LL_IoU=0.0000`, `Loss=0.6797`
  - epoch 2: `mAP50=0.0001`, `LL_IoU=0.0000`, `Loss=0.7195`
- `training output 2.txt`
  - epoch 0: `mAP50=0.0001`, `LL_IoU=0.0001`, `Loss=0.7120`
  - epoch 1: `mAP50=0.0309`, `LL_IoU=0.0000`, `Loss=0.6706`
  - epoch 2: `mAP50=0.0056`, `LL_IoU=0.0000`, `Loss=0.7038`

Common pattern:

1. detection improves briefly during warmup;
2. lane performance is effectively zero from the start;
3. once learning rate ramps further, total loss rises again and the detector collapses instead of consolidating gains.

This is why the repair set focuses on paper-alignment for the lane branch and on removing stage1 shortcuts that were not actually faithful to YOLOPv2.

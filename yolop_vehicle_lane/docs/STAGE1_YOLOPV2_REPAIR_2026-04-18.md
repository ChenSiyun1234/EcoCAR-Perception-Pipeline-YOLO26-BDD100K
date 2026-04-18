# Stage1 YOLOPv2 repair (2026-04-18)

This patch set targets the repeated stage1 failure mode visible in the two uploaded training logs: detection improves during warmup, then collapses by epoch 2, while lane IoU stays effectively zero from the start.

## Main changes

1. Added a strict `bdd100k_10class` detection protocol so stage1 can actually match the YOLOP / YOLOPv2 paper taxonomy instead of the earlier 1-class merged-vehicle shortcut.
2. Added `stage1/configs/yolopv2_paper_no_da.yaml` as the new default training / eval / export config.
3. Moved the lane decoder tap from layer 16 to layer 17 so it uses the refined end-of-FPN feature rather than the pre-CSP concat tensor.
4. Restored the lane head to a paper-aligned 2-channel bg/fg logit output and switched the lane-loss path back to 2-class focal+dice supervision, because YOLOPv2 writes the lane loss for C=2 categories.
5. Updated the lane loss and validation path so the repaired default is 2-channel/paper-aligned while older 1-channel experiments still load.
6. Switched the strict-paper config to standard cosine annealing (`SGDR: false`) while keeping the 3-epoch warmup and 640x640 train / 640x384 test protocol.

## Important limitation

This repository patch was produced by code inspection only. The dataset is not bundled in the uploaded zip, so I could not rerun training here to empirically verify the repaired configuration.

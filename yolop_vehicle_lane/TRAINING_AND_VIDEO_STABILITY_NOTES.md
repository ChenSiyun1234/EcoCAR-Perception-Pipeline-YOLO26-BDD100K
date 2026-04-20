Stage1/Stage2 training + video stability updates:
- notebook02 now defaults to YOLOP and adds a G4 RTX PRO 6000 stability profile;
- notebook02 uses cfg.TRAIN.END_EPOCH in the loop instead of a hard-coded 200;
- training loop adds optional grad clipping via TRAIN.GRAD_CLIP_NORM;
- video profile now uses rectangular letterbox inference, deterministic colors, higher conf threshold, min-area filtering, and a lightweight IoU tracker with EMA smoothing;
- plot_one_box defaults are deterministic when no color is passed.

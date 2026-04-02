This patch fixes three things the user explicitly requested:
1. Lane JSON search now follows notebook07-style candidate paths and includes raw-data debug cells in notebook 00.
2. Lane loss is now curve-to-curve geometry loss, not rigid point-index distance.
3. Video profiling notebook is now complete: inference, overlay video, lane association, temporal smoothing, CSV exports.

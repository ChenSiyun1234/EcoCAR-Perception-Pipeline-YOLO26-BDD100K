"""
src/models/multitask.py — Re-export from canonical location.

The canonical MultiTaskYOLO lives in src/multitask_model.py.
This module re-exports it so that imports from either path work.
"""

from src.multitask_model import MultiTaskYOLO, build_multitask_model

__all__ = ["MultiTaskYOLO", "build_multitask_model"]

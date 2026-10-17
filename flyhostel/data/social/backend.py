"""
numpy / cupy selection. Kernels in geometry.py take an ``xp`` argument and run unchanged
on either; everything outside the kernels is numpy.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
except ImportError:  # pragma: no cover - depends on the environment
    cp = None


def get_xp(use_gpu=False):
    if use_gpu and cp is None:
        logger.warning("use_gpu=True but cupy is not importable; falling back to numpy")
    return cp if (use_gpu and cp is not None) else np


def to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

"""Feature normalization, calibration, and ablation utilities for fatigue drift.
"""
from typing import Tuple
import numpy as np


def compute_baseline(features: np.ndarray, num_windows: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Compute baseline mean and standard deviation from the first N windows of a session.

    Default 24 windows = 2 minutes of data (at 5 seconds per window).
    """
    n = min(features.shape[0], num_windows)
    baseline_chunk = features[:n]
    mean = baseline_chunk.mean(axis=0)
    std = baseline_chunk.std(axis=0)
    std = np.clip(std, a_min=1e-6, a_max=None)  # Avoid division by zero
    return mean, std


def apply_zscore(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to features using given mean and std."""
    return (features - mean) / std


def normalize_session_features(features: np.ndarray) -> np.ndarray:
    """Normalize session features to zero mean and unit variance."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.clip(std, a_min=1e-6, a_max=None)
    return apply_zscore(features, mean, std)


def ablate_features(features: np.ndarray, group: str) -> np.ndarray:
    """Zero out specific feature groups for ablation study.

    Feature indices:
    - eye: 0 to 5 (EAR mean, EAR std, PERCLOS, blink rate, blink duration, microsleeps)
    - mouth: 6 to 8 (MAR mean, MAR max, yawning)
    - head: 9 to 13 (Pitch mean, Pitch std, Yaw mean, Roll mean, nod frequency)
    - expression: 14 to 17 (AU6 mean, AU12 mean, AU6 std, AU12 std)
    - gaze: 18 to 19 (Gaze X std, Gaze Y std)
    """
    features = features.copy()
    if group == 'eye':
        features[..., 0:6] = 0.0
    elif group == 'mouth':
        features[..., 6:9] = 0.0
    elif group == 'head':
        features[..., 9:14] = 0.0
    elif group == 'expression':
        features[..., 14:18] = 0.0
    elif group == 'gaze':
        features[..., 18:20] = 0.0
    else:
        raise ValueError(f"Unknown ablation group: {group}")
    return features

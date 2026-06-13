"""Window aggregation utilities for 5-second feature windows.
"""
from typing import List
import numpy as np
from .eye_features import detect_blinks, detect_blinks as _detect_blinks


def aggregate_window(frame_buffer: List[dict]) -> np.ndarray:
    """Aggregate list of per-frame feature dicts into 20-dim vector.
    Expects frame_buffer length ~150 (5s @ 30fps).
    Missing keys default to 0.
    """
    ear_vals   = [f.get('ear', 0.0) for f in frame_buffer]
    mar_vals   = [f.get('mar', 0.0) for f in frame_buffer]
    pitch_vals = [f.get('pitch', 0.0) for f in frame_buffer]
    yaw_vals   = [f.get('yaw', 0.0) for f in frame_buffer]
    roll_vals  = [f.get('roll', 0.0) for f in frame_buffer]
    au6_vals   = [f.get('au6', 0.0) for f in frame_buffer]
    au12_vals  = [f.get('au12', 0.0) for f in frame_buffer]
    gaze_x     = [f.get('gaze_x', 0.0) for f in frame_buffer]
    gaze_y     = [f.get('gaze_y', 0.0) for f in frame_buffer]

    blinks = detect_blinks(ear_vals)

    return np.array([
        float(np.mean(ear_vals) if ear_vals else 0.0),
        float(np.std(ear_vals) if ear_vals else 0.0),
        float(sum(1 for e in ear_vals if e < 0.2) / max(1, len(ear_vals))),
        float(len(blinks) * (60/5)),
        float(np.mean([b[2] for b in blinks]) if blinks else 0.0),
        float(0 if not ear_vals else sum(1 for e in ear_vals if e < 0.15)),
        float(np.mean(mar_vals) if mar_vals else 0.0),
        float(np.max(mar_vals) if mar_vals else 0.0),
        float(1 if any(m > 0.5 for m in mar_vals) else 0),
        float(np.mean(pitch_vals) if pitch_vals else 0.0),
        float(np.std(pitch_vals) if pitch_vals else 0.0),
        float(np.mean(yaw_vals) if yaw_vals else 0.0),
        float(np.mean(roll_vals) if roll_vals else 0.0),
        float(0),
        float(np.mean(au6_vals) if au6_vals else 0.0),
        float(np.mean(au12_vals) if au12_vals else 0.0),
        float(np.std(au6_vals) if au6_vals else 0.0),
        float(np.std(au12_vals) if au12_vals else 0.0),
        float(np.std(gaze_x) if gaze_x else 0.0),
        float(np.std(gaze_y) if gaze_y else 0.0),
    ])

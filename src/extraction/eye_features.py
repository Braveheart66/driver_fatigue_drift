"""Pure-Python eye feature utilities used by tests and extraction pipeline.
Avoids heavy deps so unit tests can run in a minimal environment.
"""
import math
from typing import List, Tuple

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(landmarks: List[Point], eye_indices: List[int]) -> float:
    """Compute Eye Aspect Ratio using 6 landmark points.
    `landmarks` is a list-like of (x,y) points; `eye_indices` selects 6 points.
    This follows the Soukupova & Čech formula.
    """
    p = [landmarks[i] for i in eye_indices]
    vertical_1 = _dist(p[1], p[5])
    vertical_2 = _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def detect_blinks(ear_sequence: List[float], threshold: float = 0.2,
                  min_frames: int = 2, max_frames: int = 10, fps: int = 30):
    """Return list of (start_idx, end_idx, duration_ms) for blinks in sequence."""
    blinks = []
    in_blink = False
    start = 0
    for i, ear in enumerate(ear_sequence):
        if ear < threshold and not in_blink:
            in_blink = True
            start = i
        elif ear >= threshold and in_blink:
            in_blink = False
            duration = i - start
            if min_frames <= duration <= max_frames:
                blinks.append((start, i, duration * (1000 / fps)))
    # handle trailing blink
    if in_blink:
        duration = len(ear_sequence) - start
        if min_frames <= duration <= max_frames:
            blinks.append((start, len(ear_sequence), duration * (1000 / fps)))
    return blinks

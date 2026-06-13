"""Process a single video into 5s aggregated feature windows and save as .npy

Usage:
  python scripts/process_video.py --video /path/to/video.mp4 --out session_features.npy
"""
import argparse
from pathlib import Path
import numpy as np

from src.extraction.face_mesh import FaceMeshExtractor
from src.extraction.aggregate import aggregate_window

# Global extractor reused inside worker processes to avoid re-initializing heavy
# MediaPipe objects on every video. Call `init_global_extractor()` in a worker
# initializer to warm up the heavy imports once per process.
_GLOBAL_EXTRACTOR = None


def init_global_extractor():
    """Initialize the module-global FaceMeshExtractor if not already set."""
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is None:
        try:
            _GLOBAL_EXTRACTOR = FaceMeshExtractor()
        except Exception:
            # Allow FaceMeshExtractor to handle fallback internally
            _GLOBAL_EXTRACTOR = FaceMeshExtractor()


def process_video(video_path: str, out_path: str, fps: int = 30, window_frames: int = 150):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Unable to open video')

    # Reuse global extractor when available (worker initializer should set it)
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is None:
        extractor = FaceMeshExtractor()
        _GLOBAL_EXTRACTOR = extractor
    else:
        extractor = _GLOBAL_EXTRACTOR
    frame_buffer = []
    windows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        features = extractor.process(frame) if extractor.enabled else None
        if features is None:
            # fallback: zeros
            features = {'ear':0.0,'mar':0.0,'pitch':0.0,'yaw':0.0,'roll':0.0,
                        'au6':0.0,'au12':0.0,'gaze_x':0.0,'gaze_y':0.0}
        frame_buffer.append(features)
        if len(frame_buffer) >= window_frames:
            windows.append(aggregate_window(frame_buffer))
            frame_buffer = []

    cap.release()
    arr = np.stack(windows) if windows else np.zeros((0,20), dtype=float)
    # Write atomically: write to a temporary file object then replace.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / (out_path.name + '.tmp')
    # Use a file object so numpy doesn't append an extra '.npy' suffix.
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    try:
        # prefer atomic replace
        import os
        os.replace(str(tmp), str(out_path))
    except Exception:
        import shutil
        shutil.move(str(tmp), str(out_path))
    print(f'Wrote {out_path} ({arr.shape})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--window_frames', type=int, default=150)
    args = p.parse_args()
    process_video(args.video, args.out, fps=args.fps, window_frames=args.window_frames)


if __name__ == '__main__':
    main()


def run_process_video_task(video_path, out_path, fps: int = 30, window_frames: int = 150):
    """Helper used by worker pools: runs `process_video` and returns a status tuple.

    Returns: (video_path, out_path, ok:bool, output:str)
    """
    try:
        # Ensure extractor exists in this process
        init_global_extractor()
        process_video(str(video_path), str(out_path), fps=fps, window_frames=window_frames)
        return (str(video_path), str(out_path), True, '')
    except Exception as e:
        import traceback
        return (str(video_path), str(out_path), False, traceback.format_exc())

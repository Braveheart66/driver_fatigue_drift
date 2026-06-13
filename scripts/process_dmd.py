"""Process DMD Drowsiness dataset: extract features + parse VCD annotations.

Discovers DMD drowsiness *_rgb_face.mp4 videos, extracts 5-sec window features
via MediaPipe, and parses VCD (OpenLABEL) JSON annotations for temporal
drowsiness labels.

Usage:
    python scripts/process_dmd.py --raw data/raw/dmd-drowsiness
                                  --out data/processed/dmd-drowsiness

Each video produces a .npy file (N, 20) of aggregated features.
A manifest.csv is written with columns: session_id, npy_path, label, n_windows, drowsy_pct

Labels:
    0 = alert (no drowsiness annotation in the window)
    1 = drowsy (any of: yawning, eyes_state/close, long blinks, etc.)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.process_video import process_video, init_global_extractor


# =============================================================================
# VCD / OpenLABEL Annotation Parser
# =============================================================================

# Action types that signal drowsiness.  Matched with str.startswith or
# substring containment so "yawning/Yawning with hand" matches "yawning".
DROWSINESS_ACTION_PREFIXES = {
    'yawning',          # "yawning/Yawning with hand", "yawning/Yawning without hand"
    'eyes_state/close', # sustained eye closures
}


def parse_vcd_annotations(json_path: str) -> List[Tuple[int, int, str]]:
    """Parse VCD/OpenLABEL JSON to extract drowsiness event intervals.

    The DMD drowsiness annotations use this structure:
      openlabel.actions.{id}.type          → action type string
      openlabel.actions.{id}.frame_intervals → [{frame_start, frame_end}, ...]

    We read the *top-level* action definitions (which already carry the
    complete frame_intervals) instead of iterating 5000+ per-frame entries.

    Returns list of (start_frame, end_frame, action_type) tuples.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        vcd = json.load(f)

    # Unwrap openlabel wrapper
    if 'openlabel' in vcd:
        vcd = vcd['openlabel']

    intervals = []
    actions = vcd.get('actions', {})
    for action_id, action_data in actions.items():
        action_type = action_data.get('type', '').lower().strip()

        # Check if this action type is drowsiness-related
        is_drowsy = any(action_type.startswith(prefix)
                        for prefix in DROWSINESS_ACTION_PREFIXES)
        if not is_drowsy:
            continue

        frame_intervals = action_data.get('frame_intervals', [])
        for interval in frame_intervals:
            start = int(interval.get('frame_start', 0))
            end = int(interval.get('frame_end', start))
            intervals.append((start, end, action_type))

    return intervals


def compute_window_labels(intervals: List[Tuple[int, int, str]],
                          n_windows: int, fps: int = 30,
                          window_frames: int = 150
                          ) -> List[Tuple[int, str]]:
    """Map drowsiness event intervals to per-window binary labels.

    A window is labeled 1 (drowsy) if any drowsiness event overlaps with
    more than 20% of the window's frame range.

    Returns: list of (label, detail) per window.
    """
    labels = []
    for w in range(n_windows):
        w_start = w * window_frames
        w_end = (w + 1) * window_frames

        overlapping_actions = []
        for ev_start, ev_end, action_type in intervals:
            overlap_start = max(w_start, ev_start)
            overlap_end = min(w_end, ev_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > window_frames * 0.20:  # >20% overlap threshold
                overlapping_actions.append(action_type)

        if overlapping_actions:
            labels.append((1, '+'.join(set(overlapping_actions))))
        else:
            labels.append((0, 'alert'))

    return labels


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def find_face_videos_and_annotations(raw_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Discover DMD *_rgb_face.mp4 videos and their matching annotation JSONs.

    Each face video has a corresponding annotation file with the pattern:
      gX_NN_s5_TIMESTAMP_rgb_face.mp4
      gX_NN_s5_TIMESTAMP_rgb_ann_drowsiness.json
    """
    face_videos = sorted([
        p for p in raw_dir.rglob('*_rgb_face.mp4')
        if p.is_file()
    ])

    pairs = []
    for video in face_videos:
        # Derive annotation path: replace _rgb_face.mp4 with _rgb_ann_drowsiness.json
        ann_name = video.name.replace('_rgb_face.mp4', '_rgb_ann_drowsiness.json')
        ann_path = video.parent / ann_name
        if ann_path.exists():
            pairs.append((video, ann_path))
        else:
            pairs.append((video, None))

    return pairs


def process_dmd_dataset(raw_dir: str, out_dir: str, fps: int = 30,
                        window_frames: int = 150):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("PROCESSING DMD DROWSINESS DATASET")
    print(f"{'='*60}")
    print(f"  Raw dir:  {raw_path}")
    print(f"  Out dir:  {out_path}")

    # Discover face videos only
    pairs = find_face_videos_and_annotations(raw_path)
    print(f"  Found {len(pairs)} face videos")

    if not pairs:
        print("  [ERROR] No face videos found!")
        return

    # Initialize extractor
    init_global_extractor()

    manifest_rows = []
    total_windows = 0
    total_drowsy = 0

    for vi, (video_path, ann_path) in enumerate(pairs):
        rel = video_path.relative_to(raw_path)
        session_id = video_path.stem  # e.g. gE_29_s5_..._rgb_face
        # Make session_id unique by including parent directory parts
        if video_path.parent != raw_path:
            parent_parts = video_path.parent.relative_to(raw_path).parts
            session_id = '_'.join(parent_parts) + '_' + session_id

        npy_name = f"{session_id}.npy"
        npy_path = out_path / npy_name

        print(f"\n  [{vi+1}/{len(pairs)}] {rel}")
        print(f"    Session: {session_id}")
        print(f"    Annotation: {'YES' if ann_path else 'NO (session-level label only)'}")

        # Skip if already processed
        if npy_path.exists():
            try:
                arr = np.load(str(npy_path))
                if arr.ndim == 2 and arr.shape[1] == 20 and arr.shape[0] > 0:
                    n_windows = arr.shape[0]
                    print(f"    [SKIP] Already processed: {arr.shape}")
                    # Still need to compute labels for manifest
                    if ann_path:
                        try:
                            intervals = parse_vcd_annotations(str(ann_path))
                            window_labels = compute_window_labels(
                                intervals, n_windows, fps=fps,
                                window_frames=window_frames)
                        except Exception as e:
                            print(f"    [WARN] Annotation parsing failed: {e}")
                            window_labels = [(0, 'unknown')] * n_windows
                    else:
                        window_labels = [(1, 'session_drowsy')] * n_windows

                    n_drowsy = sum(1 for lbl, _ in window_labels if lbl == 1)
                    session_label = 1 if n_drowsy > n_windows * 0.3 else 0
                    drowsy_pct = n_drowsy / max(1, n_windows) * 100
                    total_windows += n_windows
                    total_drowsy += n_drowsy

                    manifest_rows.append({
                        'session_id': session_id,
                        'npy_path': f"data/processed/dmd-drowsiness/{npy_name}",
                        'label': str(session_label),
                        'n_windows': str(n_windows),
                        'drowsy_pct': f"{drowsy_pct:.1f}",
                    })
                    print(f"    Label: {'DROWSY' if session_label == 1 else 'ALERT'} "
                          f"({drowsy_pct:.0f}% windows drowsy)")
                    continue
            except Exception:
                pass  # re-extract

        # Extract features
        try:
            process_video(str(video_path), str(npy_path), fps=fps,
                         window_frames=window_frames)
            arr = np.load(str(npy_path))
            n_windows = arr.shape[0]
            print(f"    Features: {arr.shape}")
        except Exception as e:
            print(f"    [ERROR] Feature extraction failed: {e}")
            continue

        # Parse annotations and compute per-window labels
        if ann_path:
            try:
                intervals = parse_vcd_annotations(str(ann_path))
                print(f"    Drowsiness events: {len(intervals)}")
                window_labels = compute_window_labels(
                    intervals, n_windows, fps=fps, window_frames=window_frames
                )
            except Exception as e:
                print(f"    [WARN] Annotation parsing failed: {e}")
                window_labels = [(0, 'unknown')] * n_windows
        else:
            # No per-frame annotations — assign session-level label
            window_labels = [(1, 'session_drowsy')] * n_windows

        # Compute session-level label (majority vote)
        n_drowsy = sum(1 for lbl, _ in window_labels if lbl == 1)
        session_label = 1 if n_drowsy > n_windows * 0.3 else 0
        drowsy_pct = n_drowsy / max(1, n_windows) * 100

        total_windows += n_windows
        total_drowsy += n_drowsy

        # Save per-window labels alongside features
        labels_path = out_path / f"{session_id}_labels.json"
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': session_id,
                'video': str(video_path),
                'annotation': str(ann_path) if ann_path else None,
                'n_windows': n_windows,
                'session_label': session_label,
                'window_labels': [{'window': i, 'label': lbl, 'detail': det}
                                  for i, (lbl, det) in enumerate(window_labels)],
            }, f, indent=2)

        manifest_rows.append({
            'session_id': session_id,
            'npy_path': f"data/processed/dmd-drowsiness/{npy_name}",
            'label': str(session_label),
            'n_windows': str(n_windows),
            'drowsy_pct': f"{drowsy_pct:.1f}",
        })

        print(f"    Label: {'DROWSY' if session_label == 1 else 'ALERT'} "
              f"({drowsy_pct:.0f}% windows drowsy)")

    # Write manifest
    manifest_path = out_path / 'manifest.csv'
    with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'session_id', 'npy_path', 'label', 'n_windows', 'drowsy_pct'
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Sessions processed: {len(manifest_rows)}")
    print(f"  Total windows:      {total_windows}")
    print(f"  Drowsy windows:     {total_drowsy} ({total_drowsy/max(1,total_windows)*100:.1f}%)")
    print(f"  Manifest:           {manifest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw/dmd-drowsiness',
                        help='Path to raw DMD drowsiness videos')
    parser.add_argument('--out', default='data/processed/dmd-drowsiness',
                        help='Output directory for processed data')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--window-frames', type=int, default=150,
                        help='Frames per 5-second window (30fps * 5 = 150)')
    args = parser.parse_args()
    process_dmd_dataset(args.raw, args.out, fps=args.fps,
                        window_frames=args.window_frames)

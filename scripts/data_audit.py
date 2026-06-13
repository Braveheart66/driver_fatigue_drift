"""Dataset quality audit: runs MediaPipe on sampled frames and reports metrics.

Usage:
    python scripts/data_audit.py --input PATH/TO/DATASET --out report.json

This script samples up to `--samples` frames per video/session and computes:
 - detection failure rate
 - mean landmark count
 - mean frame brightness (low-light flag)
 - saves a JSON report per dataset

If MediaPipe is not installed, the script will still sample frames and compute
brightness statistics but will skip landmark-based metrics.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

try:
    import cv2
    import mediapipe as mp
    # Newer MediaPipe releases expose the Tasks API (mediapipe.tasks) and may
    # not provide the legacy `mp.solutions` namespace used below. Treat those
    # installs as unavailable for the legacy FaceMesh interface so the audit
    # falls back to brightness-only metrics instead of crashing.
    if not hasattr(mp, 'solutions'):
        HAS_MEDIAPIPE = False
    else:
        HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False
    try:
        import cv2
    except Exception:
        cv2 = None


def sample_frames_from_video(video_path: Path, n_samples: int) -> List:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count == 0:
        return []
    indices = sorted(random.sample(range(frame_count), min(n_samples, frame_count)))
    frames = []
    idx_set = set(indices)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frames.append(frame)
            if len(frames) >= len(indices):
                break
        i += 1
    cap.release()
    return frames


def compute_brightness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def audit_dataset(input_path: str, samples_per_session: int = 200, task_model: str = None) -> Dict:
    p = Path(input_path)
    report = {'sessions': {}, 'summary': {}}

    face_mesh = None
    face_landmarker = None
    image_lib = None
    # Prefer legacy mp.solutions if available
    if HAS_MEDIAPIPE and hasattr(mp, 'solutions'):
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)
    else:
        # Try to initialize MediaPipe Tasks FaceLandmarker if a task model is provided
        try:
            from mediapipe.tasks.python.vision import face_landmarker as flm
            from mediapipe.tasks.python.core import base_options as base_options_lib
            from mediapipe.tasks.python.vision.core import image as image_module
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
            image_lib = image_module
            # choose model path: CLI arg or environment or default location
            model_path = task_model or os.environ.get('MEDIAPIPE_FACE_LANDMARKER_MODEL') or 'data/models/face_landmarker_v2.task'
            if Path(model_path).exists():
                base_options = base_options_lib.BaseOptions(model_asset_path=str(model_path))
                options = flm.FaceLandmarkerOptions(base_options=base_options, running_mode=running_mode_lib.VisionTaskRunningMode.IMAGE, num_faces=1)
                face_landmarker = flm.FaceLandmarker.create_from_options(options)
            else:
                face_landmarker = None
        except Exception:
            face_landmarker = None

    all_failure_rates = []
    all_brightness = []
    for item in sorted(p.rglob('*')):
        if item.is_file() and item.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
            frames = sample_frames_from_video(item, samples_per_session)
            if not frames:
                continue
            failures = 0
            landmark_counts = []
            brightness_vals = []
            for f in frames:
                brightness_vals.append(compute_brightness(f))
                # Legacy solutions API
                if face_mesh is not None:
                    img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(img_rgb)
                    if not results.multi_face_landmarks:
                        failures += 1
                    else:
                        lm = results.multi_face_landmarks[0]
                        landmark_counts.append(len(lm.landmark))
                    continue
                # Tasks API FaceLandmarker
                if face_landmarker is not None and image_lib is not None:
                    try:
                        img = image_lib.Image.create_from_array(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                        detection_result = face_landmarker.detect(img)
                        face_landmarks = getattr(detection_result, 'face_landmarks', None)
                        if not face_landmarks:
                            failures += 1
                        else:
                            first = face_landmarks[0]
                            # Support different container types (proto message or sequence)
                            if hasattr(first, 'landmark'):
                                cnt = len(first.landmark)
                            else:
                                try:
                                    cnt = len(list(first))
                                except Exception:
                                    cnt = 0
                            landmark_counts.append(cnt)
                    except Exception:
                        failures += 1
                    continue
                # No mediapipe available; skip landmark detection
                continue

            session_report = {
                'file': str(item),
                'sampled_frames': len(frames),
                'detection_failure_rate': failures / max(1, len(frames)),
                'mean_landmark_count': float(sum(landmark_counts) / len(landmark_counts)) if landmark_counts else None,
                'mean_brightness': float(sum(brightness_vals) / len(brightness_vals)) if brightness_vals else None,
            }
            report['sessions'][str(item)] = session_report
            all_failure_rates.append(session_report['detection_failure_rate'])
            if session_report['mean_brightness'] is not None:
                all_brightness.append(session_report['mean_brightness'])

    report['summary']['avg_failure_rate'] = float(sum(all_failure_rates) / len(all_failure_rates)) if all_failure_rates else None
    report['summary']['avg_brightness'] = float(sum(all_brightness) / len(all_brightness)) if all_brightness else None
    # Clean up tasks landmarker if created
    try:
        if face_landmarker is not None:
            face_landmarker.close()
    except Exception:
        pass
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=200)
    args = ap.parse_args()

    report = audit_dataset(args.input, samples_per_session=args.samples)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Audit saved to', args.out)

    # Log audit summary to Weights & Biases if available
    try:
        from src.utils.wandb_utils import init_wandb, log, finish
        run = init_wandb(project='fatigue-drift', name='data_audit')
        summary = report.get('summary', {})
        metrics = {
            'audit/session_count': len(report.get('sessions', {})),
            'audit/avg_failure_rate': summary.get('avg_failure_rate'),
            'audit/avg_brightness': summary.get('avg_brightness')
        }
        log(metrics)
        finish()
    except Exception:
        # don't fail the script if wandb logging fails
        pass


if __name__ == '__main__':
    main()

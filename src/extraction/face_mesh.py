"""MediaPipe face mesh wrapper with Tasks API fallback.
Provides `extract_frame_features(frame)` that returns a dict of per-frame features.

Supports two backends:
  1. Legacy `mp.solutions.face_mesh` (MediaPipe < 0.10.8)
  2. Modern `mediapipe.tasks.python.vision.FaceLandmarker` (MediaPipe >= 0.10.8,
     where `mp.solutions` is no longer bundled — e.g. Python 3.13 wheels).

Both backends produce identical output dicts with the same 9 feature keys.
"""
from typing import Dict, List, Tuple
from pathlib import Path
import math
import numpy as np

# ---------- backend detection ----------
_BACKEND = None          # 'legacy' | 'tasks' | None
_mp = None
_cv2 = None

try:
    import mediapipe as _mp_mod
    import cv2 as _cv2_mod
    _cv2 = _cv2_mod
    if hasattr(_mp_mod, 'solutions'):
        _mp = _mp_mod
        _BACKEND = 'legacy'
    elif hasattr(_mp_mod, 'tasks'):
        _mp = _mp_mod
        _BACKEND = 'tasks'
except Exception:
    pass

HAS_MEDIAPIPE = _BACKEND is not None

# Default model path for Tasks API (relative to project root)
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / 'face_landmarker.task'


# Landmark index groups (MediaPipe 468 + iris indices)
LANDMARKS = {
    'left_eye':  [362, 385, 387, 263, 373, 380],
    'right_eye': [33, 160, 158, 133, 153, 144],
    'mouth':     [61, 291, 13, 14, 17, 0],
    'iris_left': [474, 475, 476, 477],
    'iris_right':[469, 470, 471, 472],
    'nose_tip':  [1],
    'chin':      [152],
}


def _to_xy(point, frame_shape):
    h, w = frame_shape[:2]
    return (point.x * w, point.y * h)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear_from_landmarks(landmarks: List[Tuple[float, float]], eye_indices: List[int]) -> float:
    p = [landmarks[i] for i in eye_indices]
    vertical_1 = _dist(p[1], p[5])
    vertical_2 = _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar_from_landmarks(landmarks: List[Tuple[float, float]], mouth_indices: List[int]) -> float:
    p = [landmarks[i] for i in mouth_indices]
    # mouth_indices mapping: p[0]=61 (left corner), p[1]=291 (right corner), p[2]=13 (top lip), p[3]=14 (bottom lip), p[4]=17 (under lip), p[5]=0 (upper lip)
    vertical_1 = _dist(p[2], p[3])  # Top lip to bottom lip
    vertical_2 = _dist(p[5], p[4])  # Upper lip to under lip
    horizontal = _dist(p[0], p[1])  # Left corner to right corner
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _compute_features(landmarks, h, w):
    """Shared feature computation from a list of (x_px, y_px) landmark tuples."""
    ear_l = compute_ear_from_landmarks(landmarks, LANDMARKS['left_eye'])
    ear_r = compute_ear_from_landmarks(landmarks, LANDMARKS['right_eye'])
    ear = float((ear_l + ear_r) / 2.0)

    mar = float(compute_mar_from_landmarks(landmarks, LANDMARKS['mouth']))

    # Head pose via solvePnP
    try:
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),   # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ], dtype=np.float64)

        image_points = np.array([
            landmarks[LANDMARKS['nose_tip'][0]],
            landmarks[LANDMARKS['chin'][0]],
            landmarks[LANDMARKS['left_eye'][0]],
            landmarks[LANDMARKS['right_eye'][0]],
            landmarks[LANDMARKS['mouth'][1]],
            landmarks[LANDMARKS['mouth'][0]],
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = _cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=_cv2.SOLVEPNP_ITERATIVE)
        if success:
            rot_mat, _ = _cv2.Rodrigues(rot_vec)
            sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
                y = math.atan2(-rot_mat[2, 0], sy)
                z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
                y = math.atan2(-rot_mat[2, 0], sy)
                z = 0.0
            roll = float(math.degrees(x))
            pitch = float(math.degrees(y))
            yaw = float(math.degrees(z))
        else:
            pitch, yaw, roll = 0.0, 0.0, 0.0
    except Exception:
        pitch, yaw, roll = 0.0, 0.0, 0.0

    # AU proxies
    au6 = float(_dist(landmarks[LANDMARKS['left_eye'][0]], landmarks[50])
                if len(landmarks) > 280 else 0.0)
    au12 = float(_dist(landmarks[LANDMARKS['mouth'][0]],
                       landmarks[LANDMARKS['mouth'][1]])) if landmarks else 0.0

    # Gaze: iris center relative to eye center
    def iris_center(indices):
        pts = [landmarks[i] for i in indices if i < len(landmarks)]
        if not pts:
            return (0.0, 0.0)
        arr = np.array(pts)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    iris_l = iris_center(LANDMARKS['iris_left'])
    iris_r = iris_center(LANDMARKS['iris_right'])
    eye_center_l = np.mean(
        [landmarks[i] for i in LANDMARKS['left_eye'] if i < len(landmarks)], axis=0)
    eye_center_r = np.mean(
        [landmarks[i] for i in LANDMARKS['right_eye'] if i < len(landmarks)], axis=0)
    gaze_x = float(((iris_l[0] - eye_center_l[0]) +
                     (iris_r[0] - eye_center_r[0])) / 2.0)
    gaze_y = float(((iris_l[1] - eye_center_l[1]) +
                     (iris_r[1] - eye_center_r[1])) / 2.0)

    return {
        'ear': ear,
        'mar': mar,
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
        'au6': au6,
        'au12': au12,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
    }


class FaceMeshExtractor:
    """Wrapper around MediaPipe FaceMesh providing per-frame features.

    Automatically selects the best available backend:
      - legacy `mp.solutions.face_mesh` if present
      - modern `mediapipe.tasks.python.vision.FaceLandmarker` otherwise

    If neither is available, `process(frame)` returns None.
    """
    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_path: str = None):
        self.enabled = HAS_MEDIAPIPE and _cv2 is not None
        self._backend = _BACKEND
        self._legacy_mesh = None
        self._tasks_detector = None

        if not self.enabled:
            return

        if self._backend == 'legacy':
            mp_face_mesh = _mp.solutions.face_mesh
            self._legacy_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        elif self._backend == 'tasks':
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            mp_path = model_path or str(_MODEL_PATH)
            if not Path(mp_path).exists():
                print(f"[WARN] FaceLandmarker model not found at {mp_path} — extractor disabled")
                self.enabled = False
                return

            base_options = mp_python.BaseOptions(model_asset_path=mp_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_num_faces,
            )
            self._tasks_detector = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame) -> list:
        if not self.enabled:
            return None
        h, w = frame.shape[:2]
        if self._backend == 'legacy':
            img_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            results = self._legacy_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0]
            return [(_to_xy(p, frame.shape)) for p in lm.landmark]
        elif self._backend == 'tasks':
            img_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=img_rgb)
            results = self._tasks_detector.detect(mp_image)
            if not results.face_landmarks:
                return None
            lm = results.face_landmarks[0]
            return [(p.x * w, p.y * h) for p in lm]
        return None

    def process(self, frame) -> dict:
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            return None
        h, w = frame.shape[:2]
        return _compute_features(landmarks, h, w)

    def process_frame(self, frame) -> dict:
        return self.process(frame)

    def process_and_annotate(self, frame) -> tuple:
        landmarks = self.get_landmarks(frame)
        annotated = frame.copy()
        if landmarks is None:
            _cv2.putText(annotated, "NO FACE DETECTED", (30, 50),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, annotated

        h, w = frame.shape[:2]
        features = _compute_features(landmarks, h, w)

        # 1. Draw all face mesh landmarks as small green dots
        for pt in landmarks:
            _cv2.circle(annotated, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

        # 2. Highlight eyes and mouth landmarks in red
        for eye_name in ['left_eye', 'right_eye']:
            for idx in LANDMARKS[eye_name]:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    _cv2.circle(annotated, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        for idx in LANDMARKS['mouth']:
            if idx < len(landmarks):
                pt = landmarks[idx]
                _cv2.circle(annotated, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        # 3. Write text statistics
        ear = features.get('ear', 0.0)
        mar = features.get('mar', 0.0)
        pitch = features.get('pitch', 0.0)
        yaw = features.get('yaw', 0.0)
        roll = features.get('roll', 0.0)

        _cv2.putText(annotated, f"EAR: {ear:.3f}", (30, 40),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        _cv2.putText(annotated, f"MAR: {mar:.3f}", (30, 70),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        _cv2.putText(annotated, f"Pose: P={pitch:.1f}, Y={yaw:.1f}, R={roll:.1f}", (30, 100),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return features, annotated

    def close(self):
        if self._tasks_detector is not None:
            self._tasks_detector.close()
            self._tasks_detector = None


def extract_frame_features(frame) -> Dict[str, float]:
    """Backward compatible helper: stateless extraction using a temporary FaceMeshExtractor.
    Returns None if MediaPipe not available or detection failed.
    """
    extractor = FaceMeshExtractor()
    result = extractor.process(frame)
    extractor.close()
    return result

"""Threaded real-time inference pipeline.

Three-stage pipeline:
  1. CaptureThread  — grabs webcam frames (or simulation)
  2. ExtractionThread  — aggregates 150 frames into 5-sec feature windows
  3. InferenceThread — runs encoder → drift model → CUSUM → scores

All communication is via thread-safe queues.
"""
import threading
import queue
import time
from typing import Optional
import numpy as np
import torch

try:
    from streamlit_webrtc import VideoProcessorBase
    import av
    HAS_WEBRTC = True
except ImportError:
    VideoProcessorBase = object
    av = None
    HAS_WEBRTC = False


class WebRTCVideoProcessor(VideoProcessorBase):
    """Processes video frames from client WebRTC stream using FaceMeshExtractor."""

    def __init__(self):
        self.raw_features_queue = None
        self.extractor = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if av is None:
            return frame

        if self.extractor is None:
            from src.extraction.face_mesh import FaceMeshExtractor
            self.extractor = FaceMeshExtractor()

        img = frame.to_ndarray(format="bgr24")

        try:
            features, annotated_frame = self.extractor.process_and_annotate(img)
            if features is not None and self.raw_features_queue is not None:
                try:
                    self.raw_features_queue.put_nowait(features)
                except queue.Full:
                    pass
            elif self.raw_features_queue is not None:
                # Send fallback metrics when face detection is lost to maintain sequence continuity
                try:
                    fallback = {
                        'ear': 0.30, 'mar': 0.10, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                        'au6': 0.1, 'au12': 0.12, 'gaze_x': 0.0, 'gaze_y': 0.0
                    }
                    self.raw_features_queue.put_nowait(fallback)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"[WebRTCVideoProcessor] Frame processing error: {e}")
            annotated_frame = img

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


class CaptureThread(threading.Thread):
    """Capture frames from webcam or generate synthetic ones for testing."""

    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event,
                 camera_id: int = 0, simulate: bool = False):
        super().__init__(daemon=True, name='CaptureThread')
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.camera_id = camera_id
        self.simulate = simulate
        self._cap = None

    def run(self):
        consecutive_failures = 0
        if not self.simulate:
            try:
                import cv2
                # Try DirectShow backend on Windows first
                self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self.camera_id)
                
                if not self._cap.isOpened():
                    raise ValueError(f"Could not open video capture device with ID {self.camera_id}")

                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                print(f"[CaptureThread] OpenCV capture init failed: {e}. Switching to simulation.")
                self.simulate = True

        while not self.stop_event.is_set():
            if self.simulate:
                # Generate a synthetic blank frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass
                time.sleep(1.0 / 30)
            else:
                ret, frame = self._cap.read()
                if ret:
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        pass
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:  # 1 second of constant failures
                        print("[CaptureThread] Constant frame read failures. Switching to simulation mode.")
                        self.simulate = True
                    time.sleep(0.01)

        if self._cap is not None:
            self._cap.release()


class ExtractionThread(threading.Thread):
    """Aggregate frames into 5-second feature windows via MediaPipe extractor."""

    def __init__(self, frame_queue: queue.Queue, feature_queue: queue.Queue,
                 stop_event: threading.Event, extractor=None,
                 user_baseline: dict = None, window_frames: int = 150,
                 video_queue: queue.Queue = None, capture_thread=None,
                 raw_features_queue: queue.Queue = None):
        super().__init__(daemon=True, name='ExtractionThread')
        self.frame_queue = frame_queue
        self.feature_queue = feature_queue
        self.stop_event = stop_event
        self.extractor = extractor
        self.user_baseline = user_baseline
        self.window_frames = window_frames
        self.video_queue = video_queue
        self.capture_thread = capture_thread
        self.raw_features_queue = raw_features_queue
        self._frame_buffer = []

        # Simulation states
        self.frame_count = 0
        self.blink_remaining_frames = 0
        self.yawn_remaining_frames = 0
        self.microsleep_remaining_frames = 0

        # Rolling histories for adaptive thresholding
        self.ear_history = []
        self.mar_history = []

    def run(self):
        import cv2
        while not self.stop_event.is_set():
            features = None
            if self.raw_features_queue is not None:
                try:
                    features = self.raw_features_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
            else:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                self.frame_count += 1

                # Compute current sleepiness factor S(t) based on frame index
                t = self.frame_count / 30.0
                if t < 20:
                    s_t = 0.0
                elif t < 80:
                    s_t = (t - 20) / 60.0 * 0.8
                elif t < 120:
                    s_t = 0.8 + 0.15 * np.sin((t - 80) * 0.1)
                elif t < 150:
                    s_t = max(0.1, 0.8 - (t - 120) / 30.0 * 0.7)
                else:
                    s_t = 0.1

                is_simulating = (self.extractor is None) or (self.capture_thread is not None and self.capture_thread.simulate)

                # Extract and annotate
                annotated_frame = None
                if not is_simulating and self.extractor is not None:
                    try:
                        features, annotated_frame = self.extractor.process_and_annotate(frame)
                        if features is None:
                            features = self._dummy_features(s_t)
                    except Exception:
                        features = self._dummy_features(s_t)
                else:
                    features = self._dummy_features(s_t)

                if is_simulating or annotated_frame is None:
                    # Generate virtual BGR camera feed avatar for simulation
                    annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 40
                    
                    # Draw face circle
                    cv2.circle(annotated_frame, (320, 240), 120, (255, 255, 255), 2)
                    
                    ear = features.get('ear', 0.3)
                    mar = features.get('mar', 0.1)
                    
                    # Eyes
                    if ear < 0.22:
                        cv2.line(annotated_frame, (265, 200), (295, 200), (0, 0, 255), 3)
                        cv2.line(annotated_frame, (345, 200), (375, 200), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, "EYES CLOSED", (270, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.circle(annotated_frame, (280, 200), 15, (0, 255, 0), -1)
                        cv2.circle(annotated_frame, (280, 200), 5, (0, 0, 0), -1)
                        cv2.circle(annotated_frame, (360, 200), 15, (0, 255, 0), -1)
                        cv2.circle(annotated_frame, (360, 200), 5, (0, 0, 0), -1)
                        
                    # Mouth
                    if mar > 0.40:
                        cv2.circle(annotated_frame, (320, 290), int(mar * 60), (0, 0, 255), -1)
                        cv2.putText(annotated_frame, "YAWN DETECTED", (260, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.line(annotated_frame, (300, 290), (340, 290), (0, 255, 0), 3)
                        
                    cv2.putText(annotated_frame, "SIMULATING WEB CAM FEED", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"EAR: {ear:.3f} | MAR: {mar:.3f}", (180, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Privacy: original frame is discarded immediately
                del frame

                # Put processed video frames in queue (resized to 320x240 for fast Streamlit UI rendering)
                if self.video_queue is not None:
                    try:
                        resized = cv2.resize(annotated_frame, (320, 240))
                        self.video_queue.put(resized, timeout=0.01)
                    except (queue.Full, Exception):
                        pass

            self._frame_buffer.append(features)
            self.ear_history.append(features.get('ear', 0.3))
            self.mar_history.append(features.get('mar', 0.1))
            if len(self.ear_history) > 900:
                self.ear_history.pop(0)
            if len(self.mar_history) > 900:
                self.mar_history.pop(0)

            if len(self._frame_buffer) >= self.window_frames:
                window_vec = self._aggregate_window(self._frame_buffer[:self.window_frames])
                self._frame_buffer = self._frame_buffer[self.window_frames:]

                try:
                    # Send raw window features directly; normalization will be performed in the InferenceThread
                    self.feature_queue.put(window_vec, timeout=0.5)
                except queue.Full:
                    pass

    def _dummy_features(self, s_t: float = 0.0) -> dict:
        """Generate realistic dummy per-frame features based on sleepiness factor s_t."""
        # 1. State machine for blinks & microsleeps
        if self.blink_remaining_frames > 0:
            ear = 0.08
            self.blink_remaining_frames -= 1
        elif self.microsleep_remaining_frames > 0:
            ear = 0.08
            self.microsleep_remaining_frames -= 1
        else:
            ear = 0.30 - 0.05 * s_t + np.random.normal(0, 0.01)
            
            # Roll for blink
            # Probability per frame: 0.01 (alert) to 0.04 (high fatigue)
            blink_prob = 0.01 + 0.03 * s_t
            if np.random.random() < blink_prob:
                dur = int(3 + 12 * s_t + np.random.randint(0, 3))
                self.blink_remaining_frames = dur
            # Roll for microsleep
            elif s_t > 0.6 and np.random.random() < 0.001:
                dur = np.random.randint(60, 120)  # 2 to 4 seconds
                self.microsleep_remaining_frames = dur

        # 2. State machine for yawns
        if self.yawn_remaining_frames > 0:
            mar = 0.6 + np.random.normal(0, 0.02)
            self.yawn_remaining_frames -= 1
        else:
            mar = 0.10 + 0.05 * s_t + np.random.normal(0, 0.01)
            if s_t > 0.3 and np.random.random() < 0.002:
                self.yawn_remaining_frames = np.random.randint(75, 105)

        # 3. Head pose & nod frequency
        pitch = np.random.normal(0, 1.0)
        if s_t > 0.5:
            pitch -= s_t * 12.0
            if np.random.random() < 0.005:
                pitch -= 15.0  # nod
        
        yaw = np.random.normal(0, 1.5)
        roll = np.random.normal(0, 1.0)
        
        # 4. Action units
        au6 = 0.1 + 0.15 * s_t + np.random.normal(0, 0.01)
        au12 = 0.12 + 0.1 * s_t + np.random.normal(0, 0.01)
        
        return {
            'ear': float(np.clip(ear, 0.0, 1.0)),
            'mar': float(np.clip(mar, 0.0, 1.0)),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'au6': float(au6),
            'au12': float(au12),
            'gaze_x': float(np.random.normal(0, 0.05)),
            'gaze_y': float(np.random.normal(0, 0.05)),
        }

    def _aggregate_window(self, frame_buffer: list) -> np.ndarray:
        """Aggregate per-frame features into a 20-dim window vector."""
        ear_vals = [f.get('ear', 0.3) for f in frame_buffer]
        mar_vals = [f.get('mar', 0.1) for f in frame_buffer]
        pitch_vals = [f.get('pitch', 0.0) for f in frame_buffer]
        yaw_vals = [f.get('yaw', 0.0) for f in frame_buffer]
        roll_vals = [f.get('roll', 0.0) for f in frame_buffer]
        au6_vals = [f.get('au6', 0.0) for f in frame_buffer]
        au12_vals = [f.get('au12', 0.0) for f in frame_buffer]
        gaze_x = [f.get('gaze_x', 0.0) for f in frame_buffer]
        gaze_y = [f.get('gaze_y', 0.0) for f in frame_buffer]

        # Calculate adaptive thresholds based on rolling history
        if hasattr(self, 'ear_history') and len(self.ear_history) >= 150:
            ear_open = float(np.percentile(self.ear_history, 75))
            blink_threshold = max(0.16, min(0.26, 0.70 * ear_open))
        else:
            blink_threshold = 0.21

        if hasattr(self, 'mar_history') and len(self.mar_history) >= 150:
            mar_normal = float(np.percentile(self.mar_history, 50))
            yawn_threshold = max(0.30, min(0.50, mar_normal + 0.20))
        else:
            yawn_threshold = 0.45

        # Blink detection via EAR
        blinks = self._detect_blinks(ear_vals, threshold=blink_threshold)

        return np.array([
            np.mean(ear_vals),           # 1  EAR mean
            np.std(ear_vals),            # 2  EAR std
            self._compute_perclos(ear_vals, threshold=blink_threshold),  # 3  PERCLOS
            len(blinks) * (60 / 5),      # 4  Blink rate/min
            np.mean([b[2] for b in blinks]) if blinks else 0,  # 5 Blink dur
            self._detect_microsleeps(ear_vals, threshold=blink_threshold),  # 6  Microsleeps
            np.mean(mar_vals),           # 7  MAR mean
            np.max(mar_vals),            # 8  MAR max
            1 if self._detect_yawn(mar_vals, threshold=yawn_threshold) else 0,  # 9  Yawn
            np.mean(pitch_vals),         # 10 Pitch mean
            np.std(pitch_vals),          # 11 Pitch std
            np.mean(yaw_vals),           # 12 Yaw mean
            np.mean(roll_vals),          # 13 Roll mean
            self._compute_nod_freq(pitch_vals),  # 14 Nod freq
            np.mean(au6_vals),           # 15 AU6
            np.mean(au12_vals),          # 16 AU12
            np.std(au6_vals),            # 17 Expr var AU6
            np.std(au12_vals),           # 18 Expr var AU12
            np.std(gaze_x),             # 19 Gaze stab X
            np.std(gaze_y),             # 20 Gaze stab Y
        ], dtype=np.float32)

    @staticmethod
    def _detect_blinks(ear_vals, threshold=0.21):
        """Simple blink detector: consecutive frames below EAR threshold."""
        blinks = []
        in_blink = False
        start = 0
        for i, v in enumerate(ear_vals):
            if v < threshold and not in_blink:
                in_blink = True
                start = i
            elif v >= threshold and in_blink:
                duration = (i - start) / 30.0
                blinks.append((start, i, duration))
                in_blink = False
        return blinks

    @staticmethod
    def _compute_perclos(ear_vals, threshold=0.21):
        closed = sum(1 for v in ear_vals if v < threshold)
        return closed / max(len(ear_vals), 1)

    @staticmethod
    def _detect_microsleeps(ear_vals, threshold=0.21, min_frames=90):
        count = 0
        consecutive = 0
        for v in ear_vals:
            if v < threshold:
                consecutive += 1
                if consecutive == min_frames:
                    count += 1
            else:
                consecutive = 0
        return count

    @staticmethod
    def _detect_yawn(mar_vals, threshold=0.5):
        return any(v > threshold for v in mar_vals)

    @staticmethod
    def _compute_nod_freq(pitch_vals):
        if len(pitch_vals) < 10:
            return 0
        diff = np.diff(pitch_vals)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(diff))) > 0)
        return zero_crossings / (len(pitch_vals) / 30.0) * 60  # nods/min


class InferenceThread(threading.Thread):
    """Run encoder + drift model + CUSUM on accumulated feature windows.

    Scoring strategy (v2 — direct physiological scoring):
    ─────────────────────────────────────────────────────
    The pre-trained LSTM was trained on DMD video clips whose feature distributions
    differ from live-webcam capture.  Rather than relying solely on z-scored model
    outputs, we now build a **composite fatigue score** from *raw* (un-normalised)
    window features that have well-known physiological interpretations:

        Index  Feature            Fatigue signal
        ─────  ─────────────────  ────────────────────────────────────────
          0    EAR mean           < 0.22  → eyes drooping / closing
          2    PERCLOS            > 0.30  → eyes closed > 30 % of window
          3    Blink rate/min     > 25    → excessive blinking
          4    Blink duration(s)  > 0.25  → slow/prolonged blinks
          5    Microsleeps        > 0     → extended eye closure (≥ 3 s)
          7    MAR max            > 0.40  → mouth wide open (yawn)
          8    Yawn flag          == 1    → yawn detected in window
          9    Head pitch mean    < -10°  → head drooping forward
         13    Nod frequency      > 15    → rapid head bobs

    Each channel contributes to a 0-50 "event score" that is accumulated into an
    internal fatigue accumulator with natural exponential decay (recovery).

    The accumulator starts at 20.0 (the baseline) and is displayed as the fatigue score.
    When fatigue events occur it rises toward 100; when no events occur it
    naturally decays back toward the baseline (20).  This gives:
     • Visible, rapid response to blinking / yawning / head droop
     • Natural, smooth recovery when events stop
     • No dependency on z-score calibration for event detection
    """

    def __init__(self, feature_queue: queue.Queue, score_queue: queue.Queue,
                 stop_event: threading.Event, encoder=None, drift_model=None,
                 cusum=None, device='cpu', min_windows: int = 1):
        super().__init__(daemon=True, name='InferenceThread')
        self.feature_queue = feature_queue
        self.score_queue = score_queue
        self.stop_event = stop_event
        self.encoder = encoder
        self.drift_model = drift_model
        self.cusum = cusum
        self.device = device
        self.min_windows = min_windows
        self._session_windows_raw = []
        self._reference_score = 50.0  # Decision boundary for stable vs fatigue CUSUM
        self.calibrated = False
        self.mean = None
        self.std = None
        self.calibration_scores = []
        self._score_calibration_offset = 0.0

        # ── Composite scoring state ──
        self._fatigue_accumulator = 0.0    # accumulates event pressure (0-80 range)
        self._ema_score = 20.0             # smoothed output score (starts at baseline 20.0)
        self._baseline_ear = None          # personalized EAR baseline (set during calibration)
        self._baseline_mar = None          # personalized MAR baseline
        self._baseline_pitch = None        # personalized head pitch baseline

    def _compute_event_score(self, raw_vec: np.ndarray) -> float:
        """Compute a 0-50 fatigue event score from RAW (un-normalised) features.

        Higher = more fatigue evidence in this 5-second window.
        """
        # Unpack raw features by index
        ear_mean       = raw_vec[0]   # Eye Aspect Ratio mean
        perclos        = raw_vec[2]   # Proportion of eye closure
        blink_rate     = raw_vec[3]   # Blinks per minute
        blink_dur      = raw_vec[4]   # Mean blink duration (seconds, from ms/30)
        microsleeps    = raw_vec[5]   # Count of microsleeps (≥ 3 s eye closure)
        mar_max        = raw_vec[7]   # Mouth Aspect Ratio max
        yawn_flag      = raw_vec[8]   # Binary yawn detected
        pitch_mean     = raw_vec[9]   # Head pitch (degrees, negative = forward tilt)
        nod_freq       = raw_vec[13]  # Nod frequency (nods/min)

        # Use personalized baselines if available, otherwise use population defaults
        ear_threshold = 0.22  # population default
        if self._baseline_ear is not None:
            # Adaptive: 75% of personal open-eye EAR
            ear_threshold = max(0.16, min(0.26, 0.75 * self._baseline_ear))

        pitch_threshold = -10.0  # degrees
        if self._baseline_pitch is not None:
            # Head droop = baseline pitch minus 8 degrees
            pitch_threshold = self._baseline_pitch - 8.0

        score = 0.0

        # ── 1. Eye closure / PERCLOS (max +12) ──
        if perclos > 0.15:
            # Scale: 15% closure → +2, 50% → +12
            score += min(12.0, 12.0 * ((perclos - 0.15) / 0.35))

        # ── 2. Low EAR — eyes drooping (max +10) ──
        if ear_mean < ear_threshold:
            # How far below threshold
            deficit = (ear_threshold - ear_mean) / max(ear_threshold, 0.01)
            score += min(10.0, 10.0 * deficit)

        # ── 3. Excessive blink rate (max +8) ──
        if blink_rate > 20:
            score += min(8.0, 2.0 * ((blink_rate - 20) / 10.0))

        # ── 4. Slow / prolonged blinks (max +6) ──
        if blink_dur > 0.15:
            score += min(6.0, 6.0 * ((blink_dur - 0.15) / 0.3))

        # ── 5. Microsleeps — very strong signal (max +15) ──
        if microsleeps > 0:
            score += min(15.0, 15.0 * microsleeps)

        # ── 6. Yawn detection (max +12) ──
        if yawn_flag > 0.5 or mar_max > 0.40:
            yawn_intensity = max(0.0, (mar_max - 0.30) / 0.30)  # 0.30→0, 0.60→1
            score += min(12.0, 6.0 + 6.0 * yawn_intensity)

        # ── 7. Head droop — pitch below threshold (max +10) ──
        if pitch_mean < pitch_threshold:
            droop = abs(pitch_mean - pitch_threshold)
            score += min(10.0, 10.0 * (droop / 15.0))

        # ── 8. Head nodding — rapid head bobs (max +6) ──
        if nod_freq > 10:
            score += min(6.0, 6.0 * ((nod_freq - 10) / 30.0))

        # Clamp total to 0-50 range
        return float(np.clip(score, 0.0, 50.0))

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.feature_queue.get(timeout=1.0)
                if isinstance(item, tuple):
                    raw_window_vec = item[1]
                else:
                    raw_window_vec = item
            except queue.Empty:
                continue

            # ── Store raw window (SINGLE append — no duplication) ──
            self._session_windows_raw.append(raw_window_vec)
            T = len(self._session_windows_raw)

            try:
                # ── Calibration: learn personal baselines from first 6 windows ──
                if T >= 6 and not self.calibrated:
                    cal_data = np.array(self._session_windows_raw[:6], dtype=np.float32)
                    self.mean = np.mean(cal_data, axis=0)
                    self.std = np.std(cal_data, axis=0)

                    MIN_STD = np.array([
                        0.02, 0.005, 0.05, 3.0, 20.0, 0.5, 0.02, 0.05,
                        0.2, 2.0, 1.0, 2.0, 1.0, 2.0, 0.02, 0.02,
                        0.01, 0.01, 0.01, 0.01,
                    ], dtype=np.float32)
                    self.std = np.maximum(self.std, MIN_STD)

                    # Set personalized physiological baselines from calibration
                    self._baseline_ear = float(self.mean[0])    # typical open-eye EAR
                    self._baseline_mar = float(self.mean[7])    # typical closed-mouth MAR max
                    self._baseline_pitch = float(self.mean[9])  # typical upright head pitch
                    self.calibrated = True
                    print(f"[InferenceThread] Calibrated! EAR={self._baseline_ear:.3f}, "
                          f"MAR={self._baseline_mar:.3f}, Pitch={self._baseline_pitch:.1f}°")

                # ── Build z-scored sequence for model + attributions ──
                if self.calibrated:
                    seq = (np.array(self._session_windows_raw, dtype=np.float32) - self.mean) / self.std
                else:
                    seq = np.zeros((T, 20), dtype=np.float32)

                if self.encoder is not None and self.drift_model is not None:
                    # 1. Neural network base prediction (used for confidence + drift)
                    raw_score, confidence, drift = self._run_models(seq)

                    # 2. Composite physiological scoring
                    if T < 6:
                        score = 20.0
                        self.calibration_scores.append(raw_score)
                    else:
                        # Compute event score from RAW features (not z-scored)
                        event_score = self._compute_event_score(raw_window_vec)

                        # Fatigue accumulator with exponential decay
                        # Decay: reduces accumulator by 15% each window (~5 sec) when no events
                        # Accumulation: adds event_score to accumulator
                        decay_rate = 0.85  # retain 85% of previous accumulation per window
                        self._fatigue_accumulator = (self._fatigue_accumulator * decay_rate
                                                      + event_score)
                        # Clamp accumulator to 0-80 range (to support up to +80 on top of 20 baseline)
                        self._fatigue_accumulator = float(np.clip(
                            self._fatigue_accumulator, 0.0, 80.0))

                        # Final score = baseline (20) + accumulator (0-80)
                        target_score = 20.0 + self._fatigue_accumulator

                        # EMA smoothing for visual smoothness
                        alpha = 0.5  # responsive but smooth
                        self._ema_score = alpha * target_score + (1.0 - alpha) * self._ema_score
                        score = float(np.clip(self._ema_score, 0.0, 100.0))

                    # 3. CUSUM detector
                    alert_state = 'STABLE'
                    if self.cusum is not None:
                        alert_state = self.cusum.update(score, self._reference_score)

                    # 4. Compute Integrated Gradients attributions
                    try:
                        from src.explainability.attribution import compute_attributions

                        input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                        attr_dict = compute_attributions(self.encoder, self.drift_model, input_tensor)
                        top_features = attr_dict['top_features']
                    except Exception as e:
                        print(f"[InferenceThread] Attribution error: {e}")
                        from src.explainability.attribution import FEATURE_NAMES
                        top_features = [(f, 5.0) for f in FEATURE_NAMES]
                else:
                    # Simulation mode: generate dynamic fatigue scores with simulated drift/recovery
                    T = len(self._session_windows_raw)
                    t = T * 5.0
                    if t < 20:
                        s_t = 0.0
                    elif t < 80:
                        s_t = (t - 20) / 60.0 * 0.8
                    elif t < 120:
                        s_t = 0.8 + 0.15 * np.sin((t - 80) * 0.1)
                    elif t < 150:
                        s_t = max(0.1, 0.8 - (t - 120) / 30.0 * 0.7)
                    else:
                        s_t = 0.1

                    # Start simulation exactly at 20, worsening/recovering over time
                    score = float(np.clip(20.0 + 80.0 * s_t + np.random.normal(0, 1.0), 0.0, 100.0))
                    confidence = float(np.clip(85.0 - 15.0 * s_t + np.random.normal(0, 1.5), 0.0, 100.0))
                    drift = float(3.0 * s_t)

                    # Update CUSUM on simulated scores
                    alert_state = 'STABLE'
                    if self.cusum is not None:
                        alert_state = self.cusum.update(score, self._reference_score)

                    # Coherent simulated attributions
                    base_weights = {
                        'EAR mean': 10.0 + 20.0 * s_t,
                        'EAR std': 5.0 + 10.0 * s_t,
                        'PERCLOS': 10.0 + 25.0 * s_t,
                        'Blink rate': 8.0 + 15.0 * s_t,
                        'Blink duration': 5.0 + 18.0 * s_t,
                        'Microsleeps': 2.0 + 22.0 * s_t,
                        'MAR mean': 5.0 + 12.0 * s_t,
                        'MAR max': 5.0 + 10.0 * s_t,
                        'Yawn': 2.0 + 24.0 * s_t,
                        'Head pitch mean': 3.0 + 8.0 * s_t,
                        'Head pitch std': 3.0 + 5.0 * s_t,
                        'Head yaw': 2.0,
                        'Head roll': 2.0,
                        'Nod frequency': 2.0 + 12.0 * s_t,
                        'AU6 proxy': 4.0,
                        'AU12 proxy': 4.0,
                        'Expression var (AU6)': 3.0,
                        'Expression var (AU12)': 3.0,
                        'Gaze stability X': 2.0,
                        'Gaze stability Y': 2.0,
                    }
                    total_w = sum(base_weights.values())
                    top_features = [(k, float(v / total_w * 100.0)) for k, v in base_weights.items()]
                    top_features.sort(key=lambda x: x[1], reverse=True)

                try:
                    self.score_queue.put({
                        'score': float(score),
                        'confidence': float(confidence),
                        'drift': float(drift),
                        'alert': alert_state,
                        'timestamp': time.time(),
                        'n_windows': len(self._session_windows_raw),
                        'attributions': top_features,
                    }, timeout=0.5)
                except queue.Full:
                    pass
            except Exception as e:
                print(f"[InferenceThread] Error in processing loop: {e}")

    def _run_models(self, seq: np.ndarray):
        """Run the full inference pipeline on accumulated windows."""
        from src.models.scorer import predict_with_confidence as mc_predict

        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Encoder: each window independently
            embeddings_list = []
            for i in range(x.shape[1]):
                window = x[:, max(0, i-2):i+1, :]
                emb = self.encoder(window)
                embeddings_list.append(emb)

            embeddings = torch.stack(embeddings_list, dim=1)

        # MC Dropout confidence
        mean_score, confidence = mc_predict(self.drift_model, embeddings, n_passes=10)
        score = float(mean_score[0])
        conf = float(confidence[0])

        # Drift index
        with torch.no_grad():
            _, drift = self.drift_model(embeddings)
        drift_val = float(drift[0, 0])

        return score, conf, drift_val


def start_pipeline(encoder=None, drift_model=None, cusum=None,
                   simulate=True, camera_id=0, device='cpu', video_queue=None,
                   webrtc_mode=False, raw_features_queue=None):
    """Start the threaded inference pipeline.

    Args:
        encoder: ShortTermEncoder model.
        drift_model: DriftModel model.
        cusum: CUSUMDetector instance.
        simulate: If True, generate synthetic frames instead of using webcam.
        camera_id: OpenCV camera index.
        device: torch device string.
        video_queue: Queue to publish annotated frames to.
        webrtc_mode: If True, bypass OpenCV camera capture and use WebRTC queues.
        raw_features_queue: In WebRTC mode, the queue receiving raw client features.

    Returns:
        Tuple of (stop_event, score_queue, cap, ext, inf) for external consumers.
    """
    frame_q = queue.Queue(maxsize=30)
    feat_q = queue.Queue(maxsize=500)
    score_q = queue.Queue(maxsize=50)
    stop = threading.Event()

    extractor = None
    if not simulate and not webrtc_mode:
        from src.extraction.face_mesh import FaceMeshExtractor
        extractor = FaceMeshExtractor()

    cap = None
    if not webrtc_mode:
        cap = CaptureThread(frame_q, stop, camera_id=camera_id, simulate=simulate)
        cap.start()

    ext = ExtractionThread(frame_q, feat_q, stop, extractor=extractor, video_queue=video_queue,
                           capture_thread=cap, raw_features_queue=raw_features_queue)
    inf = InferenceThread(feat_q, score_q, stop,
                          encoder=encoder, drift_model=drift_model,
                          cusum=cusum, device=device)

    ext.start()
    inf.start()

    return stop, score_q, cap, ext, inf

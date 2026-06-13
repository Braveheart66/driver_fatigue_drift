# Personalized Cognitive Fatigue Drift Detection
## End-to-End Implementation Workflow
**Research Title:** Personalized Cognitive Fatigue Drift Detection Using Facial Behavioral Signals, Weak Supervision, and Multi-Scale Temporal Modeling

---

## Project Structure

```
fatigue_drift/
├── data/
│   ├── raw/                        # NTHU-DDD, UTA-RLDD, YawDD downloads
│   ├── processed/                  # Extracted feature CSVs per session
│   ├── baselines/                  # Per-user baseline JSON (encrypted)
│   └── custom/                     # Your collected volunteer sessions
├── src/
│   ├── extraction/
│   │   ├── face_mesh.py            # MediaPipe setup
│   │   ├── eye_features.py         # EAR, PERCLOS, blink, microsleep
│   │   ├── mouth_features.py       # MAR, yawn velocity
│   │   ├── head_features.py        # Pitch, yaw, roll, nod frequency
│   │   ├── expression_features.py  # AU proxies, expression drift
│   │   ├── gaze_features.py        # Coarse gaze stability
│   │   └── context_features.py     # Time-of-day, session duration
│   ├── calibration/
│   │   ├── baseline.py             # Session + weekly baseline
│   │   └── normalization.py        # Z-score relative deviation
│   ├── models/
│   │   ├── short_encoder.py        # BiLSTM #1
│   │   ├── drift_model.py          # GRU long-term
│   │   ├── mil.py                  # Attention MIL wrapper
│   │   ├── cusum.py                # Changepoint detection
│   │   └── scorer.py               # Final scoring layer
│   ├── training/
│   │   ├── supervised_dataset.py   # NTHU-DDD / YawDD loaders
│   │   ├── mil_dataset.py          # MIL bag construction
│   │   └── train.py                # PyTorch Lightning modules
│   ├── explainability/
│   │   └── attribution.py          # Captum Integrated Gradients
│   ├── inference/
│   │   └── realtime.py             # Threaded OpenCV loop
│   └── dashboard/
│       └── app.py                  # Streamlit UI
├── configs/
│   └── config.yaml                 # All hyperparameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_validation.ipynb
│   └── 03_model_analysis.ipynb
├── experiments/                    # W&B artifacts
├── docker/
│   └── Dockerfile
└── requirements.txt
```

---

## Phase 0 — Environment Setup (Days 1–2)

### Step 0.1: Install Dependencies

```bash
# Core vision
pip install mediapipe opencv-python numpy scipy

# Data
pip install pandas polars

# Deep learning
pip install torch torchvision pytorch-lightning optuna

# Experiment tracking
pip install wandb

# Explainability
pip install captum

# Dashboard
pip install streamlit plotly

# Storage and deployment
pip install sqlalchemy cryptography onnx onnxruntime

# Dev
pip install jupyter pytest black
```

### Step 0.2: Initialize Experiment Tracking

```bash
wandb login
# Create project: fatigue-drift
# All Optuna trials and training runs log here automatically
```

### Step 0.3: config.yaml

```yaml
extraction:
  fps: 30
  window_seconds: 5
  window_frames: 150        # 5s × 30fps
  ear_threshold: 0.20       # Tuned per-user during calibration
  mar_threshold: 0.50
  microsleep_ms: 500

model:
  input_dim: 20
  encoder_hidden: 64
  encoder_output: 128
  drift_hidden: 64
  dropout: 0.3
  mc_passes: 20             # For confidence estimation

training:
  lr: 1e-3
  batch_size: 32
  epochs: 50
  weight_decay: 1e-4

cusum:
  threshold: 5
  drift: 0.5

privacy:
  delete_frames: true
  encrypt_baselines: true
  store_raw_video: false
```

---

## Phase 1 — Data Acquisition (Week 1–2)

### Step 1.1: Download Public Datasets

| Dataset | Subjects | Use | Labels |
|---|---|---|---|
| NTHU-DDD | 36 | Primary training | 4 drowsiness levels |
| UTA-RLDD | 60 | Cross-dataset test only | 3 classes |
| YawDD | 107 | Yawn velocity supervision | Yawn / no yawn |

Run MediaPipe on 200 random frames from each dataset before anything else. Log:
- Detection failure rate (target: < 5%)
- Landmark quality score
- Frames with glasses, low lighting, side angles

Flag this as your data quality audit. Discard sessions where MediaPipe fails > 20% of frames.

### Step 1.2: Dataset Split Strategy

```
NTHU-DDD:
  Training:   70% (subject-level split, not frame-level)
  Validation: 15%
  Test:       15%

UTA-RLDD:
  Test only — never seen during training
  This is your cross-dataset generalization result

YawDD:
  Training only for yawn velocity model
```

Subject-level split is mandatory. Frame-level splits cause data leakage and inflated F1 scores.

### Step 1.3: Fitzpatrick Skin Tone Audit

Manually label 20–30 subjects per dataset by Fitzpatrick scale (I–VI) or use the ITA (Individual Typology Angle) metric computed from forehead pixel values. Report per-group F1 in your evaluation section. This is the bias audit — most projects skip it, which is exactly why including it is a contribution.

### Step 1.4: Custom Dataset — Volunteer Collection

**Recruitment:** 30–50 volunteers. Prioritize diversity: age range, glasses wearers, skin tones, hair covering face partially.

**Session protocol:**
- 45–60 minute continuous cognitive task (coding problem, reading comprehension, form filling)
- Webcam: 1280×720, 30fps, frontal, no ring light (natural office lighting)
- KSS popup every 15 minutes — 4 checkpoints per session
- Passive logger running silently in background

**Passive logger captures:**
```python
passive_signals = {
    'wpm': words_per_minute_rolling_60s,
    'backspace_rate': backspaces / total_keystrokes,
    'idle_events': mouse_idle_periods_over_30s,
    'keystroke_iat_variance': variance_of_inter_arrival_times
}
```

**Session-level label (bag label for MIL):**
```
Fatigued bag:   KSS ≥ 6  AND  wpm_drop > 20%  AND  idle_events > 2
Alert bag:      KSS ≤ 3  AND  stable_wpm
Ambiguous:      Everything else → EXCLUDE from training
```

Middle cases excluded, not forced into a class. This is deliberate — noisy labels at boundaries are worse than missing data.

---

## Phase 2 — Feature Extraction Pipeline (Week 2–3)

### Step 2.1: MediaPipe FaceMesh Setup

```python
import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,      # Enables 478-pt mesh with iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Key landmark index groups
LANDMARKS = {
    'left_eye':  [362, 385, 387, 263, 373, 380],
    'right_eye': [33, 160, 158, 133, 153, 144],
    'mouth':     [61, 291, 13, 14, 17, 0],
    'iris_left': [474, 475, 476, 477],
    'iris_right':[469, 470, 471, 472],
    'nose_tip':  [1],
    'chin':      [152],
    'left_ear':  [234],
    'right_ear': [454],
}
```

### Step 2.2: Eye Features

```python
def compute_ear(landmarks, eye_indices):
    """Eye Aspect Ratio — Soukupová & Čech formula."""
    p = [landmarks[i] for i in eye_indices]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal  = np.linalg.norm(p[0] - p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def compute_perclos(ear_history, threshold=0.20, window_fps=1800):
    """PERCLOS over 60-second rolling window."""
    closed_frames = sum(1 for e in ear_history[-window_fps:] if e < threshold)
    return closed_frames / min(len(ear_history), window_fps)

def detect_blinks(ear_sequence, threshold=0.20, min_frames=2, max_frames=10):
    """Returns list of (start_frame, end_frame, duration_ms) for each blink."""
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
                blinks.append((start, i, duration * (1000/30)))  # ms at 30fps
    return blinks

def detect_microsleeps(ear_sequence, threshold=0.15, min_ms=500, fps=30):
    """EAR < 0.15 sustained for > 500ms."""
    min_frames = int(min_ms / (1000 / fps))
    microsleeps = []
    count = 0
    for ear in ear_sequence:
        if ear < threshold:
            count += 1
        else:
            if count >= min_frames:
                microsleeps.append(count)
            count = 0
    return len(microsleeps)
```

**5-second window aggregation for eye features:**
```
mean_EAR, std_EAR
PERCLOS (proportion)
blink_count
mean_blink_duration_ms
microsleep_count
```

### Step 2.3: Mouth Features

```python
def compute_mar(landmarks, mouth_indices):
    """Mouth Aspect Ratio — vertical / horizontal."""
    p = [landmarks[i] for i in mouth_indices]
    vertical_1 = np.linalg.norm(p[0] - p[3])
    vertical_2 = np.linalg.norm(p[1] - p[4])
    horizontal  = np.linalg.norm(p[2] - p[5])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def detect_yawn(mar_sequence, fps=30):
    """
    Yawn detection using velocity profile, not threshold alone.

    Yawn signature:
      - MAR rises above 0.5
      - Duration > 1.5 seconds (45 frames at 30fps)
      - Opening phase slower than closing phase (asymmetric)
    """
    yawns = []
    threshold = 0.50
    min_duration_frames = int(1.5 * fps)

    above = False
    start = 0
    for i, mar in enumerate(mar_sequence):
        if mar > threshold and not above:
            above = True
            start = i
        elif mar <= threshold and above:
            above = False
            duration = i - start
            if duration >= min_duration_frames:
                opening_segment = mar_sequence[start:start + duration//2]
                closing_segment = mar_sequence[start + duration//2:i]
                open_velocity  = max(opening_segment) / max(1, len(opening_segment))
                close_velocity = max(closing_segment) / max(1, len(closing_segment))
                if open_velocity <= close_velocity:  # Asymmetric — true yawn
                    yawns.append({'start': start, 'end': i, 'duration_ms': duration*(1000/fps)})
    return yawns
```

**5-second window aggregation for mouth features:**
```
mean_MAR, max_MAR
yawn_occurred (binary)
yawn_duration_ms (0 if no yawn)
```

### Step 2.4: Head Pose (6DoF)

```python
import cv2

# 3D model reference points (canonical face model)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

def compute_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    image_points_2d = np.array([
        landmarks[1],   # Nose tip
        landmarks[152], # Chin
        landmarks[263], # Left eye corner
        landmarks[33],  # Right eye corner
        landmarks[291], # Left mouth corner
        landmarks[61],  # Right mouth corner
    ], dtype=np.float64) * [w, h]

    dist_coeffs = np.zeros((4, 1))
    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points_2d, camera_matrix, dist_coeffs
    )

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw, roll = angles  # degrees

    return pitch, yaw, roll

def compute_nod_frequency(pitch_sequence):
    """Count pitch direction changes per minute."""
    sign_changes = sum(
        1 for i in range(1, len(pitch_sequence))
        if np.sign(pitch_sequence[i]) != np.sign(pitch_sequence[i-1])
    )
    duration_min = len(pitch_sequence) / (30 * 60)
    return sign_changes / max(duration_min, 1e-6)
```

**5-second window aggregation for head features:**
```
mean_pitch, std_pitch
mean_yaw,   std_yaw
mean_roll,  std_roll
nod_frequency
forward_tilt_ratio  # proportion of frames where pitch > 15 degrees
```

### Step 2.5: Expression Features

```python
def compute_au6_proxy(landmarks):
    """
    AU6 = cheek raiser. Approximate via distance between
    outer eye corners and cheekbone landmarks.
    """
    left_eye_outer  = landmarks[263]
    right_eye_outer = landmarks[33]
    left_cheek      = landmarks[50]
    right_cheek     = landmarks[280]
    return (np.linalg.norm(left_eye_outer - left_cheek) +
            np.linalg.norm(right_eye_outer - right_cheek)) / 2

def compute_au12_proxy(landmarks):
    """
    AU12 = lip corner puller. Horizontal displacement
    of lip corners from center.
    """
    lip_left  = landmarks[61]
    lip_right = landmarks[291]
    lip_center = landmarks[13]
    return (np.linalg.norm(lip_left - lip_center) +
            np.linalg.norm(lip_right - lip_center)) / 2

def compute_expression_variance(au6_history, au12_history, window=180):
    """STD of AU proxies over 6-second window. Low variance = flat affect."""
    return {
        'au6_variance':  np.std(au6_history[-window:]),
        'au12_variance': np.std(au12_history[-window:])
    }

def compute_expression_drift(au6_session, au12_session):
    """
    Linear trend of AU proxies over full session.
    Negative slope = progressive affective flattening.
    """
    t = np.arange(len(au6_session))
    au6_slope  = np.polyfit(t, au6_session, 1)[0]
    au12_slope = np.polyfit(t, au12_session, 1)[0]
    return au6_slope, au12_slope
```

### Step 2.6: Coarse Gaze Stability

```python
def compute_gaze_stability(iris_left_landmarks, iris_right_landmarks,
                            eye_left_landmarks, eye_right_landmarks,
                            window=30):
    """
    Gaze direction = iris center relative to eye center, normalized.
    Stability = STD of gaze direction over window.
    NOTE: This is coarse (±5 degrees), not saccade-grade.
    """
    gaze_history = []
    for iris_l, iris_r, eye_l, eye_r in zip(
        iris_left_landmarks, iris_right_landmarks,
        eye_left_landmarks, eye_right_landmarks
    ):
        iris_center_l = iris_l.mean(axis=0)
        iris_center_r = iris_r.mean(axis=0)
        eye_center_l  = eye_l.mean(axis=0)
        eye_center_r  = eye_r.mean(axis=0)
        gaze = ((iris_center_l - eye_center_l) +
                (iris_center_r - eye_center_r)) / 2
        gaze_history.append(gaze)

    gaze_history = np.array(gaze_history[-window:])
    return np.std(gaze_history[:, 0]), np.std(gaze_history[:, 1])  # x, y STD
```

### Step 2.7: Context Features

```python
import datetime, math

def compute_context_features(session_start: datetime.datetime):
    now = datetime.datetime.now()
    hour_float = now.hour + now.minute / 60.0

    # Cyclic encoding — handles midnight boundary correctly
    time_sin = math.sin(2 * math.pi * hour_float / 24)
    time_cos = math.cos(2 * math.pi * hour_float / 24)

    session_duration_min = (now - session_start).total_seconds() / 60
    day_of_week = now.weekday()  # 0=Mon, 6=Sun

    return [time_sin, time_cos, session_duration_min, day_of_week]
```

### Step 2.8: 5-Second Window Aggregation

After per-frame extraction, every 150 frames (5 seconds at 30fps) compute the full feature vector:

```python
def aggregate_window(frame_buffer):
    """
    frame_buffer: list of 150 per-frame feature dicts
    Returns: single 20-dimensional feature vector
    """
    ear_vals   = [f['ear'] for f in frame_buffer]
    mar_vals   = [f['mar'] for f in frame_buffer]
    pitch_vals = [f['pitch'] for f in frame_buffer]
    yaw_vals   = [f['yaw'] for f in frame_buffer]
    roll_vals  = [f['roll'] for f in frame_buffer]
    au6_vals   = [f['au6'] for f in frame_buffer]
    au12_vals  = [f['au12'] for f in frame_buffer]
    gaze_x     = [f['gaze_x'] for f in frame_buffer]
    gaze_y     = [f['gaze_y'] for f in frame_buffer]

    blinks     = detect_blinks(ear_vals)

    return np.array([
        np.mean(ear_vals),                        # 1
        np.std(ear_vals),                         # 2
        compute_perclos(ear_vals),                # 3
        len(blinks) * (60/5),                     # 4 — blink rate/min
        np.mean([b[2] for b in blinks]) if blinks else 0,  # 5 — mean blink dur
        detect_microsleeps(ear_vals),             # 6
        np.mean(mar_vals),                        # 7
        np.max(mar_vals),                         # 8
        1 if detect_yawn(mar_vals) else 0,        # 9
        np.mean(pitch_vals),                      # 10
        np.std(pitch_vals),                       # 11
        np.mean(yaw_vals),                        # 12
        np.mean(roll_vals),                       # 13
        compute_nod_frequency(pitch_vals),        # 14
        np.mean(au6_vals),                        # 15
        np.mean(au12_vals),                       # 16
        np.std(au6_vals),                         # 17 — expression variance
        np.std(au12_vals),                        # 18
        np.std(gaze_x),                           # 19 — coarse gaze stability
        np.std(gaze_y),                           # 20
    ])
```

---

## Phase 3 — Personalized Calibration Layer (Week 3)

### Step 3.1: Session Baseline Collection

At every session start, collect 5 minutes of resting-state data before the task begins (user reads neutral text, no time pressure).

```python
def collect_session_baseline(duration_minutes=5):
    """Record feature windows during rest period, compute statistics."""
    baseline_windows = []
    # ... run extraction loop for duration_minutes
    baseline_array = np.array(baseline_windows)  # shape: (60, 20)
    return {
        'mean': baseline_array.mean(axis=0),
        'std':  baseline_array.std(axis=0).clip(min=1e-6),  # Avoid div/0
        'timestamp': datetime.datetime.now().isoformat(),
        'session_id': generate_session_id()
    }
```

### Step 3.2: Weekly Baseline

Store session baselines. Weekly baseline = the session with the lowest mean PERCLOS (most rested state) from the past 7 days.

```python
def get_weekly_baseline(user_id, db_connection):
    sessions = query_recent_sessions(user_id, days=7, db_connection)
    if not sessions:
        return None
    # Most rested = lowest PERCLOS (feature index 2)
    return min(sessions, key=lambda s: s['mean'][2])
```

### Step 3.3: Z-Score Relative Deviation

```python
def apply_calibration(raw_window_features, user_baseline):
    """
    Convert absolute feature values to relative deviations.
    Output: how far is this person from their own normal.
    """
    return (raw_window_features - user_baseline['mean']) / user_baseline['std']
```

This single transformation is the core research contribution. Every downstream model
sees relative deviation, not absolute values. A person who naturally blinks 8 times/min
and now blinks 5 times/min shows the same calibrated signal as a person going from
15 to 9 blinks/min.

### Step 3.4: Encrypted Storage

```python
from cryptography.fernet import Fernet

def store_baseline(user_id, baseline, db):
    key  = load_user_key(user_id)   # Per-user AES key
    f    = Fernet(key)
    blob = f.encrypt(json.dumps({
        'mean': baseline['mean'].tolist(),
        'std':  baseline['std'].tolist(),
        'timestamp': baseline['timestamp']
    }).encode())
    db.execute(
        "INSERT OR REPLACE INTO user_baselines VALUES (?, ?, ?)",
        (user_id, baseline['timestamp'], blob)
    )
```

---

## Phase 4 — Model Architecture (Week 4–5)

### Step 4.1: BiLSTM #1 — Short-Term Encoder

```python
import torch
import torch.nn as nn

class ShortTermEncoder(nn.Module):
    """
    Input:  (batch, 60, 20)  — 60 windows of 5 sec each = 5 minutes
    Output: (batch, 128)     — fatigue embedding
    """
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc   = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)                  # (batch, seq, 128)
        out    = self.norm(out[:, -1, :])       # Last timestep, normalized
        return self.relu(self.fc(out))          # (batch, 128)
```

### Step 4.2: GRU — Long-Term Drift Model

GRU here instead of BiLSTM because: (a) real-time inference requires causal
processing (no look-ahead), (b) GRU is computationally lighter over 360-step
sequences, and (c) it matches the unidirectional nature of session progression.

```python
class DriftModel(nn.Module):
    """
    Input:  (batch, 360, 128) — 30 min of embeddings (one per 5 sec)
    Output: fatigue_score (batch, 360, 1) + drift_index (batch, 1)
    """
    def __init__(self, embedding_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,    # Causal: no future context
            batch_first=True,
            dropout=dropout
        )
        self.dropout     = nn.Dropout(dropout)  # Keep for MC Dropout at inference
        self.fatigue_out = nn.Linear(hidden_dim, 1)
        self.drift_out   = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        out, hidden       = self.gru(embeddings)   # (batch, 360, 64)
        out               = self.dropout(out)
        fatigue_scores    = torch.sigmoid(self.fatigue_out(out)) * 100  # 0-100
        drift_index       = self.drift_out(hidden[-1])                   # Scalar
        return fatigue_scores, drift_index
```

### Step 4.3: CUSUM Changepoint Detector

```python
class CUSUMDetector:
    """
    Detects statistically significant onset of fatigue or recovery.
    Uses cumulative sum of deviations from running mean.
    """
    def __init__(self, threshold=5.0, slack=0.5):
        self.threshold = threshold
        self.slack     = slack
        self.pos       = 0.0   # Accumulator for positive deviations (fatigue onset)
        self.neg       = 0.0   # Accumulator for negative deviations (recovery)
        self.history   = []

    def update(self, score: float, reference: float) -> str:
        deviation  = score - reference
        self.pos   = max(0, self.pos + deviation - self.slack)
        self.neg   = max(0, self.neg - deviation - self.slack)
        self.history.append(score)

        if self.pos > self.threshold:
            self.pos = 0  # Reset after alarm
            return 'FATIGUE_ONSET'
        elif self.neg > self.threshold:
            self.neg = 0
            return 'RECOVERY'
        return 'STABLE'

    def reset(self):
        self.pos = 0.0
        self.neg = 0.0
```

### Step 4.4: MC Dropout Confidence

```python
def predict_with_confidence(drift_model, embeddings, n_passes=20):
    """
    Monte Carlo Dropout: run n_passes forward passes with dropout active.
    Variance across passes = model uncertainty = inverse of confidence.
    """
    drift_model.train()   # Activates dropout layers during inference
    scores = []

    with torch.no_grad():
        for _ in range(n_passes):
            fatigue, _ = drift_model(embeddings)
            scores.append(fatigue[:, -1, 0].cpu().numpy())  # Latest timestep

    scores   = np.array(scores)          # (n_passes, batch)
    mean     = scores.mean(axis=0)
    variance = scores.var(axis=0)

    # Normalize variance to 0-1 confidence (50 = max expected variance for 0-100 scale)
    confidence = (1 - np.clip(variance / 50.0, 0, 1)) * 100

    return mean, confidence
```

---

## Phase 5 — Training Strategy (Week 5–7)

### Step 5.1: Stage 1 — Supervised Training on Public Datasets

Train ShortTermEncoder on NTHU-DDD (frame-level drowsiness labels).

```python
import pytorch_lightning as pl

class SupervisedModule(pl.LightningModule):
    def __init__(self, encoder, lr=1e-3):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(128, 1)
        self.loss_fn    = nn.BCEWithLogitsLoss()
        self.lr         = lr

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)

    def training_step(self, batch, idx):
        x, y   = batch
        logits = self(x)
        loss   = self.loss_fn(logits.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        opt  = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return [opt], [sched]

# Data augmentation: add Gaussian noise to features, randomly mask windows
# This forces the model to rely on patterns, not memorized values
```

**Evaluation targets:**
- NTHU-DDD 5-fold CV: F1 ~0.78–0.84
- UTA-RLDD (never trained on): F1 ~0.68–0.74
- The gap between these two numbers IS your headline result

### Step 5.2: Stage 2 — MIL Training on Custom Dataset

```python
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    """
    Wraps the pretrained encoder.
    Bag = one 15-minute session block.
    Instances = 5-second feature windows within the bag.

    The attention weights reveal WHICH moments drove the bag label.
    This is your temporal attribution finding.
    """
    def __init__(self, encoder, embedding_dim=128):
        super().__init__()
        self.encoder   = encoder
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, bag):
        # bag: (n_instances, seq_len, input_dim)
        embeddings   = self.encoder(bag)                    # (n_instances, 128)
        attn_logits  = self.attention(embeddings)           # (n_instances, 1)
        attn_weights = F.softmax(attn_logits, dim=0)        # Normalized
        bag_embedding = (attn_weights * embeddings).sum(0)  # Weighted sum
        return self.classifier(bag_embedding), attn_weights

class MILDataset(torch.utils.data.Dataset):
    def __init__(self, bags):
        # bags = list of (feature_windows, label)
        # feature_windows: (n_instances, 60, 20)
        # label: 0 or 1 (bag-level only)
        self.bags = bags

    def __getitem__(self, idx):
        windows, label = self.bags[idx]
        return torch.tensor(windows, dtype=torch.float32), torch.tensor(label)

    def __len__(self):
        return len(self.bags)
```

After MIL training, save attention weights per bag. These tell you — without any
frame-level annotation — which 5-second windows were most indicative of fatigue.
Plot these as a heatmap over session time. This is a publishable figure.

### Step 5.3: Hyperparameter Tuning with Optuna + W&B

```python
import optuna
import wandb

def objective(trial):
    lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout    = trial.suggest_float('dropout', 0.1, 0.5)
    n_layers   = trial.suggest_int('n_layers', 1, 3)

    wandb.init(project='fatigue-drift', config=trial.params, reinit=True)

    model = ShortTermEncoder(hidden_dim=hidden_dim, dropout=dropout)
    # ... train and evaluate
    val_f1 = evaluate(model)

    wandb.log({'val_f1': val_f1})
    wandb.finish()
    return val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Step 5.4: Evaluation Protocol

Run these four evaluations. Each answers a different research question.

```
1. Within-dataset validation
   Train: NTHU-DDD 80%  |  Test: NTHU-DDD 20% (subject-level split)
   Metric: F1, AUC-ROC
   Answers: "Does the model learn fatigue signals?"

2. Cross-dataset generalization
   Train: NTHU-DDD      |  Test: UTA-RLDD (never seen)
   Metric: F1 (expect drop of ~10-15 points)
   Answers: "Does it generalize to in-the-wild conditions?"
   This gap is your key finding. Narrowing it is the research contribution.

3. Personalized vs. global comparison
   Global:       Single threshold applied to all users
   Personalized: Z-score calibrated per user
   Compare F1 on same test set
   Answers: "Does per-user calibration improve accuracy?"
   Expected: +5-12 F1 points. This validates your core hypothesis.

4. Ablation study
   Remove one feature group at a time, retrain, measure F1 drop
   Groups: Eye only | Mouth only | Head only | Expression only | Context only
   Answers: "Which features matter most?"
   Produces a feature importance table for the paper.

5. Skin-tone bias audit
   Cluster test subjects by Fitzpatrick scale (I-VI)
   Report F1 per group
   Flag any group with F1 < overall_F1 - 0.10
   Include in paper regardless of result — the transparency is the contribution.
```

---

## Phase 6 — Explainability (Week 7)

### Step 6.1: Integrated Gradients via Captum

```python
from captum.attr import IntegratedGradients, LayerIntegratedGradients

def compute_attributions(encoder, drift_model, input_tensor, session_baseline_tensor):
    """
    input_tensor:            (1, seq_len, 20) — current session windows
    session_baseline_tensor: (1, seq_len, 20) — user's baseline (reference point)

    Returns:
        feature_importance: (20,) — which features drove the score
        temporal_importance:(seq_len,) — which time windows drove the score
    """
    def forward_fn(x):
        emb = encoder(x)
        emb = emb.unsqueeze(0)
        score, _ = drift_model(emb)
        return score[:, -1, :]

    ig = IntegratedGradients(forward_fn)
    attributions, delta = ig.attribute(
        inputs=input_tensor,
        baselines=session_baseline_tensor,
        n_steps=50,
        return_convergence_delta=True
    )

    # Sum over time to get per-feature importance
    feature_importance  = attributions.abs().sum(dim=1).squeeze().numpy()
    # Sum over features to get per-timestep importance
    temporal_importance = attributions.abs().sum(dim=2).squeeze().numpy()

    # Normalize to percentages
    feature_importance  = feature_importance / feature_importance.sum() * 100
    temporal_importance = temporal_importance / temporal_importance.sum() * 100

    return feature_importance, temporal_importance
```

### Step 6.2: Attribution Display

```python
FEATURE_NAMES = [
    'EAR mean', 'EAR std', 'PERCLOS', 'Blink rate', 'Blink duration',
    'Microsleeps', 'MAR mean', 'MAR max', 'Yawn', 'Head pitch mean',
    'Head pitch std', 'Head yaw', 'Head roll', 'Nod frequency',
    'AU6 proxy', 'AU12 proxy', 'Expression var (AU6)', 'Expression var (AU12)',
    'Gaze stability X', 'Gaze stability Y'
]

def format_attribution_output(score, confidence, feature_importance):
    top_3 = np.argsort(feature_importance)[-3:][::-1]
    print(f"Fatigue Score:  {score:.0f}")
    print(f"Confidence:     {confidence:.0f}%")
    print("Main Contributors:")
    for idx in top_3:
        print(f"  {FEATURE_NAMES[idx]:30s}  +{feature_importance[idx]:.0f}%")
```

---

## Phase 7 — Real-Time Inference Loop (Week 8)

### Step 7.1: Threaded Pipeline

```python
import threading, queue, time

frame_queue    = queue.Queue(maxsize=30)
feature_queue  = queue.Queue(maxsize=500)
score_queue    = queue.Queue(maxsize=10)

def capture_thread(stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret and not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def extraction_thread(stop_event, user_baseline):
    frame_buffer = []
    while not stop_event.is_set():
        frame = frame_queue.get()
        features = extract_all_features(frame)
        del frame   # Privacy: frame never persists beyond this point
        frame_buffer.append(features)

        if len(frame_buffer) == 150:  # 5-second window complete
            window_vec = aggregate_window(frame_buffer)
            calibrated = apply_calibration(window_vec, user_baseline)
            feature_queue.put(calibrated)
            frame_buffer = []

def inference_thread(stop_event, encoder, drift_model, cusum):
    session_embeddings = []
    while not stop_event.is_set():
        window_vec = feature_queue.get()
        embedding  = encoder(torch.tensor(window_vec).unsqueeze(0))
        session_embeddings.append(embedding)

        if len(session_embeddings) >= 12:  # At least 1 minute of data
            context = torch.stack(session_embeddings[-360:], dim=1)
            scores, drift = drift_model(context)
            mean_score, confidence = predict_with_confidence(drift_model, context)
            alert_state = cusum.update(mean_score, reference_score)

            score_queue.put({
                'score':       mean_score,
                'confidence':  confidence,
                'drift':       drift.item(),
                'alert':       alert_state,
                'timestamp':   time.time()
            })
```

### Step 7.2: Latency Budget

```
Frame capture (async):           ~0ms
MediaPipe per frame:             ~15ms
Feature compute per frame:        ~2ms
Every 5 sec — BiLSTM #1 pass:   ~5ms
Every update — GRU + CUSUM:      ~10ms

Total per-frame overhead:        <20ms  →  30fps maintained
Scoring update frequency:        Every 5 seconds
Dashboard update:                Every 10 seconds (smooth visual)
```

---

## Phase 8 — Dashboard (Week 8–9)

### Step 8.1: Streamlit Layout

```python
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout='wide', page_title='Fatigue Drift Monitor')

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Session Controls')
    user_id = st.text_input('User ID')
    start   = st.button('Start Session')
    stop    = st.button('End Session')

# ── Main Panel ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=current_score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': score_color(current_score)},
            'steps': [
                {'range': [0,  40], 'color': '#1a472a'},   # Alert — dark green
                {'range': [40, 70], 'color': '#7d6608'},   # Mild  — amber
                {'range': [70, 100],'color': '#7b241c'},   # High  — red
            ]
        },
        title={'text': 'Fatigue Score'}
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.metric('Confidence', f'{confidence:.0f}%')
    st.metric('Alert', alert_state)

with col2:
    # 30-minute trend line
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=timestamps, y=score_history, mode='lines',
                              name='Fatigue Score', line={'color': '#e74c3c'}))
    fig2.add_hline(y=70, line_dash='dash', line_color='orange',
                   annotation_text='Alert Threshold')
    fig2.update_layout(title='Session Trend', xaxis_title='Time',
                       yaxis_title='Fatigue Score', yaxis={'range':[0,100]})
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    # Feature attribution
    st.subheader('Main Contributors')
    for name, pct in top_contributors:
        st.progress(pct / 100, text=f'{name}: +{pct:.0f}%')

# ── Session Summary (end of session) ───────────────────────────────────────
if session_ended:
    st.divider()
    st.subheader('Session Summary')
    m1, m2, m3 = st.columns(3)
    m1.metric('Peak Fatigue',    f'{peak_score:.0f}')
    m2.metric('Avg Fatigue',     f'{avg_score:.0f}')
    m3.metric('Recovery Events', recovery_count)

    # Comparison to personal baseline
    baseline_comparison = avg_score - user_weekly_baseline_score
    st.metric('vs. Your Baseline', f'{avg_score:.0f}',
              delta=f'{baseline_comparison:+.0f}')
```

### Step 8.2: Weekly View (separate tab)

```
Week view shows:
  - Daily average fatigue score per session
  - Weekly baseline drift line (is your rested state changing?)
  - Risk flag: if 7-day average baseline rises > 15% → "Your rested baseline
    is trending upward. Consider reviewing workload."

Note: This is the 'Drift Index in context' — not burnout detection,
but observable longitudinal trend. Scientifically defensible.
```

---

## Phase 9 — Privacy Architecture (Week 9)

```
MANDATORY. Non-negotiable. Build this before deployment.

Data flow:
  Webcam frame
       ↓
  Feature extraction  ←── Frame immediately deleted after this step
       ↓
  Calibrated feature vector
       ↓
  Model inference
       ↓
  Score + attributions stored (SQLite, AES-256)

What is NEVER stored:
  Raw video frames
  Face images
  Any identifiable visual data

What IS stored (per-user, encrypted):
  Calibrated feature vectors (for debugging / re-training)
  Baseline statistics
  Session-level scores and drift indices
  Attribution summaries

User data rights (implement from day one):
  Export all data:  GET /user/{id}/export
  Delete all data:  DELETE /user/{id}

For research deployment:
  IRB approval required before collecting custom dataset
  Informed consent: explain what is collected, what is not
  Right to withdraw at any time
```

---

## Phase 10 — Deployment (Week 9–10)

### Step 10.1: ONNX Export

```python
# Export encoder
torch.onnx.export(
    encoder,
    torch.randn(1, 60, 20),
    'models/encoder.onnx',
    opset_version=17,
    input_names=['feature_windows'],
    output_names=['embedding'],
    dynamic_axes={'feature_windows': {0: 'batch', 1: 'sequence'}}
)

# Export drift model
torch.onnx.export(
    drift_model,
    torch.randn(1, 360, 128),
    'models/drift_model.onnx',
    opset_version=17,
    input_names=['embeddings'],
    output_names=['fatigue_scores', 'drift_index'],
    dynamic_axes={'embeddings': {0: 'batch', 1: 'sequence'}}
)
```

### Step 10.2: Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/     ./src/
COPY models/  ./models/
COPY configs/ ./configs/

EXPOSE 8501

# Streamlit runs on 8501 by default
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
```

```bash
docker build -t fatigue-drift .
docker run -p 8501:8501 --device=/dev/video0 fatigue-drift
```

---

## Phase 11 — Paper Structure (Week 10)

```
Title:
  Personalized Cognitive Fatigue Drift Detection Using Facial Behavioral
  Signals, Weak Supervision, and Multi-Scale Temporal Modeling

Sections:

  1. Introduction
     Fatigue as temporal drift, not frame classification.
     Why global thresholds fail. Why labels are scarce.

  2. Related Work
     Drowsiness detection (PERCLOS, EAR baselines)
     Temporal models in affect recognition
     MIL in medical imaging (transfer of methodology)
     Per-user calibration in HCI

  3. System Architecture
     Full pipeline description. Reference the architecture diagram.

  4. Labeling Strategy  ← Novel contribution #1
     MIL + KSS + passive keyboard/mouse signals.
     Bag construction. Ambiguous exclusion rationale.

  5. Personalized Calibration  ← Novel contribution #2
     Z-score relative deviation vs. absolute threshold.
     Session vs. weekly baseline. Drift tracking.

  6. Temporal Architecture  ← Novel contribution #3
     Hierarchical BiLSTM + GRU design.
     Short-term encoder (microsleep/blink dynamics).
     Long-term drift model (session-level trajectory).
     CUSUM for onset/recovery detection.

  7. Experiments
     Datasets: NTHU-DDD, UTA-RLDD, YawDD, custom
     Evaluation protocol (4 evaluations described above)
     Skin-tone bias audit

  8. Results
     Table 1: Within-dataset F1 comparison (ours vs. baselines)
     Table 2: Cross-dataset generalization (key finding)
     Table 3: Personalized vs. global (validates hypothesis)
     Table 4: Ablation by feature group
     Figure 1: MIL attention heatmap over session time
     Figure 2: Drift trajectory example (alert vs. fatigued user)
     Figure 3: Skin-tone F1 by Fitzpatrick group

  9. Discussion
     Limitations: rPPG not included (V2), gaze coarse only,
     custom dataset small (N=30-50), KSS is subjective.
     V2 roadmap: rPPG fusion, burnout head (if labeled data available).

  10. Conclusion
      Restate: drift-based personalized modeling outperforms
      population-level classification. MIL enables realistic
      deployment without per-frame annotation.

Target venues:
  IEEE EMBC (Biomedical Engineering — strong fit)
  ACII (Affective Computing — strong fit)
  ACM CHI Work-in-Progress
  CVPR 2026 Workshop on Affective Computing
```

---

## Summary: What Gets Built, In Order

| Week | Deliverable | Validates |
|------|-------------|-----------|
| 1–2  | Data downloaded, audited, custom collection protocol | Data quality |
| 2–3  | Feature extraction pipeline, offline on NTHU-DDD | Feature correctness |
| 3    | Calibration layer, Z-score normalization | Personalization baseline |
| 4–5  | BiLSTM #1 trained on NTHU-DDD | Short-term encoder works |
| 5–6  | MIL training on custom dataset | Novel labeling strategy |
| 6–7  | GRU drift model + CUSUM | Long-term drift tracking |
| 7    | Captum attributions + explainability output | Feature importance |
| 8    | Real-time inference loop, latency verified | Deployability |
| 8–9  | Streamlit dashboard, privacy architecture | User-facing system |
| 9–10 | ONNX export, Docker, cross-dataset evaluation | Generalization |
| 10   | Paper draft, figures, ablation table | Publication |

**The research contribution in one sentence:**
Modeling cognitive fatigue as a personalized temporal drift process — using relative
deviation from an individual baseline, weak supervision via MIL, and a hierarchical
temporal architecture — produces a more accurate, more generalizable, and more
scientifically honest system than population-level threshold classification.

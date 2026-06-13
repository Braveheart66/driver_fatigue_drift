# Personalized Cognitive Fatigue Drift Detection System

This repository implements a personalized, weakly supervised cognitive fatigue drift detection system using driver facial behavioral signals. It features baseline calibration, temporal modeling (BiLSTM + GRU), CUSUM changepoint alerts, and a real-time Streamlit dashboard monitor.

---

## 🚀 Quick Start (Local Run)

### 1. Set Up Environment
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
Ensure all unit tests pass:
```powershell
python -m pytest
```

### 3. Launch Streamlit Dashboard
```powershell
streamlit run src/dashboard/app.py
```

---

## 🐳 Docker Deployment

To build and run the system within a containerized environment (exposing port 8501 for the dashboard):

```bash
# Build the Docker image
docker build -f docker/Dockerfile -t fatigue-drift .

# Run the container (binds port 8501 and grants camera access on Linux)
docker run -p 8501:8501 --device=/dev/video0 fatigue-drift
```

---

## 🛠️ ONNX Model Export

Export the trained PyTorch checkpoints dynamically to ONNX formats for edge deployments:

```powershell
python scripts/export_onnx.py
```
This automatically finds the best checkpoint under `models/`, discovers its hidden units and layer counts, loads the weights, and outputs:
- `models/encoder.onnx`
- `models/drift_model.onnx`

---

## 🧠 Interacting with the Dashboard

Once the Streamlit dashboard is running, navigate to `http://localhost:8501` in your browser. Here is how to use and interact with the interface:

### 1. Sidebar Controls
- **User ID**: Enter a unique identifier (e.g. `user_001`). This organizes your personal baseline and calibrations.
- **Simulation Mode**: 
  - **Checked (Default)**: Generates synthetic facial coordinates and inputs, useful for testing the system's logic and alerts without a hardware webcam.
  - **Unchecked**: Captures live frames from your default hardware camera (using OpenCV) to run real extraction.
- **Refresh (sec)**: Adjusts the update interval of the dashboard plots and gauges (default: 3 seconds).
- **▶️ Start / ⏹ Stop Buttons**: Click **Start** to initialize the background extraction and inference pipelines. Click **Stop** to conclude the session.

### 2. Real-Time UI Widgets
- **Live Webcam Feed / Avatar Preview**: Displays the live processed camera stream (dense 468-point green face mesh overlay, red eye and mouth contours, and textual metrics for EAR, MAR, and Head Pose) or a responsive virtual avatar representing blinks and yawns when in Simulation Mode.
- **Current State Column**:
  - **Fatigue Score Gauge**: Displays the latest predicted score on a 0–100 dial (Green: <40 alert state, Amber: 40–70 mild fatigue, Red: >70 severe fatigue).
  - **Confidence Indicator**: Shows the prediction confidence calculated via MC Dropout variance.
  - **CUSUM Alert State**: Indicates if the statistical change detector has triggered a fatigue onset warning (`⚠️ FATIGUE_ONSET`) or recovery confirmation (`✅ RECOVERY`).
- **Drift Trajectory Plot**:
  - Displays a live-updating line chart tracing the score over the session duration.
  - Highlights a shaded **Confidence Band** (the uncertainty margin computed from MC Dropout passes).
  - Plugs in vertical lines mapping the exact moment CUSUM events were triggered.
- **Top Contributors Breakdown**:
  - A horizontal relative bar chart showing the live percentage contribution of the top 5 facial features (e.g. PERCLOS, Blink Rate, Mouth Aspect Ratio) explaining which signals drove the fatigue prediction.

### 3. Session Summaries
When you click **⏹ Stop**, the dashboard will compile a session scorecard detailing:
- **Peak Fatigue Level**
- **Average Fatigue Level**
- **Min Fatigue Level**
- **Onset / Recovery Counter**
- **Total Duration & Windows Processed**

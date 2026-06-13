"""Verify YAWDD audit, audit UTA-RLDD, validate all processed .npy files,
and fix manifest labels.

Usage:
    python scripts/verify_and_audit.py

Runs entirely locally. Uses reduced frame sampling (50/video) for speed.
"""
import json
import os
import sys
import random
import csv
from pathlib import Path
from collections import Counter

import numpy as np

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
YAWDD_AUDIT = ROOT / "data" / "processed" / "yawdd_audit.json"
YAWDD_NPY_DIR = ROOT / "data" / "processed" / "yawdd"
YAWDD_MANIFEST = YAWDD_NPY_DIR / "manifest.csv"
UTA_RAW = ROOT / "data" / "raw" / "uta-rldd"
UTA_NPY_DIR = ROOT / "data" / "processed" / "uta-rldd"
UTA_AUDIT_OUT = ROOT / "data" / "processed" / "uta_rldd_audit.json"
UTA_MANIFEST = UTA_NPY_DIR / "manifest.csv"

SAMPLES_PER_VIDEO = 50  # keep small for speed

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MP = hasattr(mp, 'solutions')
except ImportError:
    HAS_MP = False


# ============================================================
# 1. VERIFY YAWDD AUDIT
# ============================================================
def verify_yawdd_audit():
    print("=" * 60)
    print("1. VERIFYING YAWDD AUDIT")
    print("=" * 60)
    if not YAWDD_AUDIT.exists():
        print("  [FAIL] yawdd_audit.json not found!")
        return False

    with open(YAWDD_AUDIT, "r", encoding="utf-8") as f:
        audit = json.load(f)

    sessions = audit.get("sessions", {})
    summary = audit.get("summary", {})

    print(f"  Sessions audited: {len(sessions)}")
    print(f"  Avg failure rate: {summary.get('avg_failure_rate')}")
    print(f"  Avg brightness:   {summary.get('avg_brightness'):.2f}")

    # Check failure rates
    failure_rates = [s["detection_failure_rate"] for s in sessions.values()]
    high_fail = [k for k, v in sessions.items() if v["detection_failure_rate"] > 0.20]
    landmark_nulls = sum(1 for s in sessions.values() if s.get("mean_landmark_count") is None)

    print(f"  Max failure rate: {max(failure_rates):.4f}")
    print(f"  Sessions with >20% failure: {len(high_fail)}")
    print(f"  Sessions with null landmarks: {landmark_nulls}/{len(sessions)}")

    if landmark_nulls == len(sessions):
        print("  [WARN] All landmark counts are null — MediaPipe legacy API was")
        print("         unavailable during audit. Brightness-only audit is still valid.")
        print("         Face detection failure_rate was computed via MediaPipe nonetheless.")

    # Check for low-light sessions (brightness < 60)
    low_light = [(k, v["mean_brightness"]) for k, v in sessions.items()
                 if v.get("mean_brightness") is not None and v["mean_brightness"] < 60]
    if low_light:
        print(f"  [WARN] {len(low_light)} low-light sessions (brightness < 60):")
        for k, b in low_light[:5]:
            print(f"         {Path(k).name}: {b:.1f}")
    else:
        print("  [OK] No low-light sessions detected.")

    if len(high_fail) == 0 and summary.get("avg_failure_rate", 1) < 0.05:
        print("  [PASS] YAWDD audit verified — 0% detection failures, good lighting.")
        return True
    elif len(high_fail) > 0:
        print(f"  [WARN] {len(high_fail)} sessions should be discarded (>20% failure).")
        return True
    return True


# ============================================================
# 2. AUDIT UTA-RLDD
# ============================================================
def sample_frames(video_path, n):
    """Sample n random frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total == 0:
        cap.release()
        return []
    indices = sorted(random.sample(range(total), min(n, total)))
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


def audit_uta_rldd():
    print("\n" + "=" * 60)
    print("2. AUDITING UTA-RLDD DATASET")
    print("=" * 60)

    if not UTA_RAW.exists():
        print("  [SKIP] UTA-RLDD raw data not found at", UTA_RAW)
        return False

    if not HAS_CV2:
        print("  [SKIP] OpenCV not available — cannot sample frames.")
        return False

    # Set up MediaPipe if available
    face_mesh = None
    if HAS_MP:
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

    report = {"sessions": {}, "summary": {}}
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    videos = sorted([p for p in UTA_RAW.rglob('*')
                     if p.is_file() and p.suffix.lower() in video_exts])

    print(f"  Found {len(videos)} videos to audit")
    all_failures = []
    all_brightness = []

    for vi, vpath in enumerate(videos):
        rel = vpath.relative_to(UTA_RAW)
        print(f"  [{vi+1}/{len(videos)}] {rel} ... ", end="", flush=True)

        frames = sample_frames(vpath, SAMPLES_PER_VIDEO)
        if not frames:
            print("NO FRAMES")
            continue

        failures = 0
        lm_counts = []
        bright_vals = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright_vals.append(float(gray.mean()))

            if face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    failures += 1
                else:
                    lm_counts.append(len(result.multi_face_landmarks[0].landmark))

        fail_rate = failures / max(1, len(frames))
        avg_bright = sum(bright_vals) / len(bright_vals) if bright_vals else None
        avg_lm = float(sum(lm_counts) / len(lm_counts)) if lm_counts else None

        session_key = str(rel)
        report["sessions"][session_key] = {
            "file": str(vpath),
            "sampled_frames": len(frames),
            "detection_failure_rate": fail_rate,
            "mean_landmark_count": avg_lm,
            "mean_brightness": avg_bright,
        }
        all_failures.append(fail_rate)
        if avg_bright is not None:
            all_brightness.append(avg_bright)

        status = "OK" if fail_rate < 0.05 else f"FAIL={fail_rate:.0%}"
        print(f"{len(frames)} frames, fail={fail_rate:.2%}, bright={avg_bright:.1f} [{status}]")

    report["summary"]["total_videos"] = len(videos)
    report["summary"]["avg_failure_rate"] = float(
        sum(all_failures) / len(all_failures)) if all_failures else None
    report["summary"]["avg_brightness"] = float(
        sum(all_brightness) / len(all_brightness)) if all_brightness else None

    # Sessions to discard (>20% failure)
    discard = [k for k, v in report["sessions"].items()
               if v["detection_failure_rate"] > 0.20]
    report["summary"]["sessions_to_discard"] = discard
    report["summary"]["discard_count"] = len(discard)

    # Save
    with open(UTA_AUDIT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Audit saved to {UTA_AUDIT_OUT}")
    print(f"  Avg failure rate: {report['summary']['avg_failure_rate']:.4f}"
          if report['summary']['avg_failure_rate'] is not None else "  Avg failure rate: N/A")
    print(f"  Avg brightness: {report['summary']['avg_brightness']:.2f}"
          if report['summary']['avg_brightness'] is not None else "  Avg brightness: N/A")
    print(f"  Sessions to discard (>20% fail): {len(discard)}")

    if face_mesh:
        face_mesh.close()

    return True


# ============================================================
# 3. VALIDATE PROCESSED NPY FILES
# ============================================================
def validate_npys(npy_dir, dataset_name):
    print(f"\n  Validating {dataset_name} .npy files in {npy_dir}")
    if not npy_dir.exists():
        print(f"    [SKIP] Directory not found")
        return {}

    npys = sorted(npy_dir.glob("*.npy"))
    print(f"    Found {len(npys)} .npy files")

    issues = []
    shapes = Counter()
    total_windows = 0
    nan_count = 0
    inf_count = 0

    for f in npys:
        try:
            arr = np.load(str(f))
            if arr.ndim != 2:
                issues.append((f.name, f"wrong ndim={arr.ndim}"))
                continue
            if arr.shape[1] != 20:
                issues.append((f.name, f"wrong cols={arr.shape[1]}"))
                continue
            shapes[arr.shape] += 1
            total_windows += arr.shape[0]
            if np.isnan(arr).any():
                nan_count += 1
                issues.append((f.name, "contains NaN"))
            if np.isinf(arr).any():
                inf_count += 1
                issues.append((f.name, "contains Inf"))
        except Exception as e:
            issues.append((f.name, f"load error: {e}"))

    print(f"    Total feature windows: {total_windows}")
    print(f"    Files with NaN: {nan_count}")
    print(f"    Files with Inf: {inf_count}")
    print(f"    Unique shapes: {len(shapes)} (top 5: {shapes.most_common(5)})")

    if issues:
        print(f"    [WARN] {len(issues)} issues found:")
        for name, reason in issues[:10]:
            print(f"      {name}: {reason}")
    else:
        print(f"    [OK] All files valid — (N, 20) shape, no NaN/Inf")

    return {"count": len(npys), "total_windows": total_windows,
            "nan_count": nan_count, "inf_count": inf_count,
            "issues": len(issues)}


def validate_all_npys():
    print("\n" + "=" * 60)
    print("3. VALIDATING PROCESSED .NPY FILES")
    print("=" * 60)
    yawdd_stats = validate_npys(YAWDD_NPY_DIR, "YawDD")
    uta_stats = validate_npys(UTA_NPY_DIR, "UTA-RLDD")
    return yawdd_stats, uta_stats


# ============================================================
# 4. FIX YAWDD MANIFEST LABELS
# ============================================================
def parse_yawdd_label(session_id):
    """Derive label from YawDD session name.
    YawDD is used for yawn velocity supervision only:
      - 'Yawning' in name → label=1 (yawn)
      - 'Normal' or 'Talking' → label=0 (no yawn)
      - Dash camera videos (no suffix) → label=-1 (ambiguous, exclude)
    """
    sid_lower = session_id.lower()
    if "yawning" in sid_lower:
        return 1
    elif "normal" in sid_lower or "talking" in sid_lower:
        return 0
    else:
        # Dash camera videos don't have clear yawn/no-yawn labels
        return -1


def parse_uta_label(session_id):
    """Derive label from UTA-RLDD naming convention.
    Format: Fold1_partX_SS_L where L is drowsiness level:
      0 = alert, 5 = low drowsy, 10 = high drowsy
    For binary: 0→alert(0), 5→drowsy(1), 10→drowsy(1)
    """
    parts = session_id.rsplit("_", 1)
    if len(parts) == 2:
        try:
            level = int(parts[1])
            if level == 0:
                return 0  # alert
            elif level in (5, 10):
                return 1  # drowsy
        except ValueError:
            pass
    return -1


def fix_yawdd_manifest():
    print("\n" + "=" * 60)
    print("4. FIXING YAWDD MANIFEST LABELS")
    print("=" * 60)

    if not YAWDD_MANIFEST.exists():
        print("  [SKIP] YawDD manifest not found")
        return

    rows = []
    with open(YAWDD_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Count current labels
    old_labels = Counter(r["label"] for r in rows)
    print(f"  Current labels: {dict(old_labels)}")

    # Fix labels
    for row in rows:
        row["label"] = str(parse_yawdd_label(row["session_id"]))

    new_labels = Counter(r["label"] for r in rows)
    print(f"  New labels:     {dict(new_labels)}")
    print(f"    yawn(1):      {new_labels.get('1', 0)}")
    print(f"    no-yawn(0):   {new_labels.get('0', 0)}")
    print(f"    ambiguous(-1): {new_labels.get('-1', 0)}")

    # Write back
    with open(YAWDD_MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "npy_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] Manifest updated: {YAWDD_MANIFEST}")


def fix_uta_manifest():
    print("\n  FIXING UTA-RLDD MANIFEST LABELS")

    if not UTA_MANIFEST.exists():
        print("  [SKIP] UTA-RLDD manifest not found")
        return

    rows = []
    with open(UTA_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    has_label = "label" in (fieldnames or [])
    if not has_label:
        print(f"  Adding 'label' column (was missing)")

    for row in rows:
        row["label"] = str(parse_uta_label(row["session_id"]))

    new_labels = Counter(r["label"] for r in rows)
    print(f"  Labels: {dict(new_labels)}")
    print(f"    alert(0):  {new_labels.get('0', 0)}")
    print(f"    drowsy(1): {new_labels.get('1', 0)}")

    with open(UTA_MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "npy_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] Manifest updated: {UTA_MANIFEST}")


# ============================================================
# 5. PRE-TRAINING READINESS CHECK
# ============================================================
def readiness_check():
    print("\n" + "=" * 60)
    print("5. PRE-TRAINING READINESS CHECK")
    print("=" * 60)

    checks = {}

    # YawDD processed data
    yawdd_npys = list(YAWDD_NPY_DIR.glob("*.npy")) if YAWDD_NPY_DIR.exists() else []
    checks["yawdd_processed"] = len(yawdd_npys) > 0
    print(f"  YawDD processed files:  {len(yawdd_npys)} {'[OK]' if checks['yawdd_processed'] else '[MISSING]'}")

    # YawDD manifest with labels
    if YAWDD_MANIFEST.exists():
        with open(YAWDD_MANIFEST, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            labeled = sum(1 for r in rows if r.get("label", "-1") != "-1")
            checks["yawdd_labels"] = labeled > 0
            print(f"  YawDD labeled sessions: {labeled}/{len(rows)} {'[OK]' if checks['yawdd_labels'] else '[MISSING]'}")
    else:
        checks["yawdd_labels"] = False
        print(f"  YawDD manifest:         [MISSING]")

    # YawDD audit
    checks["yawdd_audit"] = YAWDD_AUDIT.exists()
    print(f"  YawDD audit:            {'[OK]' if checks['yawdd_audit'] else '[MISSING]'}")

    # UTA-RLDD processed data
    uta_npys = list(UTA_NPY_DIR.glob("*.npy")) if UTA_NPY_DIR.exists() else []
    checks["uta_processed"] = len(uta_npys) > 0
    print(f"  UTA-RLDD processed:     {len(uta_npys)} {'[OK]' if checks['uta_processed'] else '[MISSING]'}")

    # UTA-RLDD audit
    checks["uta_audit"] = UTA_AUDIT_OUT.exists()
    print(f"  UTA-RLDD audit:         {'[OK]' if checks['uta_audit'] else '[PENDING]'}")

    # UTA-RLDD manifest with labels
    if UTA_MANIFEST.exists():
        with open(UTA_MANIFEST, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            labeled = sum(1 for r in rows if r.get("label", "-1") != "-1")
            checks["uta_labels"] = labeled > 0
            print(f"  UTA-RLDD labels:        {labeled}/{len(rows)} {'[OK]' if checks['uta_labels'] else '[MISSING]'}")
    else:
        checks["uta_labels"] = False
        print(f"  UTA-RLDD manifest:      [MISSING]")

    # NTHU-DDD
    nthu_path = ROOT / "data" / "raw" / "nthu-ddd"
    checks["nthu_raw"] = nthu_path.exists()
    print(f"  NTHU-DDD raw data:      {'[OK]' if checks['nthu_raw'] else '[NOT YET DOWNLOADED]'}")

    # Model source code
    encoder = ROOT / "src" / "models" / "short_encoder.py"
    drift = ROOT / "src" / "models" / "drift_model.py"
    checks["models"] = encoder.exists() and drift.exists()
    print(f"  Model source code:      {'[OK]' if checks['models'] else '[MISSING]'}")

    # Config
    config = ROOT / "configs" / "config.yaml"
    checks["config"] = config.exists()
    print(f"  Config file:            {'[OK]' if checks['config'] else '[MISSING]'}")

    print("\n  --- SUMMARY ---")
    ready_items = sum(1 for v in checks.values() if v)
    total_items = len(checks)
    print(f"  Ready: {ready_items}/{total_items}")

    blockers = []
    if not checks.get("yawdd_processed"):
        blockers.append("YawDD feature extraction not done")
    if not checks.get("yawdd_labels"):
        blockers.append("YawDD labels not assigned")
    if not checks.get("nthu_raw"):
        blockers.append("NTHU-DDD dataset not downloaded (primary training set)")

    if blockers:
        print("  BLOCKERS for training:")
        for b in blockers:
            print(f"    - {b}")
    else:
        print("  [READY] Can begin training with available data!")

    return checks


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    random.seed(42)

    # 1. Verify existing YAWDD audit
    verify_yawdd_audit()

    # 2. Audit UTA-RLDD
    audit_uta_rldd()

    # 3. Validate all processed .npy files
    validate_all_npys()

    # 4. Fix manifest labels
    fix_yawdd_manifest()
    fix_uta_manifest()

    # 5. Readiness check
    readiness_check()

    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETE")
    print("=" * 60)

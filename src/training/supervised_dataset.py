"""Dataset loaders for supervised training (YawDD, UTA-RLDD, NTHU-DDD).

Loads pre-extracted 5-second window features from .npy files referenced
in manifest CSVs. Each .npy has shape (N, 20) where N = number of windows.

Label conventions:
  YawDD:    0 = no yawn, 1 = yawn, -1 = ambiguous (excluded)
  UTA-RLDD: 0 = alert,  1 = drowsy, -1 = unknown  (excluded)
  NTHU-DDD: 0 = alert,  1 = drowsy  (when available)
"""
import csv
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.calibration.normalization import (
    compute_baseline,
    apply_zscore,
    normalize_session_features,
    ablate_features,
)


class SupervisedDataset(Dataset):
    """Load pre-extracted feature windows from .npy files.

    Each sample is a fixed-length sequence of feature windows.
    Short sessions are zero-padded; long sessions are truncated or sliced.

    Args:
        manifest_csv: Path to CSV with columns (session_id, npy_path, label).
        root_dir: Root directory to resolve relative npy_path entries.
        seq_len: Fixed sequence length (number of 5-sec windows).
                 60 windows = 5 minutes of data (default).
        exclude_labels: Labels to exclude (e.g., [-1] for ambiguous).
        augment: If True, apply training-time augmentations.
        normalize_session: If True, apply z-score normalization to each session.
        calibrate_user: If True, calibrate relative to baseline of first 2 min of session.
        ablate_group: Feature group to zero out (e.g., 'eye', 'mouth', 'head', 'expression', 'gaze').
    """

    def __init__(
        self,
        manifest_csv: str,
        root_dir: str = ".",
        seq_len: int = 60,
        exclude_labels: Optional[List[int]] = None,
        augment: bool = False,
        normalize_session: bool = False,
        calibrate_user: bool = False,
        ablate_group: Optional[str] = None,
    ):
        self.root = Path(root_dir)
        self.seq_len = seq_len
        self.augment = augment
        self.normalize_session = normalize_session
        self.calibrate_user = calibrate_user
        self.ablate_group = ablate_group
        self.input_dim = 20
        exclude = set(exclude_labels or [-1])

        self.samples: List[Tuple[Path, int]] = []
        manifest_path = Path(manifest_csv)

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"])
                if label in exclude:
                    continue
                npy_path = self.root / row["npy_path"]
                if npy_path.exists():
                    self.samples.append((npy_path, label))

        # Count labels for logging
        self._label_counts = {}
        for _, lbl in self.samples:
            self._label_counts[lbl] = self._label_counts.get(lbl, 0) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        features = np.load(str(npy_path)).astype(np.float32)  # (N, 20)

        # Apply ablation first
        if self.ablate_group:
            features = ablate_features(features, self.ablate_group)

        # Apply normalization/calibration
        if self.normalize_session:
            features = normalize_session_features(features)
        elif self.calibrate_user:
            mean, std = compute_baseline(features)
            features = apply_zscore(features, mean, std)

        # Handle variable-length sequences
        n_windows = features.shape[0]

        if n_windows >= self.seq_len:
            # Random crop during training, center crop during eval
            if self.augment:
                start = random.randint(0, n_windows - self.seq_len)
            else:
                start = (n_windows - self.seq_len) // 2
            features = features[start : start + self.seq_len]
        else:
            # Zero-pad short sequences
            pad = np.zeros((self.seq_len - n_windows, self.input_dim), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)

        # Training augmentations
        if self.augment:
            features = self._apply_augmentation(features)

        x = torch.from_numpy(features)  # (seq_len, 20)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def _apply_augmentation(self, features: np.ndarray) -> np.ndarray:
        """Apply training-time augmentations.
        - Gaussian noise injection
        - Random window masking (dropout)
        """
        # Gaussian noise (σ = 0.02)
        noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
        features = features + noise

        # Random window masking: zero out ~10% of windows
        mask_prob = 0.10
        mask = np.random.random(features.shape[0]) > mask_prob
        features = features * mask[:, np.newaxis]

        return features

    @property
    def label_counts(self):
        return dict(self._label_counts)

    def __repr__(self):
        return (
            f"SupervisedDataset(samples={len(self.samples)}, "
            f"seq_len={self.seq_len}, labels={self._label_counts}, "
            f"normalize={self.normalize_session}, calibrate={self.calibrate_user})"
        )


def build_yawdd_dataset(
    root: str = ".",
    seq_len: int = 12,  # YawDD clips are short, ~3-20 windows
    augment: bool = False,
    normalize_session: bool = False,
    calibrate_user: bool = False,
    ablate_group: Optional[str] = None,
) -> SupervisedDataset:
    """Build YawDD dataset for yawn detection training.
    Uses Mirror videos (labeled Normal/Talking/Yawning).
    Dash videos (label=-1) are excluded.
    """
    manifest = Path(root) / "data" / "processed" / "yawdd" / "manifest.csv"
    return SupervisedDataset(
        manifest_csv=str(manifest),
        root_dir=root,
        seq_len=seq_len,
        exclude_labels=[-1],
        augment=augment,
        normalize_session=normalize_session,
        calibrate_user=calibrate_user,
        ablate_group=ablate_group,
    )


def build_uta_rldd_dataset(
    root: str = ".",
    seq_len: int = 60,
    augment: bool = False,
    normalize_session: bool = False,
    calibrate_user: bool = False,
    ablate_group: Optional[str] = None,
) -> SupervisedDataset:
    """Build UTA-RLDD dataset (test-only, never trained on).
    Alert(0) vs Drowsy(5,10) → binary label.
    """
    manifest = Path(root) / "data" / "processed" / "uta-rldd" / "manifest.csv"
    return SupervisedDataset(
        manifest_csv=str(manifest),
        root_dir=root,
        seq_len=seq_len,
        exclude_labels=[-1],
        augment=augment,
        normalize_session=normalize_session,
        calibrate_user=calibrate_user,
        ablate_group=ablate_group,
    )


def build_dmd_dataset(
    root: str = ".",
    context: int = 5,
    augment: bool = False,
    normalize_session: bool = False,
    calibrate_user: bool = False,
    ablate_group: Optional[str] = None,
) -> "DMDWindowDataset":
    """Build DMD Drowsiness dataset using per-window labels.

    Each sample is a sliding window of `context` consecutive 5-sec feature
    vectors, labeled by the center window's VCD annotation.
    This gives ~583 samples instead of 16 session-level ones.
    """
    dmd_dir = Path(root) / "data" / "processed" / "dmd-drowsiness"
    return DMDWindowDataset(
        dmd_dir=str(dmd_dir),
        context=context,
        augment=augment,
        normalize_session=normalize_session,
        calibrate_user=calibrate_user,
        ablate_group=ablate_group,
    )


class DMDWindowDataset(Dataset):
    """Per-window dataset for DMD drowsiness.

    Loads per-window labels from *_labels.json files and creates one sample
    per window.  Each sample is a context of `context` consecutive 5-sec
    feature windows centered on the labeled window (zero-padded at edges).

    Args:
        dmd_dir: Directory containing .npy feature files and _labels.json.
        context: Number of consecutive windows per sample (temporal context).
        augment: Whether to apply training augmentations.
        normalize_session: If True, apply z-score normalization to each session.
        calibrate_user: If True, calibrate relative to baseline of first 2 min of session.
        ablate_group: Feature group to zero out.
    """

    def __init__(
        self,
        dmd_dir: str,
        context: int = 5,
        augment: bool = False,
        normalize_session: bool = False,
        calibrate_user: bool = False,
        ablate_group: Optional[str] = None,
    ):
        self.dmd_dir = Path(dmd_dir)
        self.context = context
        self.augment = augment
        self.normalize_session = normalize_session
        self.calibrate_user = calibrate_user
        self.ablate_group = ablate_group
        self.input_dim = 20

        # Collect all (npy_path, window_idx, label) tuples
        self.samples: List[Tuple[Path, int, int]] = []
        self._label_counts: dict = {}

        label_files = sorted(self.dmd_dir.glob("*_labels.json"))
        for lf in label_files:
            with open(lf, "r", encoding="utf-8") as f:
                data = json.load(f)

            session_id = data["session_id"]
            npy_path = self.dmd_dir / f"{session_id}.npy"
            if not npy_path.exists():
                continue

            for wl in data.get("window_labels", []):
                label = int(wl["label"])
                win_idx = int(wl["window"])
                self.samples.append((npy_path, win_idx, label))
                self._label_counts[label] = self._label_counts.get(label, 0) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, win_idx, label = self.samples[idx]
        all_features = np.load(str(npy_path)).astype(np.float32)  # (N, 20)

        # Apply ablation first
        if self.ablate_group:
            all_features = ablate_features(all_features, self.ablate_group)

        # Apply normalization/calibration
        if self.normalize_session:
            all_features = normalize_session_features(all_features)
        elif self.calibrate_user:
            mean, std = compute_baseline(all_features)
            all_features = apply_zscore(all_features, mean, std)

        n_windows = all_features.shape[0]

        # Extract context window centered on win_idx
        half = self.context // 2
        start = win_idx - half
        end = start + self.context

        # Build padded context
        context_features = np.zeros((self.context, self.input_dim), dtype=np.float32)
        for i in range(self.context):
            src_idx = start + i
            if 0 <= src_idx < n_windows:
                context_features[i] = all_features[src_idx]

        if self.augment:
            context_features = self._apply_augmentation(context_features)

        x = torch.from_numpy(context_features)  # (context, 20)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def _apply_augmentation(self, features: np.ndarray) -> np.ndarray:
        """Gaussian noise + random window masking."""
        noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
        features = features + noise
        mask_prob = 0.10
        mask = np.random.random(features.shape[0]) > mask_prob
        features = features * mask[:, np.newaxis]
        return features

    @property
    def label_counts(self):
        return dict(self._label_counts)

    def __repr__(self):
        return (
            f"DMDWindowDataset(samples={len(self.samples)}, "
            f"context={self.context}, labels={self._label_counts}, "
            f"normalize={self.normalize_session}, calibrate={self.calibrate_user})"
        )

"""Multiple Instance Learning dataset loaders.
Groups pre-extracted window sequences into session-level bags.
"""
import csv
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.calibration.normalization import (
    compute_baseline,
    apply_zscore,
    normalize_session_features,
    ablate_features,
)


class MILDataset(Dataset):
    """Generic MIL dataset wrapping pre-constructed bags."""
    def __init__(self, bags: List[Tuple[torch.Tensor, int]]):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        windows, label = self.bags[idx]
        return torch.as_tensor(windows, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class DMDMILDataset(Dataset):
    """Multiple Instance Learning dataset for DMD drowsiness.
    
    Each bag represents one session/video.
    Instances are sliding context windows centered on each window of the session.
    """
    def __init__(
        self,
        dmd_dir: str,
        context: int = 5,
        normalize_session: bool = False,
        calibrate_user: bool = False,
        ablate_group: Optional[str] = None,
    ):
        self.dmd_dir = Path(dmd_dir)
        self.context = context
        self.normalize_session = normalize_session
        self.calibrate_user = calibrate_user
        self.ablate_group = ablate_group
        self.input_dim = 20

        self.bags: List[torch.Tensor] = []
        self.bag_labels: List[float] = []
        self.session_ids: List[str] = []

        manifest_path = self.dmd_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row["session_id"]
                label = float(row["label"])  # Bag level label (0 or 1)
                npy_path = self.dmd_dir / f"{session_id}.npy"
                if not npy_path.exists():
                    continue

                features = np.load(str(npy_path)).astype(np.float32)  # (N, 20)

                # Apply feature ablation first
                if self.ablate_group:
                    features = ablate_features(features, self.ablate_group)

                # Apply normalization/calibration
                if self.normalize_session:
                    features = normalize_session_features(features)
                elif self.calibrate_user:
                    mean, std = compute_baseline(features)
                    features = apply_zscore(features, mean, std)

                n_windows = features.shape[0]

                # Group instances for this bag
                bag_instances = []
                half = self.context // 2
                for win_idx in range(n_windows):
                    start = win_idx - half
                    # Build padded context
                    context_features = np.zeros((self.context, self.input_dim), dtype=np.float32)
                    for i in range(self.context):
                        src_idx = start + i
                        if 0 <= src_idx < n_windows:
                            context_features[i] = features[src_idx]
                    bag_instances.append(context_features)

                # Shape: (n_instances, context, input_dim)
                self.bags.append(torch.tensor(np.array(bag_instances), dtype=torch.float32))
                self.bag_labels.append(label)
                self.session_ids.append(session_id)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], torch.tensor(self.bag_labels[idx], dtype=torch.float32)

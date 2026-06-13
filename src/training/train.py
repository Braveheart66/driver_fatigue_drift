import os
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

"""PyTorch Lightning training harness for ShortTermEncoder and DriftModel.

Supports:
  - Demo mode with synthetic data (--demo)
  - Real data from YawDD manifest (--dataset yawdd)
  - Real data from UTA-RLDD manifest (--dataset uta-rldd)
  - Class-weighted loss for imbalanced datasets (--weighted)
  - Cross-dataset evaluation (--eval-cross)
  - Feature normalization and per-user calibration (--normalize / --calibrate)
  - Feature ablation studies (--ablate <group>)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score,
)

from src.models.short_encoder import ShortTermEncoder
from src.models.drift_model import DriftModel
from src.training.supervised_dataset import (
    SupervisedDataset,
    build_yawdd_dataset,
    build_uta_rldd_dataset,
    build_dmd_dataset,
)
from src.utils.wandb_utils import init_wandb, log, finish, HAS_WANDB


class RandomSequenceDataset(Dataset):
    def __init__(self, n_samples=100, seq_len=60, input_dim=20):
        self.n = n_samples
        self.seq_len = seq_len
        self.input_dim = input_dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(self.seq_len, self.input_dim)
        y = torch.randint(0, 2, (1,)).float().squeeze()
        return x, y


def compute_pos_weight(dataset):
    """Compute positive class weight for BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive
    This upweights the minority class (yawn/drowsy) in the loss.
    """
    counts = dataset.label_counts
    n_neg = counts.get(0, 1)
    n_pos = counts.get(1, 1)
    weight = n_neg / max(n_pos, 1)
    print(f"  Class weights: neg={n_neg}, pos={n_pos}, pos_weight={weight:.3f}")
    return torch.tensor([weight], dtype=torch.float32)


class SupervisedModule(pl.LightningModule):
    """Binary classifier wrapping ShortTermEncoder."""

    def __init__(self, encoder: ShortTermEncoder, lr=1e-3, weight_decay=1e-4,
                 pos_weight=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.fc.out_features, 1)
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        # Collect predictions for epoch-level metrics
        self._val_preds = []
        self._val_labels = []

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self._val_preds.extend(preds.cpu().numpy().tolist())
        self._val_labels.extend(y.cpu().numpy().tolist())
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            preds = np.array(self._val_preds)
            labels = np.array(self._val_labels)
            f1 = f1_score(labels, preds, zero_division=0)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            self.log('val_f1', f1)
            self.log('val_precision', prec)
            self.log('val_recall', rec)
        self._val_preds.clear()
        self._val_labels.clear()

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        epoch = self.current_epoch
        t_loss = metrics.get('train_loss', float('nan'))
        t_acc = metrics.get('train_acc', float('nan'))
        v_loss = metrics.get('val_loss', float('nan'))
        v_acc = metrics.get('val_acc', float('nan'))
        v_f1 = metrics.get('val_f1', float('nan'))
        print(f"  Epoch {epoch:3d} | train_loss={t_loss:.4f} train_acc={t_acc:.4f} "
              f"| val_loss={v_loss:.4f} val_acc={v_acc:.4f} val_f1={v_f1:.4f}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return [opt], [sched]


def run_demo_training(epochs=1, batch_size=8):
    """Validate pipeline with synthetic data."""
    enc = ShortTermEncoder()
    model = SupervisedModule(enc)

    ds = RandomSequenceDataset(n_samples=64)
    dl = DataLoader(ds, batch_size=batch_size)

    wandb_run = init_wandb(project='fatigue-drift', name='demo_train')
    logger = None
    try:
        if HAS_WANDB:
            logger = WandbLogger(project='fatigue-drift', name='demo_train', log_model=False)
    except Exception:
        logger = None

    trainer = pl.Trainer(max_epochs=epochs, logger=logger if logger is not None else False,
                         enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model, dl)
    finish()


def evaluate_cross_dataset(model, dataset_name, seq_len, root=".",
                           normalize_session=False, calibrate_user=False, ablate_group=None):
    """Evaluate a trained model on a different dataset (cross-dataset generalization)."""
    print(f"\n{'='*60}")
    print(f"CROSS-DATASET EVALUATION on {dataset_name}")
    print(f"{'='*60}")

    if dataset_name == "uta-rldd":
        ds = build_uta_rldd_dataset(
            root=root, seq_len=seq_len, augment=False,
            normalize_session=normalize_session, calibrate_user=calibrate_user,
            ablate_group=ablate_group
        )
    elif dataset_name == "yawdd":
        ds = build_yawdd_dataset(
            root=root, seq_len=seq_len, augment=False,
            normalize_session=normalize_session, calibrate_user=calibrate_user,
            ablate_group=ablate_group
        )
    else:
        print(f"  Unknown dataset: {dataset_name}")
        return

    print(f"  Dataset: {ds}")
    if len(ds) == 0:
        print("  [ERROR] Empty dataset!")
        return

    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dl:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  Results on {dataset_name}:")
    print(f"  {'='*40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred=0  Pred=1")
    print(f"  Actual=0  {cm[0][0]:7d} {cm[0][1]:7d}")
    if len(cm) > 1:
        print(f"  Actual=1  {cm[1][0]:7d} {cm[1][1]:7d}")

    print(f"\n  Classification Report:")
    target_names = ['alert/no-yawn (0)', 'drowsy/yawn (1)']
    print(classification_report(all_labels, all_preds, target_names=target_names,
                                zero_division=0))

    # Confidence distribution
    print(f"  Prediction confidence stats:")
    print(f"    Mean prob:   {all_probs.mean():.4f}")
    print(f"    Std prob:    {all_probs.std():.4f}")
    print(f"    Min prob:    {all_probs.min():.4f}")
    print(f"    Max prob:    {all_probs.max():.4f}")

    return {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}


def run_real_training(dataset_name, epochs, batch_size, lr, root=".",
                      weighted=False, eval_cross=False,
                      normalize_session=False, calibrate_user=False, ablate_group=None,
                      hidden_dim=64, dropout=0.3, num_layers=2, context=5):
    """Train on real pre-extracted data."""
    print(f"\n=== Training on {dataset_name} ===")
    print(f"Params: hidden_dim={hidden_dim}, dropout={dropout}, num_layers={num_layers}, context={context}")
    print(f"Flags: normalize={normalize_session}, calibrate={calibrate_user}, ablate={ablate_group}")

    # Build dataset
    if dataset_name == "yawdd":
        ds = build_yawdd_dataset(
            root=root, seq_len=12, augment=True,
            normalize_session=normalize_session, calibrate_user=calibrate_user,
            ablate_group=ablate_group
        )
        seq_len = 12
    elif dataset_name == "uta-rldd":
        ds = build_uta_rldd_dataset(
            root=root, seq_len=60, augment=False,
            normalize_session=normalize_session, calibrate_user=calibrate_user,
            ablate_group=ablate_group
        )
        seq_len = 60
    elif dataset_name == "dmd":
        ds = build_dmd_dataset(
            root=root, context=context, augment=True,
            normalize_session=normalize_session, calibrate_user=calibrate_user,
            ablate_group=ablate_group
        )
        seq_len = context  # DMDWindowDataset uses context
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Dataset: {ds}")

    if len(ds) == 0:
        print("[ERROR] Dataset is empty! Check manifest paths and labels.")
        return

    # Compute class weight before splitting
    pos_weight = None
    if weighted:
        pos_weight = compute_pos_weight(ds)

    # Split: 80% train, 20% val
    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    enc = ShortTermEncoder(
        input_dim=20, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers
    )
    model = SupervisedModule(enc, lr=lr, pos_weight=pos_weight)

    # Callbacks
    checkpoint_dir = Path('models')
    checkpoint_dir.mkdir(exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_f1', patience=15, mode='max'),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f'{dataset_name}-best-{{epoch:02d}}-{{val_f1:.4f}}',
            monitor='val_f1',
            mode='max',
            save_top_k=1,
        ),
    ]

    # Logger
    wandb_run = init_wandb(project='fatigue-drift', name=f'train_{dataset_name}_weighted')
    logger = None
    try:
        if HAS_WANDB:
            logger = WandbLogger(
                project='fatigue-drift', name=f'train_{dataset_name}_weighted',
                log_model=False
            )
    except Exception:
        logger = None

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,  # Avoid tqdm Unicode crash on Windows cp1252
    )
    trainer.fit(model, train_dl, val_dl)

    finish()
    print(f"\n=== Training complete ===")

    # Print best checkpoint path
    best_path = callbacks[1].best_model_path
    print(f"  Best checkpoint: {best_path}")
    print(f"  Best val_f1:     {callbacks[1].best_model_score:.4f}")

    # Cross-dataset evaluation
    if eval_cross:
        # Load best checkpoint
        if best_path and Path(best_path).exists():
            print(f"\n  Loading best checkpoint for cross-dataset eval...")
            best_enc = ShortTermEncoder(
                input_dim=20, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers
            )
            best_model = SupervisedModule.load_from_checkpoint(
                best_path, encoder=best_enc,
                strict=False,  # pos_weight buffer may differ
            )
        else:
            best_model = model

        if dataset_name == "yawdd":
            evaluate_cross_dataset(
                best_model, "uta-rldd", seq_len=seq_len, root=root,
                normalize_session=normalize_session, calibrate_user=calibrate_user,
                ablate_group=ablate_group
            )
        elif dataset_name == "uta-rldd":
            evaluate_cross_dataset(
                best_model, "yawdd", seq_len=seq_len, root=root,
                normalize_session=normalize_session, calibrate_user=calibrate_user,
                ablate_group=ablate_group
            )
        elif dataset_name == "dmd":
            evaluate_cross_dataset(
                best_model, "uta-rldd", seq_len=seq_len, root=root,
                normalize_session=normalize_session, calibrate_user=calibrate_user,
                ablate_group=ablate_group
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    parser.add_argument('--dataset', type=str, choices=['yawdd', 'uta-rldd', 'dmd'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--weighted', action='store_true',
                        help='Use class-weighted loss for imbalanced datasets')
    parser.add_argument('--eval-cross', action='store_true',
                        help='Run cross-dataset evaluation after training')
    
    # Custom hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0.3, help='LSTM dropout rate')
    parser.add_argument('--num-layers', type=int, default=2, help='LSTM number of layers')
    parser.add_argument('--context', type=int, default=5, help='DMD context window size')

    # Normalization & Calibration flags
    parser.add_argument('--normalize', action='store_true', help='Enable per-session z-score normalization')
    parser.add_argument('--calibrate', action='store_true', help='Enable per-user baseline calibration')
    parser.add_argument('--ablate', type=str, choices=['eye', 'mouth', 'head', 'expression', 'gaze'],
                        help='Ablate a specific group of features (zero out)')

    args = parser.parse_args()

    if args.demo:
        run_demo_training(epochs=args.epochs, batch_size=args.batch)
    elif args.dataset:
        run_real_training(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            root=args.root,
            weighted=args.weighted,
            eval_cross=args.eval_cross,
            normalize_session=args.normalize,
            calibrate_user=args.calibrate,
            ablate_group=args.ablate,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_layers=args.num_layers,
            context=args.context,
        )
    else:
        print("Specify --demo or --dataset <name>. Use --help for options.")

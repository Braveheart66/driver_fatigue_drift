import os
import sys
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import logging
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import optuna


from src.models.short_encoder import ShortTermEncoder
from src.training.supervised_dataset import (
    build_yawdd_dataset,
    build_uta_rldd_dataset,
    build_dmd_dataset,
)
from src.training.train import SupervisedModule, compute_pos_weight

# Suppress warnings and loggers to keep output clean
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class OptunaF1Callback(pl.Callback):
    """Callback to track the best val_f1 during the trial."""
    def __init__(self):
        super().__init__()
        self.best_f1 = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_f1 = metrics.get('val_f1')
        if val_f1 is not None:
            val_f1 = val_f1.item()
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['yawdd', 'uta-rldd', 'dmd'], required=True,
                        help='Dataset to tune on')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs per trial')
    parser.add_argument('--trials', type=int, default=30, help='Number of tuning trials')
    parser.add_argument('--root', type=str, default='.', help='Project root directory')
    parser.add_argument('--weighted', action='store_true', help='Use class weights')
    parser.add_argument('--normalize', action='store_true', help='Normalize sessions')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate per user')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    def objective(trial):
        # Hyperparameter search space
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        num_layers = trial.suggest_int('num_layers', 1, 3)

        # Dataset specific parameters
        if args.dataset == "yawdd":
            seq_len = trial.suggest_int('seq_len', 5, 20)
            ds = build_yawdd_dataset(
                root=args.root, seq_len=seq_len, augment=True,
                normalize_session=args.normalize, calibrate_user=args.calibrate
            )
        elif args.dataset == "uta-rldd":
            seq_len = trial.suggest_int('seq_len', 10, 60)
            ds = build_uta_rldd_dataset(
                root=args.root, seq_len=seq_len, augment=True,
                normalize_session=args.normalize, calibrate_user=args.calibrate
            )
        elif args.dataset == "dmd":
            context = trial.suggest_int('context', 3, 10)
            ds = build_dmd_dataset(
                root=args.root, context=context, augment=True,
                normalize_session=args.normalize, calibrate_user=args.calibrate
            )

        if len(ds) == 0:
            return 0.0

        pos_weight = None
        if args.weighted:
            pos_weight = compute_pos_weight(ds)

        n_val = max(1, int(len(ds) * 0.2))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

        # Initialize model
        enc = ShortTermEncoder(
            input_dim=20, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers
        )
        model = SupervisedModule(enc, lr=lr, pos_weight=pos_weight)

        f1_cb = OptunaF1Callback()

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=[f1_cb],
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model, train_dl, val_dl)
        return f1_cb.best_f1

    print(f"\n--- Starting Optuna Tuning Study on {args.dataset} ---")
    print(f"Trials: {args.trials}, Epochs: {args.epochs}, GPU: {torch.cuda.is_available()}")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "="*40)
    print("OPTUNA TUNING STUDY COMPLETE")
    print("="*40)
    print(f"Best Trial F1: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*40)


if __name__ == '__main__':
    main()

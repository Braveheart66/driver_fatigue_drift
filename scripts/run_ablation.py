import sys
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.models.short_encoder import ShortTermEncoder
from src.training.supervised_dataset import build_dmd_dataset
from src.training.train import SupervisedModule, compute_pos_weight
from scripts.tune_optuna import OptunaF1Callback

import warnings
warnings.filterwarnings("ignore")

# Hyperparameters from Optuna best trial
LR = 0.0076
HIDDEN_DIM = 64
DROPOUT = 0.475
NUM_LAYERS = 3
CONTEXT = 3
EPOCHS = 15


def main():
    groups = [None, 'eye', 'mouth', 'head', 'expression', 'gaze']
    results = {}

    for g in groups:
        group_name = g if g is not None else "None (Full Model)"
        print(f"\n--- Running training with ablated group: {group_name} ---")
        
        ds = build_dmd_dataset(
            root='.',
            context=CONTEXT,
            augment=True,
            normalize_session=True,
            ablate_group=g
        )
        
        n_val = max(1, int(len(ds) * 0.2))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        
        enc = ShortTermEncoder(
            input_dim=20,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            num_layers=NUM_LAYERS
        )
        pos_weight = compute_pos_weight(ds)
        model = SupervisedModule(enc, lr=LR, pos_weight=pos_weight)
        
        cb = OptunaF1Callback()
        
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=[cb],
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False
        )
        trainer.fit(model, train_dl, val_dl)
        results[g] = cb.best_f1
        print(f"Best Val F1 for {group_name}: {cb.best_f1:.4f}")

    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS:")
    print("="*50)
    print(f"| Ablated Group      | Val F1 | F1 Drop |")
    print(f"| ------------------ | ------ | ------- |")
    full_f1 = results[None]
    print(f"| None (Full Model)  | {full_f1:.4f} |  0.0000 |")
    for g in ['eye', 'mouth', 'head', 'expression', 'gaze']:
        f1 = results[g]
        drop = full_f1 - f1
        print(f"| {g:<18} | {f1:.4f} | {drop:7.4f} |")
    print("="*50)


if __name__ == '__main__':
    main()

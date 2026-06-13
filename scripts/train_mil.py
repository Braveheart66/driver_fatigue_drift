import os
import sys
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.models.short_encoder import ShortTermEncoder
from src.models.mil import AttentionMIL
from src.training.mil_dataset import DMDMILDataset
from src.utils.wandb_utils import init_wandb, finish, HAS_WANDB


class MILModule(pl.LightningModule):
    """Lightning Module for AttentionMIL training."""
    def __init__(self, mil_model: AttentionMIL, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = mil_model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self._val_preds = []
        self._val_labels = []

    def forward(self, bag):
        return self.model(bag)

    def training_step(self, batch, batch_idx):
        bag, y = batch  # bag: (1, n_instances, context, input_dim), y: (1,)
        bag = bag.squeeze(0)
        
        logits, _ = self(bag)
        loss = self.loss_fn(logits.view(-1), y.view(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bag, y = batch
        bag = bag.squeeze(0)
        
        logits, _ = self(bag)
        loss = self.loss_fn(logits.view(-1), y.view(-1))
        self.log('val_loss', loss, prog_bar=True)
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        self._val_preds.extend(preds.view(-1).cpu().numpy().tolist())
        self._val_labels.extend(y.view(-1).cpu().numpy().tolist())
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            preds = np.array(self._val_preds)
            labels = np.array(self._val_labels)
            f1 = f1_score(labels, preds, zero_division=0)
            acc = accuracy_score(labels, preds)
            self.log('val_f1', f1, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return [opt], [sched]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--context', type=int, default=5, help='Context window size')
    parser.add_argument('--freeze', action='store_true', help='Freeze pretrained encoder weights')
    parser.add_argument('--encoder-checkpoint', type=str, default='models/dmd-best-epoch=01-val_f1=0.5812.ckpt',
                        help='Pretrained supervised encoder checkpoint')
    parser.add_argument('--normalize', action='store_true', help='Enable per-session z-score normalization')
    parser.add_argument('--calibrate', action='store_true', help='Enable per-user baseline calibration')
    parser.add_argument('--ablate', type=str, choices=['eye', 'mouth', 'head', 'expression', 'gaze'],
                        help='Ablate feature group')
    parser.add_argument('--root', type=str, default='.', help='Project root')
    args = parser.parse_args()

    dmd_dir = Path(args.root) / 'data' / 'processed' / 'dmd-drowsiness'
    print(f"\n=== Training MIL Model on DMD ===")
    print(f"Flags: normalize={args.normalize}, calibrate={args.calibrate}, ablate={args.ablate}, freeze={args.freeze}")

    # Build dataset
    ds = DMDMILDataset(
        dmd_dir=str(dmd_dir),
        context=args.context,
        normalize_session=args.normalize,
        calibrate_user=args.calibrate,
        ablate_group=args.ablate,
    )
    print(f"Loaded {len(ds)} bags.")

    # Split dataset
    n_val = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Load pretrained encoder
    print(f"Loading pretrained encoder from: {args.encoder_checkpoint}")
    checkpoint = torch.load(args.encoder_checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Auto-detect encoder params
    hidden_dim = 64
    num_layers = 2
    if 'encoder.lstm.weight_ih_l0' in state_dict:
        hidden_dim = state_dict['encoder.lstm.weight_ih_l0'].shape[0] // 4
    lstm_keys = [k for k in state_dict.keys() if k.startswith('encoder.lstm.weight_ih_')]
    is_bidirectional = any('reverse' in k for k in state_dict.keys() if 'lstm' in k)
    num_layers = len(lstm_keys) // 2 if is_bidirectional else len(lstm_keys)
    print(f"Detected encoder: hidden_dim={hidden_dim}, num_layers={num_layers}, bidirectional={is_bidirectional}")


    encoder = ShortTermEncoder(input_dim=20, hidden_dim=hidden_dim, num_layers=num_layers)
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            encoder_state_dict[k.replace('encoder.', '')] = v
    encoder.load_state_dict(encoder_state_dict)

    if args.freeze:
        print("Freezing encoder parameters.")
        for p in encoder.parameters():
            p.requires_grad = False

    mil_model = AttentionMIL(encoder=encoder, embedding_dim=hidden_dim * 2)
    model = MILModule(mil_model, lr=args.lr)

    checkpoint_dir = Path('models')
    checkpoint_dir.mkdir(exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_f1', patience=15, mode='max'),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='mil-best-{epoch:02d}-{val_f1:.4f}',
            monitor='val_f1',
            mode='max',
            save_top_k=1,
        ),
    ]

    logger = None
    try:
        if HAS_WANDB:
            logger = WandbLogger(project='fatigue-drift-mil', name='train_mil', log_model=False)
    except Exception:
        logger = None

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dl, val_dl)
    finish()

    best_path = callbacks[1].best_model_path
    best_score = callbacks[1].best_model_score
    print(f"\nMIL Training Complete.")
    print(f"Best Checkpoint: {best_path}")
    print(f"Best Val F1:     {best_score:.4f}")

    # Load best model for temporal attribution extraction
    if best_path and Path(best_path).exists():
        print(f"Loading best model for temporal attribution analysis...")
        best_module = MILModule.load_from_checkpoint(
            best_path, mil_model=AttentionMIL(encoder=encoder, embedding_dim=hidden_dim * 2)
        )
    else:
        best_module = model

    best_module.eval()
    device = next(best_module.parameters()).device

    # Extract attention weights for all sessions
    attribution_results = {}
    
    # Create DataLoader for all bags
    all_dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for idx, (bag, label) in enumerate(all_dl):
            session_id = ds.session_ids[idx]
            bag = bag.squeeze(0).to(device)
            logits, attn_weights = best_module(bag)
            prob = torch.sigmoid(logits).item()
            attn = attn_weights.squeeze(-1).cpu().numpy().tolist()
            
            attribution_results[session_id] = {
                'label': int(label.item()),
                'prob': prob,
                'attention_weights': attn
            }

    # Save to JSON
    out_json = checkpoint_dir / 'mil_attention_weights.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(attribution_results, f, indent=2)
    print(f"Saved attention weights to: {out_json}")

    # Try plotting
    try:
        import matplotlib.pyplot as plt
        print("Generating attention attribution heatmap...")
        
        # Select the session with highest attention weight variance or highest probability
        # Let's plot the first drowsy session and first alert session
        drowsy_sessions = [s for s, data in attribution_results.items() if data['label'] == 1]
        alert_sessions = [s for s, data in attribution_results.items() if data['label'] == 0]
        
        to_plot = []
        if drowsy_sessions:
            to_plot.append((drowsy_sessions[0], "Drowsy Session"))
        if alert_sessions:
            to_plot.append((alert_sessions[0], "Alert Session"))
            
        fig, axes = plt.subplots(len(to_plot), 1, figsize=(10, 4 * len(to_plot)), sharex=False)
        if len(to_plot) == 1:
            axes = [axes]
            
        for i, (session_id, title) in enumerate(to_plot):
            data = attribution_results[session_id]
            attn = np.array(data['attention_weights'])
            x = np.arange(len(attn)) * 5  # 5 seconds per window
            
            axes[i].plot(x, attn, marker='o', color='crimson' if data['label'] == 1 else 'forestgreen', linewidth=2)
            axes[i].fill_between(x, attn, alpha=0.3, color='crimson' if data['label'] == 1 else 'forestgreen')
            axes[i].set_title(f"{title}: {session_id} (Prob: {data['prob']:.3f})")
            axes[i].set_ylabel("Attention Weight")
            axes[i].set_xlabel("Time (seconds)")
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        out_png = checkpoint_dir / 'mil_attention_heatmap.png'
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved attention heatmap plot to: {out_png}")
    except Exception as e:
        print(f"Could not generate plot (matplotlib may be missing or running headless): {e}")


if __name__ == '__main__':
    main()

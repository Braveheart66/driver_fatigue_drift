"""Demo training script that logs synthetic metrics to Weights & Biases.

This is intended to validate WandB integration and demonstrate logging.
"""
import argparse
import random
import time

from src.utils.wandb_utils import init_wandb, log, finish


def run_demo(epochs: int = 5, steps_per_epoch: int = 10, project: str = 'fatigue-drift-demo'):
    run = init_wandb(project=project, name='demo_run')

    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for step in range(1, steps_per_epoch + 1):
            # synthetic loss that slowly decreases
            loss = max(0.01, 1.0 / (ep + step * 0.1) + random.uniform(-0.02, 0.02))
            epoch_loss += loss
            log({'train/loss': loss}, step=(ep - 1) * steps_per_epoch + step)
            time.sleep(0.01)

        avg_loss = epoch_loss / steps_per_epoch
        # synthetic val f1 improves over epochs
        val_f1 = min(0.99, 0.5 + ep * 0.08 + random.uniform(-0.02, 0.02))
        log({'epoch': ep, 'train/avg_loss': avg_loss, 'val/f1': val_f1}, step=ep)
        print(f'Epoch {ep}: avg_loss={avg_loss:.4f}, val_f1={val_f1:.3f}')

    finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--project', type=str, default='fatigue-drift-demo')
    args = parser.parse_args()
    run_demo(epochs=args.epochs, steps_per_epoch=args.steps, project=args.project)


if __name__ == '__main__':
    main()

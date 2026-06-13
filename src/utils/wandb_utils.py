"""Utility wrappers for Weights & Biases integration.
Provides safe offline fallback when no API key is present.
"""
import os
from typing import Optional, Dict

try:
    import wandb
    HAS_WANDB = True
except Exception:
    wandb = None
    HAS_WANDB = False


def init_wandb(project: str = 'fatigue-drift', name: Optional[str] = None, config: Optional[Dict] = None, offline_if_no_key: bool = True):
    """Initialize a wandb run. If no WANDB_API_KEY is set and `offline_if_no_key` is True,
    initializes wandb in offline mode (local logging).
    Returns the wandb.run object or None if wandb is not installed.
    """
    if not HAS_WANDB:
        print('wandb not installed; skipping initialization')
        return None

    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key and offline_if_no_key:
        os.environ['WANDB_MODE'] = 'offline'

    run = wandb.init(project=project, name=name, config=config, reinit=True)
    return run


def log(metrics: Dict, step: Optional[int] = None):
    if not HAS_WANDB:
        return
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish():
    if not HAS_WANDB:
        return
    try:
        wandb.finish()
    except Exception:
        pass

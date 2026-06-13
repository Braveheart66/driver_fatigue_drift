"""Monte Carlo Dropout scoring and confidence estimation.

Runs multiple stochastic forward passes with dropout active to produce
a mean prediction and a confidence score (inverse of variance).
"""
import numpy as np
import torch


def predict_with_confidence(
    drift_model,
    embeddings: torch.Tensor,
    n_passes: int = 20,
) -> tuple:
    """Run MC Dropout inference to estimate score mean and confidence.

    Args:
        drift_model: DriftModel instance (must have Dropout layers).
        embeddings: (batch, seq, embedding_dim) tensor.
        n_passes: Number of stochastic forward passes.

    Returns:
        Tuple of (mean_score, confidence) as numpy scalars or arrays.
        - mean_score: Average fatigue score (0-100).
        - confidence: 0-100 scale where 100 = fully confident (no variance).
    """
    drift_model.train()  # Activate dropout during inference
    scores = []

    with torch.no_grad():
        for _ in range(n_passes):
            fatigue, _ = drift_model(embeddings)
            # Take the latest timestep's score
            scores.append(fatigue[:, -1, 0].cpu().numpy())

    scores = np.array(scores)  # (n_passes, batch)
    mean = scores.mean(axis=0)
    variance = scores.var(axis=0)

    # Normalize variance to 0-1 confidence
    # Max expected variance for a 0-100 scale score is ~50
    confidence = (1.0 - np.clip(variance / 50.0, 0.0, 1.0)) * 100.0

    return mean, confidence

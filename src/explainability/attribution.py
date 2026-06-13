"""Explainability via Integrated Gradients (Captum).

Computes per-feature and per-timestep attributions that explain
which facial signals and time windows drove the fatigue score.
"""
import numpy as np
import torch

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

FEATURE_NAMES = [
    'EAR mean', 'EAR std', 'PERCLOS', 'Blink rate', 'Blink duration',
    'Microsleeps', 'MAR mean', 'MAR max', 'Yawn', 'Head pitch mean',
    'Head pitch std', 'Head yaw', 'Head roll', 'Nod frequency',
    'AU6 proxy', 'AU12 proxy', 'Expression var (AU6)', 'Expression var (AU12)',
    'Gaze stability X', 'Gaze stability Y',
]


def compute_attributions(
    encoder,
    drift_model,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor = None,
    n_steps: int = 50,
) -> dict:
    """Compute feature and temporal attributions using Integrated Gradients.

    Args:
        encoder: ShortTermEncoder model.
        drift_model: DriftModel that takes encoder embeddings.
        input_tensor: (1, seq_len, 20) current session windows.
        baseline_tensor: (1, seq_len, 20) user's baseline (reference).
            If None, uses a zero tensor.
        n_steps: Number of interpolation steps for IG.

    Returns:
        Dict with keys:
            'feature_importance': (20,) array — per-feature contribution %.
            'temporal_importance': (seq_len,) array — per-window contribution %.
            'top_features': list of (name, pct) tuples sorted descending.
            'raw_attributions': (seq_len, 20) numpy array.
    """
    if not HAS_CAPTUM:
        try:
            return _gradient_fallback(encoder, drift_model, input_tensor)
        except Exception:
            return _absolute_fallback(input_tensor)

    try:
        if baseline_tensor is None:
            baseline_tensor = torch.zeros_like(input_tensor)

        encoder.eval()
        drift_model.eval()

        def forward_fn(x):
            embeddings_list = []
            for i in range(x.shape[1]):
                start_idx = max(0, i - 2)
                window = x[:, start_idx:i+1, :]
                emb = encoder(window)
                embeddings_list.append(emb)
            embeddings = torch.stack(embeddings_list, dim=1)
            score, _ = drift_model(embeddings)
            return score[:, -1, :]

        ig = IntegratedGradients(forward_fn)
        attributions, delta = ig.attribute(
            inputs=input_tensor,
            baselines=baseline_tensor,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        attr_np = attributions.detach().cpu().numpy().squeeze(0)  # (seq_len, 20)
        return _format_attributions(attr_np)
    except Exception as e:
        print(f"[Attribution] IntegratedGradients failed: {e}. Falling back to gradient approximation.")
        try:
            return _gradient_fallback(encoder, drift_model, input_tensor)
        except Exception as e2:
            print(f"[Attribution] Chained fallbacks failed: {e2}. Using absolute fallback.")
            return _absolute_fallback(input_tensor)


def _absolute_fallback(input_tensor: torch.Tensor) -> dict:
    """Absolute backup fallback when all attribution methods fail."""
    num_features = len(FEATURE_NAMES)
    dummy_importance = np.ones(num_features) / num_features * 100.0
    top_features = [(FEATURE_NAMES[i], float(dummy_importance[i])) for i in range(num_features)]
    return {
        'feature_importance': dummy_importance,
        'temporal_importance': np.ones(input_tensor.shape[1]) / max(input_tensor.shape[1], 1) * 100.0,
        'top_features': top_features,
        'raw_attributions': np.zeros((input_tensor.shape[1], num_features)),
    }



def _gradient_fallback(encoder, drift_model, input_tensor: torch.Tensor) -> dict:
    """Simple gradient-based attribution when Captum is unavailable."""
    encoder.eval()
    drift_model.eval()
    input_tensor = input_tensor.detach().clone().requires_grad_(True)

    embeddings_list = []
    for i in range(input_tensor.shape[1]):
        start_idx = max(0, i - 2)
        window = input_tensor[:, start_idx:i+1, :]
        emb = encoder(window)
        embeddings_list.append(emb)
    embeddings = torch.stack(embeddings_list, dim=1)
    score, _ = drift_model(embeddings)
    target = score[:, -1, :].sum()
    target.backward()

    attr_np = input_tensor.grad.detach().cpu().numpy().squeeze(0)  # (seq_len, 20)
    return _format_attributions(attr_np)


def _format_attributions(attr_np: np.ndarray) -> dict:
    """Convert raw attribution array to structured result dict."""
    abs_attr = np.abs(attr_np)

    # Per-feature importance: sum over time
    feature_importance = abs_attr.sum(axis=0)  # (20,)
    feat_total = feature_importance.sum()
    if feat_total > 0:
        feature_importance = feature_importance / feat_total * 100.0

    # Per-timestep importance: sum over features
    temporal_importance = abs_attr.sum(axis=1)  # (seq_len,)
    temp_total = temporal_importance.sum()
    if temp_total > 0:
        temporal_importance = temporal_importance / temp_total * 100.0

    # Top features sorted
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_features = [
        (FEATURE_NAMES[i], float(feature_importance[i]))
        for i in sorted_idx
    ]

    return {
        'feature_importance': feature_importance,
        'temporal_importance': temporal_importance,
        'top_features': top_features,
        'raw_attributions': attr_np,
    }


def format_attribution_output(score: float, confidence: float, feature_importance: np.ndarray):
    """Print a human-readable attribution summary."""
    top_3 = np.argsort(feature_importance)[-3:][::-1]
    print(f"Fatigue Score:  {score:.0f}")
    print(f"Confidence:     {confidence:.0f}%")
    print("Main Contributors:")
    for idx in top_3:
        print(f"  {FEATURE_NAMES[idx]:30s}  +{feature_importance[idx]:.0f}%")

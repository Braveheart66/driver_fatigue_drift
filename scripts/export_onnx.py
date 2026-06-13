"""Export PyTorch models (ShortTermEncoder, DriftModel) to ONNX format.

Discovers training checkpoints in the models/ directory, automatically
extracts the architecture hyperparameters, loads weights, and exports
to models/encoder.onnx and models/drift_model.onnx.
"""
import os
import sys
import argparse
from pathlib import Path
import torch

# Add project root to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.short_encoder import ShortTermEncoder
from src.models.drift_model import DriftModel


def discover_hyperparameters(state_dict: dict) -> dict:
    """Dynamically determine encoder hyperparameters from a checkpoint's state dict."""
    keys = list(state_dict.keys())
    
    # Locate weight_ih_l0 key (handles lighting module wrapper prefixes)
    ih_l0_key = None
    for k in keys:
        if 'lstm.weight_ih_l0' in k:
            ih_l0_key = k
            break
            
    if ih_l0_key is None:
        raise ValueError("Could not find LSTM weights (weight_ih_l0) in checkpoint state dict.")
        
    weight_shape = state_dict[ih_l0_key].shape
    input_dim = weight_shape[1]
    hidden_dim = weight_shape[0] // 4
    
    # Count layers by identifying unique layer indices
    layers = set()
    for k in keys:
        if 'lstm.weight_ih_l' in k:
            parts = k.split('lstm.weight_ih_l')
            if len(parts) > 1:
                try:
                    layer_num = int(parts[1][0])
                    layers.add(layer_num)
                except ValueError:
                    pass
    num_layers = len(layers) if layers else 2
    
    # Locate fully connected output layer projection size
    fc_weight_key = None
    for k in keys:
        if 'fc.weight' in k:
            fc_weight_key = k
            break
            
    output_dim = 128
    if fc_weight_key is not None:
        output_dim = state_dict[fc_weight_key].shape[0]
        
    return {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'output_dim': output_dim
    }


def clean_state_dict(state_dict: dict, prefix: str = "encoder.") -> dict:
    """Filter and clean prefixes from state dict keys to fit standalone nn.Module."""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            cleaned[k[len(prefix):]] = v
        elif k.startswith("model.encoder."):
            cleaned[k[len("model.encoder."):]] = v
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to ShortTermEncoder checkpoint. If None, auto-detects best.")
    parser.add_argument('--output-dir', type=str, default='models',
                        help="Directory to save exported ONNX models.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Export ShortTermEncoder
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Auto-detect best checkpoint in models/
        ckpt_files = list(Path('models').glob('*.ckpt'))
        if ckpt_files:
            # Prefer DMD then yawdd checkpoints
            dmd_ckpts = [c for c in ckpt_files if 'dmd' in c.name]
            yawdd_ckpts = [c for c in ckpt_files if 'yawdd' in c.name]
            mil_ckpts = [c for c in ckpt_files if 'mil' in c.name]
            
            if dmd_ckpts:
                checkpoint_path = sorted(dmd_ckpts)[-1]
            elif mil_ckpts:
                checkpoint_path = sorted(mil_ckpts)[-1]
            elif yawdd_ckpts:
                checkpoint_path = sorted(yawdd_ckpts)[-1]
            else:
                checkpoint_path = sorted(ckpt_files)[-1]
            print(f"Auto-detected checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found in models/. Exporting randomly initialized encoder.")

    encoder = None
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading encoder checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        
        try:
            hparams = discover_hyperparameters(state_dict)
            print(f"Discovered Hyperparameters: {hparams}")
            encoder = ShortTermEncoder(
                input_dim=hparams['input_dim'],
                hidden_dim=hparams['hidden_dim'],
                num_layers=hparams['num_layers'],
                output_dim=hparams['output_dim']
            )
            # Try cleaning prefix "encoder." or "model.encoder."
            prefix = "encoder."
            if not any(k.startswith(prefix) for k in state_dict.keys()):
                if any(k.startswith("model.encoder.") for k in state_dict.keys()):
                    prefix = "model.encoder."
            
            cleaned_sd = clean_state_dict(state_dict, prefix=prefix)
            encoder.load_state_dict(cleaned_sd, strict=True)
            print("Successfully loaded weights into encoder.")
        except Exception as e:
            print(f"Error loading checkpoint weights: {e}. Falling back to default architecture.")
            encoder = None

    if encoder is None:
        print("Using default ShortTermEncoder architecture (input_dim=20, hidden_dim=64, num_layers=2, output_dim=128).")
        encoder = ShortTermEncoder()

    encoder.eval()
    encoder_onnx_path = output_dir / 'encoder.onnx'
    
    # 5-second sequence of 150 frames @ 30fps aggregated to 1 window
    dummy_input_encoder = torch.randn(1, 60, 20)  # (batch, sequence, input_dim)
    
    print(f"Exporting encoder to ONNX...")
    torch.onnx.export(
        encoder,
        dummy_input_encoder,
        str(encoder_onnx_path),
        opset_version=17,
        input_names=['feature_windows'],
        output_names=['embedding'],
        dynamic_axes={
            'feature_windows': {0: 'batch', 1: 'sequence'},
            'embedding': {0: 'batch'}
        }
    )
    print(f"Encoder exported to: {encoder_onnx_path}")

    # 2. Export DriftModel
    print("Instantiating DriftModel (embedding_dim=128, hidden_dim=64)...")
    drift_model = DriftModel(embedding_dim=128, hidden_dim=64)
    drift_model.eval()

    # Input: sequence of 12 to 360 windows (embeddings of dim 128)
    dummy_input_drift = torch.randn(1, 360, 128)  # (batch, sequence, embedding_dim)
    drift_onnx_path = output_dir / 'drift_model.onnx'

    print(f"Exporting drift model to ONNX...")
    torch.onnx.export(
        drift_model,
        dummy_input_drift,
        str(drift_onnx_path),
        opset_version=17,
        input_names=['embeddings'],
        output_names=['fatigue_scores', 'drift_index'],
        dynamic_axes={
            'embeddings': {0: 'batch', 1: 'sequence'},
            'fatigue_scores': {0: 'batch', 1: 'sequence'},
            'drift_index': {0: 'batch'}
        }
    )
    print(f"Drift model exported to: {drift_onnx_path}")

    # 3. Verification using onnx package if available
    try:
        import onnx
        print("\nVerifying exported ONNX models...")
        model_enc = onnx.load(str(encoder_onnx_path))
        onnx.checker.check_model(model_enc)
        print("Encoder ONNX check passed! [OK]")

        model_drift = onnx.load(str(drift_onnx_path))
        onnx.checker.check_model(model_drift)
        print("Drift model ONNX check passed! [OK]")
    except ImportError:
        print("\nonnx library not installed. Skipping model validation checks.")


if __name__ == '__main__':
    main()

import unittest

import importlib

TORCH_AVAILABLE = True
try:
    import torch
except Exception:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, 'torch not available')
class TestModelsSmoke(unittest.TestCase):
    def test_short_encoder_forward(self):
        from src.models.short_encoder import ShortTermEncoder
        enc = ShortTermEncoder()
        x = torch.randn(2, 60, 20)
        out = enc(x)
        self.assertEqual(out.shape, (2, 128))

    def test_drift_model_forward(self):
        from src.models.drift_model import DriftModel
        dm = DriftModel()
        emb = torch.randn(2, 360, 128)
        scores, drift = dm(emb)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 360)


if __name__ == '__main__':
    unittest.main()

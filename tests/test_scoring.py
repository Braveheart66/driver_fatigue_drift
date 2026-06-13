import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.cusum import CUSUMDetector
from src.models.scorer import predict_with_confidence
from src.explainability.attribution import compute_attributions, _gradient_fallback, FEATURE_NAMES


class DummyDriftModel(nn.Module):
    """A dummy model for testing MC Dropout scorer and attribution."""
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fatigue_out = nn.Linear(hidden_dim, 1)
        self.drift_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq, embedding_dim)
        out, hidden = self.gru(x)
        out = self.dropout(out)
        fatigue = torch.sigmoid(self.fatigue_out(out)) * 100.0
        drift = self.drift_out(hidden[-1])
        return fatigue, drift


class DummyEncoder(nn.Module):
    """A dummy encoder for testing explainability attribution."""
    def __init__(self, input_dim=20, output_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq, input_dim)
        out, _ = self.lstm(x)
        # return last representation
        return self.fc(out[:, -1, :])


class TestCUSUMDetector(unittest.TestCase):
    def test_cusum_stable(self):
        detector = CUSUMDetector(threshold=5.0, slack=1.0)
        # Steady inputs equal to reference
        for _ in range(5):
            state = detector.update(50.0, 50.0)
            self.assertEqual(state, 'STABLE')
        self.assertEqual(detector.pos, 0.0)
        self.assertEqual(detector.neg, 0.0)

    def test_cusum_onset(self):
        detector = CUSUMDetector(threshold=5.0, slack=0.5)
        # Incrementally increasing scores to trigger onset
        states = []
        for i in range(10):
            # score increases from 50 to 59
            state = detector.update(50.0 + i, 50.0)
            states.append(state)
        
        self.assertIn('FATIGUE_ONSET', states)
        self.assertGreaterEqual(len(detector.events), 1)
        self.assertEqual(detector.events[0][0], 'FATIGUE_ONSET')

    def test_cusum_recovery(self):
        detector = CUSUMDetector(threshold=5.0, slack=0.5)
        detector.current_state = 'FATIGUE'
        # Decreasing scores to trigger recovery
        states = []
        for i in range(10):
            state = detector.update(50.0 - i, 50.0)
            states.append(state)
        
        self.assertIn('RECOVERY', states)
        self.assertGreaterEqual(len(detector.events), 1)
        self.assertEqual(detector.events[0][0], 'RECOVERY')

    def test_resets(self):
        # Set a large threshold so update doesn't trigger a reset
        detector = CUSUMDetector(threshold=20.0, slack=0.5)
        detector.update(60.0, 50.0)
        self.assertGreater(detector.pos, 0.0)
        
        detector.reset()
        self.assertEqual(detector.pos, 0.0)
        self.assertGreater(len(detector.history), 0)
        
        detector.full_reset()
        self.assertEqual(len(detector.history), 0)
        self.assertEqual(len(detector.events), 0)


class TestScorer(unittest.TestCase):
    def test_predict_with_confidence(self):
        model = DummyDriftModel()
        embeddings = torch.randn(2, 10, 128)  # batch=2, seq=10
        
        mean, conf = predict_with_confidence(model, embeddings, n_passes=5)
        
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(conf.shape, (2,))
        self.assertTrue(np.all(mean >= 0.0) and np.all(mean <= 100.0))
        self.assertTrue(np.all(conf >= 0.0) and np.all(conf <= 100.0))


class TestAttribution(unittest.TestCase):
    def test_gradient_fallback(self):
        encoder = DummyEncoder()
        drift_model = DummyDriftModel()
        input_tensor = torch.randn(1, 10, 20)  # batch=1, seq=10, features=20
        
        res = _gradient_fallback(encoder, drift_model, input_tensor)
        
        self.assertIn('feature_importance', res)
        self.assertIn('temporal_importance', res)
        self.assertIn('top_features', res)
        
        self.assertEqual(res['feature_importance'].shape, (20,))
        self.assertEqual(res['temporal_importance'].shape, (10,))
        self.assertEqual(len(res['top_features']), 20)
        # Check sum close to 100 (relative tolerance)
        self.assertAlmostEqual(res['feature_importance'].sum(), 100.0, places=4)
        self.assertAlmostEqual(res['temporal_importance'].sum(), 100.0, places=4)

    def test_compute_attributions_dispatch(self):
        encoder = DummyEncoder()
        drift_model = DummyDriftModel()
        input_tensor = torch.randn(1, 5, 20)
        
        res = compute_attributions(encoder, drift_model, input_tensor, n_steps=5)
        
        self.assertEqual(res['feature_importance'].shape, (20,))
        self.assertEqual(res['temporal_importance'].shape, (5,))
        self.assertEqual(res['top_features'][0][0], FEATURE_NAMES[np.argmax(res['feature_importance'])])


if __name__ == '__main__':
    unittest.main()

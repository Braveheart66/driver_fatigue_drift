import unittest

from src.extraction.eye_features import compute_ear, detect_blinks


class TestEyeFeatures(unittest.TestCase):
    def test_compute_ear_basic(self):
        # Construct a simple symmetrical eye where vertical/horizontal ratios are known
        landmarks = [
            (0.0, 0.0),  # 0 left corner
            (0.0, 1.0),  # 1 top-left
            (0.5, 1.0),  # 2 top-right
            (1.0, 0.0),  # 3 right corner
            (0.5, -1.0), # 4 bottom-right
            (0.0, -1.0), # 5 bottom-left
        ]
        ear = compute_ear(landmarks, [0,1,2,3,4,5])
        self.assertGreater(ear, 0)

    def test_detect_blinks(self):
        seq = [0.3]*5 + [0.1]*3 + [0.3]*10 + [0.1]*4
        blinks = detect_blinks(seq, threshold=0.2, min_frames=2, max_frames=10, fps=30)
        # Expect two blinks detected
        self.assertGreaterEqual(len(blinks), 2)


if __name__ == '__main__':
    unittest.main()

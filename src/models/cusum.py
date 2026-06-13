"""CUSUM (Cumulative Sum) Changepoint Detector for fatigue onset/recovery.

Detects statistically significant shifts in fatigue score trajectory
by accumulating deviations from a running reference level.
"""


class CUSUMDetector:
    """Detects fatigue onset and recovery events using cumulative sum control charts.

    Attributes:
        threshold: Alarm threshold for accumulated deviation.
        slack: Allowable slack (minimum deviation to accumulate).
        pos: Positive accumulator (fatigue onset direction).
        neg: Negative accumulator (recovery direction).
        history: Full history of observed scores.
    """

    def __init__(self, threshold: float = 5.0, slack: float = 0.5):
        self.threshold = threshold
        self.slack = slack
        self.pos = 0.0
        self.neg = 0.0
        self.history: list = []
        self._events: list = []
        self.current_state = 'STABLE'  # 'STABLE' or 'FATIGUE'

    def update(self, score: float, reference: float) -> str:
        """Process a new score observation.

        Args:
            score: Current fatigue score (0-100 scale).
            reference: Reference level (e.g., running mean or baseline score).

        Returns:
            State string: 'FATIGUE_ONSET', 'RECOVERY', or 'STABLE'.
        """
        deviation = score - reference
        self.pos = max(0.0, self.pos + deviation - self.slack)
        self.neg = max(0.0, self.neg - deviation - self.slack)
        self.history.append(score)

        if self.pos > self.threshold:
            self.pos = 0.0  # Reset accumulator
            if self.current_state == 'STABLE':
                self.current_state = 'FATIGUE'
                self._events.append(('FATIGUE_ONSET', len(self.history) - 1))
                return 'FATIGUE_ONSET'
        elif self.neg > self.threshold:
            self.neg = 0.0  # Reset accumulator
            if self.current_state == 'FATIGUE':
                self.current_state = 'STABLE'
                self._events.append(('RECOVERY', len(self.history) - 1))
                return 'RECOVERY'
        return 'STABLE'

    def reset(self):
        """Reset accumulators (but keep history)."""
        self.pos = 0.0
        self.neg = 0.0
        self.current_state = 'STABLE'

    def full_reset(self):
        """Reset everything including history."""
        self.reset()
        self.history.clear()
        self._events.clear()

    @property
    def events(self):
        """Return list of (event_type, index) tuples."""
        return list(self._events)

    @property
    def state_summary(self) -> dict:
        """Return current internal state for debugging."""
        return {
            'pos_accumulator': self.pos,
            'neg_accumulator': self.neg,
            'n_observations': len(self.history),
            'n_events': len(self._events),
        }

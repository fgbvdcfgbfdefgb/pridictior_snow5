"""
Unit Tests for CPU Market Analyser.
"""

import unittest
import numpy as np
from src.simulator.market_simulator import MarketSimulator
from src.analyser.market_analyser import MarketAnalyser, MarketFeatures


class TestMarketAnalyser(unittest.TestCase):

    def setUp(self):
        self.sim = MarketSimulator(random_day=True, start_year=2023, end_year=2024, seed=42)
        self.analyser = MarketAnalyser()

    def test_feature_extraction(self):
        for _ in range(50):
            tick = self.sim.next_tick()
            features = self.analyser.process_tick(tick)

        self.assertIsInstance(features, MarketFeatures)
        self.assertEqual(len(features.feature_vector), 32)
        self.assertFalse(np.any(np.isnan(features.feature_vector)))
        self.assertFalse(np.any(np.isinf(features.feature_vector)))

        # Verify RSI bounds
        self.assertGreaterEqual(features.rsi_14, 0.0)
        self.assertLessEqual(features.rsi_14, 100.0)

    def test_feature_names_match_vector(self):
        tick = self.sim.next_tick()
        features = self.analyser.process_tick(tick)
        self.assertEqual(len(features.feature_names), len(features.feature_vector))


if __name__ == "__main__":
    unittest.main()

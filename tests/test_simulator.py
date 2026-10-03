"""
Unit Tests for Bitcoin Market Simulator.
"""

import unittest
import numpy as np
import pandas as pd
from src.simulator.market_simulator import MarketSimulator, MarketTick


class TestMarketSimulator(unittest.TestCase):

    def setUp(self):
        # Create minimal test dataset
        self.sim = MarketSimulator(random_day=True, start_year=2021, end_year=2024, seed=42)

    def test_tick_properties(self):
        tick = self.sim.next_tick()
        self.assertIsNotNone(tick)
        self.assertIsInstance(tick, MarketTick)
        self.assertGreater(tick.price, 0)
        self.assertGreaterEqual(tick.high, tick.low)
        self.assertGreater(tick.ask, tick.bid)

    def test_past_window_retrieval(self):
        past_12h = self.sim.get_past_window_prices(window_seconds=43200)
        self.assertEqual(len(past_12h), 43200)
        self.assertTrue(np.all(past_12h > 0))

    def test_future_ground_truth(self):
        gt_25m = self.sim.get_future_ground_truth(horizon_seconds=1500)
        self.assertEqual(len(gt_25m), 1500)
        self.assertTrue(np.all(gt_25m > 0))

    def test_random_day_sampling(self):
        dt = self.sim.seek_to_random_window(start_year=2022, end_year=2023, seed=99)
        self.assertGreaterEqual(dt.year, 2022)
        self.assertLessEqual(dt.year, 2023)


if __name__ == "__main__":
    unittest.main()

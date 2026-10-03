"""
Unit Tests for Price Predictor, Online Reward Trainer, and Stability Filter.
"""

import unittest
import numpy as np
from src.simulator.market_simulator import MarketSimulator
from src.analyser.market_analyser import MarketAnalyser
from src.predictor.model import HybridPredictor, MultiScaleContextEncoder
from src.predictor.online_trainer import OnlineRewardTrainer


class TestPricePredictor(unittest.TestCase):

    def setUp(self):
        self.sim = MarketSimulator(random_day=True, start_year=2024, end_year=2024, seed=10)
        self.analyser = MarketAnalyser()
        self.predictor = HybridPredictor(device="cpu", horizon_mins=25)
        self.trainer = OnlineRewardTrainer(self.predictor)

    def test_multiscale_context_encoding(self):
        raw_12h = np.ones(43200) * 50000.0
        encoded = MultiScaleContextEncoder.encode_12h_history(raw_12h, current_price=50000.0)
        self.assertEqual(len(encoded), 1980)

    def test_25min_trajectory_shape(self):
        tick = self.sim.next_tick()
        features = self.analyser.process_tick(tick)
        past_12h = self.sim.get_past_window_prices(43200)

        trajectory = self.predictor.predict_25min_trajectory(
            current_price=tick.price,
            features=features.feature_vector,
            past_12h_prices=past_12h
        )
        # 25 minutes = 1,500 seconds
        self.assertEqual(len(trajectory), 1500)
        self.assertTrue(np.all(trajectory > 0))

    def test_prediction_stability(self):
        """Tests that consecutive second predictions do not jitter erratically."""
        trajectories = []
        for _ in range(5):
            tick = self.sim.next_tick()
            features = self.analyser.process_tick(tick)
            past_12h = self.sim.get_past_window_prices(43200)

            traj = self.predictor.predict_25min_trajectory(
                current_price=tick.price,
                features=features.feature_vector,
                past_12h_prices=past_12h
            )
            trajectories.append(traj)

        # Difference between step t and step t+1 should be small and continuous
        diff = np.abs(trajectories[1][:-1] - trajectories[0][1:])
        mean_diff = np.mean(diff)
        # Price change per second should be well under 0.1%
        self.assertLess(mean_diff, 50.0)

    def test_online_reward_evaluation(self):
        future_gt = self.sim.get_future_ground_truth(1500)
        tick = self.sim.next_tick()
        features = self.analyser.process_tick(tick)
        past_12h = self.sim.get_past_window_prices(43200)

        traj = self.predictor.predict_25min_trajectory(
            current_price=tick.price,
            features=features.feature_vector,
            past_12h_prices=past_12h
        )

        eval_res = self.trainer.evaluate_step(
            timestamp=tick.timestamp,
            current_price=tick.price,
            predicted_trajectory=traj,
            future_ground_truth=future_gt,
            update_model=True
        )

        self.assertGreaterEqual(eval_res.accuracy_percentage, 0.0)
        self.assertLessEqual(eval_res.accuracy_percentage, 100.0)
        self.assertGreaterEqual(eval_res.stability_score, 0.0)
        self.assertIsInstance(eval_res.directional_accuracy, bool)


if __name__ == "__main__":
    unittest.main()

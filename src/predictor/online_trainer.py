"""
Real-Time Continuous Online Reward Trainer & Evaluation Engine.

Operates second-by-second (per market time, without static batch epochs).
Compares each second's 25-minute trajectory prediction against the stored real market ground truth.

Reward Function:
- Directional Accuracy Score: +1.0 for correct sign of 25-min price movement, -1.0 for false signal.
- Trajectory Accuracy Penalty: -lambda * Normalized RMSE across all 1,500 future seconds.
- Stability / Smoothness Reward: Penalizes erratic tick-to-tick jump discontinuities.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


@dataclass
class PredictionEvaluation:
    """Metrics comparing a 25-minute prediction against true market ground truth."""
    timestamp: int
    current_price: float
    true_final_price: float
    pred_final_price: float
    mae_dollars: float
    rmse_dollars: float
    directional_accuracy: bool    # True if predicted trend direction matched actual
    accuracy_percentage: float    # Combined 0-100% score
    realtime_reward: float        # Online reward value
    stability_score: float        # Smoothness score (0-100%)


class OnlineRewardTrainer:
    """
    Online learning and real-time validation engine.
    """

    def __init__(
        self,
        predictor_model,
        learning_rate: float = 2e-4,
        history_len: int = 1000
    ):
        self.predictor = predictor_model
        self.learning_rate = learning_rate
        self.history_len = history_len

        # Rolling history of evaluations
        self.eval_history: deque = deque(maxlen=history_len)
        self.rolling_rewards: deque = deque(maxlen=100)
        self.rolling_accuracy: deque = deque(maxlen=100)
        self.rolling_mae: deque = deque(maxlen=100)

        # Online running statistics
        self.total_steps = 0
        self.correct_directions = 0

    def evaluate_step(
        self,
        timestamp: int,
        current_price: float,
        predicted_trajectory: np.ndarray,
        future_ground_truth: np.ndarray,
        features: Optional[np.ndarray] = None,
        past_12h_prices: Optional[np.ndarray] = None,
        update_model: bool = True
    ) -> PredictionEvaluation:
        """
        Compares the 25-minute prediction against the actual true market prices.
        """
        horizon_len = min(len(predicted_trajectory), len(future_ground_truth))
        pred_sub = predicted_trajectory[:horizon_len]
        true_sub = future_ground_truth[:horizon_len]

        # 1. Error metrics
        diffs = np.abs(pred_sub - true_sub)
        mae = float(np.mean(diffs))
        rmse = float(np.sqrt(np.mean((pred_sub - true_sub) ** 2)))

        # 2. Directional Accuracy (Sign of (P_25m - P_0))
        true_delta = true_sub[-1] - current_price
        pred_delta = pred_sub[-1] - current_price
        dir_correct = (true_delta * pred_delta >= 0) or (abs(true_delta) < 1e-4 and abs(pred_delta) < 1e-4)

        # 3. Accuracy Percentage (0 - 100%)
        rel_error = mae / max(current_price, 1e-4)
        mape_score = max(0.0, 1.0 - (rel_error * 50.0)) * 50.0
        dir_score = 50.0 if dir_correct else 0.0
        accuracy_pct = mape_score + dir_score

        # 4. Smoothness / Stability penalty
        if len(pred_sub) > 2:
            d2 = np.diff(pred_sub, n=2)
            jitter = float(np.mean(d2 ** 2))
            stability = max(0.0, 100.0 - (jitter * 100.0))
        else:
            jitter = 0.0
            stability = 100.0

        # 5. Realtime Reward Function
        dir_reward = 1.0 if dir_correct else -1.0
        norm_rmse = rmse / max(current_price * 0.01, 1e-4)
        reward = float(dir_reward - 0.5 * norm_rmse - 0.05 * jitter)

        # 6. Online Gradient Step
        if update_model and features is not None and past_12h_prices is not None:
            if hasattr(self.predictor, "train_online_step"):
                self.predictor.train_online_step(current_price, features, past_12h_prices, future_ground_truth)

        eval_res = PredictionEvaluation(
            timestamp=timestamp,
            current_price=float(current_price),
            true_final_price=float(true_sub[-1]),
            pred_final_price=float(pred_sub[-1]),
            mae_dollars=round(mae, 2),
            rmse_dollars=round(rmse, 2),
            directional_accuracy=bool(dir_correct),
            accuracy_percentage=round(accuracy_pct, 2),
            realtime_reward=round(reward, 4),
            stability_score=round(stability, 2),
        )

        self.eval_history.append(eval_res)
        self.rolling_rewards.append(reward)
        self.rolling_accuracy.append(accuracy_pct)
        self.rolling_mae.append(mae)

        self.total_steps += 1
        if dir_correct:
            self.correct_directions += 1

        return eval_res

    def get_summary_stats(self) -> Dict[str, float]:
        """Returns aggregated rolling accuracy and training metrics."""
        if self.total_steps == 0:
            return {
                "total_ticks_evaluated": 0,
                "directional_accuracy_pct": 50.0,
                "avg_mae_dollars": 0.0,
                "rolling_accuracy_pct": 50.0,
                "avg_realtime_reward": 0.0,
            }

        dir_acc_total = (self.correct_directions / self.total_steps) * 100.0
        rolling_acc = float(np.mean(self.rolling_accuracy)) if len(self.rolling_accuracy) > 0 else 50.0
        avg_mae = float(np.mean(self.rolling_mae)) if len(self.rolling_mae) > 0 else 0.0
        avg_reward = float(np.mean(self.rolling_rewards)) if len(self.rolling_rewards) > 0 else 0.0

        return {
            "total_ticks_evaluated": int(self.total_steps),
            "directional_accuracy_pct": round(dir_acc_total, 2),
            "avg_mae_dollars": round(avg_mae, 2),
            "rolling_accuracy_pct": round(rolling_acc, 2),
            "avg_realtime_reward": round(avg_reward, 4),
        }

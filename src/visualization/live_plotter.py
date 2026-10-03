"""
Real-Time 1 FPS Live Animated Plotter for Bitcoin Price Predictions.

Renders:
1. Upper Chart:
   - Solid Line: Historical & Actual Market Price (last 1 hour context).
   - Dotted Line: 25-minute Model Predicted Trajectory (next 1,500s).
   - Ground Truth Marker: True market path stored in dataset.
2. Lower Chart / Bar Panel:
   - Dynamic Accuracy Bar: Directional Accuracy % and Relative Proximity.
   - Live Metric Badges: Real-time MAE ($), RMSE ($), Stability Score (%), and Online Reward.
"""

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Any, Tuple

try:
    from ..simulator.market_simulator import MarketSimulator, MarketTick
    from ..analyser.market_analyser import MarketAnalyser
    from ..predictor.model import HybridPredictor
    from ..predictor.online_trainer import OnlineRewardTrainer
except ImportError:
    from simulator.market_simulator import MarketSimulator, MarketTick
    from analyser.market_analyser import MarketAnalyser
    from predictor.model import HybridPredictor
    from predictor.online_trainer import OnlineRewardTrainer


class LivePredictorPlotter:
    """
    Renders live 1 FPS Matplotlib / Notebook animated graphs.
    """

    def __init__(
        self,
        simulator: MarketSimulator,
        analyser: MarketAnalyser,
        predictor: HybridPredictor,
        trainer: OnlineRewardTrainer,
        history_window_sec: int = 1800,  # 30 mins past history
        horizon_sec: int = 1500,         # 25 mins future horizon
    ):
        self.simulator = simulator
        self.analyser = analyser
        self.predictor = predictor
        self.trainer = trainer

        self.history_window_sec = history_window_sec
        self.horizon_sec = horizon_sec

        # Rolling buffers for plotting
        self.timestamps_hist: List[int] = []
        self.actual_prices_hist: List[float] = []
        self.last_pred_trajectory: Optional[np.ndarray] = None
        self.last_ground_truth: Optional[np.ndarray] = None
        self.last_eval = None

        self.fig = None
        self.ax_main = None
        self.ax_bar = None
        self.line_actual = None
        self.line_pred = None
        self.line_gt = None
        self.bar_acc = None

    def setup_figure(self) -> Tuple[plt.Figure, Any, Any]:
        """Configures the dark-themed financial trading visualizer layout."""
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(14, 8), dpi=100)
        gs = GridSpec(4, 1, figure=fig, hspace=0.35)

        ax_main = fig.add_subplot(gs[0:3, 0])
        ax_bar = fig.add_subplot(gs[3, 0])

        ax_main.set_title("Bitcoin Real-Time 25-Min Price Predictor (1 FPS)", fontsize=14, color="#00E5FF", pad=12, fontweight="bold")
        ax_main.set_ylabel("BTC / USD ($)", fontsize=11, color="#E0E0E0")
        ax_main.grid(True, linestyle="--", alpha=0.3, color="#404040")

        # Main lines
        (self.line_actual,) = ax_main.plot([], [], label="Actual Market Price", color="#00E676", linewidth=2.0)
        (self.line_pred,) = ax_main.plot([], [], label="Model Predicted (Next 25 Min)", color="#FF9100", linestyle=":", linewidth=2.5)
        (self.line_gt,) = ax_main.plot([], [], label="Ground Truth (Stored Data)", color="#2979FF", linestyle="--", linewidth=1.2, alpha=0.6)

        ax_main.legend(loc="upper left", framealpha=0.8, facecolor="#1E1E1E", edgecolor="#333333")

        # Lower Accuracy Bar Panel
        ax_bar.set_title("Real-Time Accuracy & Stability", fontsize=10, color="#B0BEC5", pad=6)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_ylim(-0.5, 1.5)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Accuracy / Stability Score (%)", fontsize=10, color="#E0E0E0")
        ax_bar.grid(True, axis="x", linestyle=":", alpha=0.3)

        self.bar_acc = ax_bar.barh([0.7, 0.0], [50.0, 50.0], height=0.4, color=["#00E5FF", "#76FF03"], alpha=0.85)
        ax_bar.set_yticks([0.7, 0.0])
        ax_bar.set_yticklabels(["Trajectory Match %", "Stability Score %"], fontsize=9, color="#E0E0E0")

        self.text_metrics = ax_main.text(
            0.98, 0.95, "", transform=ax_main.transAxes,
            fontsize=10, color="#FFFFFF", verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#121212", edgecolor="#00E5FF", alpha=0.9)
        )

        self.fig = fig
        self.ax_main = ax_main
        self.ax_bar = ax_bar
        return fig, ax_main, ax_bar

    def step_simulation(self) -> bool:
        """Executes a single 1-second simulation step."""
        if not self.simulator.has_next():
            return False

        future_gt = self.simulator.get_future_ground_truth(horizon_seconds=self.horizon_sec)
        tick = self.simulator.next_tick()
        if tick is None:
            return False

        features = self.analyser.process_tick(tick)
        past_12h = self.simulator.get_past_window_prices(window_seconds=43200)

        # Model Inference
        pred_trajectory = self.predictor.predict_25min_trajectory(
            current_price=tick.price,
            features=features.feature_vector,
            past_12h_prices=past_12h
        )

        # Evaluation & Online Continuous Update
        eval_res = self.trainer.evaluate_step(
            timestamp=tick.timestamp,
            current_price=tick.price,
            predicted_trajectory=pred_trajectory,
            future_ground_truth=future_gt,
            update_model=True
        )

        # Store for rendering
        self.timestamps_hist.append(tick.timestamp)
        self.actual_prices_hist.append(tick.price)

        # Keep rolling window
        if len(self.timestamps_hist) > self.history_window_sec:
            self.timestamps_hist.pop(0)
            self.actual_prices_hist.pop(0)

        self.last_pred_trajectory = pred_trajectory
        self.last_ground_truth = future_gt
        self.last_eval = eval_res
        return True

    def update_frame(self, frame_num: int):
        """Animation update function called at 1 FPS."""
        has_more = self.step_simulation()
        if not has_more or self.last_eval is None:
            return self.line_actual, self.line_pred, self.line_gt

        # X-axes: relative seconds (-history_window to +horizon_sec)
        t_hist_rel = np.arange(-len(self.actual_prices_hist) + 1, 1)
        t_future_rel = np.arange(1, self.horizon_sec + 1)

        # Update Main Graph
        self.line_actual.set_data(t_hist_rel, self.actual_prices_hist)
        self.line_pred.set_data(t_future_rel, self.last_pred_trajectory)
        self.line_gt.set_data(t_future_rel, self.last_ground_truth)

        # Set Plot Limits
        all_y = np.concatenate([self.actual_prices_hist, self.last_pred_trajectory, self.last_ground_truth])
        y_min = float(np.min(all_y)) * 0.998
        y_max = float(np.max(all_y)) * 1.002
        self.ax_main.set_xlim(-self.history_window_sec, self.horizon_sec)
        self.ax_main.set_ylim(y_min, y_max)

        # Update Accuracy Bar
        acc_pct = self.last_eval.accuracy_percentage
        stab_pct = self.last_eval.stability_score
        self.bar_acc[0].set_width(acc_pct)
        self.bar_acc[1].set_width(stab_pct)

        # Color coding
        self.bar_acc[0].set_color("#00E676" if acc_pct >= 60 else ("#FFD600" if acc_pct >= 40 else "#FF5252"))

        # Update Badge Text
        stats = self.trainer.get_summary_stats()
        dt_s = datetime.datetime.fromtimestamp(self.last_eval.timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        info_str = (
            f"Market Time: {dt_s} UTC\n"
            f"Current Price: ${self.last_eval.current_price:,.2f}\n"
            f"25-Min Forecast: ${self.last_eval.pred_final_price:,.2f} "
            f"({'▲' if self.last_eval.pred_final_price >= self.last_eval.current_price else '▼'})\n"
            f"Directional Match: {'✓ CORRECT' if self.last_eval.directional_accuracy else '✗ DIVERGENT'}\n"
            f"MAE: ${self.last_eval.mae_dollars:,.2f} | RMSE: ${self.last_eval.rmse_dollars:,.2f}\n"
            f"Rolling Win Rate: {stats['directional_accuracy_pct']}%\n"
            f"Real-Time Reward: {self.last_eval.realtime_reward:+.3f}"
        )
        self.text_metrics.set_text(info_str)

        return self.line_actual, self.line_pred, self.line_gt

    def run_live_animation(self, total_frames: int = 120, interval_ms: int = 1000):
        """Starts 1 FPS animation (1 frame per 1000ms = 1s per frame)."""
        self.setup_figure()
        anim = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=total_frames,
            interval=interval_ms,
            blit=False,
            repeat=False
        )
        return anim

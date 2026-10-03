"""
Real-Time 1 FPS Live Animated Plotter for Bitcoin Price Predictions.

Renders:
1. Upper Chart (1:1 Uncompressed Second Scale):
   - Solid Line: Historical & Actual Market Price (-25 mins past context = -1,500s).
   - Dotted Line: 25-minute Model Predicted Trajectory (0 to +25 mins = +1,500s).
   - Dashed Line: Ground Truth (0 to +25 mins = +1,500s).
   - Exact identical 1:1 linear time scale where past 25m and future 25m span equal width.
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
    Renders live 1 FPS Matplotlib / Notebook animated graphs with 1:1 uncompressed scale.
    """

    def __init__(
        self,
        simulator: MarketSimulator,
        analyser: MarketAnalyser,
        predictor: HybridPredictor,
        trainer: OnlineRewardTrainer,
        history_window_sec: int = 1500,  # 25 mins past history (1:1 match with 25m future)
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
        fig = plt.figure(figsize=(15, 8.5), dpi=100)
        gs = GridSpec(4, 1, figure=fig, hspace=0.35)

        ax_main = fig.add_subplot(gs[0:3, 0])
        ax_bar = fig.add_subplot(gs[3, 0])

        ax_main.set_title(
            "Bitcoin Real-Time 25-Min Price Predictor (1:1 Second Scale • 1 FPS)",
            fontsize=14, color="#00E5FF", pad=12, fontweight="bold"
        )
        ax_main.set_ylabel("BTC / USD ($)", fontsize=11, color="#E0E0E0")
        ax_main.set_xlabel("Timeline (Seconds from NOW: -1500s to +1500s)", fontsize=10, color="#B0BEC5")
        ax_main.grid(True, linestyle="--", alpha=0.3, color="#404040")

        # Time ticks every 5 minutes (300 seconds)
        time_ticks = [-1500, -1200, -900, -600, -300, 0, 300, 600, 900, 1200, 1500]
        time_labels = ["-25m", "-20m", "-15m", "-10m", "-5m", "NOW (0s)", "+5m", "+10m", "+15m", "+20m", "+25m"]
        ax_main.set_xticks(time_ticks)
        ax_main.set_xticklabels(time_labels, fontsize=9, color="#B0BEC5")

        # Vertical line at NOW (t=0)
        ax_main.axvline(0, color="#00E5FF", linestyle="--", linewidth=1.5, alpha=0.6, label="Current Second (t=0)")

        # Main lines
        (self.line_actual,) = ax_main.plot([], [], label="Actual Market Price (-25m to 0)", color="#00E676", linewidth=2.2)
        (self.line_pred,) = ax_main.plot([], [], label="Model Predicted (0 to +25m)", color="#FF9100", linestyle=":", linewidth=2.8)
        (self.line_gt,) = ax_main.plot([], [], label="Ground Truth (0 to +25m)", color="#2979FF", linestyle="--", linewidth=1.5, alpha=0.7)

        ax_main.legend(loc="upper left", framealpha=0.85, facecolor="#131C2E", edgecolor="#1F2D48")

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

        # Model Inference (uncompressed 1,500 future seconds)
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
            features=features.feature_vector,
            past_12h_prices=past_12h,
            update_model=True
        )

        # Store for rendering
        self.timestamps_hist.append(tick.timestamp)
        self.actual_prices_hist.append(tick.price)

        # Keep rolling window of 1,500 seconds (exact 1:1 match with future 1,500s)
        if len(self.timestamps_hist) > self.history_window_sec:
            self.timestamps_hist.pop(0)
            self.actual_prices_hist.pop(0)

        self.last_pred_trajectory = pred_trajectory
        self.last_ground_truth = future_gt
        self.last_eval = eval_res
        return True

    def update_frame(self, frame_num: int):
        """Animation update function called at 1 FPS with exact 1:1 scale."""
        has_more = self.step_simulation()
        if not has_more or self.last_eval is None:
            return self.line_actual, self.line_pred, self.line_gt

        # X-axes: relative seconds (-1500 to +1500 with exact 1:1 linear scaling)
        past_len = len(self.actual_prices_hist)
        t_hist_rel = np.arange(-past_len + 1, 1)
        t_future_rel = np.arange(1, self.horizon_sec + 1)

        # Connect t=0 boundary smoothly
        p_curr = self.actual_prices_hist[-1]
        t_future_connected = np.concatenate([[0], t_future_rel])
        pred_connected = np.concatenate([[p_curr], self.last_pred_trajectory])
        gt_connected = np.concatenate([[p_curr], self.last_ground_truth])

        # Update Main Graph
        self.line_actual.set_data(t_hist_rel, self.actual_prices_hist)
        self.line_pred.set_data(t_future_connected, pred_connected)
        self.line_gt.set_data(t_future_connected, gt_connected)

        # Set Plot Limits with 1:1 symmetric bounds (-1500s to +1500s)
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

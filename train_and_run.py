"""
End-to-End Training & Live 1 FPS Replay Runner.

CLI Commands:
1. Fast Distributed/GPU Offline Training:
   python train_and_run.py --train --ticks 5000 --gpu
2. Run Simulation on Random Day (2020-2026) at 1 FPS:
   python train_and_run.py --run-random-day --fps 1
3. Run on Specific Historical Date:
   python train_and_run.py --date 2024-03-15 --fps 1
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.simulator.market_simulator import MarketSimulator
from src.analyser.market_analyser import MarketAnalyser
from src.predictor.model import HybridPredictor
from src.predictor.online_trainer import OnlineRewardTrainer
from src.predictor.distributed import setup_gpu_compute, get_device_info


def run_hyperspeed_training(num_ticks: int = 5000, use_gpu: bool = True):
    """
    Executes distributed/single-GPU high-throughput pre-training and online calibration.
    """
    device, dev_info = setup_gpu_compute(prefer_gpu=use_gpu)
    print(f"\n==================================================================")
    print(f"🚀 BITCOIN PRICE PREDICTOR: HIGH-SPEED DISTRIBUTED TRAINING")
    print(f"==================================================================")
    print(f"Compute Device : {device.upper()} ({dev_info.get('device_name')})")
    print(f"Framework      : {dev_info.get('framework')}")
    print(f"Target Ticks   : {num_ticks:,} seconds ({(num_ticks / 3600):.1f} hours of market time)")
    print(f"Prediction Goal: 25-Minute Horizon (1,500s future trajectory)")
    print(f"------------------------------------------------------------------")

    simulator = MarketSimulator(random_day=True, start_year=2020, end_year=2026, seed=42)
    analyser = MarketAnalyser()
    predictor = HybridPredictor(device=device, horizon_mins=25)
    trainer = OnlineRewardTrainer(predictor, learning_rate=2e-4)

    start_wall = time.time()
    step_count = 0

    print(f"Starting online learning loop (sec-by-sec continuous update)...")
    while step_count < num_ticks and simulator.has_next():
        future_gt = simulator.get_future_ground_truth(horizon_seconds=1500)
        tick = simulator.next_tick()
        if tick is None:
            break

        features = analyser.process_tick(tick)
        past_12h = simulator.get_past_window_prices(window_seconds=43200)

        # Predict
        pred_trajectory = predictor.predict_25min_trajectory(
            current_price=tick.price,
            features=features.feature_vector,
            past_12h_prices=past_12h
        )

        # Online training every 5 ticks in batch mode for high throughput
        do_update = (step_count % 5 == 0)
        eval_res = trainer.evaluate_step(
            timestamp=tick.timestamp,
            current_price=tick.price,
            predicted_trajectory=pred_trajectory,
            future_ground_truth=future_gt,
            features=features.feature_vector if do_update else None,
            past_12h_prices=past_12h if do_update else None,
            update_model=do_update
        )

        step_count += 1

        if step_count % 1000 == 0 or step_count == num_ticks:
            stats = trainer.get_summary_stats()
            elapsed = time.time() - start_wall
            speed = step_count / max(0.001, elapsed)
            print(
                f"Tick {step_count:6d}/{num_ticks} | "
                f"Speed: {speed:6.0f} ticks/s | "
                f"Price: ${eval_res.current_price:9.2f} | "
                f"Dir Acc: {stats['directional_accuracy_pct']:5.1f}% | "
                f"MAE: ${eval_res.mae_dollars:6.2f} | "
                f"Reward: {eval_res.realtime_reward:+6.3f}"
            )

    total_time = time.time() - start_wall
    final_stats = trainer.get_summary_stats()
    print(f"------------------------------------------------------------------")
    print(f"✅ Training Complete in {total_time:.2f}s ({step_count/total_time:.0f} ticks/sec)")
    print(f"Final Directional Win Rate : {final_stats['directional_accuracy_pct']}%")
    print(f"Average Price Error (MAE)   : ${final_stats['avg_mae_dollars']:.2f}")
    print(f"Average Real-Time Reward   : {final_stats['avg_realtime_reward']:+.4f}")
    print(f"==================================================================\n")
    return predictor, trainer


def run_live_simulation(fps: float = 1.0, max_seconds: int = 30, random_day: bool = True):
    """
    Runs real-time live market animation replay at 1 FPS.
    """
    print(f"\n==================================================================")
    print(f"📈 BITCOIN PRICE PREDICTOR: LIVE MARKET SIMULATION (1 FPS)")
    print(f"==================================================================")
    
    simulator = MarketSimulator(random_day=random_day, start_year=2020, end_year=2026, seed=123)
    analyser = MarketAnalyser()
    predictor = HybridPredictor(device="cpu", horizon_mins=25)
    trainer = OnlineRewardTrainer(predictor)

    dt_start = datetime.datetime.fromtimestamp(simulator._timestamps[simulator.current_idx], tz=datetime.timezone.utc)
    print(f"Selected Market Day: {dt_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Granularity        : 1 Second / Step (1.0 FPS)")
    print(f"Prediction Window  : 25 Minutes Forward (1,500s)")
    print(f"------------------------------------------------------------------")

    for step in range(max_seconds):
        future_gt = simulator.get_future_ground_truth(horizon_seconds=1500)
        tick = simulator.next_tick()
        if tick is None:
            break

        features = analyser.process_tick(tick)
        past_12h = simulator.get_past_window_prices(window_seconds=43200)

        pred_trajectory = predictor.predict_25min_trajectory(
            current_price=tick.price,
            features=features.feature_vector,
            past_12h_prices=past_12h
        )

        eval_res = trainer.evaluate_step(
            timestamp=tick.timestamp,
            current_price=tick.price,
            predicted_trajectory=pred_trajectory,
            future_ground_truth=future_gt,
            features=features.feature_vector,
            past_12h_prices=past_12h,
            update_model=True
        )

        diff_pct = ((eval_res.pred_final_price - eval_res.current_price) / eval_res.current_price) * 100.0
        dir_icon = "▲" if diff_pct >= 0 else "▼"
        match_str = "✓ MATCH" if eval_res.directional_accuracy else "✗ DIVERGENT"

        # ASCII accuracy bar
        bar_len = 20
        filled = int((eval_res.accuracy_percentage / 100.0) * bar_len)
        acc_bar = "[" + "=" * filled + " " * (bar_len - filled) + "]"

        print(
            f"[{tick.datetime_str}] "
            f"Price: ${tick.price:8.2f} | "
            f"Pred(25m): ${eval_res.pred_final_price:8.2f} ({dir_icon} {diff_pct:+5.2f}%) | "
            f"{match_str:11s} | "
            f"MAE: ${eval_res.mae_dollars:6.2f} | "
            f"Acc: {acc_bar} {eval_res.accuracy_percentage:4.1f}% | "
            f"Reward: {eval_res.realtime_reward:+5.2f}"
        )

        time.sleep(1.0 / fps)

    print(f"------------------------------------------------------------------")
    print(f"Live Simulation Complete. Summary: {trainer.get_summary_stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin Price Predictor Pipeline")
    parser.add_argument("--train", action="store_true", help="Run high-speed offline training")
    parser.add_argument("--run-random-day", action="store_true", help="Run 1 FPS simulation on random day")
    parser.add_argument("--ticks", type=int, default=3000, help="Number of ticks for training")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for live replay")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")

    args = parser.parse_args()

    if args.train:
        run_hyperspeed_training(num_ticks=args.ticks, use_gpu=args.gpu)
    elif args.run_random_day:
        run_live_simulation(fps=args.fps, max_seconds=30, random_day=True)
    else:
        run_hyperspeed_training(num_ticks=2000, use_gpu=False)
        run_live_simulation(fps=10.0, max_seconds=10, random_day=True)

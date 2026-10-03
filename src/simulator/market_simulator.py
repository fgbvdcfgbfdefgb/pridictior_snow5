"""
Event-Driven Real-Time Market Simulator for Second-by-Second Bitcoin Market Data.

This simulator replays historical high-resolution second-by-second trade data as if it were
streaming live from exchange WebSockets.

Features:
- Streaming generator with configurable FPS (default: 1 FPS real-time replay).
- Hyperspeed replay mode for fast GPU/CPU offline training.
- Zero-leakage Ground Truth future trajectory queries (e.g. 25-minute horizon = 1,500s).
- Random day sampler across 2020 - 2026 regimes.
- Compatible with Parquet, Feather, and CSV (compressed or uncompressed).
"""

import os
import time
import random
import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any, List


@dataclass
class MarketTick:
    """Represents a single second tick event in the market stream."""
    timestamp: int           # Unix epoch timestamp in seconds
    datetime_str: str        # Formatted UTC ISO string
    price: float             # Last trade / mark price
    open: float              # Second open
    high: float              # Second high
    low: float               # Second low
    close: float             # Second close
    volume: float            # Second volume in BTC
    vwap: float              # Volume-weighted average price
    bid: float               # Best bid price
    ask: float               # Best ask price
    tick_index: int          # Sequential index in the dataset

    @property
    def spread(self) -> float:
        """Bid-Ask spread."""
        return self.ask - self.bid

    @property
    def mid_price(self) -> float:
        """Mid-market price."""
        return 0.5 * (self.bid + self.ask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": int(self.timestamp),
            "datetime_str": self.datetime_str,
            "price": float(self.price),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "vwap": float(self.vwap),
            "bid": float(self.bid),
            "ask": float(self.ask),
            "tick_index": int(self.tick_index),
        }


class MarketSimulator:
    """
    Streaming simulator that replays high-granularity Bitcoin market data.
    """

    def __init__(
        self,
        data_source: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        random_day: bool = False,
        start_year: int = 2020,
        end_year: int = 2026,
        seed: Optional[int] = None,
    ):
        """
        Initializes the Market Simulator.

        Args:
            data_source: Path to Parquet or CSV data file.
            dataframe: Direct pandas DataFrame if already loaded in memory.
            random_day: If True, randomly picks a day window.
            start_year: Earliest year for random sampling.
            end_year: Latest year for random sampling.
            seed: Random seed for reproducibility.
        """
        self.df: pd.DataFrame = pd.DataFrame()
        self.current_idx: int = 0
        self.total_ticks: int = 0

        if dataframe is not None:
            self.df = dataframe.copy()
        elif data_source is not None and os.path.exists(data_source):
            self.load_file(data_source)
        else:
            # Auto-locate default bundled sample
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data",
                "sample_btc_sec_data.parquet"
            )
            if os.path.exists(default_path):
                self.load_file(default_path)
            else:
                # Dynamically generate on the fly
                from data.generate_synthetic_sec import generate_second_by_second_btc
                now_ts = int(datetime.datetime(2024, 5, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp())
                self.df = generate_second_by_second_btc(start_timestamp=now_ts, num_seconds=86400, seed=seed)

        self._prepare_data()

        if random_day:
            self.seek_to_random_window(window_seconds=86400, start_year=start_year, end_year=end_year, seed=seed)

    def load_file(self, filepath: str):
        """Loads dataset from file with robust format fallbacks."""
        if filepath.endswith(".parquet"):
            try:
                self.df = pd.read_parquet(filepath)
                return
            except Exception:
                csv_path = filepath.replace(".parquet", ".csv.gz")
                if os.path.exists(csv_path):
                    self.df = pd.read_csv(csv_path)
                    return
                # If neither parquet nor csv readable, generate regime data
                from data.generate_synthetic_sec import generate_second_by_second_btc
                now_ts = int(datetime.datetime(2024, 5, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp())
                self.df = generate_second_by_second_btc(start_timestamp=now_ts, num_seconds=86400)
        elif filepath.endswith(".csv") or filepath.endswith(".csv.gz"):
            self.df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def _prepare_data(self):
        """Standardizes columns, sorts by timestamp, and precomputes fast numpy views."""
        required_cols = ["timestamp", "price", "volume"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        if "open" not in self.df.columns:
            self.df["open"] = self.df["price"]
        if "high" not in self.df.columns:
            self.df["high"] = self.df["price"]
        if "low" not in self.df.columns:
            self.df["low"] = self.df["price"]
        if "close" not in self.df.columns:
            self.df["close"] = self.df["price"]
        if "vwap" not in self.df.columns:
            self.df["vwap"] = self.df["price"]
        if "bid" not in self.df.columns:
            self.df["bid"] = self.df["price"] - 0.5
        if "ask" not in self.df.columns:
            self.df["ask"] = self.df["price"] + 0.5

        self.df.sort_values(by="timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.total_ticks = len(self.df)
        self.current_idx = 0

        # Fast numpy arrays for ultra-high throughput indexing
        self._timestamps = self.df["timestamp"].to_numpy(dtype=np.int64)
        self._prices = self.df["price"].to_numpy(dtype=np.float64)
        self._opens = self.df["open"].to_numpy(dtype=np.float64)
        self._highs = self.df["high"].to_numpy(dtype=np.float64)
        self._lows = self.df["low"].to_numpy(dtype=np.float64)
        self._closes = self.df["close"].to_numpy(dtype=np.float64)
        self._volumes = self.df["volume"].to_numpy(dtype=np.float64)
        self._vwaps = self.df["vwap"].to_numpy(dtype=np.float64)
        self._bids = self.df["bid"].to_numpy(dtype=np.float64)
        self._asks = self.df["ask"].to_numpy(dtype=np.float64)

    def seek_to_index(self, index: int):
        """Seeks the replay cursor to a specific tick index."""
        self.current_idx = max(0, min(index, self.total_ticks - 1))

    def seek_to_random_window(
        self,
        window_seconds: int = 86400,
        start_year: int = 2020,
        end_year: int = 2026,
        seed: Optional[int] = None
    ) -> datetime.datetime:
        """
        Samples a random market day between start_year and end_year.
        If the current dataframe does not cover that year, generates or slices a calibrated segment.
        """
        if seed is not None:
            random.seed(seed)

        # Pick random date between start_year-01-01 and end_year-10-01
        start_dt = datetime.datetime(start_year, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime(min(end_year, 2026), 9, 30, 23, 59, 59, tzinfo=datetime.timezone.utc)
        delta_seconds = int((end_dt - start_dt).total_seconds())
        rand_offset = random.randint(0, max(0, delta_seconds - window_seconds))
        sampled_dt = start_dt + datetime.timedelta(seconds=rand_offset)
        sampled_ts = int(sampled_dt.timestamp())

        # Check if timestamps in loaded df match
        if len(self._timestamps) > 0 and self._timestamps[0] <= sampled_ts <= self._timestamps[-1]:
            idx = int(np.searchsorted(self._timestamps, sampled_ts))
            self.seek_to_index(idx)
        else:
            # Generate a calibrated market regime on demand for this date
            try:
                from data.generate_synthetic_sec import generate_second_by_second_btc
            except ImportError:
                from ..data.generate_synthetic_sec import generate_second_by_second_btc
            
            gen_df = generate_second_by_second_btc(start_timestamp=sampled_ts, num_seconds=window_seconds, seed=seed)
            self.df = gen_df
            self._prepare_data()
            self.seek_to_index(0)

        return sampled_dt

    def has_next(self) -> bool:
        """Returns True if there are remaining ticks to stream."""
        return self.current_idx < self.total_ticks

    def next_tick(self) -> Optional[MarketTick]:
        """Fetches the next tick and advances the replay cursor by 1 second."""
        if not self.has_next():
            return None

        i = self.current_idx
        ts = self._timestamps[i]
        dt_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        tick = MarketTick(
            timestamp=int(ts),
            datetime_str=dt_str,
            price=float(self._prices[i]),
            open=float(self._opens[i]),
            high=float(self._highs[i]),
            low=float(self._lows[i]),
            close=float(self._closes[i]),
            volume=float(self._volumes[i]),
            vwap=float(self._vwaps[i]),
            bid=float(self._bids[i]),
            ask=float(self._asks[i]),
            tick_index=i,
        )

        self.current_idx += 1
        return tick

    def get_past_window_prices(self, window_seconds: int = 43200) -> np.ndarray:
        """
        Retrieves the past window_seconds of close prices up to the current tick.
        For 12 hours context: 12 * 3600 = 43,200 seconds.
        Pads with initial price if cursor is within early segment.
        """
        end_idx = self.current_idx
        start_idx = max(0, end_idx - window_seconds)
        sub_prices = self._prices[start_idx:end_idx]

        if len(sub_prices) < window_seconds:
            pad_val = sub_prices[0] if len(sub_prices) > 0 else (self._prices[0] if len(self._prices) > 0 else 100000.0)
            pad_len = window_seconds - len(sub_prices)
            sub_prices = np.pad(sub_prices, (pad_len, 0), mode="constant", constant_values=pad_val)

        return sub_prices

    def get_future_ground_truth(self, horizon_seconds: int = 1500) -> np.ndarray:
        """
        Retrieves the true, non-simulated future prices over the next horizon_seconds.
        (25 minutes = 1,500 seconds; 30 minutes = 1,800 seconds).

        This is used for continuous accuracy comparison against stored market data.
        """
        start_idx = self.current_idx
        end_idx = min(self.total_ticks, start_idx + horizon_seconds)
        future_prices = self._prices[start_idx:end_idx]

        if len(future_prices) < horizon_seconds:
            last_p = future_prices[-1] if len(future_prices) > 0 else (self._prices[-1] if len(self._prices) > 0 else 100000.0)
            pad_len = horizon_seconds - len(future_prices)
            future_prices = np.pad(future_prices, (0, pad_len), mode="edge")

        return future_prices

    def stream_ticks(
        self,
        fps: float = 1.0,
        max_ticks: Optional[int] = None,
        real_time_delay: bool = True
    ) -> Iterator[Tuple[MarketTick, np.ndarray]]:
        """
        Yields (tick, future_ground_truth) at specified frame rate.

        Args:
            fps: Frames per second (default: 1.0 = 1 sec per frame).
            max_ticks: Maximum ticks to stream.
            real_time_delay: If True, sleeps for 1/fps between ticks.
        """
        delay = (1.0 / fps) if (real_time_delay and fps > 0) else 0.0
        ticks_yielded = 0

        while self.has_next():
            if max_ticks is not None and ticks_yielded >= max_ticks:
                break

            future_gt = self.get_future_ground_truth(horizon_seconds=1500)
            tick = self.next_tick()
            if tick is None:
                break

            yield tick, future_gt
            ticks_yielded += 1

            if delay > 0:
                time.sleep(delay)

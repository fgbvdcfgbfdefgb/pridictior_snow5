"""
High-Speed Real-Time CPU Market Analyser.

Processes the streaming live market feed tick-by-tick on the CPU, maintaining rolling
ring-buffer states and computing multi-scale alpha signals and microstructure indicators.

Extracts:
1. Microstructure & Order Flow:
   - Rolling multi-horizon log returns (1s, 5s, 15s, 60s, 300s, 900s)
   - Realized Volatility & Parkinson Volatility
   - Order Flow Imbalance (OFI proxy) & Volume Delta
   - Bid-Ask Spread and Micro-Liquidity metrics
2. Meso-Scale Technical Momentum & Mean-Reversion:
   - Exponential Moving Averages (EMA 9, 21, 50, 200) & Convergence
   - Relative Strength Index (RSI 14)
   - Moving Average Convergence Divergence (MACD, Signal, Histogram)
   - Bollinger Bands (%B, Bandwidth)
   - Volume Weighted Average Price (VWAP) Deviation
3. Macro/Intraday Regime & Seasonality:
   - 12-Hour High/Low Range Position
   - Cyclical Time of Day Sine/Cosine Features
   - Trend Acceleration & Kurtosis

Features are normalized, clipped, and formatted into fixed-length vectors suitable for GPU inference.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
try:
    from ..simulator.market_simulator import MarketTick
except ImportError:
    from simulator.market_simulator import MarketTick


@dataclass
class MarketFeatures:
    """Dataclass holding extracted real-time signals."""
    feature_vector: np.ndarray      # 1D float32 numpy array of shape (32,)
    feature_names: List[str]
    current_price: float
    vwap: float
    rsi_14: float
    macd: float
    bb_percent_b: float
    realized_vol_60s: float
    spread_bps: float
    trend_signal: float

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.feature_names, self.feature_vector)}


class MarketAnalyser:
    """
    CPU-optimized streaming market signal extractor.
    """

    FEATURE_NAMES = [
        "log_ret_1s", "log_ret_5s", "log_ret_15s", "log_ret_30s", "log_ret_60s", "log_ret_300s", "log_ret_900s",
        "realized_vol_60s", "realized_vol_300s", "realized_vol_900s",
        "ema_9_diff_pct", "ema_21_diff_pct", "ema_50_diff_pct", "ema_200_diff_pct",
        "ema_cross_9_21", "ema_cross_21_50",
        "rsi_14_norm",
        "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "bb_percent_b", "bb_bandwidth",
        "vwap_diff_pct",
        "volume_intensity_60s", "volume_delta_proxy",
        "spread_bps",
        "micro_acceleration",
        "range_12h_pos",
        "sin_time_of_day", "cos_time_of_day",
        "skewness_proxy", "volatility_trend"
    ]

    def __init__(self, buffer_size: int = 43200):
        """
        Args:
            buffer_size: Maximum rolling history size in seconds (43,200s = 12 hours).
        """
        self.buffer_size = buffer_size
        self.count = 0

        # Ring buffers for fast sliding window updates
        self.prices = np.zeros(buffer_size, dtype=np.float64)
        self.volumes = np.zeros(buffer_size, dtype=np.float64)
        self.highs = np.zeros(buffer_size, dtype=np.float64)
        self.lows = np.zeros(buffer_size, dtype=np.float64)
        self.vwaps = np.zeros(buffer_size, dtype=np.float64)
        self.timestamps = np.zeros(buffer_size, dtype=np.int64)

        # Online stateful indicators
        self.ema_9 = None
        self.ema_21 = None
        self.ema_50 = None
        self.ema_200 = None

        self.ema_12_macd = None
        self.ema_26_macd = None
        self.macd_signal = None

        # RSI online state
        self.rsi_avg_gain = 0.0
        self.rsi_avg_loss = 0.0
        self.rsi_period = 14

        # Smoothing weights
        self.alpha_9 = 2.0 / (9.0 + 1.0)
        self.alpha_21 = 2.0 / (21.0 + 1.0)
        self.alpha_50 = 2.0 / (50.0 + 1.0)
        self.alpha_200 = 2.0 / (200.0 + 1.0)
        self.alpha_12 = 2.0 / (12.0 + 1.0)
        self.alpha_26 = 2.0 / (26.0 + 1.0)
        self.alpha_9_sig = 2.0 / (9.0 + 1.0)

    def reset(self):
        """Resets the analyser state."""
        self.count = 0
        self.prices.fill(0.0)
        self.volumes.fill(0.0)
        self.highs.fill(0.0)
        self.lows.fill(0.0)
        self.vwaps.fill(0.0)
        self.timestamps.fill(0)
        self.ema_9 = None
        self.ema_21 = None
        self.ema_50 = None
        self.ema_200 = None
        self.ema_12_macd = None
        self.ema_26_macd = None
        self.macd_signal = None
        self.rsi_avg_gain = 0.0
        self.rsi_avg_loss = 0.0

    def process_tick(self, tick: MarketTick) -> MarketFeatures:
        """
        Ingests a new tick and extracts all 32 alpha signals in sub-millisecond CPU time.
        """
        p = float(tick.price)
        v = float(tick.volume)
        h = float(tick.high)
        l = float(tick.low)
        vwap = float(tick.vwap)
        ts = int(tick.timestamp)

        # Store in rolling circular buffers
        idx = self.count % self.buffer_size
        self.prices[idx] = p
        self.volumes[idx] = v
        self.highs[idx] = h
        self.lows[idx] = l
        self.vwaps[idx] = vwap
        self.timestamps[idx] = ts
        self.count += 1

        n = min(self.count, self.buffer_size)

        # 1. Update EMAs
        if self.ema_9 is None:
            self.ema_9 = p
            self.ema_21 = p
            self.ema_50 = p
            self.ema_200 = p
            self.ema_12_macd = p
            self.ema_26_macd = p
            self.macd_signal = 0.0
        else:
            self.ema_9 = self.alpha_9 * p + (1.0 - self.alpha_9) * self.ema_9
            self.ema_21 = self.alpha_21 * p + (1.0 - self.alpha_21) * self.ema_21
            self.ema_50 = self.alpha_50 * p + (1.0 - self.alpha_50) * self.ema_50
            self.ema_200 = self.alpha_200 * p + (1.0 - self.alpha_200) * self.ema_200

            self.ema_12_macd = self.alpha_12 * p + (1.0 - self.alpha_12) * self.ema_12_macd
            self.ema_26_macd = self.alpha_26 * p + (1.0 - self.alpha_26) * self.ema_26_macd

        macd_raw = self.ema_12_macd - self.ema_26_macd
        self.macd_signal = self.alpha_9_sig * macd_raw + (1.0 - self.alpha_9_sig) * self.macd_signal
        macd_hist = macd_raw - self.macd_signal

        # 2. Update RSI
        if self.count > 1:
            prev_idx = (self.count - 2) % self.buffer_size
            p_prev = self.prices[prev_idx]
            delta = p - p_prev
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)

            alpha_rsi = 1.0 / self.rsi_period
            self.rsi_avg_gain = alpha_rsi * gain + (1.0 - alpha_rsi) * self.rsi_avg_gain
            self.rsi_avg_loss = alpha_rsi * loss + (1.0 - alpha_rsi) * self.rsi_avg_loss

            if self.rsi_avg_loss == 0.0:
                rsi_val = 100.0
            else:
                rs = self.rsi_avg_gain / self.rsi_avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_val = 50.0

        # 3. Multi-horizon log returns
        def get_lag_price(lag_sec: int) -> float:
            if n <= lag_sec:
                first_idx = 0 if self.count <= self.buffer_size else (self.count % self.buffer_size)
                return self.prices[first_idx]
            target_idx = (self.count - 1 - lag_sec) % self.buffer_size
            return self.prices[target_idx]

        p_1s = get_lag_price(1)
        p_5s = get_lag_price(5)
        p_15s = get_lag_price(15)
        p_30s = get_lag_price(30)
        p_60s = get_lag_price(60)
        p_300s = get_lag_price(300)
        p_900s = get_lag_price(900)

        ret_1s = np.log(max(p, 1e-4) / max(p_1s, 1e-4))
        ret_5s = np.log(max(p, 1e-4) / max(p_5s, 1e-4))
        ret_15s = np.log(max(p, 1e-4) / max(p_15s, 1e-4))
        ret_30s = np.log(max(p, 1e-4) / max(p_30s, 1e-4))
        ret_60s = np.log(max(p, 1e-4) / max(p_60s, 1e-4))
        ret_300s = np.log(max(p, 1e-4) / max(p_300s, 1e-4))
        ret_900s = np.log(max(p, 1e-4) / max(p_900s, 1e-4))

        # 4. Realized Volatility
        def calc_vol(window_sec: int) -> float:
            w = min(n, window_sec)
            if w < 2:
                return 0.001
            indices = [(self.count - 1 - k) % self.buffer_size for k in range(w)]
            pts = self.prices[indices]
            rets = np.diff(np.log(np.maximum(pts, 1e-4)))
            return float(np.std(rets) * np.sqrt(86400))  # annualized-like daily scale

        rvol_60 = calc_vol(60)
        rvol_300 = calc_vol(300)
        rvol_900 = calc_vol(900)

        # 5. Bollinger Bands (20 periods of 60s or last 60s)
        w_bb = min(n, 60)
        bb_indices = [(self.count - 1 - k) % self.buffer_size for k in range(w_bb)]
        bb_prices = self.prices[bb_indices]
        bb_mean = float(np.mean(bb_prices))
        bb_std = float(np.std(bb_prices)) + 1e-6
        bb_upper = bb_mean + 2.0 * bb_std
        bb_lower = bb_mean - 2.0 * bb_std
        bb_percent_b = (p - bb_lower) / (bb_upper - bb_lower) if (bb_upper > bb_lower) else 0.5
        bb_bandwidth = (bb_upper - bb_lower) / max(bb_mean, 1e-4)

        # 6. VWAP deviation
        vwap_diff_pct = (p - vwap) / max(vwap, 1e-4)

        # 7. Volume Intensity & Delta Proxy
        w_vol = min(n, 60)
        vol_indices = [(self.count - 1 - k) % self.buffer_size for k in range(w_vol)]
        avg_vol = float(np.mean(self.volumes[vol_indices])) + 1e-6
        volume_intensity = min(5.0, v / avg_vol)
        
        # Order flow buying/selling pressure proxy: position within bar * volume
        bar_range = max(h - l, 1e-4)
        flow_imbalance = ((p - l) / bar_range - 0.5) * 2.0  # [-1.0, 1.0]
        volume_delta = flow_imbalance * volume_intensity

        # 8. Spread
        spread_bps = (tick.spread / max(p, 1e-4)) * 10000.0

        # 9. Micro-acceleration
        micro_acc = (ret_1s - (np.log(max(p_1s, 1e-4) / max(get_lag_price(2), 1e-4))))

        # 10. 12-hour high/low range
        w_12h = min(n, 43200)
        range_indices = [(self.count - 1 - k) % self.buffer_size for k in range(w_12h)]
        h_12h = float(np.max(self.highs[range_indices]))
        l_12h = float(np.min(self.lows[range_indices]))
        range_12h_pos = (p - l_12h) / max(h_12h - l_12h, 1e-4) if h_12h > l_12h else 0.5

        # 11. Cyclical time
        sec_in_day = ts % 86400
        sin_tod = np.sin(sec_in_day * 2.0 * np.pi / 86400.0)
        cos_tod = np.cos(sec_in_day * 2.0 * np.pi / 86400.0)

        # 12. Volatility trend & Skewness
        vol_trend = (rvol_60 - rvol_900) / max(rvol_900, 1e-4)
        skew_proxy = float(np.mean(((bb_prices - bb_mean) / bb_std)**3)) if len(bb_prices) > 3 else 0.0

        # Assemble normalized feature vector
        vec = np.array([
            np.clip(ret_1s * 500.0, -5.0, 5.0),
            np.clip(ret_5s * 250.0, -5.0, 5.0),
            np.clip(ret_15s * 150.0, -5.0, 5.0),
            np.clip(ret_30s * 100.0, -5.0, 5.0),
            np.clip(ret_60s * 70.0, -5.0, 5.0),
            np.clip(ret_300s * 30.0, -5.0, 5.0),
            np.clip(ret_900s * 15.0, -5.0, 5.0),
            np.clip(rvol_60 * 100.0, 0.0, 10.0),
            np.clip(rvol_300 * 100.0, 0.0, 10.0),
            np.clip(rvol_900 * 100.0, 0.0, 10.0),
            np.clip((p - self.ema_9) / max(p, 1e-4) * 500.0, -5.0, 5.0),
            np.clip((p - self.ema_21) / max(p, 1e-4) * 350.0, -5.0, 5.0),
            np.clip((p - self.ema_50) / max(p, 1e-4) * 200.0, -5.0, 5.0),
            np.clip((p - self.ema_200) / max(p, 1e-4) * 100.0, -5.0, 5.0),
            np.clip((self.ema_9 - self.ema_21) / max(p, 1e-4) * 500.0, -5.0, 5.0),
            np.clip((self.ema_21 - self.ema_50) / max(p, 1e-4) * 350.0, -5.0, 5.0),
            (rsi_val - 50.0) / 50.0,  # [-1, 1]
            np.clip(macd_raw / max(p, 1e-4) * 1000.0, -5.0, 5.0),
            np.clip(self.macd_signal / max(p, 1e-4) * 1000.0, -5.0, 5.0),
            np.clip(macd_hist / max(p, 1e-4) * 2000.0, -5.0, 5.0),
            np.clip(bb_percent_b * 2.0 - 1.0, -3.0, 3.0),
            np.clip(bb_bandwidth * 500.0, 0.0, 10.0),
            np.clip(vwap_diff_pct * 300.0, -5.0, 5.0),
            np.clip(volume_intensity - 1.0, -1.0, 5.0),
            np.clip(volume_delta, -3.0, 3.0),
            np.clip(spread_bps / 5.0, 0.0, 5.0),
            np.clip(micro_acc * 1000.0, -5.0, 5.0),
            (range_12h_pos - 0.5) * 2.0,  # [-1, 1]
            sin_tod,
            cos_tod,
            np.clip(skew_proxy, -3.0, 3.0),
            np.clip(vol_trend, -3.0, 3.0),
        ], dtype=np.float32)

        trend_score = float(0.4 * vec[10] + 0.3 * vec[16] + 0.3 * vec[19])

        return MarketFeatures(
            feature_vector=vec,
            feature_names=self.FEATURE_NAMES,
            current_price=p,
            vwap=vwap,
            rsi_14=rsi_val,
            macd=macd_raw,
            bb_percent_b=bb_percent_b,
            realized_vol_60s=rvol_60,
            spread_bps=spread_bps,
            trend_signal=trend_score,
        )

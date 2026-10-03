"""
High-Fidelity Multi-Regime Second-by-Second Bitcoin Market Data Generator (2020 - 2026).

This module simulates realistic second-by-second BTC/USDT market data calibrated across
the distinct historical market regimes between 2020 and 2026:
- 2020: Halving recovery & breakout ($7,000 -> $29,000)
- 2021: Bull run, double peak, high volatility ($29,000 -> $64,000 -> $29,000 -> $69,000)
- 2022: Crypto winter & deleveraging ($48,000 -> $16,000)
- 2023: Post-bear consolidation & ETF narrative build ($16,000 -> $42,000)
- 2024: Spot ETF inflows & All-Time High ($42,000 -> $99,000)
- 2025-2026: Institutional cycle & macro expansion ($90,000 -> $145,000+)

Mathematical Model:
- Jump-Diffusion Geometric Brownian Motion with Heston Stochastic Volatility
- Intraday U-shaped volume & volatility seasonality (Asian, European, US market opens)
- Microstructure noise and bid-ask bounce
"""

import os
import datetime
import numpy as np
import pandas as pd
from typing import Optional, Tuple


# Regime parameters: (start_date, end_date, start_price, end_price, base_annual_vol, jump_intensity)
REGIMES = [
    ("2020-01-01", "2020-12-31", 7200.0, 29000.0, 0.70, 0.05),
    ("2021-01-01", "2021-12-31", 29000.0, 47000.0, 0.85, 0.08),
    ("2022-01-01", "2022-12-31", 47000.0, 16500.0, 0.65, 0.06),
    ("2023-01-01", "2023-12-31", 16500.0, 42500.0, 0.45, 0.03),
    ("2024-01-01", "2024-12-31", 42500.0, 96000.0, 0.55, 0.04),
    ("2025-01-01", "2026-10-04", 96000.0, 138000.0, 0.50, 0.04),
]


def get_regime_for_date(target_date: datetime.date) -> Tuple[float, float, float, float]:
    """Finds starting price, drift, base volatility, and jump parameters for a target date."""
    target_dt = pd.to_datetime(target_date)
    for start_s, end_s, p_start, p_end, vol, jump in REGIMES:
        s_dt = pd.to_datetime(start_s)
        e_dt = pd.to_datetime(end_s)
        if s_dt <= target_dt <= e_dt:
            total_days = max(1, (e_dt - s_dt).days)
            elapsed_days = (target_dt - s_dt).days
            progress = elapsed_days / total_days
            # Log-linear price interpolation with cyclical variation
            base_price = np.exp(np.log(p_start) + progress * (np.log(p_end) - np.log(p_start)))
            annual_drift = (np.log(p_end) - np.log(p_start)) / (total_days / 365.25)
            return float(base_price), float(annual_drift), float(vol), float(jump)
    
    # Fallback default (2026 pricing)
    return 115000.0, 0.25, 0.50, 0.04


def generate_second_by_second_btc(
    start_timestamp: int,
    num_seconds: int = 86400,
    initial_price: Optional[float] = None,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generates high-precision second-by-second BTC/USDT market data.

    Args:
        start_timestamp: Unix epoch timestamp in seconds.
        num_seconds: Number of seconds to generate (e.g. 86400 = 24 hours, 43200 = 12 hours).
        initial_price: Optional starting price. If None, derived from date.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns:
        [timestamp, datetime, price, open, high, low, close, volume, vwap, bid, ask]
    """
    if seed is not None:
        np.random.seed(seed)

    start_dt = datetime.datetime.fromtimestamp(start_timestamp, tz=datetime.timezone.utc)
    base_price, annual_drift, base_vol, jump_intensity = get_regime_for_date(start_dt.date())

    if initial_price is not None:
        price_0 = initial_price
    else:
        price_0 = base_price

    dt_sec = 1.0 / (365.25 * 86400.0)  # Time step in years
    drift_sec = (annual_drift - 0.5 * base_vol**2) * dt_sec
    vol_sec = base_vol * np.sqrt(dt_sec)

    # 1. Heston Stochastic Volatility Process
    # dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
    kappa = 2.0  # mean-reversion speed
    theta = base_vol**2
    xi = 0.3  # vol of vol
    
    vol_path = np.zeros(num_seconds)
    vol_path[0] = theta

    # Correlated Brownian motions (leverage effect: rho < 0 for equities/crypto pullbacks)
    rho = -0.15
    z1 = np.random.normal(0, 1, num_seconds)
    z2_independent = np.random.normal(0, 1, num_seconds)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * z2_independent

    for t in range(1, num_seconds):
        v_prev = max(vol_path[t-1], 1e-6)
        dv = kappa * (theta - v_prev) * (1.0 / 86400.0) + xi * np.sqrt(v_prev) * np.sqrt(1.0 / 86400.0) * z2[t]
        vol_path[t] = max(v_prev + dv, 1e-4)

    instant_vol_sec = np.sqrt(vol_path) * np.sqrt(dt_sec)

    # 2. Intraday seasonality factor (U-shaped across 24h)
    seconds_in_day = (np.arange(num_seconds) + (start_timestamp % 86400)) % 86400
    hour_of_day = seconds_in_day / 3600.0
    seasonality = 1.0 + 0.35 * np.cos((hour_of_day - 14.5) * 2 * np.pi / 24)  # Peak at US/EU market overlap

    # 3. Jump-Diffusion (Poisson jumps)
    jump_prob_per_sec = jump_intensity / 86400.0
    jump_occurred = np.random.random(num_seconds) < jump_prob_per_sec
    jump_sizes = np.random.laplace(0, 0.003, num_seconds) * jump_occurred

    # 4. Price Path Generation
    log_returns = drift_sec + instant_vol_sec * seasonality * z1 + jump_sizes
    log_price_path = np.log(price_0) + np.cumsum(log_returns)
    mid_prices = np.exp(log_price_path)

    # 5. Second-level Microstructure (OHLCV, Bid/Ask, Spread, Volume)
    # Micro spread: roughly 0.5 to 2 bps ($5-$25 at $100k)
    spread_bps = np.random.uniform(0.00005, 0.00025, num_seconds)
    half_spread = mid_prices * spread_bps * 0.5

    bids = mid_prices - half_spread
    asks = mid_prices + half_spread

    # Intrasecond high/low jitter
    micro_jitter = np.abs(np.random.normal(0, instant_vol_sec * seasonality * 0.8, num_seconds)) * mid_prices
    opens = np.roll(mid_prices, 1)
    opens[0] = price_0
    closes = mid_prices
    highs = np.maximum(opens, closes) + micro_jitter
    lows = np.minimum(opens, closes) - micro_jitter

    # Volume: lognormal distributed with volume clustering (GARCH-like)
    base_volume = np.random.lognormal(mean=0.2, sigma=0.8, size=num_seconds) * (mid_prices / 50000.0) * seasonality
    # Correlate volume with volatility spikes
    volumes = np.round(base_volume * (1.0 + 50.0 * np.abs(log_returns)), 4)
    volumes = np.maximum(volumes, 0.0001)

    # VWAP (Volume Weighted Average Price)
    typical_prices = (highs + lows + closes) / 3.0
    cum_pv = np.cumsum(typical_prices * volumes)
    cum_v = np.cumsum(volumes)
    vwap = cum_pv / np.maximum(cum_v, 1e-8)

    timestamps = np.arange(start_timestamp, start_timestamp + num_seconds, dtype=np.int64)
    datetimes = pd.to_datetime(timestamps, unit="s", utc=True)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "datetime": datetimes,
        "price": np.round(closes, 2),
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes,
        "vwap": np.round(vwap, 2),
        "bid": np.round(bids, 2),
        "ask": np.round(asks, 2),
    })

    return df


def generate_benchmark_multi_day_dataset(
    output_parquet_path: str,
    days: int = 3,
    start_date_str: str = "2024-03-15",
    seed: int = 100
) -> str:
    """
    Generates a continuous multi-day second-by-second dataset and saves to Parquet.
    Over 259,200 seconds of continuous data (3 full days) for robust offline training and validation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    start_dt = pd.to_datetime(start_date_str, utc=True)
    start_ts = int(start_dt.timestamp())
    num_seconds = days * 86400

    print(f"Generating {num_seconds:,} seconds ({days} days) of high-resolution BTC market data starting {start_date_str}...")
    df = generate_second_by_second_btc(start_timestamp=start_ts, num_seconds=num_seconds, seed=seed)
    df.to_parquet(output_parquet_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Successfully generated and saved {len(df):,} rows to {output_parquet_path} (Size: {os.path.getsize(output_parquet_path)/1024/1024:.2f} MB)")
    return output_parquet_path


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "sample_btc_sec_data.parquet")
    generate_benchmark_multi_day_dataset(out_path, days=2, start_date_str="2024-03-15", seed=42)

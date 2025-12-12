# indicators.py

import numpy as np
import pandas as pd


def rsi(series, window=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic_kd(df, k_window=14, d_window=3):
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_window).mean()
    return k, d


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def williams_r(df, window=14):
    high_max = df["High"].rolling(window).max()
    low_min = df["Low"].rolling(window).min()
    wr = -100 * (high_max - df["Close"]) / (high_max - low_min)
    return wr


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add oscillators, MAs, time features. Drops initial NaNs from rolling."""
    df = df.copy()
    df = df.sort_values("datetime")

    # Prices / returns
    df["log_close"] = np.log(df["Close"])
    df["ret_1"] = df["Close"].pct_change()
    df["ret_15m"] = df["log_close"].diff()

    # Indicators
    df["rsi14"] = rsi(df["Close"], 14)
    df["stoch_k"], df["stoch_d"] = stochastic_kd(df, 14, 3)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["Close"], 12, 26, 9)
    df["williams_r"] = williams_r(df, 14)

    # SMAs / EMAs
    for w in [20, 50, 100]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
    for w in [9, 21, 50]:
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # Volume feature
    vol_log = np.log1p(df["Volume"])
    df["vol_z"] = (vol_log - vol_log.rolling(100).mean()) / \
                  (vol_log.rolling(100).std() + 1e-6)

    # Time features
    dt = df["datetime"]
    df["minute"] = dt.dt.hour * 60 + dt.dt.minute
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / (24 * 60))
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / (24 * 60))
    df["dow"] = dt.dt.weekday
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 5)

    # Drop rows with NaNs from rolling indicators
    df = df.dropna().reset_index(drop=True)
    return df

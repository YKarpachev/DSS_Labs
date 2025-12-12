# data.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_PATH, SEQ_LEN, HORIZON, BATCH_SIZE,
    BEAR_THRESHOLD, BULL_THRESHOLD
)
from indicators import add_features


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y_reg, y_cls):
        self.X = X
        self.y_reg = y_reg
        self.y_cls = y_cls

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


def make_splits(df: pd.DataFrame):
    """70/15/15 split by chronological order."""
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)
    return df_train, df_val, df_test


def build_supervised(df: pd.DataFrame, feature_cols):
    """
    Build [X, y_reg, y_cls] supervised data.

    X: [num_samples, SEQ_LEN, num_features]
    y_reg: [num_samples, HORIZON] future percentage returns
           r_k = C_{t+k} / C_{t-1} - 1 for k = 1..HORIZON
    y_cls: [num_samples] 0/1/2 (bear/neutral/bull) based on final future return
    """
    closes = df["Close"].values
    feats = df[feature_cols].values.astype(np.float32)

    X, y_reg, y_cls = [], [], []
    n = len(df)

    for i in range(SEQ_LEN, n - HORIZON):
        last_close = closes[i - 1]
        future_closes = closes[i:i + HORIZON]

        # Skip weird data (zero / negative closes)
        if last_close <= 0 or np.any(future_closes <= 0):
            continue

        # --- inputs: past SEQ_LEN feature rows ---
        X.append(feats[i-SEQ_LEN:i])

        # --- regression target: future percentage returns vs last_close ---
        # r_k = C_{i+k} / C_{i-1} - 1
        future_rets = (future_closes / last_close) - 1.0
        y_reg.append(future_rets.astype(np.float32))

        # --- classification target: bull / bear / neutral ---
        final_future = future_closes[-1]
        fut_ret = (final_future / last_close) - 1.0

        if fut_ret <= BEAR_THRESHOLD:
            label = 0  # bear
        elif fut_ret >= BULL_THRESHOLD:
            label = 2  # bull
        else:
            label = 1  # neutral

        y_cls.append(label)

    X = np.array(X, dtype=np.float32)
    y_reg = np.array(y_reg, dtype=np.float32)
    y_cls = np.array(y_cls, dtype=np.int64)

    # Just a sanity check you can keep or drop:
    # print("Supervised shapes:", X.shape, y_reg.shape, y_cls.shape)

    return X, y_reg, y_cls


def prepare_data():
    """Load CSV, add features, split by time, scale, return loaders + metadata."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = add_features(df)

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "ret_1", "ret_15m",
        "rsi14", "stoch_k", "stoch_d",
        "macd", "macd_signal", "macd_hist",
        "williams_r",
        "sma_20", "sma_50", "sma_100",
        "ema_9", "ema_21", "ema_50",
        "vol_z",
        "minute_sin", "minute_cos",
        "dow_sin", "dow_cos",
    ]

    df_train, df_val, df_test = make_splits(df)

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    def transform_and_build(df_split):
        df_split = df_split.copy()
        df_split[feature_cols] = scaler.transform(df_split[feature_cols].values)
        return build_supervised(df_split, feature_cols)

    X_train, y_reg_train, y_cls_train = transform_and_build(df_train)
    X_val, y_reg_val, y_cls_val = transform_and_build(df_val)
    X_test, y_reg_test, y_cls_test = transform_and_build(df_test)

    train_ds = TimeSeriesDataset(X_train, y_reg_train, y_cls_train)
    val_ds = TimeSeriesDataset(X_val, y_reg_val, y_cls_val)
    test_ds = TimeSeriesDataset(X_test, y_reg_test, y_cls_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, feature_cols, df

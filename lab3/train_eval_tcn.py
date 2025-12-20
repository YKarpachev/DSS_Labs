import os, json, math, argparse
import numpy as np
import pandas as pd
import joblib

from dataclasses import asdict, dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import weight_norm


def ema(s: pd.Series, span: int) -> pd.Series:
    """
    Compute the exponential moving average (EMA) of a series.

    Args:
        s: Input time series (e.g., Close prices).
        span: EMA span parameter (roughly comparable to window size).

    Returns:
        EMA series aligned to `s` index.
    """
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using exponentially-smoothed gains/losses.

    RSI is computed from price changes:
      - Separate positive (up) and negative (down) deltas.
      - Smooth both with EWM(alpha=1/period).
      - RSI = 100 - 100 / (1 + RS), where RS = smoothed_up / smoothed_down.

    Args:
        close: Close price series.
        period: RSI lookback period.

    Returns:
        RSI series in [0, 100], aligned to `close` index.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1 / period, adjust=False).mean() / (
        down.ewm(alpha=1 / period, adjust=False).mean() + 1e-12
    )
    return 100 - (100 / (1 + rs))


def true_range(high, low, close):
    """
    Compute the True Range (TR), a volatility measure.

    TR[t] = max(
        |High[t] - Low[t]|,
        |High[t] - Close[t-1]|,
        |Low[t]  - Close[t-1]|
    )

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.

    Returns:
        True Range series aligned to inputs.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high, low, close, period: int = 14):
    """
    Compute the Average True Range (ATR) using exponential smoothing.

    ATR is an EWM-smoothed version of True Range.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR smoothing period.

    Returns:
        ATR series aligned to inputs.
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    """
    Compute MACD (Moving Average Convergence Divergence).

    MACD line = EMA(close, fast) - EMA(close, slow)
    Signal line = EMA(MACD line, signal)
    Histogram = MACD line - Signal line

    Args:
        close: Close price series.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        Tuple of (macd_line, signal_line, hist) as series aligned to `close`.
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close, period=20, nstd=2.0):
    """
    Compute Bollinger Bands and derived features.

    Mid = rolling mean(close, period)
    Upper/Lower = Mid ± nstd * rolling_std(close, period)

    Also returns:
      - width = (upper - lower) / mid
      - pct_b = (close - lower) / (upper - lower)

    Args:
        close: Close price series.
        period: Rolling window length.
        nstd: Number of standard deviations for bands.

    Returns:
        (mid, upper, lower, width, pct_b) as series aligned to `close`.
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    width = (upper - lower) / (mid + 1e-12)
    pct_b = (close - lower) / (upper - lower + 1e-12)
    return mid, upper, lower, width, pct_b


def adx(high, low, close, period=14):
    """
    Compute ADX (Average Directional Index) and directional indicators (+DI, -DI).

    Steps:
      - Compute directional movement +DM and -DM from high/low changes.
      - Compute ATR (EWM-smoothed TR).
      - Compute +DI, -DI as smoothed DM / ATR.
      - DX = 100 * |+DI - -DI| / (+DI + -DI)
      - ADX = EWM-smoothed DX

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Smoothing period.

    Returns:
        (adx, plus_di, minus_di) as series aligned to inputs.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        / (atr_ + 1e-12)
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        / (atr_ + 1e-12)
    )
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_, plus_di, minus_di


def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP (Volume Weighted Average Price), resetting each calendar day.

    VWAP for each timestamp t within a day is:
      VWAP[t] = cumulative_sum(typical_price * volume) / cumulative_sum(volume)

    Typical price = (High + Low + Close) / 3

    Requires:
      - datetime-like index (used to group by date)
      - columns: High, Low, Close, Volume

    Args:
        df: DataFrame with OHLCV columns and datetime index.

    Returns:
        VWAP series aligned to df.index.
    """
    dt = df.index
    date_key = dt.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    grp = pd.Series(date_key, index=df.index)
    cum_pv = pv.groupby(grp).cumsum()
    cum_v = df["Volume"].groupby(grp).cumsum()
    return cum_pv / (cum_v + 1e-12)


def add_user_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical-indicator features and heuristic regime/score labels.

    Adds (among others):
      - ema20, ema50, ema_slope
      - rsi14
      - bb_width, bb_pctb
      - atr14, atr_pct
      - macd_hist, macd_hist_chg
      - adx14, pdi14, mdi14
      - vwap, vwap_dist

    Also constructs a heuristic `signal_score` based on multiple rules:
      - RSI overbought/oversold
      - MACD histogram sign and change
      - EMA alignment and slope
      - Bollinger %B extremes
      - VWAP distance
      - ADX trend direction

    Then maps that score to:
      - state_5: {strong sell, sell, neutral, buy, strong buy}
      - momentum_3: {bull, bear, neutral} based on EMA/ADX conditions

    Note:
        The heuristic label columns (signal_score/state_5/momentum_3) are not
        necessarily used by the model unless included in `feature_cols`.

    Args:
        df: DataFrame with OHLCV columns and datetime index.

    Returns:
        Copy of df with added indicator/label columns.
    """
    out = df.copy()
    out["ema20"] = ema(out["Close"], 20)
    out["ema50"] = ema(out["Close"], 50)
    out["ema_slope"] = out["ema20"].diff(4) / (out["ema20"].shift(4) + 1e-12)

    out["rsi14"] = rsi(out["Close"], 14)

    _, _, _, out["bb_width"], out["bb_pctb"] = bollinger(out["Close"], 20, 2.0)
    out["atr14"] = atr(out["High"], out["Low"], out["Close"], 14)
    out["atr_pct"] = out["atr14"] / (out["Close"] + 1e-12)

    _, _, out["macd_hist"] = macd(out["Close"], 12, 26, 9)
    out["macd_hist_chg"] = out["macd_hist"].diff()

    out["adx14"], out["pdi14"], out["mdi14"] = adx(
        out["High"], out["Low"], out["Close"], 14
    )

    out["vwap"] = intraday_vwap(out)
    out["vwap_dist"] = (out["Close"] - out["vwap"]) / (out["Close"] + 1e-12)

    score = pd.Series(0.0, index=out.index)

    score += np.where(out["rsi14"] <= 25, +2.0, 0.0)
    score += np.where((out["rsi14"] > 25) & (out["rsi14"] <= 40), +1.0, 0.0)
    score += np.where((out["rsi14"] >= 60) & (out["rsi14"] < 75), -1.0, 0.0)
    score += np.where(out["rsi14"] >= 75, -2.0, 0.0)

    score += np.where((out["macd_hist"] > 0) & (out["macd_hist_chg"] > 0), +1.5, 0.0)
    score += np.where((out["macd_hist"] < 0) & (out["macd_hist_chg"] < 0), -1.5, 0.0)

    score += np.where(
        (out["Close"] > out["ema20"]) & (out["ema20"] > out["ema50"]), +1.5, 0.0
    )
    score += np.where(
        (out["Close"] < out["ema20"]) & (out["ema20"] < out["ema50"]), -1.5, 0.0
    )
    score += np.where(out["ema_slope"] > 0.002, +0.5, 0.0)
    score += np.where(out["ema_slope"] < -0.002, -0.5, 0.0)

    score += np.where(out["bb_pctb"] < 0.1, +0.5, 0.0)
    score += np.where(out["bb_pctb"] > 0.9, -0.5, 0.0)

    score += np.where(out["vwap_dist"] > 0.002, +0.5, 0.0)
    score += np.where(out["vwap_dist"] < -0.002, -0.5, 0.0)

    strong_trend = out["adx14"] > 20
    score += np.where(strong_trend & (out["pdi14"] > out["mdi14"]), +0.5, 0.0)
    score += np.where(strong_trend & (out["mdi14"] > out["pdi14"]), -0.5, 0.0)

    out["signal_score"] = score

    def to_state5(x: float) -> str:
        """
        Map a continuous signal score to a 5-state discrete label.

        Args:
            x: Signal score value.

        Returns:
            One of: "strong sell", "sell", "neutral", "buy", "strong buy".
        """
        if x <= -3.0:
            return "strong sell"
        if x <= -1.0:
            return "sell"
        if x < 1.0:
            return "neutral"
        if x < 3.0:
            return "buy"
        return "strong buy"

    out["state_5"] = out["signal_score"].apply(to_state5)

    bull = (out["ema20"] > out["ema50"]) & (out["ema_slope"] > 0) & (out["adx14"] > 18)
    bear = (out["ema20"] < out["ema50"]) & (out["ema_slope"] < 0) & (out["adx14"] > 18)
    out["momentum_3"] = np.where(bull, "bull", np.where(bear, "bear", "neutral"))
    return out


def add_targets(df: pd.DataFrame, horizons=(1, 2, 3)) -> pd.DataFrame:
    """
    Add supervised learning targets for multiple future horizons.

    For each horizon h:
      - ret_h: log return over h steps: log(Close[t+h] / Close[t])
      - dir_h: direction label: 1 if ret_h > 0 else 0

    Notes:
      - Uses Close.shift(-h), so the last h rows will have NaNs for ret_h/dir_h.

    Args:
        df: Input dataframe containing at least the "Close" column.
        horizons: Iterable of integer horizons (in bars/rows).

    Returns:
        Copy of df with added columns: ret_{h}, dir_{h} for each horizon.
    """
    out = df.copy()
    for h in horizons:
        out[f"ret_{h}"] = (np.log(out["Close"].shift(-h) / out["Close"])).astype(
            np.float32
        )
        out[f"dir_{h}"] = (out[f"ret_{h}"] > 0).astype(np.float32)
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the model feature set.

    Steps:
      1) Calls `add_user_labels` to add indicators.
      2) Adds additional statistical and time-of-day features.
      3) Returns both the enriched dataframe and the explicit list of model inputs.

    The returned `feature_cols` is the authoritative list of numeric features that
    will be used by `build_sequences` to create model inputs.

    Args:
        df: DataFrame with OHLCV columns and datetime index.

    Returns:
        A tuple of:
          - out: DataFrame with all engineered features.
          - feature_cols: list of feature column names used as model inputs.
    """
    out = add_user_labels(df)

    out["log_ret1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_ret4"] = np.log(out["Close"] / out["Close"].shift(4))
    out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["co_change"] = (out["Close"] - out["Open"]) / (out["Open"] + 1e-12)

    out["vol_log"] = np.log(out["Volume"].replace(0, np.nan))
    out["vol_z_96"] = (out["vol_log"] - out["vol_log"].rolling(96).mean()) / (
        out["vol_log"].rolling(96).std(ddof=0) + 1e-12
    )

    out["rv_32"] = out["log_ret1"].rolling(32).std(ddof=0)
    out["rv_96"] = out["log_ret1"].rolling(96).std(ddof=0)

    out["z_close_96"] = (out["Close"] - out["Close"].rolling(96).mean()) / (
        out["Close"].rolling(96).std(ddof=0) + 1e-12
    )

    minutes = out.index.hour * 60 + out.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    out["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    out["dow"] = out.index.dayofweek.astype(np.float32)

    feature_cols = [
        "log_ret1",
        "log_ret4",
        "hl_range",
        "co_change",
        "vol_log",
        "vol_z_96",
        "rv_32",
        "rv_96",
        "z_close_96",
        "rsi14",
        "macd_hist",
        "macd_hist_chg",
        "atr_pct",
        "bb_width",
        "bb_pctb",
        "ema20",
        "ema50",
        "ema_slope",
        "adx14",
        "pdi14",
        "mdi14",
        "vwap_dist",
        "tod_sin",
        "tod_cos",
        "dow",
    ]
    return out, feature_cols


class SeqDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-multi-horizon targets.

    Stores:
      - X: input sequences shaped (N, C, L)
      - y_ret: regression targets shaped (N, H)
      - y_dir: classification targets shaped (N, H)

    __getitem__ returns tensors:
      (X_i [C,L], y_ret_i [H], y_dir_i [H])
    """

    def __init__(self, X: np.ndarray, y_ret: np.ndarray, y_dir: np.ndarray):
        """
        Args:
            X: Input array of shape (N, C, L).
            y_ret: Regression targets of shape (N, H).
            y_dir: Classification targets of shape (N, H).
        """
        self.X = X
        self.y_ret = y_ret
        self.y_dir = y_dir

    def __len__(self):
        """Return number of samples N."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Fetch one sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
              - X[idx] float32 tensor shaped (C, L)
              - y_ret[idx] float32 tensor shaped (H,)
              - y_dir[idx] float32 tensor shaped (H,)
        """
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_ret[idx], dtype=torch.float32),
            torch.tensor(self.y_dir[idx], dtype=torch.float32),
        )


def build_sequences(df_feat: pd.DataFrame, feature_cols, lookback, horizons):
    """
    Convert a feature dataframe into sliding-window sequences suitable for a TCN.

    For each time index t (where a full lookback window and future horizons exist):
      - X[t] is built from the last `lookback` rows of `feature_cols`,
        transposed to shape (C, L) where:
          C = len(feature_cols)
          L = lookback
      - y_ret[t] is the vector [ret_h] for each horizon h
      - y_dir[t] is the vector [dir_h] for each horizon h

    Samples containing NaN/inf in inputs or targets are skipped.

    Args:
        df_feat: DataFrame containing feature columns and target columns ret_h/dir_h.
        feature_cols: List of feature column names to use as inputs.
        lookback: Number of past bars/rows in each input sequence.
        horizons: Iterable of integer horizons for targets.

    Returns:
        Tuple:
          - X: np.ndarray (N, C, L)
          - y_ret: np.ndarray (N, H)
          - y_dir: np.ndarray (N, H)
          - t_idx: np.ndarray (N,) indices into df_feat corresponding to each sample's t
    """
    max_h = max(horizons)
    start_t = lookback - 1
    end_t = len(df_feat) - max_h - 1

    feat_mat = df_feat[feature_cols].values.astype(np.float32)

    X_list, yret_list, ydir_list, t_idx = [], [], [], []
    for t in range(start_t, end_t + 1):
        x = feat_mat[t - lookback + 1 : t + 1].T  # (C, L)
        yret = np.array(
            [df_feat.iloc[t][f"ret_{h}"] for h in horizons], dtype=np.float32
        )
        ydir = np.array(
            [df_feat.iloc[t][f"dir_{h}"] for h in horizons], dtype=np.float32
        )

        if (
            np.any(~np.isfinite(x))
            or np.any(~np.isfinite(yret))
            or np.any(~np.isfinite(ydir))
        ):
            continue

        X_list.append(x)
        yret_list.append(yret)
        ydir_list.append(ydir)
        t_idx.append(t)

    X = np.stack(X_list)
    y_ret = np.stack(yret_list)
    y_dir = np.stack(ydir_list)
    t_idx = np.array(t_idx, dtype=np.int64)
    return X, y_ret, y_dir, t_idx


class Chomp1d(nn.Module):
    """
    Remove extra timesteps introduced by padding in causal Conv1d.

    In a causal TCN, conv layers use padding so the convolution can be computed
    without shrinking the sequence length. Chomp1d then trims the padded tail to
    ensure the output at time t does not depend on future timesteps.
    """

    def __init__(self, chomp_size: int):
        """
        Args:
            chomp_size: Number of timesteps to remove from the end of the sequence.
                        Typically equals the padding used in the Conv1d.
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, L + chomp_size) if padded.

        Returns:
            Tensor with last `chomp_size` timesteps removed: shape (B, C, L).
        """
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    A residual Temporal Convolutional Network (TCN) block.

    Structure:
      (Conv1d -> Chomp -> GELU -> Dropout) x 2
      + residual connection (optionally 1x1 downsample to match channels)
      -> GELU

    Uses dilated convolutions to expand receptive field while keeping depth manageable.
    """

    def __init__(self, in_ch, out_ch, k, dilation, dropout):
        """
        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            k: Convolution kernel size.
            dilation: Dilation factor for Conv1d.
            dropout: Dropout probability.
        """
        super().__init__()
        padding = (k - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, in_ch, L)

        Returns:
            Output tensor of shape (B, out_ch, L)
        """
        y = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.act2(self.chomp2(self.conv2(y))))
        res = x if self.downsample is None else self.downsample(x)
        return self.act(y + res)


class TCNModel(nn.Module):
    """
    Multi-task TCN model for multi-horizon forecasting.

    Input:
      x: (B, C, L) where C = num_features, L = lookback

    Shared trunk:
      Stack of TemporalBlocks with increasing dilation.

    Heads:
      - Regression head predicts future log-returns for each horizon.
      - Classification head predicts logits for future direction (up/down) for each horizon.
    """

    def __init__(
        self, num_features, horizons=3, channels=(96, 96, 96, 96), k=3, dropout=0.15
    ):
        """
        Args:
            num_features: Number of input feature channels C.
            horizons: Number of horizons H to predict (size of outputs).
            channels: Tuple of channel sizes for each TemporalBlock.
            k: Convolution kernel size.
            dropout: Dropout probability used in blocks and heads.
        """
        super().__init__()
        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(in_ch, out_ch, k=k, dilation=2**i, dropout=dropout)
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(channels[-1])

        self.reg_head = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizons),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizons),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor shaped (B, C, L).

        Returns:
            Tuple:
              - ret: predicted log-returns, shape (B, H)
              - logits: classification logits (pre-sigmoid), shape (B, H)
        """
        z = self.tcn(x)          # (B, channels[-1], L)
        last = z[:, :, -1]       # (B, channels[-1])
        last = self.norm(last)   # (B, channels[-1])
        ret = self.reg_head(last)
        logits = self.cls_head(last)
        return ret, logits


@dataclass
class Config:
    """
    Configuration container for data paths, model hyperparameters, and training settings.

    Attributes:
        csv: Path to the OHLCV CSV input.
        out_dir: Directory where model/scaler/artifacts will be saved.
        lookback: Number of past bars per input sequence.
        horizons: Tuple of forecast horizons in bars.
        batch: Batch size for DataLoaders.
        epochs: Maximum training epochs.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        dropout: Dropout probability in model.
        alpha_dir: Weight multiplier for direction classification loss.
        grad_clip: Max norm for gradient clipping.
        test_frac: Fraction of samples reserved for chronological test split.
        device: Torch device string ("cuda" if available else "cpu").
        seed: Random seed for reproducibility.
    """
    csv: str
    out_dir: str
    lookback: int = 256
    horizons: tuple = (1, 2, 3)
    batch: int = 128
    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-3
    dropout: float = 0.15
    alpha_dir: float = 0.7
    grad_clip: float = 1.0
    test_frac: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def seed_all(seed: int):
    """
    Seed numpy and PyTorch RNGs for reproducibility.

    Args:
        seed: Seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, opt, loss_ret_fn, loss_dir_fn, cfg: Config):
    """
    Train the model for one epoch.

    Computes a multi-task loss:
      loss = loss_ret + cfg.alpha_dir * loss_dir
    where:
      - loss_ret is regression loss on future returns
      - loss_dir is BCE-with-logits loss on direction labels

    Applies gradient clipping and performs optimizer steps.

    Args:
        model: TCNModel (or compatible) instance.
        loader: DataLoader yielding (X, y_ret, y_dir).
        opt: Optimizer (AdamW).
        loss_ret_fn: Regression loss function (e.g., SmoothL1Loss).
        loss_dir_fn: Classification loss function (e.g., BCEWithLogitsLoss).
        cfg: Config containing device, alpha_dir, grad_clip, etc.

    Returns:
        Average training loss over all samples in the loader.
    """
    model.train()
    total = 0.0
    for X, y_ret, y_dir in loader:
        X, y_ret, y_dir = X.to(cfg.device), y_ret.to(cfg.device), y_dir.to(cfg.device)

        opt.zero_grad(set_to_none=True)
        pred_ret, pred_logits = model(X)

        loss_ret = loss_ret_fn(pred_ret, y_ret)
        loss_dir = loss_dir_fn(pred_logits, y_dir)
        loss = loss_ret + cfg.alpha_dir * loss_dir

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, loss_ret_fn, loss_dir_fn, cfg: Config):
    """
    Evaluate the average multi-task loss on a validation/test DataLoader.

    Uses the same combined loss as training:
      loss = loss_ret + cfg.alpha_dir * loss_dir

    Args:
        model: Model to evaluate.
        loader: DataLoader yielding (X, y_ret, y_dir).
        loss_ret_fn: Regression loss function.
        loss_dir_fn: Classification loss function.
        cfg: Config containing device and alpha_dir.

    Returns:
        Average loss over all samples in the loader.
    """
    model.eval()
    total = 0.0
    for X, y_ret, y_dir in loader:
        X, y_ret, y_dir = X.to(cfg.device), y_ret.to(cfg.device), y_dir.to(cfg.device)
        pred_ret, pred_logits = model(X)

        loss_ret = loss_ret_fn(pred_ret, y_ret)
        loss_dir = loss_dir_fn(pred_logits, y_dir)
        loss = loss_ret + cfg.alpha_dir * loss_dir
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def collect_probs_and_labels(model, X: np.ndarray, y_dir: np.ndarray, cfg: Config):
    """
    Run the model on a full input array and return sigmoid probabilities for direction.

    Note:
        This function runs the entire X tensor at once (no batching). For very large
        datasets, this may be memory-heavy.

    Args:
        model: Trained model.
        X: Input array of shape (N, C, L).
        y_dir: Direction labels array of shape (N, H) (returned unchanged).
        cfg: Config containing device.

    Returns:
        Tuple:
          - probs: np.ndarray of shape (N, H), probabilities in [0,1]
          - y_dir: the provided labels (np.ndarray)
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(cfg.device)
    _, logits = model(X_t)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs, y_dir


def best_threshold_for_precision(
    probs: np.ndarray, y: np.ndarray, min_positives: int = 50
):
    """
    Choose a classification threshold that maximizes precision, subject to a minimum
    number of predicted positives.

    Searches thresholds on a fixed grid in [0.05, 0.95]. For each threshold t:
      - pred = probs >= t
      - if pred.sum() < min_positives: skip (avoids degenerate high-precision from tiny samples)
      - compute precision and keep the best

    Args:
        probs: Probability predictions (N,) for a single horizon.
        y: Ground-truth binary labels (N,).
        min_positives: Minimum count of predicted positives required to consider a threshold.

    Returns:
        (best_threshold, best_precision)
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_p = 0.5, -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        if pred.sum() < min_positives:
            continue
        p = precision_score(y, pred, zero_division=0)
        if p > best_p:
            best_p, best_t = p, t
    return best_t, best_p


def main():
    """
    End-to-end training + evaluation entrypoint.

    Pipeline:
      1) Parse CLI args (csv path, out dir, lookback, epochs, batch).
      2) Load OHLCV CSV, sort by datetime, set datetime as index.
      3) Build features and targets.
      4) Build sequences (X, y_ret, y_dir) using lookback windows.
      5) Chronological split: last `test_frac` samples as test.
      6) Fit StandardScaler on training period rows only; transform all features.
      7) Rebuild sequences using scaled features.
      8) Split train into train/val (val is the last 15% of train).
      9) Train TCN with early stopping based on validation loss.
     10) On validation, choose per-horizon probability thresholds maximizing precision.
     11) Evaluate on test split and report metrics per horizon.
     12) Save artifacts:
          - artifacts.json (config, feature_cols, thresholds, metrics)
          - scaler.joblib
          - model.pt

    CLI example:
        python train_eval_tcn.py --csv nvda_15m_clean.csv --out artifacts_nvda_tcn --lookback 256 --epochs 30 --batch 128
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help="Input OHLCV CSV with columns: datetime,Open,High,Low,Close,Volume",
    )
    ap.add_argument(
        "--out", default="artifacts_tcn", help="Output directory for artifacts"
    )
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    cfg = Config(
        csv=args.csv,
        out_dir=args.out,
        lookback=args.lookback,
        epochs=args.epochs,
        batch=args.batch,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_all(cfg.seed)

    df = (
        pd.read_csv(cfg.csv, parse_dates=["datetime"])
        .sort_values("datetime")
        .set_index("datetime")
    )

    df_feat, feature_cols = build_features(df)
    df_feat = add_targets(df_feat, cfg.horizons)

    X, y_ret, y_dir, t_idx = build_sequences(
        df_feat, feature_cols, cfg.lookback, cfg.horizons
    )

    n = len(X)
    test_n = int(math.ceil(n * cfg.test_frac))
    train_n = n - test_n

    X_train_full, yret_train_full, ydir_train_full = (
        X[:train_n],
        y_ret[:train_n],
        y_dir[:train_n],
    )
    X_test, yret_test, ydir_test = X[train_n:], y_ret[train_n:], y_dir[train_n:]

    last_train_t = t_idx[train_n - 1]
    train_rows = df_feat.iloc[: last_train_t + 1]

    scaler = StandardScaler()
    scaler.fit(train_rows[feature_cols].dropna().values)
    df_feat_scaled = df_feat.copy()
    df_feat_scaled[feature_cols] = scaler.transform(df_feat_scaled[feature_cols].values)

    Xs, y_ret_s, y_dir_s, _ = build_sequences(
        df_feat_scaled, feature_cols, cfg.lookback, cfg.horizons
    )
    X_train_full, yret_train_full, ydir_train_full = (
        Xs[:train_n],
        y_ret_s[:train_n],
        y_dir_s[:train_n],
    )
    X_test, yret_test, ydir_test = Xs[train_n:], y_ret_s[train_n:], y_dir_s[train_n:]

    val_frac = 0.15
    val_n = int(math.ceil(len(X_train_full) * val_frac))
    tr_n = len(X_train_full) - val_n

    X_tr, yret_tr, ydir_tr = (
        X_train_full[:tr_n],
        yret_train_full[:tr_n],
        ydir_train_full[:tr_n],
    )
    X_val, yret_val, ydir_val = (
        X_train_full[tr_n:],
        yret_train_full[tr_n:],
        ydir_train_full[tr_n:],
    )

    train_loader = DataLoader(
        SeqDataset(X_tr, yret_tr, ydir_tr),
        batch_size=cfg.batch,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        SeqDataset(X_val, yret_val, ydir_val), batch_size=cfg.batch, shuffle=False
    )

    model = TCNModel(
        num_features=len(feature_cols),
        horizons=len(cfg.horizons),
        channels=(96, 96, 96, 96),
        k=3,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )

    loss_ret_fn = nn.SmoothL1Loss(beta=0.5)
    loss_dir_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    patience = 7
    bad = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, opt, loss_ret_fn, loss_dir_fn, cfg
        )
        va_loss = eval_loss(model, val_loader, loss_ret_fn, loss_dir_fn, cfg)
        scheduler.step(va_loss)

        print(f"epoch {epoch:02d} | train_loss {tr_loss:.6f} | val_loss {va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs_test, y_test_dir = collect_probs_and_labels(model, X_test, ydir_test, cfg)

    probs_val, y_val_dir_np = collect_probs_and_labels(model, X_val, ydir_val, cfg)
    thresholds = []
    for i, h in enumerate(cfg.horizons):
        t, p = best_threshold_for_precision(
            probs_val[:, i], y_val_dir_np[:, i].astype(int), min_positives=50
        )
        thresholds.append(float(t))
        print(f"[val] horizon {h}: best precision {p:.4f} at threshold {t:.2f}")

    report = {}
    for i, h in enumerate(cfg.horizons):
        y_true = y_test_dir[:, i].astype(int)
        y_prob = probs_test[:, i]
        y_pred = (y_prob >= thresholds[i]).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")

        report[f"h{h}"] = {
            "threshold": thresholds[i],
            "precision": float(prec),
            "accuracy": float(acc),
            "f1": float(f1),
            "auc": float(auc),
            "positives_pred": int(y_pred.sum()),
            "positives_true": int(y_true.sum()),
        }

    print("\n=== TEST REPORT (last 15% samples) ===")
    print(json.dumps(report, indent=2))

    artifacts = {
        "config": asdict(cfg),
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "report_test": report,
    }
    with open(os.path.join(cfg.out_dir, "artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    joblib.dump(scaler, os.path.join(cfg.out_dir, "scaler.joblib"))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))

    print(f"\nsaved artifacts to: {cfg.out_dir}")


if __name__ == "__main__":
    main()

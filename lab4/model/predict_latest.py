import os, json, argparse
from contextlib import nullcontext

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def df_row_to_payload(out: pd.DataFrame) -> dict:
    """
    Convert a one-row output DataFrame into a JSON-serializable payload.

    The function:
      - takes the first row of `out` (assumes `out` has at least one row),
      - adds a "timestamp" field from the DataFrame index (assumes datetime index),
      - converts numpy scalar types to native Python types,
      - converts NaN to None for JSON compatibility.

    Args:
        out: DataFrame with a datetime index and at least one row. Typically the
             prediction output DataFrame built in `main()`.

    Returns:
        A dict containing:
          - "timestamp": ISO-8601 string
          - all columns from the first row converted to JSON-friendly types
    """
    row = out.iloc[0].to_dict()
    ts = out.index[0]
    payload = {"timestamp": ts.isoformat()}

    for k, v in row.items():
        if isinstance(v, (np.floating, np.float32, np.float64)):
            payload[k] = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            payload[k] = int(v)
        elif pd.isna(v):
            payload[k] = None
        else:
            payload[k] = v
    return payload


def ema(s: pd.Series, span: int) -> pd.Series:
    """
    Compute the exponential moving average (EMA) of a series.

    Args:
        s: Input time series (e.g., Close prices).
        span: EMA span parameter.

    Returns:
        EMA series aligned to `s` index.
    """
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using exponentially-smoothed gains/losses.

    Steps:
      - delta = diff(close)
      - up = positive deltas, down = negative deltas
      - smooth up/down via EWM(alpha=1/period)
      - RS = smoothed_up / smoothed_down
      - RSI = 100 - 100 / (1 + RS)

    Args:
        close: Close price series.
        period: RSI smoothing period.

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
    Compute True Range (TR), a volatility measure.

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
    Compute Average True Range (ATR) using exponential smoothing.

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
        period: Rolling window size.
        nstd: Number of standard deviations for band distance.

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
      - Compute +DM / -DM from high/low changes.
      - Compute ATR-like EWM smoothing of TR.
      - Compute +DI / -DI as smoothed DM divided by ATR.
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
    Compute intraday VWAP (Volume Weighted Average Price), resetting each day.

    VWAP within a day is:
      cumulative_sum(typical_price * volume) / cumulative_sum(volume)

    Typical price = (High + Low + Close) / 3.

    This implementation groups rows by calendar date derived from df.index.

    Args:
        df: DataFrame with datetime index and columns High/Low/Close/Volume.

    Returns:
        VWAP series aligned to df.index.
    """
    dt = df.index
    date_key = dt.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    grp = pd.Series(date_key, index=df.index)
    return pv.groupby(grp).cumsum() / (df["Volume"].groupby(grp).cumsum() + 1e-12)


def add_user_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicator features and heuristic regime/score labels to the OHLCV DataFrame.

    Adds indicator columns:
      - ema20, ema50, ema_slope
      - rsi14
      - bb_width, bb_pctb
      - atr14, atr_pct
      - macd_hist, macd_hist_chg
      - adx14, pdi14, mdi14
      - vwap, vwap_dist

    Also builds a heuristic `signal_score` combining multiple rule-based signals,
    then maps it to:
      - state_5: {strong sell, sell, neutral, buy, strong buy}
      - momentum_3: {bull, bear, neutral}

    Note:
        These heuristic labels are mainly for reporting/inspection; they are not
        automatically used as model inputs unless included in the feature list.

    Args:
        df: DataFrame with OHLCV columns and datetime index.

    Returns:
        Copy of df with additional indicator and label columns.
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
            x: Signal score.

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature dataframe used for inference.

    This mirrors the training-time feature engineering (minus targets), producing:
      - technical indicators via `add_user_labels`
      - returns/statistical features: log_ret1/log_ret4, RV measures, z-scores
      - intraday time encodings: tod_sin/tod_cos
      - day-of-week feature

    Args:
        df: OHLCV DataFrame with datetime index.

    Returns:
        DataFrame with additional engineered feature columns.
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
    return out


class Chomp1d(nn.Module):
    """
    Remove extra timesteps introduced by padding in causal Conv1d.

    In a causal TCN, Conv1d uses padding to avoid shrinking sequence length.
    Chomp1d trims the padded tail to prevent information leakage from "future"
    positions introduced by padding.
    """

    def __init__(self, chomp_size: int):
        """
        Args:
            chomp_size: Number of timesteps to remove from the end.
                        Typically equals the Conv1d padding size.
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: Tensor shaped (B, C, L + chomp_size) if padded.

        Returns:
            Tensor shaped (B, C, L) after trimming.
        """
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    A residual TCN block with two dilated causal Conv1d layers.

    Structure:
      (Conv1d -> Chomp -> GELU -> Dropout) x 2
      + residual (optionally 1x1 Conv for channel matching)
      -> GELU

    Uses weight normalization for Conv1d layers.
    """

    def __init__(self, in_ch, out_ch, k, dilation, dropout):
        """
        Args:
            in_ch: Input channels.
            out_ch: Output channels.
            k: Kernel size.
            dilation: Dilation factor for temporal receptive field expansion.
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
            x: Input tensor (B, in_ch, L)

        Returns:
            Output tensor (B, out_ch, L)
        """
        y = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.act2(self.chomp2(self.conv2(y))))
        res = x if self.downsample is None else self.downsample(x)
        return self.act(y + res)


class TCNModel(nn.Module):
    """
    Multi-task TCN model used for inference.

    Inputs:
      x: (B, C, L) where:
         - C = num_features
         - L = lookback

    Outputs:
      - ret: predicted log-returns per horizon, shape (B, H)
      - logits: direction logits per horizon, shape (B, H)
    """

    def __init__(
        self, num_features, horizons=3, channels=(96, 96, 96, 96), k=3, dropout=0.15
    ):
        """
        Args:
            num_features: Number of input feature channels.
            horizons: Number of horizons to predict (H).
            channels: Channel sizes for each TemporalBlock.
            k: Kernel size.
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
            x: Input tensor of shape (B, C, L).

        Returns:
            Tuple:
              - ret: float tensor of shape (B, H) with predicted log-returns
              - logits: float tensor of shape (B, H) with direction logits (pre-sigmoid)
        """
        z = self.tcn(x)
        last = self.norm(z[:, :, -1])
        ret = self.reg_head(last)
        logits = self.cls_head(last)
        return ret, logits


@torch.no_grad()
def predict_one(model, x_one: np.ndarray, device: str, use_amp: bool = True):
    """
    Run inference on a single (C, L) window and return predictions.

    Uses mixed precision autocast on CUDA by default to speed up inference.

    Args:
        model: Loaded TCNModel.
        x_one: Input window of shape (C, L) with float32 features.
        device: Torch device string ("cpu" or "cuda...").
        use_amp: If True and device is CUDA, use autocast(fp16) for inference.

    Returns:
        Tuple:
          - pred_logret: np.ndarray of shape (H,), predicted log-returns per horizon
          - prob_up: np.ndarray of shape (H,), sigmoid probabilities of upward move
    """
    model.eval()
    x = torch.tensor(x_one[None, ...], dtype=torch.float32, device=device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device.startswith("cuda"))
        else nullcontext()
    )
    with amp_ctx:
        ret, logits = model(x)
        prob = torch.sigmoid(logits)
    return ret.float().cpu().numpy()[0], prob.float().cpu().numpy()[0]


def main():
    """
    CLI entrypoint to predict the latest available window from an OHLCV CSV.

    Inputs:
      - --input: OHLCV CSV file (datetime, Open, High, Low, Close, Volume)
      - --artifacts: directory containing:
          - artifacts.json (feature_cols, thresholds, config)
          - scaler.joblib
          - model.pt
      - --device: cpu/cuda selection
      - --no-amp: disable mixed precision on CUDA
      - --format: stdout output format: json, kv, or csv
      - --output: optional path to write a one-row CSV

    Steps:
      1) Load artifacts.json to get feature_cols, thresholds, lookback, horizons, dropout.
      2) Load the scaler and the trained model weights.
      3) Read the input CSV and compute features (same as training).
      4) Apply the saved scaler to `feature_cols`.
      5) Find the most recent lookback window with finite values (no NaNs/Infs).
      6) Run the model on that window:
          - pred_logret_h (per horizon)
          - prob_up_h (per horizon)
      7) Build a one-row output DataFrame containing:
          - Close and heuristic labels (state_5/momentum_3/signal_score)
          - selected raw indicator columns for inspection
          - predictions, thresholds, and predicted directions
      8) Print in requested format and optionally write CSV to --output.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--output", default="", help="If set, writes one-row CSV here")
    ap.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument(
        "--format",
        choices=["json", "kv", "csv"],
        default="json",
        help="Output format for stdout (json/kv/csv). If --output is set, CSV is written there too.",
    )
    args = ap.parse_args()

    with open(os.path.join(args.artifacts, "artifacts.json"), "r") as f:
        art = json.load(f)

    feature_cols = art["feature_cols"]
    thresholds = art["thresholds"]
    lookback = int(art["config"]["lookback"])
    horizons = art["config"]["horizons"]
    dropout = float(art["config"]["dropout"])

    scaler = joblib.load(os.path.join(args.artifacts, "scaler.joblib"))

    df = (
        pd.read_csv(args.input, parse_dates=["datetime"])
        .sort_values("datetime")
        .set_index("datetime")
    )

    df_feat_raw = build_features(df.copy())
    df_feat = df_feat_raw.copy()

    df_feat[feature_cols] = scaler.transform(df_feat[feature_cols].values)
    feat_mat = df_feat[feature_cols].values.astype(np.float32)

    # Find latest valid lookback window (no NaNs/Infs).
    max_t = len(df_feat) - 1
    t = None
    window = None
    for cand in range(max_t, lookback - 2, -1):
        w = feat_mat[cand - lookback + 1 : cand + 1]
        if np.isfinite(w).all():
            t = cand
            window = w
            break
    if t is None or window is None:
        raise RuntimeError("No valid latest window found (features contain NaNs).")

    x = window.T  # (C, L)
    ts = df_feat.index[t]
    close_t = float(df.loc[ts, "Close"])

    model = TCNModel(
        num_features=len(feature_cols),
        horizons=len(horizons),
        channels=(96, 96, 96, 96),
        k=3,
        dropout=dropout,
    )
    model.load_state_dict(
        torch.load(os.path.join(args.artifacts, "model.pt"), map_location="cpu")
    )
    model.to(args.device)

    pred_logret, prob_up = predict_one(model, x, args.device, use_amp=(not args.no_amp))

    out = pd.DataFrame(index=[ts])
    out["Close"] = close_t
    out["state_5"] = df_feat_raw.loc[ts, "state_5"]
    out["momentum_3"] = df_feat_raw.loc[ts, "momentum_3"]
    out["signal_score"] = float(df_feat_raw.loc[ts, "signal_score"])

    raw_cols = [
        "rsi14",
        "macd_hist",
        "macd_hist_chg",
        "adx14",
        "pdi14",
        "mdi14",
        "ema20",
        "ema50",
        "ema_slope",
        "bb_width",
        "bb_pctb",
        "atr_pct",
        "vwap_dist",
        "vol_log",
        "vol_z_96",
        "rv_32",
        "rv_96",
        "z_close_96",
    ]
    for c in raw_cols:
        if c in df_feat_raw.columns:
            out[c] = float(df_feat_raw.loc[ts, c])

    for i, h in enumerate(horizons):
        out[f"pred_logret_{h}"] = float(pred_logret[i])
        out[f"pred_ret_{h}"] = float(np.expm1(pred_logret[i]))
        out[f"prob_up_{h}"] = float(prob_up[i])
        out[f"pred_dir_{h}"] = int(prob_up[i] >= thresholds[i])
        out[f"threshold_{h}"] = float(thresholds[i])

    payload = df_row_to_payload(out)

    if args.format == "json":
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    elif args.format == "kv":
        for k in sorted(payload.keys()):
            print(f"{k}={payload[k]}")
    elif args.format == "csv":
        print(out.to_csv(index=True).strip())

    if args.output:
        out.to_csv(args.output)


if __name__ == "__main__":
    main()

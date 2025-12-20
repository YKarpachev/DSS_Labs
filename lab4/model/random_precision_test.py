import os, json, argparse, math
from contextlib import nullcontext

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from sklearn.metrics import precision_score, accuracy_score, f1_score


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / (down.ewm(alpha=1/period, adjust=False).mean() + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period: int = 14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def macd(close, fast=12, slow=26, signal=9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, period=20, nstd=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    width = (upper - lower) / (mid + 1e-12)
    pct_b = (close - lower) / (upper - lower + 1e-12)
    return mid, upper, lower, width, pct_b

def adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    adx_ = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_, plus_di, minus_di

def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    dt = df.index
    date_key = dt.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    grp = pd.Series(date_key, index=df.index)
    return pv.groupby(grp).cumsum() / (df["Volume"].groupby(grp).cumsum() + 1e-12)

def add_user_labels(df: pd.DataFrame) -> pd.DataFrame:
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

    out["adx14"], out["pdi14"], out["mdi14"] = adx(out["High"], out["Low"], out["Close"], 14)

    out["vwap"] = intraday_vwap(out)
    out["vwap_dist"] = (out["Close"] - out["vwap"]) / (out["Close"] + 1e-12)
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_user_labels(df)

    out["log_ret1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_ret4"] = np.log(out["Close"] / out["Close"].shift(4))
    out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["co_change"] = (out["Close"] - out["Open"]) / (out["Open"] + 1e-12)

    out["vol_log"] = np.log(out["Volume"].replace(0, np.nan))
    out["vol_z_96"] = (out["vol_log"] - out["vol_log"].rolling(96).mean()) / (out["vol_log"].rolling(96).std(ddof=0) + 1e-12)

    out["rv_32"] = out["log_ret1"].rolling(32).std(ddof=0)
    out["rv_96"] = out["log_ret1"].rolling(96).std(ddof=0)

    out["z_close_96"] = (out["Close"] - out["Close"].rolling(96).mean()) / (out["Close"].rolling(96).std(ddof=0) + 1e-12)

    minutes = out.index.hour * 60 + out.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    out["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    out["dow"] = out.index.dayofweek.astype(np.float32)
    return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout):
        super().__init__()
        padding = (k - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.GELU()

    def forward(self, x):
        y = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.act2(self.chomp2(self.conv2(y))))
        res = x if self.downsample is None else self.downsample(x)
        return self.act(y + res)

class TCNModel(nn.Module):
    def __init__(self, num_features, horizons=3, channels=(96, 96, 96, 96), k=3, dropout=0.15):
        super().__init__()
        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, k=k, dilation=2**i, dropout=dropout))
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
        z = self.tcn(x)
        last = self.norm(z[:, :, -1])
        ret = self.reg_head(last)
        logits = self.cls_head(last)
        return ret, logits


@torch.no_grad()
def predict_batch(model, X: np.ndarray, device: str, use_amp: bool = True, batch_size: int = 512):
    model.eval()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.startswith("cuda")) else nullcontext()

    pred_prob = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        with amp_ctx:
            _, logits = model(xb)
            prob = torch.sigmoid(logits)
        pred_prob.append(prob.detach().float().cpu().numpy())
        del xb, logits, prob
    return np.vstack(pred_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--n", type=int, default=2000, help="How many random samples from latest 15%")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.artifacts, "artifacts.json"), "r") as f:
        art = json.load(f)

    feature_cols = art["feature_cols"]
    thresholds = np.array(art["thresholds"], dtype=float)
    lookback = int(art["config"]["lookback"])
    horizons = list(art["config"]["horizons"])
    dropout = float(art["config"]["dropout"])
    test_frac = float(art["config"].get("test_frac", 0.15))

    max_h = max(horizons)

    scaler = joblib.load(os.path.join(args.artifacts, "scaler.joblib"))
    df = pd.read_csv(args.input, parse_dates=["datetime"]).sort_values("datetime").set_index("datetime")

    df_feat = build_features(df.copy())
    df_feat[feature_cols] = scaler.transform(df_feat[feature_cols].values)
    feat_mat = df_feat[feature_cols].values.astype(np.float32)

    start_t = lookback - 1
    end_t = len(df) - max_h - 1
    finite_rows = np.isfinite(feat_mat).all(axis=1).astype(np.int32)

    kernel = np.ones(lookback, dtype=np.int32)
    valid_win = np.convolve(finite_rows, kernel, mode="valid") == lookback

    all_t = np.arange(start_t, start_t + len(valid_win))
    all_t = all_t[valid_win]
    all_t = all_t[all_t <= end_t]

    if len(all_t) < 10:
        raise RuntimeError("Not enough valid candidate points for testing.")

    split = int(math.floor(len(all_t) * (1.0 - test_frac)))
    test_t = all_t[split:]
    if len(test_t) == 0:
        raise RuntimeError("Test split is empty (check lookback / data length).")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(test_t))
    chosen_t = rng.choice(test_t, size=n, replace=False)
    chosen_t.sort()

    X = np.stack([feat_mat[t - lookback + 1 : t + 1].T for t in chosen_t], axis=0)

    model = TCNModel(num_features=len(feature_cols), horizons=len(horizons),
                     channels=(96, 96, 96, 96), k=3, dropout=dropout)
    model.load_state_dict(torch.load(os.path.join(args.artifacts, "model.pt"), map_location="cpu"))
    model.to(args.device)

    prob_up = predict_batch(model, X, args.device, use_amp=(not args.no_amp), batch_size=args.batch)

    close = df["Close"].values.astype(np.float64)
    y_true = []
    y_pred = []
    for i, t in enumerate(chosen_t):
        for j, h in enumerate(horizons):
            true_dir = int(close[t + h] > close[t])
            pred_dir = int(prob_up[i, j] >= thresholds[j])
            y_true.append(true_dir)
            y_pred.append(pred_dir)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    prec_micro = precision_score(y_true, y_pred, zero_division=0)
    acc_micro = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, zero_division=0)

    print(f"Random test points: {n} (from latest {int(test_frac*100)}% of valid samples)")
    print(f"Overall (micro) precision: {prec_micro:.4f}")
    print(f"Overall (micro) accuracy:  {acc_micro:.4f}")
    print(f"Overall (micro) f1:        {f1_micro:.4f}")
    print(f"Predicted positives: {int(y_pred.sum())} / {len(y_pred)}")

    for j, h in enumerate(horizons):
        yt = np.array([int(close[t + h] > close[t]) for t in chosen_t], dtype=int)
        yp = (prob_up[:, j] >= thresholds[j]).astype(int)
        p = precision_score(yt, yp, zero_division=0)
        a = accuracy_score(yt, yp)
        f = f1_score(yt, yp, zero_division=0)
        print(f"h{h} precision={p:.4f} accuracy={a:.4f} f1={f:.4f}  positives_pred={int(yp.sum())}/{len(yp)}")


# python random_precision_test.py --input nvda_15m_clean.csv --artifacts artifacts_nvda_tcn --n 2000 --seed 7                                                                                                                                         (base)
if __name__ == "__main__":
    main()

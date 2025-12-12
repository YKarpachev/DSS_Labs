# inference_dss.py

import joblib
import torch
import numpy as np
import pandas as pd

from config import DEVICE, MODEL_PATH, SEQ_LEN
from indicators import add_features
from model import LSTMDSS


# --- Rating rules ---

def rate_rsi(rsi_value):
    if rsi_value >= 70:
        return "strong sell"
    elif rsi_value >= 60:
        return "sell"
    elif rsi_value <= 30:
        return "strong buy"
    elif rsi_value <= 40:
        return "buy"
    else:
        return "neutral"


def rate_stoch(k, d):
    val = k
    if val >= 80:
        return "strong sell"
    elif val >= 60:
        return "sell"
    elif val <= 20:
        return "strong buy"
    elif val <= 40:
        return "buy"
    else:
        return "neutral"


def rate_macd(macd_val, signal_val, hist_val):
    if hist_val >= 0 and macd_val > signal_val:
        if hist_val > 0.5:
            return "strong buy"
        else:
            return "buy"
    elif hist_val <= 0 and macd_val < signal_val:
        if hist_val < -0.5:
            return "strong sell"
        else:
            return "sell"
    else:
        return "neutral"


def rate_ma(price, fast_ma, slow_ma):
    if fast_ma > slow_ma and price > fast_ma > slow_ma:
        return "strong buy"
    elif fast_ma > slow_ma and price > fast_ma:
        return "buy"
    elif fast_ma < slow_ma and price < fast_ma < slow_ma:
        return "strong sell"
    elif fast_ma < slow_ma and price < fast_ma:
        return "sell"
    else:
        return "neutral"


def load_model_and_meta():
    meta = joblib.load(MODEL_PATH.with_suffix(".meta.pkl"))
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = LSTMDSS(
        input_size=len(feature_cols),
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, scaler, feature_cols


def dss_inference(df_raw: pd.DataFrame):
    """Main DSS function using raw cleaned OHLCV dataframe."""
    from config import DATA_PATH  # only for path if needed

    model, scaler, feature_cols = load_model_and_meta()

    df = add_features(df_raw.copy())
    df = df.sort_values("datetime").reset_index(drop=True)

    # last SEQ_LEN rows for sequence
    window = df.iloc[-SEQ_LEN:].copy()
    last_row = window.iloc[-1]

    feats = window[feature_cols].values.astype(np.float32)
    feats_scaled = scaler.transform(feats)
    X = torch.tensor(feats_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_reg, logits = model(X)
        pred_rets = pred_reg.squeeze(0).cpu().numpy()  # r_k = C_{t+k}/C_{t-1} - 1
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # reconstruct future closes from percentage returns
    last_close = float(last_row["Close"])
    pred_closes = last_close * (1.0 + pred_rets)

    # safety: avoid negative prices if model outputs crazy returns
    pred_closes = np.maximum(pred_closes, 0.01)

    momentum_map = {0: "bear", 1: "neutral", 2: "bull"}
    momentum = momentum_map[int(probs.argmax())]

    # Indicators for DSS rating
    rsi_val = float(last_row["rsi14"])
    stoch_k_val = float(last_row["stoch_k"])
    stoch_d_val = float(last_row["stoch_d"])
    macd_val = float(last_row["macd"])
    macd_sig_val = float(last_row["macd_signal"])
    macd_hist_val = float(last_row["macd_hist"])

    price = float(last_row["Close"])
    sma20 = float(last_row["sma_20"])
    sma50 = float(last_row["sma_50"])
    sma100 = float(last_row["sma_100"])
    ema21 = float(last_row["ema_21"])

    scores = {
        "RSI14": {
            "value": rsi_val,
            "rating": rate_rsi(rsi_val),
        },
        "Stoch(14,3,3)": {
            "value": stoch_k_val,
            "rating": rate_stoch(stoch_k_val, stoch_d_val),
        },
        "MACD(12,26,9)": {
            "value": macd_val,
            "hist": macd_hist_val,
            "rating": rate_macd(macd_val, macd_sig_val, macd_hist_val),
        },
        "MA_20_50": {
            "fast": sma20,
            "slow": sma50,
            "rating": rate_ma(price, sma20, sma50),
        },
        "MA_50_100": {
            "fast": sma50,
            "slow": sma100,
            "rating": rate_ma(price, sma50, sma100),
        },
        "EMA_21_vs_SMA50": {
            "ema21": ema21,
            "sma50": sma50,
            "rating": rate_ma(price, ema21, sma50),
        },
    }

    result = {
        "latest_datetime": str(last_row["datetime"]),
        "last_close": price,
        "predicted_next_5_closes": pred_closes.tolist(),
        "momentum": momentum,
        "momentum_probabilities": {
            "bear": float(probs[0]),
            "neutral": float(probs[1]),
            "bull": float(probs[2]),
        },
        "indicator_scores": scores,
    }
    return result


if __name__ == "__main__":
    from config import DATA_PATH
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    res = dss_inference(df_raw)
    import pprint
    pprint.pprint(res)

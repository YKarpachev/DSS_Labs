import math
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 1. CONFIG
CSV_PATH = "NVDA_CLEAN_15M.csv"
SEQ_LEN = 128  # past 128 candles (~4–5 trading days)
PRED_HORIZON = 5  # next 5 candles
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 2. DATA LOADING & FEATURES
def load_and_engineer(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.drop_duplicates(subset=["datetime"])

    df["Close_shift1"] = df["Close"].shift(1)
    df["log_ret_close"] = np.log(df["Close"] / df["Close_shift1"])

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close_shift1"]
    df["oc_gap"] = (df["Open"] - df["Close_shift1"]) / df["Close_shift1"]

    df["log_volume"] = np.log(df["Volume"] + 1)

    dt = df["datetime"]
    minutes = dt.dt.hour * 60 + dt.dt.minute
    df["sin_tod"] = np.sin(2 * math.pi * minutes / (24 * 60))
    df["cos_tod"] = np.cos(2 * math.pi * minutes / (24 * 60))

    dow = dt.dt.dayofweek  # 0 = Monday, ..., 6 = Sunday (we mostly care about 0-4)
    df["sin_dow"] = np.sin(2 * math.pi * dow / 7)
    df["cos_dow"] = np.cos(2 * math.pi * dow / 7)

    # Optional: rolling volatility of log returns
    window_vol = 64
    df["ret_vol_64"] = df["log_ret_close"].rolling(window_vol).std()

    # Drop rows with NaNs from shifts/rolling
    df = df.dropna().reset_index(drop=True)

    # Define feature columns
    feature_cols = [
        "log_ret_close",
        "hl_range",
        "oc_gap",
        "log_volume",
        "sin_tod",
        "cos_tod",
        "sin_dow",
        "cos_dow",
        "ret_vol_64",
    ]

    # Target: future log returns of Close for next PRED_HORIZON steps
    # We build a matrix y where each column is log_ret_close shifted into the future.
    for h in range(1, PRED_HORIZON + 1):
        df[f"y_ret_h{h}"] = df["log_ret_close"].shift(-h)

    # Drop tail rows that don't have full future horizon
    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y_cols = [f"y_ret_h{h}" for h in range(1, PRED_HORIZON + 1)]
    y = df[y_cols].values.astype(np.float32)
    times = df["datetime"].values  # for reference / debugging

    return X, y, times, feature_cols


# 3. SPLIT + SCALING + SEQUENCE DATASET
class SeqDataset(Dataset):
    def __init__(self, X, y, indices, seq_len):
        self.X = X
        self.y = y
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        start = i - self.seq_len
        end = i
        x_seq = self.X[start:end]
        y_vec = self.y[i]
        return torch.from_numpy(x_seq), torch.from_numpy(y_vec)


def make_splits(X, y, times, seq_len, pred_horizon):
    N = len(X)
    # Earliest usable target index
    first_target = seq_len
    last_target = N - 1  # we already ensured y is valid up to N-1 via dropna

    indices = np.arange(first_target, last_target + 1)

    # chronological split
    train_end = int(0.7 * len(indices))
    val_end = int(0.85 * len(indices))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Fit scaler on TRAIN **input** only
    scaler = StandardScaler()
    max_train_row = train_idx[-1]
    scaler.fit(X[: max_train_row + 1])

    X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, train_idx, val_idx, test_idx


# 4. MODEL: LSTM ENCODER + MLP
class LSTMForecaster(nn.Module):
    def __init__(
        self, n_features, hidden_size=128, num_layers=2, pred_horizon=5, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_horizon),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        # Use last layer's hidden state as summary: (num_layers, batch, hidden) -> (batch, hidden)
        last_hidden = h_n[-1]
        pred = self.fc(last_hidden)
        return pred


# 5. TRAINING LOOP
def train_model():
    X, y, times, feature_cols = load_and_engineer(CSV_PATH)
    X_scaled, y, scaler, train_idx, val_idx, test_idx = make_splits(
        X, y, times, SEQ_LEN, PRED_HORIZON
    )

    n_features = X_scaled.shape[1]

    train_ds = SeqDataset(X_scaled, y, train_idx, SEQ_LEN)
    val_ds = SeqDataset(X_scaled, y, val_idx, SEQ_LEN)
    test_ds = SeqDataset(X_scaled, y, test_idx, SEQ_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMForecaster(
        n_features,
        hidden_size=128,
        num_layers=2,
        pred_horizon=PRED_HORIZON,
        dropout=0.2,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "seq_len": SEQ_LEN,
                "pred_horizon": PRED_HORIZON,
            }

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}")

    torch.save(best_state, "nvda_lstm_dss.pt")
    print("Saved best model to nvda_lstm_dss.pt")


# 6. DSS: LIVE PREDICTION INTERFACE
def load_dss(model_path="nvda_lstm_dss.pt"):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    n_features = len(checkpoint["feature_cols"])

    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=128,
        num_layers=2,
        pred_horizon=checkpoint["pred_horizon"],
        dropout=0.0,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    seq_len = checkpoint["seq_len"]
    pred_horizon = checkpoint["pred_horizon"]

    return model, scaler, feature_cols, seq_len, pred_horizon


def preprocess_latest_window(df_window, scaler, feature_cols, seq_len):
    """
    df_window: DataFrame with at least seq_len rows, same raw columns as original CSV.
    Returns: tensor (1, seq_len, n_features)
    """
    # (Assuming df_window is already sorted and has enough history)
    df = df_window.copy()

    df["Close_shift1"] = df["Close"].shift(1)
    df["log_ret_close"] = np.log(df["Close"] / df["Close_shift1"])
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close_shift1"]
    df["oc_gap"] = (df["Open"] - df["Close_shift1"]) / df["Close_shift1"]
    df["log_volume"] = np.log(df["Volume"] + 1)

    dt = df["datetime"]
    minutes = dt.dt.hour * 60 + dt.dt.minute
    df["sin_tod"] = np.sin(2 * math.pi * minutes / (24 * 60))
    df["cos_tod"] = np.cos(2 * math.pi * minutes / (24 * 60))

    dow = dt.dt.dayofweek
    df["sin_dow"] = np.sin(2 * math.pi * dow / 7)
    df["cos_dow"] = np.cos(2 * math.pi * dow / 7)

    df["ret_vol_64"] = df["log_ret_close"].rolling(64).std()

    df = df.dropna().reset_index(drop=True)

    if len(df) < len(df_window):  # due to dropna
        # ensure we still have at least seq_len rows
        df = df.iloc[-len(df_window) :]

    if len(df) < seq_len:
        raise ValueError("Not enough history to build a full sequence")

    feat = df[feature_cols].values.astype(np.float32)
    feat_scaled = scaler.transform(feat)

    x_seq = feat_scaled[-seq_len:]
    x_tensor = torch.from_numpy(x_seq).unsqueeze(0)
    return x_tensor


def predict_next5_closes(df_window, model, scaler, feature_cols, seq_len, pred_horizon):
    """
    df_window: DataFrame of last >= seq_len+some rows with datetime,Open,High,Low,Close,Volume.
    Returns: predicted close prices for next pred_horizon steps.
    """
    model.eval()
    x_tensor = preprocess_latest_window(df_window, scaler, feature_cols, seq_len)
    x_tensor = x_tensor.to(DEVICE)

    with torch.no_grad():
        pred_log_returns = model(x_tensor).cpu().numpy().flatten()  # (pred_horizon,)

    last_close = df_window["Close"].iloc[-1]
    future_prices = []
    cum_log_ret = 0.0
    for h in range(pred_horizon):
        cum_log_ret += float(pred_log_returns[h])
        future_price = last_close * math.exp(cum_log_ret)
        future_prices.append(future_price)

    return np.array(pred_log_returns), np.array(future_prices)


if __name__ == "__main__":
    train_model()

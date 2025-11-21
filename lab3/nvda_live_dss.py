import pandas as pd
from nvda_dss_lstm import load_dss, predict_next5_closes

df = pd.read_csv("NVDA_CLEAN_15M.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

model, scaler, feature_cols, seq_len, pred_horizon = load_dss("nvda_lstm_dss.pt")

# Take the last (seq_len + 100) rows for safety
df_window = df.iloc[-(seq_len + 100) :].copy()

pred_log_rets, pred_closes = predict_next5_closes(
    df_window, model, scaler, feature_cols, seq_len, pred_horizon
)

print("Predicted log-returns:", pred_log_rets)
print("Predicted next 5 Close prices:", pred_closes)

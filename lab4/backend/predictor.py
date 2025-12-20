from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch


class Predictor:
    def __init__(
        self,
        model_dir: Path,
        artifacts_dir: Path,
        device: str | None = None,
        use_amp: bool = True,
    ):
        model_dir = model_dir.resolve()
        sys.path.insert(0, str(model_dir))

        from predict_latest import (
            build_features,
            TCNModel,
            predict_one,
            df_row_to_payload,
        )

        self.build_features = build_features
        self.TCNModel = TCNModel
        self.predict_one = predict_one
        self.df_row_to_payload = df_row_to_payload

        self.artifacts_dir = artifacts_dir.resolve()

        with open(self.artifacts_dir / "artifacts.json", "r") as f:
            art = json.load(f)

        self.feature_cols = art["feature_cols"]
        self.thresholds = art["thresholds"]
        self.lookback = int(art["config"]["lookback"])
        self.horizons = art["config"]["horizons"]
        self.dropout = float(art["config"]["dropout"])

        self.scaler = joblib.load(self.artifacts_dir / "scaler.joblib")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")

        self.model = self.TCNModel(
            num_features=len(self.feature_cols),
            horizons=len(self.horizons),
            channels=(96, 96, 96, 96),
            k=3,
            dropout=self.dropout,
        )
        self.model.load_state_dict(
            torch.load(self.artifacts_dir / "model.pt", map_location="cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def _latest_valid_window(self, feat_mat: np.ndarray) -> tuple[int, np.ndarray]:
        max_t = len(feat_mat) - 1
        for cand in range(max_t, self.lookback - 2, -1):
            w = feat_mat[cand - self.lookback + 1 : cand + 1]
            if np.isfinite(w).all():
                return cand, w
        raise RuntimeError(
            "No valid latest window found (features contain NaNs or not enough usable history)."
        )

    @torch.no_grad()
    def predict_from_csv(self, csv_path: Path) -> dict:
        df = pd.read_csv(csv_path, on_bad_lines="skip")

        if "datetime" not in df.columns:
            raise ValueError(
                f"Feed CSV missing 'datetime' column. Columns: {list(df.columns)}"
            )

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = (
            df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
        )

        required = ["Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in df.columns:
                raise ValueError(
                    f"Feed CSV missing '{c}' column. Columns: {list(df.columns)}"
                )
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=required)

        df_feat_raw = self.build_features(df.copy())
        df_feat = df_feat_raw.copy()

        missing_feat = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing_feat:
            raise RuntimeError(
                f"Feature build mismatch; missing feature columns: {missing_feat}"
            )

        df_feat[self.feature_cols] = self.scaler.transform(
            df_feat[self.feature_cols].values
        )
        feat_mat = df_feat[self.feature_cols].values.astype(np.float32)

        t, window = self._latest_valid_window(feat_mat)
        x = window.T  # (C, L)

        ts = df_feat.index[t]
        close_t = float(df.loc[ts, "Close"])

        pred_logret, prob_up = self.predict_one(
            self.model, x, self.device, use_amp=self.use_amp
        )

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

        for i, h in enumerate(self.horizons):
            out[f"pred_logret_{h}"] = float(pred_logret[i])
            out[f"pred_ret_{h}"] = float(np.expm1(pred_logret[i]))
            out[f"prob_up_{h}"] = float(prob_up[i])
            out[f"pred_dir_{h}"] = int(prob_up[i] >= self.thresholds[i])
            out[f"threshold_{h}"] = float(self.thresholds[i])

        return self.df_row_to_payload(out)

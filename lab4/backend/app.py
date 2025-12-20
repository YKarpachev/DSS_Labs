from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import Predictor
from streamer import CandleStreamer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_DIR = (BASE_DIR.parent / "model").resolve()
ARTIFACTS_DIR = (MODEL_DIR / "artifacts_nvda_tcn").resolve()

FEED_CSV = DATA_DIR / "nvda_15m_clean_to_feed.csv"
ADD_FROM_CSV = DATA_DIR / "nvda_15m_clean_to_add_from.csv"
STATE_JSON = DATA_DIR / "sim_state.json"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


predictor = Predictor(model_dir=MODEL_DIR, artifacts_dir=ARTIFACTS_DIR)
streamer = CandleStreamer(
    add_from_path=ADD_FROM_CSV, feed_path=FEED_CSV, state_path=STATE_JSON
)

io_lock = Lock()

from fastapi import Query, HTTPException
import pandas as pd


@app.get("/candles")
def candles(limit: int = Query(240, ge=10, le=5000)):
    if not FEED_CSV.exists():
        raise HTTPException(400, f"Feed CSV not found: {FEED_CSV}")

    try:
        df = pd.read_csv(FEED_CSV, on_bad_lines="skip")

        if "datetime" not in df.columns:
            raise HTTPException(
                500, f"Feed CSV has no 'datetime' column. Columns: {list(df.columns)}"
            )

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime").tail(limit)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                raise HTTPException(
                    500, f"Feed CSV missing '{col}' column. Columns: {list(df.columns)}"
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        out = []
        for _, r in df.iterrows():
            out.append(
                {
                    "time": int(pd.Timestamp(r["datetime"]).timestamp()),
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "volume": None if pd.isna(r["Volume"]) else float(r["Volume"]),
                }
            )
        return out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"/candles failed: {type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/predict")
def predict():
    with io_lock:
        if not FEED_CSV.exists():
            raise HTTPException(400, "Feed CSV not found")

        try:
            return predictor.predict_from_csv(FEED_CSV)
        except Exception as e:
            raise HTTPException(500, f"/predict failed: {type(e).__name__}: {e}")


@app.post("/next_candle")
def next_candle():
    """
    Move exactly 1 candle from add_from.csv into feed.csv, then return that candle.
    """
    with io_lock:
        try:
            candle = streamer.pop_next_candle()
            streamer.append_to_feed(candle)
            return {"added": candle}
        except Exception as e:
            raise HTTPException(409, str(e))

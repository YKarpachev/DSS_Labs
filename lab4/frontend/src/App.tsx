import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  createChart,
  CandlestickSeries,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  type CandlestickData,
} from "lightweight-charts";

const API = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
};

type PredictPayload = {
  timestamp: string;
  [k: string]: string | number | null;
};

type Recommendation = "Strong Sell" | "Sell" | "Neutral" | "Buy" | "Strong Buy";

function toRecommendation(meanProbUp: number | null): Recommendation {
  if (meanProbUp == null || !Number.isFinite(meanProbUp)) return "Neutral";
  if (meanProbUp >= 0.8) return "Strong Buy";
  if (meanProbUp >= 0.6) return "Buy";
  if (meanProbUp <= 0.2) return "Strong Sell";
  if (meanProbUp <= 0.4) return "Sell";
  return "Neutral";
}

function fmtNum(v: unknown, digits = 4): string {
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(digits);
  return "—";
}

function fmtPct(v: unknown, digits = 2): string {
  if (typeof v === "number" && Number.isFinite(v))
    return `${(v * 100).toFixed(digits)}%`;
  return "—";
}

export default function App() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [pred, setPred] = useState<PredictPayload | null>(null);
  const [loadingCandles, setLoadingCandles] = useState(false);
  const [loadingPred, setLoadingPred] = useState(false);
  const [loadingNext, setLoadingNext] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const chartDivRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const roRef = useRef<ResizeObserver | null>(null);

  async function loadCandles() {
    setLoadingCandles(true);
    setErr(null);
    try {
      const res = await fetch(`${API}/candles?limit=240`);
      if (!res.ok) throw new Error(`candles: ${res.status} ${res.statusText}`);
      const data = (await res.json()) as Candle[];
      setCandles(data);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoadingCandles(false);
    }
  }

  async function makePrediction() {
    setLoadingPred(true);
    setErr(null);
    try {
      const res = await fetch(`${API}/predict`);
      if (!res.ok) throw new Error(`predict: ${res.status} ${res.statusText}`);
      const data = (await res.json()) as PredictPayload;
      setPred(data);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoadingPred(false);
    }
  }

  async function nextCandle() {
    setLoadingNext(true);
    setErr(null);
    try {
      const res = await fetch(`${API}/next_candle`, { method: "POST" });
      if (!res.ok)
        throw new Error(`next_candle: ${res.status} ${res.statusText}`);

      setPred(null);
      await loadCandles();
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoadingNext(false);
    }
  }

  const meanProbUp = useMemo(() => {
    if (!pred) return null;
    const probs: number[] = [];
    for (const [k, v] of Object.entries(pred)) {
      if (
        k.startsWith("prob_up_") &&
        typeof v === "number" &&
        Number.isFinite(v)
      )
        probs.push(v);
    }
    if (!probs.length) return null;
    return probs.reduce((a, b) => a + b, 0) / probs.length;
  }, [pred]);

  const rec = useMemo(() => toRecommendation(meanProbUp), [meanProbUp]);

  const predRows = useMemo(() => {
    if (!pred) return [];
    const horizons: string[] = [];
    for (const k of Object.keys(pred)) {
      if (k.startsWith("pred_ret_")) horizons.push(k.replace("pred_ret_", ""));
    }
    horizons.sort((a, b) => Number(a) - Number(b));

    return horizons.map((h) => {
      const ret = pred[`pred_ret_${h}`];
      const dir = pred[`pred_dir_${h}`];
      const prob = pred[`prob_up_${h}`];
      const thr = pred[`threshold_${h}`];

      const returnSign =
        typeof ret === "number" && Number.isFinite(ret)
          ? ret > 0
            ? "UP"
            : "DOWN"
          : "—";

      const signal =
        typeof dir === "number" && Number.isFinite(dir)
          ? dir === 1
            ? "BUY"
            : "SELL"
          : "—";

      return { h, ret, returnSign, signal, prob, thr };
    });
  }, [pred]);

  useEffect(() => {
    const el = chartDivRef.current;
    if (!el) return;
    if (chartRef.current) return;

    const getSize = () => ({
      w: Math.max(300, el.clientWidth || 0),
      h: Math.max(260, el.clientHeight || 0),
    });

    const { w, h } = getSize();

    const chart = createChart(el, {
      width: w,
      height: h,
      layout: {
        textColor: "#e5e7eb",
        background: { type: "solid", color: "#0b1220" },
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.06)" },
        horzLines: { color: "rgba(255,255,255,0.06)" },
      },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.12)" },
      timeScale: { borderColor: "rgba(255,255,255,0.12)" },
      crosshair: { mode: 0 },
    });

    const series = chart.addSeries(CandlestickSeries);

    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      const s = getSize();
      chart.applyOptions({ width: s.w, height: s.h });
    });
    ro.observe(el);
    roRef.current = ro;

    return () => {
      ro.disconnect();
      roRef.current = null;
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

    const data: CandlestickData[] = candles.map((c) => ({
      time: c.time as UTCTimestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    series.setData(data);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  useEffect(() => {
    loadCandles();
  }, []);

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">DSS Trading Simulator</div>
          <div className="subtitle">Backend: {API}</div>
        </div>

        <div className={`badge badge-${rec.replace(" ", "-").toLowerCase()}`}>
          {rec}
          <span className="badge-sub">
            {meanProbUp == null ? "" : ` (mean p(up)=${meanProbUp.toFixed(3)})`}
          </span>
        </div>
      </header>

      {err && <div className="error">Error: {err}</div>}

      <section className="cards">
        <div className="card">
          <div className="cardLabel">Timestamp</div>
          <div className="cardValue">{pred?.timestamp ?? "—"}</div>
        </div>
        <div className="card">
          <div className="cardLabel">Close</div>
          <div className="cardValue">{fmtNum(pred?.Close, 2)}</div>
        </div>
        <div className="card">
          <div className="cardLabel">RSI (14)</div>
          <div className="cardValue">{fmtNum(pred?.rsi14, 2)}</div>
        </div>
        <div className="card">
          <div className="cardLabel">ADX (14)</div>
          <div className="cardValue">{fmtNum(pred?.adx14, 2)}</div>
        </div>
        <div className="card">
          <div className="cardLabel">MACD Hist</div>
          <div className="cardValue">{fmtNum(pred?.macd_hist, 4)}</div>
        </div>
        <div className="card">
          <div className="cardLabel">Signal score</div>
          <div className="cardValue">{fmtNum(pred?.signal_score, 4)}</div>
        </div>
      </section>

      <div className="mainGrid">
        <section className="panel chartPanel">
          <div className="panelHeader">
            <div className="panelTitle">Candles</div>
            <div className="panelMeta">
              {loadingCandles ? "Loading…" : `Loaded: ${candles.length}`}
            </div>
          </div>
          <div className="chartWrap" ref={chartDivRef} />
        </section>

        <section className="panel actionsPanel">
          <div className="panelHeader">
            <div className="panelTitle">Actions</div>
            <div className="panelMeta">
              Predict uses current feed.csv. Next candle appends 1 row and
              clears predictions.
            </div>
          </div>

          <div className="actions">
            <button
              className="btn"
              onClick={makePrediction}
              disabled={loadingPred || loadingNext}
            >
              {loadingPred ? "Predicting…" : "Make predictions"}
            </button>

            <button
              className="btn btnSecondary"
              onClick={nextCandle}
              disabled={loadingNext || loadingPred}
            >
              {loadingNext ? "Adding…" : "Next candle"}
            </button>

            <button
              className="btn btnGhost"
              onClick={loadCandles}
              disabled={loadingCandles || loadingNext}
            >
              {loadingCandles ? "Refreshing…" : "Refresh candles"}
            </button>
          </div>

          <div className="predBox">
            <div className="predTitle">Predicted returns</div>

            {!pred && (
              <div className="muted">
                No predictions yet. Click “Make predictions”.
              </div>
            )}

            {pred && (
              <div className="tableWrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th>Pred return</th>
                      <th>Return sign</th>
                      <th>Signal (p(up) ≥ threshold)</th>
                      <th>p(up)</th>
                      <th>Threshold</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predRows.map((r) => (
                      <tr key={r.h}>
                        <td>{r.h}</td>
                        <td>{fmtPct(r.ret, 2)}</td>
                        <td>{r.returnSign}</td>
                        <td>{r.signal}</td>
                        <td>{fmtNum(r.prob, 3)}</td>
                        <td>{fmtNum(r.thr, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

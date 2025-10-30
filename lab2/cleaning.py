import numpy as np, pandas as pd

INPUT_PATH = "kaggle_nvda_processed.csv"
OUTPUT_PATH = "NVDA_CLEAN_15M.csv"

BAR_MINUTES = 15
RET_MAX_ABS = 0.35
RET_MAD_Z = 9.0
ROLL_BARS_RET = 200
VOL_MAD_Z = 9.0
ROLL_BARS_VOL = 500
MIN_WINDOW = 100
REQUIRE_POS_VOL = True
KEEP_ONLY_RTH = True
UNIFORMIZE_GRID = True
DROP_BAD_DAYS = True
MAX_MISS_PER_DAY = 4


def pct(s, p):
    s = s.dropna()
    return float(np.nanpercentile(s, p)) if len(s) else np.nan


def madz(x, w, mn):
    med = x.rolling(w, min_periods=mn).median()
    mad = (x - med).abs().rolling(w, min_periods=mn).median()
    return (x - med) / (1.4826 * mad.replace(0, np.nan))


def record(steps, name, before, after, notes=None):
    steps.append((name, before, after, before - after, notes or {}))


def fmt_report(meta, steps, stats):
    lines = ["[Meta]"]
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n[Steps]")
    lines.append(f"{'Step':28} {'Before':>12} {'After':>12}  Dropped  Notes")
    for n, b, a, d, nt in steps:
        note = "; ".join(f"{k}={v}" for k, v in nt.items()) if nt else ""
        lines.append(f"{n:28} {b:>12,} {a:>12,}  {d:>7,}  {note}")
    lines.append("\n[Post-clean stats]")
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def read_clean(path):
    df = pd.read_csv(path)
    need = {"datetime", "Open", "High", "Low", "Close", "Volume"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"Missing columns: {miss}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(
        subset=list(need)).sort_values("datetime").drop_duplicates(
            subset=["datetime"]))


def filter_rth(df):
    if not KEEP_ONLY_RTH: return df
    wd = df["datetime"].dt.weekday < 5
    from datetime import time
    rth = (df["datetime"].dt.time >= time(9, 30)) & (df["datetime"].dt.time
                                                     <= time(16, 0))
    return df[wd & rth].copy()


def enforce_candles(df):
    pos = (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    vol = (df["Volume"] > 0) if REQUIRE_POS_VOL else (df["Volume"] >= 0)
    body = (df["High"] >= df[["Open", "Close"]].max(
        axis=1)) & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
    shape = (df["High"] >= df["Low"])
    return df[pos & vol & body & shape].copy(), {
        "neg_price":
        int((~(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)).sum()),
        "nonpos_vol":
        int((~(df["Volume"] > 0)
             if REQUIRE_POS_VOL else ~(df["Volume"] >= 0)).sum()),
        "high<low":
        int((df["High"] < df["Low"]).sum()),
        "body_outside":
        int((~body).sum())
    }


def uniform_grid(df):
    if not UNIFORMIZE_GRID: return df
    dates = sorted(df["datetime"].dt.date.unique())
    if not dates: return df
    opens = pd.to_datetime([f"{d} 09:30:00" for d in dates])
    closes = pd.to_datetime([f"{d} 16:00:00" for d in dates])
    idx = []
    for o, c in zip(opens, closes):
        if o.weekday() < 5:
            idx.append(pd.date_range(o, c, freq=f"{BAR_MINUTES}min"))
    if not idx: return df.iloc[0:0].copy()
    grid = pd.DatetimeIndex(np.concatenate(idx)).unique().sort_values()
    left = df.set_index("datetime")
    left = left[~left.index.duplicated(keep="first")]
    out = left.reindex(grid)
    out.index.name = "datetime"
    return out.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]).reset_index()


def add_session_cols(df):
    df = df.copy()
    minutes = df["datetime"].diff().dt.total_seconds().div(60)
    df["new_session"] = minutes.isna() | (minutes > BAR_MINUTES)
    df["ret"] = df["Close"].pct_change().where((minutes == BAR_MINUTES)
                                               & (~df["new_session"]))
    return df


def drop_ret_outliers(df):
    df = df.copy()
    hard_ok = df["ret"].abs() <= RET_MAX_ABS
    hard_ok |= df["ret"].isna()
    z = madz(df["ret"].abs(), ROLL_BARS_RET, MIN_WINDOW)
    keep = hard_ok & ((z <= RET_MAD_Z) | z.isna())
    return df[keep].copy()


def drop_vol_outliers(df):
    df = df.copy()
    lv = np.log1p(df["Volume"].clip(lower=0))
    z = madz(lv, ROLL_BARS_VOL, MIN_WINDOW)
    return df[(z.abs() <= VOL_MAD_Z) | z.isna()].copy()


def drop_bad_days(df):
    if not DROP_BAD_DAYS: return df
    expected = int(round(390 / BAR_MINUTES))  # 26
    cnt = df.groupby(df["datetime"].dt.date)["datetime"].count()
    bad = cnt[cnt < (expected - MAX_MISS_PER_DAY)].index
    return df[~df["datetime"].dt.date.isin(bad)].copy(), {
        "days_before":
        int(cnt.index.nunique()),
        "days_after":
        int(df[~df["datetime"].dt.date.isin(bad)]
            ["datetime"].dt.date.nunique()),
        "dropped_days":
        int(len(bad)),
        "avg_bars_on_dropped_days":
        float(cnt.loc[bad].mean()) if len(bad) else None,
        "expected_bars_per_day":
        expected
    }


def final_qc(df):
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
    return df


def post_stats(df):
    tmp = add_session_cols(df.copy())
    mins = df["datetime"].diff().dt.total_seconds().div(60)
    intra = mins.where(~tmp["new_session"]).dropna()
    r = tmp["ret"].dropna()
    lv = np.log1p(df["Volume"].clip(lower=0))
    return {
        "rows":
        f"{len(df):,}",
        "date_range":
        f"{df['datetime'].min()} → {df['datetime'].max()}",
        "unique_days":
        int(df["datetime"].dt.date.nunique()),
        "bars_expected_per_day(15m_RTH)":
        int(round(390 / 15)),
        "median_gap_minutes_intra":
        float(np.nanmedian(intra)) if len(intra) else np.nan,
        "pct_gaps_eq_15m_intra":
        float((intra == 15).mean() * 100) if len(intra) else np.nan,
        "returns.count":
        int(r.count()),
        "returns.mean":
        float(r.mean()),
        "returns.std":
        float(r.std()),
        "returns.abs.p95":
        pct(r.abs(), 95),
        "returns.abs.p99":
        pct(r.abs(), 99),
        "volume.mean":
        float(df["Volume"].mean()),
        "volume.median":
        float(df["Volume"].median()),
        "log_volume.std":
        float(lv.std()),
        "log_volume.p95":
        pct(lv, 95),
    }


if __name__ == "__main__":
    steps = []
    df = read_clean(INPUT_PATH)
    meta = dict(input_file=INPUT_PATH,
                rows_loaded=len(df),
                start=str(df["datetime"].min()),
                end=str(df["datetime"].max()),
                config=dict(BAR_MINUTES=BAR_MINUTES,
                            RET_MAX_ABS=RET_MAX_ABS,
                            RET_MAD_Z=RET_MAD_Z,
                            VOL_MAD_Z=VOL_MAD_Z,
                            ROLL_BARS_RET=ROLL_BARS_RET,
                            ROLL_BARS_VOL=ROLL_BARS_VOL,
                            MIN_WINDOW=MIN_WINDOW,
                            KEEP_ONLY_RTH=KEEP_ONLY_RTH,
                            UNIFORMIZE_GRID=UNIFORMIZE_GRID,
                            DROP_BAD_DAYS=DROP_BAD_DAYS,
                            MAX_MISS_PER_DAY=MAX_MISS_PER_DAY,
                            REQUIRE_POS_VOL=REQUIRE_POS_VOL))
    b = len(df)
    df = filter_rth(df)
    record(steps, "filter_market_hours", b, len(df))

    b = len(df)
    df2, notes = enforce_candles(df)
    df = df2
    record(steps, "enforce_candle_validity", b, len(df), notes)

    b = len(df)
    df = uniform_grid(df)
    record(steps, "uniformize_grid_rth", b, len(df))

    b = len(df)
    df = add_session_cols(df)
    record(steps, "add_session_keys", b, len(df))
    b = len(df)
    df = add_session_cols(df)
    record(steps, "time_aware_returns", b, len(df))

    def hard_note(before_df, after_df):
        x = before_df.dropna(subset=["ret"])
        return {
            "hard_guard_exceeded": int((x["ret"].abs() > RET_MAX_ABS).sum())
        }

    b = len(df)
    df = drop_ret_outliers(df)
    record(steps, "drop_price_outliers", b, len(df), hard_note(df, df))

    b = len(df)
    df = drop_vol_outliers(df)
    record(steps, "drop_volume_outliers", b, len(df))

    b = len(df)
    df2, notes = drop_bad_days(df)
    df = df2
    record(steps, "drop_days_with_many_missing", b, len(df), notes)

    b = len(df)
    df = df.drop(columns=["new_session", "ret"], errors="ignore")
    df = final_qc(df)
    record(steps, "final_qc", b, len(df))

    df.to_csv(OUTPUT_PATH, index=False)
    stats = post_stats(df)
    report_txt = fmt_report(meta, steps, stats)
    base = OUTPUT_PATH.rsplit(".", 1)[0]
    with open(base + "_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"Saved cleaned dataset: {OUTPUT_PATH} ({len(df):,} rows)")
    print(f"Report: {base}_report.txt")

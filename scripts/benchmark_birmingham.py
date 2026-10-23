"""Benchmark the forecasting approach on REAL public data (UCI "Parking Birmingham").

1. Download dataset.csv from https://archive.ics.uci.edu/dataset/482/parking+birmingham
   and save it as data/birmingham.csv
2. Run:  python scripts/benchmark_birmingham.py --csv data/birmingham.csv

It predicts occupancy 60 minutes ahead for each car park and compares with two baselines:
"nothing changes" and "same time yesterday". Use the printed numbers on your Feasibility slide.
The dataset has readings roughly every 30 min between 08:00 and 16:30 (Oct-Dec 2016).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["ts"] = pd.to_datetime(df["LastUpdated"])
    df["rate"] = (df["Occupancy"] / df["Capacity"]).clip(0, 1)
    df["slot"] = df["ts"].dt.round("30min")
    return df.groupby(["SystemCodeNumber", "slot"], as_index=False).agg(rate=("rate", "mean"), cap=("Capacity", "first"))


def features(g):
    g = g.sort_values("slot").set_index("slot")
    r = g["rate"]
    f = pd.DataFrame({"rate_now": r})
    # a lag/target only counts if it is exactly 30/60 min (or 1 day) away, so overnight gaps are skipped
    for name, mins in (("lag30", 30), ("lag60", 60), ("lag1d", 1440)):
        f[name] = r.reindex(r.index - pd.Timedelta(minutes=mins)).values
    f["target"] = r.reindex(r.index + pd.Timedelta(minutes=60)).values
    f["hour"] = f.index.hour + f.index.minute / 60
    f["dow"] = f.index.dayofweek
    return f.dropna(subset=["target", "lag30", "lag60"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/birmingham.csv")
    args = ap.parse_args()
    d = load(args.csv)
    frames = []
    for i, (code, g) in enumerate(d.groupby("SystemCodeNumber")):
        f = features(g)
        f["park"] = i
        frames.append(f)
    all_ = pd.concat(frames)
    cut = all_.index.max() - pd.Timedelta(days=14)
    tr, te = all_[all_.index <= cut], all_[all_.index > cut]
    cols = ["rate_now", "lag30", "lag60", "hour", "dow", "park"]
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=0,
                                      categorical_features=[cols.index("park")]).fit(tr[cols], tr["target"])
    pred = np.clip(m.predict(te[cols]), 0, 1)
    mae = np.abs(pred - te["target"]).mean()
    naive = np.abs(te["rate_now"] - te["target"]).mean()
    day = te.dropna(subset=["lag1d"])
    yday = np.abs(day["lag1d"].values - day["target"].values).mean() if len(day) else float("nan")
    print(f"Car parks: {all_['park'].nunique()}   train rows: {len(tr)}   test rows: {len(te)} (last 14 days)")
    print(f"Model MAE (occupancy rate, +60 min):       {mae*100:.2f} percentage points")
    print(f"Baseline 'nothing changes':                 {naive*100:.2f} percentage points")
    print(f"Baseline 'same time yesterday' (approx.):   {yday*100:.2f} percentage points")
    print(f"Improvement over 'nothing changes':         {100*(1-mae/naive):.1f}%")


if __name__ == "__main__":
    main()

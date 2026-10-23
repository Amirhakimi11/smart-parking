"""Next-hour occupancy forecaster (scikit-learn HistGradientBoosting).

One model predicts occupancy rate at +15/+30/+45/+60 minutes for every zone.
"""
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import MODEL_PATH
from .simulator import FIXED_HOLIDAYS

HORIZONS = [15, 30, 45, 60]
FEATURES = ["horizon", "hour_sin", "hour_cos", "dow", "is_weekend", "is_holiday",
            "is_payday", "rate_now", "rate_15", "rate_30", "rate_60", "trend", "zone_idx"]


def calendar_feats(ts: pd.DatetimeIndex) -> pd.DataFrame:
    hour = ts.hour + ts.minute / 60
    dow = ts.dayofweek
    hol = np.array([(m, d) in FIXED_HOLIDAYS for m, d in zip(ts.month, ts.day)])
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": dow,
        "is_weekend": ((dow >= 5) | hol).astype(int),
        "is_holiday": hol.astype(int),
        "is_payday": ((ts.day >= 25) | (ts.day <= 2)).astype(int),
    })


def build_frame(readings: pd.DataFrame, caps: dict, zone_order: list, horizons=HORIZONS) -> pd.DataFrame:
    """readings: columns zone_id, ts, occupied  ->  supervised learning table."""
    out = []
    for zone, g in readings.groupby("zone_id"):
        s = g.set_index("ts")["occupied"].sort_index()
        s = s[~s.index.duplicated()].resample("15min").mean().interpolate(limit=4)
        rate = (s / caps[zone]).clip(0, 1)
        for h in horizons:
            f = pd.DataFrame({"rate_now": rate, "rate_15": rate.shift(1),
                              "rate_30": rate.shift(2), "rate_60": rate.shift(4)})
            f["target"] = rate.shift(-(h // 15))
            cal = calendar_feats(f.index + pd.Timedelta(minutes=h))
            cal.index = f.index
            f = f.join(cal)
            f["horizon"] = h
            f["trend"] = f["rate_now"] - f["rate_30"]
            f["zone_idx"] = zone_order.index(zone)
            f["zone_id"] = zone
            f["ts"] = f.index
            out.append(f.dropna())
    return pd.concat(out, ignore_index=True)


def _model():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=6, random_state=0,
        categorical_features=[FEATURES.index("zone_idx")])


def train(readings: pd.DataFrame, caps: dict, test_days: int = 10, save: bool = True) -> dict:
    zone_order = sorted(caps)
    df = build_frame(readings, caps, zone_order)
    cutoff = df["ts"].max() - pd.Timedelta(days=test_days)
    tr, te = df[df["ts"] <= cutoff], df[df["ts"] > cutoff]
    model = _model().fit(tr[FEATURES], tr["target"])
    pred = np.clip(model.predict(te[FEATURES]), 0, 1)
    cap_arr = te["zone_id"].map(caps).values
    err = np.abs(pred - te["target"].values)
    base_err = np.abs(te["rate_now"].values - te["target"].values)   # "nothing changes" baseline
    m60 = (te["horizon"] == 60).values
    metrics = {
        "train_rows": int(len(tr)), "test_rows": int(len(te)), "test_days": test_days,
        "mae_rate": float(err.mean()),
        "mae_bays": float((err * cap_arr).mean()),
        "baseline_mae_rate": float(base_err.mean()),
        "baseline_mae_bays": float((base_err * cap_arr).mean()),
        "mae_rate_60min": float(err[m60].mean()),
        "baseline_mae_rate_60min": float(base_err[m60].mean()),
        "mae_bays_60min": float((err[m60] * cap_arr[m60]).mean()),
        "model": "HistGradientBoostingRegressor",
    }
    metrics["improvement_vs_baseline_pct"] = round(
        100 * (1 - metrics["mae_rate"] / max(metrics["baseline_mae_rate"], 1e-9)), 1)
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": model, "zone_order": zone_order, "metrics": metrics}, MODEL_PATH)
    return metrics


class Forecaster:
    def __init__(self):
        self.bundle = None
        self.load()

    def load(self):
        if MODEL_PATH.exists():
            self.bundle = joblib.load(MODEL_PATH)
        return self.bundle is not None

    @property
    def ready(self):
        return self.bundle is not None

    @property
    def metrics(self):
        return self.bundle["metrics"] if self.bundle else {}

    def predict(self, now, states: dict) -> dict:
        """states: zone_id -> dict(rate_now, rate_15, rate_30, rate_60).
        returns zone_id -> {horizon: predicted_rate}"""
        order = self.bundle["zone_order"]
        rows, keys = [], []
        for zid, st in states.items():
            if zid not in order:
                continue
            for h in HORIZONS:
                t = pd.DatetimeIndex([pd.Timestamp(now + timedelta(minutes=h))])
                cal = calendar_feats(t).iloc[0].to_dict()
                rows.append({**cal, **st, "horizon": h, "trend": st["rate_now"] - st["rate_30"],
                             "zone_idx": order.index(zid)})
                keys.append((zid, h))
        if not rows:
            return {}
        X = pd.DataFrame(rows)[FEATURES]
        y = np.clip(self.bundle["model"].predict(X), 0, 1)
        out = {}
        for (zid, h), v in zip(keys, y):
            out.setdefault(zid, {})[h] = float(v)
        return out

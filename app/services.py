"""Query helpers that turn raw readings into dashboard-ready JSON."""
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from .db import Prediction, Reading, Zone
from .predictor import HORIZONS

LEVEL_LABEL = {"available": "Available", "busy": "Busy", "almost_full": "Almost full",
               "full": "Full", "closed": "Closed"}


def level_for(rate: float, is_open: bool = True) -> str:
    if not is_open:
        return "closed"
    if rate < 0.60:
        return "available"
    if rate < 0.85:
        return "busy"
    if rate < 0.97:
        return "almost_full"
    return "full"


def load_recent(s, now, minutes=95):
    rows = s.execute(select(Reading.zone_id, Reading.ts, Reading.occupied)
                     .where(Reading.ts >= now - timedelta(minutes=minutes))
                     .order_by(Reading.ts)).all()
    by = {}
    for z, t, o in rows:
        by.setdefault(z, []).append((t, o))
    return by


def _nearest(series, t, tol_min=20):
    best = None
    for ts, v in series:
        d = abs((ts - t).total_seconds())
        if best is None or d < best[0]:
            best = (d, v)
    return best[1] if best and best[0] <= tol_min * 60 else None


def zone_states(zones, recent):
    states, latest = {}, {}
    for z in zones:
        ser = recent.get(z.id)
        if not ser:
            continue
        cap = max(z.capacity, 1)
        t_now, o_now = ser[-1]

        def r(minutes):
            v = _nearest(ser, t_now - timedelta(minutes=minutes))
            return min(1.0, (v if v is not None else o_now) / cap)

        states[z.id] = {"rate_now": min(1.0, o_now / cap), "rate_15": r(15), "rate_30": r(30), "rate_60": r(60)}
        latest[z.id] = (t_now, o_now)
    return states, latest


def snapshot(s, forecaster, now, vision_status=None, scenario=None):
    zones = s.query(Zone).order_by(Zone.sort).all()
    recent = load_recent(s, now)
    states, latest = zone_states(zones, recent)
    preds = forecaster.predict(now, states) if forecaster.ready and states else {}

    floors, order = {}, []
    for z in zones:
        occ = latest.get(z.id, (None, 0))[1]
        occ = min(occ, z.capacity)
        avail = max(z.capacity - occ, 0) if z.is_open else 0
        rate = occ / max(z.capacity, 1)
        fc = {}
        for h in HORIZONS:
            p = preds.get(z.id, {}).get(h, rate)
            po = int(round(p * z.capacity))
            fc[str(h)] = {"occupied": po, "available": max(z.capacity - po, 0) if z.is_open else 0, "rate": round(p, 3)}
        zd = {"id": z.id, "name": z.name, "floor": z.floor, "capacity": z.capacity, "occupied": occ,
              "available": avail, "rate": round(rate, 3), "is_open": z.is_open, "source": z.source,
              "level": level_for(rate, z.is_open), "forecast": fc,
              "level_60": level_for(fc["60"]["rate"], z.is_open)}
        if z.floor not in floors:
            floors[z.floor] = {"floor": z.floor, "name": z.floor_name, "zones": []}
            order.append(z.floor)
        floors[z.floor]["zones"].append(zd)

    def roll(zs):
        cap = sum(z["capacity"] for z in zs)
        occ = sum(z["occupied"] for z in zs)
        rate = occ / max(cap, 1)
        fc = {}
        for h in HORIZONS:
            po = sum(z["forecast"][str(h)]["occupied"] for z in zs)
            av = sum(z["forecast"][str(h)]["available"] for z in zs)
            fc[str(h)] = {"occupied": po, "available": av, "rate": round(po / max(cap, 1), 3)}
        all_closed = all(not z["is_open"] for z in zs)
        return {"capacity": cap, "occupied": occ, "available": sum(z["available"] for z in zs),
                "rate": round(rate, 3), "level": level_for(rate, not all_closed),
                "forecast": fc, "level_60": level_for(fc["60"]["rate"], not all_closed)}

    out_floors = []
    for code in order:
        f = floors[code]
        f.update(roll(f["zones"]))
        out_floors.append(f)
    all_zones = [z for f in out_floors for z in f["zones"]]
    totals = roll(all_zones) if all_zones else {}

    rec = None
    cands = [f for f in out_floors if f["forecast"]["30"]["available"] > 0]
    if cands:
        best = max(cands, key=lambda f: f["forecast"]["30"]["available"])
        rec = {"floor": best["floor"], "name": best["name"], "available_now": best["available"],
               "available_30": best["forecast"]["30"]["available"]}
    return {"updated_at": now.isoformat(timespec="seconds"), "totals": totals, "floors": out_floors,
            "recommendation": rec, "vision": vision_status, "scenario": scenario,
            "model_ready": forecaster.ready}


def history_series(s, now, scope="ALL", hours=6):
    zones = s.query(Zone).all()
    if scope == "ALL":
        ids = [z.id for z in zones]
    else:
        ids = [z.id for z in zones if z.floor == scope or z.id == scope]
    if not ids:
        return {"capacity": 0, "points": []}
    cap = sum(z.capacity for z in zones if z.id in ids)
    rows = s.execute(select(Reading.zone_id, Reading.ts, Reading.occupied)
                     .where(Reading.ts >= now - timedelta(hours=hours), Reading.zone_id.in_(ids))).all()
    if not rows:
        return {"capacity": cap, "points": []}
    df = pd.DataFrame(rows, columns=["zone_id", "ts", "occupied"])
    df["bucket"] = pd.to_datetime(df["ts"]).dt.floor("15min")
    g = df.groupby(["bucket", "zone_id"])["occupied"].mean().unstack().ffill().bfill()
    tot = g.sum(axis=1)
    return {"capacity": cap, "points": [{"t": t.isoformat(), "occupied": int(round(v))} for t, v in tot.items()]}


def accuracy(s, now, hours=6):
    zones = {z.id: z for z in s.query(Zone).all()}
    preds = s.execute(select(Prediction.zone_id, Prediction.target_ts, Prediction.predicted_occupied)
                      .where(Prediction.horizon_min == 60, Prediction.target_ts <= now,
                             Prediction.target_ts >= now - timedelta(hours=hours))).all()
    if not preds:
        return {"n": 0, "points": []}
    rd = s.execute(select(Reading.zone_id, Reading.ts, Reading.occupied)
                   .where(Reading.ts >= now - timedelta(hours=hours, minutes=10))).all()
    rdf = pd.DataFrame(rd, columns=["zone_id", "ts", "occupied"]).sort_values("ts")
    by = {z: g.reset_index(drop=True) for z, g in rdf.groupby("zone_id")}
    recs = []
    for zid, tgt, p in preds:
        g = by.get(zid)
        if g is None or g.empty:
            continue
        i = (g["ts"] - tgt).abs().idxmin()
        if abs((g.loc[i, "ts"] - tgt).total_seconds()) > 300:
            continue
        recs.append((zid, pd.Timestamp(tgt).floor("15min"), float(p), float(g.loc[i, "occupied"])))
    if not recs:
        return {"n": 0, "points": []}
    df = pd.DataFrame(recs, columns=["zone_id", "t", "pred", "actual"])
    df["abs_err"] = (df["pred"] - df["actual"]).abs()
    df["cap"] = df["zone_id"].map(lambda z: zones[z].capacity)
    agg = df.groupby("t")[["pred", "actual"]].sum().reset_index()
    return {"n": int(len(df)),
            "mae_bays": round(float(df["abs_err"].mean()), 2),
            "mae_pct": round(float((df["abs_err"] / df["cap"]).mean() * 100), 2),
            "points": [{"t": r.t.isoformat(), "pred": round(r.pred), "actual": round(r.actual)}
                       for r in agg.itertuples()]}

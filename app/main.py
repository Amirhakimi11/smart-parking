import asyncio
import csv
import hashlib
import hmac
import io
import threading
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, insert, select

from . import predictor, services
from .config import (ADMIN_PASSWORD, CAMERA_ZONE, DATABASE_URL, SECRET_KEY, SEED_DAYS,
                     SITE_NAME, STATIC_DIR, now_local)
from .db import Prediction, Reading, SessionLocal, Zone, init_db
from .simulator import ZONES, Simulator, bulk_insert_readings, generate_history
from .vision import VisionService

sim = Simulator()
vision = VisionService(sim)
forecaster = predictor.Forecaster()


# ---------------------------------------------------------------- startup helpers
def seed_zones():
    with SessionLocal() as s:
        existing = {z.id: z for z in s.query(Zone).all()}
        for i, (zid, fl, fname, name, cap, _k) in enumerate(ZONES):
            if zid == CAMERA_ZONE:
                cap = vision.total_bays
            src = "yolov8" if (zid == CAMERA_ZONE and vision.mode == "yolo") else "simulated"
            if zid in existing:
                existing[zid].source = src
                if zid == CAMERA_ZONE:
                    existing[zid].capacity = cap
            else:
                s.add(Zone(id=zid, floor=fl, floor_name=fname, name=name, capacity=cap, source=src, sort=i))
        s.commit()


def seed_history_if_empty():
    with SessionLocal() as s:
        n = s.scalar(select(func.count(Reading.id)))
        caps = {z.id: z.capacity for z in s.query(Zone).all()}
    if n == 0:
        print(f"[startup] seeding {SEED_DAYS} days of synthetic history ...")
        rows = generate_history(caps, now_local(), SEED_DAYS)
        bulk_insert_readings(rows)
        print(f"[startup] inserted {len(rows)} readings")


def load_all_readings():
    with SessionLocal() as s:
        rows = s.execute(select(Reading.zone_id, Reading.ts, Reading.occupied)).all()
        caps = {z.id: z.capacity for z in s.query(Zone).all()}
    return pd.DataFrame(rows, columns=["zone_id", "ts", "occupied"]), caps


def train_if_needed():
    if forecaster.ready:
        return
    print("[startup] training forecasting model ...")
    df, caps = load_all_readings()
    m = predictor.train(df, caps)
    forecaster.load()
    print("[startup] model MAE (bays):", round(m["mae_bays"], 2), "| baseline:", round(m["baseline_mae_bays"], 2))


def backfill_predictions():
    """Create the last few hours of +60 min predictions so accuracy charts are not empty."""
    with SessionLocal() as s:
        if s.scalar(select(func.count(Prediction.id))):
            return
        now = now_local()
        rows = s.execute(select(Reading.zone_id, Reading.ts, Reading.occupied)
                         .where(Reading.ts >= now - timedelta(hours=8))).all()
        caps = {z.id: z.capacity for z in s.query(Zone).all()}
        if not rows:
            return
        df = pd.DataFrame(rows, columns=["zone_id", "ts", "occupied"])
        fr = predictor.build_frame(df, caps, forecaster.bundle["zone_order"], horizons=[60])
        fr = fr[(fr["ts"] >= pd.Timestamp(now - timedelta(hours=5))) & (fr["ts"] <= pd.Timestamp(now - timedelta(minutes=60)))]
        if fr.empty:
            return
        y = forecaster.bundle["model"].predict(fr[predictor.FEATURES]).clip(0, 1)
        s.execute(insert(Prediction), [
            {"zone_id": r.zone_id, "made_at": r.ts.to_pydatetime(), "target_ts": (r.ts + pd.Timedelta(minutes=60)).to_pydatetime(),
             "horizon_min": 60, "predicted_occupied": float(p * caps[r.zone_id])}
            for r, p in zip(fr.itertuples(), y)])
        s.commit()


def log_predictions():
    now = now_local()
    with SessionLocal() as s:
        zones = s.query(Zone).all()
        states, _ = services.zone_states(zones, services.load_recent(s, now))
        preds = forecaster.predict(now, states)
        caps = {z.id: z.capacity for z in zones}
        rows = [{"zone_id": zid, "made_at": now, "target_ts": now + timedelta(minutes=60), "horizon_min": 60,
                 "predicted_occupied": float(v[60] * caps[zid])} for zid, v in preds.items()]
        if rows:
            s.execute(insert(Prediction), rows)
            s.commit()


def prediction_loop():
    while True:
        time.sleep(300)
        try:
            if forecaster.ready:
                log_predictions()
        except Exception as exc:
            print("[predictions] error:", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_zones()
    seed_history_if_empty()
    train_if_needed()
    sim.init_state()
    if vision.mode == "yolo":
        sim.skip_zones.add(CAMERA_ZONE)
    sim.start()
    vision.start()
    backfill_predictions()
    threading.Thread(target=prediction_loop, daemon=True, name="pred-log").start()
    print("[startup] ready - vision mode:", vision.mode)
    yield
    sim.running = False
    vision.running = False


app = FastAPI(title="AI Smart Parking & Congestion Forecaster", lifespan=lifespan)


# ---------------------------------------------------------------- auth (admin)
def _sign(msg: str) -> str:
    return hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    exp = str(int(time.time()) + 12 * 3600)
    return f"{exp}.{_sign(exp)}"


def require_admin(authorization: Optional[str] = Header(None)):
    token = (authorization or "").replace("Bearer ", "")
    try:
        exp, sig = token.split(".")
        if int(exp) > time.time() and hmac.compare_digest(sig, _sign(exp)):
            return True
    except Exception:
        pass
    raise HTTPException(status_code=401, detail="Admin login required")


class Login(BaseModel):
    password: str


class ZonePatch(BaseModel):
    capacity: Optional[int] = None
    is_open: Optional[bool] = None


class Scenario(BaseModel):
    name: str
    minutes: int = 10


# ---------------------------------------------------------------- public API
@app.get("/api/health")
def health():
    return {"ok": True, "site": SITE_NAME, "db": "postgresql" if DATABASE_URL.startswith("postgresql") else "sqlite",
            "vision": vision.status, "model_ready": forecaster.ready}


@app.get("/api/overview")
def overview():
    with SessionLocal() as s:
        data = services.snapshot(s, forecaster, now_local(), dict(vision.status), sim.scenario)
    data["site"] = SITE_NAME
    return data


@app.get("/api/history")
def history(scope: str = "ALL", hours: int = 6):
    with SessionLocal() as s:
        return services.history_series(s, now_local(), scope, min(max(hours, 1), 48))


@app.get("/api/stream")
async def stream():
    async def gen():
        while True:
            f = vision.frame()
            if f:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")
            await asyncio.sleep(0.15)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot.jpg")
def snapshot_jpg():
    f = vision.frame()
    if not f:
        raise HTTPException(503, "No frame yet")
    return Response(f, media_type="image/jpeg")


# ---------------------------------------------------------------- admin API
@app.post("/api/admin/login")
def login(body: Login):
    if not hmac.compare_digest(body.password, ADMIN_PASSWORD):
        raise HTTPException(401, "Wrong password")
    return {"token": make_token()}


@app.get("/api/admin/summary", dependencies=[Depends(require_admin)])
def admin_summary():
    with SessionLocal() as s:
        data = services.snapshot(s, forecaster, now_local(), dict(vision.status), sim.scenario)
        data["readings_total"] = s.scalar(select(func.count(Reading.id)))
        data["accuracy"] = services.accuracy(s, now_local())
        zones = s.query(Zone).order_by(Zone.sort).all()
        data["zones_flat"] = [{"id": z.id, "name": z.name, "floor": z.floor_name, "capacity": z.capacity,
                               "is_open": z.is_open, "source": z.source} for z in zones]
    data["metrics"] = forecaster.metrics
    data["site"] = SITE_NAME
    data["db"] = "postgresql" if DATABASE_URL.startswith("postgresql") else "sqlite"
    alerts = []
    for f in data["floors"]:
        for z in f["zones"]:
            if z["is_open"] and z["level_60"] in ("almost_full", "full") and z["level"] not in ("almost_full", "full"):
                alerts.append({"zone": z["name"], "message": f"{z['name']} is forecast to be {z['level_60'].replace('_', ' ')} within 60 min"})
            elif z["is_open"] and z["level"] == "full":
                alerts.append({"zone": z["name"], "message": f"{z['name']} is full now"})
    data["alerts"] = alerts
    return data


@app.patch("/api/admin/zones/{zone_id}", dependencies=[Depends(require_admin)])
def patch_zone(zone_id: str, body: ZonePatch):
    with SessionLocal() as s:
        z = s.get(Zone, zone_id)
        if not z:
            raise HTTPException(404, "Zone not found")
        if body.capacity is not None:
            if body.capacity < 1 or body.capacity > 2000:
                raise HTTPException(400, "Capacity must be 1-2000")
            z.capacity = body.capacity
            sim.caps[zone_id] = body.capacity
        if body.is_open is not None:
            z.is_open = body.is_open
        s.commit()
    return {"ok": True}


@app.post("/api/admin/scenario", dependencies=[Depends(require_admin)])
def set_scenario(body: Scenario):
    try:
        sim.set_scenario(body.name, body.minutes)
    except ValueError:
        raise HTTPException(400, "Scenario must be normal, peak, event or quiet")
    return {"scenario": sim.scenario, "until": sim.scenario_until.isoformat() if sim.scenario_until else None}


@app.get("/api/admin/export.csv", dependencies=[Depends(require_admin)])
def export_csv(hours: int = 24):
    with SessionLocal() as s:
        rows = s.execute(select(Reading.ts, Reading.zone_id, Reading.occupied)
                         .where(Reading.ts >= now_local() - timedelta(hours=min(hours, 168)))
                         .order_by(Reading.ts).limit(100000)).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp", "zone_id", "occupied"])
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=parking_readings.csv"})


# ---------------------------------------------------------------- pages
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")

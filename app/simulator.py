"""Demand model + simulated sensors.

Everything produced here is SYNTHETIC data shaped like a Malaysian shopping mall
(lunch bump, evening peak, heavier weekends, payday and public-holiday boosts).
Replace it with real sensor / CCTV counts by writing rows into the `readings` table.
"""
import threading
import time
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import insert

from .config import CAMERA_ZONE, SIM_INTERVAL, now_local
from .db import Reading, SessionLocal, Zone

# id, floor code, floor name, zone name, capacity, "attractiveness" k (how early it fills)
ZONES = [
    ("RF-A", "RF", "Rooftop", "Rooftop A", 30, 0.62),
    ("RF-B", "RF", "Rooftop", "Rooftop B", 30, 0.55),
    ("L1-A", "L1", "Level 1", "Level 1 A", 80, 1.22),
    ("L1-B", "L1", "Level 1", "Level 1 B", 80, 1.12),
    ("B1-A", "B1", "Basement 1", "Basement 1 A", 110, 1.05),
    ("B1-B", "B1", "Basement 1", "Basement 1 B", 110, 0.98),
    ("B2-A", "B2", "Basement 2", "Basement 2 A", 90, 0.82),
    ("B2-B", "B2", "Basement 2", "Basement 2 B", 90, 0.72),
]
K = {z[0]: z[5] for z in ZONES}

_WEEKDAY = [(0, .02), (8, .05), (10, .20), (12, .52), (13.5, .62), (15, .42),
            (17, .55), (19, .74), (21, .56), (22.5, .10), (24, .02)]
_WEEKEND = [(0, .02), (8, .06), (10, .30), (12, .60), (14, .78), (16, .86),
            (18, .92), (20, .88), (21.5, .55), (22.5, .12), (24, .02)]
FIXED_HOLIDAYS = {(1, 1), (5, 1), (8, 31), (9, 16), (12, 25)}   # extend with Hari Raya, CNY, Deepavali...


def is_holiday(ts) -> bool:
    return (ts.month, ts.day) in FIXED_HOLIDAYS


def is_payday(ts) -> bool:
    return ts.day >= 25 or ts.day <= 2


def base_load(ts: datetime) -> float:
    """Overall mall load 0..~1.1 at a timestamp."""
    hour = ts.hour + ts.minute / 60
    weekend = ts.weekday() >= 5 or is_holiday(ts)
    pts = _WEEKEND if weekend else _WEEKDAY
    xs, ys = zip(*pts)
    load = float(np.interp(hour, xs, ys))
    if is_holiday(ts):
        load *= 1.10
    if is_payday(ts):
        load *= 1.08
    return load


def generate_history(capacities: dict, end: datetime, days: int, seed: int = 7):
    """Create `days` of 15-minute synthetic readings ending at `end`."""
    rng = np.random.default_rng(seed)
    end = end.replace(second=0, microsecond=0, minute=(end.minute // 15) * 15)
    stamps = [end - timedelta(minutes=15 * i) for i in range(days * 96)][::-1]
    base = np.array([base_load(t) for t in stamps])
    day_factor = {}
    for t in stamps:
        d = t.date()
        if d not in day_factor:
            day_factor[d] = rng.normal(1.0, 0.06)
    dayf = np.array([day_factor[t.date()] for t in stamps])
    rows = []
    for zid, cap in capacities.items():
        noise = np.zeros(len(stamps))
        for i in range(1, len(stamps)):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.025)
        rate = np.clip(base * K[zid] * dayf + noise, 0, 1)
        occ = np.rint(rate * cap).astype(int)
        rows += [{"zone_id": zid, "ts": t, "occupied": int(o)} for t, o in zip(stamps, occ)]
    return rows


def bulk_insert_readings(rows, chunk=4000):
    with SessionLocal() as s:
        for i in range(0, len(rows), chunk):
            s.execute(insert(Reading), rows[i:i + chunk])
        s.commit()


class Simulator:
    """Writes one new reading per zone every SIM_INTERVAL seconds."""

    SCENARIOS = {"normal": 1.0, "peak": 1.35, "event": 1.6, "quiet": 0.6}

    def __init__(self):
        self.rate = {}
        self.caps = {}
        self.scenario = "normal"
        self.scenario_until = None
        self.skip_zones = set()      # zones fed by a real source (YOLOv8)
        self.running = False
        self.rng = np.random.default_rng()

    def init_state(self):
        with SessionLocal() as s:
            for z in s.query(Zone).all():
                self.caps[z.id] = z.capacity
                self.rate[z.id] = min(1.0, base_load(now_local()) * K.get(z.id, 1.0))

    def set_scenario(self, name: str, minutes: int = 10):
        if name not in self.SCENARIOS:
            raise ValueError("unknown scenario")
        self.scenario = name
        self.scenario_until = None if name == "normal" else now_local() + timedelta(minutes=minutes)

    def surge(self) -> float:
        if self.scenario_until and now_local() > self.scenario_until:
            self.scenario, self.scenario_until = "normal", None
        return self.SCENARIOS[self.scenario]

    def current_occupied(self, zid: str) -> int:
        return int(round(self.rate.get(zid, 0) * self.caps.get(zid, 0)))

    def step(self):
        now = now_local()
        surge = self.surge()
        rows = []
        for zid, cap in self.caps.items():
            target = min(1.0, base_load(now) * K.get(zid, 1.0) * surge)
            r = self.rate[zid]
            r += (target - r) * 0.12 + self.rng.normal(0, 0.008)
            self.rate[zid] = float(np.clip(r, 0, 1))
            if zid not in self.skip_zones:
                rows.append({"zone_id": zid, "ts": now, "occupied": int(round(self.rate[zid] * cap))})
        if rows:
            bulk_insert_readings(rows)

    def start(self):
        self.running = True

        def loop():
            while self.running:
                try:
                    self.step()
                except Exception as exc:  # keep the thread alive
                    print("[simulator] error:", exc)
                time.sleep(SIM_INTERVAL)

        threading.Thread(target=loop, daemon=True, name="simulator").start()

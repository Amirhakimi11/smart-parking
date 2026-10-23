"""Live parking-bay detection with YOLOv8.

- If data/demo/parking.mp4 exists, every frame is analysed by YOLOv8 (vehicle classes)
  and each bay polygon in data/slots.json is marked free / occupied.
- If no video is present, a clearly-labelled SIMULATED top-down feed is drawn so the
  dashboard still works. (Nothing is faked as "detected" in that mode.)
"""
import json
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from sqlalchemy import insert

from .config import CAMERA_ZONE, SLOTS_PATH, VIDEO_PATH, YOLO_WEIGHTS, now_local
from .db import Reading, SessionLocal

VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
GREEN, RED, AMBER = (90, 190, 60), (60, 60, 230), (0, 176, 255)


def default_bays(w=960, h=540, per_row=15, rows=2):
    """Fallback bay layout (2 rows x 15) used for the simulated feed."""
    bays = []
    bw, bh = 52, 120
    x0 = (w - per_row * bw) // 2
    for r in range(rows):
        y0 = 90 if r == 0 else h - 90 - bh
        for i in range(per_row):
            x = x0 + i * bw
            bays.append([[x + 3, y0], [x + bw - 3, y0], [x + bw - 3, y0 + bh], [x + 3, y0 + bh]])
    return {"frame": [w, h], "bays": bays}


def load_slots():
    if SLOTS_PATH.exists():
        return json.loads(SLOTS_PATH.read_text())
    return default_bays()


class VisionService:
    def __init__(self, simulator):
        self.sim = simulator
        self.slots = load_slots()
        self.mode = "yolo" if VIDEO_PATH.exists() else "simulated"
        self.latest = None
        self.running = False
        self.status = {"mode": self.mode, "fps": 0.0, "occupied": 0, "total": len(self.slots["bays"]),
                       "model": YOLO_WEIGHTS if self.mode == "yolo" else None,
                       "updated_at": None, "error": None}
        self._lock = threading.Lock()
        self._perm = np.random.default_rng(3).permutation(len(self.slots["bays"]))

    @property
    def total_bays(self):
        return len(self.slots["bays"])

    # ---------- shared helpers ----------
    def _set(self, jpeg, occupied, fps):
        with self._lock:
            self.latest = jpeg
            self.status.update(occupied=int(occupied), fps=round(fps, 1),
                               updated_at=now_local().isoformat(timespec="seconds"))

    def frame(self):
        with self._lock:
            return self.latest

    def _write_reading(self, occupied):
        try:
            with SessionLocal() as s:
                s.execute(insert(Reading), [{"zone_id": CAMERA_ZONE, "ts": now_local(), "occupied": int(occupied)}])
                s.commit()
        except Exception as exc:
            print("[vision] db error:", exc)

    # ---------- simulated feed ----------
    def _run_sim(self):
        w, h = self.slots["frame"]
        bays = [np.array(b, np.int32) for b in self.slots["bays"]]
        rng = np.random.default_rng(11)
        colors = [(200, 200, 205), (70, 70, 190), (150, 90, 40), (60, 140, 90), (40, 40, 45), (170, 170, 60)]
        car_color = [colors[i] for i in rng.integers(0, len(colors), len(bays))]
        while self.running:
            t0 = time.time()
            n = min(self.sim.current_occupied(CAMERA_ZONE), len(bays))
            occupied_idx = set(self._perm[:n].tolist())
            img = np.full((h, w, 3), (62, 64, 68), np.uint8)
            for i, poly in enumerate(bays):
                cv2.polylines(img, [poly], True, (235, 235, 235), 1)
                if i in occupied_idx:
                    x, y, bw, bh = cv2.boundingRect(poly)
                    cv2.rectangle(img, (x + 8, y + 12), (x + bw - 8, y + bh - 12), car_color[i], -1)
                    cv2.rectangle(img, (x + 12, y + 30), (x + bw - 12, y + bh - 30), (30, 30, 30), 2)
            cv2.rectangle(img, (0, 0), (w, 34), (20, 20, 20), -1)
            cv2.putText(img, "SIMULATED FEED - add data/demo/parking.mp4 to run live YOLOv8", (12, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._set(buf.tobytes(), n, 4.0)
            time.sleep(max(0.0, 0.25 - (time.time() - t0)))

    # ---------- real YOLOv8 feed ----------
    def _run_yolo(self):
        try:
            from ultralytics import YOLO
            model = YOLO(YOLO_WEIGHTS)
        except Exception as exc:
            self.status["error"] = f"YOLO load failed: {exc}"
            print("[vision]", self.status["error"])
            self.mode = self.status["mode"] = "simulated"
            return self._run_sim()

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        skip = max(0, int(round(src_fps / 6)) - 1)      # analyse about 6 frames per second
        sw, sh = self.slots["frame"]
        bays_src = [np.array(b, np.float32) for b in self.slots["bays"]]
        last_db, fps_ema = 0.0, 0.0
        while self.running:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            for _ in range(skip):
                cap.grab()
            fh, fw = frame.shape[:2]
            sx, sy = fw / sw, fh / sh
            bays = [(b * [sx, sy]).astype(np.int32) for b in bays_src]

            res = model.predict(frame, imgsz=640, conf=0.25, classes=VEHICLE_CLASSES, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
            anchors = []
            for x1, y1, x2, y2 in boxes:
                anchors.append(((x1 + x2) / 2, (y1 + y2) / 2))
                anchors.append(((x1 + x2) / 2, y1 + 0.65 * (y2 - y1)))   # lower body point helps on angled cameras

            occupied = 0
            for poly in bays:
                busy = any(cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0 for px, py in anchors)
                occupied += busy
                cv2.polylines(frame, [poly], True, RED if busy else GREEN, 2)
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), AMBER, 1)
                cv2.putText(frame, f"{c:.2f}", (int(x1), max(12, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER, 1)
            cv2.rectangle(frame, (0, 0), (fw, 30), (20, 20, 20), -1)
            cv2.putText(frame, f"YOLOv8  occupied {occupied}/{len(bays)}  free {len(bays) - occupied}",
                        (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            dt = time.time() - t0
            fps_ema = 0.8 * fps_ema + 0.2 * (1 / max(dt, 1e-3))
            ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (min(fw, 960), int(fh * min(fw, 960) / fw))),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._set(buf.tobytes(), occupied, fps_ema)
            if time.time() - last_db > 10:
                self._write_reading(occupied)
                last_db = time.time()

    def start(self):
        self.running = True
        target = self._run_yolo if self.mode == "yolo" else self._run_sim
        threading.Thread(target=target, daemon=True, name="vision").start()

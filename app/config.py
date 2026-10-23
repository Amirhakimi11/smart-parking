import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

_db = os.getenv("DATABASE_URL", "").strip()
if not _db:
    (BASE / "data").mkdir(exist_ok=True)
    _db = f"sqlite:///{BASE / 'data' / 'parking.db'}"
elif _db.startswith("postgres://"):
    _db = _db.replace("postgres://", "postgresql+psycopg2://", 1)
elif _db.startswith("postgresql://"):
    _db = _db.replace("postgresql://", "postgresql+psycopg2://", 1)
DATABASE_URL = _db

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
SITE_NAME = os.getenv("SITE_NAME", "KL East Mall (demo)")
SIM_INTERVAL = int(os.getenv("SIM_INTERVAL", "15"))      # seconds between simulated readings
SEED_DAYS = int(os.getenv("SEED_DAYS", "60"))            # days of synthetic history to create
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
CAMERA_ZONE = "RF-A"                                     # the zone watched by the YOLOv8 camera

VIDEO_PATH = BASE / "data" / "demo" / "parking.mp4"
SLOTS_PATH = BASE / "data" / "slots.json"
MODEL_PATH = BASE / "models" / "occupancy_model.joblib"
STATIC_DIR = BASE / "static"

TZ = ZoneInfo("Asia/Kuala_Lumpur")


def now_local() -> datetime:
    """Current Kuala Lumpur time as a naive datetime (all DB timestamps use this)."""
    return datetime.now(TZ).replace(tzinfo=None)

"""Wipe all tables (zones, readings, predictions) and the saved model so the app re-seeds on next start.

Use it after you change data/demo/parking.mp4 or data/slots.json, because the camera zone's capacity changes.
Usage:  python scripts/reset_db.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import MODEL_PATH
from app.db import Base, engine


def main():
    if input("This deletes ALL parking data in the configured database. Type YES to continue: ").strip() != "YES":
        return print("Cancelled.")
    Base.metadata.drop_all(engine)
    MODEL_PATH.unlink(missing_ok=True)
    print("Done. Start the server again to re-seed.")


if __name__ == "__main__":
    main()

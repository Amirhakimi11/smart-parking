"""Re-train the forecasting model from whatever is in the database.

Usage:  python scripts/train_model.py
Run it again later when real readings have accumulated (or after loading your own data).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import select

from app import predictor
from app.db import Reading, SessionLocal, Zone, init_db


def main():
    init_db()
    with SessionLocal() as s:
        rows = s.execute(select(Reading.zone_id, Reading.ts, Reading.occupied)).all()
        caps = {z.id: z.capacity for z in s.query(Zone).all()}
    if not rows:
        sys.exit("No readings in the database yet. Start the server once so it seeds data.")
    m = predictor.train(pd.DataFrame(rows, columns=["zone_id", "ts", "occupied"]), caps)
    print("Saved models/occupancy_model.joblib")
    for k, v in m.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()

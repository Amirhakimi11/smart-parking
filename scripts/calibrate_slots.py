"""Draw the parking bays on the first frame of your video (one-time setup).

Usage:   python scripts/calibrate_slots.py --video data/demo/parking.mp4

For each ROW of bays:
  1. Click its 4 outer corners:  top-left -> top-right -> bottom-right -> bottom-left
  2. Type how many bays are in that row (digits), press Enter. The row is split evenly.
Keys:  u = undo last row   s = save and quit   q = quit without saving
Result is written to data/slots.json (the app reads it automatically).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def split_row(corners, n):
    """corners = [TL, TR, BR, BL] -> list of n bay polygons (4 points each)."""
    tl, tr, br, bl = [np.array(c, float) for c in corners]
    bays = []
    for i in range(n):
        a, b = i / n, (i + 1) / n
        t0, t1 = tl + (tr - tl) * a, tl + (tr - tl) * b
        b0, b1 = bl + (br - bl) * a, bl + (br - bl) * b
        bays.append([[round(float(x)), round(float(y))] for x, y in (t0, t1, b1, b0)])
    return bays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "demo" / "parking.mp4"))
    ap.add_argument("--out", default=str(ROOT / "data" / "slots.json"))
    ap.add_argument("--frame", type=int, default=0, help="which frame to draw on")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Cannot read a frame from {args.video}")
    H, W = frame.shape[:2]
    scale = min(1.0, 1280 / W)
    view = cv2.resize(frame, (int(W * scale), int(H * scale))) if scale < 1 else frame.copy()

    rows, clicks, typing = [], [], ""

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow("Calibrate bays")
    cv2.setMouseCallback("Calibrate bays", on_mouse)
    while True:
        img = view.copy()
        for row in rows:
            for poly in row:
                cv2.polylines(img, [(np.array(poly) * scale).astype(np.int32)], True, (90, 190, 60), 1)
        for i, (x, y) in enumerate(clicks):
            cv2.circle(img, (int(x * scale), int(y * scale)), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (int(x * scale) + 6, int(y * scale) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(clicks) < 4:
            msg = f"Click corner {len(clicks) + 1}/4 (TL, TR, BR, BL).  Rows: {len(rows)}  Bays: {sum(len(r) for r in rows)}"
        else:
            msg = f"How many bays in this row? {typing}_   (Enter to confirm)"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(img, msg, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Calibrate bays", img)
        k = cv2.waitKey(30) & 0xFF
        if len(clicks) == 4:
            if ord("0") <= k <= ord("9"):
                typing += chr(k)
            elif k in (8, 127):
                typing = typing[:-1]
            elif k in (13, 10) and typing:
                n = int(typing)
                if 1 <= n <= 80:
                    rows.append(split_row(clicks, n))
                clicks.clear()
                typing = ""
            continue
        if k == ord("u") and rows:
            rows.pop()
        elif k == ord("s"):
            bays = [p for r in rows for p in r]
            if not bays:
                print("Nothing to save yet.")
                continue
            Path(args.out).write_text(json.dumps({"frame": [W, H], "bays": bays}))
            print(f"Saved {len(bays)} bays to {args.out}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

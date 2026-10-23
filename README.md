---
title: AI Smart Parking and Congestion Forecaster
emoji: 🅿️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# AI Smart Parking & Congestion Forecaster

Sime Darby Property University Hackathon 2026 · Theme: Township of Tomorrow · Asset: retail mall car park (demo site: KL East Mall)

Drivers see live free bays per level, a next-hour AI forecast, and a live camera view where YOLOv8 marks each bay free or taken. Operators get an admin dashboard with alerts, zone controls, forecast accuracy and data export.

## How it fits together

```
 Camera / demo video ──► YOLOv8 (vehicle detection) ──► bay polygons ──► free / taken
                                                                       │
 Simulated sensors (other levels) ─────────────────────────────────────┤
                                                                       ▼
                                          Cloud PostgreSQL  (zones, readings, predictions)
                                                                       │
                                     scikit-learn forecaster (+15/+30/+45/+60 min)
                                                                       │
                                    FastAPI  ──►  User page (/)   Admin page (/admin)
```

* **Real in this prototype:** YOLOv8 inference on a video feed, the forecasting model, the cloud database, the web app.
* **Simulated in this prototype:** the car-park history and the sensors on every zone except the camera zone. The history is synthetic, shaped like a Malaysian mall (lunch bump, evening peak, heavier weekends, payday and fixed public holidays). Say this clearly in your pitch.

---

## 1. Run it on your laptop (VS Code)

Needs Python 3.10, 3.11 or 3.12 and VS Code with the Python extension.

Windows (PowerShell):
```powershell
cd smart-parking
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```
macOS / Linux:
```bash
cd smart-parking
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 (users) and http://localhost:8000/admin (password is `ADMIN_PASSWORD` in `.env`, default `admin123`).

The first start creates the tables, seeds 60 days of synthetic history, trains the model and starts the live feed. Allow 30 to 90 seconds. In VS Code choose the `.venv` interpreter (Ctrl+Shift+P, "Python: Select Interpreter").

Tip: with `--reload` the server restarts whenever you save a file, which re-runs the startup. Drop `--reload` for demos.

## 2. Connect the cloud database

The app uses SQLite until you set `DATABASE_URL`. Any PostgreSQL works. Neon is the quickest:

1. Create a free account at neon.tech and create a project.
2. Copy the connection string. It looks like `postgresql://user:password@ep-xxxx.neon.tech/neondb?sslmode=require`.
3. Paste it into `.env` as `DATABASE_URL=...` and restart the server.
4. Open http://localhost:8000/api/health. `"db":"postgresql"` means it worked. Tables are created automatically.

Supabase works the same way: use its PostgreSQL connection string (the pooler one is the most reliable) and URL-encode special characters in the password.

If you change database, the history is seeded again into the new one.

## 3. Switch on real YOLOv8 detection

Without a video the app shows a clearly labelled simulated feed. To run the real model:

1. Get a short parking-lot video filmed from a high angle so bays and cars are clearly visible (search "parking lot aerial" or "car park CCTV" on Pexels or Pixabay and check the licence, or film your own).
2. Compress it to under 10 MB and about 20 seconds so it deploys easily:
   ```
   ffmpeg -i input.mp4 -t 20 -vf scale=1280:-2 -an -crf 28 data/demo/parking.mp4
   ```
3. Draw the bays once:
   ```
   python scripts/calibrate_slots.py --video data/demo/parking.mp4
   ```
   For each row of bays click the 4 outer corners (top-left, top-right, bottom-right, bottom-left), type the number of bays, press Enter. Press `s` to save to `data/slots.json`.
4. Reset the data once, because the camera zone's capacity is now your bay count: `python scripts/reset_db.py` (type YES). This wipes the tables in whichever database `.env` points to.
5. Start the server. The camera panel now shows YOLOv8 boxes and green/red bays, and the zone `Rooftop A` takes its live count from the camera.

`yolov8n.pt` downloads automatically the first time. Detection counts cars, motorbikes, buses and trucks. If a bay is wrongly marked, redraw it a little tighter, or raise the camera angle.

To use a real CCTV stream later, replace the video path in `app/vision.py` (`cv2.VideoCapture`) with the camera's RTSP URL.

## 4. Datasets

| Dataset | Use it for | Link |
|---|---|---|
| Parking Birmingham (UCI) | Real occupancy time series. Run `scripts/benchmark_birmingham.py` to get a real-data accuracy number for your Feasibility slide. 35,717 rows, 30 car parks, Oct to Dec 2016. CC BY 4.0 | https://archive.ics.uci.edu/dataset/482/parking+birmingham |
| PKLot (Hugging Face mirrors) | Real parking-lot camera images labelled free/occupied. Use for test frames, screenshots, or fine-tuning YOLO later | https://huggingface.co/datasets/Voxel51/PKLot and https://huggingface.co/datasets/ShantyCam/pklot |

Cite the Birmingham dataset (Stolfi, D. 2017, UCI Machine Learning Repository, https://doi.org/10.24432/C51K5Z) and PKLot (Almeida et al., 2015, Expert Systems with Applications 42(11)). Check each licence before redistributing the files.

Benchmark on real data:
```
python scripts/benchmark_birmingham.py --csv data/birmingham.csv
```
Quote only the numbers this prints for the real file.

## 5. Deploy (Hugging Face Spaces, free)

YOLOv8 plus PyTorch needs more memory than most free hosts offer. A Hugging Face Space (Docker) gives 16 GB RAM on the free CPU tier.

1. Create a Hugging Face account, then New Space, SDK = Docker (blank), public.
2. Put this folder in a Git repo and push it to the Space (the Space gives you the `git` URL). The `README.md` header already contains the settings it needs.
3. In the Space, open Settings, Variables and secrets, and add secrets: `DATABASE_URL` (your Neon string), `ADMIN_PASSWORD`, `SECRET_KEY` (any long random text).
4. Wait for the build (about 5 to 10 minutes). Your public URL is `https://<user>-<space-name>.hf.space`. Use `/admin` for the dashboard.

Things to know:
* Free Spaces sleep when idle. Open the link a few minutes before judges do.
* Commit `data/demo/parking.mp4` and `data/slots.json` (they are not git-ignored). Videos over 10 MB need Git LFS.
* The model retrains from the database on each restart (about 20 seconds).
* Backup plan if hosting fails: run locally and share it with a free tunnel (`cloudflared tunnel --url http://localhost:8000`).

Render or Railway free tiers often run out of memory with PyTorch. If you must use them, use a paid instance of 2 GB or more.

## 6. Hackathon submission checklist (deadline 11:59 PM, 23 October 2026)

1. Pitch deck: maximum 10 main slides, the rest in an appendix.
2. Video or live demo link: 5 minutes showing the solution.
3. Prototype / code link: public GitHub repository plus the live URL.
4. Supporting documents (optional): dataset sources, benchmark output, architecture diagram.

Suggested 5-minute video:
* 0:00 Problem in one sentence and who it hurts (shoppers circling, queues at ramps, operators blind to congestion)
* 0:30 User page: free bays per level, best level to try, forecast chart
* 1:30 Levels list: tap a floor, show zones, capacity, status now and in 60 minutes
* 2:15 Camera panel: YOLOv8 marking bays
* 3:00 Admin: alerts, close a zone, press "Peak hour" and show the forecast reacting
* 4:00 Model quality: error in bays versus the "nothing changes" baseline, plus the Birmingham result
* 4:30 Impact, feasibility (existing CCTV, cloud, low cost) and rollout (pilot one level for 3 months)

## 7. Project layout

```
app/
  main.py        FastAPI app, startup, admin login, endpoints
  config.py      settings from .env
  db.py          tables: zones, readings, predictions
  simulator.py   demand model, history generator, simulated sensors
  predictor.py   forecasting model (train + predict)
  vision.py      YOLOv8 bay detection and MJPEG frames
  services.py    queries that build dashboard data
static/          index.html, admin.html, app.js, admin.js, styles.css, vendor/chart.umd.js
scripts/         calibrate_slots.py, train_model.py, benchmark_birmingham.py, reset_db.py
Dockerfile       for Hugging Face Spaces / Docker
```

Main endpoints: `GET /api/overview`, `GET /api/history?scope=ALL|L1|L1-A`, `GET /api/stream` (MJPEG), `POST /api/admin/login`, `GET /api/admin/summary`, `PATCH /api/admin/zones/{id}`, `POST /api/admin/scenario`, `GET /api/admin/export.csv`. Interactive docs at `/docs`.

## 8. Troubleshooting

* **Page loads but numbers stay at "--"**: open the browser console. Usually the server is still seeding. Wait one minute and refresh.
* **`ModuleNotFoundError`**: the virtual environment is not active. Re-run the activate command.
* **Database errors on Neon/Supabase**: the URL must keep `?sslmode=require`; passwords with `@` or `#` must be URL-encoded.
* **Calibration window does not open**: run it on the normal desktop, not inside WSL or a remote shell.
* **Slow video**: lower the video resolution to 1280 wide, or use a shorter loop.

## 9. Honest limits and next steps

* Forecast accuracy shown for the simulated mall is measured against simulated data, so it only proves the pipeline. The Birmingham benchmark is the real-data evidence.
* One camera covers one zone. A real rollout adds a camera or entry/exit counter per zone, and can use existing CCTV.
* Privacy: only vehicle counts are stored. No plates or faces are saved. Review PDPA obligations before any real deployment.
* Next steps: add real holidays (Hari Raya, Chinese New Year, Deepavali) to `FIXED_HOLIDAYS` in `app/simulator.py`, event calendars, weather, and LED signage output.

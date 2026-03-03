from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import math
import os
import requests
import pandas as pd
import io
import glob

app = FastAPI(title="Load Forecast Interface Prototype")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Historical store (prototype)
# ----------------------------
HISTORY_STORE: dict[str, list[list[float]]] = {}
HISTORY_KEEP_LAST = 14
HISTORICAL_LOAD_CSV = os.getenv("HISTORICAL_LOAD_CSV", "data/historical_load.csv")
USE_PRED_HISTORY_BASELINE_FALLBACK = os.getenv("USE_PRED_HISTORY_BASELINE_FALLBACK", "false").lower() == "true"
RESIDENTIAL_BASELINE_CSV = os.getenv(
    "RESIDENTIAL_BASELINE_CSV",
    r"c:\Users\14184\Downloads\BV06 - Residential Energy Consumption Data (2020-2024) - Jan. 2020 (1).csv",
)
LAST_RESIDENTIAL_BASELINE_PATH: str | None = None


@app.get("/")
def home():
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


# ----------------------------
# Preset locations (OpenWeather coords)
# ----------------------------
PRESET_LOCATIONS = {
    "toronto":   {"label": "Toronto, ON",        "lat": 43.7001, "lon": -79.4163},
    "ontario":   {"label": "Ontario (Ottawa)",   "lat": 45.4211, "lon": -75.6903},
    "alberta":   {"label": "Alberta (Edmonton)", "lat": 53.5501, "lon": -113.4687},
    "vancouver": {"label": "Vancouver, BC",      "lat": 49.2497, "lon": -123.1193},
}

OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"


# ----------------------------
# Requests
# ----------------------------
class ModelRequest(BaseModel):
    location_key: str = Field(..., description="Preset location key: toronto/ontario/alberta/vancouver")
    start_iso: str | None = Field(None)
    horizon_hours: int = Field(24, ge=1, le=168)


class AllRequest(BaseModel):
    # Each model has its own dropdown / location key
    res_location_key: str = Field("toronto")
    com_location_key: str = Field("toronto")
    ind_location_key: str = Field("ontario")

    start_iso: str | None = Field(None)
    horizon_hours: int = Field(24, ge=1, le=168)


# ----------------------------
# Helpers
# ----------------------------
def _parse_start(start_iso: str | None) -> datetime:
    if start_iso:
        return datetime.fromisoformat(start_iso)
    return datetime.now().replace(minute=0, second=0, microsecond=0)


def resolve_preset(key: str) -> tuple[str, float, float]:
    k = (key or "").strip().lower()
    if k not in PRESET_LOCATIONS:
        raise HTTPException(status_code=400, detail=f"Unknown location key '{key}'.")
    p = PRESET_LOCATIONS[k]
    return p["label"], float(p["lat"]), float(p["lon"])


def _fetch_openweather_current(lat: float, lon: float) -> dict | None:
    """
    Fetch current weather snapshot from OpenWeather.
    Returns None if API key is missing or request fails.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        res = requests.get(OPENWEATHER_CURRENT_URL, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return {
            "timestamp_utc": datetime.fromtimestamp(int(data["dt"])),
            "temp_c": float(data["main"]["temp"]),
            "relative_humidity_pct": float(data["main"]["humidity"]),
        }
    except Exception:
        return None


def mock_weather(location_label: str, lat: float, lon: float, start: datetime, horizon_hours: int):
    """
    Demo-only weather. Later replace with real OpenWeather call using lat/lon.
    """
    # Toronto override: use live OpenWeather current conditions if available.
    live = None
    if location_label == "Toronto, ON":
        live = _fetch_openweather_current(lat, lon)

    ts, temp, rh = [], [], []
    for i in range(horizon_hours):
        t = start + timedelta(hours=i)
        if live is not None:
            # OpenWeather /weather is a current snapshot (not hourly forecast).
            # We reuse it for each horizon step until a forecast endpoint is added.
            temp_val = live["temp_c"]
            rh_val = live["relative_humidity_pct"]
        else:
            temp_val = 10 + 7 * math.sin((i / 24) * 2 * math.pi)
            rh_val = 55 + 10 * math.cos((i / 24) * 2 * math.pi)
        ts.append(t.isoformat(timespec="minutes"))
        temp.append(round(temp_val, 2))
        rh.append(round(rh_val, 2))

    return {
        "module": "weather_openweather_current" if live is not None else "weather_mock",
        "location": location_label,
        "lat": lat,
        "lon": lon,
        "timestamps": ts,
        "temperature_C": temp,
        "relative_humidity_pct": rh,
    }


def mock_forecast(sector: str, weather_payload: dict):
    """
    Mock forecast for commercial/industrial (until you plug real ML for them).
    """
    ts = weather_payload["timestamps"]
    temps = weather_payload["temperature_C"]

    allowed = {"res", "com", "ind"}
    if sector not in allowed:
        return {"error": f"sector must be one of {sorted(allowed)}"}

    if sector == "res":
        scale, base_level = 1.0, 90
    elif sector == "com":
        scale, base_level = 1.4, 120
    else:  # ind
        scale, base_level = 1.9, 150

    yhat = []
    for i, temp in enumerate(temps):
        daily = 22 * math.sin((i / 24) * 2 * math.pi)
        temp_effect = max(0.0, (18 - temp)) * 1.2
        pred = (base_level + daily + temp_effect) * scale
        yhat.append(round(pred, 2))

    return {
        "module": f"forecast_{sector}",
        "sector": sector,
        "unit": "kW",
        "timestamps": ts,
        "predicted_load": yhat,
    }


# ----------------------------
# History helpers (baseline)
# ----------------------------
def _history_key(location_key: str, sector: str) -> str:
    return f"{location_key}::{sector}"


def _update_history(location_key: str, sector: str, predicted: list[float]):
    key = _history_key(location_key, sector)
    HISTORY_STORE.setdefault(key, [])
    HISTORY_STORE[key].append(predicted)
    if len(HISTORY_STORE[key]) > HISTORY_KEEP_LAST:
        HISTORY_STORE[key] = HISTORY_STORE[key][-HISTORY_KEEP_LAST:]


def _historical_baseline(location_key: str, sector: str, horizon: int) -> list[float] | None:
    key = _history_key(location_key, sector)
    runs = HISTORY_STORE.get(key, [])
    if not runs:
        return None

    min_len = min(min(len(r) for r in runs), horizon)
    baseline = []
    for i in range(min_len):
        baseline.append(sum(r[i] for r in runs) / len(runs))

    if len(baseline) < horizon:
        baseline += [baseline[-1]] * (horizon - len(baseline))

    return [round(x, 2) for x in baseline]


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _historical_baseline_from_actual_csv(
    location_key: str, sector: str, target_timestamps: list[str]
) -> list[float] | None:
    """
    Build baseline from historical actual load CSV.
    Expected columns (flexible names/case):
      - timestamp column: timestamp|datetime|dt|time
      - load column: load|actual_load|demand|kwh|kw
    Optional filter columns:
      - sector column: sector|model
      - location column: location_key|location
    """
    if not os.path.isfile(HISTORICAL_LOAD_CSV):
        return None

    try:
        df = pd.read_csv(HISTORICAL_LOAD_CSV)
    except Exception:
        return None

    ts_col = _find_first_col(df, ["timestamp", "datetime", "dt", "time"])
    load_col = _find_first_col(df, ["load", "actual_load", "demand", "kwh", "kw"])
    if ts_col is None or load_col is None:
        return None

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df[load_col] = pd.to_numeric(df[load_col], errors="coerce")
    df = df.dropna(subset=[ts_col, load_col])
    if df.empty:
        return None

    sector_col = _find_first_col(df, ["sector", "model"])
    if sector_col is not None:
        df = df[df[sector_col].astype(str).str.lower() == str(sector).lower()]
        if df.empty:
            return None

    location_col = _find_first_col(df, ["location_key", "location"])
    if location_col is not None:
        # Match both key and common label text where available.
        lk = str(location_key).lower()
        df_loc = df[df[location_col].astype(str).str.lower() == lk]
        if not df_loc.empty:
            df = df_loc

    df["hour"] = df[ts_col].dt.hour
    df["dow"] = df[ts_col].dt.dayofweek

    baseline: list[float] = []
    for ts in target_timestamps:
        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t):
            baseline.append(float("nan"))
            continue

        # Prefer same hour + weekday, fallback to same hour.
        subset = df[(df["hour"] == int(t.hour)) & (df["dow"] == int(t.dayofweek))]
        if subset.empty:
            subset = df[df["hour"] == int(t.hour)]

        if subset.empty:
            baseline.append(float("nan"))
        else:
            baseline.append(float(subset[load_col].mean()))

    if all(pd.isna(v) for v in baseline):
        return None
    return [round(float(v), 2) if not pd.isna(v) else None for v in baseline]  # type: ignore[list-item]


def _historical_baseline_from_residential_csv(target_timestamps: list[str]) -> list[float] | None:
    """
    Build residential baseline from Toronto residential consumption CSV.
    Expected columns:
      - Date
      - Hour (1-24)
      - Total Consumption (kWh)
    """
    global LAST_RESIDENTIAL_BASELINE_PATH

    candidate_paths: list[str] = []

    env_candidates = (os.getenv("RESIDENTIAL_BASELINE_CSVS") or "").strip()
    if env_candidates:
        candidate_paths.extend([p.strip() for p in env_candidates.split(";") if p.strip()])

    candidate_paths.append(RESIDENTIAL_BASELINE_CSV)
    candidate_paths.extend(glob.glob("data/*Residential*Energy*Consumption*.csv"))
    candidate_paths.extend(glob.glob("data/*residential*.csv"))

    user_profile = os.getenv("USERPROFILE")
    if user_profile:
        candidate_paths.extend(glob.glob(os.path.join(user_profile, "Downloads", "*Residential*Energy*Consumption*.csv")))
        candidate_paths.extend(glob.glob(os.path.join(user_profile, "Downloads", "*residential*.csv")))

    csv_path = next((p for p in candidate_paths if p and os.path.isfile(p)), None)
    if not csv_path:
        LAST_RESIDENTIAL_BASELINE_PATH = None
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        LAST_RESIDENTIAL_BASELINE_PATH = csv_path
        return None

    date_col = _find_first_col(df, ["Date", "date"])
    hour_col = _find_first_col(df, ["Hour", "hour", "hour_ending", "he"])
    load_col = _find_first_col(
        df,
        [
            "Total Consumption (kWh)",
            "total_consumption_kwh",
            "consumption_kwh",
            "load",
            "demand",
            "kwh",
            "kw",
        ],
    )
    if date_col is None or hour_col is None or load_col is None:
        LAST_RESIDENTIAL_BASELINE_PATH = csv_path
        return None

    temp = df.copy()
    type_col = _find_first_col(temp, ["Type", "type", "Sector", "sector"])
    if type_col is not None:
        temp = temp[temp[type_col].astype(str).str.lower() == "residential"]
        if temp.empty:
            LAST_RESIDENTIAL_BASELINE_PATH = csv_path
            return None

    temp[hour_col] = pd.to_numeric(temp[hour_col], errors="coerce")
    temp[load_col] = pd.to_numeric(temp[load_col], errors="coerce")
    temp["dt"] = pd.to_datetime(
        temp[date_col].astype(str) + " " + (temp[hour_col].astype("Int64") - 1).astype(str) + ":00",
        errors="coerce",
    )
    temp = temp.dropna(subset=["dt", load_col])
    if temp.empty:
        LAST_RESIDENTIAL_BASELINE_PATH = csv_path
        return None

    temp["hour"] = temp["dt"].dt.hour
    temp["dow"] = temp["dt"].dt.dayofweek

    baseline: list[float] = []
    for ts in target_timestamps:
        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t):
            baseline.append(float("nan"))
            continue

        # Prefer same hour + weekday, fallback to same hour.
        subset = temp[(temp["hour"] == int(t.hour)) & (temp["dow"] == int(t.dayofweek))]
        if subset.empty:
            subset = temp[temp["hour"] == int(t.hour)]

        if subset.empty:
            baseline.append(float("nan"))
        else:
            baseline.append(float(subset[load_col].mean()))

    if all(pd.isna(v) for v in baseline):
        LAST_RESIDENTIAL_BASELINE_PATH = csv_path
        return None
    LAST_RESIDENTIAL_BASELINE_PATH = csv_path
    return [round(float(v), 2) if not pd.isna(v) else None for v in baseline]  # type: ignore[list-item]


# ----------------------------
# ML support: build 168h past window (for LSTM)
# ----------------------------
def build_past_168_window(label: str, lat: float, lon: float, start: datetime) -> list[dict]:
    """
    Build 168 hours of past weather ending at `start` using mock weather for now.
    Your ML expects: temp, humidity, dewpoint + dt per hour.
    """
    past_start = start - timedelta(hours=168)
    w = mock_weather(label, lat, lon, past_start, 168)

    rows = []
    for i, ts in enumerate(w["timestamps"]):
        dt = datetime.fromisoformat(ts)
        temp = float(w["temperature_C"][i])
        hum = float(w["relative_humidity_pct"][i])

        # TEMP dewpoint placeholder:
        # Replace with real dewpoint from API or dewpoint formula later
        dew = temp - 2.0

        rows.append({"dt": dt, "temp": temp, "humidity": hum, "dewpoint": dew})
    return rows


def _ml_predict_residential_24h(window_rows: list[dict]) -> list[float]:
    """
    Lazy-import the ML module so the app can start even if tensorflow fails.
    Supports either function name your file might have.
    """
    try:
        from ml import toronto_res_model as res_model  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML import failed (tensorflow/joblib?): {e}")

    predictor = getattr(res_model, "predict_next_24", None) or getattr(res_model, "predict_next_24h_kwh", None)
    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="ML prediction function not found. Expected predict_next_24 or predict_next_24h_kwh."
        )

    try:
        y = predictor(window_rows)
        return [float(v) for v in y]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {e}")


# ----------------------------
# Debug endpoints (VERY useful on Azure)
# ----------------------------
@app.get("/api/debug/files")
def debug_files():
    """
    Lets you confirm Azure actually deployed your model/scaler files.
    """
    out = {
        "cwd": os.getcwd(),
        "root_list": os.listdir(".")[:200],
        "models_exists": os.path.isdir("models"),
        "models_files": os.listdir("models") if os.path.isdir("models") else [],
        "ml_exists": os.path.isdir("ml"),
        "ml_files": os.listdir("ml") if os.path.isdir("ml") else [],
    }
    return out


@app.get("/api/debug/try_load_ml")
def debug_try_load_ml():
    """
    Attempts to import and load the ML artifacts (will error if tensorflow or files missing).
    """
    try:
        from ml import toronto_res_model  # noqa
        return {"ok": True, "msg": "Imported ml.toronto_res_model successfully"}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/api/locations")
def list_locations():
    return {"locations": PRESET_LOCATIONS}


@app.post("/api/weather")
def api_weather(req: ModelRequest):
    start = _parse_start(req.start_iso)
    label, lat, lon = resolve_preset(req.location_key)
    return mock_weather(label, lat, lon, start, req.horizon_hours)


@app.post("/api/forecast/res")
def api_forecast_res(req: ModelRequest):
    """
    Residential ML endpoint (24-hour forecast) using your LSTM + scalers.
    """
    start = _parse_start(req.start_iso)
    label, lat, lon = resolve_preset(req.location_key)

    # Build 168h history window
    window_rows = build_past_168_window(label, lat, lon, start)

    # ML predict 24h
    yhat = _ml_predict_residential_24h(window_rows)

    # Build next 24 hourly timestamps
    ts = [(start + timedelta(hours=i + 1)).isoformat(timespec="minutes") for i in range(24)]

    out = {
        "module": "forecast_res_ml",
        "sector": "res",
        "unit": "kWh",
        "timestamps": ts,
        "predicted_load": [round(float(v), 2) for v in yhat],
        "location": label,
        "lat": lat,
        "lon": lon,
        "historical_baseline": _historical_baseline_from_residential_csv(ts),
        "baseline_source": "residential_csv",
        "baseline_source_path": LAST_RESIDENTIAL_BASELINE_PATH,
    }
    if out["historical_baseline"] is None:
        out["historical_baseline"] = _historical_baseline_from_actual_csv(req.location_key, "res", ts)
        out["baseline_source"] = "actual_history_csv"
        out["baseline_source_path"] = HISTORICAL_LOAD_CSV if out["historical_baseline"] is not None else None
    if out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        out["historical_baseline"] = _historical_baseline(req.location_key, "res", 24)
        out["baseline_source"] = "prediction_history_fallback"
        out["baseline_source_path"] = None
    if out["historical_baseline"] is None:
        out["baseline_source"] = "none"
        out["baseline_source_path"] = None

    # store history (use 24 horizon for ML)
    _update_history(req.location_key, "res", out["predicted_load"])
    return out


@app.post("/api/forecast/com")
def api_forecast_com(req: ModelRequest):
    """
    Commercial: still mock (until you have com model).
    """
    start = _parse_start(req.start_iso)
    label, lat, lon = resolve_preset(req.location_key)
    w = mock_weather(label, lat, lon, start, req.horizon_hours)

    sector = "com"
    out = mock_forecast(sector, w)
    baseline = _historical_baseline_from_actual_csv(req.location_key, sector, out["timestamps"])
    baseline_source = "actual_history_csv"
    if baseline is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        baseline = _historical_baseline(req.location_key, sector, req.horizon_hours)
        baseline_source = "prediction_history_fallback"
    if baseline is None:
        baseline_source = "none"
    _update_history(req.location_key, sector, out["predicted_load"])

    out["historical_baseline"] = baseline
    out["baseline_source"] = baseline_source
    out["location"] = label
    out["lat"] = lat
    out["lon"] = lon
    return out


@app.post("/api/forecast/ind")
def api_forecast_ind(req: ModelRequest):
    """
    Industrial: still mock (until you have ind model).
    """
    start = _parse_start(req.start_iso)
    label, lat, lon = resolve_preset(req.location_key)
    w = mock_weather(label, lat, lon, start, req.horizon_hours)

    sector = "ind"
    out = mock_forecast(sector, w)
    baseline = _historical_baseline_from_actual_csv(req.location_key, sector, out["timestamps"])
    baseline_source = "actual_history_csv"
    if baseline is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        baseline = _historical_baseline(req.location_key, sector, req.horizon_hours)
        baseline_source = "prediction_history_fallback"
    if baseline is None:
        baseline_source = "none"
    _update_history(req.location_key, sector, out["predicted_load"])

    out["historical_baseline"] = baseline
    out["baseline_source"] = baseline_source
    out["location"] = label
    out["lat"] = lat
    out["lon"] = lon
    return out


@app.post("/api/run_all")
def api_run_all(req: AllRequest):
    """
    Runs Residential (ML) + Commercial (mock) + Industrial (mock).
    Each model uses its own location dropdown selection.
    """
    start = _parse_start(req.start_iso)

    # ---- Residential (ML 24h) ----
    res_label, res_lat, res_lon = resolve_preset(req.res_location_key)
    res_window = build_past_168_window(res_label, res_lat, res_lon, start)
    res_yhat = _ml_predict_residential_24h(res_window)
    res_ts = [(start + timedelta(hours=i + 1)).isoformat(timespec="minutes") for i in range(24)]

    res_out = {
        "module": "forecast_res_ml",
        "sector": "res",
        "unit": "kWh",
        "timestamps": res_ts,
        "predicted_load": [round(float(v), 2) for v in res_yhat],
        "historical_baseline": _historical_baseline_from_residential_csv(res_ts),
        "baseline_source": "residential_csv",
        "baseline_source_path": LAST_RESIDENTIAL_BASELINE_PATH,
    }
    if res_out["historical_baseline"] is None:
        res_out["historical_baseline"] = _historical_baseline_from_actual_csv(req.res_location_key, "res", res_ts)
        res_out["baseline_source"] = "actual_history_csv"
        res_out["baseline_source_path"] = HISTORICAL_LOAD_CSV if res_out["historical_baseline"] is not None else None
    if res_out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        res_out["historical_baseline"] = _historical_baseline(req.res_location_key, "res", 24)
        res_out["baseline_source"] = "prediction_history_fallback"
        res_out["baseline_source_path"] = None
    if res_out["historical_baseline"] is None:
        res_out["baseline_source"] = "none"
        res_out["baseline_source_path"] = None
    _update_history(req.res_location_key, "res", res_out["predicted_load"])

    res_weather = mock_weather(res_label, res_lat, res_lon, start, req.horizon_hours)

    # ---- Commercial (mock) ----
    com_label, com_lat, com_lon = resolve_preset(req.com_location_key)
    com_weather = mock_weather(com_label, com_lat, com_lon, start, req.horizon_hours)
    com_out = mock_forecast("com", com_weather)
    com_out["historical_baseline"] = _historical_baseline_from_actual_csv(
        req.com_location_key, "com", com_out["timestamps"]
    )
    com_out["baseline_source"] = "actual_history_csv"
    if com_out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        com_out["historical_baseline"] = _historical_baseline(req.com_location_key, "com", req.horizon_hours)
        com_out["baseline_source"] = "prediction_history_fallback"
    if com_out["historical_baseline"] is None:
        com_out["baseline_source"] = "none"
    _update_history(req.com_location_key, "com", com_out["predicted_load"])

    # ---- Industrial (mock) ----
    ind_label, ind_lat, ind_lon = resolve_preset(req.ind_location_key)
    ind_weather = mock_weather(ind_label, ind_lat, ind_lon, start, req.horizon_hours)
    ind_out = mock_forecast("ind", ind_weather)
    ind_out["historical_baseline"] = _historical_baseline_from_actual_csv(
        req.ind_location_key, "ind", ind_out["timestamps"]
    )
    ind_out["baseline_source"] = "actual_history_csv"
    if ind_out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        ind_out["historical_baseline"] = _historical_baseline(req.ind_location_key, "ind", req.horizon_hours)
        ind_out["baseline_source"] = "prediction_history_fallback"
    if ind_out["historical_baseline"] is None:
        ind_out["baseline_source"] = "none"
    _update_history(req.ind_location_key, "ind", ind_out["predicted_load"])

    return {
        "module": "pipeline_all",
        "start": start.isoformat(timespec="minutes"),
        "horizon_hours": req.horizon_hours,
        "residential": {"forecast": res_out, "weather": res_weather},
        "commercial": {"forecast": com_out, "weather": com_weather},
        "industrial": {"forecast": ind_out, "weather": ind_weather},
    }


@app.post("/api/train")
async def api_train(file: UploadFile | None = File(default=None), notes: str | None = Form(default=None)):
    """
    Lightweight training endpoint to make Train UI functional.
    This validates and inspects uploaded CSV; real model fitting can be added later.
    """
    notes_text = (notes or "").strip()

    if file is None:
        return {
            "ok": True,
            "module": "train_preview",
            "message": "Train request received (no CSV uploaded).",
            "trained": False,
            "rows": 0,
            "columns": [],
            "numeric_columns": [],
            "notes": notes_text,
        }

    if not str(file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Training file must be a .csv")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read uploaded file: {e}")

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    return {
        "ok": True,
        "module": "train_preview",
        "message": f"Training data accepted: {len(df)} rows, {len(df.columns)} columns.",
        "trained": False,
        "filename": file.filename,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "numeric_columns": [str(c) for c in numeric_cols],
        "notes": notes_text,
    }

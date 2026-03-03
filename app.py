from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import math
import os
import requests

app = FastAPI(title="Load Forecast Interface Prototype")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Historical store (prototype)
# ----------------------------
HISTORY_STORE: dict[str, list[list[float]]] = {}
HISTORY_KEEP_LAST = 14


@app.get("/")
def home():
    return FileResponse("static/index.html")


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


def _historical_baseline(
    location_key: str,
    sector: str,
    horizon: int,
    current_predicted: list[float] | None = None,
    min_diff: float = 1e-6,
) -> list[float] | None:
    key = _history_key(location_key, sector)
    runs = HISTORY_STORE.get(key, [])
    if not runs:
        return None

    # If current prediction is provided, drop prior runs that are effectively identical.
    # This avoids plotting a "historical" line that is exactly the same as current.
    if current_predicted:
        filtered = []
        for r in runs:
            n = min(len(r), len(current_predicted), horizon)
            if n == 0:
                continue
            max_abs_diff = max(abs(float(r[i]) - float(current_predicted[i])) for i in range(n))
            if max_abs_diff > min_diff:
                filtered.append(r)
        runs = filtered
        if not runs:
            return None

    min_len = min(min(len(r) for r in runs), horizon)
    baseline = []
    for i in range(min_len):
        baseline.append(sum(r[i] for r in runs) / len(runs))

    if len(baseline) < horizon:
        baseline += [baseline[-1]] * (horizon - len(baseline))

    return [round(x, 2) for x in baseline]


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
        "historical_baseline": _historical_baseline(
            req.location_key, "res", 24, current_predicted=[round(float(v), 2) for v in yhat]
        ),
    }

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
    baseline = _historical_baseline(
        req.location_key, sector, req.horizon_hours, current_predicted=out["predicted_load"]
    )
    _update_history(req.location_key, sector, out["predicted_load"])

    out["historical_baseline"] = baseline
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
    baseline = _historical_baseline(
        req.location_key, sector, req.horizon_hours, current_predicted=out["predicted_load"]
    )
    _update_history(req.location_key, sector, out["predicted_load"])

    out["historical_baseline"] = baseline
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
        "historical_baseline": _historical_baseline(
            req.res_location_key, "res", 24, current_predicted=[round(float(v), 2) for v in res_yhat]
        ),
    }
    _update_history(req.res_location_key, "res", res_out["predicted_load"])

    res_weather = mock_weather(res_label, res_lat, res_lon, start, req.horizon_hours)

    # ---- Commercial (mock) ----
    com_label, com_lat, com_lon = resolve_preset(req.com_location_key)
    com_weather = mock_weather(com_label, com_lat, com_lon, start, req.horizon_hours)
    com_out = mock_forecast("com", com_weather)
    com_out["historical_baseline"] = _historical_baseline(
        req.com_location_key, "com", req.horizon_hours, current_predicted=com_out["predicted_load"]
    )
    _update_history(req.com_location_key, "com", com_out["predicted_load"])

    # ---- Industrial (mock) ----
    ind_label, ind_lat, ind_lon = resolve_preset(req.ind_location_key)
    ind_weather = mock_weather(ind_label, ind_lat, ind_lon, start, req.horizon_hours)
    ind_out = mock_forecast("ind", ind_weather)
    ind_out["historical_baseline"] = _historical_baseline(
        req.ind_location_key, "ind", req.horizon_hours, current_predicted=ind_out["predicted_load"]
    )
    _update_history(req.ind_location_key, "ind", ind_out["predicted_load"])

    return {
        "module": "pipeline_all",
        "start": start.isoformat(timespec="minutes"),
        "horizon_hours": req.horizon_hours,
        "residential": {"forecast": res_out, "weather": res_weather},
        "commercial": {"forecast": com_out, "weather": com_weather},
        "industrial": {"forecast": ind_out, "weather": ind_weather},
    }

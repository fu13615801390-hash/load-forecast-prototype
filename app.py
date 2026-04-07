from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
from threading import Lock
import math
import os
import re
import tempfile
import requests
import pandas as pd
import io
import glob
import numpy as np
import joblib
import uuid
import zipfile
import copy
from zoneinfo import ZoneInfo

app = FastAPI(title="Load Forecast Interface")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Historical store (prototype)
# ----------------------------
HISTORY_STORE: dict[str, list[list[float]]] = {}
HISTORY_KEEP_LAST = 14
WEATHER_UPLOAD_STORE: dict[str, dict] = {}
HISTORICAL_LOAD_CSV = os.getenv("HISTORICAL_LOAD_CSV", "data/historical_load.csv")
USE_PRED_HISTORY_BASELINE_FALLBACK = os.getenv("USE_PRED_HISTORY_BASELINE_FALLBACK", "false").lower() == "true"
RESIDENTIAL_BASELINE_CSV = os.getenv(
    "RESIDENTIAL_BASELINE_CSV",
    r"c:\Users\14184\Downloads\BV06 - Residential Energy Consumption Data (2020-2024) - Jan. 2020 (1).csv",
)
LAST_RESIDENTIAL_BASELINE_PATH: str | None = None
TRAINED_USER_MODEL_PATH = os.path.join("models", "trained", "userModel.keras")
TRAINED_COM_MODEL_DIR = os.path.join("models", "trained_commercial")
TRAINED_IND_MODEL_DIR = os.path.join("models", "trained_industrial")
COMMERCIAL_MODEL_DIR = os.path.join("models", "commercial")
ONTARIO_INDUSTRIAL_MODEL_DIR = os.path.join("models", "ontario_industrial")
ALBERTA_COMMERCIAL_MODEL_DIR = os.path.join("models", "alberta_commercial")
ALBERTA_INDUSTRIAL_MODEL_DIR = os.path.join("models", "alberta_industrial")
BC_COMMERCIAL_MODEL_DIR = os.path.join("models", "bc_commercial")
BC_INDUSTRIAL_MODEL_DIR = os.path.join("models", "bc_industrial")
DISPLAY_ENERGY_UNIT = "MW"
KWH_TO_MW = 1.0 / 1000.0
KW_TO_MW = 1.0 / 1000.0
APP_TIMEZONE = ZoneInfo("America/Toronto")
USER_TRAINED_SECTOR_DIRS = {
    "res": os.path.join("models", "trained"),
    "com": TRAINED_COM_MODEL_DIR,
    "ind": TRAINED_IND_MODEL_DIR,
}


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
    return {"status": "ok", "time": _toronto_now().isoformat(timespec="seconds")}


# ----------------------------
# Preset locations (OpenWeather coords)
# ----------------------------
PRESET_LOCATIONS = {
    "toronto":   {"label": "Toronto, ON",        "lat": 43.7001, "lon": -79.4163},
    "ontario":   {"label": "Ontario",            "lat": 45.4211, "lon": -75.6903},
    "alberta":   {"label": "Alberta",            "lat": 53.5501, "lon": -113.4687},
    "vancouver": {"label": "British Columbia",   "lat": 49.2497, "lon": -123.1193},
}

OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_HTTP_TIMEOUT = (
    float(os.getenv("WEATHER_HTTP_CONNECT_TIMEOUT_SECONDS", "15")),
    float(os.getenv("WEATHER_HTTP_READ_TIMEOUT_SECONDS", "180")),
)
WEATHER_CACHE_TTL_SECONDS = int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "300"))

_HTTP_SESSION = requests.Session()
_WEATHER_CACHE: dict[tuple, tuple[datetime, dict | None]] = {}
_WEATHER_CACHE_LOCK = Lock()
_ACTUAL_HISTORY_CACHE: dict[str, tuple[float, pd.DataFrame | None]] = {}
_ACTUAL_HISTORY_CACHE_LOCK = Lock()
_RESIDENTIAL_BASELINE_CACHE: dict[str, tuple[float, pd.DataFrame | None]] = {}
_RESIDENTIAL_BASELINE_CACHE_LOCK = Lock()


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
    return _toronto_now().replace(minute=0, second=0, microsecond=0)


def _toronto_now() -> datetime:
    return datetime.now(APP_TIMEZONE).replace(tzinfo=None)


def _toronto_naive_from_unix_timestamp(value: int | float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone(APP_TIMEZONE).replace(tzinfo=None)


def resolve_preset(key: str) -> tuple[str, float, float]:
    k = (key or "").strip().lower()
    if k not in PRESET_LOCATIONS:
        raise HTTPException(status_code=400, detail=f"Unknown location key '{key}'.")
    p = PRESET_LOCATIONS[k]
    return p["label"], float(p["lat"]), float(p["lon"])


def _openweather_api_key() -> str | None:
    return os.getenv("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHER_APPID")


def _cache_now() -> datetime:
    return datetime.now()


def _get_ttl_cache_value(cache: dict, key: tuple, ttl_seconds: int):
    now = _cache_now()
    with _WEATHER_CACHE_LOCK:
        cached = cache.get(key)
        if cached is None:
            return None
        expires_at, payload = cached
        if expires_at <= now:
            cache.pop(key, None)
            return None
        return copy.deepcopy(payload)


def _set_ttl_cache_value(cache: dict, key: tuple, value, ttl_seconds: int):
    expires_at = _cache_now() + timedelta(seconds=max(1, ttl_seconds))
    with _WEATHER_CACHE_LOCK:
        cache[key] = (expires_at, copy.deepcopy(value))


def _fetch_openweather_current(lat: float, lon: float) -> dict | None:
    """
    Fetch current weather snapshot from OpenWeather.
    Returns None if API key is missing or request fails.
    """
    api_key = _openweather_api_key()
    if not api_key:
        return None

    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    cache_key = ("ow_current", round(float(lat), 4), round(float(lon), 4))
    cached = _get_ttl_cache_value(_WEATHER_CACHE, cache_key, WEATHER_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        res = _HTTP_SESSION.get(OPENWEATHER_CURRENT_URL, params=params, timeout=WEATHER_HTTP_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        payload = {
            "timestamp_utc": _toronto_naive_from_unix_timestamp(int(data["dt"])),
            "temp_c": float(data["main"]["temp"]),
            "relative_humidity_pct": float(data["main"]["humidity"]),
        }
        _set_ttl_cache_value(_WEATHER_CACHE, cache_key, payload, WEATHER_CACHE_TTL_SECONDS)
        return payload
    except Exception:
        return None


def _fetch_openweather_forecast(lat: float, lon: float, start: datetime, horizon_hours: int) -> dict | None:
    """
    Fetch forecast from OpenWeather /forecast (3-hour steps) and map to hourly timestamps.
    """
    api_key = _openweather_api_key()
    if not api_key:
        return None

    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    cache_key = (
        "ow_forecast",
        round(float(lat), 4),
        round(float(lon), 4),
        start.isoformat(timespec="hours"),
        int(horizon_hours),
    )
    cached = _get_ttl_cache_value(_WEATHER_CACHE, cache_key, WEATHER_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        res = _HTTP_SESSION.get(OPENWEATHER_FORECAST_URL, params=params, timeout=WEATHER_HTTP_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        items = data.get("list", [])
        if not items:
            return None

        pts = []
        for it in items:
            dt = _toronto_naive_from_unix_timestamp(int(it["dt"]))
            temp = float(it["main"]["temp"])
            hum = float(it["main"]["humidity"])
            pts.append((dt, temp, hum))
        if not pts:
            return None

        ts, temp, rh = [], [], []
        for i in range(horizon_hours):
            t = start + timedelta(hours=i)
            # nearest-neighbor mapping from 3-hour forecast to hourly axis
            nearest = min(pts, key=lambda p: abs((p[0] - t).total_seconds()))
            ts.append(t.isoformat(timespec="minutes"))
            temp.append(round(float(nearest[1]), 2))
            rh.append(round(float(nearest[2]), 2))

        payload = {
            "module": "weather_openweather_forecast",
            "location": "",
            "lat": lat,
            "lon": lon,
            "timestamps": ts,
            "temperature_C": temp,
            "relative_humidity_pct": rh,
        }
        _set_ttl_cache_value(_WEATHER_CACHE, cache_key, payload, WEATHER_CACHE_TTL_SECONDS)
        return payload
    except Exception:
        return None


def _synthetic_weather_series(lat: float, lon: float, start: datetime, horizon_hours: int) -> tuple[list[float], list[float]]:
    """
    Deterministic fallback weather that still varies by location and season.
    This keeps forecasts from collapsing to the exact same weather profile when
    the live API is unavailable.
    """
    temps: list[float] = []
    humidity: list[float] = []

    lat_abs = abs(float(lat))
    lon_abs = abs(float(lon))
    coord_phase = math.radians((lat_abs * 3.7 + lon_abs * 1.9) % 360.0)
    daily_amp = 5.0 + min(lat_abs / 18.0, 4.0)
    seasonal_amp = 7.5 + min(lat_abs / 10.0, 6.0)
    baseline = 14.0 - min(lat_abs / 7.5, 8.0) - min(lon_abs / 90.0, 2.0)
    humidity_base = 58.0 + min(lon_abs / 10.0, 14.0) - min(lat_abs / 12.0, 7.0)

    for i in range(horizon_hours):
        t = start + timedelta(hours=i)
        seasonal_angle = (t.timetuple().tm_yday / 365.25) * (2 * math.pi)
        daily_angle = ((t.hour + (lon / 15.0)) / 24.0) * (2 * math.pi)

        temp_val = (
            baseline
            + seasonal_amp * math.sin(seasonal_angle - 1.2 + coord_phase * 0.35)
            + daily_amp * math.sin(daily_angle - math.pi / 2 + coord_phase)
        )
        rh_val = (
            humidity_base
            + 10.0 * math.cos(daily_angle + coord_phase * 0.7)
            - 6.0 * math.sin(seasonal_angle + coord_phase * 0.2)
        )

        temps.append(round(temp_val, 2))
        humidity.append(round(min(95.0, max(25.0, rh_val)), 2))

    return temps, humidity


def mock_weather(location_label: str, lat: float, lon: float, start: datetime, horizon_hours: int):
    """
    Weather provider with automatic fallback order:
    1) OpenWeather forecast (3h forecast mapped to hourly)
    2) OpenWeather current snapshot (reused across horizon)
    3) Synthetic mock weather
    """
    live_forecast = _fetch_openweather_forecast(lat, lon, start, horizon_hours)
    if live_forecast is not None:
        live_forecast["location"] = location_label
        return live_forecast

    live = _fetch_openweather_current(lat, lon)

    ts, temp, rh = [], [], []
    for i in range(horizon_hours):
        t = start + timedelta(hours=i)
        if live is not None:
            # OpenWeather /weather is a current snapshot (not hourly forecast).
            # We reuse it for each horizon step until a forecast endpoint is added.
            temp_val = live["temp_c"]
            rh_val = live["relative_humidity_pct"]
        ts.append(t.isoformat(timespec="minutes"))
        if live is not None:
            temp.append(round(temp_val, 2))
            rh.append(round(rh_val, 2))

    if live is None:
        temp, rh = _synthetic_weather_series(lat, lon, start, horizon_hours)

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


def _parse_uploaded_weather_csv(content: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    cols = list(df.columns)

    ts_col = _find_first_col(df, ["timestamp", "datetime", "dt", "time", "Date/Time (LST)", "date/time (lst)"])
    if ts_col is None:
        ts_col = next((c for c in cols if "date/time" in c.lower()), None)

    temp_col = _find_first_col(df, ["temperature_C", "temp_c", "temp", "temperature"])
    if temp_col is None:
        temp_col = next(
            (
                c
                for c in cols
                if ("temp" in c.lower()) and ("dew" not in c.lower()) and ("flag" not in c.lower())
            ),
            None,
        )

    hum_col = _find_first_col(df, ["relative_humidity_pct", "humidity_percent", "humidity", "rh", "Rel Hum (%)"])
    if hum_col is None:
        hum_col = next((c for c in cols if ("rel hum" in c.lower()) and ("flag" not in c.lower())), None)

    dew_col = _find_first_col(df, ["dew_point_temp_c", "dew point temp (°c)", "dew_point_c", "dewpoint_c", "dewpoint"])
    if dew_col is None:
        dew_col = next((c for c in cols if ("dew" in c.lower()) and ("flag" not in c.lower())), None)

    if ts_col is None or temp_col is None or hum_col is None:
        raise HTTPException(
            status_code=400,
            detail="Weather CSV must include timestamp, temperature_C (or temp), and relative_humidity_pct (or humidity).",
        )

    keep_cols = [ts_col, temp_col, hum_col] + ([dew_col] if dew_col is not None else [])
    tmp = df[keep_cols].copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp[temp_col] = pd.to_numeric(tmp[temp_col], errors="coerce")
    tmp[hum_col] = pd.to_numeric(tmp[hum_col], errors="coerce")
    if dew_col is not None:
        tmp[dew_col] = pd.to_numeric(tmp[dew_col], errors="coerce")
    tmp = tmp.dropna(subset=[ts_col, temp_col, hum_col]).sort_values(ts_col).reset_index(drop=True)
    if tmp.empty:
        raise HTTPException(status_code=400, detail="Weather CSV has no valid rows after parsing.")

    dew_values: list[float]
    if dew_col is not None:
        dew_series = tmp[dew_col]
        if dew_series.isna().all():
            dew_values = []
        else:
            dew_values = [round(float(v), 4) if not pd.isna(v) else None for v in dew_series.tolist()]  # type: ignore[list-item]
    else:
        dew_values = []

    return {
        "timestamps": [d.isoformat(timespec="minutes") for d in tmp[ts_col].tolist()],
        "temperature_C": [round(float(v), 4) for v in tmp[temp_col].tolist()],
        "relative_humidity_pct": [round(float(v), 4) for v in tmp[hum_col].tolist()],
        "dew_point_C": dew_values,
    }


def _with_datetime_column(df: pd.DataFrame, col_name: str = "__dt") -> pd.DataFrame:
    """
    Build a datetime column from either timestamp OR Date+Hour fields.
    """
    ts_col = _find_first_col(df, ["timestamp", "datetime", "dt", "time"])
    date_col = _find_first_col(df, ["Date", "date"])
    hour_col = _find_first_col(df, ["Hour", "hour", "hour_ending", "he"])

    out = df.copy()
    if ts_col is not None:
        out[col_name] = pd.to_datetime(out[ts_col], errors="coerce")
    elif date_col is not None and hour_col is not None:
        out[hour_col] = pd.to_numeric(out[hour_col], errors="coerce")
        out[col_name] = pd.to_datetime(
            out[date_col].astype(str) + " " + (out[hour_col].astype("Int64") - 1).astype(str) + ":00",
            errors="coerce",
        )
    else:
        raise ValueError("Need timestamp column or Date+Hour columns.")

    out = out.dropna(subset=[col_name]).copy()
    return out


def _parse_weather_training_df(content: bytes) -> pd.DataFrame:
    """
    Parse uploaded weather CSV into [__dt, temperature_C, relative_humidity_pct].
    """
    parsed = _parse_uploaded_weather_csv(content)
    dew_values = parsed.get("dew_point_C") or []
    if dew_values and len(dew_values) == len(parsed["timestamps"]):
        dew_series = pd.to_numeric(pd.Series(dew_values), errors="coerce")
    else:
        temp_series = pd.to_numeric(pd.Series(parsed["temperature_C"]), errors="coerce")
        rh_series = pd.to_numeric(pd.Series(parsed["relative_humidity_pct"]), errors="coerce")
        # Magnus approximation for dew point in Celsius.
        alpha = np.log((rh_series.clip(lower=1e-6)) / 100.0) + (17.625 * temp_series) / (243.04 + temp_series)
        dew_series = 243.04 * alpha / (17.625 - alpha)

    wdf = pd.DataFrame(
        {
            "__dt": pd.to_datetime(parsed["timestamps"], errors="coerce"),
            "temperature_C": pd.to_numeric(pd.Series(parsed["temperature_C"]), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(pd.Series(parsed["relative_humidity_pct"]), errors="coerce"),
            "dew_point_C": dew_series,
        }
    )
    wdf = (
        wdf.dropna(subset=["__dt", "temperature_C", "relative_humidity_pct", "dew_point_C"])
        .sort_values("__dt")
        .reset_index(drop=True)
    )
    return wdf


def _safe_filename_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return text.strip("._") or "file"


def _build_combined_training_dataframe(df: pd.DataFrame, weather_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict | None]:
    weather_merge_stats: dict | None = None

    try:
        load_df = _with_datetime_column(df, "__dt")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Power CSV datetime parse failed: {e}")

    target_col = _find_first_col(
        load_df,
        [
            "TOTAL_CONSUMPTION (kWh)",
            "Total Consumption (kWh)",
            "total_consumption_kwh",
            "consumption_kwh",
            "predicted_load",
            "load",
            "actual_load",
            "demand",
            "kwh",
            "kw",
        ],
    )
    if target_col is None:
        raise HTTPException(status_code=400, detail="Power CSV must include a load column such as Total Consumption (kWh).")

    load_df[target_col] = pd.to_numeric(load_df[target_col], errors="coerce")
    load_df["__dt_hour"] = pd.to_datetime(load_df["__dt"]).dt.floor("h")
    combined = load_df[["__dt_hour", target_col]].rename(
        columns={"__dt_hour": "datetime", target_col: "TOTAL_CONSUMPTION (kWh)"}
    )

    if weather_df is not None:
        weather_hourly = weather_df.copy()
        weather_hourly["__dt_hour"] = pd.to_datetime(weather_hourly["__dt"]).dt.floor("h")
        merged = combined.merge(
            weather_hourly[["__dt_hour", "temperature_C", "relative_humidity_pct", "dew_point_C"]],
            left_on="datetime",
            right_on="__dt_hour",
            how="left",
        )
        matched = int(merged["temperature_C"].notna().sum())
        if matched < max(12, int(0.05 * len(merged))):
            raise HTTPException(
                status_code=400,
                detail="Weather merge matched too few rows. Check timestamp alignment/timezone between load and weather CSVs.",
            )
        weather_merge_stats = {
            "weather_rows": int(len(weather_hourly)),
            "power_rows_after_dt_parse": int(len(load_df)),
            "merged_rows": int(len(merged)),
            "weather_matched_rows": matched,
        }
        combined = merged.drop(columns=["__dt_hour"], errors="ignore")
        combined = combined.rename(
            columns={
                "temperature_C": "Temp (°C)",
                "relative_humidity_pct": "Rel Hum (%)",
                "dew_point_C": "Dew Point Temp (°C)",
            }
        )
    else:
        temp_col = _find_first_col(load_df, ["Temp (°C)", "Temp (C)", "temperature_C", "temp_c", "temp"])
        hum_col = _find_first_col(load_df, ["Rel Hum (%)", "relative_humidity_pct", "humidity_percent", "humidity"])
        dew_col = _find_first_col(load_df, ["Dew Point Temp (°C)", "Dew Point Temp (C)", "dew_point_c", "dewpoint_c", "dewpoint"])
        if temp_col is None or hum_col is None:
            raise HTTPException(
                status_code=400,
                detail="Training requires either a separate weather CSV or weather columns in the power CSV.",
            )
        combined["Temp (°C)"] = pd.to_numeric(load_df[temp_col], errors="coerce")
        combined["Rel Hum (%)"] = pd.to_numeric(load_df[hum_col], errors="coerce")
        if dew_col is not None:
            combined["Dew Point Temp (°C)"] = pd.to_numeric(load_df[dew_col], errors="coerce")
        else:
            alpha = np.log((combined["Rel Hum (%)"].clip(lower=1e-6)) / 100.0) + (
                17.625 * combined["Temp (°C)"]
            ) / (243.04 + combined["Temp (°C)"])
            combined["Dew Point Temp (°C)"] = 243.04 * alpha / (17.625 - alpha)

    combined = combined[
        ["datetime", "TOTAL_CONSUMPTION (kWh)", "Temp (°C)", "Rel Hum (%)", "Dew Point Temp (°C)"]
    ].copy()
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    for col in ["TOTAL_CONSUMPTION (kWh)", "Temp (°C)", "Rel Hum (%)", "Dew Point Temp (°C)"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.dropna().drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return combined, weather_merge_stats


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


def _load_actual_history_df(csv_path: str) -> pd.DataFrame | None:
    try:
        mtime = os.path.getmtime(csv_path)
    except OSError:
        return None

    with _ACTUAL_HISTORY_CACHE_LOCK:
        cached = _ACTUAL_HISTORY_CACHE.get(csv_path)
        if cached is not None and cached[0] == mtime:
            return cached[1].copy() if cached[1] is not None else None

    try:
        raw_df = pd.read_csv(csv_path)
    except Exception:
        parsed = None
    else:
        ts_col = _find_first_col(raw_df, ["timestamp", "datetime", "dt", "time"])
        load_col = _find_first_col(raw_df, ["load", "actual_load", "demand", "kwh", "kw"])
        if ts_col is None or load_col is None:
            parsed = None
        else:
            parsed = raw_df.copy()
            parsed[ts_col] = pd.to_datetime(parsed[ts_col], errors="coerce")
            parsed[load_col] = pd.to_numeric(parsed[load_col], errors="coerce")
            parsed = parsed.dropna(subset=[ts_col, load_col]).copy()
            if parsed.empty:
                parsed = None
            else:
                parsed["ts"] = parsed[ts_col]
                parsed["load"] = parsed[load_col]
                sector_col = _find_first_col(parsed, ["sector", "model"])
                if sector_col is not None:
                    parsed["sector"] = parsed[sector_col].astype(str).str.lower()
                location_col = _find_first_col(parsed, ["location_key", "location"])
                if location_col is not None:
                    parsed["location_key"] = parsed[location_col].astype(str).str.lower()
                parsed["hour"] = parsed["ts"].dt.hour
                parsed["dow"] = parsed["ts"].dt.dayofweek
                keep_cols = ["ts", "load", "hour", "dow"]
                if "sector" in parsed.columns:
                    keep_cols.append("sector")
                if "location_key" in parsed.columns:
                    keep_cols.append("location_key")
                parsed = parsed[keep_cols].reset_index(drop=True)

    with _ACTUAL_HISTORY_CACHE_LOCK:
        _ACTUAL_HISTORY_CACHE[csv_path] = (mtime, parsed.copy() if parsed is not None else None)
    return parsed.copy() if parsed is not None else None


def _load_residential_baseline_df(csv_path: str) -> pd.DataFrame | None:
    try:
        mtime = os.path.getmtime(csv_path)
    except OSError:
        return None

    with _RESIDENTIAL_BASELINE_CACHE_LOCK:
        cached = _RESIDENTIAL_BASELINE_CACHE.get(csv_path)
        if cached is not None and cached[0] == mtime:
            return cached[1].copy() if cached[1] is not None else None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        parsed = None
    else:
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
            parsed = None
        else:
            temp = df.copy()
            type_col = _find_first_col(temp, ["Type", "type", "Sector", "sector"])
            if type_col is not None:
                temp = temp[temp[type_col].astype(str).str.lower() == "residential"]
            if temp.empty:
                parsed = None
            else:
                temp[hour_col] = pd.to_numeric(temp[hour_col], errors="coerce")
                temp[load_col] = pd.to_numeric(temp[load_col], errors="coerce")
                temp["dt"] = pd.to_datetime(
                    temp[date_col].astype(str) + " " + (temp[hour_col].astype("Int64") - 1).astype(str) + ":00",
                    errors="coerce",
                )
                temp = temp.dropna(subset=["dt", load_col]).copy()
                if temp.empty:
                    parsed = None
                else:
                    temp["load"] = temp[load_col]
                    temp["hour"] = temp["dt"].dt.hour
                    temp["dow"] = temp["dt"].dt.dayofweek
                    parsed = temp[["dt", "load", "hour", "dow"]].reset_index(drop=True)

    with _RESIDENTIAL_BASELINE_CACHE_LOCK:
        _RESIDENTIAL_BASELINE_CACHE[csv_path] = (mtime, parsed.copy() if parsed is not None else None)
    return parsed.copy() if parsed is not None else None


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

    df = _load_actual_history_df(HISTORICAL_LOAD_CSV)
    if df is None:
        return None
    df = df.copy()

    if "sector" in df.columns:
        df = df[df["sector"].astype(str).str.lower() == str(sector).lower()]
        if df.empty:
            return None

    if "location_key" in df.columns:
        lk = str(location_key).lower()
        df_loc = df[df["location_key"].astype(str).str.lower() == lk]
        if not df_loc.empty:
            df = df_loc

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
            baseline.append(float(subset["load"].mean()))

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

    temp = _load_residential_baseline_df(csv_path)
    LAST_RESIDENTIAL_BASELINE_PATH = csv_path
    if temp is None:
        return None
    if temp.empty:
        return None

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
            baseline.append(float(subset["load"].mean()))

    if all(pd.isna(v) for v in baseline):
        return None
    return [round(float(v), 2) if not pd.isna(v) else None for v in baseline]  # type: ignore[list-item]


def _load_residential_history_df() -> pd.DataFrame | None:
    """
    Load historical residential series as [dt, load] for lag-feature forecasting.
    """
    candidate_paths: list[str] = []
    env_candidates = (os.getenv("RESIDENTIAL_BASELINE_CSVS") or "").strip()
    if env_candidates:
        candidate_paths.extend([p.strip() for p in env_candidates.split(";") if p.strip()])
    candidate_paths.append(RESIDENTIAL_BASELINE_CSV)

    csv_path = next((p for p in candidate_paths if p and os.path.isfile(p)), None)
    if not csv_path:
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
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
        return None

    temp = df.copy()
    type_col = _find_first_col(temp, ["Type", "type", "Sector", "sector"])
    if type_col is not None:
        temp = temp[temp[type_col].astype(str).str.lower() == "residential"]
    if temp.empty:
        return None

    temp[hour_col] = pd.to_numeric(temp[hour_col], errors="coerce")
    temp[load_col] = pd.to_numeric(temp[load_col], errors="coerce")
    temp["dt"] = pd.to_datetime(
        temp[date_col].astype(str) + " " + (temp[hour_col].astype("Int64") - 1).astype(str) + ":00",
        errors="coerce",
    )
    temp = temp.dropna(subset=["dt", load_col]).sort_values("dt")
    if temp.empty:
        return None

    out = temp[["dt", load_col]].copy()
    out.columns = ["dt", "load"]
    out["load"] = pd.to_numeric(out["load"], errors="coerce")
    out = out.dropna(subset=["load"]).reset_index(drop=True)
    return out if not out.empty else None


# ----------------------------
# ML support: build 168h past window (for LSTM)
# ----------------------------
def build_past_168_window(label: str, lat: float, lon: float, start: datetime) -> list[dict]:
    """
    Build 168 hours of past weather ending at `start` using mock weather for now.
    The residential ML modules derive their own calendar features from:
    temp, humidity, dewpoint, dt.
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
    Prefer the newer 10-feature Keras forecaster, then fall back to the legacy module.
    """
    import_errors: list[str] = []

    for module_name in ("toronto_res_forecast", "toronto_res_model"):
        try:
            if module_name == "toronto_res_forecast":
                from ml import toronto_res_forecast as res_model  # type: ignore
            else:
                from ml import toronto_res_model as res_model  # type: ignore
        except Exception as e:
            import_errors.append(f"{module_name}: {e}")
            continue

        predictor = getattr(res_model, "predict_next_24", None) or getattr(res_model, "predict_next_24h_kwh", None)
        if predictor is None:
            import_errors.append(f"{module_name}: missing predict_next_24/predict_next_24h_kwh")
            continue

        try:
            y = predictor(window_rows)
            return [float(v) for v in y]
        except Exception as e:
            import_errors.append(f"{module_name}: {e}")

    detail = "ML prediction failed."
    if import_errors:
        detail = detail + " " + " | ".join(import_errors)
    raise HTTPException(status_code=500, detail=detail)


def _build_train_dataframe(df: pd.DataFrame):
    """
    Build supervised train frame from uploaded CSV.
    Supports:
      - timestamp-style columns OR Date+Hour columns
      - common load target column names
      - optional weather columns (temp/humidity/dewpoint)
    """
    target_col = _find_first_col(
        df,
        [
            "Total Consumption (kWh)",
            "total_consumption_kwh",
            "consumption_kwh",
            "predicted_load",
            "load",
            "actual_load",
            "demand",
            "kwh",
            "kw",
        ],
    )
    if target_col is None:
        raise ValueError("Could not find load target column (e.g. Total Consumption (kWh), load, kwh).")

    ts_col = _find_first_col(df, ["timestamp", "datetime", "dt", "time"])
    date_col = _find_first_col(df, ["Date", "date"])
    hour_col = _find_first_col(df, ["Hour", "hour", "hour_ending", "he"])

    work = df.copy()
    if ts_col is not None:
        work["dt"] = pd.to_datetime(work[ts_col], errors="coerce")
    elif date_col is not None and hour_col is not None:
        work[hour_col] = pd.to_numeric(work[hour_col], errors="coerce")
        work["dt"] = pd.to_datetime(
            work[date_col].astype(str) + " " + (work[hour_col].astype("Int64") - 1).astype(str) + ":00",
            errors="coerce",
        )
    else:
        raise ValueError("Need timestamp column or Date+Hour columns in training CSV.")

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=["dt", target_col]).sort_values("dt").reset_index(drop=True)
    if len(work) < 96:
        raise ValueError("Need at least 96 clean hourly rows to train.")

    # Time/calendar features
    work["hour"] = work["dt"].dt.hour
    work["dow"] = work["dt"].dt.dayofweek
    work["month"] = work["dt"].dt.month
    work["is_weekend"] = work["dow"].isin([5, 6]).astype(int)
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24.0)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24.0)

    # Lag features from target
    work["lag_1h"] = work[target_col].shift(1)
    work["lag_24h"] = work[target_col].shift(24)

    # Optional weather features if present in same CSV
    temp_col = _find_first_col(work, ["temperature_C", "temp_c", "temp", "temperature"])
    hum_col = _find_first_col(work, ["relative_humidity_pct", "humidity_percent", "humidity", "rh"])
    dew_col = _find_first_col(work, ["dewpoint", "dew_point", "dewpoint_c"])

    feature_cols = ["hour_sin", "hour_cos", "dow", "month", "is_weekend", "lag_1h", "lag_24h"]

    if temp_col is not None:
        work[temp_col] = pd.to_numeric(work[temp_col], errors="coerce")
        work["hdd"] = (18.0 - work[temp_col]).clip(lower=0.0)
        feature_cols += [temp_col, "hdd"]
    if hum_col is not None:
        work[hum_col] = pd.to_numeric(work[hum_col], errors="coerce")
        feature_cols.append(hum_col)
    if dew_col is not None:
        work[dew_col] = pd.to_numeric(work[dew_col], errors="coerce")
        feature_cols.append(dew_col)

    work = work.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    if len(work) < 72:
        raise ValueError("Not enough rows after feature building (need >= 72).")

    X = work[feature_cols].astype(float)
    y = work[target_col].astype(float)
    return work, X, y, feature_cols, target_col


def _predict_residential_with_user_model(
    start: datetime,
    horizon_hours: int,
    label: str,
    lat: float,
    lon: float,
    weather_override: dict | None = None,
    timestamps_override: list[datetime] | None = None,
) -> list[float]:
    """
    Load the user-trained residential model from models/trained/.
    """
    from ml import user_res_forecast  # type: ignore

    window_rows = build_past_168_window(label, lat, lon, start)
    yhat = user_res_forecast.predict_next_24(window_rows)
    return _expand_to_horizon([float(v) for v in yhat], horizon_hours)


def _expand_to_horizon(values: list[float], horizon: int) -> list[float]:
    if horizon <= 0:
        return []
    if not values:
        return [0.0] * horizon
    out = list(values)
    while len(out) < horizon:
        out.extend(values[: max(1, horizon - len(out))])
    return out[:horizon]


def _convert_kwh_list_to_mw(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    out: list[float] = []
    for value in values:
        if value is None:
            out.append(None)  # type: ignore[arg-type]
        else:
            out.append(round(float(value) * KWH_TO_MW, 4))
    return out


def _convert_kw_list_to_mw(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    out: list[float] = []
    for value in values:
        if value is None:
            out.append(None)  # type: ignore[arg-type]
        else:
            out.append(round(float(value) * KW_TO_MW, 4))
    return out


def _normalize_display_unit(payload: dict) -> dict:
    unit = str(payload.get("unit") or "").strip()
    if unit == "kWh":
        payload["predicted_load"] = _convert_kwh_list_to_mw(payload.get("predicted_load"))
        payload["historical_baseline"] = _convert_kwh_list_to_mw(payload.get("historical_baseline"))
        payload["unit"] = "MW"
    elif unit == "kW":
        payload["predicted_load"] = _convert_kw_list_to_mw(payload.get("predicted_load"))
        payload["historical_baseline"] = _convert_kw_list_to_mw(payload.get("historical_baseline"))
        payload["unit"] = "MW"
    elif unit == "MWh":
        payload["unit"] = "MW"
    return payload


def _predict_user_sector_with_weather_payload(
    sector: str,
    weather_payload: dict,
    location: str,
    lat: float | str,
    lon: float | str,
) -> dict:
    from ml.user_sector_model import predict_user_sector_from_weather  # type: ignore

    model_dir = USER_TRAINED_SECTOR_DIRS.get(sector)
    if not model_dir:
        raise ValueError(f"Unsupported user-trained sector '{sector}'.")

    predicted = predict_user_sector_from_weather(weather_payload, model_dir)
    return {
        "module": f"forecast_{sector}_user_ml",
        "sector": sector,
        "unit": predicted["unit"],
        "timestamps": predicted["timestamps"],
        "predicted_load": predicted["predicted_load"],
        "location": location,
        "lat": lat,
        "lon": lon,
        "model_source": predicted["model_source"],
    }


def _user_model_artifact_paths_for_download(sector: str) -> tuple[list[str], str]:
    sector_key = str(sector or "res").strip().lower()
    if sector_key == "res":
        return (
            [
                os.path.join("models", "trained", "userModel.keras"),
                os.path.join("models", "trained", "user_res_scaler_x.save"),
                os.path.join("models", "trained", "user_res_scaler_y.save"),
            ],
            "user_res_training_artifacts.zip",
        )
    if sector_key == "com":
        return (
            [
                os.path.join(TRAINED_COM_MODEL_DIR, "model.joblib"),
                os.path.join(TRAINED_COM_MODEL_DIR, "metadata.json"),
            ],
            "user_com_training_artifacts.zip",
        )
    if sector_key == "ind":
        return (
            [
                os.path.join(TRAINED_IND_MODEL_DIR, "model.joblib"),
                os.path.join(TRAINED_IND_MODEL_DIR, "metadata.json"),
            ],
            "user_ind_training_artifacts.zip",
        )
    raise HTTPException(status_code=400, detail=f"Unsupported training sector '{sector}'.")


def _predict_commercial_24h(location_key: str) -> dict:
    label, lat, lon = resolve_preset(location_key)
    if location_key in {"toronto", "ontario"}:
        from ml import commercial_forecast  # type: ignore

        forecast_df = commercial_forecast.forecast_next_24h_load(
            lat=lat,
            lon=lon,
            model_dir=COMMERCIAL_MODEL_DIR,
            timezone="America/Toronto",
        )

        ts = [pd.Timestamp(v).isoformat(timespec="minutes") for v in forecast_df["forecast_time"].tolist()]
        yhat = [round(float(v), 2) for v in forecast_df["predicted_load_kWh"].tolist()]

        return {
            "module": "forecast_com_ml",
            "sector": "com",
            "unit": "kWh",
            "timestamps": ts,
            "predicted_load": yhat,
            "location": label,
            "lat": lat,
            "lon": lon,
            "model_source": "azure_commercial_cnn",
        }

    from ml import provincial_dual_input_forecast  # type: ignore

    if location_key == "alberta":
        model_dir = ALBERTA_COMMERCIAL_MODEL_DIR
        timezone_name = "America/Edmonton"
        model_source = "azure_alberta_commercial_cnn"
        module_name = "forecast_alberta_commercial_ml"
    elif location_key == "vancouver":
        model_dir = BC_COMMERCIAL_MODEL_DIR
        timezone_name = "America/Vancouver"
        model_source = "azure_bc_commercial_cnn"
        module_name = "forecast_bc_commercial_ml"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported commercial location key '{location_key}'.")

    forecast_df = provincial_dual_input_forecast.forecast_next_24h_load(
        lat=lat,
        lon=lon,
        model_dir=model_dir,
        timezone=timezone_name,
    )
    ts = [pd.Timestamp(v).isoformat(timespec="minutes") for v in forecast_df["forecast_time"].tolist()]
    yhat = [round(float(v), 2) for v in forecast_df["predicted_load"].tolist()]

    return {
        "module": module_name,
        "sector": "com",
        "unit": "MW",
        "timestamps": ts,
        "predicted_load": yhat,
        "location": label,
        "lat": lat,
        "lon": lon,
        "model_source": model_source,
    }


def _predict_alberta_24h(location_key: str, start: datetime) -> dict:
    label, lat, lon = resolve_preset(location_key)
    from ml import alberta_model  # type: ignore

    window_rows = build_past_168_window(label, lat, lon, start)
    yhat = [round(float(v), 2) for v in alberta_model.predict_next_24(window_rows)]
    ts = [(start + timedelta(hours=i + 1)).isoformat(timespec="minutes") for i in range(len(yhat))]

    return {
        "module": "forecast_alberta_ml",
        "sector": "res",
        "unit": "MW",
        "timestamps": ts,
        "predicted_load": yhat,
        "location": label,
        "lat": lat,
        "lon": lon,
        "model_source": "alberta_provincial_lstm",
        "historical_baseline": None,
        "baseline_source": "none",
        "baseline_source_path": None,
    }


def _predict_bc_24h(location_key: str, start: datetime) -> dict:
    label, lat, lon = resolve_preset(location_key)
    from ml import bc_model  # type: ignore

    window_rows = build_past_168_window(label, lat, lon, start)
    yhat = [round(float(v), 2) for v in bc_model.predict_next_24(window_rows)]
    ts = [(start + timedelta(hours=i + 1)).isoformat(timespec="minutes") for i in range(len(yhat))]

    return {
        "module": "forecast_bc_ml",
        "sector": "res",
        "unit": "MW",
        "timestamps": ts,
        "predicted_load": yhat,
        "location": label,
        "lat": lat,
        "lon": lon,
        "model_source": "bc_provincial_lstm",
        "historical_baseline": None,
        "baseline_source": "none",
        "baseline_source_path": None,
    }


def _predict_ontario_industrial_24h(location_key: str) -> dict:
    label, lat, lon = resolve_preset(location_key)
    from ml import ontario_industrial_forecast  # type: ignore

    forecast_df = ontario_industrial_forecast.forecast_next_24h_load(
        lat=lat,
        lon=lon,
        model_dir=ONTARIO_INDUSTRIAL_MODEL_DIR,
        timezone="America/Toronto",
    )

    ts = [pd.Timestamp(v).isoformat(timespec="minutes") for v in forecast_df["forecast_time"].tolist()]
    yhat = [round(float(v), 2) for v in forecast_df["predicted_load_MW"].tolist()]

    return {
        "module": "forecast_ontario_industrial_ml",
        "sector": "ind",
        "unit": "MW",
        "timestamps": ts,
        "predicted_load": yhat,
        "location": label,
        "lat": lat,
        "lon": lon,
        "model_source": "azure_ontario_industrial_cnn",
    }


def _predict_provincial_industrial_24h(location_key: str) -> dict:
    label, lat, lon = resolve_preset(location_key)
    from ml import provincial_dual_input_forecast  # type: ignore

    if location_key == "alberta":
        model_dir = ALBERTA_INDUSTRIAL_MODEL_DIR
        timezone_name = "America/Edmonton"
        module_name = "forecast_alberta_industrial_ml"
        model_source = "azure_alberta_industrial_cnn"
    elif location_key == "vancouver":
        model_dir = BC_INDUSTRIAL_MODEL_DIR
        timezone_name = "America/Vancouver"
        module_name = "forecast_bc_industrial_ml"
        model_source = "azure_bc_industrial_cnn"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported industrial location key '{location_key}'.")

    forecast_df = provincial_dual_input_forecast.forecast_next_24h_load(
        lat=lat,
        lon=lon,
        model_dir=model_dir,
        timezone=timezone_name,
    )
    ts = [pd.Timestamp(v).isoformat(timespec="minutes") for v in forecast_df["forecast_time"].tolist()]
    yhat = [round(float(v), 2) for v in forecast_df["predicted_load"].tolist()]

    return {
        "module": module_name,
        "sector": "ind",
        "unit": "MW",
        "timestamps": ts,
        "predicted_load": yhat,
        "location": label,
        "lat": lat,
        "lon": lon,
        "model_source": model_source,
    }


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


@app.get("/api/debug/try_load_commercial_ml")
def debug_try_load_commercial_ml():
    """
    Attempts to import and run the commercial model loader.
    """
    try:
        from ml import commercial_forecast  # type: ignore

        commercial_forecast._load_artifacts(COMMERCIAL_MODEL_DIR)
        return {"ok": True, "msg": "Imported and loaded ml.commercial_forecast successfully"}
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


@app.post("/api/upload/weather_csv")
async def api_upload_weather_csv(file: UploadFile = File(...)):
    if not str(file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv weather file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded weather CSV is empty.")

    parsed = _parse_uploaded_weather_csv(content)
    file_id = uuid.uuid4().hex
    WEATHER_UPLOAD_STORE[file_id] = parsed

    return {
        "ok": True,
        "file_id": file_id,
        "filename": file.filename,
        "rows": len(parsed["timestamps"]),
        "columns": ["timestamp", "temperature_C", "relative_humidity_pct"],
    }


def _get_uploaded_weather(file_id: str) -> dict:
    payload = WEATHER_UPLOAD_STORE.get(file_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Uploaded weather file_id not found: {file_id}")
    return payload


def _forecast_res_with_weather_payload(weather_payload: dict, location_key: str = "toronto") -> dict:
    label, lat, lon = resolve_preset(location_key)
    ts_all = weather_payload.get("timestamps", [])
    temp_all = weather_payload.get("temperature_C", [])
    hum_all = weather_payload.get("relative_humidity_pct", [])
    n = min(24, len(ts_all), len(temp_all), len(hum_all))
    if n <= 0:
        raise HTTPException(status_code=400, detail="Uploaded weather payload has no usable rows.")

    ts = ts_all[:n]
    temp = temp_all[:n]
    hum = hum_all[:n]
    ts_dt = [datetime.fromisoformat(str(t)) for t in ts]
    start = ts_dt[0] - timedelta(hours=1)

    model_source = "user_trained_rf"
    try:
        yhat = _predict_residential_with_user_model(
            start,
            n,
            label,
            lat,
            lon,
            weather_override={"temperature_C": temp, "relative_humidity_pct": hum},
            timestamps_override=ts_dt,
        )
    except Exception:
        try:
            window_rows = build_past_168_window(label, lat, lon, start)
            yhat = _ml_predict_residential_24h(window_rows)[:n]
            model_source = "default_res_lstm"
        except Exception:
            fallback = mock_forecast(
                "res",
                {"timestamps": ts, "temperature_C": temp, "relative_humidity_pct": hum},
            )
            yhat = [float(v) for v in fallback["predicted_load"][:n]]
            model_source = "res_weather_rule_fallback"

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
        "model_source": model_source,
    }
    if out["historical_baseline"] is None:
        out["historical_baseline"] = _historical_baseline_from_actual_csv(location_key, "res", ts)
        out["baseline_source"] = "actual_history_csv"
        out["baseline_source_path"] = HISTORICAL_LOAD_CSV if out["historical_baseline"] is not None else None
    if out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        out["historical_baseline"] = _historical_baseline(location_key, "res", n)
        out["baseline_source"] = "prediction_history_fallback"
        out["baseline_source_path"] = None
    if out["historical_baseline"] is None:
        out["baseline_source"] = "none"
        out["baseline_source_path"] = None

    _update_history(location_key, "res", out["predicted_load"])
    return _normalize_display_unit(out)


@app.post("/api/forecast/res_from_upload/{file_id}")
def api_forecast_res_from_upload(file_id: str):
    weather_payload = _get_uploaded_weather(file_id)
    out = _forecast_res_with_weather_payload(weather_payload, location_key="toronto")
    out["input_source"] = "uploaded_csv"
    return out


@app.post("/api/forecast/com_from_upload/{file_id}")
def api_forecast_com_from_upload(file_id: str):
    weather_payload = _get_uploaded_weather(file_id)
    n = min(24, len(weather_payload["timestamps"]), len(weather_payload["temperature_C"]))
    w = {
        "timestamps": weather_payload["timestamps"][:n],
        "temperature_C": weather_payload["temperature_C"][:n],
        "relative_humidity_pct": weather_payload["relative_humidity_pct"][:n],
        "dew_point_C": (weather_payload.get("dew_point_C") or [])[:n],
    }
    try:
        out = _predict_user_sector_with_weather_payload("com", w, "Uploaded weather CSV", "", "")
    except Exception:
        out = mock_forecast("com", w)
        out["location"] = "Uploaded weather CSV"
        out["lat"] = ""
        out["lon"] = ""
    out["baseline_source"] = "none"
    out["historical_baseline"] = None
    out["input_source"] = "uploaded_csv"
    return _normalize_display_unit(out)


@app.post("/api/forecast/ind_from_upload/{file_id}")
def api_forecast_ind_from_upload(file_id: str):
    weather_payload = _get_uploaded_weather(file_id)
    n = min(24, len(weather_payload["timestamps"]), len(weather_payload["temperature_C"]))
    w = {
        "timestamps": weather_payload["timestamps"][:n],
        "temperature_C": weather_payload["temperature_C"][:n],
        "relative_humidity_pct": weather_payload["relative_humidity_pct"][:n],
        "dew_point_C": (weather_payload.get("dew_point_C") or [])[:n],
    }
    try:
        out = _predict_user_sector_with_weather_payload("ind", w, "Uploaded weather CSV", "", "")
    except Exception:
        out = mock_forecast("ind", w)
        out["location"] = "Uploaded weather CSV"
        out["lat"] = ""
        out["lon"] = ""
    out["baseline_source"] = "none"
    out["historical_baseline"] = None
    out["input_source"] = "uploaded_csv"
    return _normalize_display_unit(out)


@app.post("/api/run_all_from_upload/{file_id}")
def api_run_all_from_upload(file_id: str):
    weather_payload = _get_uploaded_weather(file_id)
    n = min(24, len(weather_payload["timestamps"]), len(weather_payload["temperature_C"]))
    w = {
        "timestamps": weather_payload["timestamps"][:n],
        "temperature_C": weather_payload["temperature_C"][:n],
        "relative_humidity_pct": weather_payload["relative_humidity_pct"][:n],
        "dew_point_C": (weather_payload.get("dew_point_C") or [])[:n],
    }
    res_out = _forecast_res_with_weather_payload(w, location_key="toronto")
    try:
        com_out = _normalize_display_unit(_predict_user_sector_with_weather_payload("com", w, "Uploaded weather CSV", "", ""))
    except Exception:
        com_out = _normalize_display_unit(mock_forecast("com", w))
    try:
        ind_out = _normalize_display_unit(_predict_user_sector_with_weather_payload("ind", w, "Uploaded weather CSV", "", ""))
    except Exception:
        ind_out = _normalize_display_unit(mock_forecast("ind", w))
    com_out["historical_baseline"] = None
    ind_out["historical_baseline"] = None

    return {
        "module": "pipeline_all_upload",
        "residential": {"forecast": res_out, "weather": w},
        "commercial": {"forecast": com_out, "weather": w},
        "industrial": {"forecast": ind_out, "weather": w},
    }


@app.post("/api/forecast/res")
def api_forecast_res(req: ModelRequest):
    """
    Residential ML endpoint (24-hour forecast) using your LSTM + scalers.
    """
    start = _parse_start(req.start_iso)
    label, lat, lon = resolve_preset(req.location_key)

    horizon = 24

    if req.location_key == "alberta":
        out = _predict_alberta_24h(req.location_key, start)
        out["input_source"] = "live_api_or_fallback_weather"
    elif req.location_key == "vancouver":
        out = _predict_bc_24h(req.location_key, start)
        out["input_source"] = "live_api_or_fallback_weather"
    else:
        # Toronto uses the trained Toronto/user residential path.
        model_source = "user_trained_rf"
        try:
            yhat = _predict_residential_with_user_model(start, horizon, label, lat, lon)
        except Exception:
            window_rows = build_past_168_window(label, lat, lon, start)
            yhat = _expand_to_horizon(_ml_predict_residential_24h(window_rows), horizon)
            model_source = "default_res_lstm"

        ts = [(start + timedelta(hours=i + 1)).isoformat(timespec="minutes") for i in range(horizon)]
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
            "model_source": model_source,
            "input_source": "live_api_or_fallback_weather",
        }

    ts = out["timestamps"]

    if out["historical_baseline"] is None:
        out["historical_baseline"] = _historical_baseline_from_actual_csv(req.location_key, "res", ts)
        out["baseline_source"] = "actual_history_csv"
        out["baseline_source_path"] = HISTORICAL_LOAD_CSV if out["historical_baseline"] is not None else None
    if out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        out["historical_baseline"] = _historical_baseline(req.location_key, "res", horizon)
        out["baseline_source"] = "prediction_history_fallback"
        out["baseline_source_path"] = None
    if out["historical_baseline"] is None:
        out["baseline_source"] = "none"
        out["baseline_source_path"] = None

    # store history (use 24 horizon for ML)
    _update_history(req.location_key, "res", out["predicted_load"])
    return _normalize_display_unit(out)


@app.post("/api/forecast/com")
def api_forecast_com(req: ModelRequest):
    """
    Commercial forecast using the Azure CNN weather model when available.
    """
    sector = "com"
    start = _parse_start(req.start_iso)
    label, lat, lon = resolve_preset(req.location_key)
    try:
        w = mock_weather(label, lat, lon, start, req.horizon_hours)
        out = _predict_user_sector_with_weather_payload(sector, w, label, lat, lon)
        out["input_source"] = "live_api_or_fallback_weather"
    except Exception:
        try:
            out = _predict_commercial_24h(req.location_key)
            out["input_source"] = "model_weather_pipeline"
        except Exception as e:
            w = mock_weather(label, lat, lon, start, req.horizon_hours)
            out = mock_forecast(sector, w)
            out["location"] = label
            out["lat"] = lat
            out["lon"] = lon
            out["model_source"] = "mock_fallback"
            out["model_error"] = str(e)
            out["input_source"] = "live_api_or_fallback_weather"

    baseline = _historical_baseline_from_actual_csv(req.location_key, sector, out["timestamps"])
    baseline_source = "actual_history_csv"
    horizon = len(out["predicted_load"])
    if baseline is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        baseline = _historical_baseline(req.location_key, sector, horizon)
        baseline_source = "prediction_history_fallback"
    if baseline is None:
        baseline_source = "none"
    _update_history(req.location_key, sector, out["predicted_load"])

    out["historical_baseline"] = baseline
    out["baseline_source"] = baseline_source
    return _normalize_display_unit(out)


@app.post("/api/forecast/ind")
def api_forecast_ind(req: ModelRequest):
    """
    Industrial: uses model-backed forecasts for Ontario, Alberta, and British Columbia,
    with mock fallback for other locations or load failures.
    """
    start = _parse_start(req.start_iso)
    sector = "ind"
    label, lat, lon = resolve_preset(req.location_key)
    try:
        weather_payload = mock_weather(label, lat, lon, start, req.horizon_hours)
        out = _predict_user_sector_with_weather_payload(sector, weather_payload, label, lat, lon)
        out["input_source"] = "live_api_or_fallback_weather"
    except Exception:
        if req.location_key == "ontario":
            try:
                out = _predict_ontario_industrial_24h(req.location_key)
                out["input_source"] = "model_weather_pipeline"
            except Exception as e:
                w = mock_weather(label, lat, lon, start, req.horizon_hours)
                out = mock_forecast(sector, w)
                out["location"] = label
                out["lat"] = lat
                out["lon"] = lon
                out["model_source"] = "mock_fallback"
                out["model_error"] = str(e)
                out["input_source"] = "live_api_or_fallback_weather"
        elif req.location_key == "alberta":
            try:
                out = _predict_provincial_industrial_24h(req.location_key)
                out["input_source"] = "model_weather_pipeline"
            except Exception as e:
                w = mock_weather(label, lat, lon, start, req.horizon_hours)
                out = mock_forecast(sector, w)
                out["location"] = label
                out["lat"] = lat
                out["lon"] = lon
                out["model_source"] = "mock_fallback"
                out["model_error"] = str(e)
                out["input_source"] = "live_api_or_fallback_weather"
        elif req.location_key == "vancouver":
            try:
                out = _predict_provincial_industrial_24h(req.location_key)
                out["input_source"] = "model_weather_pipeline"
            except Exception as e:
                w = mock_weather(label, lat, lon, start, req.horizon_hours)
                out = mock_forecast(sector, w)
                out["location"] = label
                out["lat"] = lat
                out["lon"] = lon
                out["model_source"] = "mock_fallback"
                out["model_error"] = str(e)
                out["input_source"] = "live_api_or_fallback_weather"
        else:
            w = mock_weather(label, lat, lon, start, req.horizon_hours)
            out = mock_forecast(sector, w)
            out["location"] = label
            out["lat"] = lat
            out["lon"] = lon
            out["input_source"] = "live_api_or_fallback_weather"

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
    return _normalize_display_unit(out)


@app.post("/api/run_all")
def api_run_all(req: AllRequest):
    """
    Runs Residential (ML) + Commercial + Industrial (mock).
    Each model uses its own location dropdown selection.
    """
    start = _parse_start(req.start_iso)

    # ---- Residential ----
    horizon = 24
    res_label, res_lat, res_lon = resolve_preset(req.res_location_key)
    if req.res_location_key == "alberta":
        res_out = _predict_alberta_24h(req.res_location_key, start)
    elif req.res_location_key == "vancouver":
        res_out = _predict_bc_24h(req.res_location_key, start)
    else:
        res_model_source = "user_trained_rf"
        try:
            res_yhat = _predict_residential_with_user_model(start, horizon, res_label, res_lat, res_lon)
        except Exception:
            res_window = build_past_168_window(res_label, res_lat, res_lon, start)
            res_yhat = _expand_to_horizon(_ml_predict_residential_24h(res_window), horizon)
            res_model_source = "default_res_lstm"
        res_ts = [(start + timedelta(hours=i + 1)).isoformat(timespec="minutes") for i in range(horizon)]

        res_out = {
            "module": "forecast_res_ml",
            "sector": "res",
            "unit": "kWh",
            "timestamps": res_ts,
            "predicted_load": [round(float(v), 2) for v in res_yhat],
            "historical_baseline": _historical_baseline_from_residential_csv(res_ts),
            "baseline_source": "residential_csv",
            "baseline_source_path": LAST_RESIDENTIAL_BASELINE_PATH,
            "model_source": res_model_source,
        }
    res_ts = res_out["timestamps"]
    if res_out["historical_baseline"] is None:
        res_out["historical_baseline"] = _historical_baseline_from_actual_csv(req.res_location_key, "res", res_ts)
        res_out["baseline_source"] = "actual_history_csv"
        res_out["baseline_source_path"] = HISTORICAL_LOAD_CSV if res_out["historical_baseline"] is not None else None
    if res_out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        res_out["historical_baseline"] = _historical_baseline(req.res_location_key, "res", horizon)
        res_out["baseline_source"] = "prediction_history_fallback"
        res_out["baseline_source_path"] = None
    if res_out["historical_baseline"] is None:
        res_out["baseline_source"] = "none"
        res_out["baseline_source_path"] = None
    _update_history(req.res_location_key, "res", res_out["predicted_load"])
    res_out = _normalize_display_unit(res_out)

    res_weather = mock_weather(res_label, res_lat, res_lon, start, horizon)

    # ---- Commercial ----
    com_label, com_lat, com_lon = resolve_preset(req.com_location_key)
    com_weather = mock_weather(com_label, com_lat, com_lon, start, horizon)
    try:
        com_out = _predict_user_sector_with_weather_payload("com", com_weather, com_label, com_lat, com_lon)
        com_out["input_source"] = "live_api_or_fallback_weather"
    except Exception:
        try:
            com_out = _predict_commercial_24h(req.com_location_key)
            com_weather = {
                "module": "weather_model_driven",
                "location": com_label,
                "lat": com_lat,
                "lon": com_lon,
                "timestamps": com_out["timestamps"],
                "temperature_C": [],
                "relative_humidity_pct": [],
            }
            com_out["input_source"] = "model_weather_pipeline"
        except Exception as e:
            com_out = mock_forecast("com", com_weather)
            com_out["location"] = com_label
            com_out["lat"] = com_lat
            com_out["lon"] = com_lon
            com_out["model_source"] = "mock_fallback"
            com_out["model_error"] = str(e)
            com_out["input_source"] = "live_api_or_fallback_weather"
    com_out["historical_baseline"] = _historical_baseline_from_actual_csv(
        req.com_location_key, "com", com_out["timestamps"]
    )
    com_out["baseline_source"] = "actual_history_csv"
    if com_out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        com_out["historical_baseline"] = _historical_baseline(req.com_location_key, "com", horizon)
        com_out["baseline_source"] = "prediction_history_fallback"
    if com_out["historical_baseline"] is None:
        com_out["baseline_source"] = "none"
    _update_history(req.com_location_key, "com", com_out["predicted_load"])
    com_out = _normalize_display_unit(com_out)

    # ---- Industrial ----
    ind_label, ind_lat, ind_lon = resolve_preset(req.ind_location_key)
    ind_weather = mock_weather(ind_label, ind_lat, ind_lon, start, horizon)
    try:
        ind_out = _predict_user_sector_with_weather_payload("ind", ind_weather, ind_label, ind_lat, ind_lon)
        ind_out["input_source"] = "live_api_or_fallback_weather"
    except Exception:
        if req.ind_location_key == "ontario":
            try:
                ind_out = _predict_ontario_industrial_24h(req.ind_location_key)
                ind_weather = {
                    "module": "weather_model_driven",
                    "location": ind_label,
                    "lat": ind_lat,
                    "lon": ind_lon,
                    "timestamps": ind_out["timestamps"],
                    "temperature_C": [],
                    "relative_humidity_pct": [],
                }
                ind_out["input_source"] = "model_weather_pipeline"
            except Exception as e:
                ind_out = mock_forecast("ind", ind_weather)
                ind_out["location"] = ind_label
                ind_out["lat"] = ind_lat
                ind_out["lon"] = ind_lon
                ind_out["model_source"] = "mock_fallback"
                ind_out["model_error"] = str(e)
                ind_out["input_source"] = "live_api_or_fallback_weather"
        elif req.ind_location_key == "alberta":
            try:
                ind_out = _predict_provincial_industrial_24h(req.ind_location_key)
                ind_weather = {
                    "module": "weather_model_driven",
                    "location": ind_label,
                    "lat": ind_lat,
                    "lon": ind_lon,
                    "timestamps": ind_out["timestamps"],
                    "temperature_C": [],
                    "relative_humidity_pct": [],
                }
                ind_out["input_source"] = "model_weather_pipeline"
            except Exception as e:
                ind_out = mock_forecast("ind", ind_weather)
                ind_out["location"] = ind_label
                ind_out["lat"] = ind_lat
                ind_out["lon"] = ind_lon
                ind_out["model_source"] = "mock_fallback"
                ind_out["model_error"] = str(e)
                ind_out["input_source"] = "live_api_or_fallback_weather"
        elif req.ind_location_key == "vancouver":
            try:
                ind_out = _predict_provincial_industrial_24h(req.ind_location_key)
                ind_weather = {
                    "module": "weather_model_driven",
                    "location": ind_label,
                    "lat": ind_lat,
                    "lon": ind_lon,
                    "timestamps": ind_out["timestamps"],
                    "temperature_C": [],
                    "relative_humidity_pct": [],
                }
                ind_out["input_source"] = "model_weather_pipeline"
            except Exception as e:
                ind_out = mock_forecast("ind", ind_weather)
                ind_out["location"] = ind_label
                ind_out["lat"] = ind_lat
                ind_out["lon"] = ind_lon
                ind_out["model_source"] = "mock_fallback"
                ind_out["model_error"] = str(e)
                ind_out["input_source"] = "live_api_or_fallback_weather"
        else:
            ind_out = mock_forecast("ind", ind_weather)
            ind_out["location"] = ind_label
            ind_out["lat"] = ind_lat
            ind_out["lon"] = ind_lon
            ind_out["input_source"] = "live_api_or_fallback_weather"
    ind_out["historical_baseline"] = _historical_baseline_from_actual_csv(
        req.ind_location_key, "ind", ind_out["timestamps"]
    )
    ind_out["baseline_source"] = "actual_history_csv"
    if ind_out["historical_baseline"] is None and USE_PRED_HISTORY_BASELINE_FALLBACK:
        ind_out["historical_baseline"] = _historical_baseline(req.ind_location_key, "ind", horizon)
        ind_out["baseline_source"] = "prediction_history_fallback"
    if ind_out["historical_baseline"] is None:
        ind_out["baseline_source"] = "none"
    _update_history(req.ind_location_key, "ind", ind_out["predicted_load"])
    ind_out = _normalize_display_unit(ind_out)

    return {
        "module": "pipeline_all",
        "start": start.isoformat(timespec="minutes"),
        "horizon_hours": horizon,
        "residential": {"forecast": res_out, "weather": res_weather},
        "commercial": {"forecast": com_out, "weather": com_weather},
        "industrial": {"forecast": ind_out, "weather": ind_weather},
    }


@app.post("/api/train")
async def api_train(
    file: UploadFile | None = File(default=None),
    weather_file: UploadFile | None = File(default=None),
    expect_rows: int | None = Form(default=None),
    auto_trim_to_expected: bool = Form(default=True),
    notes: str | None = Form(default=None),
):
    """
    Train the active residential Keras model from uploaded data and return metrics.
    This updates the artifacts used by the website forecast flow.
    """
    notes_text = (notes or "").strip()

    if file is None:
        raise HTTPException(status_code=400, detail="Please upload a training CSV file.")

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

    weather_merge_stats: dict | None = None
    weather_filename: str | None = None
    parsed_weather_df: pd.DataFrame | None = None
    if weather_file is not None:
        if weather_file.filename and not str(weather_file.filename).lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Weather training file must be a .csv")

        try:
            weather_content = await weather_file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to read weather training file: {e}")
        if not weather_content:
            raise HTTPException(status_code=400, detail="Uploaded weather training CSV is empty.")

        try:
            wdf = _parse_weather_training_df(weather_content)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Weather training CSV parse failed: {e}")

        if wdf.empty:
            raise HTTPException(status_code=400, detail="Weather training CSV has no valid rows.")
        parsed_weather_df = wdf
        weather_filename = weather_file.filename

    combined_df, weather_merge_stats = _build_combined_training_dataframe(df, parsed_weather_df)
    if weather_merge_stats is not None:
        weather_merge_stats["weather_filename"] = weather_filename

    combined_dir = os.path.join("data", "combined_training")
    os.makedirs(combined_dir, exist_ok=True)
    combined_name = f"{_safe_filename_part(file.filename or 'training')}_combined.csv"
    combined_path = os.path.join(combined_dir, combined_name)
    combined_df.to_csv(combined_path, index=False)

    df = combined_df
    input_rows = int(len(df))
    expected = int(expect_rows) if expect_rows is not None else None
    if expected is not None and input_rows != expected:
        if auto_trim_to_expected and input_rows > expected:
            try:
                temp_for_trim = df.copy()
                temp_for_trim["__dt"] = pd.to_datetime(temp_for_trim["datetime"], errors="coerce")
                temp_for_trim = temp_for_trim.sort_values("__dt").reset_index(drop=True)
                df = temp_for_trim.tail(expected).drop(columns=["__dt"], errors="ignore").reset_index(drop=True)
                input_rows = int(len(df))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to auto-trim training CSV to {expected} rows: {e}",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Training CSV must contain exactly {expected} rows after merge/cleaning. Found {input_rows}.",
            )

    try:
        from ml.user_res_trainer import train_user_lstm  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to load Keras trainer: {e}")

    try:
        result = train_user_lstm(
            df=df,
            output_dir="models",
            source_name=file.filename or "uploaded.csv",
            notes=notes_text,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")

    result["input_rows"] = input_rows
    result["weather_merge"] = weather_merge_stats
    result["combined_training_csv"] = combined_path
    result["combined_columns"] = [str(c) for c in df.columns]
    return result


@app.get("/api/train/download_artifacts")
def api_train_download_artifacts():
    artifact_paths = [
        os.path.join("models", "trained", "userModel.keras"),
        os.path.join("models", "trained", "user_res_scaler_x.save"),
        os.path.join("models", "trained", "user_res_scaler_y.save"),
    ]
    missing = [p for p in artifact_paths if not os.path.isfile(p)]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing trained artifact files: {missing}")

    zip_path = os.path.join(tempfile.gettempdir(), f"user_res_training_artifacts_{uuid.uuid4().hex}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in artifact_paths:
            zf.write(path, arcname=os.path.basename(path))

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="user_res_training_artifacts.zip",
    )

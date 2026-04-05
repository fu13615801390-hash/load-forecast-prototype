from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

TEMP_COL = "Temp (\N{DEGREE SIGN}C)"
HUMIDITY_COL = "Rel Hum (%)"
DEWPOINT_COL = "Dew Point Temp (\N{DEGREE SIGN}C)"
WIND_COL = "Wind Spd (km/h)"
WEATHER_HTTP_TIMEOUT = (
    float(os.getenv("WEATHER_HTTP_CONNECT_TIMEOUT_SECONDS", "15")),
    float(os.getenv("WEATHER_HTTP_READ_TIMEOUT_SECONDS", "180")),
)

_ARTIFACT_CACHE: dict[str, tuple[object, object, object, dict]] = {}
_LOAD_MODEL = None


def build_time_features(df: pd.DataFrame, base_temp: float = 18.0) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["HDD"] = np.maximum(0, base_temp - df[TEMP_COL])
    df["CDD"] = np.maximum(0, df[TEMP_COL] - base_temp)
    return df


def _resolve_model_loader():
    global _LOAD_MODEL
    if _LOAD_MODEL is not None:
        return _LOAD_MODEL

    try:
        from tensorflow.keras.models import load_model as tf_load_model  # type: ignore

        _LOAD_MODEL = tf_load_model
        return _LOAD_MODEL
    except Exception:
        pass

    try:
        from keras.models import load_model as keras_load_model  # type: ignore

        _LOAD_MODEL = keras_load_model
        return _LOAD_MODEL
    except Exception as e:
        raise ImportError("No working Keras loader found for the provincial dual-input forecast model.") from e


def _load_artifacts(model_dir: str | os.PathLike[str]):
    model_dir = Path(model_dir).resolve()
    cache_key = str(model_dir)
    cached = _ARTIFACT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model_candidates = [
        model_dir / "cnn_weather_only_dual_input.keras",
        model_dir / "best_model.keras",
    ]
    model_path = next((p for p in model_candidates if p.is_file()), model_candidates[0])
    feature_scaler_path = model_dir / "feature_scaler.pkl"
    target_scaler_path = model_dir / "target_scaler.pkl"
    config_path = model_dir / "config.json"

    missing = [str(p) for p in (model_path, feature_scaler_path, target_scaler_path, config_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing provincial dual-input model artifacts: {missing}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_loader = _resolve_model_loader()
    model = model_loader(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    artifacts = (model, feature_scaler, target_scaler, config)
    _ARTIFACT_CACHE[cache_key] = artifacts
    return artifacts


def fetch_weather(
    lat: float,
    lon: float,
    feature_cols: list[str],
    timezone: str,
    past_hours: int,
    future_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_fields = ["temperature_2m", "relative_humidity_2m"]
    if DEWPOINT_COL in feature_cols:
        hourly_fields.append("dew_point_2m")
    if WIND_COL in feature_cols:
        hourly_fields.append("wind_speed_10m")

    params: dict[str, object] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_fields),
        "past_hours": past_hours + 6,
        "forecast_hours": future_hours + 6,
        "timezone": timezone,
    }
    if "wind_speed_10m" in hourly_fields:
        params["wind_speed_unit"] = "kmh"

    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=WEATHER_HTTP_TIMEOUT)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    data = {
        "timestamp": pd.to_datetime(hourly["time"]),
        TEMP_COL: hourly["temperature_2m"],
        HUMIDITY_COL: hourly["relative_humidity_2m"],
    }
    if "dew_point_2m" in hourly:
        data[DEWPOINT_COL] = hourly["dew_point_2m"]
    if "wind_speed_10m" in hourly:
        data[WIND_COL] = hourly["wind_speed_10m"]

    df = pd.DataFrame(data).dropna()

    now_local = pd.Timestamp.now(tz=timezone)
    current_hour = now_local.floor("h").tz_localize(None)
    last_completed_hour = current_hour - pd.Timedelta(hours=1)
    first_forecast_hour = current_hour + pd.Timedelta(hours=1)

    past_start = last_completed_hour - pd.Timedelta(hours=past_hours - 1)
    future_end = first_forecast_hour + pd.Timedelta(hours=future_hours - 1)

    past = df[(df["timestamp"] >= past_start) & (df["timestamp"] <= last_completed_hour)].copy()
    future = df[(df["timestamp"] >= first_forecast_hour) & (df["timestamp"] <= future_end)].copy()
    if len(past) != past_hours or len(future) != future_hours:
        raise ValueError(f"Unexpected weather window sizes. Got past={len(past)}, future={len(future)}")

    return past.reset_index(drop=True), future.reset_index(drop=True)


def forecast_next_24h_load(
    lat: float,
    lon: float,
    model_dir: str | os.PathLike[str],
    timezone: str,
) -> pd.DataFrame:
    model, feature_scaler, target_scaler, config = _load_artifacts(model_dir)
    feature_cols = config["feature_cols"]
    past_hours = int(config.get("past_hours", 72))
    future_hours = int(config.get("future_hours", 24))
    base_temp = float(config.get("base_temp_c", 18.0))
    model_timezone = str(config.get("model_calendar_timezone") or timezone)

    past, future = fetch_weather(
        lat=lat,
        lon=lon,
        feature_cols=feature_cols,
        timezone=model_timezone,
        past_hours=past_hours,
        future_hours=future_hours,
    )
    past = build_time_features(past, base_temp=base_temp)
    future = build_time_features(future, base_temp=base_temp)

    x_past = past[feature_cols].to_numpy(dtype=float).reshape(1, past_hours, len(feature_cols))
    x_future = future[feature_cols].to_numpy(dtype=float).reshape(1, future_hours, len(feature_cols))

    x_past = feature_scaler.transform(x_past.reshape(-1, len(feature_cols))).reshape(1, past_hours, len(feature_cols))
    x_future = feature_scaler.transform(x_future.reshape(-1, len(feature_cols))).reshape(1, future_hours, len(feature_cols))

    y_pred = model.predict([x_past, x_future], verbose=0)
    y_pred = target_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).reshape(future_hours)

    return pd.DataFrame(
        {
            "forecast_time": future["timestamp"],
            "predicted_load": y_pred,
        }
    )

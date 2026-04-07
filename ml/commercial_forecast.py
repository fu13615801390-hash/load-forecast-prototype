from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta
import copy

import joblib
import numpy as np
import pandas as pd
import requests

TEMP_COL = "Temp (\N{DEGREE SIGN}C)"
HUMIDITY_COL = "Rel Hum (%)"
DEWPOINT_COL = "Dew Point Temp (\N{DEGREE SIGN}C)"
WEATHER_HTTP_TIMEOUT = (
    float(os.getenv("WEATHER_HTTP_CONNECT_TIMEOUT_SECONDS", "15")),
    float(os.getenv("WEATHER_HTTP_READ_TIMEOUT_SECONDS", "180")),
)

_MODEL = None
_FEATURE_SCALER = None
_TARGET_SCALER = None
_CONFIG = None
_LOAD_MODEL = None
_ARTIFACT_LOCK = Lock()
_WEATHER_CACHE: dict[tuple, tuple[datetime, tuple[pd.DataFrame, pd.DataFrame]]] = {}
_WEATHER_CACHE_LOCK = Lock()
_HTTP_SESSION = requests.Session()
WEATHER_CACHE_TTL_SECONDS = int(os.getenv("MODEL_WEATHER_CACHE_TTL_SECONDS", "300"))


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
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

    base_temp = 18.0
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
        raise ImportError("No working Keras loader found for the commercial model.") from e


def _load_artifacts(model_dir: str | os.PathLike[str]):
    global _MODEL, _FEATURE_SCALER, _TARGET_SCALER, _CONFIG
    if _MODEL is not None and _FEATURE_SCALER is not None and _TARGET_SCALER is not None and _CONFIG is not None:
        return _MODEL, _FEATURE_SCALER, _TARGET_SCALER, _CONFIG

    with _ARTIFACT_LOCK:
        if _MODEL is not None and _FEATURE_SCALER is not None and _TARGET_SCALER is not None and _CONFIG is not None:
            return _MODEL, _FEATURE_SCALER, _TARGET_SCALER, _CONFIG

        model_dir = Path(model_dir)
        model_path = model_dir / "cnn_weather_only_dual_input.keras"
        feature_scaler_path = model_dir / "feature_scaler.pkl"
        target_scaler_path = model_dir / "target_scaler.pkl"
        config_path = model_dir / "config.json"

        missing = [str(p) for p in (model_path, feature_scaler_path, target_scaler_path, config_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing commercial model artifacts: {missing}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        model_loader = _resolve_model_loader()
        model = model_loader(model_path)
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)

        _MODEL = model
        _FEATURE_SCALER = feature_scaler
        _TARGET_SCALER = target_scaler
        _CONFIG = config
    return _MODEL, _FEATURE_SCALER, _TARGET_SCALER, _CONFIG


def _get_cached_weather(cache_key: tuple):
    now = datetime.now()
    with _WEATHER_CACHE_LOCK:
        cached = _WEATHER_CACHE.get(cache_key)
        if cached is None:
            return None
        expires_at, payload = cached
        if expires_at <= now:
            _WEATHER_CACHE.pop(cache_key, None)
            return None
        past, future = payload
        return past.copy(), future.copy()


def _set_cached_weather(cache_key: tuple, past: pd.DataFrame, future: pd.DataFrame):
    expires_at = datetime.now() + timedelta(seconds=max(1, WEATHER_CACHE_TTL_SECONDS))
    with _WEATHER_CACHE_LOCK:
        _WEATHER_CACHE[cache_key] = (expires_at, (past.copy(), future.copy()))


def fetch_weather(lat: float, lon: float, timezone: str = "America/Toronto") -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_key = ("commercial", round(float(lat), 4), round(float(lon), 4), timezone)
    cached = _get_cached_weather(cache_key)
    if cached is not None:
        return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m",
        "past_hours": 72,
        "forecast_hours": 24,
        "timezone": timezone,
    }

    response = _HTTP_SESSION.get(url, params=params, timeout=WEATHER_HTTP_TIMEOUT)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            TEMP_COL: hourly["temperature_2m"],
            HUMIDITY_COL: hourly["relative_humidity_2m"],
            DEWPOINT_COL: hourly["dew_point_2m"],
        }
    ).dropna()

    past = df.iloc[:72].copy()
    future = df.iloc[72:96].copy()
    if len(past) != 72 or len(future) != 24:
        raise ValueError(f"Unexpected weather window sizes. Got past={len(past)}, future={len(future)}")

    _set_cached_weather(cache_key, past, future)
    return past, future


def forecast_next_24h_load(
    lat: float,
    lon: float,
    model_dir: str | os.PathLike[str],
    timezone: str = "America/Toronto",
) -> pd.DataFrame:
    model, feature_scaler, target_scaler, config = _load_artifacts(model_dir)
    feature_cols = config["feature_cols"]

    past, future = fetch_weather(lat, lon, timezone=timezone)
    past = build_time_features(past)
    future = build_time_features(future)

    X_past = past[feature_cols].to_numpy(dtype=float).reshape(1, 72, len(feature_cols))
    X_future = future[feature_cols].to_numpy(dtype=float).reshape(1, 24, len(feature_cols))

    X_past = feature_scaler.transform(X_past.reshape(-1, len(feature_cols))).reshape(1, 72, len(feature_cols))
    X_future = feature_scaler.transform(X_future.reshape(-1, len(feature_cols))).reshape(1, 24, len(feature_cols))

    y_pred = model.predict([X_past, X_future], verbose=0)
    y_pred = target_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).reshape(24)

    return pd.DataFrame(
        {
            "forecast_time": future["timestamp"],
            "predicted_load_kWh": y_pred,
            "predicted_load_MW": y_pred / 1000.0,
        }
    )

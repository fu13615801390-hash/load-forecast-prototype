from __future__ import annotations

import os

import joblib
import numpy as np

WINDOW_HOURS = 168
FEATURE_COLS = [
    "temp",
    "humidity",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "HDD",
    "CDD",
    "month_sin",
    "month_cos",
]

_model = None
_scaler_x = None
_scaler_y = None
_load_model = None


def _paths():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models", "british_columbia")
    return {
        "models_dir": models_dir,
        "model": os.path.join(models_dir, "BC_test.keras"),
        "sx": os.path.join(models_dir, "scaler_BC_x.save"),
        "sy": os.path.join(models_dir, "scaler_BC_y.save"),
    }


def _resolve_model_loader():
    global _load_model
    if _load_model is not None:
        return _load_model

    try:
        from tensorflow.keras.models import load_model as tf_load_model  # type: ignore

        _load_model = tf_load_model
        return _load_model
    except Exception:
        pass

    try:
        from keras.models import load_model as keras_load_model  # type: ignore

        _load_model = keras_load_model
        return _load_model
    except Exception as e:
        raise ImportError("No working Keras loader found for the British Columbia model.") from e


def load_artifacts():
    global _model, _scaler_x, _scaler_y
    if _model is not None and _scaler_x is not None and _scaler_y is not None:
        return

    p = _paths()
    missing = [name for name, path in {"model": p["model"], "sx": p["sx"], "sy": p["sy"]}.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing British Columbia model files in {p['models_dir']}: {missing}")

    model_loader = _resolve_model_loader()
    _model = model_loader(p["model"])
    _scaler_x = joblib.load(p["sx"])
    _scaler_y = joblib.load(p["sy"])


def _feature_row(row: dict) -> list[float]:
    dt = row["dt"]
    temp = float(row["temp"])
    humidity = float(row["humidity"])

    return [
        temp,
        humidity,
        float(np.sin(2 * np.pi * dt.hour / 24.0)),
        float(np.cos(2 * np.pi * dt.hour / 24.0)),
        1.0 if dt.weekday() in (5, 6) else 0.0,
        max(0.0, 18.0 - temp),
        max(0.0, temp - 18.0),
        float(np.sin(2 * np.pi * dt.month / 12.0)),
        float(np.cos(2 * np.pi * dt.month / 12.0)),
    ]


def build_X(window_rows: list[dict]):
    load_artifacts()
    if len(window_rows) < WINDOW_HOURS:
        raise ValueError(f"Need {WINDOW_HOURS} rows, got {len(window_rows)}")

    rows = window_rows[-WINDOW_HOURS:]
    X = np.asarray([_feature_row(row) for row in rows], dtype=float)
    X_scaled = _scaler_x.transform(X)
    return X_scaled.reshape(1, WINDOW_HOURS, len(FEATURE_COLS))


def predict_next_24(window_rows: list[dict]) -> list[float]:
    load_artifacts()
    X = build_X(window_rows)
    y_scaled = _model.predict(X, verbose=0)
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y = _scaler_y.inverse_transform(y_scaled).flatten()
    return [float(v) for v in y[:24]]

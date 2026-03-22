import os

import joblib
import numpy as np

WINDOW_HOURS = 168
FORECAST_HOURS = 24
FEATURE_COLS = [
    "temp",
    "humidity",
    "dewpoint",
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
    models_dir = os.path.join(base_dir, "models", "trained")
    return {
        "models_dir": models_dir,
        "model": os.path.join(models_dir, "userModel.keras"),
        "sx": os.path.join(models_dir, "user_scaler_x.save"),
        "sy": os.path.join(models_dir, "user_scaler_y.save"),
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
        raise ImportError("No working Keras loader found for the user-trained residential model.") from e


def load_artifacts():
    global _model, _scaler_x, _scaler_y
    if _model is not None and _scaler_x is not None and _scaler_y is not None:
        return

    p = _paths()
    missing = [name for name, path in {"model": p["model"], "sx": p["sx"], "sy": p["sy"]}.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing user-trained residential files in {p['models_dir']}: {missing}")

    model_loader = _resolve_model_loader()
    _model = model_loader(p["model"])
    _scaler_x = joblib.load(p["sx"])
    _scaler_y = joblib.load(p["sy"])


def _coerce_dt(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _feature_row(row):
    dt = _coerce_dt(row["dt"])
    temp = float(row["temp"])
    humidity = float(row["humidity"])
    dewpoint = float(row["dewpoint"])
    return {
        "temp": temp,
        "humidity": humidity,
        "dewpoint": dewpoint,
        "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24.0)),
        "is_weekend": 1.0 if dt.weekday() in (5, 6) else 0.0,
        "HDD": max(0.0, 18.0 - temp),
        "CDD": max(0.0, temp - 18.0),
        "month_sin": float(np.sin(2 * np.pi * dt.month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * dt.month / 12.0)),
    }


def build_X(window_rows):
    load_artifacts()
    if len(window_rows) < WINDOW_HOURS:
        raise ValueError(f"Need {WINDOW_HOURS} rows, got {len(window_rows)}")

    rows = window_rows[-WINDOW_HOURS:]
    features = [[_feature_row(row)[col] for col in FEATURE_COLS] for row in rows]
    X = np.asarray(features, dtype=float)
    X_scaled = _scaler_x.transform(X)
    return X_scaled.reshape(1, WINDOW_HOURS, len(FEATURE_COLS))


def predict_next_24(window_rows):
    load_artifacts()
    X = build_X(window_rows)
    pred_scaled = _model.predict(X, verbose=0)
    pred_scaled = np.asarray(pred_scaled).reshape(-1, 1)
    pred_kwh = _scaler_y.inverse_transform(pred_scaled).flatten()
    return pred_kwh[:FORECAST_HOURS]

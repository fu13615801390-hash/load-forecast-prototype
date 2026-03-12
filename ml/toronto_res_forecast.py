import os
from glob import glob

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
    models_dir = os.path.join(base_dir, "models")
    preferred_model = os.getenv("TORONTO_RES_KERAS_MODEL", "toronto_test.keras")
    fallback_model = os.getenv("TORONTO_RES_KERAS_FALLBACK_MODEL", "toronto_final_model.keras")
    return {
        "models_dir": models_dir,
        "model_candidates": [
            os.path.join(models_dir, preferred_model),
            os.path.join(models_dir, fallback_model),
        ],
        "sx": os.path.join(models_dir, "scaler_x.save"),
        "sy": os.path.join(models_dir, "scaler_y.save"),
    }


def _pick_existing_path(preferred_path, pattern):
    if os.path.isfile(preferred_path):
        return preferred_path

    matches = sorted(glob(pattern))
    for match in matches:
        if os.path.isfile(match):
            return match
    return preferred_path


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
        raise ImportError(
            "No working Keras loader found. Install or repair TensorFlow/Keras."
        ) from e


def _resolve_model_path():
    paths = _paths()
    for path in paths["model_candidates"]:
        if os.path.isfile(path):
            return path

    models_dir = paths["models_dir"]
    fallback_patterns = [
        os.path.join(models_dir, "toronto*.keras"),
        os.path.join(models_dir, "*.keras"),
    ]
    for pattern in fallback_patterns:
        matches = sorted(glob(pattern))
        for match in matches:
            if os.path.isfile(match):
                return match
    raise FileNotFoundError(
        "Missing Keras model file. Expected one of: "
        + ", ".join(paths["model_candidates"])
    )


def load_artifacts():
    global _model, _scaler_x, _scaler_y
    if _model is not None and _scaler_x is not None and _scaler_y is not None:
        return

    paths = _paths()
    scaler_x_path = _pick_existing_path(paths["sx"], os.path.join(paths["models_dir"], "scaler_x*.save"))
    scaler_y_path = _pick_existing_path(paths["sy"], os.path.join(paths["models_dir"], "scaler_y*.save"))
    missing = [
        name
        for name, path in {"sx": scaler_x_path, "sy": scaler_y_path}.items()
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(f"Missing scaler files in {paths['models_dir']}: {missing}")

    try:
        model_loader = _resolve_model_loader()
        model = model_loader(_resolve_model_path())
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
    except Exception:
        _model = None
        _scaler_x = None
        _scaler_y = None
        raise

    _model = model
    _scaler_x = scaler_x
    _scaler_y = scaler_y


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


def predict_next_24h_kwh(window_rows):
    return predict_next_24(window_rows)


def predict_next_24h_mw(window_rows):
    pred_kwh = predict_next_24(window_rows)
    return np.asarray(pred_kwh, dtype=float) / 1000.0

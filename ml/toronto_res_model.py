import os
from glob import glob

import joblib
import numpy as np

WINDOW_HOURS = 168
FEATURE_COUNT = 7
FORECAST_HOURS = 24

_model = None
_scaler_x = None
_scaler_y = None
_load_model = None


def _paths():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    return {
        "models_dir": models_dir,
        "model": os.path.join(models_dir, "toronto_final_model.keras"),
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
            "No working Keras loader found. Install/repair TensorFlow or install standalone keras."
        ) from e


def load_artifacts():
    global _model, _scaler_x, _scaler_y
    if _model is not None and _scaler_x is not None and _scaler_y is not None:
        return

    p = _paths()
    model_path = _pick_existing_path(p["model"], os.path.join(p["models_dir"], "toronto*.keras"))
    scaler_x_path = _pick_existing_path(p["sx"], os.path.join(p["models_dir"], "scaler_x*.save"))
    scaler_y_path = _pick_existing_path(p["sy"], os.path.join(p["models_dir"], "scaler_y*.save"))
    missing = [
        name
        for name, path in {"model": model_path, "sx": scaler_x_path, "sy": scaler_y_path}.items()
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(f"Missing model files in {p['models_dir']}: {missing}")

    try:
        model_loader = _resolve_model_loader()
        model = model_loader(model_path)
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
    except ModuleNotFoundError as e:
        msg = str(e).lower()
        if "sklearn" in msg:
            raise ModuleNotFoundError(
                "Missing dependency 'scikit-learn'. Install with: pip install scikit-learn"
            ) from e
        if "tensorflow" in msg or "keras" in msg:
            raise ModuleNotFoundError(
                "Missing/broken TensorFlow/Keras dependency. Reinstall TensorFlow (or install keras)."
            ) from e
        raise
    except Exception:
        _model = None
        _scaler_x = None
        _scaler_y = None
        raise

    _model = model
    _scaler_x = scaler_x
    _scaler_y = scaler_y


def hour_sin_cos(hour: int):
    return np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)


def hdd(temp):
    return max(0.0, 18.0 - float(temp))


def build_X(window_rows):
    if len(window_rows) < WINDOW_HOURS:
        raise ValueError(f"Need {WINDOW_HOURS} rows, got {len(window_rows)}")

    feats = []
    for r in window_rows[-WINDOW_HOURS:]:
        dt = r["dt"]
        temp = float(r["temp"])
        hum = float(r["humidity"])
        dew = float(r["dewpoint"])

        hs, hc = hour_sin_cos(dt.hour)
        is_weekend = 1 if dt.weekday() in (5, 6) else 0
        feats.append([temp, hum, dew, hs, hc, is_weekend, hdd(temp)])

    X = np.array(feats, dtype=float)
    Xs = _scaler_x.transform(X)
    return Xs.reshape(1, WINDOW_HOURS, FEATURE_COUNT)


def predict_next_24(window_rows):
    load_artifacts()
    X = build_X(window_rows)
    y_scaled = _model.predict(X, verbose=0)
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    y = _scaler_y.inverse_transform(y_scaled).flatten()
    return y


def predict_next_24h_kwh(window_rows):
    return predict_next_24(window_rows)

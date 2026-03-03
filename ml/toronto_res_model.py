# ml/toronto_res_model.py
import os
import numpy as np
import joblib

WINDOW_HOURS = 168
FEATURE_COUNT = 7
FORECAST_HOURS = 24

_model = None
_scaler_x = None
_scaler_y = None
_load_model = None

def _paths():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    models_dir = os.path.join(base_dir, "models")
    return {
        "models_dir": models_dir,
        "model": os.path.join(models_dir, "toronto_final_model.keras"),
        "sx": os.path.join(models_dir, "scaler_x.save"),
        "sy": os.path.join(models_dir, "scaler_y.save"),
    }

def _resolve_model_loader():
    """
    Resolve a compatible Keras model loader at runtime.
    This avoids import-time crashes when TensorFlow is missing/broken.
    """
    global _load_model
    if _load_model is not None:
        return _load_model

    # Preferred path when TensorFlow is installed correctly.
    try:
        from tensorflow.keras.models import load_model as tf_load_model  # type: ignore
        _load_model = tf_load_model
        return _load_model
    except Exception:
        pass

    # Fallback for environments with standalone keras.
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
    # Only short-circuit when all artifacts are loaded.
    if _model is not None and _scaler_x is not None and _scaler_y is not None:
        return

    p = _paths()
    missing = [k for k in ["model","sx","sy"] if not os.path.isfile(p[k])]
    if missing:
        raise FileNotFoundError(f"Missing model files in {p['models_dir']}: {missing}")

    try:
        # Load into locals first so partially-loaded globals never leak.
        model_loader = _resolve_model_loader()
        model = model_loader(p["model"])
        scaler_x = joblib.load(p["sx"])
        scaler_y = joblib.load(p["sy"])
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
        # Reset globals on any failure to avoid stale half-loaded state.
        _model = None
        _scaler_x = None
        _scaler_y = None
        raise

    _model = model
    _scaler_x = scaler_x
    _scaler_y = scaler_y

def hour_sin_cos(hour: int):
    return np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)

def hdd(temp):
    return max(0.0, 18.0 - float(temp))

def build_X(window_rows):
    """
    window_rows: list length 168
    each row: dict with keys: dt (datetime), temp, humidity, dewpoint
    """
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

    X = np.array(feats, dtype=float)              # (168,7)
    Xs = _scaler_x.transform(X)                   # (168,7)
    return Xs.reshape(1, WINDOW_HOURS, FEATURE_COUNT)  # (1,168,7)

def predict_next_24(window_rows):
    load_artifacts()
    X = build_X(window_rows)
    y_scaled = _model.predict(X, verbose=0)
    y_scaled = np.array(y_scaled).reshape(-1, 1)   # (24,1)
    y = _scaler_y.inverse_transform(y_scaled).flatten()
    return y


def predict_next_24h_kwh(window_rows):
    """
    Backward-compatible alias used by older backend code paths.
    """
    return predict_next_24(window_rows)

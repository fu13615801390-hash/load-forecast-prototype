from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "temp",
    "humidity",
    "dewpoint",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "HDD",
    "CDD",
]

DEFAULT_UNITS = {
    "com": "kWh",
    "ind": "MW",
}


def _clean_column_name(value: str) -> str:
    return str(value).replace("\ufeff", "").strip()


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {_clean_column_name(c).lower(): c for c in df.columns}
    for candidate in candidates:
        key = _clean_column_name(candidate).lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _prepare_training_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = df.copy()
    work.columns = [_clean_column_name(c) for c in work.columns]

    datetime_col = _find_first_col(work, ["datetime", "timestamp", "date_time", "date/time"])
    if datetime_col is None:
        raise ValueError("Could not find datetime column in training CSV.")

    target_col = _find_first_col(
        work,
        [
            "TOTAL_CONSUMPTION (kWh)",
            "Total Consumption (kWh)",
            "total_consumption (kWh)",
            "total_consumption_kwh",
            "load",
            "kwh",
            "kw",
            "mw",
        ],
    )
    if target_col is None:
        raise ValueError("Could not find target load column in training CSV.")

    temp_col = _find_first_col(work, ["Temp (°C)", "Temp (C)", "Temp (掳C)", "temperature_C", "temp_c", "temp"])
    hum_col = _find_first_col(work, ["Rel Hum (%)", "relative_humidity_pct", "humidity_percent", "humidity"])
    dew_col = _find_first_col(
        work,
        ["Dew Point Temp (°C)", "Dew Point Temp (C)", "Dew Point Temp (掳C)", "dewpoint", "dewpoint_c"],
    )
    if temp_col is None or hum_col is None:
        raise ValueError("Training CSV must include temperature and humidity columns.")

    work["datetime"] = pd.to_datetime(work[datetime_col].astype(str).str.strip(), errors="coerce")
    work = work.dropna(subset=["datetime"]).drop_duplicates(subset="datetime").sort_values("datetime")

    rename_map = {
        target_col: "target_load",
        temp_col: "temp",
        hum_col: "humidity",
    }
    if dew_col is not None:
        rename_map[dew_col] = "dewpoint"
    work = work.rename(columns=rename_map)

    numeric_cols = ["target_load", "temp", "humidity"]
    if "dewpoint" in work.columns:
        numeric_cols.append("dewpoint")
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "dewpoint" not in work.columns:
        work["dewpoint"] = work["temp"]

    work = work.dropna(subset=["target_load", "temp", "humidity"]).reset_index(drop=True)
    if len(work) < 96:
        raise ValueError("Need at least 96 clean hourly rows to train.")

    dt = work["datetime"]
    work["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    work["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    work["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    work["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    work["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    work["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    work["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    work["HDD"] = (18.0 - work["temp"]).clip(lower=0.0)
    work["CDD"] = (work["temp"] - 18.0).clip(lower=0.0)
    return work, "target_load"


def _fmt_metric(value: float | None, decimals: int = 6, suffix: str = "") -> str:
    if value is None or np.isnan(value):
        return f"n/a{suffix}"
    return f"{float(value):.{decimals}f}{suffix}"


def _build_feature_frame(weather_df: pd.DataFrame) -> pd.DataFrame:
    features = weather_df.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
    features["temp"] = pd.to_numeric(features["temp"], errors="coerce")
    features["humidity"] = pd.to_numeric(features["humidity"], errors="coerce")
    features["dewpoint"] = pd.to_numeric(features["dewpoint"], errors="coerce")
    features = features.dropna(subset=["timestamp", "temp", "humidity", "dewpoint"]).reset_index(drop=True)

    dt = features["timestamp"]
    features["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    features["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    features["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    features["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    features["HDD"] = (18.0 - features["temp"]).clip(lower=0.0)
    features["CDD"] = (features["temp"] - 18.0).clip(lower=0.0)
    return features


def train_user_sector_model(
    df: pd.DataFrame,
    output_dir: str,
    sector: str,
    source_name: str = "uploaded.csv",
    notes: str = "",
):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception as e:
        raise RuntimeError(
            "Training dependencies are not available in the current Python environment. Install/repair scikit-learn."
        ) from e

    sector_key = str(sector or "").strip().lower()
    if sector_key not in DEFAULT_UNITS:
        raise ValueError("Sector must be 'com' or 'ind' for this trainer.")

    work, target_col = _prepare_training_dataframe(df)
    X = work[FEATURE_COLS].astype(float).to_numpy()
    y = work[target_col].astype(float).to_numpy()

    train_end = int(len(work) * 0.7)
    val_end = int(len(work) * 0.85)
    if train_end < 24 or val_end <= train_end or val_end >= len(work):
        raise ValueError("Dataset too small for train/validation/test split.")

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae = float(mean_absolute_error(y_val, val_pred))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_r2 = float(r2_score(y_val, val_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    test_r2 = float(r2_score(y_test, test_pred))

    metrics_text = "\n".join(
        [
            f"Validation MAE:      {_fmt_metric(val_mae, 4)}",
            f"Validation RMSE:     {_fmt_metric(val_rmse, 4)}",
            f"Validation R2:       {_fmt_metric(val_r2, 4)}",
            "",
            "Test Metrics",
            f"Test RMSE:           {_fmt_metric(test_rmse, 4)}",
            f"Test R2:             {_fmt_metric(test_r2, 4)}",
        ]
    )

    trained_dir = Path(output_dir)
    trained_dir.mkdir(parents=True, exist_ok=True)
    model_path = trained_dir / "model.joblib"
    metadata_path = trained_dir / "metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "sector": sector_key,
        "unit": DEFAULT_UNITS[sector_key],
        "feature_columns": list(FEATURE_COLS),
        "source_name": source_name,
        "notes": notes,
        "rows": int(len(work)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sample = work.iloc[train_end : min(train_end + 24, val_end)].copy()
    sample_pred = model.predict(sample[FEATURE_COLS].astype(float).to_numpy())

    return {
        "ok": True,
        "module": f"train_{sector_key}_rf",
        "message": "Training completed.",
        "trained": True,
        "filename": source_name,
        "rows": int(len(work)),
        "columns": [str(c) for c in df.columns],
        "feature_columns": list(FEATURE_COLS),
        "target_column": target_col,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "metrics": {
            "mae": round(val_mae, 4),
            "rmse": round(val_rmse, 4),
            "r2": round(val_r2, 4),
        },
        "validation": {
            "timestamps": [pd.Timestamp(ts).isoformat(timespec="minutes") for ts in sample["datetime"].tolist()],
            "actual": [round(float(v), 4) for v in sample[target_col].tolist()],
            "predicted": [round(float(v), 4) for v in sample_pred.tolist()],
        },
        "test_metrics": {
            "r2": round(test_r2, 4),
            "rmse": round(test_rmse, 4),
        },
        "metrics_text": metrics_text,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "unit": metadata["unit"],
        "notes": notes,
        "sector": sector_key,
    }


def load_user_sector_artifacts(model_dir: str | os.PathLike[str]) -> tuple[object, dict]:
    model_path = Path(model_dir) / "model.joblib"
    metadata_path = Path(model_dir) / "metadata.json"
    missing = [str(p) for p in (model_path, metadata_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing user-trained sector files: {missing}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_user_sector_from_weather(weather_payload: dict, model_dir: str | os.PathLike[str]) -> dict:
    model, metadata = load_user_sector_artifacts(model_dir)

    timestamps = list(weather_payload.get("timestamps") or [])
    temp_values = list(weather_payload.get("temperature_C") or [])
    humidity_values = list(weather_payload.get("relative_humidity_pct") or [])
    dew_values = list(weather_payload.get("dew_point_C") or [])
    if len(dew_values) != len(timestamps):
        dew_values = temp_values

    weather_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": temp_values,
            "humidity": humidity_values,
            "dewpoint": dew_values,
        }
    )
    features = _build_feature_frame(weather_df)
    if features.empty:
        raise ValueError("Uploaded weather payload has no usable rows for prediction.")

    yhat = model.predict(features[FEATURE_COLS].astype(float).to_numpy())
    return {
        "timestamps": [pd.Timestamp(ts).isoformat(timespec="minutes") for ts in features["timestamp"].tolist()],
        "predicted_load": [round(float(v), 4) for v in yhat.tolist()],
        "unit": str(metadata.get("unit") or "load"),
        "sector": str(metadata.get("sector") or ""),
        "model_source": f"user_trained_{metadata.get('sector', 'sector')}_rf",
    }

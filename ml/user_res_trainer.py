import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

INPUT_WINDOW = 168
OUTPUT_WINDOW = 24
DEFAULT_EPOCHS = 100
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


def _clean_column_name(value: str) -> str:
    text = str(value).strip()
    replacements = {
        "脗": "",
        "Â": "",
        "掳": "°",
        "(C)": "(°C)",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


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
        raise ValueError(
            "Could not find datetime column. Expected one of ['datetime', 'timestamp', 'date_time', 'date/time']."
        )

    target_col = _find_first_col(
        work,
        [
            "TOTAL_CONSUMPTION (kWh)",
            "Total Consumption (kWh)",
            "total_consumption (kWh)",
            "total_consumption_kwh",
            "load",
            "kwh",
        ],
    )
    if target_col is None:
        raise ValueError("Could not find target load column.")

    temp_col = _find_first_col(work, ["Temp (°C)", "temperature_C", "temp_c", "temp"])
    hum_col = _find_first_col(work, ["Rel Hum (%)", "relative_humidity_pct", "humidity_percent", "humidity"])
    dew_col = _find_first_col(work, ["Dew Point Temp (°C)", "dewpoint", "dewpoint_c"])

    if temp_col is None or hum_col is None:
        raise ValueError("Combined CSV must include temperature and humidity columns.")

    work["datetime"] = pd.to_datetime(work[datetime_col].astype(str).str.strip(), errors="coerce")
    work = work.dropna(subset=["datetime"]).drop_duplicates(subset="datetime").sort_values("datetime")
    work = work.set_index("datetime")

    rename_map = {
        target_col: "Total Consumption (kWh)",
        temp_col: "temp",
        hum_col: "humidity",
    }
    if dew_col is not None:
        rename_map[dew_col] = "dewpoint"
    work = work.rename(columns=rename_map)

    numeric_cols = ["Total Consumption (kWh)", "temp", "humidity"]
    if "dewpoint" in work.columns:
        numeric_cols.append("dewpoint")
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "dewpoint" not in work.columns:
        work["dewpoint"] = work["temp"]

    work[["temp", "humidity", "dewpoint"]] = work[["temp", "humidity", "dewpoint"]].ffill().bfill()
    work = work.dropna(subset=["Total Consumption (kWh)"])
    if len(work) < INPUT_WINDOW + OUTPUT_WINDOW + 1:
        raise ValueError("Need at least 193 clean hourly rows to train this 168->24 LSTM.")

    work["hour_sin"] = np.sin(2 * np.pi * work.index.hour / 24.0)
    work["hour_cos"] = np.cos(2 * np.pi * work.index.hour / 24.0)
    work["month_sin"] = np.sin(2 * np.pi * work.index.month / 12.0)
    work["month_cos"] = np.cos(2 * np.pi * work.index.month / 12.0)
    work["is_weekend"] = work.index.dayofweek.isin([5, 6]).astype(int)
    work["HDD"] = work["temp"].apply(lambda x: max(0.0, 18.0 - float(x)))
    work["CDD"] = work["temp"].apply(lambda x: max(0.0, float(x) - 18.0))
    return work, "Total Consumption (kWh)"


def _df_to_X_y(X_data, y_data, timestamps, window_size=INPUT_WINDOW, forecast_horizon=OUTPUT_WINDOW):
    X_out, y_out, ts_out = [], [], []
    limit = len(X_data) - window_size - forecast_horizon + 1
    for i in range(limit):
        X_out.append(X_data[i : i + window_size])
        y_out.append(y_data[i + window_size : i + window_size + forecast_horizon])
        ts_out.append(timestamps[i + window_size : i + window_size + forecast_horizon])
    return np.array(X_out), np.array(y_out), np.array(ts_out, dtype=object)


def _inverse_any(y_scaled_any, scaler):
    return scaler.inverse_transform(np.asarray(y_scaled_any).reshape(-1, 1)).flatten()


def _calculate_nmape(y_true_scaled, y_pred_scaled, scaler):
    true_real = _inverse_any(y_true_scaled, scaler)
    pred_real = _inverse_any(y_pred_scaled, scaler)
    denom = np.sum(np.abs(true_real))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(true_real - pred_real)) / denom * 100.0)


def _calculate_rse(y_true_real, y_pred_real):
    num = np.sum((y_true_real - y_pred_real) ** 2)
    den = np.sum((y_true_real - np.mean(y_true_real)) ** 2)
    return float(num / den) if den != 0 else np.nan


def _calculate_mase(y_true_real, y_pred_real, train_series_real, m=1):
    mae = np.mean(np.abs(y_true_real - y_pred_real))
    if len(train_series_real) <= m:
        return np.nan
    scale = np.mean(np.abs(train_series_real[m:] - train_series_real[:-m]))
    return float(mae / scale) if scale != 0 else np.nan


def _calculate_rmse(y_true_real, y_pred_real):
    return float(np.sqrt(np.mean((y_true_real - y_pred_real) ** 2)))


def train_user_lstm(
    df: pd.DataFrame,
    output_dir: str,
    source_name: str = "uploaded.csv",
    notes: str = "",
    epochs: int = DEFAULT_EPOCHS,
):
    try:
        import joblib
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.layers import Bidirectional, Dense, Dropout, InputLayer, LSTM
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.metrics import RootMeanSquaredError
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.optimizers import Adam
    except Exception as e:
        raise RuntimeError(
            "Training dependencies are not available in the current Python environment. "
            "Install/repair tensorflow, scikit-learn, and joblib."
        ) from e

    work, target_col = _prepare_training_dataframe(df)

    X_inputs = work[FEATURE_COLS].astype(float)
    y_target = work[[target_col]].astype(float)
    timestamps = work.index.to_numpy()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X_inputs)
    y_scaled = scaler_y.fit_transform(y_target)

    X, y, y_timestamps = _df_to_X_y(X_scaled, y_scaled, timestamps)
    if len(X) < 8:
        raise ValueError("Not enough training windows after preprocessing.")

    train_v = int(len(X) * 0.7)
    val_v = int(len(X) * 0.85)
    if train_v < 1 or val_v <= train_v or val_v >= len(X):
        raise ValueError("Dataset too small for train/validation/test split.")

    X_train, y_train = X[:train_v], y[:train_v]
    X_val, y_val = X[train_v:val_v], y[train_v:val_v]
    X_test, y_test = X[val_v:], y[val_v:]
    ts_val = y_timestamps[train_v:val_v]

    model = Sequential(
        [
            InputLayer((INPUT_WINDOW, len(FEATURE_COLS))),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(OUTPUT_WINDOW, activation="linear"),
        ]
    )

    os.makedirs(output_dir, exist_ok=True)
    active_model_path = os.path.join(output_dir, "toronto_test.keras")
    scaler_x_path = os.path.join(output_dir, "scaler_x.save")
    scaler_y_path = os.path.join(output_dir, "scaler_y.save")
    alias_model_path = os.path.join(output_dir, "userModel.keras")
    alias_scaler_x_path = os.path.join(output_dir, "scaler_x.save")
    alias_scaler_y_path = os.path.join(output_dir, "scaler_y.save")

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.0005),
        metrics=[RootMeanSquaredError()],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max(1, int(epochs)),
        batch_size=32,
        callbacks=[
            ModelCheckpoint(active_model_path, save_best_only=True),
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ],
        verbose=0,
    )

    model.save(active_model_path)
    if active_model_path != alias_model_path:
        model.save(alias_model_path)
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    if scaler_x_path != alias_scaler_x_path:
        joblib.dump(scaler_x, alias_scaler_x_path)
    if scaler_y_path != alias_scaler_y_path:
        joblib.dump(scaler_y, alias_scaler_y_path)

    model1 = load_model(active_model_path)
    train_preds = model1.predict(X_train, verbose=0)
    val_preds = model1.predict(X_val, verbose=0)
    test_preds = model1.predict(X_test, verbose=0)

    val_true_real = _inverse_any(y_val, scaler_y)
    val_pred_real = _inverse_any(val_preds, scaler_y)
    test_true_real = _inverse_any(y_test, scaler_y)
    test_pred_real = _inverse_any(test_preds, scaler_y)
    train_series_real = _inverse_any(y_scaled[: train_v + INPUT_WINDOW + OUTPUT_WINDOW - 1], scaler_y)

    val_nmape = _calculate_nmape(y_val, val_preds, scaler_y)
    test_nmape = _calculate_nmape(y_test, test_preds, scaler_y)
    val_rse = _calculate_rse(val_true_real, val_pred_real)
    test_rse = _calculate_rse(test_true_real, test_pred_real)
    val_mase = _calculate_mase(val_true_real, val_pred_real, train_series_real, m=1)
    test_mase = _calculate_mase(test_true_real, test_pred_real, train_series_real, m=1)
    val_r2 = float(r2_score(val_true_real, val_pred_real))
    test_r2 = float(r2_score(test_true_real, test_pred_real))
    val_rmse = _calculate_rmse(val_true_real, val_pred_real)
    test_rmse = _calculate_rmse(test_true_real, test_pred_real)
    val_mae = float(mean_absolute_error(val_true_real, val_pred_real))

    last_window = X_scaled[-INPUT_WINDOW:]
    prediction_scaled = model1.predict(last_window.reshape(1, INPUT_WINDOW, len(FEATURE_COLS)), verbose=0)
    future_forecast_kwh = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten().tolist()

    sample_idx = 0
    validation_timestamps = [pd.Timestamp(ts).isoformat(timespec="minutes") for ts in ts_val[sample_idx].tolist()]
    validation_actual = scaler_y.inverse_transform(y_val[sample_idx].reshape(-1, 1)).flatten().tolist()
    validation_predicted = scaler_y.inverse_transform(val_preds[sample_idx].reshape(-1, 1)).flatten().tolist()

    frontend_metrics = {
        "dataFileName": source_name,
        "inputWindow": INPUT_WINDOW,
        "outputWindow": OUTPUT_WINDOW,
        "epochs": max(1, int(epochs)),
        "files": {
            "bestCheckpointFile": active_model_path,
            "finalModelFile": active_model_path,
            "scalerXFile": scaler_x_path,
            "scalerYFile": scaler_y_path,
        },
        "validation": {
            "nMAPE": None if np.isnan(val_nmape) else float(val_nmape),
            "RSE": None if np.isnan(val_rse) else float(val_rse),
            "MASE": None if np.isnan(val_mase) else float(val_mase),
            "R2": float(val_r2),
            "RMSE": float(val_rmse),
        },
        "test": {
            "nMAPE": None if np.isnan(test_nmape) else float(test_nmape),
            "RSE": None if np.isnan(test_rse) else float(test_rse),
            "MASE": None if np.isnan(test_mase) else float(test_mase),
            "R2": float(test_r2),
            "RMSE": float(test_rmse),
        },
        "futureForecastKwh": [float(v) for v in future_forecast_kwh],
    }

    return {
        "ok": True,
        "module": "train_res_lstm",
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
            "nmape": None if np.isnan(val_nmape) else round(float(val_nmape), 4),
            "rse": None if np.isnan(val_rse) else round(float(val_rse), 4),
            "mase": None if np.isnan(val_mase) else round(float(val_mase), 4),
        },
        "validation": {
            "timestamps": validation_timestamps,
            "actual": [round(float(v), 4) for v in validation_actual],
            "predicted": [round(float(v), 4) for v in validation_predicted],
        },
        "test_metrics": {
            "nmape": None if np.isnan(test_nmape) else round(float(test_nmape), 4),
            "rse": None if np.isnan(test_rse) else round(float(test_rse), 4),
            "mase": None if np.isnan(test_mase) else round(float(test_mase), 4),
            "r2": round(test_r2, 4),
            "rmse": round(test_rmse, 4),
        },
        "future_forecast_kwh": [round(float(v), 4) for v in future_forecast_kwh],
        "frontend_metrics": frontend_metrics,
        "model_path": active_model_path,
        "scaler_x_path": scaler_x_path,
        "scaler_y_path": scaler_y_path,
        "unit": "kWh",
        "notes": notes,
    }

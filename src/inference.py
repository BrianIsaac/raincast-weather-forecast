import requests
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict
import argparse
from src.model import TLSTMEnsembleTrainer

def read_time_df_call() -> pd.DataFrame:
    base_urls = {
        "wind_speed": "https://api-open.data.gov.sg/v2/real-time/api/wind-speed",
        "air_temperature": "https://api-open.data.gov.sg/v2/real-time/api/air-temperature",
        "wind_direction": "https://api-open.data.gov.sg/v2/real-time/api/wind-direction",
        "relative_humidity": "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity",
        "rainfall": "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
    }

    def fetch_all_pages(url: str, date: str) -> list:
        print(f"Calling API for {url.split('/')[-1]} on {date}")
        all_readings = []
        pagination_token = None

        while True:
            params = {'date': date}
            if pagination_token:
                params['paginationToken'] = pagination_token
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json().get('data', {})
            all_readings.extend(payload.get('readings', []))
            pagination_token = payload.get('paginationToken')
            if not pagination_token:
                break
        return all_readings

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    date_strs = [yesterday.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")]

    dfs = []
    for var, url in base_urls.items():
        all_var_readings = []
        for date_str in date_strs:
            try:
                readings_by_time = fetch_all_pages(url, date_str)
                for entry in readings_by_time:
                    timestamp = entry.get("timestamp")
                    values = entry.get("data", [])
                    if values:
                        avg_val = sum(d['value'] for d in values) / len(values)
                        all_var_readings.append({"timestamp": timestamp, var: avg_val})
            except Exception as e:
                print(f"Error fetching {var} for {date_str}: {e}")

        df_var = pd.DataFrame(all_var_readings)
        if not df_var.empty:
            df_var['timestamp'] = pd.to_datetime(df_var['timestamp'])
            df_var = df_var.groupby('timestamp', as_index=False).mean()
            df_var = df_var.set_index('timestamp').sort_index()
            dfs.append(df_var)

    if dfs:
        df_combined = pd.concat(dfs, axis=1).sort_index()
        return df_combined.resample("h").mean()
    else:
        return pd.DataFrame()

def predict_next_rainfall(df: pd.DataFrame, model_ckpt_path: str) -> float:
    LOOKBACK, HORIZON, TARGET_COL, HURDLE, DELTA = 24, 1, "rainfall", True, True
    assert df.shape[0] >= LOOKBACK, f"Need at least {LOOKBACK} rows. Got {df.shape[0]}."
    df = df.sort_index().copy()
    df["elapsed_time"] = df.index.to_series().diff().dt.total_seconds().div(3600).fillna(0)
    df["target"] = df[TARGET_COL].shift(-HORIZON)
    if HURDLE:
        df["occurrence"] = (df["target"] > 0).astype(int)
    if DELTA:
        for col in [c for c in df.columns if c not in ("elapsed_time", "target", "occurrence")]:
            df[f"{col}_delta"] = df[col].diff()
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    for m in range(1, 13): 
        df[f"m_{m}"] = 0
    for h in range(24): 
        df[f"h_{h}"] = 0
    for idx in df.index:
        df.at[idx, f"m_{df.at[idx, 'month']}"] = 1
        df.at[idx, f"h_{df.at[idx, 'hour']}"] = 1
    df.drop(columns=["month", "hour"], inplace=True)
    df = df.dropna()
    if df.shape[0] < LOOKBACK:
        raise ValueError(f"Only {df.shape[0]} valid rows after preprocessing (need at least {LOOKBACK})")
    df = df.iloc[-LOOKBACK:]
    exclude = {"elapsed_time", "target", "occurrence"}
    columns = list(df.columns)
    feature_cols = [c for c in columns if c not in exclude]
    feature_idxs = [columns.index(c) for c in feature_cols]
    arr = df.values
    X = arr[:, feature_idxs].astype(np.float32)
    T = arr[:, columns.index("elapsed_time"):columns.index("elapsed_time")+1].astype(np.float32)
    X_tensor = torch.tensor(X[np.newaxis, :, :])
    T_tensor = torch.tensor(T[np.newaxis, :, :])
    _, _, F = X_tensor.shape

    def strip_prefix(state_dict, prefix="_orig_mod."):
        return OrderedDict((k[len(prefix):] if k.startswith(prefix) else k, v)
                           for k, v in state_dict.items())

    ensemble = TLSTMEnsembleTrainer(
        input_dim=F, hidden_dim=64, fc_dim=32,
        lr_reg=1e-3, lr_cls=1e-3, use_compile=False
    )
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    ensemble.reg_trainer.model.load_state_dict(strip_prefix(ckpt["regressor_state_dict"]))
    ensemble.cls_trainer.model.load_state_dict(strip_prefix(ckpt["classifier_state_dict"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_tensor.to(device)
    T_tensor = T_tensor.to(device)
    ensemble.reg_trainer.model.to(device).eval()
    ensemble.cls_trainer.model.to(device).eval()

    with torch.no_grad():
        reg_out = ensemble.reg_trainer.model(X_tensor, T_tensor)
        cls_prob = torch.sigmoid(ensemble.cls_trainer.model(X_tensor, T_tensor))
        return max((reg_out * cls_prob).item(), 0.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print("Fetching latest weather data...")
    df_weather = read_time_df_call()

    if df_weather.empty or df_weather.shape[0] < 24:
        print("Insufficient data for prediction.")
    else:
        pred = predict_next_rainfall(df_weather, args.model_path)
        print(f"Predicted rainfall for next hour: {pred:.3f} mm")\
        
# python inference.py --model_path ./models/Model/weather.pth

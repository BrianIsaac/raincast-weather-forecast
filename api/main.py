import io
import contextlib
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.inference import read_time_df_call, predict_next_rainfall

app = FastAPI()

MODEL_CKPT_PATH = "./models/Model/weather.pth"
LOOKBACK = 24

@app.get("/")
def root():
    return {"message": "Rainfall Forecasting API is running."}


@app.get("/predict")
def predict_rainfall():
    log_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(log_buffer):
            df_weather = read_time_df_call()
            if df_weather.empty or df_weather.shape[0] < LOOKBACK:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "detail": "Insufficient data for prediction.",
                        "logs": log_buffer.getvalue()
                    }
                )

            prediction = predict_next_rainfall(df_weather, MODEL_CKPT_PATH)

            # Extract rainfall history
            df_rainfall = df_weather[["rainfall"]].copy().dropna()
            df_rainfall.index = df_rainfall.index.strftime("%Y-%m-%d %H:%M:%S")
            rainfall_history = df_rainfall.reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records")

        return {
            "predicted_rainfall_mm": round(prediction, 3),
            "status": "success",
            "logs": log_buffer.getvalue(),
            "rainfall_history": rainfall_history
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Prediction failed: {str(e)}",
                "logs": log_buffer.getvalue()
            }
        )


# PYTHONPATH=. uvicorn api.main:app --reload


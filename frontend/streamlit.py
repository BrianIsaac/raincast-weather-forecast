import os
import streamlit as st
import requests
import pandas as pd
import altair as alt

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/predict")

st.title("🌧️ Rainfall Prediction Dashboard")
st.markdown("Click below to fetch real-time weather data and predict rainfall for the **next hour**.")

if st.button("📡 Predict Rainfall"):
    with st.spinner("Fetching data and running model..."):
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prediction completed")

                predicted_value = result["predicted_rainfall_mm"]
                st.metric(label="Predicted Rainfall (mm)", value=predicted_value)

                # Prepare DataFrame
                history = result.get("rainfall_history", [])
                if history:
                    df = pd.DataFrame(history)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp").sort_index()

                    # Add predicted value as the last row
                    last_time = df.index[-1]
                    next_time = last_time + pd.Timedelta(hours=1)
                    df.loc[next_time] = predicted_value
                    df["source"] = "Actual"
                    df.iloc[-1, df.columns.get_loc("source")] = "Predicted"

                    # Keep only last 5 rows (4 actual + 1 predicted)
                    df_trimmed = df.tail(5).reset_index()

                    # Altair chart
                    chart = alt.Chart(df_trimmed).mark_line(point=True).encode(
                        x=alt.X("timestamp:T", title="Time"),
                        y=alt.Y("rainfall:Q", title="Rainfall (mm)"),
                        color=alt.Color("source:N", scale=alt.Scale(domain=["Actual", "Predicted"], range=["#1f77b4", "#e74c3c"])),
                        tooltip=["timestamp:T", "rainfall:Q", "source:N"]
                    ).properties(
                        title="Rainfall Forecast: Last 4 Hours + Next Prediction",
                        width="container"
                    )

                    st.altair_chart(chart, use_container_width=True)

                with st.expander("🔍 Prediction Logs"):
                    st.text(result.get("logs", "No logs available."))

            else:
                error = response.json()
                st.error(f"❌ {error.get('detail', 'Unknown error')}")
                with st.expander("Debug Logs"):
                    st.text(error.get("logs", "No logs available."))

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

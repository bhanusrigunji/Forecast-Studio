from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Forecast Studio", page_icon="📈", layout="wide")


@dataclass
class ForecastResult:
    forecast: pd.Series
    actual: pd.Series
    predicted: pd.Series
    mae: float
    rmse: float
    mape: float | None


def sample_data() -> pd.DataFrame:
    """A deterministic monthly example dataset, so the app works immediately."""
    rng = np.random.default_rng(12)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    t = np.arange(len(dates))
    seasonal = 18 * np.sin(2 * np.pi * t / 12) + 7 * np.cos(2 * np.pi * t / 12)
    values = 100 + 1.15 * t + seasonal + rng.normal(0, 4, len(t))
    return pd.DataFrame({"Date": dates, "Sales": np.round(values, 2)})


def prepare_data(df: pd.DataFrame, date_col: str, value_col: str, freq: str) -> pd.Series:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().sort_values(date_col)
    if data.empty:
        raise ValueError("No usable rows found. Check the date and numeric value columns.")
    # Combine duplicate dates, align to a regular calendar, then fill short gaps.
    series = data.groupby(date_col)[value_col].mean().sort_index()
    series = series.resample(freq).mean()
    series = series.interpolate(method="time", limit_direction="both").dropna()
    if len(series) < 12:
        raise ValueError("Please provide at least 12 usable time periods for a meaningful forecast.")
    series.name = value_col
    return series


def linear_features(index: np.ndarray, period: int) -> np.ndarray:
    angle = 2 * np.pi * index / period
    return np.column_stack([index, np.sin(angle), np.cos(angle)])


def fit_forecast(series: pd.Series, horizon: int, model_name: str, seasonal_period: int) -> ForecastResult:
    values = series.to_numpy(dtype=float)
    # Keep the latest 20% for a time-ordered holdout; train only on earlier data.
    test_size = max(2, min(int(round(len(values) * 0.2)), len(values) // 3))
    split = len(values) - test_size
    if split < 6:
        raise ValueError("There is not enough history to train and evaluate the model.")

    if model_name == "Linear trend + seasonality":
        train_idx = np.arange(split)
        test_idx = np.arange(split, len(values))
        model = LinearRegression().fit(linear_features(train_idx, seasonal_period), values[:split])
        test_pred = model.predict(linear_features(test_idx, seasonal_period))
        final_model = LinearRegression().fit(linear_features(np.arange(len(values)), seasonal_period), values)
        future_idx = np.arange(len(values), len(values) + horizon)
        future_values = final_model.predict(linear_features(future_idx, seasonal_period))
    else:
        # Random forest learns from the most recent values and calendar trend.
        lags = min(max(3, seasonal_period), max(3, len(values) // 4))

        def make_rows(end: int) -> tuple[np.ndarray, np.ndarray]:
            rows, targets = [], []
            for i in range(lags, end):
                row = [values[i - lag] for lag in range(1, lags + 1)]
                row += [i, np.sin(2 * np.pi * i / seasonal_period), np.cos(2 * np.pi * i / seasonal_period)]
                rows.append(row)
                targets.append(values[i])
            return np.asarray(rows), np.asarray(targets)

        x_train, y_train = make_rows(split)
        x_test, _ = make_rows(len(values))
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test[-test_size:])
        final_x, final_y = make_rows(len(values))
        final_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
        final_model.fit(final_x, final_y)
        history = list(values)
        future_values = []
        for i in range(len(values), len(values) + horizon):
            row = [history[-lag] for lag in range(1, lags + 1)]
            row += [i, np.sin(2 * np.pi * i / seasonal_period), np.cos(2 * np.pi * i / seasonal_period)]
            next_value = float(final_model.predict([row])[0])
            future_values.append(next_value)
            history.append(next_value)

    actual = pd.Series(values[split:], index=series.index[split:], name="Actual")
    predicted = pd.Series(test_pred, index=actual.index, name="Predicted")
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = actual.abs() > 1e-9
    mape = float((np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero]).mean()) * 100) if nonzero.any() else None
    future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=series.index.freq)[1:]
    forecast = pd.Series(future_values, index=future_index, name="Forecast")
    return ForecastResult(forecast, actual, predicted, mae, rmse, mape)


st.title("📈 Forecast Studio")
st.caption("Explore historical trends, evaluate a forecasting model, and estimate what may come next.")

with st.sidebar:
    st.header("Your data")
    upload = st.file_uploader("Upload a CSV file", type=["csv"], help="Include one date column and one numeric measure column.")
    st.caption("No file? A sample monthly sales dataset is ready to explore.")

try:
    raw = pd.read_csv(upload) if upload is not None else sample_data()
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if len(raw.columns) < 2:
    st.error("The CSV needs at least two columns: a date and a numeric value.")
    st.stop()

with st.sidebar:
    date_guess = next((c for c in raw.columns if "date" in str(c).lower() or "time" in str(c).lower()), raw.columns[0])
    date_col = st.selectbox("Date column", raw.columns, index=list(raw.columns).index(date_guess))
    numeric_cols = [c for c in raw.columns if pd.to_numeric(raw[c], errors="coerce").notna().any() and c != date_col]
    if not numeric_cols:
        st.error("No numeric value columns found in this CSV.")
        st.stop()
    value_col = st.selectbox("Value to forecast", numeric_cols)
    freq_label = st.selectbox("Time interval", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=2)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}[freq_label]
    model_name = st.selectbox("Forecast model", ["Linear trend + seasonality", "Random forest"])
    horizon = st.slider("Periods to forecast", 1, 36, 12)
    default_period = {"Daily": 7, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": 1}[freq_label]
    season_period = st.number_input("Season length (periods)", min_value=1, max_value=365, value=default_period)

try:
    series = prepare_data(raw, date_col, value_col, freq)
    result = fit_forecast(series, horizon, model_name, int(season_period))
except Exception as exc:
    st.error(str(exc))
    st.stop()

left, center, right = st.columns(3)
left.metric("History", f"{len(series):,} periods")
center.metric("Holdout MAE", f"{result.mae:,.2f}")
right.metric("Holdout RMSE", f"{result.rmse:,.2f}")
if result.mape is not None:
    st.caption(f"Holdout MAPE: {result.mape:.1f}% · Metrics use the most recent portion of history as a chronological holdout.")
else:
    st.caption("MAPE is unavailable because holdout values are all zero.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical", mode="lines", line=dict(color="#64748b", width=2)))
fig.add_trace(go.Scatter(x=result.actual.index, y=result.predicted.values, name="Holdout prediction", mode="lines+markers", line=dict(color="#f59e0b", dash="dot")))
fig.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast.values, name="Future forecast", mode="lines+markers", line=dict(color="#2563eb", width=3)))
fig.update_layout(title=f"{value_col}: history and forecast", xaxis_title="Date", yaxis_title=value_col, hovermode="x unified", legend=dict(orientation="h", y=1.12), margin=dict(l=20, r=20, t=70, b=20))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Forecast values"):
    forecast_df = result.forecast.rename(value_col).rename_axis("Date").reset_index()
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")

with st.expander("Data preview and notes"):
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
    st.write(f"Cleaned range: **{series.index.min().date()}** to **{series.index.max().date()}**. Duplicate timestamps were averaged, missing intervals were interpolated, and rows without a valid date or numeric value were dropped.")
    st.info("Forecasts are estimates based on past patterns. They do not account for known future events or causal factors. Compare the holdout metrics with your domain knowledge before making decisions.")

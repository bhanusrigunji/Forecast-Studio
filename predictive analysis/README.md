# Forecast Studio

A small, executable forecasting project based on the description in `Description.txt`. It demonstrates data cleaning, regression and machine-learning forecasts, chronological model evaluation, and interactive visualization.

## Run on Windows

1. Install Python 3.10 or newer.
2. Double-click `run.bat`.
3. On its first run, the launcher creates a local `.venv` and installs the packages in `requirements.txt` (an internet connection is needed for this step).
4. The app opens in your browser. Use the sample data immediately or upload a CSV containing a date/time column and a numeric column.

You can also start it from a terminal in this folder:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

## Features

- Select date and value columns, then choose daily, weekly, monthly, quarterly, or yearly intervals.
- Clean invalid rows, average duplicate timestamps, and fill missing periods by time interpolation.
- Compare a linear trend with seasonal terms or a random forest using recent lags.
- Evaluate on a chronological holdout using MAE, RMSE, and MAPE.
- View history, holdout predictions, and future estimates together; download future estimates as CSV.

The included monthly sample dataset is generated in the app and requires no separate data file. Forecasts are illustrative estimates and should be checked against domain knowledge.

import logging
import warnings

# Suppress logs
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Libraries
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_pipeline(file_path):
    """
    Runs the full sales forecasting pipeline
    Returns metrics + predictions for Streamlit use
    """

    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv(file_path, encoding='cp1252')

    # =========================
    # Preprocessing
    # =========================
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('Order Date')
    df = df.dropna()

    # Monthly aggregation
    df = df.groupby(pd.Grouper(key='Order Date', freq='ME')).sum().reset_index()

    # =========================
    # Prepare for Prophet
    # =========================
    df_prophet = df[['Order Date', 'Sales']]
    df_prophet.columns = ['ds', 'y']

    # Train-test split
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet[:train_size]
    test = df_prophet[train_size:]

    # =========================
    # Train model
    # =========================
    model = Prophet()
    model.fit(train)

    # =========================
    # Predictions
    # =========================
    future = model.make_future_dataframe(periods=len(test), freq='ME')
    forecast = model.predict(future)

    pred = forecast[['ds', 'yhat']]
    comparison = test.merge(pred, on='ds')

    # =========================
    # Evaluation Metrics
    # =========================
    mae = mean_absolute_error(comparison['y'], comparison['yhat'])
    rmse = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))

    # ✅ Safe MAPE
    mape = np.mean(
        np.abs((comparison['y'] - comparison['yhat']) /
               np.where(comparison['y'] == 0, 1, comparison['y']))
    ) * 100

    # =========================
    # Future Forecast (6 months)
    # =========================
    future_6 = model.make_future_dataframe(periods=6, freq='ME')
    forecast_6 = model.predict(future_6)

    # =========================
    # Return results
    # =========================
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "comparison": comparison,
        "forecast": forecast,
        "forecast_6": forecast_6,
        "model": model
    }
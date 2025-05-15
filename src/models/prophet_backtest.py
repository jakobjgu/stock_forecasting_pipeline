import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.models.prophet_model import prepare_prophet_input


def prophet_backtest(
    series: pd.Series,
    cutoff_dates: list[str],
    forecast_horizon: int = 12,
    freq: str = "MS"
) -> pd.DataFrame:
    """
    Perform Prophet backtesting over a list of historical cutoff dates.
    
    Returns a DataFrame with cutoff, MAE, RMSE, and sample forecast window.
    """
    series = series.sort_index().dropna()
    df_prophet = prepare_prophet_input(series)

    results = []

    for cutoff in cutoff_dates:
        cutoff = pd.to_datetime(cutoff)
        train_df = df_prophet[df_prophet["ds"] <= cutoff]

        # Train model
        model = Prophet()
        model.fit(train_df)

        # Forecast
        future = model.make_future_dataframe(periods=forecast_horizon, freq=freq)
        forecast = model.predict(future)

        # Actuals in the forecast range
        y_true = df_prophet.set_index("ds").reindex(forecast["ds"]).loc[cutoff:].iloc[1:forecast_horizon+1]["y"]
        y_pred = forecast.set_index("ds").loc[y_true.index]["yhat"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        results.append({
            "cutoff": cutoff.date(),
            "horizon": forecast_horizon,
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(results)

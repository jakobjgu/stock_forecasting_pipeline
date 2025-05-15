import pandas as pd
from prophet import Prophet
from src.models.prophet_model import prepare_prophet_input


def forecast_sp500_with_regressors(
    sp500: pd.Series,
    cpi_forecast: pd.DataFrame,
    usd_eur_forecast: pd.DataFrame,
    periods: int,
    freq: str = "B"
) -> pd.DataFrame:
    """
    Forecast SP500 using CPI and USD/EUR as external regressors.

    Assumes cpi_forecast and usd_eur_forecast come from Prophet (with 'ds' and 'yhat').
    """
    df = prepare_prophet_input(sp500)
    df["cpi"] = cpi_forecast.set_index("ds").reindex(df["ds"])["yhat"].values
    df["usd_eur"] = usd_eur_forecast.set_index("ds").reindex(df["ds"])["yhat"].values

    model = Prophet()
    model.add_regressor("cpi")
    model.add_regressor("usd_eur")
    model.fit(df.dropna())  # Only use rows where all regressors are present

    # Prepare future DataFrame
    future = model.make_future_dataframe(periods=periods, freq=freq)
    future["cpi"] = cpi_forecast.set_index("ds").reindex(future["ds"])["yhat"].values
    future["usd_eur"] = usd_eur_forecast.set_index("ds").reindex(future["ds"])["yhat"].values

    forecast = model.predict(future)
    return forecast

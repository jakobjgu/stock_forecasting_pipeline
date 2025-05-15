import pandas as pd
from prophet import Prophet
from src.models.prophet_model import prepare_prophet_input

def prepare_prophet_input(series: pd.Series) -> pd.DataFrame:
    """
    Convert a time-indexed pd.Series into Prophet's expected format.
    """
    df = series.reset_index()
    df.columns = ["ds", "y"]
    return df


def train_prophet(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on the given DataFrame with columns ['ds', 'y'].
    """
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model


def make_future_forecast(model: Prophet, periods: int = 30, freq: str = "D") -> pd.DataFrame:
    """
    Generate a forecast for the specified number of periods into the future.
    Set `freq='MS'` for monthly data.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def forecast_series(
    series: pd.Series,
    periods: int = 30,
    freq: str = "D",
    name: str = "series",
    daily_seasonality: bool = True,
    yearly_seasonality: bool = True,
    return_model: bool = False
) -> tuple[pd.DataFrame, Prophet] | pd.DataFrame:
    """
    Forecast a univariate time series using Prophet.

    Parameters:
    - series: time-indexed pd.Series
    - periods: number of future periods to forecast
    - freq: frequency of the forecast (e.g. 'D', 'MS', 'B')
    - name: optional name for printing/logging
    - daily_seasonality / yearly_seasonality: Prophet seasonality toggles
    - return_model: if True, returns (forecast_df, model)

    Returns:
    - forecast_df, or (forecast_df, model) if return_model=True
    """
    print(f"Forecasting {name} for {periods} periods ({freq})...")

    # Prepare input
    df = prepare_prophet_input(series)

    # Fit model
    model = Prophet(daily_seasonality=daily_seasonality, yearly_seasonality=yearly_seasonality)
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return (forecast, model) if return_model else forecast


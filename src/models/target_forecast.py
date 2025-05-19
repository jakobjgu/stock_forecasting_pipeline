import pandas as pd
from prophet import Prophet
from src.utils.prophet_helpers import prepare_prophet_input

def forecast_target(
    target: pd.Series,
    periods: int,
    freq: str = "MS"
) -> pd.DataFrame:
    """
    Forecast target values using Prophet with no external regressors.

    Parameters:
    - target: pd.Series
        Actual target values with a datetime index, used for model training.
    - periods: int
        Number of periods to forecast into the future.
    - freq: str
        Frequency of the time series (e.g., 'MS' for month start, 'B' for business day).

    Returns:
    - pd.DataFrame
        A Prophet forecast DataFrame for target including 'ds', 'yhat', 'yhat_lower', and 'yhat_upper'.
    """
    # Step 1: Prepare training DataFrame
    df = prepare_prophet_input(target)

    # Step 2: Fit model without regressors
    model = Prophet()
    model.fit(df.dropna())

    # Step 3: Prepare future dates and forecast
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast

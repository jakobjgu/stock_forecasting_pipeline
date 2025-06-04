import pandas as pd
from prophet import Prophet
from src.utils.prophet_helpers import (
    prepare_prophet_input,
    align_forecast_regressor,
    series_to_prophet_df
    )


def forecast_target_with_regressors(
    target: pd.Series,
    actual_regressors: dict[str, pd.Series],
    forecast_regressors: dict[str, pd.DataFrame],
    periods: int,
    freq: str = "MS"
) -> pd.DataFrame:
    """
    Forecast Target values using Prophet with multiple dynamic external regressors.

    Parameters:
    - Target: pd.Series
        Time series of target values with a datetime index.
    - actual_regressors: dict[str, pd.Series]
        Dictionary of actual regressor series (must match index of target series).
    - forecast_regressors: dict[str, pd.DataFrame]
        Dictionary of Prophet-forecasted regressors with columns ['ds', 'yhat'].
    - periods: int
        Number of periods to forecast into the future.
    - freq: str
        Frequency of the time series (e.g., 'B', 'MS').

    Returns:
    - pd.DataFrame
        Prophet forecast including yhat, yhat_lower, yhat_upper, and regressor contributions.
    """


    # Step 1: Prepare training DataFrame
    df = prepare_prophet_input(target)  # adds 'ds' and 'y'

    # Add all aligned actual regressors
    model = Prophet()
    for name, series in actual_regressors.items():
        reg_df = series_to_prophet_df(series)
        df[name] = align_forecast_regressor(reg_df, df["ds"], freq=freq, column="y").values
        model.add_regressor(name)

    # Step 2: Fit model
    model.fit(df.dropna())

    # Step 3: Prepare future DataFrame
    future = model.make_future_dataframe(periods=periods, freq=freq)

    for name, forecast_df in forecast_regressors.items():
        future[name] = align_forecast_regressor(forecast_df, future["ds"], freq=freq, column="yhat").values

    # Step 4: Generate forecast
    forecast = model.predict(future)

    return forecast

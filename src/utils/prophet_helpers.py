import pandas as pd
from src.utils.preprocess import save_series_to_csv
from prophet import Prophet

def prepare_prophet_input(series: pd.Series) -> pd.DataFrame:
    """
    Convert a time-indexed pd.Series into Prophet's expected format.
    """
    df = series.reset_index()
    df.columns = ["ds", "y"]
    return df

import pandas as pd


def align_forecast_regressor(
    forecast_df: pd.DataFrame,
    future_ds: pd.Series,
    column: str = "yhat",
    freq: str = "MS"
) -> pd.Series:
    """
    Align and resample a forecast or actual time series to match a target date index.

    Parameters:
    - forecast_df: must have 'ds' and the specified value column
    - future_ds: the target dates to align to
    - column: the column in forecast_df to align ('y' for actuals, 'yhat' for forecast)
    - freq: resample frequency, default 'MS' (month start)

    Returns:
    - A Series aligned to future_ds with forward-filled values
    """
    if column not in forecast_df.columns:
        raise ValueError(f"Expected column '{column}' not found in forecast_df. Columns: {forecast_df.columns.tolist()}")

    future_index = pd.DatetimeIndex(future_ds).normalize()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"]).dt.normalize()
    forecast_df[column] = forecast_df[column].ffill()

    forecast_df = forecast_df.dropna(subset=[column])
    forecast_daily = (
        forecast_df.set_index("ds")[column]
        .resample(freq)
        .ffill()
    )
    forecast_daily.name = column  # restore name so it's accessible in later steps
    forecast_daily = forecast_daily.dropna()

    # Prepare aligned forecast regressor and export for inspection
    aligned = forecast_daily.reindex(future_index).ffill()
    save_series_to_csv(aligned, "aligned_regressor.csv", destination_folder='temp')

    if aligned.isna().any():
        print("⚠️ Debugging alignment failure:")
        print("Forecast index head:", forecast_daily.index[:5])
        print("Future index head:", future_index[:5])
        print("Missing values after alignment:", aligned.isna().sum())
        raise ValueError(f"NaNs remain in regressor '{column}' after resample + alignment.")

    return aligned


def series_to_prophet_df(series: pd.Series, name="y") -> pd.DataFrame:
    df = series.copy()
    df.name = name
    df = df.reset_index()
    df.columns = ["ds", name]
    return df


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

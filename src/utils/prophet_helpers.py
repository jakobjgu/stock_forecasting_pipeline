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
    # Validate input
    if "ds" not in forecast_df.columns:
        raise ValueError("forecast_df must contain a 'ds' column")
    if column not in forecast_df.columns:
        raise ValueError(f"Expected column '{column}' not found in forecast_df. Available columns: {forecast_df.columns.tolist()}")

    # # Inspect types for debugging
    # print(f"\n🔍 Aligning regressor '{column}'")
    # print("forecast_df['ds'] dtype:", forecast_df["ds"].dtype)
    # print("future_ds dtype:", future_ds.dtype)

    # Normalize dates
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"]).dt.normalize()
    future_index = pd.to_datetime(future_ds).dt.normalize()

    # Forward-fill values
    forecast_df[column] = forecast_df[column].ffill()
    forecast_df = forecast_df.dropna(subset=[column])

    # Resample to monthly frequency
    forecast_series = forecast_df.set_index("ds")[column].resample(freq).ffill()
    forecast_series.name = column
    forecast_series = forecast_series.dropna()

    # Align with future index
    aligned = forecast_series.reindex(future_index).ffill()

    # Debug output
    # print("🔢 Aligned series preview:")
    # print(aligned.head())
    # print(f"🕒 forecast_series index head: {forecast_series.index[:3]}")
    # print(f"🕒 future_index head: {future_index[:3]}")

    if aligned.isna().any():
        print("⚠️ Missing values found in aligned series:")
        print(aligned[aligned.isna()])
        raise ValueError(f"NaNs remain in regressor '{column}' after alignment")

    # Save for inspection
    save_series_to_csv(aligned, "aligned_regressor.csv", destination_folder="temp")
    return aligned


def series_to_prophet_df(series: pd.Series) -> pd.DataFrame:
    df = series.copy()
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp(how="start")
    # df.index = pd.to_datetime(df.index)  # ← force uniform datetime format
    return pd.DataFrame({"ds": df.index, "y": df.values})


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

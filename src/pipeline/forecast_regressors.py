from src.utils.preprocess import load_series_from_csv, save_series_to_csv
from src.utils.prophet_helpers import forecast_series
from pathlib import Path

def forecast_all_regressors(regressors: list[dict], freq: str, horizon: int, out_dir: Path) -> tuple[dict, dict]:
    """
    Load, preprocess, and forecast all regressors defined in the pipeline config.

    Parameters:
    - regressors: List of dicts with 'name' and 'label' for each regressor
    - freq: Resampling frequency (e.g. 'MS')
    - horizon: Number of periods to forecast
    - out_dir: Folder to save forecast CSVs

    Returns:
    - actuals: Dict of cleaned input Series
    - forecasts: Dict of forecast DataFrames
    """
    actuals, forecasts = {}, {}

    for reg in regressors:
        name = reg["name"]
        label = reg["label"]
        print(f"⏳ Processing regressor: {label}...")

        series = load_series_from_csv(f"{name}.csv", origin_folder="raw").resample(freq).ffill().bfill()
        save_series_to_csv(series, f"{name}_resampled.csv", destination_folder="processed")
        actuals[name] = series

        forecast = forecast_series(series, periods=horizon, freq=freq, name=label)
        forecast.to_csv(out_dir / f"{name}_forecast.csv", index=False)
        forecasts[name] = forecast

    return actuals, forecasts

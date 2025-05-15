from pathlib import Path
from src.utils.preprocess import load_series_from_csv
from src.models.prophet_model import forecast_series


def forecast_and_save_series(name: str, filename: str, periods: int, freq: str):
    """
    Load, forecast, and save the forecast of a univariate series.
    """
    series = load_series_from_csv(filename)
    forecast = forecast_series(series, periods=periods, freq=freq, name=name)

    output_path = Path("outputs/data/forecasts")
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / f"{name.lower()}_forecast.csv"
    forecast.to_csv(out_file, index=False)
    print(f"Saved forecast for {name} to {out_file}")


def main():
    forecast_and_save_series("CPI", "cpi.csv", periods=24, freq="MS")
    forecast_and_save_series("USD_EUR", "usd_eur.csv", periods=90, freq="B")


if __name__ == "__main__":
    main()

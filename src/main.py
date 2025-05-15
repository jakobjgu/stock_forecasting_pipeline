from src.utils.preprocess import load_series_from_csv
from src.models.prophet_model import forecast_series
from src.models.sp500_forecast_with_regressors import forecast_sp500_with_regressors
from pathlib import Path


def main():
    print("\n🔄 Step 1: Forecast CPI and USD/EUR...")
    cpi = load_series_from_csv("cpi.csv")
    usd_eur = load_series_from_csv("usd_eur.csv")

    cpi_forecast = forecast_series(cpi, periods=24, freq="MS", name="CPI")
    usd_forecast = forecast_series(usd_eur, periods=90, freq="B", name="USD_EUR")

    print("\n📈 Step 2: Forecast SP500 using CPI and USD_EUR as regressors...")
    sp500 = load_series_from_csv("sp500.csv")
    sp500_forecast = forecast_sp500_with_regressors(sp500, cpi_forecast, usd_forecast, periods=90)

    # Save outputs
    out_dir = Path("outputs/data/forecasts")
    out_dir.mkdir(parents=True, exist_ok=True)
    cpi_forecast.to_csv(out_dir / "cpi_forecast.csv", index=False)
    usd_forecast.to_csv(out_dir / "usd_eur_forecast.csv", index=False)
    sp500_forecast.to_csv(out_dir / "sp500_forecast_with_regressors.csv", index=False)
    print("\n✅ All forecasts saved to outputs/data/forecasts/")


if __name__ == "__main__":
    main()

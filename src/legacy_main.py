from src.pipeline.ingest_economic_data import fetch_economic_data_series
from src.pipeline.forecast_regressors import forecast_all_regressors
from src.pipeline.forecast_target import run_backtests
from pathlib import Path
import sys


def main(fetch=False):
    # Global Configuration
    FREQ = "MS"
    FORECAST_HORIZON = 120  # number of periods of the main FREQ. For MS, the 120 equates to a 10 year forecasting horizon
    TARGET_SERIES = {"name": "nasdaq", "label": "NASDAQ Index"}
    REGRESSORS = [
        {"name": "cpi", "label": "CPI"},
        {"name": "unrate", "label": "Unemployment Rate"},
        {"name": "umcsent", "label": "Consumer Sentiment"},
        {"name": "treasury_3m", "label": "3M Treasury"},
        {"name": "treasury_10y", "label": "10Y Treasury"},
        {"name": "yield_spread", "label": "Yield Spread"},
    ]
    CUTOFFS = ["2000-01-01", "2010-01-01", "2020-01-01"]
    OUT_DIR = Path("outputs/forecasts")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if fetch:
        print("\n🔄 Fetching latest economic data...")
        fetch_economic_data_series()
    else:
        print("\n⏩ Skipping data fetch. Using existing local files.")

    # 🔄 Step 2: Forecast regressors
    print("\n📈 Forecasting regressors...")
    actuals, forecasts = forecast_all_regressors(REGRESSORS, freq=FREQ, horizon=FORECAST_HORIZON, out_dir=OUT_DIR)

    # 🔄 Step 3–5: Run backtests for the target series
    print(f"\n📊 Running backtests for target: {TARGET_SERIES['label']}")
    run_backtests(
        target_series=TARGET_SERIES,
        actuals=actuals,
        forecasts=forecasts,
        freq=FREQ,
        cutoffs=CUTOFFS,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "false"
    if arg in {"true", "t", "1"}:
        main(fetch=True)
    elif arg in {"false", "f", "0"}:
        main(fetch=False)
    else:
        print('⚠️ Invalid input. Use "true" or "false" to indicate whether to fetch fresh FRED data.')

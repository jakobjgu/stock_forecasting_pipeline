from pathlib import Path
import sys
from src.pipeline.ingest_economic_data import fetch_economic_data_series
from src.pipeline.forecast_regressors import forecast_all_regressors
from src.models.forecaster_factory import create_forecaster
from src.utils.preprocess import load_and_trim_target_series
from src.pipeline.forecast_target import run_cutoff_backtests
from src.utils.metrics import save_backtest_metrics


def main(fetch=False, model_type="prophet"):
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

    # Step 1: Fetch data
    if fetch:
        print("\n Fetching latest economic data...")
        fetch_economic_data_series()
    else:
        print("\n Skipping data fetch. Using existing local files.")

    # Step 2: Forecast regressors
    print("\nForecasting regressors...")
    actuals, forecasts = forecast_all_regressors(REGRESSORS, freq=FREQ, horizon=FORECAST_HORIZON, out_dir=OUT_DIR)

    # Step 3: Instantiate model using factory
    forecaster = create_forecaster(
        model_type=model_type,
        target=TARGET_SERIES["name"],
        use_regressors=True
    )

    # Step 4: Load and trim target to match regressor timeframe
    target_trimmed, actuals_trimmed = load_and_trim_target_series(
        target_info=TARGET_SERIES,
        actuals=actuals,
        freq=FREQ
    )

    # Step 5: Run backtests per cutoff
    metrics_df = run_cutoff_backtests(
        target_series_info=TARGET_SERIES,
        target_trimmed=target_trimmed,
        actuals_trimmed=actuals_trimmed,
        forecasts=forecasts,
        freq=FREQ,
        cutoffs=CUTOFFS,
        out_dir=OUT_DIR,
        forecaster=forecaster,
    )

    # Step 6: Save backtest metrics
    save_backtest_metrics(metrics_df, target_name=TARGET_SERIES["name"], out_dir=OUT_DIR)


if __name__ == "__main__":
    fetch_flag = sys.argv[1].lower() if len(sys.argv) > 1 else "false"
    model_arg = sys.argv[2].lower() if len(sys.argv) > 2 else "prophet"

    if fetch_flag in {"true", "t", "1"}:
        fetch = True
    elif fetch_flag in {"false", "f", "0"}:
        fetch = False
    else:
        print('Invalid fetch input. Use "true" or "false" to indicate whether to fetch fresh data.')
        sys.exit(1)

    main(fetch=fetch, model_type=model_arg)

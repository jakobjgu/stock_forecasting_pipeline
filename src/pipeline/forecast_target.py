import pandas as pd
from pathlib import Path
from src.utils.preprocess import load_series_from_csv, save_series_to_csv
from src.utils.evaluation import evaluate_and_save_forecast
from src.models.target_forecast_with_regressors import forecast_target_with_regressors
from src.models.target_forecast import forecast_target

def run_backtests(
    target_series: dict,
    actuals: dict,
    forecasts: dict,
    freq: str,
    cutoffs: list[str],
    out_dir: Path
):
    """
    Run forecasts (with and without regressors) for a given target series across cutoff dates.

    Parameters:
    - target_series: dict with keys 'name' and 'label'
    - actuals: dict of cleaned Series
    - forecasts: dict of forecasted DataFrames
    - freq: Resample frequency
    - cutoffs: list of date strings
    - out_dir: path for saving outputs
    """
    print(f"\n📈 Loading and preprocessing target: {target_series['label']}")
    target = load_series_from_csv(f"{target_series['name']}.csv", origin_folder="raw").resample(freq).ffill().bfill()
    save_series_to_csv(target, f"{target_series['name']}_resampled.csv", destination_folder="processed")

    print("\n🧹 Trimming to common timeframe...")
    all_series = list(actuals.values()) + [target]
    min_common = max(s.index.min() for s in all_series)
    max_common = min(s.index.max() for s in all_series)

    trimmed = {
        name: s[(s.index >= min_common) & (s.index <= max_common)]
        for name, s in actuals.items()
    }
    target_trim = target[(target.index >= min_common) & (target.index <= max_common)]

    for name, series in trimmed.items():
        save_series_to_csv(series, f"{name}_trimmed.csv", destination_folder="processed")
    save_series_to_csv(target_trim, f"{target_series['name']}_trimmed.csv", destination_folder="processed")

    print("\n🔁 Running backtests...")
    metrics_log = []

    for cutoff_str in cutoffs:
        cutoff = pd.to_datetime(cutoff_str)
        print(f"🔸 Cutoff: {cutoff.date()}")

        target_train = target_trim[target_trim.index <= cutoff]
        n_periods = (2030 - cutoff.year) * 12 + (1 - cutoff.month)

        forecast_with = forecast_target_with_regressors(
            target=target_train,
            actual_regressors=trimmed,
            forecast_regressors=forecasts,
            periods=n_periods,
            freq=freq
        )

        forecast_base = forecast_target(
            target=target_train,
            periods=n_periods,
            freq=freq
        )

        metrics_with = evaluate_and_save_forecast(
            forecast_df=forecast_with,
            actual_series=target_trim,
            cutoff=cutoff,
            forecast_label=f"{target_series['name']} (with regressors)",
            save_name=f"{target_series['name']}_forecast_cutoff_{cutoff.year}",
            output_dir=out_dir
        )
        metrics_base = evaluate_and_save_forecast(
            forecast_df=forecast_base,
            actual_series=target_trim,
            cutoff=cutoff,
            forecast_label=f"{target_series['name']} (baseline)",
            save_name=f"{target_series['name']}_baseline_cutoff_{cutoff.year}",
            output_dir=out_dir
        )

        metrics_with["model"] = "with_regressors"
        metrics_base["model"] = "baseline"
        metrics_log += [metrics_with, metrics_base]

    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(out_dir / f"{target_series['name']}_backtest_metrics.csv", index=False)
    print(f"\n📊 Saved metrics to outputs/data/forecasts/{target_series['name']}_backtest_metrics.csv")

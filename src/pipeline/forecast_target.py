import pandas as pd
from pathlib import Path
from src.utils.evaluation import evaluate_and_save_forecast
from src.models.forecaster_definitions import ProphetForecaster


def run_cutoff_backtests(
    target_series_info: dict,
    target_trimmed: pd.Series,
    actuals_trimmed: dict[str, pd.Series],
    forecasts: dict[str, pd.DataFrame],
    freq: str,
    cutoffs: list[str],
    out_dir: Path,
    forecaster,
) -> None:
    """
    Run backtests across multiple cutoff dates using a trained forecasting model.

    Parameters:
    - target_series_info: dict with keys 'name' and 'label'
    - target_trimmed: pd.Series, trimmed target series
    - actuals_trimmed: dict of trimmed actual regressor series
    - forecasts: dict of forecasted regressor DataFrames
    - freq: time frequency (e.g., 'MS')
    - cutoffs: list of cutoff date strings
    - out_dir: output directory path
    - forecaster: a model instance supporting .fit(), .predict(), and .reset()
    """
    metrics_log = []

    print("📅 Final trimmed range:")
    print("Target:", target_trimmed.index.min(), "→", target_trimmed.index.max())
    for k, v in actuals_trimmed.items():
        print(f"{k}: {v.index.min()} → {v.index.max()}")

    for cutoff_str in cutoffs:
        cutoff = pd.to_datetime(cutoff_str)
        print(f"\n🔸 Cutoff: {cutoff.date()}")

        target_train = target_trimmed[target_trimmed.index <= cutoff]
        if target_train.empty:
            print(f"⚠️ Skipping cutoff {cutoff.date()} — no data in target series before this date.")
            continue

        n_periods = (2030 - cutoff.year) * 12 + (1 - cutoff.month)

        forecaster.reset()
        forecaster.fit(target=target_train, actual_regressors=actuals_trimmed)
        forecast_with = forecaster.predict(periods=n_periods, forecast_regressors=forecasts)

        model_name = forecaster.name.lower().replace(" ", "_")
        label = f"{target_series_info['name']} ({model_name} with regressors)"
        save_name = f"{target_series_info['name']}_{model_name}_forecast_cutoff_{cutoff.year}"

        metrics_with = evaluate_and_save_forecast(
            forecast_df=forecast_with,
            actual_series=target_trimmed,
            cutoff=cutoff,
            forecast_label=label,
            save_name=save_name,
            output_dir=out_dir,
        )
        metrics_with["model"] = f"{model_name}_with_regressors"
        metrics_log.append(metrics_with)

        if forecaster.__class__.__name__.lower().startswith("prophet"):
            baseline_forecaster = ProphetForecaster(
                target=target_series_info["name"],
                use_regressors=False,
                freq=freq
            )
            baseline_forecaster.fit(target=target_train)
            forecast_base = baseline_forecaster.predict(periods=n_periods)

            metrics_base = evaluate_and_save_forecast(
                forecast_df=forecast_base,
                actual_series=target_trimmed,
                cutoff=cutoff,
                forecast_label=f"{target_series_info['name']} ({model_name} baseline)",
                save_name=f"{target_series_info['name']}_{model_name}_baseline_cutoff_{cutoff.year}",
                output_dir=out_dir,
            )
            metrics_base["model"] = f"{model_name}_baseline"
            metrics_log.append(metrics_base)

    metrics_df = pd.DataFrame(metrics_log)

    return metrics_df

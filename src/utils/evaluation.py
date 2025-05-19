from pathlib import Path
import pandas as pd
from src.utils.plotting import plot_forecast_with_actuals
from src.utils.error_calculations import compute_forecast_errors

def evaluate_and_save_forecast(
    forecast_df: pd.DataFrame,
    actual_series: pd.Series,
    cutoff: pd.Timestamp,
    forecast_label: str,
    save_name: str,
    output_dir: Path
) -> dict:
    """
    Evaluate forecast against actuals, save results, and generate plot.

    Parameters:
    - forecast_df: pd.DataFrame
        The forecasted data from Prophet.
    - actual_series: pd.Series
        The actual SP500 data to compare against.
    - cutoff: pd.Timestamp
        The training cutoff date.
    - label: str
        Label for the plot and output.
    - save_name: str
        Base filename (without extension) to save CSV and PNG.
    - output_dir: Path
        Directory where output files should be saved.

    Returns:
    - dict with keys: 'cutoff', 'mae', 'rmse'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save forecast CSV
    forecast_path = output_dir / f"{save_name}.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # Plot forecast
    plot_forecast_with_actuals(
        forecast_df=forecast_df,
        actual_series=actual_series,
        label=forecast_label,
        title=f"{forecast_label} (cutoff {cutoff.year})",
        train_cutoff=cutoff,
        save_path=f"outputs/figures/{save_name}.png"
    )

    # Evaluate forecast
    forecast_df = forecast_df.set_index("ds")
    y_pred = forecast_df.loc[cutoff:, "yhat"]
    y_true = actual_series[actual_series.index >= cutoff]

    aligned = y_true.loc[y_true.index.isin(y_pred.index)]
    y_pred = y_pred.loc[y_pred.index.isin(aligned.index)]

    mae, rmse = compute_forecast_errors(aligned, y_pred)

    print(f"📉 {forecast_label} MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return {"cutoff": str(cutoff.date()), "mae": mae, "rmse": rmse}

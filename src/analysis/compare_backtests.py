import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_backtests(target_name: str):
    """
    Compare forecasts with and without regressors for a given target series.
    Generates RMSE bar plots and forecast subplots for each cutoff date.

    Parameters:
    - target_name (str): The name of the target series (e.g., 'sp500', 'nasdaq')
    """

    metrics_path = Path(f"outputs/forecasts/{target_name}_backtest_metrics.csv")
    forecast_dir = Path("outputs/forecasts")
    actual_path = Path(f"data/processed/{target_name}_trimmed.csv")

    metrics_df = pd.read_csv(metrics_path)
    actual_series = pd.read_csv(actual_path, index_col=0, parse_dates=True).squeeze()

    cutoffs = sorted(metrics_df["cutoff"].unique())
    colors = sns.color_palette("tab10", n_colors=len(cutoffs))
    color_map = dict(zip(cutoffs, colors))
    end_forecast_date = pd.to_datetime("2030-01-01")

    # --- RMSE comparison bar chart ---
    pivot_df = metrics_df.pivot(index="cutoff", columns="model", values="rmse").reset_index()
    x = range(len(pivot_df))
    bar_width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - bar_width / 2 for i in x], pivot_df["baseline"], width=bar_width, label="Baseline", color="gray")
    plt.bar([i + bar_width / 2 for i in x], pivot_df["with_regressors"], width=bar_width, label="With Regressors", color="steelblue")
    plt.xticks(ticks=x, labels=pivot_df["cutoff"])
    plt.xlabel("Cutoff Year")
    plt.ylabel("RMSE")
    plt.title(f"{target_name.upper()} Forecast RMSE: Baseline vs With Regressors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{target_name}_rmse_comparison_bar.png")
    plt.show()

    # --- Forecast subplots ---
    fig_height = 4 * len(cutoffs) * 0.7
    fig, axes = plt.subplots(len(cutoffs), 1, figsize=(14, fig_height), sharex=True)
    if len(cutoffs) == 1:
        axes = [axes]

    for ax, cutoff in zip(axes, cutoffs):
        cutoff_date = pd.to_datetime(cutoff)
        color = color_map[cutoff]

        # Actual
        ax.plot(actual_series.index, actual_series.values, label="Actual", color="black", linewidth=1.5)

        # With regressors
        forecast_file = forecast_dir / f"{target_name}_forecast_cutoff_{cutoff_date.year}.csv"
        forecast_df = pd.read_csv(forecast_file, parse_dates=["ds"])
        forecast_df = forecast_df[forecast_df["ds"] <= end_forecast_date]
        ax.plot(forecast_df["ds"], forecast_df["yhat"], label="With Regressors", color=color)

        if "yhat_lower" in forecast_df.columns:
            ax.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                            color=color, alpha=0.2, label="Uncertainty")

        # Baseline
        baseline_file = forecast_dir / f"{target_name}_baseline_cutoff_{cutoff_date.year}.csv"
        if baseline_file.exists():
            baseline_df = pd.read_csv(baseline_file, parse_dates=["ds"])
            baseline_df = baseline_df[baseline_df["ds"] <= end_forecast_date]
            ax.plot(baseline_df["ds"], baseline_df["yhat"], linestyle="--", label="No Regressors", color=color)

        # Forecast horizon
        ax.axvspan(cutoff_date, end_forecast_date, color="gray", alpha=0.05)
        ax.set_title(f"{target_name.upper()} Forecast from Cutoff {cutoff}")
        ax.set_ylabel(f"{target_name.upper()} Index")
        ax.grid(True)
        ax.legend(loc="upper left")

        # Metrics
        row_reg = metrics_df[(metrics_df["cutoff"] == cutoff) & (metrics_df["model"] == "with_regressors")].squeeze()
        row_base = metrics_df[(metrics_df["cutoff"] == cutoff) & (metrics_df["model"] == "baseline")].squeeze()

        if not row_reg.empty and not row_base.empty:
            text = (
                f"With Regressors → MAE: {row_reg['mae']:.2f} | RMSE: {row_reg['rmse']:.2f}\n"
                f"No Regressors   → MAE: {row_base['mae']:.2f} | RMSE: {row_base['rmse']:.2f}"
            )
            ax.text(
                0.5, 0.98, text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
            )

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{target_name}_forecast_subplots.png")
    plt.show()


# Entry point
if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "sp500"
    compare_backtests(target_name=arg)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.style.use("fivethirtyeight")  # Optional but nice

def plot_forecast_with_actuals(
    forecast_df: pd.DataFrame,
    actual_series: pd.Series,
    label: str = "Forecast",
    title: str = "Forecast vs Actual",
    column: str = "yhat",
    train_cutoff: pd.Timestamp = None,
    save_path: str = None
):
    """
    Plot Prophet forecast alongside actual historical values, including uncertainty intervals.

    Parameters:
    - forecast_df: Prophet forecast output with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    - actual_series: historical pd.Series with datetime index
    - label: label for the forecast line
    - title: plot title
    - column: forecast_df column to use (default: 'yhat')
    """

    forecast = forecast_df.set_index("ds")

    plt.figure(figsize=(12, 5))
    
    # Plot actual values
    plt.plot(actual_series, label="Actual", linewidth=2)

    # Plot forecast line
    plt.plot(forecast[column], label=label, linestyle="--")

    # Plot uncertainty band
    if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
        plt.fill_between(
            forecast.index,
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="lightblue",
            alpha=0.3,
            label="Uncertainty"
        )

    # Forecast start marker
    plt.axvline(actual_series.index.max(), color="gray", linestyle=":", label="Forecast start")
    if train_cutoff:
        plt.axvline(train_cutoff, color="red", linestyle="--", label="Train/Test Split")
        plt.axvspan(train_cutoff, forecast_df["ds"].max(), color="gray", alpha=0.1, label="Test Period")


    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📁 Saved plot to: {save_path}")
    
    plt.show()

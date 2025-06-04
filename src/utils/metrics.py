import pandas as pd
from pathlib import Path

def save_backtest_metrics(metrics_df: pd.DataFrame, target_name: str, out_dir: Path) -> None:
    """
    Save a DataFrame of backtest metrics to disk.
    """
    path = out_dir / f"{target_name}_backtest_metrics.csv"
    metrics_df.to_csv(path, index=False)
    print(f"\n📈 Saved metrics to {path}")

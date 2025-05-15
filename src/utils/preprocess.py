import pandas as pd
from typing import Dict
from settings import ROOT_PATH
from pathlib import Path

def load_series_from_csv(filename: str, folder: Path = ROOT_PATH / "data" / "raw") -> pd.Series:
    """
    Load a time series CSV from a consistent project-root-relative location.
    """
    filepath = folder / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    return df["value"]


def preprocess_series(
    series: pd.Series,
    freq: str = "B",        # business daily frequency
    method: str = "ffill",  # 'ffill', 'bfill', 'interpolate'
) -> pd.Series:
    """
    Resample and fill missing values in a time series.
    - freq: target frequency ('B', 'D', 'M', etc.)
    - method: how to fill missing values
    """
    series = series.sort_index()
    series = series.asfreq(freq)

    if method == "interpolate":
        series = series.interpolate()
    elif method == "ffill":
        series = series.ffill()
    elif method == "bfill":
        series = series.bfill()
    else:
        raise ValueError(f"Unknown fill method: {method}")

    return series



def add_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Add percent change over the specified lag."""
    return series.pct_change(periods=periods)


def merge_series_dict(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Merge multiple time series (with datetime index) into a single DataFrame.
    Each key in the dict becomes a column name.
    """
    df = pd.concat(series_dict.values(), axis=1)
    df.columns = list(series_dict.keys())
    return df


def add_derived_features(
    df: pd.DataFrame,
    pct_change_lag: int = 1,
    rolling_windows: list = [5, 20],
) -> pd.DataFrame:
    """
    Add derived features:
    - Percent change (returns)
    - Rolling mean and std for specified windows

    Returns a new DataFrame with original + derived columns.
    """
    df_features = df.copy()

    for col in df.columns:
        # Percent change
        df_features[f"{col}_pct_change"] = df[col].pct_change(periods=pct_change_lag, fill_method=None)

        # Rolling features
        for window in rolling_windows:
            df_features[f"{col}_rollmean_{window}"] = df[col].rolling(window=window).mean()
            df_features[f"{col}_rollstd_{window}"] = df[col].rolling(window=window).std()

    return df_features


def train_test_split_time_series(
    df: pd.DataFrame,
    split_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series DataFrame into train and test sets while preserving order.
    
    Parameters:
    - df: full DataFrame with time index
    - split_ratio: fraction of data to use for training (default 80%)

    Returns:
    - (train_df, test_df)
    """
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test

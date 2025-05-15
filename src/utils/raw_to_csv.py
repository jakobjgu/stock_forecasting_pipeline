import pandas as pd
from pathlib import Path
from settings import ROOT_PATH

def save_series_to_csv(series: pd.Series, filename: str, folder: Path = ROOT_PATH / "data" / "processed"):
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / filename
    df = series.to_frame(name="value")
    df.to_csv(filepath, index_label="date")
    print(f"Saved to: {filepath}")


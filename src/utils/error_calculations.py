from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def compute_forecast_errors(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """Compute MAE and RMSE from two aligned series."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse
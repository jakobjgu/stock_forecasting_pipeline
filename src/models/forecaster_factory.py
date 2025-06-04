from src.models.forecaster_definitions import ProphetForecaster
from src.models.forecaster_definitions import XGBoostForecaster

def create_forecaster(model_type: str, target: str, use_regressors: bool = True):
    """
    Factory to return a new forecaster instance based on model type.

    Parameters:
    - model_type: e.g. "prophet" or "xgboost"
    - target: name of the target series (e.g., 'sp500')
    - use_regressors: whether to include regressors (only applies to Prophet)

    Returns:
    - An instance of ProphetForecaster or XGBoostForecaster
    """
    model_type = model_type.lower()

    if model_type == "prophet":
        return ProphetForecaster(target=target, use_regressors=use_regressors)
    elif model_type == "xgboost":
        return XGBoostForecaster(target=target)
    elif model_type.startswith("xgboost_minimal"):
        return XGBoostForecaster()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

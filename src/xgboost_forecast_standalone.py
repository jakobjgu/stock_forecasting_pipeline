import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Load target and regressor data
def load_series(file_path):
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df["value"]

# Forecast each regressor with Prophet
def forecast_regressor(series, periods, freq='MS'):
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"]

# Build features with lags and optional regressors
def make_features(series, lags=12, regressors=None):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    if regressors:
        for name, reg_series in regressors.items():
            df[name] = reg_series.reindex(df.index)
    return df.dropna()

# Train XGBoost model
def train_xgboost(X, y):
    model = XGBRegressor(n_estimators=200, objective="reg:squarederror")
    model.fit(X, y)
    return model

# Forecast with trained model
def forecast_xgboost(model, y_hist, periods, lags=12, regressors=None, freq="MS"):
    forecast = []
    index = pd.date_range(start=y_hist.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)

    for current_date in index:
        lag_features = [y_hist.iloc[-i] for i in range(1, lags + 1)][::-1]
        if regressors:
            reg_values = []
            for name in regressors:
                reg_series = regressors[name]
                val = reg_series.get(current_date, np.nan)
                if np.isnan(val):
                    break  # Skip prediction if any regressor value is missing
                reg_values.append(val)
            else:
                features = np.array(lag_features + reg_values).reshape(1, -1)
                y_pred = model.predict(features)[0]
                forecast.append({"ds": current_date, "yhat": y_pred})
                y_hist.loc[current_date] = y_pred
                continue
            break  # exit if break in inner loop
        else:
            features = np.array(lag_features).reshape(1, -1)
            y_pred = model.predict(features)[0]
            forecast.append({"ds": current_date, "yhat": y_pred})
            y_hist.loc[current_date] = y_pred

    return pd.DataFrame(forecast)

# Load data
target = load_series("data/processed/nasdaq_trimmed.csv")
regressors_raw = {
    "cpi": load_series("data/processed/cpi_trimmed.csv"),
    "unrate": load_series("data/processed/unrate_trimmed.csv"),
    "umcsent": load_series("data/processed/umcsent_trimmed.csv")
}

# Parameters
LAGS = 12
FREQ = "MS"
PERIODS = 60

# Forecast regressors with Prophet
forecasted_regressors = {
    name: forecast_regressor(series, periods=PERIODS, freq=FREQ) for name, series in regressors_raw.items()
}

# Align training data
df_train = make_features(target, lags=LAGS, regressors=regressors_raw)
X_train = df_train.drop(columns=["y"])
y_train = df_train["y"]

# Train models
model_with_regressors = train_xgboost(X_train, y_train)

df_train_no_reg = make_features(target, lags=LAGS, regressors=None)
X_train_no_reg = df_train_no_reg.drop(columns=["y"])
y_train_no_reg = df_train_no_reg["y"]
model_no_regressors = train_xgboost(X_train_no_reg, y_train_no_reg)

# Forecasts
forecast_with_reg = forecast_xgboost(model_with_regressors, target.copy(), PERIODS, lags=LAGS, regressors=forecasted_regressors, freq=FREQ)
forecast_no_reg = forecast_xgboost(model_no_regressors, target.copy(), PERIODS, lags=LAGS, regressors=None, freq=FREQ)

target.index = pd.to_datetime(target.index)
forecast_no_reg["ds"] = pd.to_datetime(forecast_no_reg["ds"])
forecast_with_reg["ds"] = pd.to_datetime(forecast_with_reg["ds"])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(forecast_with_reg["ds"], forecast_with_reg["yhat"], label="With Regressors (XGBoost)", color="green")
plt.plot(forecast_no_reg["ds"], forecast_no_reg["yhat"], label="No Regressors (XGBoost)", color="blue")
target.plot(label="historical", color="black")
plt.legend()
plt.title("NASDAQ Forecast: XGBoost With vs Without Regressors")
plt.grid(True)
plt.tight_layout()

# Save results
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/xgboost_nasdaq_forecast_comparison.png")

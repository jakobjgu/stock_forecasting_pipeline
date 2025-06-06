from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional
from prophet import Prophet
from xgboost import XGBRegressor
from src.utils.prophet_helpers import (
    prepare_prophet_input,
    align_forecast_regressor,
    series_to_prophet_df,
)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    Each forecaster should implement the fit and predict methods.
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def fit(self, series: pd.Series, exogenous: dict[str, pd.Series] = None):
        """
        Fit the forecasting model.

        Parameters:
        - series: pd.Series
            Time series to forecast. Index must be datetime.
        - exogenous: dict[str, pd.Series], optional
            Dictionary of exogenous variables. Each value is a time-aligned series.
        """
        pass

    @abstractmethod
    def predict(self, periods: int, freq: str) -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.

        Parameters:
        - periods: int
            Number of periods to forecast.
        - freq: str
            Frequency string (e.g., "MS" for month-start).

        Returns:
        - pd.DataFrame
            Forecast with at least a 'ds' column (dates) and a 'yhat' column (forecasted values).
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state between runs (e.g., model object, flags, cached data)."""
        pass


class ProphetForecaster(BaseForecaster):
    def __init__(self, target: str, use_regressors: bool = False, freq: str = "MS", name: str = None):
        """
        Prophet-based forecasting class supporting optional external regressors.

        Attributes are set during `.fit()` and `.predict()`.
        This class is compatible with multiple forecasting models by separating training and prediction steps.
        """
        super().__init__(name=name or f"Prophet_{target}")
        self.target_name = target
        self.use_regressors = use_regressors
        self.freq = freq
        self.model = Prophet()
        self.fitted = False
        self.regressor_names = []
        
    def reset(self):
        self.model = Prophet()
        self.df = None
        self.fitted = False
        self.regressor_names = []

    def fit(self, target: pd.Series, actual_regressors: Optional[dict[str, pd.Series]] = None):
        """
        Fit the Prophet model to the target series with optional external regressors.
        """
        self.df = prepare_prophet_input(target)

        if self.use_regressors and actual_regressors:
            for name, series in actual_regressors.items():
                reg_df = series_to_prophet_df(series)
                aligned = align_forecast_regressor(reg_df, self.df["ds"], column="y", freq=self.freq)

                # print(f"✅ Adding regressor '{name}' to training data...")
                # print(aligned.head())

                self.df[name] = aligned.values
                self.model.add_regressor(name)
                self.regressor_names.append(name)

        self.model.fit(self.df.dropna())
        self.fitted = True

    def predict(
        self,
        periods: int,
        forecast_regressors: Optional[dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Generate forecast from the trained model, optionally with forecasted regressors.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before calling predict().")

        future = self.model.make_future_dataframe(periods=periods, freq=self.freq)

        if self.use_regressors and forecast_regressors:
            for name in self.regressor_names:
                forecast_df = forecast_regressors.get(name)
                if forecast_df is not None:
                    aligned = align_forecast_regressor(forecast_df, future["ds"], column="yhat", freq=self.freq)
                    future[name] = aligned.values

        forecast = self.model.predict(future)
        return forecast


# class XGBoostForecaster(BaseForecaster):
#     def __init__(self, target: str, name: str = None, lags: int = 12):
#         super().__init__(name=name or f"XGBoost_{target}")
#         self.target = target
#         self.lags = lags
#         self.model = XGBRegressor(
#             n_estimators=300,
#             max_depth=4,
#             learning_rate=0.05,
#             objective="reg:squarederror"
#         )
#         self.train_series = None
#         self.actual_regressors = None
#         self.forecast_regressors = None
#         self.fitted = False

#     def reset(self):
#         self.model = XGBRegressor(
#             n_estimators=300,
#             max_depth=4,
#             learning_rate=0.05,
#             objective="reg:squarederror"
#         )
#         self.train_series = None
#         self.actual_regressors = None
#         self.forecast_regressors = None
#         self.fitted = False

#     def _make_features(
#         self,
#         series: pd.Series,
#         regressors: Optional[dict[str, pd.Series]]
#     ) -> pd.DataFrame:
#         df = pd.DataFrame({"y": series})
#         for lag in range(1, self.lags + 1):
#             df[f"lag_{lag}"] = df["y"].shift(lag)

#         if regressors:
#             for name, r in regressors.items():
#                 df[name] = r.reindex(df.index)

#         df = df.dropna()
#         self.scaler = StandardScaler()
#         scaled = self.scaler.fit_transform(df)

#         return pd.DataFrame(scaled, index=df.index, columns=df.columns)

#     def fit(self, target: pd.Series, actual_regressors: Optional[dict[str, pd.Series]] = None):
#         self.train_series = target
#         self.actual_regressors = actual_regressors

#         df = self._make_features(target, actual_regressors)
#         X_train = df.drop(columns=["y"])

#         # Fit the scaler only on X_train (not including "y")
#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X_train)

#         self.model.fit(X_scaled, df["y"])
#         self.fitted = True
#         self.X_train = X_train
#         self.fitted_values = self.model.predict(X_scaled)
#         self.fitted_index = X_train.index


#     def predict(
#         self,
#         periods: int,
#         freq: str = "MS",
#         forecast_regressors: Optional[dict[str, pd.DataFrame]] = None
#     ) -> pd.DataFrame:
#         if not self.fitted:
#             raise RuntimeError("XGBoost model must be fit before calling predict().")

#         self.forecast_regressors = forecast_regressors
#         y_hist = self.train_series.copy()
#         future_index = pd.date_range(start=y_hist.index[-1] + pd.tseries.frequencies.to_offset(freq),
#                                      periods=periods, freq=freq)
#         forecast_values = []

#         feature_cols = list(self.scaler.feature_names_in_)
#         if "y" in feature_cols:
#             feature_cols.remove("y")

#         for date in future_index:
#             lag_features = [y_hist.iloc[-i] for i in range(1, self.lags + 1)][::-1]
#             reg_features = []
#             if forecast_regressors:
#                 for name, df in forecast_regressors.items():
#                     reg_df = df.set_index("ds")
#                     reg_val = reg_df.loc[date, "yhat"] if date in reg_df.index else np.nan
#                     reg_features.append(reg_val)

#             X_input = pd.DataFrame([lag_features + reg_features], columns=feature_cols)
           
#             # Scale and predict
#             X_scaled = self.scaler.transform(X_input)

#             y_pred = self.model.predict(X_scaled)[0]

#             forecast_values.append({"ds": date, "yhat": y_pred})
#             y_hist.loc[date] = y_pred

#         forecast_df = pd.DataFrame(forecast_values)
#         fitted_df = pd.DataFrame({
#             "ds": self.fitted_index,
#             "yhat": self.fitted_values
#         })

#         return pd.concat([fitted_df, forecast_df], ignore_index=True)
    

class XGBoostForecaster(BaseForecaster):
    def __init__(self, name: str = "XGBoost", lags: int = 12, use_regressors: bool = True):
        super().__init__(name=name)
        self.lags = lags
        self.use_regressors = use_regressors
        self.model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, objective="reg:squarederror")
        self.scaler = None
        self.fitted = False

    def reset(self):
        self.__init__(name=self.name, lags=self.lags, use_regressors=self.use_regressors)

    def _make_features(self, series: pd.Series, regressors: Optional[dict[str, pd.Series]]) -> pd.DataFrame:
        df = pd.DataFrame({"y": series})
        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)

        if self.use_regressors and regressors:
            for name, r in regressors.items():
                df[name] = r.reindex(df.index)

        df = df.dropna()
        return df

    def fit(self, target: pd.Series, actual_regressors: Optional[dict[str, pd.Series]] = None):
        df = self._make_features(target, actual_regressors)
        X_train = df.drop(columns=["y"])
        y_train = df["y"]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.train_series = target
        self.actual_regressors = actual_regressors
        self.fitted = True
        self.X_train_index = X_train.index
        self.y_fitted = self.model.predict(X_scaled)

        self.fitted_values = self.model.predict(X_train)
        self.fitted_index = y_train.index

    def predict(
        self,
        periods: int,
        freq: str = "MS",
        forecast_regressors: Optional[dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("XGBoost model must be fit before calling predict().")

        y_hist = self.train_series.copy()
        future_index = pd.date_range(start=y_hist.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                    periods=periods, freq=freq)

        forecast_values = []
        all_inputs = []  # for diagnostic drift check

        for date in future_index:
            lag_features = [y_hist.iloc[-i] for i in range(1, self.lags + 1)][::-1]
            reg_features = []

            if forecast_regressors:
                for name, df in forecast_regressors.items():
                    reg_df = df.set_index("ds")
                    val = reg_df.loc[date, "yhat"] if date in reg_df.index else np.nan
                    reg_features.append(val)

            features = lag_features + reg_features
            X_input = pd.DataFrame([features], columns=self.scaler.feature_names_in_)

            # NaN check
            if X_input.isnull().any().any():
                print(f"❌ NaNs in prediction input at {date}:")
                print(X_input)
                continue

            # Scale + Predict
            X_scaled = self.scaler.transform(X_input)
            y_pred = self.model.predict(X_scaled)[0]

            # Logging for diagnostics
            print(f"\n📆 Predicting for: {date}")
            print("Lag features:", np.round(lag_features, 2))
            print("Regressor features:", np.round(reg_features, 2))
            print("→ Raw prediction:", y_pred)

            forecast_values.append({"ds": date, "yhat": y_pred})
            y_hist.loc[date] = y_pred  # roll forward for next step
            all_inputs.append(features)

        # Diagnostics
        if all_inputs:
            drift = np.array(all_inputs[-1]) - np.array(all_inputs[0])
            print("\n📉 Input drift (last - first):", np.round(drift, 2))

        forecast_df = pd.DataFrame(forecast_values)
        fitted_df = pd.DataFrame({
            "ds": self.fitted_index,
            "yhat": self.fitted_values
        })

        return pd.concat([fitted_df, forecast_df], ignore_index=True)


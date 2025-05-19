# 📈 Forecasting Pipeline: Process Overview

This project builds a reproducible economic forecasting pipeline using the FRED and Yahoo Finance APIs, Prophet models, and configurable regressors. Below is a step-by-step walkthrough of how the pipeline operates.

---

## ✅ Step 1: Run the Pipeline

Use:

```bash
python -m src.main [true|false]
```

Pass `true` (or `t`, `1`) to fetch the latest data, or `false`, `f`, or `0` to use existing local files.

---

## 🔄 Step 2: Fetch Economic Data (Optional)

If the `fetch=True` flag is used:

- The main script calls `fetch_economic_data_series` from `src.pipeline.ingest_economic_data`
- That script imports individual functions from `src.utils.fetch_economic_data`
- Each function (e.g., `fetch_full_sp500_history`, `fetch_unemployment_rate`, etc.) loads a specific indicator from FRED or Yahoo Finance
- The resulting `pandas.Series` objects are saved as `.csv` files in `data/raw/`

---

## 🧼 Step 3: Resample and Forecast Regressors

For each regressor (CPI, UNRATE, UMCSENT, etc.), the main script calls `forecast_all_regressors` from `src.pipeline.forecast_regressors`, which in turn:

- Loads the `.csv` from `data/raw/`
- Resamples to the global frequency (`FREQ`) set in `main.py`
- Fills missing values using forward- and back-fill
- Saves the cleaned series to `data/processed/`
- Forecasts 120 future periods (or any configured horizon)
- Saves the forecast to `outputs/data/forecasts/`

The number of forecast periods is dynamically computed from the current date to the configured forecast horizon (e.g., year 2030), ensuring flexibility across targets.

---

## 📊 Step 4: Process Target Series

The main script calls `run_backtests` from `src.pipeline.forecast_target`, which executes the following steps:
- Load the target series (e.g., SP500, NASDAQ) from `data/raw/<target_name>.csv`, based on the configuration in `main.py`
- Resample to the global frequency (`FREQ`)
- Save the cleaned series to `data/processed/<target_name>_resampled.csv`
- Perform alignment, forecasting, and evaluation as detailed in steps 5–7

---

## 🧭 Step 5: Align All Series to a Common Time Range

Before regressors can be used for forecasting, all series (target + regressors) must be trimmed to a **shared time window** where data is available for all of them. This ensures valid training data.

**Details:**
- Each series (e.g., SP500, CPI, UMCSENT) is trimmed to the overlap of their available dates.
- The trimmed series are saved to `data/processed/*_trimmed.csv` for inspection.

**Important Alignment Logic:**
- When the `regressor-enhanced Prophet model` is trained or forecasted, it relies on the function `align_forecast_regressor()` from `src/utils/prophet_helpers.py`.
- This utility:
  - Converts forecast DataFrames into time-aligned `pd.Series`
  - Resamples to the global frequency (`FREQ`) using forward-fill
  - Aligns regressor data to the target forecast horizon (`future["ds"]`)
  - Raises an explicit error if `NaN`s remain after alignment — helping catch subtle issues early

This alignment step was one of the most complex parts of the pipeline, especially when mixing monthly and daily data, or series with differing start dates.

---

## 🔁 Step 6: Backtesting Loop

For each specified cutoff date (e.g., `2000-01-01`, `2010-01-01`, `2020-01-01`), the pipeline:

1. Splits the target series into training (pre-cutoff) and test (post-cutoff)
2. Trains and forecasts using two models:
   - A baseline Prophet model (no regressors)
   - A regressor-enhanced Prophet model
3. Saves forecasts to clearly named `.csv` files:
   - `<target>_forecast_cutoff_<year>.csv`
   - `<target>_baseline_cutoff_<year>.csv`
4. Generates visualizations with uncertainty intervals
5. Computes **MAE** and **RMSE**
6. Logs the metrics for later comparison

---

## 📁 Outputs

- Forecasts saved as `.csv` in `outputs/data/forecasts/`
- Figures saved as `.png` in `outputs/figures/`
- Backtest metrics logged to `target_backtest_metrics.csv`

---

## 🔍 Step 7: Evaluate Forecast Accuracy

For each forecast (with and without regressors), the pipeline evaluates two core metrics:

- **MAE** (Mean Absolute Error): average magnitude of errors, regardless of direction
- **RMSE** (Root Mean Squared Error): penalizes larger errors more heavily

These metrics are computed on the forecast period after each cutoff and stored in:

```
outputs/data/forecasts/target_backtest_metrics.csv
```

The script `src.analysis.compare_backtests.py` uses this file to generate:
- RMSE bar charts comparing model performance
- Annotated subplots showing accuracy across time

---

## 🚀 Step 8: Running the Full Pipeline

To run the pipeline end-to-end:

1. **Ensure economic indicators are defined**  
   Each indicator (regressor or target) must have a corresponding fetch function in:  
   `src/utils/fetch_economic_data.py`

2. **Set pipeline configuration**  
   At the top of `main.py`, define:
   - `FREQ`: the global sampling frequency (e.g. `"MS"` or `"B"`)
   - Forecasting horizon: e.g., 120 months

3. **Configure regressors and target**  
   Define:
   - Regressors in a list of dictionaries in `main.py`
   - Target series (e.g., SP500, NASDAQ) as a dictionary

4. **Run the pipeline**  
   From the project root:

   ```bash
   python -m src.main true    # fetch fresh data  
   python -m src.main false   # use existing .csv files  
   ```

5. **Visualize results**  
   Generate summary plots:

   ```bash
   python -m src.analysis.compare_backtests <target_series_name>  # defaults to sp500 if no argument is provided
   ```

   This creates:
   - Subplots of actual vs predicted forecasts across cutoff years
   - RMSE comparison bar chart

---

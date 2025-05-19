### 📐 Forecast Error Metrics

#### Mean Absolute Error (MAE)

**Definition**: The average of the absolute differences between predicted and actual values.

**Formula**:
MAE = (1/n) * sum(|y_i - ŷ_i|)

**Interpretation**:
- MAE is measured in the same units as the target variable (e.g., SP500 index points).
- It gives a straightforward average error magnitude.
- Less sensitive to large outliers than RMSE.
- MAE = 100 means the forecast is off by 100 points on average.

---

#### Root Mean Squared Error (RMSE)

**Definition**: The square root of the average of the squared differences between predicted and actual values.

**Formula**:
RMSE = sqrt((1/n) * sum((y_i - ŷ_i)^2))

**Interpretation**:
- Also measured in the same units as the target.
- RMSE penalizes large errors more than MAE due to squaring.
- More sensitive to extreme deviations.
- Useful when large errors are especially costly or informative.

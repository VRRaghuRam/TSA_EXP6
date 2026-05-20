# Ex.No: 6               HOLT WINTERS METHOD
### Date: 20-05-2026

## NAME : Raghu Ram VR
## REG.NO : 212224220075




### AIM:
   To implement Hold vinters method for time series and analysis.

## REQUIREMENTS:
   Dataset - salesdata.csv
   Software - google collab
   
   
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# ==========================================
# LOAD DATASET
# ==========================================
data = pd.read_csv(
    "marketing_campaign_performance_10000.csv",
    parse_dates=['StartDate']
)

# ==========================================
# DISPLAY DATA
# ==========================================
print("First 5 Rows:\n")
print(data.head())

print("\nColumn Names:\n")
print(data.columns)

# ==========================================
# SET DATE COLUMN AS INDEX
# ==========================================
data.set_index('StartDate', inplace=True)

# ==========================================
# MONTHLY RESAMPLING
# ==========================================
data_monthly = data[['Revenue_USD']].resample('MS').sum()

print("\nMonthly Data:\n")
print(data_monthly.head())

print("\nTotal Monthly Records:", len(data_monthly))

# ==========================================
# PLOT ORIGINAL DATA
# ==========================================
plt.figure(figsize=(12,5))

plt.plot(
    data_monthly.index,
    data_monthly['Revenue_USD']
)

plt.title('Monthly Revenue')

plt.xlabel('Date')

plt.ylabel('Revenue_USD')

plt.grid(True)

plt.show()

# ==========================================
# SCALE DATA
# ==========================================
scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(
    data_monthly.values.reshape(-1,1)
)

scaled_data = pd.Series(
    scaled_values.flatten(),
    index=data_monthly.index
)

# ==========================================
# PLOT SCALED DATA
# ==========================================
plt.figure(figsize=(12,5))

plt.plot(
    scaled_data.index,
    scaled_data
)

plt.title('Scaled Revenue Data')

plt.xlabel('Date')

plt.ylabel('Scaled Revenue')

plt.grid(True)

plt.show()

# ==========================================
# SEASONAL DECOMPOSITION
# Only if enough observations exist
# ==========================================
if len(data_monthly) >= 6:

    decomposition = seasonal_decompose(
        data_monthly,
        model='additive',
        period=2
    )

    decomposition.plot()

    plt.show()

# ==========================================
# AVOID ZERO VALUES
# ==========================================
scaled_data = scaled_data + 1

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================
train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]

test_data = scaled_data[train_size:]

# ==========================================
# CHOOSE SAFE SEASONAL PERIOD
# ==========================================
seasonal_period = 2

# ==========================================
# BUILD HOLT-WINTERS MODEL
# ==========================================
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=seasonal_period,
    initialization_method='estimated'
).fit()

# ==========================================
# FORECAST TEST DATA
# ==========================================
test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

# ==========================================
# VISUAL EVALUATION
# ==========================================
plt.figure(figsize=(12,5))

ax = train_data.plot(
    label='Train Data'
)

test_data.plot(
    ax=ax,
    label='Actual Test Data'
)

test_predictions_add.plot(
    ax=ax,
    label='Predicted Data'
)

plt.legend()

plt.title('Visual Evaluation')

plt.xlabel('Date')

plt.ylabel('Scaled Revenue')

plt.grid(True)

plt.show()

# ==========================================
# RMSE CALCULATION
# ==========================================
rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

print("\nRMSE:", rmse)

# ==========================================
# FINAL MODEL
# ==========================================
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=seasonal_period,
    initialization_method='estimated'
).fit()

# ==========================================
# FUTURE FORECAST
# ==========================================
future_steps = max(2, int(len(data_monthly) / 4))

final_predictions = final_model.forecast(
    steps=future_steps
)

# ==========================================
# CREATE FUTURE DATES
# ==========================================
future_dates = pd.date_range(
    start=data_monthly.index[-1] + pd.offsets.MonthBegin(),
    periods=future_steps,
    freq='MS'
)

forecast_df = pd.DataFrame({
    'Forecast': final_predictions.values
}, index=future_dates)

# ==========================================
# FINAL FORECAST PLOT
# ==========================================
plt.figure(figsize=(12,5))

ax = scaled_data.plot(
    label='Original Data'
)

forecast_df['Forecast'].plot(
    ax=ax,
    label='Forecast'
)

plt.legend()

plt.xlabel('Date')

plt.ylabel('Scaled Revenue')

plt.title('Future Revenue Forecast using Holt-Winters Method')

plt.grid(True)

plt.show()

# ==========================================
# DISPLAY FORECAST VALUES
# ==========================================
print("\nFuture Forecast Values:\n")

print(forecast_df)

```

### OUTPUT:


<img width="1001" height="470" alt="image" src="https://github.com/user-attachments/assets/2045ce31-c344-4361-88ce-a7262a3b43b2" />
<img width="1001" height="470" alt="image" src="https://github.com/user-attachments/assets/84b19a39-358d-4393-ab67-f4cf24142b5f" />
<img width="629" height="469" alt="image" src="https://github.com/user-attachments/assets/fff17956-0c5d-4da7-a26d-6fd6d1c491d8" />
<img width="1014" height="486" alt="image" src="https://github.com/user-attachments/assets/91409596-ce73-4a01-9690-1f0e69a2905e" />
<img width="1001" height="486" alt="image" src="https://github.com/user-attachments/assets/cd5a76fd-c27a-4488-b049-ec2c6fe52ec1" />









### RESULT:
Thus the program run successfully based on the Holt Winters Method model.

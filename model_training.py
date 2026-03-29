# ============================================
# WEEKLY SALES PREDICTION MODEL TRAINING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("weekly_sales.csv")

# Convert to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Sort chronologically (CRITICAL for time series)
df = df.sort_values('InvoiceDate')

print("Dataset Loaded Successfully")
print(df.head())


# ============================================
# 2. Feature Engineering (CRITICAL STEP)
# ============================================

# Time features
df['Week_Number'] = np.arange(len(df))
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.quarter

# Lag features (previous weeks sales)
df['Lag_1'] = df['Total_Weekly_Sales'].shift(1)
df['Lag_2'] = df['Total_Weekly_Sales'].shift(2)
df['Lag_3'] = df['Total_Weekly_Sales'].shift(3)
df['Lag_4'] = df['Total_Weekly_Sales'].shift(4)
df['Lag_5'] = df['Total_Weekly_Sales'].shift(5)
df['Lag_6'] = df['Total_Weekly_Sales'].shift(6)
df['Lag_7'] = df['Total_Weekly_Sales'].shift(7)
df['Lag_8'] = df['Total_Weekly_Sales'].shift(8)

# Rolling average feature
df['Rolling_Mean_4'] = df['Total_Weekly_Sales'].shift(1).rolling(4).mean()
df['Rolling_Mean_8'] = df['Total_Weekly_Sales'].shift(1).rolling(8).mean()
# Drop missing values created by lagging
df = df.dropna()

print("\nFeature Engineering Completed")


# ============================================
# 3. Define Features and Target
# ============================================

features = [

'Transaction_Count',

'Customer_Count',

'Quantity_Sold',

'Unique_Products_Sold',

'Avg_Order_Value',

'Avg_Quantity_Per_Transaction',

'Lag_1',
'Lag_2',
'Lag_3'

]



X = df[features]
y = df['Total_Weekly_Sales']


# ============================================
# 4. Train-Test Split (Chronological)
# ============================================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

dates_test = df['InvoiceDate'][split_index:]


print("\nTraining Size:", len(X_train))
print("Testing Size:", len(X_test))


# ============================================
# 5. Initialize Models
# ============================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
}

# ============================================
# 6. Train, Predict and Evaluate
# ============================================

results = {}
predictions = {}

print("\nModel Training Started...\n")

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = (mae, rmse, r2)

    print(f"{name} Completed")


# ============================================
# 7. Visualization: Actual vs Predicted
# ============================================

plt.figure(figsize=(14,7))

plt.plot(dates_test, y_test.values,
         label="Actual Sales",
         linewidth=3)

for name, pred in predictions.items():

    plt.plot(dates_test, pred,
             label=name,
             linestyle='--')

plt.title("Weekly Sales Prediction: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid()

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================
# 8. Print Model Performance
# ============================================

print("\n===================================")
print("MODEL PERFORMANCE RESULTS")
print("===================================")

best_model = None
best_r2 = -999

for name, (mae, rmse, r2) in results.items():

    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = name


print("\n===================================")
print(f"BEST MODEL: {best_model}")
print(f"BEST R2 SCORE: {best_r2:.4f}")
print("===================================")


# Convert test results to DataFrame
output_df = pd.DataFrame({
    'InvoiceDate': dates_test,
    'Actual_Sales': y_test.values
})

# Add predictions from each model
for name, pred in predictions.items():
    col_name = name.replace(" ", "_") + "_Prediction"
    output_df[col_name] = pred

# Save to CSV
output_df.to_csv("predictions.csv", index=False)

print("\nPredictions saved as predictions.csv")
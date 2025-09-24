import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np


# Load data
data_path = "M:\\ExpensePrediction\\data\\Expenses.xlsx"
df = pd.read_excel(data_path)

# Ensure Value Date is datetime
df["Value Date"] = pd.to_datetime(df["Value Date"], errors="coerce")

# Create Year-Month field
df["Yr_Month"] = df["Value Date"].dt.to_period("M")

# Group by Yr_Month and sum withdrawals
monthly_expenses = df.groupby("Yr_Month")["Withdrawal Amt."].sum().reset_index()
monthly_expenses["Yr_Month"] = monthly_expenses["Yr_Month"].astype(str)

# Create lag features
for i in range(1, 4):
    monthly_expenses[f"Prev_{i}"] = monthly_expenses["Withdrawal Amt."].shift(i)

# Optional: Add month as a feature
monthly_expenses["Month"] = pd.to_datetime(monthly_expenses["Yr_Month"]).dt.month

# Drop rows with NaN
monthly_expenses = monthly_expenses.dropna()

# Features and target
X = monthly_expenses[["Prev_1", "Prev_2", "Prev_3", "Month"]]
y = monthly_expenses["Withdrawal Amt."]

# Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Save model
save_dir = r"M:\ExpensePrediction\scripts"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "xgb_expense_model.pkl")
joblib.dump(model, save_path)

#os.makedirs("models", exist_ok=True)
#joblib.dump(model, "models/xgb_expense_model.pkl")

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("✅ XGBoost model trained and saved as models/xgb_expense_model.pkl")
print(f"Training R² Score: {r2_score(y_train, y_pred_train):.2f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.2f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Test RMSE: {rmse:.2f}")
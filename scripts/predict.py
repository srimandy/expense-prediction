import os
import joblib
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# Load model
model_path = r"M:\ExpensePrediction\models\xgb_expense_model.pkl"
model = joblib.load(model_path)

# Load and prepare data
data_path = r"M:\ExpensePrediction\data\expenses.xlsx"
df = pd.read_excel(data_path)
df["Value Date"] = pd.to_datetime(df["Value Date"], errors="coerce")
df["Yr_Month"] = df["Value Date"].dt.to_period("M")

# Aggregate by month
monthly_expenses = df.groupby("Yr_Month")["Withdrawal Amt."].sum().reset_index()
monthly_expenses["Yr_Month"] = monthly_expenses["Yr_Month"].astype(str)

# Get last known 3 months
last_3 = monthly_expenses.tail(3)["Withdrawal Amt."].values.tolist()

# Start from last known month
last_month_str = monthly_expenses["Yr_Month"].iloc[-1]
last_month = pd.to_datetime(last_month_str) + MonthEnd(1)

# Predict next 12 months
future_preds = []
for i in range(12):
    if len(last_3) < 3:
        break

    # Features = Prev_1, Prev_2, Prev_3, Month
    month_num = (last_month + pd.DateOffset(months=i)).month
    features = [last_3[-1], last_3[-2], last_3[-3], month_num]

    pred = model.predict([features])[0]

    future_preds.append({
        "Year-Month": (last_month + pd.DateOffset(months=i)).strftime("%Y-%m"),
        "Predicted Expense": round(pred, 2)
    })

    last_3.append(pred)  # roll forward

# Save to Excel
pred_df = pd.DataFrame(future_preds)
os.makedirs("output", exist_ok=True)
pred_df.to_excel("output/future_expense_forecast.xlsx", index=False)

print("✅ Forecast saved to output/future_expense_forecast.xlsx")
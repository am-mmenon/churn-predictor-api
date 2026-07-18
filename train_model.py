import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# 1. Load the mock data
df = pd.read_csv("customer_data.csv")
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# 2. Set snapshot date
snapshot_date = pd.to_datetime("2026-07-18")

# 3. Calculate RFM Metrics
rfm = df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (snapshot_date - x.max()).days,
    'CustomerID': 'count',
    'TransactionAmount': 'sum'
}).rename(columns={
    'TransactionDate': 'recency',
    'CustomerID': 'frequency',
    'TransactionAmount': 'monetary'
}).reset_index()

# 4. Create a Training Label (Target)
# Rule: If a customer hasn't bought anything in more than 30 days, we label them as Churned (1). Otherwise (0).
rfm['churn'] = np.where(rfm['recency'] > 30, 1, 0)

# Features (Inputs to AI) and Target (What AI predicts)
X = rfm[['recency', 'frequency', 'monetary']]
y = rfm['churn']

print("Training the XGBoost Model...")

# 5. Initialize and Train XGBoost
model = XGBClassifier(eval_metric='logloss')
model.fit(X, y)

# 6. Save the trained model structure to a file
joblib.dump(model, "churn_model.pkl")
print("Success! Model trained and saved as 'churn_model.pkl'")
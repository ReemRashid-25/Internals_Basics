import pandas as pd
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Load data
df1 = pd.read_csv("data/training_data.csv")
df2 = pd.read_csv("data/new_data.csv")

combined = pd.concat([df1, df2])

X = combined.drop("fare_amount", axis=1)
y = combined["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load champion
champion = joblib.load("models/tuned_model.pkl")

champion_preds = champion.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_preds)

# Retrain
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

re_preds = model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, re_preds)

improvement = champion_mae - retrained_mae

action = "promoted" if improvement >= 0.3 else "kept_champion"

output = {
    "original_data_rows": len(df1),
    "new_data_rows": len(df2),
    "combined_data_rows": len(combined),
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": 0.3,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 DONE")

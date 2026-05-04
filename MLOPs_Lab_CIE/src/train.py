import joblib
import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("urbanride-fare-amount")

results = []

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge()
}
os.makedirs("models", exist_ok=True)
best_rmse = float("inf")
best_model_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.pkl")

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })
        mlflow.set_tag("experiment_type", "baseline_comparison")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model
            
joblib.dump(best_model, "models/best_model.pkl")		
# Save JSON
output = {
    "experiment_name": "urbanride-fare-amount",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 DONE")

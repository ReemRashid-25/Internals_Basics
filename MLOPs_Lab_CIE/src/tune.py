import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/training_data.csv")

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

rf = RandomForestRegressor(random_state=42)

mlflow.set_experiment("urbanride-fare-amount")

with mlflow.start_run(run_name="tuning-urbanride"):
    search = RandomizedSearchCV(
        rf,
        param_grid,
        n_iter=10,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    joblib.dump(best_model, "models/tuned_model.pkl")

    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 10,
        "best_params": search.best_params_,
        "best_mae": mae,
        "best_cv_mae": -search.best_score_,
        "parent_run_name": "tuning-urbanride"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 DONE")

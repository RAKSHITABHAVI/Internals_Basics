import pandas as pd
import mlflow
import mlflow.sklearn
import json, os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/training_data.csv")

X = df[["ad_spend","audience_size","creative_score","is_retargeting"]]
y = df["click_through_rate"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("adpulse-click-through-rate")

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor()
}

results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        mape = (abs((y_test - preds) / y_test)).mean()

        # Log to MLflow
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        mlflow.set_tag("domain", "digital_advertising")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

# Best model
best = min(results, key=lambda x: x["mae"])

os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump({
        "experiment_name": "adpulse-click-through-rate",
        "models": results,
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"]
    }, f, indent=4)

print("Task 1 DONE")
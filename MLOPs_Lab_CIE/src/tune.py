import pandas as pd
import json, os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv("data/training_data.csv")

X = df[["ad_spend","audience_size","creative_score","is_retargeting"]]
y = df["click_through_rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_grid = {
    "n_estimators": [50,150],
    "learning_rate": [0.05,0.1,0.2],
    "max_depth": [3,5,10]
}

model = GradientBoostingRegressor()

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error"
)

grid.fit(X_train, y_train)

best_params = grid.best_params_
best_mae = -grid.best_score_

os.makedirs("results", exist_ok=True)

with open("results/step2_s2.json","w") as f:
    json.dump({
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": 18,
        "best_params": best_params,
        "best_mae": best_mae,
        "best_cv_mae": best_mae,
        "parent_run_name": "tuning-adpulse"
    }, f, indent=4)

print("Task 2 DONE")
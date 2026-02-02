import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score


# CONFIG
DATA_FILE = "real_household_daily_data.csv"
MODEL_FILE = "daily_electricity_model.pkl"

FEATURE_COLS = [
    "household_size",
    "has_ac",
    "ac_hours_day",
    "rice_uses_day",
    "tv_hours_day",
    "has_wifi",
    "heat_index",
    "is_weekend"
]

TARGET_COL = "daily_kwh"


# LOAD DATA
df = pd.read_csv(DATA_FILE)

X = df[FEATURE_COLS]
y = df[TARGET_COL]


# MODEL FACTORY
def create_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

# 80/20 TRAIN–TEST SPLIT (REPORTING)

print("\n========== 80/20 TRAIN–TEST SPLIT ==========\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

split_model = create_model()
split_model.fit(X_train, y_train)

split_preds = split_model.predict(X_test)

split_mae = mean_absolute_error(y_test, split_preds)
split_r2 = r2_score(y_test, split_preds)

print(f"MAE: {split_mae:.3f}")
print(f"R² : {split_r2:.3f}")


#5-FOLD CROSS VALIDATION (PRIMARY)
print("\n========== 5-FOLD CROSS VALIDATION ==========\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    model = create_model()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mae_scores.append(mae)
    r2_scores.append(r2)

    print(f"Fold {fold}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R² : {r2:.3f}\n")

print("===================================")
print(f"Average MAE: {np.mean(mae_scores):.3f}")
print(f"Average R² : {np.mean(r2_scores):.3f}")
print("===================================\n")

# FINAL MODEL (TRAIN ON ALL DATA)
final_model = create_model()
final_model.fit(X, y)

joblib.dump(final_model, MODEL_FILE)

print("Final model trained on full dataset.")
print("Model saved as:", MODEL_FILE)

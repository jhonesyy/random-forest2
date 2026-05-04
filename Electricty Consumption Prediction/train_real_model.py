import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# CONFIG
DATA_FILE = "cleaned_electricity_dataset.csv"
MODEL_FILE = "daily_electricity_model.pkl"

FEATURE_COLS = [
    "household_size", "has_ac", "ac_hours_day", "rice_uses_day",
    "tv_hours_day", "has_wifi", "heat_index", "is_weekend"
]
TARGET_COL = "daily_kwh"

print("=== ELECTRICITY CONSUMPTION PREDICTION MODEL ===\n")

# 1. Load Data
df = pd.read_csv(DATA_FILE)
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

def create_model():
    return RandomForestRegressor(
        n_estimators=400,      # Increased slightly
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

# 2. Train-Test Split Evaluation
print("\n=== 80/20 Train-Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = create_model()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"MAE : {mean_absolute_error(y_test, preds):.3f} kWh")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f} kWh")
print(f"R²  : {r2_score(y_test, preds):.4f}")

# 3. Cross Validation
print("\n=== 5-Fold Cross Validation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

for fold, (tr, te) in enumerate(kf.split(X), 1):
    m = create_model()
    m.fit(X.iloc[tr], y.iloc[tr])
    preds_fold = m.predict(X.iloc[te])
    mae = mean_absolute_error(y.iloc[te], preds_fold)
    mae_scores.append(mae)
    print(f"Fold {fold}: MAE = {mae:.3f} kWh")

print(f"\nAverage MAE: {np.mean(mae_scores):.3f} kWh")

# 4. Final Model
print("\n=== Training Final Model ===")
final_model = create_model()
final_model.fit(X, y)

joblib.dump(final_model, MODEL_FILE)
print(f"✅ Model successfully saved as: {MODEL_FILE}")
print(f"Model Size: {os.path.getsize(MODEL_FILE) / 1024:.1f} KB")

# Feature Importance (Useful for debugging)
importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
print("\nTop Feature Importances:")
print(importances.sort_values(ascending=False))

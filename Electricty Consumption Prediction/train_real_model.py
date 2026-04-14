import pandas as pd
import joblib
import numpy as np
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

print("=== ELECTRICITY CONSUMPTION PREDICTION MODEL EVALUATION ===\n")

# 1. DATA PREPARATION AND PREPROCESSING
print("========== 1. DATA PREPARATION ==========\n")
df = pd.read_csv(DATA_FILE)
print(f"Dataset Shape: {df.shape}")
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("\nSelected Features:", FEATURE_COLS)
print("Target:", TARGET_COL)
print("\n→ Data is cleaned. 8 features selected based on domain knowledge.\n")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

def create_model():
    return RandomForestRegressor(
        n_estimators=300, max_depth=18, min_samples_split=3,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

# 2. MODEL TRAINING AND TESTING
print("========== 2. MODEL TRAINING & TESTING ==========\n")

# 80/20 Split
print("--- 80/20 Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = create_model()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"MAE : {mean_absolute_error(y_test, preds):.3f} kWh")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f} kWh")
print(f"R²  : {r2_score(y_test, preds):.3f}\n")

# 5-Fold Cross Validation
print("--- 5-Fold Cross Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores, r2_scores = [], []

for fold, (tr, te) in enumerate(kf.split(X), 1):
    model = create_model()
    model.fit(X.iloc[tr], y.iloc[tr])
    preds = model.predict(X.iloc[te])
    mae = mean_absolute_error(y.iloc[te], preds)
    r2 = r2_score(y.iloc[te], preds)
    mae_scores.append(mae)
    r2_scores.append(r2)
    print(f"Fold {fold}: MAE={mae:.3f}  R²={r2:.3f}")

print("\nAverage MAE :", round(np.mean(mae_scores), 3))
print("Average R²  :", round(np.mean(r2_scores), 3))

# Final Model
final_model = create_model()
final_model.fit(X, y)
joblib.dump(final_model, MODEL_FILE)
print("\nFinal model trained on full data and saved as:", MODEL_FILE)

# 3. COMPARATIVE ANALYSIS
print("\n========== 3. COMPARATIVE ANALYSIS ==========\n")
small_df = df.sample(n=min(1000, len(df)), random_state=42)
X_s, y_s = small_df[FEATURE_COLS], small_df[TARGET_COL]

kf_small = KFold(n_splits=5, shuffle=True, random_state=42)
small_mae = []
for tr, te in kf_small.split(X_s):
    m = create_model()
    m.fit(X_s.iloc[tr], y_s.iloc[tr])
    small_mae.append(mean_absolute_error(y_s.iloc[te], m.predict(X_s.iloc[te])))

print(f"Full Dataset Avg MAE : {np.mean(mae_scores):.3f}")
print(f"1000 Records Avg MAE : {np.mean(small_mae):.3f}")
print("→ More data improves model stability.")

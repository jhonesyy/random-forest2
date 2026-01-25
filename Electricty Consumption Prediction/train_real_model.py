import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "real_household_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "real_electricity_model.pkl")

FEATURE_COLS = [
    'household_size',
    'has_ac',
    'ac_hours_day',
    'rice_uses_day',
    'tv_hours_day',
    'has_wifi'
]
TARGET_COL = 'monthly_kwh'


# Load & Clean

print("Loading training data...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Raw rows: {len(df)}")
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    exit(1)

available_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
if len(available_cols) < len(FEATURE_COLS) + 1:
    print("ERROR: Missing required columns")
    print("Expected:", FEATURE_COLS + [TARGET_COL])
    print("Found:", list(df.columns))
    exit(1)

df = df[available_cols].copy()

# Force numeric & drop bad rows
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna()
print(f"Valid rows after cleaning: {len(df_clean)}")

if len(df_clean) < 15:
    print("WARNING: Very few rows — model will be unreliable.")
    print(df_clean.head(10))
    exit(1)

print("\nData Summary:")
print(df_clean.describe().round(2))


# Train / Test Split
X = df_clean[FEATURE_COLS]
y = df_clean[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Training on {len(X_train)} | Testing on {len(X_test)}")


# Model
rf = RandomForestRegressor(
    n_estimators=300,        
    max_depth=18,            
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Evaluation
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print("\nPerformance:")
print(f"Train MAE: {train_mae:.1f} kWh")
print(f"Test  MAE: {test_mae:.1f} kWh")
print(f"Test R²:  {test_r2:.3f}")

print("\nFeature Importance:")
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(imp.round(4))

# Save
joblib.dump(rf, MODEL_FILE)
print(f"\nModel saved → {MODEL_FILE}")
print("Done. You can now run the Flask app.")
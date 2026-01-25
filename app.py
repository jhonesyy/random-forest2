from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ========================
# CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master_meralco_appliances.csv")
MODEL_FILE = os.path.join(BASE_DIR, "real_electricity_model.pkl")

# Update this monthly from Meralco announcements / official site
# As of Jan 2026: ≈12.95 – 13.0 PHP/kWh (generation + distribution + others)
RATE_PER_KWH = 12.95

FEATURE_COLS = [
    'household_size',
    'has_ac',
    'ac_hours_day',
    'rice_uses_day',
    'tv_hours_day',
    'has_wifi'
]

# Minimal always-on / miscellaneous load (lights, standby, small devices)
ALWAYS_ON_DAILY_BASE = 0.7          # fixed
ALWAYS_ON_PER_PERSON = 0.12         # additional per household member

# ========================
# LOAD DATA & MODEL
# ========================
try:
    master = pd.read_csv(MASTER_CSV)
    rf_model = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"Error loading data/model: {e}")
    master = pd.DataFrame()
    rf_model = None

# Group appliances by category for dropdowns
grouped_models = {}
for _, row in master.iterrows():
    grouped_models.setdefault(row['Category'], []).append(row)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST' and rf_model is not None:
        try:
            household_size = int(request.form.get('household_size', 4))
            total_daily_kwh = 0.0

            rf_features = {
                "household_size": household_size,
                "has_ac": 0,
                "ac_hours_day": 0.0,
                "rice_uses_day": 0.0,
                "tv_hours_day": 0.0,
                "has_wifi": 0
            }

            appliance_count = 0

            # Process each appliance slot
            for i in range(1, 6):
                cat = request.form.get(f'category_{i}')
                model_name = request.form.get(f'model_{i}')
                usage = float(request.form.get(f'usage_{i}', 0) or 0)

                # Server-side clamp
                usage = max(0, min(usage, 24))

                if not cat or not model_name:
                    continue

                row = master[(master['Category'] == cat) & (master['Model'] == model_name)]
                if row.empty:
                    continue

                row = row.iloc[0]
                kwh_value = float(row['kWh_value'])
                unit = str(row['Usage_Unit']).lower()

                if unit in ['day', 'daily']:
                    daily = kwh_value
                elif unit in ['hour', 'hours']:
                    daily = kwh_value * usage
                elif unit in ['use', 'uses', 'cycle']:
                    daily = kwh_value * usage
                elif unit in ['minute', 'minutes']:
                    daily = kwh_value * (usage / 60)
                else:
                    daily = 0

                total_daily_kwh += daily
                appliance_count += 1

                # Feature updates
                if cat == "Air Conditioner":
                    rf_features["has_ac"] = 1
                    rf_features["ac_hours_day"] = max(rf_features["ac_hours_day"], usage)

                if "RICE" in str(model_name).upper():
                    rf_features["rice_uses_day"] += min(usage, 5)  # cap crazy values

                if cat == "Gadgets / AV / Computers" and "TV" in str(model_name).upper():
                    rf_features["tv_hours_day"] += usage

                if cat in ["Gadgets / AV / Computers", "Wi-Fi / Always-on"]:
                    rf_features["has_wifi"] = 1

            # Add minimal always-on/misc load
            misc_daily = ALWAYS_ON_DAILY_BASE + (ALWAYS_ON_PER_PERSON * household_size)
            total_daily_kwh += misc_daily

            base_monthly = total_daily_kwh * 30

            # ML adjustment
            X = pd.DataFrame([rf_features], columns=FEATURE_COLS)
            tree_preds = np.array([tree.predict(X.values)[0] for tree in rf_model.estimators_])

            ml_mean = tree_preds.mean()

            # Adaptive weighting: trust ML more when user entered very few appliances
            ml_weight = 0.55 if appliance_count <= 2 else 0.40
            predicted_monthly = (1 - ml_weight) * base_monthly + ml_weight * ml_mean

            # Confidence range from tree predictions
            low = np.percentile(tree_preds, 10)
            high = np.percentile(tree_preds, 90)

            bill = round(predicted_monthly * RATE_PER_KWH, 2)

            # Explanation drivers
            drivers = []
            if rf_features["has_ac"]:
                drivers.append(f"Air conditioner ({rf_features['ac_hours_day']:.1f} hrs/day)")
            if rf_features["tv_hours_day"] > 4:
                drivers.append(f"High TV usage ({rf_features['tv_hours_day']:.1f} hrs/day)")
            if household_size >= 5:
                drivers.append("Large household size")
            if rf_features["rice_uses_day"] > 2:
                drivers.append("Frequent rice cooker use")
            if not drivers:
                drivers.append("General household & miscellaneous usage")

            result = {
                "base_kwh": round(base_monthly, 1),
                "predicted_kwh": round(predicted_monthly, 1),
                "bill": f"{bill:,.2f}",
                "bill_low": round(low * RATE_PER_KWH, 2),
                "bill_high": round(high * RATE_PER_KWH, 2),
                "drivers": drivers
            }

        except Exception as e:
            print(f"Calculation error: {e}")
            result = {"error": "Something went wrong during calculation. Please try again."}

    return render_template(
        'index.html',
        grouped_models=grouped_models,
        result=result
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
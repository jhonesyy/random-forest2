from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "master_meralco_appliances.csv")
MODEL_FILE = os.path.join(BASE_DIR, "real_electricity_model.pkl")

RATE_PER_KWH = 12.95
DAYS_IN_MONTH = 30

FEATURE_COLS = [
    'household_size',
    'has_ac',
    'ac_hours_day',
    'rice_uses_day',
    'tv_hours_day',
    'has_wifi'
]

ALWAYS_ON_DAILY_BASE = 0.7
ALWAYS_ON_PER_PERSON = 0.12

master = pd.read_csv(MASTER_CSV)
rf_model = joblib.load(MODEL_FILE)

grouped_models = {}
for _, row in master.iterrows():
    grouped_models.setdefault(row['Category'], []).append(row)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        household_size = int(request.form.get("household_size", 4))
        total_daily_kwh = 0
        appliance_count = 0

        rf_features = {
            "household_size": household_size,
            "has_ac": 0,
            "ac_hours_day": 0,
            "rice_uses_day": 0,
            "tv_hours_day": 0,
            "has_wifi": 0
        }

        for i in range(1, 6):
            cat = request.form.get(f"category_{i}")
            model_name = request.form.get(f"model_{i}")
            usage = float(request.form.get(f"usage_{i}", 0) or 0)

            if not cat or not model_name:
                continue

            row = master[
                (master["Category"] == cat) &
                (master["Model"] == model_name)
            ]

            if row.empty:
                continue

            row = row.iloc[0]
            kwh = float(row["kWh_value"])
            unit = row["Usage_Unit"].lower()

            if unit in ["hour", "hours"]:
                daily = kwh * usage
            elif unit in ["use", "uses", "cycle"]:
                daily = kwh * usage
            else:
                daily = kwh

            total_daily_kwh += daily
            appliance_count += 1

            if cat == "Air Conditioner":
                rf_features["has_ac"] = 1
                rf_features["ac_hours_day"] = max(rf_features["ac_hours_day"], usage)

            if "RICE" in model_name.upper():
                rf_features["rice_uses_day"] += usage

            if "TV" in model_name.upper():
                rf_features["tv_hours_day"] += usage

            if cat in ["Wi-Fi / Always-on", "Gadgets / AV / Computers"]:
                rf_features["has_wifi"] = 1

        misc = ALWAYS_ON_DAILY_BASE + ALWAYS_ON_PER_PERSON * household_size
        total_daily_kwh += misc

        base_monthly = total_daily_kwh * DAYS_IN_MONTH
        X = pd.DataFrame([rf_features], columns=FEATURE_COLS)

        tree_preds = np.array([
            tree.predict(X.values)[0]
            for tree in rf_model.estimators_
        ])

        ml_weight = 0.4 if appliance_count > 2 else 0.55
        predicted_monthly = (1 - ml_weight) * base_monthly + ml_weight * tree_preds.mean()

        daily_kwh = predicted_monthly / DAYS_IN_MONTH
        daily_bill = daily_kwh * RATE_PER_KWH

        result = {
            "daily_kwh": round(daily_kwh, 2),
            "base_kwh": round(total_daily_kwh, 2),
            "daily_bill": f"{daily_bill:,.2f}",
            "bill_low": round(np.percentile(tree_preds, 10) / DAYS_IN_MONTH * RATE_PER_KWH, 2),
            "bill_high": round(np.percentile(tree_preds, 90) / DAYS_IN_MONTH * RATE_PER_KWH, 2),
            "drivers": [
                "Air conditioner usage" if rf_features["has_ac"] else None,
                "Large household" if household_size >= 5 else None
            ]
        }

        result["drivers"] = [d for d in result["drivers"] if d]

    return render_template("index.html", grouped_models=grouped_models, result=result)


if __name__ == "__main__":
    app.run(debug=True)

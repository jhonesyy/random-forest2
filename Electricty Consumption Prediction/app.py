from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = "meralco_electricity_calculator_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "daily_electricity_model.pkl")
APPLIANCE_FILE = os.path.join(BASE_DIR, "master_meralco_appliances.csv")
CUSTOM_FILE = os.path.join(BASE_DIR, "custom_appliances.csv")

RATE_PER_KWH = 12.95

# ✅ FIX: remove wrong indentation
FEATURE_COLS = [
    "household_size", "has_ac", "ac_hours_day", "rice_uses_day",
    "tv_hours_day", "has_wifi", "heat_index", "is_weekend"
]

# Load ML model
model = joblib.load(MODEL_FILE)

# Load appliance datasets
appliances_df = pd.read_csv(APPLIANCE_FILE)

# Create custom CSV if not exists
if not os.path.exists(CUSTOM_FILE):
    pd.DataFrame(columns=["Category", "Model", "kWh_value", "Usage_Unit", "Notes"]).to_csv(CUSTOM_FILE, index=False)

custom_df = pd.read_csv(CUSTOM_FILE)

# Combine built-in + custom appliances
combined_df = pd.concat([appliances_df, custom_df], ignore_index=True)

grouped_models = {
    cat: rows.to_dict(orient="records")
    for cat, rows in combined_df.groupby("Category")
}

appliance_lookup = combined_df.set_index("Model").to_dict("index")
all_models = combined_df.to_dict(orient="records")


# ✅ FIX: route must NOT be indented
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    num_appliances = 5
    custom_count = 1

    if request.method == "POST":
        action = request.form.get("action", "")

        try:
            num_appliances = int(request.form.get("num_appliances", 5))
        except:
            num_appliances = 5

        try:
            custom_count = int(request.form.get("custom_count", 1))
        except:
            custom_count = 1

        if action == "add_more":
            num_appliances = min(num_appliances + 2, 20)

        elif action == "add_custom":
            custom_count = min(custom_count + 1, 10)

        elif action == "calculate":
            num_appliances = 5

            try:
                household_size = min(int(request.form.get("household_size", 4)), 12)
            except:
                household_size = 4

            try:
                heat_index = float(request.form.get("heat_index", 40.0))
            except:
                heat_index = 40.0

            is_weekend = 1 if request.form.get("day_type") in ["Saturday", "Sunday"] else 0

            base_kwh = 0.0
            ac_hours = 0.0
            rice_uses = 0.0
            tv_hours = 0.0
            has_ac = 0
            has_wifi = 0
            drivers = []

            for i in range(1, 21):
                model_name = request.form.get(f"model_{i}")
                usage_str = request.form.get(f"usage_{i}")

                if not model_name or not usage_str:
                    continue

                try:
                    usage = float(usage_str)
                    if usage <= 0:
                        continue
                except:
                    continue

                row = appliance_lookup.get(model_name)
                if not row:
                    continue

                kwh_value = row["kWh_value"]
                unit = row.get("Usage_Unit", "hour")

                if unit == "hour":
                    kwh = kwh_value * usage
                elif unit == "use":
                    kwh = kwh_value * usage
                elif unit == "minutes":
                    kwh = kwh_value * (usage / 60)
                else:
                    kwh = kwh_value

                base_kwh += kwh

                if row["Category"] == "Air Conditioner":
                    has_ac = 1
                    ac_hours += usage
                if "Rice Cooker" in row.get("Notes", ""):
                    rice_uses += usage
                if "TV" in row.get("Notes", ""):
                    tv_hours += usage
                if row["Category"] == "Wi-Fi / Always-on":
                    has_wifi = 1

                drivers.append(f"{row['Category']} - {row.get('Notes') or row.get('Model')}")

            X = pd.DataFrame([{
                "household_size": household_size,
                "has_ac": has_ac,
                "ac_hours_day": round(ac_hours, 1),
                "rice_uses_day": round(rice_uses, 1),
                "tv_hours_day": round(tv_hours, 1),
                "has_wifi": has_wifi,
                "heat_index": heat_index,
                "is_weekend": is_weekend
            }], columns=FEATURE_COLS)

            ml_kwh = model.predict(X)[0]
            final_kwh = (base_kwh * 0.6) + (ml_kwh * 0.4)
            daily_bill = final_kwh * RATE_PER_KWH

            result = {
                "daily_kwh": round(final_kwh, 2),
                "daily_bill": round(daily_bill, 2),
                "base_kwh": round(base_kwh, 2),
                "bill_low": round(daily_bill * 0.9, 2),
                "bill_high": round(daily_bill * 1.1, 2),
                "drivers": list(set(drivers))[:6]
            }


    return render_template(
        "index.html",
        result=result,
        grouped_models=grouped_models,
        all_models=all_models,
        num_appliances=num_appliances,
        custom_count=custom_count,
        request=request
    )



if __name__ == "__main__":
    app.run(debug=True)

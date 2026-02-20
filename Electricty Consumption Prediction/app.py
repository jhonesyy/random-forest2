from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "daily_electricity_model.pkl")
APPLIANCE_FILE = os.path.join(BASE_DIR, "master_meralco_appliances.csv")

RATE_PER_KWH = 12.95

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

model = joblib.load(MODEL_FILE)
appliances_df = pd.read_csv(APPLIANCE_FILE)

grouped_models = {
    cat: rows.to_dict(orient="records")
    for cat, rows in appliances_df.groupby("Category")
}

appliance_lookup = appliances_df.set_index("Model").to_dict("index")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    num_appliances = 5  # default starting number

    if request.method == "POST":
        action = request.form.get("action", "calculate")

        # Get current number of shown appliance fields
        try:
            num_appliances = int(request.form.get("num_appliances", 5))
        except (ValueError, TypeError):
            num_appliances = 5

        if action == "add_more":
            # Just add 2 more slots, no calculation, no reset
            num_appliances += 2
            num_appliances = min(num_appliances, 20)  # reasonable max

        elif action == "calculate":
            # Reset to 5 slots when calculating
            num_appliances = 5

            household_size_raw = request.form.get("household_size", "4")
            try:
                household_size = int(household_size_raw)
                if household_size > 12:
                    household_size = 12
            except ValueError:
                household_size = 4

            heat_index_raw = request.form.get("heat_index", "40")
            try:
                heat_index = float(heat_index_raw)
            except ValueError:
                heat_index = 40.0

            day_type = request.form.get("day_type", "Weekday")
            is_weekend = 1 if day_type in ["Saturday", "Sunday"] else 0

            base_kwh = 0.0
            ac_hours = 0
            rice_uses = 0
            tv_hours = 0
            has_ac = 0
            has_wifi = 0
            drivers = []

            for i in range(1, 21):  # safe upper limit
                model_name = request.form.get(f"model_{i}")
                usage_str = request.form.get(f"usage_{i}")

                if not model_name or not usage_str:
                    continue

                try:
                    usage = float(usage_str)
                except ValueError:
                    continue

                row = appliance_lookup.get(model_name)
                if not row:
                    continue

                kwh_value = row["kWh_value"]
                unit = row["Usage_Unit"]

                if unit == "hour":
                    kwh = kwh_value * usage
                elif unit == "use":
                    kwh = kwh_value * usage
                elif unit == "minutes":
                    kwh = kwh_value * (usage / 60)
                else:
                    kwh = kwh_value

                base_kwh += kwh

                notes = row.get("Notes", "")

                if row["Category"] == "Air Conditioner":
                    has_ac = 1
                    ac_hours += usage

                if "Rice Cooker" in notes:
                    rice_uses += usage

                if "TV" in notes:
                    tv_hours += usage

                if row["Category"] == "Wi-Fi / Always-on":
                    has_wifi = 1

                drivers.append(f"{row['Category']} – {notes or model_name}")

            X = pd.DataFrame([{
                "household_size": household_size,
                "has_ac": has_ac,
                "ac_hours_day": ac_hours,
                "rice_uses_day": rice_uses,
                "tv_hours_day": tv_hours,
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
                "drivers": list(set(drivers))[:5]
            }

    return render_template(
        "index.html",
        result=result,
        grouped_models=grouped_models,
        num_appliances=num_appliances
    )


if __name__ == "__main__":
    app.run(debug=True)

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "rf_model.pkl"

STATES = {
    1: "Andhra Pradesh",
    2: "Arunachal Pradesh",
    3: "Assam",
    4: "Bihar",
    5: "Chhattisgarh",
    6: "Goa",
    7: "Gujarat",
    8: "Haryana",
    9: "Himachal Pradesh",
    10: "Jammu & Kashmir",
    11: "Jharkhand",
    12: "Karnataka",
    13: "Kerala",
    14: "Madhya Pradesh",
    15: "Maharashtra",
    16: "Manipur",
    17: "Meghalaya",
    18: "Mizoram",
    19: "Nagaland",
    20: "Odisha",
    21: "Punjab",
    22: "Rajasthan",
    23: "Sikkim",
    24: "Tamil Nadu",
    25: "Telangana",
    26: "Tripura",
    27: "Uttar Pradesh",
    28: "Uttarakhand",
    29: "West Bengal",
    30: "Andaman & Nicobar",
    31: "Chandigarh",
    32: "Delhi",
    33: "Ladakh",
    34: "Lakshadweep",
    35: "Puducherry",
    36: "Dadra & Nagar Haveli",
}

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTHLY_NORMAL_HINT = {
    1: 3.0, 2: 4.0, 3: 6.0, 4: 12.0, 5: 18.0, 6: 35.0,
    7: 45.0, 8: 40.0, 9: 28.0, 10: 15.0, 11: 8.0, 12: 4.0,
}

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def rain_category(mm: float) -> dict:
    if mm < 2.5:
        label, emoji, color = "Trace / dry", "🌤️", "#94A3B8"
    elif mm < 15:
        label, emoji, color = "Light rain", "🌦️", "#38BDF8"
    elif mm < 64:
        label, emoji, color = "Moderate rain", "🌧️", "#2563EB"
    elif mm < 124:
        label, emoji, color = "Heavy rain", "⛈️", "#1E40AF"
    else:
        label, emoji, color = "Very heavy rain", "🌩️", "#1E3A8A"
    return {"label": label, "emoji": emoji, "color": color}


def predict_rainfall(state_code: int, year: int, month: int, day: int, normal: float) -> dict:
    model = load_model()
    row = pd.DataFrame([{
        "state_code": state_code,
        "year": year,
        "month": month,
        "day": day,
        "normal": normal,
    }])
    prediction = max(0.0, float(model.predict(row)[0]))
    delta = prediction - normal
    category = rain_category(prediction)
    state_name = STATES.get(state_code, "Unknown")

    return {
        "prediction": round(prediction, 2),
        "normal": round(normal, 2),
        "delta": round(delta, 2),
        "state_name": state_name,
        "month_name": MONTH_NAMES[month],
        "category": category,
    }

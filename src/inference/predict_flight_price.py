import pickle
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/flight_price_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"] if isinstance(saved, dict) else saved
except FileNotFoundError:
    model = None
    print("[WARN] flight_price_model.pkl not found.")

# All cities the model knows about
FROM_CITIES = [
    "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
    "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)",
    "Salvador (BH)", "Sao Paulo (SP)"
]

TO_CITIES = [
    "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
    "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)",
    "Salvador (BH)", "Sao Paulo (SP)"
]

FLIGHT_TYPES = ["economic", "firstClass", "premium"]   # economic = base (dropped)
AGENCIES     = ["CloudFy", "FlyingDrops", "Rainbow"]   # CloudFy  = base (dropped)


def predict_flight_price(
    travel_code: int,
    user_code: int,
    time: float,
    distance: float,
    year: int,
    month: int,
    day: int,
    from_city: str,
    to_city: str,
    flight_type: str,
    agency: str,
) -> float:
    if model is None:
        raise RuntimeError("Flight price model is not loaded.")

    # ── Base numeric features ──────────────────────────────────
    row = {
        "travelCode": travel_code,
        "userCode":   user_code,
        "time":       time,
        "distance":   distance,
        "year":       year,
        "month":      month,
        "day":        day,
    }

    # ── One-hot: from_<city> ───────────────────────────────────
    for city in FROM_CITIES:
        row[f"from_{city}"] = 1 if from_city == city else 0

    # ── One-hot: to_<city> ─────────────────────────────────────
    for city in TO_CITIES:
        row[f"to_{city}"] = 1 if to_city == city else 0

    # ── One-hot: flightType (drop 'economic') ──────────────────
    for ft in ["firstClass", "premium"]:
        row[f"flightType_{ft}"] = 1 if flight_type == ft else 0

    # ── One-hot: agency (drop 'CloudFy') ──────────────────────
    for ag in ["FlyingDrops", "Rainbow"]:
        row[f"agency_{ag}"] = 1 if agency == ag else 0

    df = pd.DataFrame([row])
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)
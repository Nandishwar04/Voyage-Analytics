from flask import Flask, request, jsonify
from src.inference.predict_gender import predict_gender
from src.inference.predict_flight_price import predict_flight_price
from src.inference.predict_recommendations import get_recommendations

app = Flask(__name__)


# ── Health check ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ML Classification & Prediction API"})


# ── 1. Gender Classification ──────────────────────────────────
@app.route("/predict_gender", methods=["POST"])
def predict_gender_route():
    """
    POST /predict_gender
    Body:    { "name": "Alice", "age": 30 }
    Returns: { "name": "Alice", "predicted_gender": "Female" }
    """
    data = request.get_json(force=True)
    missing = [f for f in ("name", "age") if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        gender = predict_gender(data["name"], int(data["age"]))
        return jsonify({"name": data["name"], "predicted_gender": gender})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 2. Flight Price Prediction ────────────────────────────────
@app.route("/predict_flight_price", methods=["POST"])
def predict_flight_price_route():
    data = request.get_json(force=True)

    required = [
        "travel_code", "user_code", "time", "distance",
        "year", "month", "day",
        "from_city", "to_city", "flight_type", "agency"
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        price = predict_flight_price(
            travel_code = int(data["travel_code"]),
            user_code   = int(data["user_code"]),
            time        = float(data["time"]),
            distance    = float(data["distance"]),
            year        = int(data["year"]),
            month       = int(data["month"]),
            day         = int(data["day"]),
            from_city   = data["from_city"],
            to_city     = data["to_city"],
            flight_type = data["flight_type"],
            agency      = data["agency"],
        )
        return jsonify({
            "from":              data["from_city"],
            "to":                data["to_city"],
            "flight_type":       data["flight_type"],
            "agency":            data["agency"],
            "predicted_price_usd": price,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 3. SVD Recommendations ────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend_route():
    """
    POST /recommend
    Body:    { "user_id": 42, "item_ids": [101, 202, 303], "top_n": 3 }
    Returns: { "user_id": 42, "recommendations": [...] }
    """
    data = request.get_json(force=True)
    missing = [f for f in ("user_id", "item_ids") if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        top_n = int(data.get("top_n", 5))
        recs = get_recommendations(
            user_id=int(data["user_id"]),
            item_ids=data["item_ids"],
            top_n=top_n,
        )
        return jsonify({"user_id": data["user_id"], "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
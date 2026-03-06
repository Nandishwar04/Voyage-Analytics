from flask import Flask, request, jsonify
from src.inference.predict_gender import predict_gender


app = Flask(__name__)


@app.route("/predict_gender", methods=["POST"])
def predict():

    data = request.json

    name = data["name"]
    age = data["age"]

    gender = predict_gender(name, age)

    return jsonify({
        "name": name,
        "predicted_gender": gender
    })


if __name__ == "__main__":
    app.run(debug=True)
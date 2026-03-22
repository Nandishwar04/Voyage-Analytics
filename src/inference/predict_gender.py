import pickle
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/gender_model.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except FileNotFoundError:
    model = None
    print("[WARN] gender_model.pkl not found. Place it in the models/ directory.")


def predict_gender(name: str, age: int) -> str:
    """
    Predict gender based on name and age.
    Args:
        name: Person's first name
        age:  Person's age (1-100)
    Returns:
        Predicted gender label string (e.g. 'Male' / 'Female')
    """
    if model is None:
        raise RuntimeError("Gender model is not loaded.")

    data = {
        "name": [name.lower()],
        "age": [age],
    }
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return prediction[0]